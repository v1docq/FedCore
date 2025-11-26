"""Base quantization model abstraction for FedCore.

This module defines :class:`BaseQuantizer`, a compression wrapper that adds
quantization capabilities on top of :class:`BaseNeuralModel`. It:

* configures quantization mode (dynamic, static, QAT) and backend;
* builds appropriate QConfig mappings and allowed module sets;
* prepares the model graph for quantization (uninplace, fusion, QDQ wrapping);
* injects quantization hooks (:class:`DynamicQuantizationHook`,
  :class:`StaticQuantizationHook`, :class:`QATHook`) into the training loop;
* exposes a unified ``fit / predict`` interface compatible with other
  FedCore compression models.
"""

from copy import deepcopy
import traceback
import logging

import torch
import torch.ao.nn.quantized.dynamic as nnqd
from torch import nn, optim
from torch.ao.quantization import (
    convert, prepare, prepare_qat, propagate_qconfig_, quantize_dynamic)
from torch.ao.quantization.qconfig import (
    default_qconfig, default_dynamic_qconfig,
    float16_static_qconfig, float16_dynamic_qconfig,
    get_default_qat_qconfig, float_qparams_weight_only_qconfig, QConfig)
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.quantization_mappings import (
    get_embedding_qat_module_mappings,
    DEFAULT_STATIC_QUANT_MODULE_MAPPINGS,
    DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS,
    DEFAULT_QAT_MODULE_MAPPINGS)

from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedcore.algorithm.base_compression_model import BaseCompressionModel
from fedcore.algorithm.quantization.utils import (
    ParentalReassembler, QDQWrapper, uninplace, get_flattened_qconfig_dict)
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel, BaseNeuralForecaster
from fedcore.models.network_impl.utils.trainer_factory import create_trainer_from_input_data
from fedcore.models.network_impl.utils.hooks import BaseHook
from fedcore.architecture.computational.devices import default_device
from fedcore.algorithm.quantization.hooks import QuantizationHooks
from fedcore.repository.constant_repository import TorchLossesConstant
from fedcore.models.network_impl.utils.hooks import Optimizers
from fedcore.tools.registry.model_registry import ModelRegistry


class BaseQuantizer(BaseCompressionModel):
    """Base class for model quantization in FedCore.

    This class wraps a neural model and configures PyTorch quantization
    workflows, including post-training dynamic/static quantization and
    quantization-aware training (QAT). It:

    * selects quantization mode and backend based on configuration and device;
    * creates a flattened qconfig mapping for the model;
    * derives the set of module types allowed to be quantized;
    * prepares the model for quantization (graph reassembly, fusion, Q/DQ
      wrappers);
    * attaches appropriate quantization hooks to the trainer.

    Parameters
    ----------
    params : dict, optional
        Configuration dictionary. Common keys include:

        * ``"quant_type"`` – quantization mode, one of
          :class:`QuantMode` values (default: ``QuantMode.DYNAMIC.value``);
        * ``"backend"`` – quantization backend
          (e.g. ``"fbgemm"``, ``"qnnpack"``; default ``"fbgemm"``);
        * ``"dtype"`` – target quantization dtype
          (e.g. ``torch.qint8`` or ``torch.float16``; default ``torch.qint8``);
        * ``"allow_emb"`` – whether to quantize embedding layers
          (default ``False``);
        * ``"allow_conv"`` – whether to quantize convolution layers
          (default ``True``; can be overridden by device/dtype);
        * ``"inplace"`` – whether quantization should be done in-place
          on the model (default ``False``; used mostly for API compatibility);
        * ``"quant_each"`` – epoch at which quantization hook should fire
          (or interval, depending on hook logic; default ``-1`` – disabled);
        * ``"prepare_qat_after_epoch"`` – epoch to call
          ``prepare_qat`` for QAT mode (default ``1``);
        * Any additional keys supported by
          :class:`BaseCompressionModel` / :class:`BaseNeuralModel`.
    """

    def __init__(self, params: dict = {}):
        super().__init__(params)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Quantizing params
        self.device = params.get("device", default_device())
        self.quant_type = params.get("quant_type", 'dynamic')
        self.backend = params.get("backend", 'fbgemm')
        self.dtype = params.get("dtype", torch.qint8)
        self.allow_emb = params.get("allow_emb", False)
        self.allow_conv = params.get("allow_conv", True)
        self.inplace = params.get("inplace", False)
        
        # self.device = default_device('cpu') if self.quant_type == 'qat' else self.device
        self.dtype = torch.float16 if self.device.type == "cuda" else self.dtype
        self.backend = 'onednn' if self.device.type == ("cuda") else self.backend
        self.allow_conv = False if self.dtype == torch.float16 else self.allow_conv
        torch.backends.quantized.engine = self.backend
        
        self.qconfig = params.get("qconfig", self.get_qconfig())
        self.allow = self._set_allow_mappings()
        
        # QAT params
        self.qat_params = params.get("qat_params", dict())
        if not self.qat_params:
            self.qat_params = dict()
            self.qat_params["epochs"] = 2
            self.qat_params["optimizer"] = 'adam'
            self.qat_params["criterion"] = 'cross_entropy'
            self.qat_params["lr"] = 0.001
        self.optimizer = Optimizers[self.qat_params.get("optimizer", 'adam')]
        self.criterion = TorchLossesConstant[self.qat_params.get("criterion", 'cross_entropy')]

        # Hooks initialization
        self._hooks = [QuantizationHooks]
        self._init_empty_object()
        self._quantization_index = 0

        self.logger.info(f"[INIT] quant_type: {self.quant_type}, backend: {self.backend}, device: {self.device}")
        self.logger.info(f"[INIT] dtype: {self.dtype}, allow_embedding: {self.allow_emb}, allow_convolution: {self.allow_conv}")
        self.logger.info(f"[INIT] qconfig: {self.qconfig}")

    def __repr__(self):
        """Return a short string representation with quantization mode."""
        return f"{self.quant_type.upper()} Quantization"

    def _init_empty_object(self):
        self.history = {'train_loss': [], 'val_loss': []}
        self._on_epoch_end = []
        self._on_epoch_start = []

    def _init_hooks(self, input_data):
        self._on_epoch_end = []
        self._on_epoch_start = []

        hook_params = {
            "input_data": input_data,
            "dtype": self.dtype,
            "epochs": self.qat_params.get("epochs", 2),
            "optimizer": self.optimizer.value,
            "criterion": self.criterion.value(),
            "lr": self.qat_params.get("lr", 0.001),
            "device": self.device
        }

        for hook_elem in QuantizationHooks:
            hook: BaseHook = hook_elem.value(hook_params, self.quant_model)
            if hook._hook_place == 'post':
                self._on_epoch_end.append(hook)
            else:
                self._on_epoch_start.append(hook)

    def _set_allow_mappings(self):
        mapping_dict = {
            'dynamic': DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS,
            'static': DEFAULT_STATIC_QUANT_MODULE_MAPPINGS,
            'qat': DEFAULT_QAT_MODULE_MAPPINGS
        }.get(self.quant_type)
        if self.allow_emb and self.quant_type == 'qat':
            mapping_dict.update(get_embedding_qat_module_mappings().keys())
        if not self.allow_conv:
            conv_set = {nn.Conv1d, nn.Conv2d, nn.Conv3d,
                        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d}
            mapping_dict = {k: v for k, v in mapping_dict.items() if k not in conv_set}
        return set(mapping_dict)

    def get_qconfig(self):
        """Create and flatten a global QConfig mapping for the model.

        The base qconfig is chosen based on ``self.dtype`` and
        ``self.quant_type``:

        * for ``torch.qint8``:
          * dynamic – :data:`default_dynamic_qconfig`;
          * static – :data:`default_qconfig`;
          * qat – :func:`get_default_qat_qconfig(self.backend)`;
        * for ``torch.float16``:
          * dynamic – :data:`float16_dynamic_qconfig`;
          * static – :data:`float16_static_qconfig`;
          * qat – :func:`get_default_qat_qconfig(self.backend)`.

        A :class:`QConfigMapping` with global qconfig is then built and,
        if embeddings are allowed, specialized qconfigs for
        :class:`nn.Embedding` / :class:`nn.EmbeddingBag` with
        :data:`float_qparams_weight_only_qconfig` are added.

        Finally, the mapping is flattened to a dict via
        :func:`get_flattened_qconfig_dict`.

        Returns
        -------
        dict
            Flattened qconfig mapping suitable for propagation with
            :func:`propagate_qconfig_`.
        """
        if self.dtype == torch.qint8:
            qconfig = {
                'dynamic': default_dynamic_qconfig,
                'static': default_qconfig,
                'qat': get_default_qat_qconfig(self.backend)
            }.get(self.quant_type, default_qconfig)
        elif self.dtype == torch.float16:
            qconfig = {
                'dynamic': float16_dynamic_qconfig,
                'static': float16_static_qconfig,
                'qat': get_default_qat_qconfig(self.backend)
            }.get(self.quant_type, default_qconfig)

        qconfig_mapping = QConfigMapping().set_global(qconfig)
        if self.allow_emb:
            qconfig_mapping.set_object_type(nn.Embedding, float_qparams_weight_only_qconfig)
            qconfig_mapping.set_object_type(nn.EmbeddingBag, float_qparams_weight_only_qconfig)

        self.logger.info(f"[QCONFIG] Created qconfig mapping: {qconfig_mapping}")
        return get_flattened_qconfig_dict(qconfig_mapping)

    def _get_example_input(self, input_data: InputData):
        """Extract a single batch from validation data as example input.

        Parameters
        ----------
        input_data : InputData
            Fedot input data that contains ``features.val_dataloader``.

        Returns
        -------
        torch.Tensor
            Example input tensor moved to ``self.device``.
        """
        loader = input_data.val_dataloader
        example_input, _ = next(iter(loader))
        self.logger.info(f"[DATA] Example input shape: {example_input.shape}")
        return example_input.to(self.device)

    def _init_model(self, input_data):
        model = input_data.target
        if isinstance(model, str):
            loaded = torch.load(model, map_location=self.device)
            if isinstance(loaded, dict) and "model" in loaded:
                model = loaded["model"]
            else:
                model = loaded

        self.model_before = model.to(self.device)

        self.trainer = create_trainer_from_input_data(input_data, self.qat_params)

        self.trainer.model = self.model_before
        self.quant_model = self.model_before.eval()

        self._model_registry = ModelRegistry()
        self._quantization_index += 1
        if self._model_id_before:
            self._model_registry.update_metrics(
                fedcore_id=self._fedcore_id,
                model_id=self._model_id_before,
                metrics={},
                stage="before",
                mode=self.__class__.__name__
            )

        self.logger.info("[MODEL] Model initialized for quantization (no deepcopy).")

    def _prepare_model(self, input_data: InputData):
        """Prepare ``model_after`` for quantization.

        This method executes several steps required before applying
        quantization transforms:

        1. Call :func:`uninplace` to replace in-place activations (e.g. ReLU)
           with out-of-place versions to avoid issues with quantization.
        2. Reassemble the model graph using :class:`ParentalReassembler`
           to ensure a consistent module hierarchy.
        3. If the model exposes ``fuse_model``, call it to fuse modules
           (Conv+BN+ReLU, etc.) before quantization.
        4. Propagate qconfig using :func:`propagate_qconfig_` with
           ``self.qconfig``.
        5. Store an example batch for calibration via :meth:`_get_example_input`.
        6. Insert Q/DQ wrappers at model entry/exit points through
           :meth:`QDQWrapper.add_quant_entry_exit`.

        If any step fails, an error is printed and a deep copy of
        ``model_before`` in eval mode is returned.

        Parameters
        ----------
        input_data : InputData
            Input data used to derive an example batch for calibration.

        Returns
        -------
        torch.nn.Module or None
            On error, a copy of the original model in eval mode is returned;
            on success, ``self.model_after`` is prepared in-place and
            ``None`` is implicitly returned.
        """
        try:
            uninplace(self.quant_model)
            self.quant_model = ParentalReassembler.reassemble(self.quant_model)
            if hasattr(self.quant_model, 'fuse_model'):
                self.quant_model.fuse_model()
                self.logger.info("[PREPARE] fuse_model executed.")
            propagate_qconfig_(self.quant_model, self.qconfig)
            for name, module in self.quant_model.named_modules():
                self.logger.info(f"Module: {name}, qconfig: {module.qconfig}")
            self.data_batch_for_calib = self._get_example_input(input_data)

            QDQWrapper.add_quant_entry_exit(
                self.quant_model, *(self.data_batch_for_calib,), allow=self.allow, mode=self.quant_type
            )

            if self.quant_type == 'dynamic':
                self.quant_model.eval()
            elif self.quant_type == 'static':
                prepare(self.quant_model, inplace=True)
                self.quant_model.eval()
                with torch.no_grad():
                    for batch, _ in input_data.val_dataloader:
                        self.quant_model(batch.to(self.device))
            elif self.quant_type == 'qat':
                self.quant_model.train()
                prepare_qat(self.quant_model, inplace=True)

            self.logger.info("[PREPARE] Model prepared successfully.")
            return self.quant_model

        except Exception as e:
            self.logger.info("[PREPARE ERROR] Exception during preparation:")
            traceback.print_exc()
            return self.model_before.eval()

    def fit(self, input_data: InputData):
        """Run training with quantization hooks and return the quantized model.

        This method prepares the trainer and models via
        ``_prepare_trainer_and_model_to_fit`` (which internally calls
        :meth:`_init_trainer_model_before_model_after_and_incapsulate_hooks`),
        then runs ``self.trainer.fit``.

        If an exception occurs during training or quantization, the error
        is printed, and ``model_after`` is replaced with a deep copy of
        ``model_before`` in eval mode.

        Parameters
        ----------
        input_data : InputData
            Fedot input data used for training and calibration.

        Returns
        -------
        torch.nn.Module
            Quantized model (or original model in case of failure), stored
            in ``self.model_after``.
        """
        self._init_model(input_data)
        self._init_hooks(input_data)
        self.quant_model = self._prepare_model(input_data)
        
        hook_args = {'model': self.quant_model}
        for hook in self._on_epoch_end:
            if hook.trigger(self.quant_type):
                hook.action(self.quant_type, hook_args)

        try:
            if self.quant_type == 'dynamic':
                self.quant_model = quantize_dynamic(
                    self.quant_model,
                    qconfig_spec=self.allow,
                    dtype=self.dtype,
                    inplace=True)
            else:
                convert(self.quant_model, inplace=True)

            self.logger.info("[FIT] Quantization performed successfully.")
            self.model_after = self.quant_model

            if self.quant_type == 'qat':
                self.model_after._is_quantized = True

            if self._model_id_after:
                self._model_registry.update_metrics(
                    fedcore_id=self._fedcore_id,
                    model_id=self._model_id_after,
                    metrics={},
                    stage="after",
                    mode=self.__class__.__name__
                )
        except Exception as e:
            self.logger.info("[FIT ERROR] Exception during quantization:")
            traceback.print_exc()
            self.model_after = self.model_before.eval()
            
        return self.model_after

    def predict_for_fit(self, input_data: InputData, output_mode: str = "fedcore"):
        """Return model object after `fit` for integration in FedCore pipelines.

        Parameters
        ----------
        input_data : InputData
            Input data (unused, kept for API compatibility).
        output_mode : str, optional
            If ``"fedcore"`` (default), return quantized model
            ``self.model_after``; otherwise return ``self.model_before``.

        Returns
        -------
        torch.nn.Module
            Selected model instance.
        """
        return self.model_after if output_mode == "fedcore" else self.model_before

    def predict(self, input_data: InputData, output_mode: str = "fedcore"):
        """Run prediction with either quantized or original model.

        Parameters
        ----------
        input_data : InputData
            Data for inference.
        output_mode : str, optional
            If ``"fedcore"`` (default), use quantized model
            ``self.model_after``; otherwise use the original unquantized
            model ``self.model_before``.

        Returns
        -------
        Any
            Output of :meth:`BaseNeuralModel.predict` for the selected model.
        """
        self.trainer.model = self.model_after if output_mode == "fedcore" else self.model_before
        return self.trainer.predict(input_data, output_mode)
