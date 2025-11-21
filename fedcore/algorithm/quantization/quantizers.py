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

import torch
from torch import nn
from torch.ao.quantization import propagate_qconfig_
from torch.ao.quantization.qconfig import (
    default_qconfig, default_dynamic_qconfig,
    float16_static_qconfig, float16_dynamic_qconfig,
    get_default_qat_qconfig, float_qparams_weight_only_qconfig)
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.quantization_mappings import (
    get_embedding_qat_module_mappings,
    DEFAULT_STATIC_QUANT_MODULE_MAPPINGS,
    DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS,
    DEFAULT_QAT_MODULE_MAPPINGS)

from fedot.core.data.data import InputData
from fedcore.algorithm.base_compression_model import BaseCompressionModel
from fedcore.algorithm.quantization.utils import (
    ParentalReassembler, QDQWrapper, uninplace, get_flattened_qconfig_dict)
from fedcore.api.api_configs import QuantMode
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
from fedcore.algorithm.quantization.hooks import AbstractQuantizationHook, DynamicQuantizationHook, QATHook, StaticQuantizationHook


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

    DEFAULT_HOOKS: list[type[AbstractQuantizationHook]] = [DynamicQuantizationHook, StaticQuantizationHook, QATHook]

    def __init__(self, params: dict = {}):
        super().__init__(params)
        
        # Quantizing params
        self.quant_type: str = params.get("quant_type", QuantMode.DYNAMIC.value)
        self.backend = params.get("backend", 'fbgemm')
        self.dtype = params.get("dtype", torch.qint8)
        self.allow_emb = params.get("allow_emb", False)
        self.allow_conv = params.get("allow_conv", True)
        self.inplace = params.get("inplace", False)
        self.quant_each = params.get("quant_each", -1)
        self.prepare_qat_after_epoch = params.get("prepare_qat_after_epoch", 1)
        
        self._change_params_considering_device_type()
        
        self.qconfig = params.get("qconfig", self.get_qconfig())
        self.allowed_quant_module_mapping = self._set_allowed_quant_module_mappings()

        print(f"[INIT] quant_type: {self.quant_type}, backend: {self.backend}, device: {self.device}")
        print(f"[INIT] dtype: {self.dtype}, allow_embedding: {self.allow_emb}, allow_convolution: {self.allow_conv}")
        print(f"[INIT] qconfig: {self.qconfig}")

    
    def _change_params_considering_device_type(self):
        """Adjust device-dependent quantization settings.

        This helper adapts certain parameters to the current device:

        * on CUDA devices, forces ``dtype = torch.float16`` and backend
          ``'onednn'``, since int8 backends may not be available;
        * disables convolution quantization when using float16
          (``allow_conv = False``), as some backends do not support
          conv+float16 combinations.

        Notes
        -----
        The actual compatibility constraints depend on PyTorch and backend
        versions; this method encodes a conservative default strategy.
        """
        """Corrects device-sensitive parameters like dtype or backend library for compatibility with
        concrete hardware
        """
        if (self.device.type == "cuda"):
            self.dtype = torch.float16
            self.backend = 'onednn'

        self.allow_conv = False if self.dtype == torch.float16 else self.allow_conv

    def __repr__(self):
        """Return a short string representation with quantization mode."""
        return f"{self.quant_type.upper()} Quantization"

    def _set_allowed_quant_module_mappings(self) -> set[nn.Module]:
        """Derive the set of module types that are allowed to be quantized.

        The mapping is selected based on ``self.quant_type``:

        * ``"dynamic"`` – uses :data:`DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS`;
        * ``"static"`` – uses :data:`DEFAULT_STATIC_QUANT_MODULE_MAPPINGS`;
        * ``"qat"`` – uses :data:`DEFAULT_QAT_MODULE_MAPPINGS`.

        Additional adjustments:

        * if ``allow_emb`` is enabled and mode is QAT, embedding QAT mappings
          are merged in;
        * if ``allow_conv`` is ``False``, all convolution and transposed
          convolution layers are removed from the mapping.

        Returns
        -------
        set[nn.Module]
            Set of module classes that should be considered for quantization.
        """
        mapping_dict = {
            'dynamic': DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS,
            'static': DEFAULT_STATIC_QUANT_MODULE_MAPPINGS,
            'qat': DEFAULT_QAT_MODULE_MAPPINGS
        }.get(self.quant_type)
        if self.allow_emb and self.quant_type is QuantMode.QAT.value:
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

        print(f"[QCONFIG] Created qconfig mapping: {qconfig_mapping}")
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
        loader = input_data.features.val_dataloader
        example_input, _ = next(iter(loader))
        print(f"[DATA] Example input shape: {example_input.shape}")
        return example_input.to(self.device)

    def _prepare_model_after_for_quantizing(self, input_data: InputData):
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
            uninplace(self.model_after) #skip connection operations and other may work incorrect with nn.Relu(inplace=True)
            # https://stackoverflow.com/questions/69913781/is-it-true-that-inplace-true-activations-in-pytorch-make-sense-only-for-infere
            self.model_after = ParentalReassembler.reassemble(self.model_after)
            if hasattr(self.model_after, 'fuse_model'):
                self.model_after.fuse_model()
                print("[PREPARE] fuse_model executed.")
            propagate_qconfig_(self.model_after, self.qconfig)
            for name, module in self.model_after.named_modules():
                print(f"Module: {name}, qconfig: {module.qconfig}")
            self.data_batch_for_calib = self._get_example_input(input_data)

            QDQWrapper.add_quant_entry_exit(
                self.model_after, *(self.data_batch_for_calib,), allow=self.allowed_quant_module_mapping, mode=self.quant_type
            )

            print("[PREPARE] Model prepared successfully.")

        except Exception as e:
            print("[PREPARE ERROR] Exception during preparation:")
            traceback.print_exc()
            return deepcopy(self.model_before).eval()
        
    def _init_trainer_model_before_model_after_and_incapsulate_hooks(self, input_data):
        """Initialize trainer, models and attach quantization hooks.

        This method:

        1. Filters quantization hooks using
           :meth:`BaseNeuralModel.filter_hooks_by_params` and
           :data:`DEFAULT_HOOKS`.
        2. Instantiates the selected quantization hooks with current
           quantization parameters (epoch schedule, dtype, allowed modules,
           QAT preparation epoch, backend).
        3. Calls the base
           :meth:`BaseCompressionModel._init_trainer_model_before_model_after`
           to create ``trainer``, ``model_before`` and ``model_after`` with
           these hooks attached.
        4. Runs :meth:`_prepare_model_after_for_quantizing` on
           ``model_after`` to reassemble and configure it for quantization.

        Parameters
        ----------
        input_data : InputData
            Data and configuration object used to build the trainer and
            base model.
        """
        additional_hooks = BaseNeuralModel.filter_hooks_by_params(self.params, self.DEFAULT_HOOKS)
        additional_hooks = [quant_hook_type(
                                self.quant_each,
                                self.dtype, 
                                self.allowed_quant_module_mapping, 
                                self.prepare_qat_after_epoch,
                                self.backend) 
                            for quant_hook_type in additional_hooks]
        self._init_trainer_model_before_model_after(input_data, additional_hooks)
        self._prepare_model_after_for_quantizing(input_data)

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
        super()._prepare_trainer_and_model_to_fit(input_data)
        try:
            self.trainer.fit(input_data)
            print("[FIT] Quantization performed successfully.")

        except Exception as e:
            print("[FIT ERROR] Exception during quantization:")
            traceback.print_exc()
            self.model_after = deepcopy(self.model_before).eval()
            
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
