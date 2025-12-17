from copy import deepcopy
import traceback
import logging

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
from fedcore.algorithm.low_rank.reassembly.core_reassemblers import ParentalReassembler
from fedcore.algorithm.quantization.utils import (
    QDQWrapper, uninplace, get_flattened_qconfig_dict)
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
        self.logger = logging.getLogger(self.__class__.__name__)

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

        self._quantization_index = 0

        self.logger.info(f"[INIT] quant_type: {self.quant_type}, backend: {self.backend}, device: {self.device}")
        self.logger.info(f"[INIT] dtype: {self.dtype}, allow_embedding: {self.allow_emb}, allow_convolution: {self.allow_conv}")
        self.logger.info(f"[INIT] qconfig: {self.qconfig}")

    
    def _change_params_considering_device_type(self):
        if (self.device.type == "cuda"):
            self.dtype = torch.float16
            self.backend = 'onednn'

        self.allow_conv = False if self.dtype == torch.float16 else self.allow_conv

    def __repr__(self):
        """Return a short string representation with quantization mode."""
        return f"{self.quant_type.upper()} Quantization"

    def _set_allowed_quant_module_mappings(self) -> set[nn.Module]:
        """Derive the set of module types that are allowed to be quantized."""
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
        loader = input_data.features.val_dataloader
        example_input, _ = next(iter(loader))
        self.logger.info(f"[DATA] Example input shape: {example_input.shape}")
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

            self.logger.info("[PREPARE] Model prepared successfully.")

        except Exception as e:
            self.logger.info("[PREPARE ERROR] Exception during preparation:")
            traceback.print_exc()
            return deepcopy(self.model_before).eval()
        
    def _init_trainer_model_before_model_after_and_incapsulate_hooks(self, input_data):
        additional_hooks = BaseNeuralModel.filter_hooks_by_params(self.params, self.DEFAULT_HOOKS)
        additional_hooks = [quant_hook_type(
                                self.quant_each,
                                self.dtype, 
                                self.allowed_quant_module_mapping, 
                                self.prepare_qat_after_epoch,
                                self.backend) 
                            for quant_hook_type in additional_hooks]
        self._init_trainer_model_before_model_after(input_data, additional_hooks)
        
        self._quantization_index += 1 #TODO also broken one?
        if self._model_id_before: #TODO probably broken that "if" after BIG MERGE, check PR "Model registry #33" for find sense
            self._registry.update_metrics(
                fedcore_id=self._fedcore_id,
                model_id=self._model_id_before,
                metrics={},
                stage="before",
                mode=self.__class__.__name__
            )
        self._prepare_model_after_for_quantizing(input_data)

    def fit(self, input_data: InputData):
        """Run training with quantization hooks and return the quantized model.
        """
        super()._prepare_trainer_and_model_to_fit(input_data)
        try:
            self.trainer.fit(input_data)
            print("[FIT] Quantization performed successfully.")

            if self._model_id_after: #TODO probably broken if and it's logic, see RP "Model registry #33" for find sense
                self._registry.update_metrics(
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
        """Return model object after `fit` for integration in FedCore pipelines."""
        return self.model_after if output_mode == "fedcore" else self.model_before

    def predict(self, input_data: InputData, output_mode: str = "fedcore"):
        """Run prediction with either quantized or original model.
        """
        self.trainer.model = self.model_after if output_mode == "fedcore" else self.model_before
        return self.trainer.predict(input_data, output_mode)
