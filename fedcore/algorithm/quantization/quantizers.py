"""
Quantization pipeline base class for FedCore.

This module provides :class:`BaseQuantizer`, a high-level orchestrator for
PyTorch quantization workflows covering:
  * **Dynamic PTQ** (post-training quantization without calibration),
  * **Static PTQ** (with calibration on a validation dataloader),
  * **QAT** (quantization-aware training).

Core responsibilities
---------------------
- Initialize a copy of the model to be quantized and select the proper trainer.
- Build a ``qconfig`` mapping and allowed module set for the chosen mode/dtype.
- Prepare the model (fusing, qconfig propagation, Q/DQ wrapping, calibration).
- Invoke quantization hooks (see ``fedcore.algorithm.quantization.hooks``)
  to run mode-specific actions (e.g., dynamic setup, calibration loop, QAT).
- Convert the prepared model to its quantized form and return it.

Notes
-----
- The original model remains in ``self.model_before``; the quantized artifact
  is kept in ``self.model_after``.
- Hooks are registered under the summon key ``'quantization'``.
"""

from copy import deepcopy
from itertools import chain
from typing import Optional
import traceback

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
from fedcore.models.network_impl.hooks import BaseHook
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.algorithm.quantization.hooks import QuantizationHooks
from fedcore.repository.constanst_repository import TorchLossesConstant
from fedcore.models.network_impl.hooks import Optimizers


class BaseQuantizer(BaseCompressionModel):
    """
    High-level quantization orchestrator supporting dynamic PTQ, static PTQ, and QAT.

    Configuration is provided via ``params`` and determines device, backend,
    dtype, allowed module set, qconfig mapping, and (for QAT) training
    hyperparameters. The typical flow is:

        1) :meth:`_init_model` — copy & place model to device, select trainer.
        2) :meth:`_init_hooks` — construct quantization hooks for the chosen mode.
        3) :meth:`_prepare_model` — uninplace, (re)assemble, fuse, propagate qconfig,
           attach Q/DQ wrappers, and (if needed) run calibration / enable QAT.
        4) :meth:`fit` — execute hooks, run ``convert``/``quantize_dynamic``,
           and return the quantized model in ``self.model_after``.

    Attributes
    ----------
    device : torch.device
        Target device for quantization phases.
    quant_type : {'dynamic','static','qat'}
        Quantization workflow to execute.
    backend : str
        Quantization backend engine (e.g., 'fbgemm', 'onednn').
    dtype : torch.dtype
        Quantized dtype (e.g., ``torch.qint8``) or ``torch.float16`` for FP16 path.
    allow_emb : bool
        Whether to allow embedding modules in the mapping (QAT only).
    allow_conv : bool
        Whether to allow convolution modules in the mapping.
    inplace : bool
        Preference for in-place transformations (passed to PyTorch APIs where applicable).
    qconfig : dict
        Flattened qconfig mapping prepared by :meth:`get_qconfig`.
    allow : set[type]
        Allowed module set derived from PyTorch default mappings and flags.
    qat_params : dict
        QAT hyperparameters (epochs, optimizer, criterion, lr).
    optimizer : Optimizers
        Registry entry for optimizer used in QAT hooks.
    criterion : TorchLossesConstant
        Registry entry for loss used in QAT hooks.
    _hooks : list
        Hook group used to instantiate quantization hooks.

    See Also
    --------
    fedcore.algorithm.quantization.hooks : Hook implementations for each mode.
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        """Initialize quantization settings, qconfig, and hook registry.

        Parameters
        ----------
        params : Optional[OperationParameters], default={}
            Recognized keys:
              - ``device`` (torch.device or str), ``quant_type`` ('dynamic'|'static'|'qat'),
              - ``backend`` (str), ``dtype`` (torch.dtype),
              - ``allow_emb`` (bool), ``allow_conv`` (bool), ``inplace`` (bool),
              - ``qconfig`` (optional prebuilt mapping),
              - ``qat_params`` dict with 'epochs'|'optimizer'|'criterion'|'lr'.
        """
        super().__init__(params)
        
        # Quantizing params
        self.device = params.get("device", default_device())
        self.quant_type = params.get("quant_type", 'dynamic')
        self.backend = params.get("backend", 'fbgemm')
        self.dtype = params.get("dtype", torch.qint8)
        self.allow_emb = params.get("allow_emb", False)
        self.allow_conv = params.get("allow_conv", True)
        self.inplace = params.get("inplace", False)
        
        # Device/dtype/backend interplay (kept as in original implementation)
        # - Prefer FP16 on CUDA; restrict conv quant if FP16 is chosen.
        # - Switch backend on CUDA to 'onednn' (as in original code).
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

        print(f"[INIT] quant_type: {self.quant_type}, backend: {self.backend}, device: {self.device}")
        print(f"[INIT] dtype: {self.dtype}, allow_embedding: {self.allow_emb}, allow_convolution: {self.allow_conv}")
        print(f"[INIT] qconfig: {self.qconfig}")

    def __repr__(self):
        """Return human-readable identifier of the configured quantization type."""
        return f"{self.quant_type.upper()} Quantization"

    def _init_empty_object(self):
        """Initialize history containers and hook lists (internal helper)."""
        self.history = {'train_loss': [], 'val_loss': []}
        self._on_epoch_end = []
        self._on_epoch_start = []

    def _init_hooks(self, input_data):
        """Instantiate quantization hooks for the current mode.

        Parameters
        ----------
        input_data : InputData
            Data container; used to pass dataloaders into hooks.
        """
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
        """Build the set of allowed module types for quantization conversion.

        The base mapping is chosen from PyTorch defaults according to
        ``self.quant_type`` and then optionally adjusted:
          * add embedding modules for QAT when ``allow_emb`` is True,
          * drop convolutional modules when ``allow_conv`` is False.

        Returns
        -------
        set[type]
            Set of module classes that are allowed to be quantized/converted.
        """
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
        """Create and flatten a global qconfig mapping for the chosen mode/dtype.

        For INT8, chooses between ``default_dynamic_qconfig`` / ``default_qconfig`` /
        ``get_default_qat_qconfig``. For FP16, uses the corresponding float16 configs.

        If ``allow_emb`` is True, embedding modules are configured with
        ``float_qparams_weight_only_qconfig``.

        Returns
        -------
        dict
            A flattened qconfig dictionary suitable for propagation utilities.
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
        """Fetch one validation batch for calibration/shape checks and move to device."""
        loader = input_data.features.val_dataloader
        example_input, _ = next(iter(loader))
        print(f"[DATA] Example input shape: {example_input.shape}")
        return example_input.to(self.device)

    def _init_model(self, input_data):
        """Copy the base model to device, select trainer, and build a quant copy.

        Parameters
        ----------
        input_data : InputData
            Container holding the source model and task metadata.
        """
        self.model_before = input_data.target.to(self.device)
        if input_data.task.task_type.value.__contains__('forecasting'):
            self.trainer = BaseNeuralForecaster(self.qat_params)
        else:
            self.trainer = BaseNeuralModel(self.qat_params)
        self.trainer.model = self.model_before
        self.quant_model = deepcopy(self.model_before).eval()
        print("[MODEL] Model initialized and copied for quantization.")

    def _prepare_model(self, input_data: InputData):
        """Prepare the model for the selected quantization mode.

        Steps
        -----
        1) Make modules out-of-place friendly (``uninplace``) and reassemble parents.
        2) Fuse model if it provides ``fuse_model`` (common in ResNet-like backbones).
        3) Propagate qconfig through the module tree.
        4) Collect example input and wrap the model with Q/DQ entry/exit points.
        5) For each mode:
            - **dynamic**: switch to eval mode.
            - **static** : ``prepare`` and run calibration on val dataloader, eval mode.
            - **qat**    : ``prepare_qat`` and switch to train mode.

        Returns
        -------
        nn.Module
            Prepared model ready for conversion / quantize_dynamic.

        Notes
        -----
        Any exception reverts preparation and supplies a clean eval copy of the
        original model to keep the pipeline robust.
        """
        try:
            uninplace(self.quant_model)
            self.quant_model = ParentalReassembler.reassemble(self.quant_model)
            if hasattr(self.quant_model, 'fuse_model'):
                self.quant_model.fuse_model()
                print("[PREPARE] fuse_model executed.")
            propagate_qconfig_(self.quant_model, self.qconfig)
            for name, module in self.quant_model.named_modules():
                print(f"Module: {name}, qconfig: {module.qconfig}")
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
                    for batch, _ in input_data.features.val_dataloader:
                        self.quant_model(batch.to(self.device))
            elif self.quant_type == 'qat':
                self.quant_model.train()
                prepare_qat(self.quant_model, inplace=True)

            print("[PREPARE] Model prepared successfully.")
            return self.quant_model

        except Exception as e:
            print("[PREPARE ERROR] Exception during preparation:")
            traceback.print_exc()
            return deepcopy(self.model_before).eval()

    def fit(self, input_data: InputData):
        """Execute the full quantization flow and return the quantized model.

        Parameters
        ----------
        input_data : InputData
            Data container with model and loaders.

        Returns
        -------
        nn.Module
            Quantized model instance (also stored in ``self.model_after``).
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

            print("[FIT] Quantization performed successfully.")
            self.model_after = self.quant_model
            if self.quant_type == 'qat':
                self.model_after._is_quantized = True 

        except Exception as e:
            print("[FIT ERROR] Exception during quantization:")
            traceback.print_exc()
            self.model_after = deepcopy(self.model_before).eval()
            
        return self.model_after

    def predict_for_fit(self, input_data: InputData, output_mode: str = "fedcore"):
        """Return the model object used during fit-time predictions (compat shim)."""
        return self.model_after if output_mode == "fedcore" else self.model_before

    def predict(self, input_data: InputData, output_mode: str = "fedcore"):
        """Forward to the trainer using either the quantized or original model."""
        self.trainer.model = self.model_after if output_mode == "fedcore" else self.model_before
        return self.trainer.predict(input_data, output_mode)
