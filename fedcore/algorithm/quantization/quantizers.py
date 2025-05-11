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
    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)
        
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

        print(f"[INIT] quant_type: {self.quant_type}, backend: {self.backend}, device: {self.device}")
        print(f"[INIT] dtype: {self.dtype}, allow_embedding: {self.allow_emb}, allow_convolution: {self.allow_conv}")
        print(f"[INIT] qconfig: {self.qconfig}")

    def __repr__(self):
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
        loader = input_data.features.val_dataloader
        example_input, _ = next(iter(loader))
        print(f"[DATA] Example input shape: {example_input.shape}")
        return example_input.to(self.device)

    def _init_model(self, input_data):
        self.model_before = input_data.target.to(self.device)
        if input_data.task.task_type.value.__contains__('forecasting'):
            self.trainer = BaseNeuralForecaster(self.qat_params)
        else:
            self.trainer = BaseNeuralModel(self.qat_params)
        self.trainer.model = self.model_before
        self.quant_model = deepcopy(self.model_before).eval()
        print("[MODEL] Model initialized and copied for quantization.")

    def _prepare_model(self, input_data: InputData):
        try:
            uninplace(self.quant_model)

            if hasattr(self.quant_model, 'fuse_model'):
                self.quant_model.fuse_model()
                print("[PREPARE] fuse_model executed.")

            self.quant_model = ParentalReassembler.reassemble(self.quant_model)
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
        return self.model_after if output_mode == "fedcore" else self.model_before

    def predict(self, input_data: InputData, output_mode: str = "fedcore"):
        self.trainer.model = self.model_after if output_mode == "fedcore" else self.model_before
        return self.trainer.predict(input_data, output_mode)