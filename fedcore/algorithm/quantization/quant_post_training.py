from typing import Optional
from copy import deepcopy

from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters

from fedcore.algorithm.base_compression_model import BaseCompressionModel
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
from fedcore.neural_compressor import quantization
from fedcore.neural_compressor.config import PostTrainingQuantConfig
from fedcore.repository.constanst_repository import default_device
from .utils import uninplace, ParentalReassembler, reset_qconfig, QDQWrapper, QDQWrapping
from torch.ao.quantization import (
    quantize, quantize_dynamic, quantize_qat, 
    propagate_qconfig_, 
    convert,
    prepare
)
from torch.ao.quantization.quantization_mappings import get_default_dynamic_quant_module_mappings
from torch.ao.quantization.qconfig import default_dynamic_qconfig, float_qparams_weight_only_qconfig, default_qconfig
from torch.ao.quantization.qconfig_mapping import QConfigMapping, get_default_qconfig_mapping
from torch.ao.quantization.qconfig_mapping_utils import get_flattened_qconfig_dict
import torch.nn as nn
import torch.ao.nn.quantized.dynamic as nnqd

class PostTrainingQuantization(BaseCompressionModel):
    pass

class QuantPostModel(BaseCompressionModel):
    """Class responsible for Pruning model implementation.
    Example:
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)
        self.epochs = params.get("epochs", 5)
        self.learning_rate = params.get("learning_rate", 0.001)
        self.backend = params.get('backend', 'x86')
        self.qconfig = params.get("qconfig", None)
        self.allow_conv = params.get('allow_conv', True)
        self.allow_emb = params.get('allow_emb', True)
        self.inplace = params.get('inplace', False)
        self.trainer = BaseNeuralModel(params)
        self.device = default_device()

    def get_qconfig(self):
        qconfig = QConfigMapping().set_global(default_qconfig)
        if self.allow_emb:
            qconfig.set_object_type(nn.Embedding, float_qparams_weight_only_qconfig)
        return get_flattened_qconfig_dict(qconfig)   

    def _prepare_model(self, input_data: InputData, supplementary_data=None):
        model = (input_data.target
            if "predict" not in vars(input_data)
            else input_data.predict)
        if not self.inplace:
            model = deepcopy(model)
        model.eval()

        uninplace(model)
        model = ParentalReassembler.reassemble(model, self.params.get('additional_mapping', None))
        self.qconfig = self.get_qconfig()
        propagate_qconfig_(model, self.qconfig)
        b = self.__get_example_input(input_data)
        QDQWrapper.add_quant_entry_exit(model, b)
        prepare(model, inplace=True)
        self.trainer.model = model
        return model

    def __repr__(self):
        return f"{self.quant_type.upper()} Quantization"

    def fit(self, input_data: InputData, supplementary_data=None):
        self.model = self._prepare_model(
            input_data, supplementary_data
        )
        self.trainer.fit(input_data)
        convert(self.model, inplace=True)
        self.optimised_model = self.model.to('cpu')
        self.model._is_quantized = True

    def predict_for_fit(self, input_data: InputData, output_mode: str = "compress"):
        self.trainer.model = (
            self.optimised_model if output_mode == "compress" else self.model
        )
        return self.trainer.predict(input_data, output_mode)

    def predict(self, input_data: InputData, output_mode: str = "compress"):
        self.trainer.model = (
            self.optimised_model if output_mode == "compress" else self.model
        )
        return self.trainer.predict(input_data, output_mode)
    
    def __get_example_input(self, input_data: InputData):
        b = next(iter(input_data.features.calib_dataloader))
        if isinstance(b, (list, tuple)) and len(b) == 2:
            return b[0]
        return b
    

class QuantDynamicModel(BaseCompressionModel):
    """Class responsible for Dynamic Post Quantiization implementation.
    Example:
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)
        self.epochs = params.get("epochs", 5)
        self.learning_rate = params.get("learning_rate", 0.001)
        self.backend = params.get('backend', 'x86')
        self.dtype = params.get('dtype', None)
        self.qconfig = params.get(
            "qconfig", None
        )
        self.allow_emb = params.get('allow_emb', True)
        self.allow_conv = params.get('allow_conv', False)
        self.inplace = params.get('inplace', False)
        self.trainer = BaseNeuralModel(params)
        self.device = default_device()

    def get_qconfig(self,):
        if not (self.dtype or self.qconfig):
            qconfig = QConfigMapping().set_global(default_dynamic_qconfig)
            if self.allow_emb:
                qconfig.set_object_type(nn.Embedding, float_qparams_weight_only_qconfig)
            return get_flattened_qconfig_dict(qconfig)            

    def _prepare_model(self, input_data: InputData, supplementary_data=None):
        model = (input_data.target
            if "predict" not in vars(input_data)
            else input_data.predict)
        if not self.inplace:
            model = deepcopy(model)
        model.eval()
        uninplace(model)
        model = ParentalReassembler.reassemble(model, self.params.get('additional_mapping', None))
        if self.qconfig:
            propagate_qconfig_(model, self.qconfig)
        return model

    def __repr__(self):
        return f"Dynamic Quantization"

    def fit(self, input_data: InputData, supplementary_data=None):
        self.model = self._prepare_model(
            input_data, supplementary_data
        )
        quantize_dynamic(self.model, self.get_qconfig(), self.dtype,
                         self.__update_mappings(),
                         inplace=True)
        self.optimised_model = self.model.to('cpu')
        self.model._is_quantized = True

    def __update_mappings(self):
        mapping = get_default_dynamic_quant_module_mappings()
        additional = {
            nn.Conv1d: nnqd.Conv1d,
            nn.Conv2d: nnqd.Conv2d,
            nn.Conv3d: nnqd.Conv3d,
            nn.ConvTranspose1d: nnqd.ConvTranspose1d,
            nn.ConvTranspose2d: nnqd.ConvTranspose2d,
            nn.ConvTranspose3d: nnqd.ConvTranspose3d,
        }
        if self.allow_conv:
            mapping.update(additional)
        return mapping


    def predict_for_fit(self, input_data: InputData, output_mode: str = "compress"):
        self.trainer.model = (
            self.optimised_model if output_mode == "compress" else self.model
        )
        return self.trainer.predict(input_data, output_mode)

    def predict(self, input_data: InputData, output_mode: str = "compress"):
        self.trainer.model = (
            self.optimised_model if output_mode == "compress" else self.model
        )
        return self.trainer.predict(input_data, output_mode)
