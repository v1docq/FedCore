from copy import deepcopy
from typing import Optional


import torch
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from torch import optim, nn
from torch.ao.quantization.quantize import (convert, prepare_qat, propagate_qconfig_,)
from torch.ao.quantization.qconfig import get_default_qat_qconfig
from torch.ao.quantization.qconfig_mapping import get_default_qat_qconfig_mapping
# from torch.ao.quantization.q
from fedcore.algorithm.quantization.utils import ParentalReassembler, QDQWrapper, uninplace

from fedcore.algorithm.base_compression_model import BaseCompressionModel
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
# from fedcore.neural_compressor.config import QuantizationAwareTrainingConfig
# from fedcore.neural_compressor.training import prepare_compression


class QuantAwareModel(BaseCompressionModel):
    """Quantization aware training
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)
        self.epochs = params.get("epochs", 5)
        self.learning_rate = params.get("learning_rate", 0.001)
        self.backend = params.get('backend', 'x86')
        self.qconfig = params.get(
            "qconfig", get_default_qat_qconfig_mapping(self.backend).to_dict()
        )
        self.inplace = params.get('inplace', False)
        self.trainer = BaseNeuralModel(params)
        self.device = default_device()

    def _prepare_model(self, input_data: InputData, supplementary_data=None):
        model = (input_data.target
            if "predict" not in vars(input_data)
            else input_data.predict)
        if not self.inplace:
            model = deepcopy(model)
        ### TODO add check whether it is 
        uninplace(model)
        model = ParentalReassembler.reassemble(model, self.params.get('additional_mapping', None))
        propagate_qconfig_(model, self.qconfig)
        b = self.__get_example_input(input_data)
        QDQWrapper.add_quant_entry_exit(model, b)
        model.train()
        prepare_qat(model, inplace=True)
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
