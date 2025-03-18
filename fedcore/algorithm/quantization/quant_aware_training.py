from copy import deepcopy
from typing import Optional


import torch
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from torch import optim, nn
from torch.ao.quantization.quantize import (convert, prepare_qat, propagate_qconfig_,)
from torch.ao.quantization.qconfig import get_default_qat_qconfig
from fedcore.algorithm.quantization.utils import ParentalReassembler, QDQWrapper, uninplace
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from fedcore.algorithm.base_compression_model import BaseCompressionModel
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
from fedcore.algorithm.quantization.utils import get_flattened_qconfig_dict
from torch.ao.quantization.qconfig import float_qparams_weight_only_qconfig

from torch.ao.quantization.quantization_mappings import DEFAULT_QAT_MODULE_MAPPINGS, get_embedding_qat_module_mappings

class QuantAwareModel(BaseCompressionModel):
    """Quantization aware training
    """
    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)
        self.device = default_device()
        params.update(device=self.device)
        self.epochs = params.get("epochs", 5)
        self.learning_rate = params.get("learning_rate", 0.001)
        self.backend = params.get('backend', 'x86')
        self.allow_emb = params.get('allow_emb', False)
        self.allow: set = params.get('allow', set(DEFAULT_QAT_MODULE_MAPPINGS))
        self.inplace = params.get('inplace', False)
        self.trainer = BaseNeuralModel(params)
        self.qconfig = params.get(
            "qconfig", self.get_qconfig()
        )
        if self.allow_emb:
            self.allow.update(get_embedding_qat_module_mappings().keys())

    def get_qconfig(self):
        qconfig = (QConfigMapping().set_global(get_default_qat_qconfig(self.backend))
            .set_object_type(nn.Embedding, 
                                float_qparams_weight_only_qconfig if self.allow_emb else None)
            .set_object_type(nn.EmbeddingBag, 
                                float_qparams_weight_only_qconfig if self.allow_emb else None)
            .set_object_type(nn.MultiheadAttention, None)
        )
        return get_flattened_qconfig_dict(qconfig)  

    def _prepare_model(self, input_data: InputData, supplementary_data=None):
        model = (input_data.target
            if "predict" not in vars(input_data)
            else input_data.predict)
        if not self.inplace:
            model = deepcopy(model)
        ### TODO add check whether it is 
        # uninplace(model)
        model = ParentalReassembler.reassemble(model, self.params.get('additional_mapping', None))
        propagate_qconfig_(model, self.qconfig)
        b = self._get_example_input(input_data).to(self.device)
        QDQWrapper.add_quant_entry_exit(model, b, allow=self.allow, mode='qat')
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
        self.device = torch.device('cpu')
        self.trainer.device = self.device
        self.model.to(self.device)
        convert(self.model, inplace=True)
        self.optimised_model = self.model
        self.model._is_quantized = True

    def predict_for_fit(self, input_data: InputData, output_mode: str = 'fedcore'):
        self.trainer.model = (
            self.optimised_model if output_mode == 'fedcore' else self.model
        )
        return self.trainer.predict(input_data, output_mode)

    def predict(self, input_data: InputData, output_mode: str = 'fedcore'):
        self.trainer.model = (
            self.optimised_model if output_mode == 'fedcore' else self.model
        )
        return self.trainer.predict(input_data, output_mode)
