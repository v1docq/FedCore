from fedcore.algorithm.base_compression_model import BaseCompressionModel
from copy import deepcopy

import numpy as np
from fedot.core.data.data import InputData
from fedcore.algorithm.base_compression_model import BaseCompressionModel
import torch_pruning as tp
from typing import Optional
import torch
from fedot.core.operations.operation_parameters import OperationParameters
from torchvision.models import resnet18
from fedcore.models.model_repository import PRUNER_MODELS
from fedcore.neural_compressor import PostTrainingQuantConfig
from fedcore.neural_compressor import quantization


class QuantPostModel(BaseCompressionModel):
    """Class responsible for Pruning model implementation.
    Example:
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)
        self.epochs = params.get('epochs', 5)
        self.quantisation_config = params.get('quantisation_config', PostTrainingQuantConfig())
        self.quantisation_model = quantization

    def __repr__(self):
        return 'PostQuantisation'

    def fit(self,
            input_data: InputData):
        self.quantisation_model.fit(model=input_data.supplementary_data['model_to_quant'],
                                    conf=self.quantisation_config,
                                    calib_dataloader=input_data.supplementary_data['validation_data']
                                    )

    def predict_for_fit(self,
                        input_data: InputData, output_mode: str = 'default') -> np.array:
        return self.model

    def predict(self,
                input_data: InputData, output_mode: str = 'default') -> np.array:
        self.quantisation_model.save("./output")
        return self.predict_for_fit(input_data, output_mode)
