from fedcore.algorithm.base_compression_model import BaseCompressionModel
from copy import deepcopy

import numpy as np
from fedot.core.data.data import InputData
from fedcore.algorithm.base_compression_model import BaseCompressionModel
from typing import Optional
from fedot.core.operations.operation_parameters import OperationParameters

from fedcore.data.data import CompressionInputData
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
            input_data: CompressionInputData):
        self.model = self.quantisation_model.fit(model=input_data.target,
                                                 conf=self.quantisation_config,
                                                 calib_dataloader=input_data.calib_dataloader
                                                 )

    def predict_for_fit(self,
                        input_data: InputData, output_mode: str = 'default') -> np.array:
        return self.model

    def predict(self,
                input_data: InputData, output_mode: str = 'default') -> np.array:
        return self.predict_for_fit(input_data, output_mode)
