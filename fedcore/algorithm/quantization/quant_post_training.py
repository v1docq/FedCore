from typing import Optional

from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters

from fedcore.algorithm.base_compression_model import BaseCompressionModel
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
from fedcore.neural_compressor import quantization
from fedcore.neural_compressor.config import PostTrainingQuantConfig
from fedcore.repository.constanst_repository import default_device


class QuantPostModel(BaseCompressionModel):
    """Class responsible for Pruning model implementation.
    Example:
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)
        self.epochs = params.get("epochs", 5)
        self.learning_rate = params.get("learning_rate", 0.001)
        self.quantisation_config = params.get(
            "quantisation_config", PostTrainingQuantConfig()
        )
        self.quantisation_model = quantization
        self.trainer = BaseNeuralModel(params)
        self.device = default_device()

    def __repr__(self):
        return "PostQuantisation"

    def fit(self, input_data: InputData):
        self.model = (
            input_data.target
            if "predict" not in vars(input_data)
            else input_data.predict
        )
        self.optimized_model = self.quantisation_model.fit(
            model=self.model,
            conf=self.quantisation_config,
            calib_dataloader=input_data.features.calib_dataloader,
        )
        self.optimized_model.to(default_device())

    def predict_for_fit(self, input_data: InputData, output_mode: str = "compress"):
        self.trainer.model = (
            self.optimized_model if output_mode == "compress" else self.model
        )
        return self.trainer.predict(input_data, output_mode)

    def predict(self, input_data: InputData, output_mode: str = "compress"):
        self.trainer.model = (
            self.optimized_model if output_mode == "compress" else self.model
        )
        return self.trainer.predict(input_data, output_mode)
