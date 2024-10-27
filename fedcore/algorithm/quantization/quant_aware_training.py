from typing import Optional

import torch
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from torch import optim, nn

from fedcore.algorithm.base_compression_model import BaseCompressionModel
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
from fedcore.neural_compressor.config import QuantizationAwareTrainingConfig
from fedcore.neural_compressor.training import prepare_compression


class QuantAwareModel(BaseCompressionModel):
    """Class responsible for Pruning model implementation.
    Example:
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)
        self.epochs = params.get("epochs", 5)
        self.loss = params.get("loss", nn.CrossEntropyLoss())
        self.learning_rate = params.get("lr", 3e-3)
        self.optimizer = params.get("optimizer", optim.AdamW)
        self.scheduler = params.get("optimizer", torch.optim.lr_scheduler.StepLR)
        self.quantisation_config = params.get(
            "quantisation_config", QuantizationAwareTrainingConfig()
        )
        self.quantisation_model = prepare_compression
        self.device = default_device()
        self.trainer = BaseNeuralModel(params)

    def __repr__(self):
        return "QuantisationAware"

    def _init_model(self, input_data):
        self.quantisation_model = self.quantisation_model(
            input_data.target, self.quantisation_config
        )
        self.trainer.model = self.quantisation_model.model

    def fit(self, input_data: InputData):
        self._init_model(input_data)
        self.optimised_model = self.trainer.fit(
            input_data,
            supplementary_data={
                "strategy": "quant_aware",
                "callback": self.quantisation_model,
            },
        )

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
