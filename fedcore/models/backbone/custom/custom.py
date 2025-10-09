from typing import Any, Optional, Callable
import torch
from torch import nn, optim, Tensor

from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.data.data import InputData
from fedcore.architecture.computational.devices import default_device
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
from fedcore.repository.constant_repository import (
    CROSS_ENTROPY,
    MULTI_CLASS_CROSS_ENTROPY,
    MSE,
)


class TorchModel:
    def __init__(self, path_to_model: str, custom_callable_object: Callable = None):
        if custom_callable_object is None:
            self.model = torch.load(path_to_model, map_location=default_device())
        else:
            self.model = custom_callable_object
            self.model.load_state_dict(torch.load(path_to_model, 
                                                  weights_only=True, 
                                                  map_location=default_device()))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implements the forward method of the model and returns predictions."""
        x = x.to(default_device())
        return self.model(x)


class CustomModel(BaseNeuralModel):
    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)
        print(params)
        self.path_to_model = params.get("path_to_model", None)
        self.framework = params.get("framework", "pytorch")
        self.custom_callable_object = params.get("custom_callable_object", None)
        self._init_model()

    def _init_model(self):
        if self.framework == "pytorch":
            self.model = TorchModel(
                path_to_model=self.path_to_model,
                custom_callable_object=self.custom_callable_object
            )
            self.model_for_inference = TorchModel(
                path_to_model=self.path_to_model,
                custom_callable_object=self.custom_callable_object
            ).model
            self.model = self.model.model

    def _prepare_model(self, input_data):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        if self.task_type == "classification":
            if input_data.shape[0] == 2:
                loss_fn = CROSS_ENTROPY()
            else:
                loss_fn = MULTI_CLASS_CROSS_ENTROPY()
        else:
            loss_fn = MSE()
        return loss_fn, optimizer

    def _predict_model(self, x_test):
        self.model.eval()
        x_test = Tensor(x_test).to(default_device())
        pred = self.model(x_test)
        return self._convert_predict(pred)
    
    def load_model(self, input_data: InputData, path: str):
        self._prepare_model(input_data)
        self.model.eval()