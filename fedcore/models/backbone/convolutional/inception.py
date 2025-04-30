from typing import Optional

from fastai.torch_core import Module
from fastcore.meta import delegates
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from torch import Tensor
from torch import nn, optim

from fedcore.architecture.comptutaional.devices import default_device
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
from fedcore.models.network_modules.layers.pooling_layers import GAP1d
from fedcore.models.network_modules.layers.special import InceptionBlock, InceptionModule

BASE_INCEPTIONTIME_PARAMS = {'activation': "ReLU",
                             'number_of_filters': 32,
                             'residual': True,
                             'base_kernel_size': 40,
                             'bottleneck': True}


@delegates(InceptionModule.__init__)
class InceptionTime(Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            depth: int = 6,
            custom_params: dict = None,
            **kwargs):
        super().__init__()
        self.inception_block = InceptionBlock(input_dim=input_dim,
                                              depth=depth,
                                              **custom_params)
        self.gap = GAP1d(1)
        self.fc = nn.Linear(custom_params['number_of_filters'] * 4, output_dim)

    def forward(self, x):
        x = self.inception_block(x)
        x = self.gap(x)
        x = self.fc(x)
        return x


class InceptionTimeModel(BaseNeuralModel):
    """Class responsible for InceptionTime model implementation.

    Attributes:
        self.num_features: int, the number of features.

    Example:
        To use this operation you can create pipeline as follows::
            from fedot.core.pipelines.pipeline_builder import PipelineBuilder
            from examples.fedot.fedot_ex import init_input_data
            from fedot_ind.tools.loader import DataLoader
            from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
            train_data, test_data = DataLoader(dataset_name='Lightning7').load_data()
            input_data = init_input_data(train_data[0], train_data[1])
            val_data = init_input_data(test_data[0], test_data[1])
            with IndustrialModels():
                pipeline = PipelineBuilder().add_node('inception_model', params={'epochs': 100,
                                                                                 'batch_size': 10}).build()
                pipeline.fit(input_data)
                target = pipeline.predict(val_data).predict
                metric = evaluate_metric(target=test_data[1], prediction=target)

    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)

        self.input_dim = params.model_architecture.get("input_dim", 1)
        self.output_dim = params.model_architecture.get("output_dim", 1)
        self.depth = params.model_architecture.get("depth", 3)
        self.custom_params = params.model_architecture.get("custom_params", BASE_INCEPTIONTIME_PARAMS)
        self._init_model()

    def __repr__(self):
        return "InceptionNN"

    def _init_model(self):
        self.model = InceptionTime(input_dim=self.input_dim, output_dim=self.output_dim,
                                   depth=self.depth, custom_params=self.custom_params)

    def forward(self, *inputs):
        return self.model(*inputs)
