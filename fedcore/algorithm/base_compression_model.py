import os

import torch_pruning as tp
from typing import Optional
import numpy as np
import torch
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters

from fedcore.data.data import CompressionInputData


class BaseCompressionModel:
    """Class responsible for NN model implementation.

    Attributes:
        self.num_features: int, the number of features.

    Example:
        To use this operation you can create pipeline as follows::
            from fedot.core.pipelines.pipeline_builder import PipelineBuilder
            from examples.fedot.fedot_ex import init_input_data
            from fedot_ind.tools.loader import DataLoader
            from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

            train_data, test_data = DataLoader(dataset_name='Ham').load_data()
            with IndustrialModels():
                pipeline = PipelineBuilder().add_node('resnet_model').add_node('rf').build()
                input_data = init_input_data(train_data[0], train_data[1])
                pipeline.fit(input_data)
                features = pipeline.predict(input_data)
                print(features)
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        self.num_classes = params.get('num_classes', None)
        self.epochs = params.get('epochs', 10)
        self.batch_size = params.get('batch_size', 16)
        self.activation = params.get('activation', 'ReLU')
        self.learning_rate = 0.001
        self.model = None
        self.model_for_inference = None

    def _save_and_clear_cache(self):
        try:
            # the pruned model
            state_dict = tp.state_dict(self.model)
            torch.save(state_dict, 'pruned.pth')
            new_model = self.model.eval()
            # load the pruned state_dict into the unpruned model.
            loaded_state_dict = torch.load('pruned.pth', map_location='cpu')
            tp.load_state_dict(new_model, state_dict=loaded_state_dict)
        except Exception:
            self.model.zero_grad()  # Remove gradients
            prefix = f'model_{self.__repr__()}_activation_{self.activation}_epochs_{self.epochs}_bs_{self.batch_size}.pt'
            torch.save(self.model.state_dict(), prefix)
            del self.model
            with torch.no_grad():
                torch.cuda.empty_cache()
            self.model = self.model_for_inference.model.to(torch.device('cpu'))
            self.model.load_state_dict(torch.load(
                prefix, map_location=torch.device('cpu')))
            os.remove(prefix)

    def _fit_model(self, ts: CompressionInputData, split_data: bool = False):
        pass

    def _predict_model(self, x_test, output_mode: str = 'default'):
        pass

    def fit(self,
            input_data: CompressionInputData):
        """
        Method for feature generation for all series
        """
        self.num_classes = input_data.num_classes
        self.target = input_data.target
        self.task_type = input_data.task
        self._fit_model(input_data)
        self._save_and_clear_cache()

    def predict(self,
                input_data: CompressionInputData, output_mode: str = 'default') -> np.array:
        """
        Method for feature generation for all series
        """
        return self._predict_model(input_data, output_mode)

    def predict_for_fit(self,
                        input_data: CompressionInputData, output_mode: str = 'default') -> np.array:
        """
        Method for feature generation for all series
        """
        return self._predict_model(input_data, output_mode)
