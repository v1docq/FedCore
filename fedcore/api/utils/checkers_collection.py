import logging
from copy import deepcopy
from typing import Callable, Union

import torch
import numpy as np
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import (
    Task,
    TaskTypesEnum,
)

from fedcore.architecture.comptutaional.devices import default_device
from fedcore.data.data import CompressionInputData, CompressionOutputData
from fedcore.models.backbone.backbone_loader import load_backbone
from fedcore.repository.config_repository import TASK_MAPPING
from pymonad.maybe import Maybe

import torch.optim.adam


class DataCheck:
    """Class for checking and preprocessing input data for Fedot AutoML.

    Args:
        input_data: Input data in tuple format (X, y) or Fedot InputData object.
        task: Machine learning task, either "classification" or "regression".

    Attributes:
        logger (logging.Logger): Logger instance for logging messages.
        input_data (InputData): Preprocessed and initialized Fedot InputData object.
        task (str): Machine learning task for the dataset.

    """

    def __init__(self, peft_task=None, model=None, learning_params=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.task = peft_task
        self.model = model
        self.learning_params = learning_params
        self._init_dummy_val()

    def _init_dummy_val(self):
        self.fedot_dummy_task = Task(TaskTypesEnum.classification)
        self.fedot_dummy_idx = np.arange(1)
        self.fedot_dummy_datatype = DataTypesEnum.image

    def init_model_from_backbone(self, input_data: Union[CompressionInputData, InputData]):
        model_is_pretrain_torch_backbone = isinstance(self.model, str)
        model_is_pretrain_backbone_with_weights = isinstance(self.model, dict)
        model_is_custom_callable_object = isinstance(self.model, Callable)

        if model_is_pretrain_torch_backbone or model_is_pretrain_backbone_with_weights:
            torch_model = load_backbone(torch_model=self.model,
                                        model_params=self.learning_params)
            torch_model = self._check_optimised_model(torch_model, input_data)
            if model_is_pretrain_backbone_with_weights:
                if hasattr(torch_model, 'load_model'):
                    torch_model.load_model(self.model['path_to_model'])
                else:
                    loaded_state_dict = torch.load(self.model['path_to_model'], weights_only=True,
                                                   map_location=default_device())
                    verified_state_dict = self._check_state_dict(loaded_state_dict, input_data)
                    torch_model.load_state_dict(verified_state_dict)
        elif model_is_custom_callable_object:
            torch_model = self.model

        return torch_model

    def init_input_data(self, compression_dataset: CompressionInputData = None, manually_done=False) -> None:
        """Initializes the `input_data` attribute based on its type.

        If a tuple (X, y) is provided, it converts it to a Fedot InputData object
        with appropriate data types and task information. If an existing InputData
        object is provided, it checks if it requires further initialization.

        Raises:
            ValueError: If the input data format is invalid.

        """
        if self.learning_params is not None:
            model_params = self.learning_params.model_architecture
            if any([model_params.input_dim is None, model_params.output_dim is None]):
                model_params.input_dim = compression_dataset.input_dim
                model_params.output_dim = compression_dataset.num_classes

        input_data = InputData(
            features=compression_dataset,  # CompressionInputData object
            idx=self.fedot_dummy_idx,  # dummy value
            task=compression_dataset.task,
            data_type=self.fedot_dummy_datatype,  # dummy value
            supplementary_data=compression_dataset.supplementary_data,
        )
        torch_model = self.init_model_from_backbone(input_data)

        input_data.target = torch_model  # model for compression
        input_data.features.target = torch_model  # model for compression
        input_data.supplementary_data.is_auto_preprocessed = True
        return input_data

    def _check_dataloader(self, input_data: InputData):
        """Checks and preprocesses the features in the input data.

        - Replaces NaN and infinite values with 0.
        - Converts features to torch format using NumpyConverter.

        """
        return input_data

    def _check_optimised_model(self, model: Callable, input_data: Union[CompressionInputData, InputData]):
        """Checks and preprocesses the features in the input data.

        - Replaces NaN and infinite values with 0.
        - Converts features to torch format using NumpyConverter.

        """
        if not input_data.task.task_type.value == 'classification':
            return model
        else:
            model_layers = list(model.modules())
            output_layer = model_layers[-1]
            if hasattr(input_data.features, 'num_classes'):
                n_classes = input_data.features.num_classes
            else:
                n_classes = input_data.num_classes
            if output_layer.weight.shape[0] != n_classes:
                output_layer.weight = torch.nn.Parameter(output_layer.weight[:n_classes, :])
                output_layer.bias = torch.nn.Parameter(output_layer.bias[:n_classes])
                output_layer.out_features = n_classes
        return model

    def _check_state_dict(self, state_dict: dict, input_data: InputData):
        if not input_data.task.task_type.value == 'classification':
            return state_dict
        else:
            layers = list(state_dict.keys())
            output_layer_weight = layers[-2]
            output_layer_bias = layers[-1]
            n_classes = input_data.features.num_classes
            if state_dict[output_layer_weight].shape[0] != n_classes:
                state_dict[output_layer_weight] = torch.nn.Parameter(state_dict[output_layer_weight][:n_classes, :])
                state_dict[output_layer_bias] = torch.nn.Parameter(state_dict[output_layer_bias][:n_classes])
        return state_dict

    def check_input_data(self, input_data: [InputData, CompressionInputData] = None) -> InputData:
        """Checks and preprocesses the input data for Fedot AutoML.

        Performs the following steps:
            1. Initializes the `input_data` attribute based on its type.
            2. Checks and preprocesses the features (replacing NaNs, converting to torch format).
            3. Checks and preprocesses the target variable (encoding labels, casting to float).

        Returns:
            InputData: The preprocessed and initialized Fedot InputData object.

        """
        self.input_data = Maybe.insert(self.init_input_data(input_data)). \
            then(self._check_dataloader). \
            maybe(None, lambda data: data)
        return self.input_data


class ApiConfigCheck:
    def __init__(self):
        pass

    def compare_configs(self, original, updated):
        """Compares two nested dictionaries"""

        changes = []

        def recursive_compare(orig, upd, path):
            all_keys = orig.keys() | upd.keys()
            for key in all_keys:
                orig_val = orig.get(key, "<MISSING>")
                upd_val = upd.get(key, "<MISSING>")

                if isinstance(orig_val, dict) and isinstance(upd_val, dict):
                    recursive_compare(orig_val, upd_val, path + [key])
                elif orig_val != upd_val:
                    changes.append(f"{' -> '.join(map(str, path + [key]))} -> Changed value {orig_val} to {upd_val}")

        for sub_config in original.keys():
            if sub_config in updated:
                recursive_compare(original[sub_config], updated[sub_config], [sub_config])
            else:
                changes.append(f"{sub_config} -> Removed completely")

        for i in changes:
            print('>>>', i)
        return "\n".join(changes) if changes else "No changes detected."

    def update_config_with_kwargs(self, config_to_update, **kwargs):
        """ Recursively update config dictionary with provided keyword arguments. """

        # prevent inplace changes to the original config
        config = deepcopy(config_to_update)

        def recursive_update(d, key, value):
            if key in d:
                d[key] = value
                # print(f'Updated {key} with {value}')
            for k, v in d.items():
                if isinstance(v, dict):
                    recursive_update(v, key, value)

        # we select automl problem
        # assert 'task' in kwargs, 'Problem type is not provided'
        # problem_type = kwargs['task']
        # config['automl_config'] = TASK_MAPPING[problem_type]

        # change MEGA config with keyword arguments
        for param, value in kwargs.items():
            recursive_update(config, param, value)

        return config
