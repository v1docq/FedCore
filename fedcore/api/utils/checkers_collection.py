import logging
from copy import deepcopy
from typing import Callable

import torch
import numpy as np
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import (
    Task,
    TaskTypesEnum,
)

from fedcore.data.data import CompressionInputData, CompressionOutputData
from fedcore.models.backbone.backbone_loader import load_backbone
from fedcore.repository.config_repository import TASK_MAPPING


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

    def __init__(self, peft_task=None, optimised_model=None, learning_params=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.task = peft_task
        self.model = optimised_model
        self.learning_params = learning_params
        self._init_dummy_val()

    def _init_dummy_val(self):
        self.fedot_dummy_task = Task(TaskTypesEnum.classification)
        self.fedot_dummy_idx = np.arange(1)
        self.fedot_dummy_datatype = DataTypesEnum.image

    def _init_input_data(self, input_data: [InputData, CompressionInputData, dict] = None, manually_done=False) -> None:
        """Initializes the `input_data` attribute based on its type.

        If a tuple (X, y) is provided, it converts it to a Fedot InputData object
        with appropriate data types and task information. If an existing InputData
        object is provided, it checks if it requires further initialization.

        Raises:
            ValueError: If the input data format is invalid.

        """
        model_params = self.learning_params.get('model_params', None)
        if isinstance(input_data, dict):
            compression_dataset = CompressionInputData(
                calib_dataloader=input_data['val_dataloader'],
                train_dataloader=input_data['train_dataloader'],
                test_dataloader=input_data['test_dataloader'],
                input_dim=model_params['input_dim'] if model_params is not None else 1
            )
            compression_dataset.supplementary_data.is_auto_preprocessed = True
        else:
            if input_data.supplementary_data.is_auto_preprocessed:
                compression_dataset = input_data
            else:
                compression_dataset, torch_model = input_data

        if self.model is not None:
            if isinstance(self.model, str):
                torch_model = load_backbone(self.model)
            elif isinstance(self.model, dict):
                torch_model = load_backbone(torch_model=self.model['model_type'],
                                            model_params=self.learning_params)
                torch_model.load_model(self.model['path_to_model'])
            elif isinstance(self.model, Callable):
                torch_model = self.model
            else:
                torch_model = None
            compression_dataset.target = torch_model

        self.input_data = InputData(
            features=compression_dataset,  # CompressionInputData object
            idx=self.fedot_dummy_idx,  # dummy value
            features_names=compression_dataset.num_classes,  # CompressionInputData attribute
            task=self.fedot_dummy_task,  # dummy value
            data_type=self.fedot_dummy_datatype,  # dummy value
            target=torch_model,  # model for compression
            supplementary_data=compression_dataset.supplementary_data,
        )
        self.input_data.supplementary_data.is_auto_preprocessed = True

    def _check_dataloader(self):
        """Checks and preprocesses the features in the input data.

        - Replaces NaN and infinite values with 0.
        - Converts features to torch format using NumpyConverter.

        """

    def _check_optimised_model(self):
        """Checks and preprocesses the features in the input data.

        - Replaces NaN and infinite values with 0.
        - Converts features to torch format using NumpyConverter.

        """

    def check_input_data(self, input_data: [InputData, CompressionInputData] = None) -> InputData:
        """Checks and preprocesses the input data for Fedot AutoML.

        Performs the following steps:
            1. Initializes the `input_data` attribute based on its type.
            2. Checks and preprocesses the features (replacing NaNs, converting to torch format).
            3. Checks and preprocesses the target variable (encoding labels, casting to float).

        Returns:
            InputData: The preprocessed and initialized Fedot InputData object.

        """
        self._init_input_data(input_data)
        self._check_dataloader()
        self._check_optimised_model()
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
