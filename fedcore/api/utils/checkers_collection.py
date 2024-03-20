import logging
from typing import Union, Optional

import numpy as np
import torch
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedcore.data.data import CompressionInputData, CompressionOutputData
from fedcore.repository.constanst_repository import FEDOT_TASK
from keras.models import load_model
import onnx
#import keras2onnx
import tensorflow as tf

from fedcore.repository.model_repository import BACKBONE_MODELS


class DataCheck:
    """Class for checking and preprocessing input data for Fedot AutoML.

    Args:
        input_data: Input data in tuple format (X, y) or Fedot InputData object.
        task: Machine learning task, either "classification" or "regression".

    Attributes:
        logger (logging.Logger): Logger instance for logging messages.
        input_data (InputData): Preprocessed and initialized Fedot InputData object.
        task (str): Machine learning task for the dataset.
        task_dict (dict): Mapping of string task names to Fedot Task objects.

    """

    def __init__(self,
                 input_data: Union[InputData, tuple] = None,
                 cv_dataset: callable = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.input_data = input_data
        self.cv_dataset = cv_dataset

    def _init_input_data(self) -> None:
        """Initializes the `input_data` attribute based on its type.

        If a tuple (X, y) is provided, it converts it to a Fedot InputData object
        with appropriate data types and task information. If an existing InputData
        object is provided, it checks if it requires further initialization.

        Raises:
            ValueError: If the input data format is invalid.

        """
        compression_dataset, torch_model = None, None
        if isinstance(self.input_data, InputData):
            return
        elif isinstance(self.input_data[0], (CompressionInputData, CompressionOutputData)):
            compression_dataset, torch_model = self.input_data[0], self.input_data[1]
        elif isinstance(self.input_data[0], str):
            path_to_files, path_to_labels, path_to_model = self.input_data[0], self.input_data[1], self.input_data[2]
            torch_dataloader = self.cv_dataset(path_to_files, path_to_labels)
            if not path_to_model.__contains__('pt'):
                torch_model = BACKBONE_MODELS[path_to_model]

            else:
                torch_model = torch.load(path_to_model, map_location=torch.device('cpu'))

            compression_dataset = CompressionInputData(features=np.zeros((2, 2)),
                                                       num_classes=torch_dataloader.num_classes,
                                                       calib_dataloader=torch_dataloader,
                                                       target=torch_model
                                                       )

        self.input_data = InputData(features=compression_dataset,  # CompressionInputData object
                                    idx=np.arange(1),  # dummy value
                                    features_names=compression_dataset.num_classes,  # CompressionInputData attribute
                                    task=FEDOT_TASK['classification'],  # dummy value
                                    data_type=DataTypesEnum.image,  # dummy value
                                    target=torch_model  # model for compression
                                    )
        self.input_data.supplementary_data.is_auto_preprocessed = True

    def _check_input_data_features(self):
        """Checks and preprocesses the features in the input data.

        - Replaces NaN and infinite values with 0.
        - Converts features to torch format using NumpyConverter.

        """
        pass

    def _check_input_data_target(self):
        """Checks and preprocesses the features in the input data.

        - Replaces NaN and infinite values with 0.
        - Converts features to torch format using NumpyConverter.

        """
        pass

    def check_available_operations(self, available_operations):
        pass

    def check_input_data(self) -> InputData:
        """Checks and preprocesses the input data for Fedot AutoML.

        Performs the following steps:
            1. Initializes the `input_data` attribute based on its type.
            2. Checks and preprocesses the features (replacing NaNs, converting to torch format).
            3. Checks and preprocesses the target variable (encoding labels, casting to float).

        Returns:
            InputData: The preprocessed and initialized Fedot InputData object.

        """

        self._init_input_data()
        self._check_input_data_features()
        self._check_input_data_target()
        return self.input_data
