import logging
from typing import Union

import numpy as np
import torch
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from pymonad.either import Either
from torch.utils.data import DataLoader
from tqdm import tqdm

from fedcore.architecture.utils.loader import collate
from fedcore.data.data import CompressionInputData, CompressionOutputData
from fedcore.repository.constanst_repository import FEDOT_TASK
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

    """

    def __init__(self, input_data: Union[InputData, tuple] = None, task=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.input_data = input_data
        self.task = task

    def _check_od_dataset(self, dataset_type):
        list(self.input_data.classes.keys())
        target = []
        loader = DataLoader(
            self.input_data, batch_size=1, shuffle=False, collate_fn=collate
        )
        gen = (
            target.append((targets[0]["boxes"], targets[0]["labels"]))
            for i, (images, targets) in enumerate(tqdm(loader, desc="Fitting"))
        )
        return gen

    def _check_directory_dataset(self, dataset_type):
        if dataset_type == "fedcore":
            return self.input_data[0], self.input_data[1]
        else:
            path_to_files, path_to_labels, path_to_model = (
                self.input_data[0],
                self.input_data[1],
                self.input_data[2],
            )
            torch_dataloader = self.cv_dataset(path_to_files, path_to_labels)
            if not path_to_model.__contains__("pt"):
                torch_model = BACKBONE_MODELS[path_to_model]
            else:
                torch_model = torch.load(
                    path_to_model, map_location=torch.device("cpu")
                )

            compression_dataset = CompressionInputData(
                features=np.zeros((2, 2)),
                num_classes=torch_dataloader.num_classes,
                calib_dataloader=torch_dataloader,
                target=torch_model,
            )
        return compression_dataset, torch_model

    def _init_input_data(self, manually_done=False) -> None:
        """Initializes the `input_data` attribute based on its type.

        If a tuple (X, y) is provided, it converts it to a Fedot InputData object
        with appropriate data types and task information. If an existing InputData
        object is provided, it checks if it requires further initialization.

        Raises:
            ValueError: If the input data format is invalid.

        """
        if not manually_done:
            object_detection_scenario = self.task == "detection"
            fedcore_scenario = isinstance(
                self.input_data[0], (CompressionInputData, CompressionOutputData)
            )
            custom_scenario = "fedcore" if fedcore_scenario else "directory"

            compression_dataset, torch_model = Either(
                value="detection", monoid=[custom_scenario, object_detection_scenario]
            ).either(
                left_function=lambda dataset_type: self._check_directory_dataset(
                    dataset_type
                ),
                right_function=lambda dataset_type: self._check_od_dataset(
                    dataset_type
                ),
            )
        else:
            compression_dataset, torch_model = self.input_data

        self.input_data = InputData(
            features=compression_dataset,  # CompressionInputData object
            idx=np.arange(1),  # dummy value
            features_names=compression_dataset.num_classes,  # CompressionInputData attribute
            task=FEDOT_TASK["classification"],  # dummy value
            data_type=DataTypesEnum.image,  # dummy value
            target=torch_model,  # model for compression
            supplementary_data=compression_dataset.supplementary_data,
        )
        self.input_data.supplementary_data.is_auto_preprocessed = True

    def _check_input_data_features(self):
        """Checks and preprocesses the features in the input data.

        - Replaces NaN and infinite values with 0.
        - Converts features to torch format using NumpyConverter.

        """

    def _check_input_data_target(self):
        """Checks and preprocesses the features in the input data.

        - Replaces NaN and infinite values with 0.
        - Converts features to torch format using NumpyConverter.

        """

    def check_input_data(self, manually_done=False) -> InputData:
        """Checks and preprocesses the input data for Fedot AutoML.

        Performs the following steps:
            1. Initializes the `input_data` attribute based on its type.
            2. Checks and preprocesses the features (replacing NaNs, converting to torch format).
            3. Checks and preprocesses the target variable (encoding labels, casting to float).

        Returns:
            InputData: The preprocessed and initialized Fedot InputData object.

        """
        self._init_input_data(manually_done)
        self._check_input_data_features()
        self._check_input_data_target()
        return self.input_data
