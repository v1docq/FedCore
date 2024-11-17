from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from fedcore.architecture.abstraction.delegator import DelegatorFactory
from fedcore.data.data import CompressionInputData
from fedcore.repository.constanst_repository import FEDOT_TASK


def get_compression_input(
    model,
    train_dataloader,
    calib_dataloader,
    task="classification",
    num_classes=None,
    train_loss=None,
):
    input_data = CompressionInputData(
        features=np.zeros((2, 2)),
        train_dataloader=train_dataloader,
        calib_dataloader=calib_dataloader,
        task=FEDOT_TASK[task],
        num_classes=num_classes or len(train_dataloader.dataset.get('classes', 0)),
        target=model,
    )
    input_data.supplementary_data.is_auto_preprocessed = True
    input_data.supplementary_data.col_type_ids = {"loss": train_loss}
    return input_data


def check_multivariate_data(data: pd.DataFrame) -> tuple:
    """
    Checks if the provided pandas DataFrame contains multivariate data.

    Args:
        data (pd.DataFrame): The DataFrame to be analyzed.

    Returns:
        bool: True if the DataFrame contains multivariate data (nested columns), False otherwise.
    """
    if not isinstance(data, pd.DataFrame):
        return len(data.shape) > 2, data
    else:
        return isinstance(data.iloc[0, 0], pd.Series), data.values


def init_input_data(
    X: pd.DataFrame, y: Optional[np.ndarray], task: str = "classification"
) -> InputData:
    """
    Initializes a Fedot InputData object from input features and target.

    Args:
        X: The DataFrame containing features.
        y: The NumPy array containing target values.
        task: The machine learning task type ("classification" or "regression"). Defaults to "classification".

    Returns:
        InputData: The initialized Fedot InputData object.

    """

    is_multivariate_data, features = check_multivariate_data(X)
    task_dict = {
        "classification": Task(TaskTypesEnum.classification),
        "regression": Task(TaskTypesEnum.regression),
    }

    if y is not None and isinstance(y[0], np.str_) and task == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    elif y is not None and isinstance(y[0], np.str_) and task == "regression":
        y = y.astype(float)

    data_type = DataTypesEnum.image if is_multivariate_data else DataTypesEnum.table
    input_data = InputData(
        idx=np.arange(len(X)),
        features=np.array(features.tolist()).astype(float),
        target=y.reshape(-1, 1) if y is not None else y,
        task=task_dict[task],
        data_type=data_type,
    )

    if input_data.target is not None:
        if task == "regression":
            input_data.target = input_data.target.squeeze()
        elif task == "classification":
            input_data.target[input_data.target == -1] = 0

    # Replace NaN and infinite values with 0 in features
    input_data.features = np.where(
        np.isnan(input_data.features), 0, input_data.features
    )
    input_data.features = np.where(
        np.isinf(input_data.features), 0, input_data.features
    )

    return input_data


class DataLoaderHandler:
    __non_included_kwargs = {'check_worker_number_rationality'}

    @staticmethod
    def limited_generator(gen, max_batches, enumerate=False):
        i = 0
        for elem in gen:
            if i >= max_batches:
                break
            yield (i, elem) if enumerate else elem
            i += 1

    @staticmethod
    def __X2Xy(collate_fn):
        def wrapped(batch, *args, **kwargs):
            return collate_fn(batch, *args, **kwargs), [None] * len(batch)
        return wrapped

    collate_modes = {
        'X2Xy': __X2Xy,
        'pass': lambda x: x
    }

    @classmethod
    def __clean_dict(cls, d: dict, is_iterable=False):
      d = {attr: val for attr, val in d.items() if not (attr.startswith('_') or attr in cls.__non_included_kwargs)}

      if is_iterable:
        d.pop('sampler', None)

      if any((d.get(k, False) for k in ('batch_size', 'shuffle', 'sampler', 'drop_last'))):
        d.pop('batch_sampler', None)
      if any((d.get(k, False) for k in ('batch_size', 'shuffle', 'sampler', 'drop_last'))):
        d.pop('batch_sampler', None)

      if d.get('batch_size', None):
        d.pop('drop_last', None)
      if d.get('drop_last', False):
        d.pop('batch_size', None)
      return d

    @classmethod
    def check_convert(cls, dataloader: DataLoader, mode: Union[None, str, Callable] = None, max_batches:int = None, enumerate=False) -> DataLoader:
        batch = cls.__get_batch_sample(dataloader)
        dl_params = {attr: getattr(dataloader, attr) for attr in dir(dataloader)}
        dl_params = cls.__clean_dict(dataloader.__dict__, hasattr(dataloader.dataset, '__iter__'))
        modified1, dl_params = cls.__substitute_collate_fn(dl_params, batch, mode)
        if modified1:
          dataloader = DataLoader(**dl_params)
        if max_batches or enumerate:
          dataloader = cls.limit_batches(dataloader, max_batches, enumerate)
        return dataloader

    @classmethod
    def __get_batch_sample(cls, dataloader: DataLoader):
        for b in dataloader:
            return b

    @classmethod
    def __substitute_collate_fn(cls, dl_params: dict, batch: Any, mode: Union[None, str, Callable]):
        modified = True
        type_ = mode
        if isinstance(mode, Callable):
            collate_fn = mode
        elif isinstance(mode, str):
            collate_fn = cls.collate_modes[mode]
        else:
            type_ = cls.__check_type(batch)
            collate_fn = cls.collate_modes[type_]
        if type_ == 'pass':
            modified = False
        dl_params['collate_fn'] = collate_fn(dl_params['collate_fn'])
        return modified, dl_params

    @staticmethod
    def limit_batches(dataloader, max_batches, enumerate=False):
      if max_batches is None and not enumerate:
          return dataloader
      return DataLoaderHandler.__substitute_iter(dataloader, max_batches, enumerate)
      
    @staticmethod
    def __substitute_iter(iterable, max_batches=None, enumerate=False):
        max_batches = max_batches or float('inf')
        def newiter(_):
            return iter(DataLoaderHandler.limited_generator(iterable, max_batches, enumerate))
        return DelegatorFactory.create_delegator_inst(iterable, {'__iter__': newiter})

    @staticmethod
    def __check_type(batch) -> str:
        return 'pass' if isinstance(batch, (tuple, list)) else 'X2Xy'
