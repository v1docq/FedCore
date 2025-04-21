from copy import deepcopy

import numpy as np
from typing import Callable, Union
from pymonad.either import Either

from fedcore.architecture.comptutaional.devices import default_device
from fedcore.architecture.dataset.api_loader import ApiLoader
from fedcore.data.data import CompressionInputData
from fedcore.repository.constanst_repository import DEFAULT_TORCH_DATASET


def convert_callable_loader(input_data):
    train_data = CompressionInputData(
        val_dataloader=input_data['val_dataloader'],
        train_dataloader=input_data['train_dataloader'],
        test_dataloader=input_data['test_dataloader']
    )
    train_data.supplementary_data.is_auto_preprocessed = True
    target_list = []
    for batch in train_data.test_dataloader:
        batch_list = [b.to(default_device()) for b in batch]
        target_list.append(batch_list[-1].cpu().detach().numpy().squeeze())
    train_data.target = np.concatenate(target_list)
    return train_data


def load_data(source: Union[str, Callable] = None, loader_params: dict = None):
    source_is_dir = isinstance(source, str)
    source_is_loader = isinstance(source, Callable)
    data_loader = ApiLoader(load_source=source, loader_params=loader_params)
    loader_type = "benchmark" if source_is_dir and source in DEFAULT_TORCH_DATASET.keys() else loader_params['data_type']

    train_data = Either(value=loader_params, monoid=[loader_type, source_is_loader]). \
        either(left_function=data_loader.load_data, right_function=lambda custom_params: source(custom_params))
    if source_is_loader:
        train_data = convert_callable_loader(train_data)
    return train_data
