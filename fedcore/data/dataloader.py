from copy import deepcopy

from typing import Callable, Union
from pymonad.either import Either
from fedcore.architecture.dataset.api_loader import ApiLoader
from fedcore.repository.constanst_repository import DEFAULT_TORCH_DATASET


def load_data(source: Union[str, Callable] = None, loader_params: dict = None):
    source_is_dir = isinstance(source, str)
    source_is_loader = isinstance(source, Callable)
    data_loader = ApiLoader(load_source=source, loader_params=loader_params)
    if source_is_dir:
        loader_type = "benchmark" if source in DEFAULT_TORCH_DATASET.keys() else "directory"
    else:
        loader_type = "torchvision"

    train_data = Either(value=loader_params, monoid=[loader_type, source_is_loader]). \
        either(left_function=data_loader.load_data, right_function=lambda custom_params:source(custom_params))
    test_data = deepcopy(train_data)
    test_data.train_dataloader = train_data.calib_dataloader
    test_data.target = train_data.train_dataloader.dataset.targets
    return train_data, test_data
