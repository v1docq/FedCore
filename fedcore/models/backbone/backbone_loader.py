from fedcore.repository.model_repository import BACKBONE_MODELS
from typing import Callable, Union


def load_backbone(torch_model: Union[str, dict], model_params: dict = None):
    is_path_to_torch_weight = isinstance(torch_model, dict)
    if is_path_to_torch_weight:
        torch_model = BACKBONE_MODELS[torch_model['model_type']](model_params)
    else:
        torch_model = BACKBONE_MODELS[torch_model](model_params)
    return torch_model
