import torch

from fedcore.repository.model_repository import BACKBONE_MODELS
from typing import Callable, Union


def load_backbone(torch_model: Union[str, Callable], model_impl: Callable = None, model_params:dict = None):
    is_path_to_torch_weight = isinstance(torch_model,str)
    is_backbone_torch = torch_model in BACKBONE_MODELS.keys()
    if is_backbone_torch:
        torch_model = BACKBONE_MODELS[torch_model](model_params) if is_backbone_torch else torch_model
    elif is_path_to_torch_weight:
        torch_model = model_impl.load_state_dict(torch.load(torch_model, weights_only=True))
    else:
        torch_model = torch_model

    return torch_model
