"""This module contains functions for working with singular value decomposition.

Model decomposition, pruning by threshold, decomposed model loading.
"""

from typing import Callable, Optional

import torch
from torch.nn.modules import Module


from fedcore.models.network_impl.layers import IDecomposed
from fedcore.repository.constanst_repository import DECOMPOSABLE_LAYERS, FORWARD_MODE

def decompose_module(
    model: Module,
    decomposing_mode: Optional[str] = None,
    forward_mode: str = FORWARD_MODE,
) -> None:
    """Replace decomposable layers with their decomposed analogues in module (in-place).

    Args:
        model: Decomposable module.
        decomposing_mode: ``'channel'`` or ``'spatial'`` weights reshaping method.
            If ``None`` replace layers without decomposition.
        forward_mode: ``'one_layer'``, ``'two_layers'`` or ``'three_layers'`` forward pass calculation method.
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            decompose_module(
                module, decomposing_mode=decomposing_mode, forward_mode=forward_mode
            )
        decomposed_analogue = _map_decomposed_cls(module)
        if decomposed_analogue is not None:
            new_module = decomposed_analogue(
                module, decomposing_mode, forward_mode
            )
            setattr(model, name, new_module)


def _load_svd_params(model, state_dict, prefix="") -> None:
    """Loads state_dict to DecomposedConv2d layers in model."""
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            _load_svd_params(module, state_dict, prefix=f"{prefix}{name}.")

        if isinstance(module, IDecomposed):
            module.set_U_S_Vh(
                u=state_dict[f"{prefix}{name}.U"],
                s=state_dict[f"{prefix}{name}.S"],
                vh=state_dict[f"{prefix}{name}.Vh"],
            )


def load_svd_state_dict(
    model: Module,
    decomposing_mode: str,
    state_dict_path: str,
    forward_mode: str = FORWARD_MODE,
) -> None:
    """Loads SVD state_dict to model.

    Args:
        model: An instance of the base model.
        decomposing_mode: ``'channel'`` or ``'spatial'`` weights reshaping method.
        state_dict_path: Path to state_dict file.
        forward_mode: ``'one_layer'``, ``'two_layers'`` or ``'three_layers'`` forward pass calculation method.
    """
    state_dict = torch.load(state_dict_path, map_location="cpu")
    decompose_module(
        model=model, decomposing_mode=decomposing_mode, forward_mode=forward_mode
    )
    _load_svd_params(model, state_dict)
    model.load_state_dict(state_dict)


def _map_decomposed_cls(inst: torch.nn.Module) -> Optional[IDecomposed]:
    for decomposable, decomposed in DECOMPOSABLE_LAYERS.items():
        if isinstance(inst, decomposable) and not isinstance(inst, IDecomposed):
            return decomposed
