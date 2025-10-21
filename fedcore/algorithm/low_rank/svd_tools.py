"""This module contains helper functions for working with singular value
decomposition (SVD)–style factorization in FedCore.

It covers three common tasks:
    1) Model decomposition — replacing decomposable layers with their
       low-rank analogs (:func:`decompose_module`).
    2) Threshold/rank pruning — performed elsewhere, but this module provides
       utilities to load factorized weights back into a model.
    3) Loading a decomposed model from a saved ``state_dict``
       (:func:`load_svd_state_dict`).

Notes
-----
* Decomposition is applied recursively to all child modules.
* Only layers listed in ``DECOMPOSABLE_LAYERS`` are replaced; types from
  ``PROHIBIT_TO_DECOMPOSE`` are explicitly skipped.
* For composed forward (one/two/three-layer computation paths), see
  ``compose_mode`` argument used by decomposed layer implementations.
"""

from typing import Callable, Literal, Optional

import torch
from torch.nn.modules import Module

from fedcore.architecture.comptutaional.devices import extract_device
from fedcore.models.network_impl.decomposed_layers import IDecomposed
from fedcore.repository.constanst_repository import (
    DECOMPOSABLE_LAYERS,
    COMPOSE_MODE,
    PROHIBIT_TO_DECOMPOSE,
)

__all__ = [
    "decompose_module",
    "load_svd_state_dict",
]


def decompose_module(
    model: Module,
    decomposing_mode: Optional[str] = True,
    decomposer: Literal["svd", "cur", "rsvd"] = "svd",
    compose_mode: str = None,
) -> None:
    """Replace decomposable layers with their decomposed analogues (in-place).

    The function walks the module tree recursively and, for each child layer
    whose type appears in ``DECOMPOSABLE_LAYERS``, constructs the corresponding
    decomposed layer and swaps it into the parent module. Newly created layers
    are placed on the same device as the original model.

    Parameters
    ----------
    model : torch.nn.Module
        Root module to transform in place.
    decomposing_mode : Optional[str], default=True
        Weights reshaping strategy expected by the decomposed layer
        implementation, e.g. ``'channel'`` or ``'spatial'``.
        If ``None``, layers are wrapped without factorization (pass-through
        mode), depending on implementation.
    decomposer : {'svd', 'cur', 'rsvd'}, default='svd'
        Low-rank algorithm used inside the decomposed analogs.
    compose_mode : Optional[str], default=None
        Forward computation strategy for decomposed layers:
        typically one of ``'one_layer'``, ``'two_layers'`` or ``'three_layers'``.

    Returns
    -------
    None
        Operates in place; the passed ``model`` is modified.
    """
    device = extract_device(model)
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            decompose_module(
                module,
                decomposing_mode=decomposing_mode,
                decomposer=decomposer,
                compose_mode=compose_mode,
            )
        decomposed_analogue = _map_decomposed_cls(module)
        if decomposed_analogue is not None:
            new_module = decomposed_analogue(
                module,
                decomposing_mode=decomposing_mode,
                decomposer=decomposer,
                compose_mode=compose_mode,
            ).to(device)
            setattr(model, name, new_module)


def _load_svd_params(model, state_dict, prefix: str = "") -> None:
    """Load factorized parameters (U, S, Vh) into decomposed layers.

    This helper walks the module tree and, for each :class:`IDecomposed` layer,
    reads its factor tensors from the provided ``state_dict`` by the following
    keys (with the accumulated ``prefix``):

        ``{prefix}{name}.U``, ``{prefix}{name}.S``, ``{prefix}{name}.Vh``

    Parameters
    ----------
    model : torch.nn.Module
        Model with (already) decomposed layers.
    state_dict : Mapping[str, torch.Tensor]
        A state dictionary containing factor tensors saved earlier.
    prefix : str, default=""
        Hierarchical prefix for nested modules during recursion.

    Returns
    -------
    None
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            _load_svd_params(model=module, state_dict=state_dict, prefix=f"{prefix}{name}.")

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
    compose_mode: str = COMPOSE_MODE,
) -> None:
    """Load a saved SVD ``state_dict`` into a (possibly plain) model.

    The function ensures structural compatibility by first calling
    :func:`decompose_module` with the provided ``decomposing_mode`` /
    ``compose_mode`` and then injecting the factorized tensors from disk.

    Parameters
    ----------
    model : torch.nn.Module
        An instance of the base model to populate.
    decomposing_mode : str
        Weights reshaping mode expected by the decomposed layers
        (e.g., ``'channel'`` or ``'spatial'``).
    state_dict_path : str
        Filesystem path to a serialized ``state_dict`` created for a decomposed model.
    compose_mode : str, default=COMPOSE_MODE
        Forward composition strategy for decomposed layers (one/two/three-layer).

    Returns
    -------
    None
    """
    state_dict = torch.load(state_dict_path, map_location="cpu")
    decompose_module(model=model, decomposing_mode=decomposing_mode, compose_mode=compose_mode)
    _load_svd_params(model, state_dict)
    model.load_state_dict(state_dict)


def _map_decomposed_cls(inst: torch.nn.Module) -> Optional[IDecomposed]:
    """Resolve the decomposed analogue class for a given layer instance.

    Iterates over mapping ``DECOMPOSABLE_LAYERS`` and returns a decomposed class
    suitable for ``inst`` if:
        * the instance type is allowed (not listed in ``PROHIBIT_TO_DECOMPOSE``),
        * ``inst`` is an instance of a decomposable source type,
        * and ``inst`` is not already an :class:`IDecomposed`.

    Parameters
    ----------
    inst : torch.nn.Module
        Layer instance to check.

    Returns
    -------
    Optional[type[IDecomposed]]
        Decomposed analogue (class), or ``None`` if no replacement is needed.
    """
    for decomposable, decomposed in DECOMPOSABLE_LAYERS.items():
        if (
            not type(isinstance) in PROHIBIT_TO_DECOMPOSE
            and isinstance(inst, decomposable)
            and not isinstance(inst, IDecomposed)
        ):
            return decomposed
