"""
Rank pruning utilities for low-rank (SVD-based) decomposed layers.

This module implements a thresholding workflow that reduces the effective
rank of layers represented as (U, S, Vh) factors. The main entry point,
:func:`rank_threshold_pruning`, selects the number of singular components
to keep according to several strategies derived from the singular values `S`:

Strategies
----------
- ``"quantile"``            : keep components whose singular values exceed
                              the specified quantile of `S`.
- ``"explained_variance"``  : keep the smallest number of components whose
                              cumulative squared singular values exceed the
                              threshold (i.e., retain at least that share of
                              Frobenius energy).
- ``"energy"``              : apply a softmax-like normalization over `S`
                              (via ``exp(S) / sum(exp(S))``) and keep components
                              whose normalized energy exceeds the threshold.
- ``"absolute_sum"``        : keep the smallest number of components whose
                              cumulative absolute singular values exceed the
                              threshold relative to the total absolute sum.

Integration
-----------
The target layer must implement the ``IDecomposed`` protocol used across
FedCore's low-rank stack and expose:
    - ``get_U_S_Vh() -> Tuple[Tensor, Tensor, Tensor]``
    - ``set_U_S_Vh(U, S, Vh) -> None``
    - ``_anti_three_layers_compose() -> None`` (to ensure factorized state)
    - ``_get_threshold() -> Optional[float]`` (optional per-layer override)

The pruning is performed **in-place** and prints a short summary with the
fraction of parameters retained after pruning.

Notes
-----
``round_to_times`` is a convenience to align the kept rank to hardware-friendly
multiples (e.g., 4/8/16) for some architectures.
"""

from enum import Enum
from functools import partial
from joblib import cpu_count
from math import floor, ceil

import torch

from fedcore.models.network_impl.decomposed_layers import IDecomposed
from fedcore.architecture.utils.misc import _contiguous, count_params
from fedcore.architecture.utils.misc import EnumNoValue

__all__ = [
    'rank_threshold_pruning',
    'S_STRATEGIES'
]


def rank_threshold_pruning(
        decomposed_module: IDecomposed,
        threshold: float = 0.75,
        strategy: str = "explained_variance",
        module_name: str = "",
        round_to_times=4
) -> None:
    """
    Prune a decomposed layer to a lower rank based on singular values (in-place).

    The function selects the number of singular components to keep using the
    chosen ``strategy`` and updates the layer's factors ``(U, S, Vh)`` accordingly.

    Parameters
    ----------
    decomposed_module : IDecomposed
        A layer exposing SVD-like factors and ``set_U_S_Vh`` / ``get_U_S_Vh`` API.
    threshold : float, default=0.75
        Strategy-specific threshold in the open interval (0, 1]. For example,
        with ``strategy="explained_variance"``, it is the target share of
        cumulative squared singular values to retain.
    strategy : str, default="explained_variance"
        Rank selection rule. One of: ``"quantile"``, ``"explained_variance"``,
        ``"energy"``, ``"absolute_sum"``.
    module_name : str, default=""
        Optional human-readable name used only in the printed summary.
    round_to_times : int, default=4
        Round the selected number of components up to a multiple of this value
        (useful for some hardware kernels). Must be a positive integer.

    Raises
    ------
    AssertionError
        If ``threshold`` is not in the interval (0, 1].
    ValueError
        If ``strategy`` is unknown.

    Notes
    -----
    - If the layer reports a per-layer threshold via ``_get_threshold()``,
      it overrides the function-level ``threshold``.
    - If the module has attribute ``S`` and it is ``None``, pruning is skipped.

    Side Effects
    ------------
    Updates the layer's internal factors via ``set_U_S_Vh`` and prints a short
    summary of the retained parameter ratio.
    """
    assert 0 < threshold <= 1, "Threshold must be in the range (0, 1]"

    if hasattr(decomposed_module, 'S') and decomposed_module.S is None:
        return
    decomposed_module._anti_three_layers_compose()

    U, S, Vh = decomposed_module.get_U_S_Vh()
    

    threshold = decomposed_module._get_threshold() or threshold  # for cases of per-layer adaptive thresholds

    if strategy in SLRStrategies:
        indices = _apply_S_strategy(S, strategy, threshold, round_to_times=round_to_times)
        initial_size = count_params(decomposed_module)
    else:
        # TODO Grad-based & approx. error
        raise ValueError(f'Unknown strategy: `{strategy}`')
    decomposed_module.set_U_S_Vh(
        _contiguous(U[:, indices]),
        _contiguous(S[indices]),
        _contiguous(Vh[indices, :]),
    )
    print(
        "After rank pruning left only {} % of {} layer params".format(
            round(100 * (count_params(decomposed_module) / initial_size)), module_name
        )
    )


def _apply_S_strategy(S, strategy, threshold, round_to_times=1):
    """
    Translate a strategy name into a concrete component count and return indices.

    Parameters
    ----------
    S : torch.Tensor
        1D tensor of singular values.
    strategy : str
        One of the keys recognized by :class:`SLRStrategiesEnum`.
    threshold : float
        Strategy-specific threshold in (0, 1].
    round_to_times : int, default=1
        Round the kept component count up to a multiple of this value.

    Returns
    -------
    torch.Tensor
        Indices of the top ``n_components`` singular values to keep (sorted
        descending by singular value magnitude).
    """
    S, indices = S.sort(descending=True)
    n_components = SLRStrategiesEnum[strategy].value(S, threshold)
    # n_cpu = cpu_count()
    # channels_per_device = max(floor(n_components / n_cpu), 1)
    # n_components = channels_per_device * n_cpu
    n_components = ceil(n_components / round_to_times) * round_to_times  # for architecture
    n_components = min(n_components, len(indices))
    return indices[:n_components]


def _quantile_strategy(S, threshold) -> int:
    """
    Keep components whose singular values exceed the given quantile.

    Parameters
    ----------
    S : torch.Tensor
        1D tensor of singular values (unsorted is fine).
    threshold : float
        Quantile in (0, 1]; e.g., 0.75 keeps values above the 75th percentile.

    Returns
    -------
    int
        Number of components to keep (at least 1).
    """
    thr_value = torch.quantile(S, threshold)
    n_components = max((S > thr_value).sum().item(), 1)
    return n_components


def _explained_variance_strategy(S, threshold):
    """
    Keep the smallest number of components explaining at least ``threshold`` energy.

    Energy is measured as the cumulative fraction of squared singular values:
        cumsum(S^2) / sum(S^2).

    Parameters
    ----------
    S : torch.Tensor
        1D tensor of singular values (assumed sorted descending by caller).
    threshold : float
        Target cumulative share in (0, 1].

    Returns
    -------
    int
        Number of components to keep (at least 1).
    """
    explained_variance = torch.cumsum(torch.square(S), 0)  # .div_(n_samples - 1) scaling by scalar doesn't matter
    explained_variance /= (explained_variance[-1].item())
    n_components = max((explained_variance > threshold).sum().item(), 1)
    return n_components


def _energy_strategy(S, threshold):
    """
    Keep components whose softmax-like energy exceeds ``threshold``.

    We compute normalized energies as:
        energies = exp(S) / sum(exp(S)).

    Parameters
    ----------
    S : torch.Tensor
        1D tensor of singular values (assumed sorted descending by caller).
    threshold : float
        Per-component energy cutoff in (0, 1].

    Returns
    -------
    int
        Number of components to keep (at least 1).
    """
    energies = torch.exp(S)
    energies = energies / torch.sum(energies)
    n_components = max((energies > threshold).sum().item(), 1)
    return n_components

    
def _abssum_strategy(S, threshold):
    """
    Keep the smallest number of components whose cumulative |S| crosses the threshold.

    The fraction is computed as:
        cumsum(|S|) / sum(|S|).

    Parameters
    ----------
    S : torch.Tensor
        1D tensor of singular values (assumed sorted descending by caller).
    threshold : float
        Target cumulative share in (0, 1].

    Returns
    -------
    int
        Number of components to keep (at least 1).
    """
    abssums = torch.cumsum(S, 0)
    abssums /= (abssums[-1].item())
    n_components = max((abssums > threshold).sum().item(), 1)
    return n_components


class SLRStrategiesEnum(Enum):
    """Enumeration of singular-value–based rank pruning strategies.

    Members map strategy names to their respective selector functions which
    return the number of components to keep given ``S`` and ``threshold``.
    """
    quantile = partial(_quantile_strategy)
    explained_variance = partial(_explained_variance_strategy)
    energy = partial(_energy_strategy)
    absolute_sum = partial(_abssum_strategy)


# ``EnumNoValue`` provides a convenience wrapper for membership / lookup semantics.
SLRStrategies = EnumNoValue(SLRStrategiesEnum)
