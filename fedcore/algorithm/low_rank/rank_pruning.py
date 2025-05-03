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
    """Prune the weight matrices to the threshold (in-place).
    Args:
        conv: The optimizable layer.
        threshold: hyperparameter must be in the range (0, 1].
    Raises:
        Assertion Error: If ``threshold`` is not in (0, 1].
    """
    assert 0 < threshold <= 1, "Threshold must be in the range (0, 1]"

    if hasattr(decomposed_module, 'S') and decomposed_module.S is None:
        return
    if hasattr(decomposed_module, '_anti_three_layers_compose'):
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
    S, indices = S.sort(descending=True)
    n_components = SLRStrategiesEnum[strategy].value(S, threshold)
    # n_cpu = cpu_count()
    # channels_per_device = max(floor(n_components / n_cpu), 1)
    # n_components = channels_per_device * n_cpu
    n_components = ceil(n_components / round_to_times) * round_to_times  # for architecture
    n_components = min(n_components, len(indices))
    return indices[:n_components]


def _quantile_strategy(S, threshold) -> int:
    thr_value = torch.quantile(S, threshold)
    n_components = max((S > thr_value).sum().item(), 1)
    return n_components


def _explained_variance_strategy(S, threshold):
    explained_variance = torch.cumsum(torch.square(S), 0)  # .div_(n_samples - 1) scaling by scalar doesn't matter
    explained_variance /= (explained_variance[-1].item())
    n_components = max((explained_variance > threshold).sum().item(), 1)
    return n_components


def _energy_strategy(S, threshold):
    energies = torch.exp(S)
    energies = energies / torch.sum(energies)
    n_components = max((energies > threshold).sum().item(), 1)
    return n_components

    
def _abssum_strategy(S, threshold):
    abssums = torch.cumsum(S, 0)
    abssums /= (abssums[-1].item())
    n_components = max((abssums > threshold).sum().item(), 1)
    return n_components


class SLRStrategiesEnum(Enum):
    """Rank pruning strategies based on singular values"""
    quantile = partial(_quantile_strategy)
    explained_variance = partial(_explained_variance_strategy)
    energy = partial(_energy_strategy)
    absolute_sum = partial(_abssum_strategy)


SLRStrategies = EnumNoValue(SLRStrategiesEnum)
