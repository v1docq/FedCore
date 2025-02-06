import torch
from fedcore.models.network_impl.layers import IDecomposed
from fedcore.algorithm.low_rank.svd_tools import _contiguous, count_params
from joblib import cpu_count
from math import floor

__all__ = [
    'rank_threshold_pruning'
]

def rank_threshold_pruning(
    decomposed_module: IDecomposed,
    threshold: float = 0.75,
    strategy: str = "explained_variance",
    module_name: str = "",
) -> None:
    """Prune the weight matrices to the threshold (in-place).
    Args:
        conv: The optimizable layer.
        threshold: hyperparameter must be in the range (0, 1].
    Raises:
        Assertion Error: If ``threshold`` is not in (0, 1].
    """
    assert 0 < threshold <= 1, "Threshold must be in the range (0, 1]"
    U, S, Vh = decomposed_module.get_U_S_Vh()

    threshold = decomposed_module._get_threshold() or threshold # for cases of per-layer adaptive thresholds

    if strategy in S_STRATEGIES:
        indices = _apply_S_strategy(S, strategy, threshold)
        initial_size = count_params(decomposed_module)
    else:
        # TODO Grad-based & approx. error
        return
    # channels_per_device = floor(n_components / n_cpu)
    # n_components = channels_per_device * n_cpu
    decomposed_module.set_U_S_Vh(
        _contiguous(U[:, indices]),
        _contiguous(S[indices]),
        _contiguous(Vh[indices, :]),
    )
    print(
        "After rank pruning left only {} % of {} layer params".format(
            100 * (count_params(decomposed_module) / initial_size), module_name
        )
    )

def _apply_S_strategy(S, strategy, threshold):
    n_cpu = cpu_count()
    S, indices = S.sort(descending=True)
    n_components = S_STRATEGIES[strategy](S, threshold)
    channels_per_device = max(floor(n_components / n_cpu), 1)
    n_components = channels_per_device * n_cpu
    return indices[:n_components]

def _quantile_strategy(S, threshold) -> int:
    thr_value = torch.quantile(S, threshold)
    n_components = max((S > thr_value).sum().item(), 1)
    # n_components = indices.cpu().detach().numpy().max() - n_components
    return n_components

def _explained_variance_strategy(S, threshold):
    explained_variance = torch.cumsum(torch.square(S), 0) #.div_(n_samples - 1) scaling by scalar doesn't matter
    explained_variance /= (explained_variance[-1].item())
    n_components = max(torch.where(explained_variance <= threshold)[0].max() + 1, 1)
    return n_components

def _energy_strategy(S, threshold):
    energies = torch.exp(S)
    n_components = max(torch.where(energies > threshold)[0].max(0) + 1, 1)
    return n_components
    
def _abssum_strategy(S, threshold):
    abssums = torch.cumsum(S, 0)
    abssums /= (abssums[-1].item())
    n_components = max(torch.where(abssums <= threshold)[0].max() + 1, 1)
    return n_components

# See no reason to explicitly demonstrate in in constant repository

S_STRATEGIES = {
    'quantile': _quantile_strategy,
    'explained_variance': _explained_variance_strategy,
    'energy': _energy_strategy,
    'absolute_sum': _abssum_strategy
}