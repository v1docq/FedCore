import torch
from fedcore.models.network_impl.hooks import BaseHook
from joblib import cpu_count
from math import floor, ceil
from numpy import diff, abs as npabs

from fedcore.architecture.abstraction.accessor import Accessor
from fedcore.architecture.utils.misc import filter_kw_universal
from fedcore.models.network_impl.decomposed_layers import IDecomposed
from fedcore.architecture.utils.misc import _contiguous, count_params

__all__ = [
    'rank_threshold_pruning',
    'DynamicRankPruner',
    'OnetimeRankPruner',
]

class DynamicRankPruner(BaseHook):
    _SUMMON_KEY = 'n_plateau'

    def __init__(self, params, model):
        super().__init__(params, model)
        self.n_plateau : int= params.get('n_plateau', 5)
        self.pl_thr : float = params.get('pl_thr', 1e-2)
        self.sv_thr : float = params.get('sv_thr', 1e-5)
        self.traced_layers: dict = self._prepare_mapping()
        self.traced_layers_ksis = self._get_ksis()
        self.rank_attr = '_effective_rank'

    def _S_gen(self, model: torch.nn.Module):
        decomposed_layers = (layer for layer in model.modules() if isinstance(layer, IDecomposed))
        nparams_of_decomposed = (np for layer in decomposed_layers
                                 for np in layer.named_parameters()
                                    if np[0].endswith("S"))
        return nparams_of_decomposed

    def _prepare_mapping(self):
        """Returns initial rank of weight matrices estimated via SVD"""
        return {name: [torch.sum((S > self.sv_thr))] for name, S in self._S_gen(self.model)}
    
    def _get_ksis(self):
        return {name: self.traced_layers[name][0] / self._estimate_stable_rank(S) for name, S in self._S_gen(self.model)}
    
    def _estimate_stable_rank(self, S: torch.Tensor):
        return (S ** 2).sum() / S.max() ** 2
    
    def _update_stable_ranks(self):
        for name, history in self.traced_layers.items():
            history.append(
                self.traced_layers_ksis[name] * self._estimate_stable_rank(
                    Accessor.get_module(self.model, name)
                )
            )

    def trigger(self, *args, **kwargs) -> dict:
        self._estimate_stable_rank()
        to_prune = {}
        for n, history in self.traced_layers:
            if (npabs(diff(history[-max(len(history), self.n_plateau + 1):])) < self.pl_thr).all():
                to_prune[n] = ceil(history[-1])
        return to_prune
    
    def action(self, epoch, trigger_result, *args, **kwargs):
        for name, rank in trigger_result.items():
            layer_name = '.'.join(name.split('.')[:-1])
            layer = Accessor.get_module(self.model, layer_name)
            setattr(layer, self.rank_attr, rank)
            self.traced_layers.pop(name, None)


class OnetimeRankPruner(BaseHook):
    _SUMMON_KEY = 'rank_prune_each'

    def __init__(self, params, model: torch.nn.Module):
        # non_adaptive_threshold: float = 0.75,
        # strategy: str = "explained_variance",
        # module_name: str = "",
        # epochs: int = 1,
        # rank_prune_each: int = -1):
        super().__init__(params, model)

    def trigger(self, epoch, kws) -> bool:
        rank_prune_each = self.params.get('rank_prune_each', -1)
        if not rank_prune_each:
            return False
        if rank_prune_each != -1:
            return not epoch % rank_prune_each
        else:
            return epoch == self.params.get('epochs', 1)
    
    def action(self, epoch, trigger_result):
        non_adaptive_threshold = self.params.get('non_adaptive_threshold', .75)
        strategy = self.params.get('strategy', 'explained_variance')
        for name, module in self.model.named_modules():
            if isinstance(module, IDecomposed): 
                rank_threshold_pruning(decomposed_module=module,
                                       threshold=non_adaptive_threshold,
                                       strategy=strategy,
                                       module_name=name)  


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

def _apply_S_strategy(S, strategy, threshold, round_factor=1):
    n_cpu = cpu_count()
    S, indices = S.sort(descending=True)
    n_components = S_STRATEGIES[strategy](S, threshold)
    channels_per_device = max(floor(n_components / n_cpu), 1)
    n_components = channels_per_device * n_cpu
    n_components = ceil(n_components / round_factor) * round_factor # for architecture 
    return indices[:n_components]

def _quantile_strategy(S, threshold) -> int:
    thr_value = torch.quantile(S, threshold)
    n_components = max((S > thr_value).sum().item(), 1)
    return n_components

def _explained_variance_strategy(S, threshold):
    explained_variance = torch.cumsum(torch.square(S), 0) #.div_(n_samples - 1) scaling by scalar doesn't matter
    explained_variance /= (explained_variance[-1].item())
    n_components = max(torch.where(explained_variance <= threshold)[0].max() + 1, 1)
    return n_components

def _energy_strategy(S, threshold):
    energies = torch.exp(S)
    energies = energies / torch.sum(energies)
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