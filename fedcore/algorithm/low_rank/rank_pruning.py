"""Rank policies evaluated on a true, nonnegative singular spectrum.

explained_variance retains squared Frobenius norm; absolute_sum retains nuclear
norm; energy retains cumulative softmax mass (not Frobenius energy). Quantile
keeps a threshold fraction of components, including ties deterministically.
"""
from enum import Enum
from functools import partial
from math import ceil
import torch
from fedcore.models.network_impl.decomposed_layers import IDecomposed


def _validate(S, threshold, round_to_times=1):
    if not 0 < threshold <= 1:
        raise ValueError('Threshold must be in (0, 1]')
    if isinstance(round_to_times,bool) or not isinstance(round_to_times,int) or round_to_times <= 0:
        raise ValueError('round_to_times must be a positive integer')
    if S.ndim != 1 or not S.numel() or not torch.isfinite(S).all():
        raise ValueError('A nonempty finite one-dimensional spectrum is required')
    if (S < 0).any():
        raise ValueError('Rank policies require singular values; canonicalize trained factors')


def _first_reaching(masses, threshold):
    if threshold == 1:
        # Preserve every nonzero contribution, independently of accumulation rounding.
        nz = torch.nonzero(masses > 0).reshape(-1)
        return int(nz[-1])+1 if len(nz) else 1
    if not torch.any(masses > 0):
        return 1  # zero operator: retain one zero component for valid layer shapes
    cumulative = masses.to(torch.float64).cumsum(0)
    target = cumulative[-1]*threshold
    return min(int(torch.searchsorted(cumulative,target).item())+1,len(masses))


def _explained_variance_strategy(S, threshold):
    scale = S.max()
    return _first_reaching((S/scale).square() if scale > 0 else S,threshold)


def _abssum_strategy(S, threshold):
    scale = S.max()
    return _first_reaching(S/scale if scale > 0 else S,threshold)


def _energy_strategy(S, threshold):
    return _first_reaching(torch.softmax(S.to(torch.float64),dim=0),threshold)


def _quantile_strategy(S, threshold):
    return max(1,ceil(threshold*len(S)))


class SLRStrategiesEnum(Enum):
    quantile = partial(_quantile_strategy)
    explained_variance = partial(_explained_variance_strategy)
    energy = partial(_energy_strategy)
    absolute_sum = partial(_abssum_strategy)


SLRStrategies = tuple(member.name for member in SLRStrategiesEnum)
S_STRATEGIES = SLRStrategies


def _apply_S_strategy(S,strategy,threshold,round_to_times=1):
    _validate(S,threshold,round_to_times)
    if strategy not in SLRStrategies:
        raise ValueError(f'Unknown strategy: {strategy}')
    sorted_values,indices = S.sort(descending=True,stable=True)
    rank = SLRStrategiesEnum[strategy].value(sorted_values,threshold)
    return indices[:min(ceil(rank/round_to_times)*round_to_times,len(S))]


def rank_threshold_pruning_in_place(decomposed_module:IDecomposed,threshold=.75,
                                    strategy='explained_variance',module_name='',round_to_times=4):
    if not isinstance(decomposed_module,IDecomposed):
        raise TypeError('Expected a decomposed layer')
    threshold = decomposed_module._get_threshold() or threshold
    # Re-SVD makes decisions invariant to all equivalent trained factorizations,
    # including CUR initialization; the error bound is for this actual operator.
    matrix = decomposed_module.factor_matrix().detach()
    if not torch.isfinite(matrix).all():
        raise ValueError('Cannot prune a nonfinite operator')
    with torch.no_grad():
        u,s,vh = torch.linalg.svd(matrix,full_matrices=False)
        spectra = s.unbind(0) if s.ndim == 2 else (s,)
        indices = [_apply_S_strategy(v,strategy,threshold,round_to_times) for v in spectra]
        # Uniform rank per group permits a grouped kernel; use the largest request.
        rank = max(len(i) for i in indices)
        decomposed_module.set_U_S_Vh(u[...,:rank],s[...,:rank],vh[...,:rank,:])
    decomposed_module.rank_pruning_info = {'method':'operator_svd','strategy':strategy,
                                         'threshold':float(threshold),'rank':rank}


rank_threshold_pruning = rank_threshold_pruning_in_place
__all__ = ['rank_threshold_pruning','rank_threshold_pruning_in_place','S_STRATEGIES']
