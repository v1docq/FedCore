"""
Training-time hooks for low-rank rank pruning in FedCore.

This module provides two hook implementations that can be attached to a model
to reduce the effective rank of decomposed layers (`IDecomposed`) during or
after training:

- OnetimeRankPruner:   one-shot threshold-based pruning on a chosen epoch/interval.
- DynamicRankPruner:   adaptive "cuttlefish" strategy that detects plateaus in
                       effective rank evolution and sets a target rank accordingly.
- LRHooks:             small enum-style registry for convenient selection.

Integration
-----------
Hooks are discovered/instantiated by FedCore's training loop via their
`_SUMMON_KEY` and executed around a fixed `_hook_place`. Both hooks cooperate
with `rank_threshold_pruning` and `Accessor` utilities and expect decomposed
layers to expose singular values `S` (or compatible attributes) and to be able
to compose weights for inference when pruning is finished.
"""

from enum import Enum
from math import floor, ceil

import torch
from numpy import diff, abs as npabs


from fedcore.algorithm.low_rank.rank_pruning import rank_threshold_pruning
from fedcore.architecture.abstraction.accessor import Accessor
from fedcore.models.network_impl.hooks import BaseHook
from fedcore.models.network_impl.decomposed_layers import IDecomposed


class OnetimeRankPruner(BaseHook):
    """
    One-shot rank pruning hook.

    Performs a single pass over decomposed layers (`IDecomposed`) and prunes
    singular vectors according to the specified strategy and threshold. The hook
    is triggered either on a specific epoch or at a periodic interval defined
    by `rank_prune_each`. After pruning, layers compose their weights for
    inference-time use.

    Parameters in `params`
    ----------------------
    rank_prune_each : int
        -1 to trigger on final epoch, or N to prune every N epochs.
    non_adaptive_threshold : float
        Threshold value for pruning criterion (depends on `strategy`).
    strategy : str, default='explained_variance'
        Rank selection strategy passed to `rank_threshold_pruning`.
    epochs : int
        Total number of epochs; used when `rank_prune_each == -1`.
    """
    _SUMMON_KEY = ('rank_prune_each', 'strategy')
    _hook_place = 50

    def __init__(self, params, model):
        super().__init__(params, model)
        self.__done = False

    @classmethod
    def check_init(cls, params):
        """
        Decide whether to instantiate this hook.

        Here we check that a non-empty `strategy` is provided and is not 'cuttlefish'
        (which is reserved for a dynamic pruner variant).
        """
        strategy = params.get('strategy', '') 
        if strategy and strategy != 'cuttlefish':
            return True
        return False

    def trigger(self, epoch, kws) -> bool:
        """
        Determine whether pruning should occur at the current epoch.

        Logic
        -----
        - If already executed once, never trigger again.
        - If `rank_prune_each` is falsy (0 or None), do not trigger.
        - If `rank_prune_each` > 0, trigger every N epochs.
        - If `rank_prune_each` == -1, trigger on the final epoch (`epochs` in params).

        Parameters
        ----------
        epoch : int
            Current epoch number (1-based or 0-based — consistent with caller).
        kws : dict
            Unused here; reserved for future extensions.

        Returns
        -------
        bool
            True if the hook should run this epoch.
        """
        if self.__done:
            return False
        rank_prune_each = self.params.get('rank_prune_each', -1)
        if not rank_prune_each:
            return False
        if rank_prune_each != -1:
            return not epoch % rank_prune_each
        else:
            return epoch == self.params.get('epochs', 1)

    def action(self, epoch, kws):
        """
        Apply rank pruning to all `IDecomposed` layers and finalize weights.

        Parameters
        ----------
        epoch : int
            Current epoch.
        kws : dict
            Unused here.
        """
        self.__done = True
        non_adaptive_threshold = self.params.get('non_adaptive_threshold', .75)
        strategy = self.params.get('strategy', 'explained_variance')
        for name, module in self.model.named_modules():
            if isinstance(module, IDecomposed): 
                rank_threshold_pruning(decomposed_module=module,
                                       threshold=non_adaptive_threshold,
                                       strategy=strategy,
                                       module_name=name)
                module.compose_weight_for_inference()


class DynamicRankPruner(BaseHook):
    """
    Adaptive rank pruning hook (a.k.a. "cuttlefish" strategy).

    Tracks the evolution of effective ranks (by examining singular values `S`
    inside decomposed layers) and prunes when a plateau is detected. A plateau
    is defined by small absolute differences over the last `n_plateau` steps.

    Parameters in `params`
    ----------------------
    n_plateau : int, default=5
        Window size to check for plateau behavior.
    pl_thr : float, default=1e-2
        Threshold for absolute diff between consecutive stable-rank estimates.
    sv_thr : float, default=1e-5
        Threshold to count non-zero singular values when estimating rank.

    Notes
    -----
    This hook sets an attribute (default: `_effective_rank`) on the affected
    layer when a plateau is found; the layer is expected to react to this
    attribute downstream (e.g., during forward or recompose steps).
    """
    _SUMMON_KEY = 'n_plateau'
    _hook_place = 50

    def __init__(self, params, model):
        super().__init__(params, model)
        self.n_plateau : int= params.get('n_plateau', 5)
        self.pl_thr : float = params.get('pl_thr', 1e-2)
        self.sv_thr : float = params.get('sv_thr', 1e-5)
        self.traced_layers: dict = self._prepare_mapping()
        self.traced_layers_ksis = self._get_ksis()
        self.rank_attr = '_effective_rank'

    def _S_gen(self, model: torch.nn.Module):
        """
        Generator over named parameters of decomposed layers that correspond to `S`.

        Yields
        ------
        Iterator[Tuple[str, torch.Tensor]]
            Pairs `(name, tensor)` for parameters ending with "S".
        """
        decomposed_layers = (layer for layer in model.modules() if isinstance(layer, IDecomposed))
        nparams_of_decomposed = (np for layer in decomposed_layers
                                 for np in layer.named_parameters()
                                    if np[0].endswith("S"))
        return nparams_of_decomposed

    def _prepare_mapping(self):
        """
        Initialize a history map of estimated ranks for each tracked `S` parameter.

        Returns
        -------
        dict
            `{param_name: [initial_rank]}` where `initial_rank` is the count
            of singular values greater than `sv_thr`.
        """
        return {name: [torch.sum((S > self.sv_thr))] for name, S in self._S_gen(self.model)}
    
    def _get_ksis(self):
        """
        Compute scaling coefficients (ksi) per tracked parameter.

        For each `S`, we compute `ksi = initial_rank / stable_rank(S)`,
        where `stable_rank(S) = sum(S^2) / max(S)^2`. These coefficients are
        later used to normalize subsequent stable-rank estimates.
        """
        return {name: self.traced_layers[name][0] / self._estimate_stable_rank(S) for name, S in self._S_gen(self.model)}
    
    def _estimate_stable_rank(self, S: torch.Tensor):
        """
        Estimate stable rank from singular values.

        stable_rank(S) = ||S||_2^2 / ||S||_∞^2 = sum(S^2) / max(S)^2

        Parameters
        ----------
        S : torch.Tensor
            1D tensor of singular values.

        Returns
        -------
        torch.Tensor
            Scalar tensor with stable-rank estimate.
        """
        return (S ** 2).sum() / S.max() ** 2
    
    def _update_stable_ranks(self):
        """
        Append a new stable-rank estimate to each parameter's history.

        Uses the stored `ksi` scaling and the current module's attribute
        (looked up via `Accessor`) to obtain the latest `S`. The concrete
        mechanics of retrieving `S` from the module are assumed to be
        implemented in the decomposed layer type.
        """
        for name, history in self.traced_layers.items():
            history.append(
                self.traced_layers_ksis[name] * self._estimate_stable_rank(
                    Accessor.get_module(self.model, name)
                )
            )

    def trigger(self, epoch, kws) -> dict:
        """
        Decide which layers to prune based on plateau detection.

        For each tracked parameter, we look at the last `n_plateau + 1` values
        in its history and check if all consecutive differences are below
        `pl_thr`. If so, we mark the corresponding layer for pruning to the
        nearest integer `ceil(history[-1])`.

        Parameters
        ----------
        epoch : int
            Current epoch (unused in logic).
        kws : dict
            Unused here; reserved for extension.

        Returns
        -------
        dict
            Mapping `{param_name: target_rank}` for layers to be pruned this step.
        """
        self._estimate_stable_rank()
        to_prune = {}
        for n, history in self.traced_layers:
            if (npabs(diff(history[-max(len(history), self.n_plateau + 1):])) < self.pl_thr).all():
                to_prune[n] = ceil(history[-1])
        self.trigger_result = to_prune
        return to_prune
    
    def action(self, epoch, kws):
        """
        Apply the decided rank updates to target layers.

        Sets `self.rank_attr` (default `_effective_rank`) on matched layers,
        then removes them from further tracking.

        Parameters
        ----------
        epoch : int
            Current epoch.
        kws : dict
            Unused here.
        """
        for name, rank in self.trigger_result.items():
            layer_name = '.'.join(name.split('.')[:-1])
            layer = Accessor.get_module(self.model, layer_name)
            setattr(layer, self.rank_attr, rank)
            self.traced_layers.pop(name, None)  


class LRHooks(Enum):
    """
    Registry of low-rank hooks for convenience.

    Members
    -------
    onetime : type[OnetimeRankPruner]
        One-shot pruning hook.
    cuttlefish : type[DynamicRankPruner]
        Plateau-based dynamic pruning hook.
    """
    onetime = OnetimeRankPruner
    cuttlefish = DynamicRankPruner
