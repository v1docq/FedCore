"""Hooks for rank pruning of low-rank–decomposed layers.

This module defines hook classes that perform one-shot or dynamic rank pruning
for layers implementing :class:`IDecomposed`. The hooks are intended to be
registered in :class:`BaseNeuralModel` and use its ``params`` and ``model``
attributes to control pruning behaviour.
"""

from enum import Enum
from math import floor, ceil

import torch
from numpy import diff, abs as npabs


from fedcore.algorithm.low_rank.rank_pruning import rank_threshold_pruning
from fedcore.architecture.abstraction.accessor import Accessor
from fedcore.models.network_impl.utils.hooks import BaseHook
from fedcore.models.network_impl.decomposed_layers import IDecomposed


class OnetimeRankPruner(BaseHook):
    """Single-shot rank pruning hook for decomposed layers.

    This hook performs rank pruning exactly once, when the configured epoch
    is reached. It scans all modules of the attached model, finds instances
    of :class:`IDecomposed`, and applies threshold-based pruning to their
    singular values (or other rank-defining parameters), followed by composing
    weights for inference.

    The hook reads the following parameters from ``self.params`` / trainer
    configuration:

    * ``rank_prune_each`` (int): epoch at which to run pruning.
    * ``non_adaptive_threshold`` (float): threshold for pruning singular values.
    * ``strategy`` (str): pruning strategy, forwarded to
      :func:`rank_threshold_pruning_in_place`.
    """
    _SUMMON_KEY = ('rank_prune_each', 'strategy')
    _hook_place = 50

    def __init__(self, params, model):
        """Initialize the one-time rank pruner."""
        super().__init__(params, model)
        self.__done = False

    @classmethod
    def check_init(cls, params):
        """Check whether this hook should be instantiated from config.

        Parameters
        ----------
        d : dict
            Configuration dictionary (usually a subset of trainer params).

        Returns
        -------
        bool
            ``True`` if the hook should be created for the given config,
            ``False`` otherwise.

        Notes
        -----
        The hook is enabled when the ``strategy`` key is present and differs
        from ``"cuttlefish"`` (which is reserved for dynamic pruning).
        """
        strategy = params.get('strategy', '') 
        if strategy and strategy != 'cuttlefish':
            return True
        return False

    def trigger(self, epoch, kws) -> bool:
        """Decide whether pruning should be executed at the current epoch.

        Parameters
        ----------
        epoch : int
            Current training epoch (0-based or 1-based, depending on trainer).
        kws : dict
            Additional keyword arguments passed by the trainer (unused here).

        Returns
        -------
        bool
            ``True`` if the pruning action should be executed at this epoch,
            ``False`` otherwise.
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
        """Perform one-shot rank pruning of all decomposed layers.

        Parameters
        ----------
        epoch : int
            Current training epoch (unused, provided for hook API).
        kws : dict
            Additional keyword arguments passed by the trainer (unused).

        Notes
        -----
        After pruning, ``compose_weight_for_inference`` is called on each
        pruned :class:`IDecomposed` layer to update its weight representation.
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
    """Dynamic (plateau-based) rank pruning hook for decomposed layers.

    This hook tracks effective ranks of :class:`IDecomposed` layers over
    training and performs pruning when the rank estimate enters a plateau
    region. The rank is estimated via a stable-rank–like functional of the
    singular-value tensor ``S`` and is adjusted according to an initial
    normalization factor.

    Expected parameters in the attached trainer:

    * ``n_plateau`` (int): window length (in epochs) to detect plateaus.
    * ``pl_thr`` (float): threshold for changes in rank history to be treated
      as a plateau.
    * ``sv_thr`` (float): singular-value threshold for initial rank estimate.
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
        """Yield (name, S) pairs for all decomposed layers' singular values.

        Parameters
        ----------
        model : torch.nn.Module
            The model whose modules should be scanned.

        Yields
        ------
        Tuple[str, torch.nn.Parameter]
            Names and parameters whose names end with ``\"S\"`` belonging to
            modules implementing :class:`IDecomposed`.
        """
        decomposed_layers = (layer for layer in model.modules() if isinstance(layer, IDecomposed))
        nparams_of_decomposed = (np for layer in decomposed_layers
                                 for np in layer.named_parameters()
                                    if np[0].endswith("S"))
        return nparams_of_decomposed

    def _prepare_mapping(self):
        """Initialize history of rank estimates for all tracked parameters.

        Returns
        -------
        dict[str, list[int]]
            Mapping from parameter name to a list containing the initial
            (integer) rank estimate computed from its singular values.
        """
        """Returns initial rank of weight matrices estimated via SVD"""
        return {name: [torch.sum((S > self.sv_thr))] for name, S in self._S_gen(self.model)}
    
    def _get_ksis(self):
        """Compute normalization factors for stable-rank tracking.

        Returns
        -------
        dict[str, float]
            Mapping from parameter name to the ratio between the initial
            rank estimate and its stable-rank estimate.
        """
        return {name: self.traced_layers[name][0] / self._estimate_stable_rank(S) for name, S in self._S_gen(self.model)}
    
    def _estimate_stable_rank(self, S: torch.Tensor):
        """Estimate a stable-rank–like measure from singular values.

        Parameters
        ----------
        S : torch.Tensor
            Singular values (or analogous spectrum) of a decomposed matrix.

        Returns
        -------
        torch.Tensor
            Stable-rank estimate defined as ``sum(S**2) / max(S)**2``.
        """
        return (S ** 2).sum() / S.max() ** 2
    
    def _update_stable_ranks(self):
        """Update rank histories for all tracked parameters.

        Notes
        -----
        For each tracked parameter, the new entry in its history is computed
        as ``ksi * stable_rank(current_S)``, where ``ksi`` is the
        normalization factor computed at initialization time and
        ``current_S`` is obtained via :class:`Accessor` from the model.
        """
        for name, history in self.traced_layers.items():
            history.append(
                self.traced_layers_ksis[name] * self._estimate_stable_rank(
                    Accessor.get_module(self.model, name)
                )
            )

    def trigger(self, epoch, kws) -> dict:
        """Check whether any layer has reached a plateau and should be pruned.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        kws : dict
            Additional keyword arguments passed by the trainer (unused).

        Returns
        -------
        bool
            ``True`` if at least one parameter should be pruned according to
            the plateau criterion, ``False`` otherwise.

        Side Effects
        ------------
        Populates ``self.trigger_result`` with a mapping from parameter name
        to the target rank (integer) for pruning.
        """
        self._estimate_stable_rank()
        to_prune = {}
        for n, history in self.traced_layers:
            if (npabs(diff(history[-max(len(history), self.n_plateau + 1):])) < self.pl_thr).all():
                to_prune[n] = ceil(history[-1])
        self.trigger_result = to_prune
        return to_prune
    
    def action(self, epoch, kws):
        """Apply effective-rank updates to layers marked for pruning.

        Parameters
        ----------
        epoch : int
            Current training epoch (unused, provided for hook API).
        kws : dict
            Additional keyword arguments passed by the trainer (unused).

        Notes
        -----
        For each entry in ``self.trigger_result`` the corresponding module is
        located via :class:`Accessor`, its ``_effective_rank`` attribute is
        set to the target rank, and the parameter is removed from the tracking
        dictionary so that it is not updated further.
        """
        for name, rank in self.trigger_result.items():
            layer_name = '.'.join(name.split('.')[:-1])
            layer = Accessor.get_module(self.model, layer_name)
            setattr(layer, self.rank_attr, rank)
            self.traced_layers.pop(name, None)  

class LRHooks(Enum):
    """Enumeration of available low-rank pruning hook types.

    Members
    -------
    onetime
        One-shot threshold-based rank pruning
        (:class:`OnetimeRankPruner`).
    cuttlefish
        Dynamic plateau-based rank pruning
        (:class:`DynamicRankPruner`).
    """
    onetime = OnetimeRankPruner
    cuttlefish = DynamicRankPruner
