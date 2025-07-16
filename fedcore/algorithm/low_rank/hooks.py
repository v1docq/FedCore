from enum import Enum
from math import floor, ceil

import torch
from numpy import diff, abs as npabs


from fedcore.algorithm.low_rank.rank_pruning import rank_threshold_pruning, rank_attr_pruning
from fedcore.architecture.abstraction.accessor import Accessor
from fedcore.models.network_impl.hooks import BaseHook
from fedcore.models.network_impl.decomposed_layers import IDecomposed


class OnetimeRankPruner(BaseHook):
    _SUMMON_KEY = ('rank_prune_each', 'strategy')
    _hook_place = 50

    def __init__(self, params, model):
        super().__init__(params, model)
        self.__done = False

    @classmethod
    def check_init(cls, params):
        strategy = params.get('strategy', '') 
        if strategy and strategy != 'cuttlefish':
            return True
        return False

    def trigger(self, epoch, kws) -> bool:
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
        decomposed_layers = (layer for layer in model.modules() if isinstance(layer, IDecomposed))
        nparams_of_decomposed = (np for layer in decomposed_layers
                                 for np in layer.named_parameters()
                                    if np[0].endswith("S"))
        return nparams_of_decomposed

    def _prepare_mapping(self):
        """Returns initial rank of weight matrices estimated via SVD"""
        return {name: [torch.sum((S > self.sv_thr))] + [0] * (self.n_plateau - 1) for name, S in self._S_gen(self.model)}
    
    def _get_ksis(self):
        return {name: self.traced_layers[name][0] / self._estimate_stable_rank(S) for name, S in self._S_gen(self.model)}
    
    def _estimate_stable_rank(self, S: torch.Tensor):
        return (S ** 2).sum() / S.max() ** 2
    
    def _update_stable_ranks(self, epoch):
        for name, history in self.traced_layers.items():
            history[epoch % self.n_plateau] = self.traced_layers_ksis[name] * self._estimate_stable_rank(Accessor.get_module(self.model, name))


    def trigger(self, epoch, kws) -> dict:
        self._update_stable_ranks()
        to_prune = {}
        for n, history in self.traced_layers:
            if (npabs(diff(history)) < self.pl_thr).all():
                to_prune[n] = ceil(history[-1])
        self.trigger_result = to_prune
        return to_prune
    
    def action(self, epoch, kws):
        for name, rank in self.trigger_result.items():
            layer_name = '.'.join(name.split('.')[:-1])
            layer = Accessor.get_module(self.model, layer_name)
            setattr(layer, self.rank_attr, rank)
            strategy = self.params.get('strategy', 'explained_variance')
            if isinstance(layer, IDecomposed): 
                rank_attr_pruning(decomposed_module=layer,
                                    strategy=strategy,
                                    module_name=layer_name,
                                    rank_attr=rank)
            layer.compose_weight_for_inference() 
            

class LRHooks(Enum):
    onetime = OnetimeRankPruner
    cuttlefish = DynamicRankPruner