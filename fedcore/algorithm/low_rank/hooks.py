from enum import Enum
from math import floor, ceil

import torch
from numpy import diff, abs as npabs

from fedcore.repository.constanst_repository import DECOMPOSE_MODE
from fedcore.algorithm.low_rank.svd_tools import decompose_module
from fedcore.algorithm.low_rank.rank_pruning import rank_threshold_pruning
from fedcore.architecture.abstraction.accessor import Accessor
from fedcore.models.network_impl.hooks import BaseHook
from fedcore.models.network_impl.decomposed_layers import IDecomposed


class Decomposer(BaseHook):
    _hook_place = -110
    _SUMMON_KEY = 'rank_prune_each'

    _gathering = ('asvd',)

    @classmethod
    def check_init(cls, d):
        return True
    
    def __init__(self, params, model):
        super().__init__(params, model)
        self.__done = False
        self.__gathered = False
        self.decomposing_mode = params.get("decomposing_mode", DECOMPOSE_MODE)
        self.decomposer = params.get('decomposer', 'svd')
        self.compose_mode = params.get("compose_mode", None)

    def trigger(self, epoch, kws):
        if self.__done:
            return False
        if self.decomposer != 'asvd' or epoch >= 1: # check numeration of epochs . here assumed 0-based
            return True
        return False
        
    def action(self, epoch, kws):
        if self.decomposer in self._gathering and not self.__gathered:
            # logic
            
            self._hook_place = 1
            self.__gathered = True
        decompose_module(self.model, decomposer=self.decomposer, 
                         decomposing_mode=self.decomposing_mode,
                         compose_mode=self.compose_mode)
            
        self.__done = True
        
    


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

    def trigger(self, epoch, kws) -> dict:
        self._estimate_stable_rank()
        to_prune = {}
        for n, history in self.traced_layers:
            if (npabs(diff(history[-max(len(history), self.n_plateau + 1):])) < self.pl_thr).all():
                to_prune[n] = ceil(history[-1])
        self.trigger_result = to_prune
        return to_prune
    
    def action(self, epoch, kws):
        for name, rank in self.trigger_result.items():
            layer_name = '.'.join(name.split('.')[:-1])
            layer = Accessor.get_module(self.model, layer_name)
            setattr(layer, self.rank_attr, rank)
            self.traced_layers.pop(name, None)  


class ASVDPruner(OnetimeRankPruner):
    _SUMMON_KEY = ('rank_prune_each', 'strategy')
    _hook_place = 0

    rank_attr = 'channel_activation'

    def __init__(self, params, model: torch.nn.Module):
        super().__init__(params, model)
        self.__name_to_activation_sum = {}
        self.__name_to_activation_num = {}
        self.__handles = []
        self._gathered = False


    def __aggregate_activations(self, module, args, output, name):
        self.__name_to_activation_sum[name] = self.__name_to_activation_sum.get(name, 0.) + torch.sum(torch.abs(output), dim=0)
        self.__name_to_activation_num[name] = self.__name_to_activation_num.get(name, 0) + output.size(0)

    def __register_activation_gatherer(self):
        for name, layer in self.model.named_modules():
            if not isinstance(layer, IDecomposed):
                continue
            layer: torch.nn.Module = layer
            from functools import partial
            self.__handles.append(
                layer.register_forward_hook(partial(self.__aggregate_activations, name=name))
            )
            self._gathered = True


    def action(self, epoch, kws):
        if not self._gathered:
            self.__register_activation_gatherer()
        else:
            for name in self.__name_to_activation_sum:
                layer = self.model.get_submodule(name)
                layer._spectrum_source = self.rank_attr
                setattr(layer, self.rank_attr, self.__name_to_activation_sum[name] / self.__name_to_activation_num[name])
            super().action(epoch, kws)
            self._reset()
                

    def _reset(self):
        for layer in self.model.modules():
            if not isinstance(layer, IDecomposed):
                continue
            delattr(layer, self.rank_attr)
            layer._spectrum_source = 'S'
        for handle in self.__handles:
            handle.remove() 
        self.__handles.clear()
        self.__name_to_activation_num.clear()
        self.__name_to_activation_sum.clear()
        self._gathered = False
        


class LRHooks(Enum):
    onetime = OnetimeRankPruner
    cuttlefish = DynamicRankPruner