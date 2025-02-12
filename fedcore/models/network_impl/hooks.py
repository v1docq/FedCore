import os
from abc import abstractmethod, ABC
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Set

import torch

from numpy import diff, abs as npabs

from fedcore.architecture.abstraction.accessor import Accessor
from fedcore.models.network_impl.layers import IDecomposed


def now_for_file():
    return datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

class BaseHook(ABC):
    _KEYS: Set[str]

    def __init__(self, params, model):
        self.params : dict = params
        self.model : torch.nn.Module = model

    def __call__(self, epoch, **kws):
        result = self.trigger(epoch, kws)
        if result:
            self.action(epoch, kws)

    @abstractmethod
    def trigger(self, epoch, kws):
        pass
    
    @abstractmethod
    def action(self, epoch, kws):
        pass

    def _filter_kw(self):
        pass

class DynamicRankPruner(BaseHook):
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
    
    def action(self, *args, **kwargs):
        to_prune = self.trigger()
        if not to_prune:
            return
        for name, rank in to_prune.items():
            layer_name = '.'.join(name.split('.')[:-1])
            layer = Accessor.get_module(self.model, layer_name)
            setattr(layer, self.rank_attr, rank)
            self.traced_layers.pop(name, None)

class Saver(BaseHook):
    def __init__(self, params, model):
        super().__init__(params, model)
        self.save_each = params.get('save_each', False)
        self.checkpoint_folder = params.get('checkpoint_folder', '.')
    
    def trigger(self, epoch, kw) -> bool:
        if not self.save_each:
            return False
        if self.save_each != -1:
            return not epoch % self.save_each
        else:
            return epoch == self.params.get('epochs', 0)
        
    def action(self, epoch, kw):
        name = kw.get(name, '') or self.params.get('name', '')
        path_pref = Path(self.checkpoint_folder)
        save_only = self.params.get('save_only', '')
        to_save = self.model if not save_only else Accessor.get_module(self.model, save_only)
        try:
            path = path_pref.joinpath(f"model_{name}{now_for_file()}_{epoch}.pth")
            torch.save(
                to_save,
                path,
            )
        except Exception as x:
            if os.path.exists(path):
                os.remove(path)
            print('Basic saving failed. Trying to use jit. \nReason: ', x.args[0])
            try:
                path = path_pref.joinpath(f"model_{name}{now_for_file()}_{epoch}_jit.pth")
                torch.jit.save(torch.jit.script(to_save), path)
            except Exception as x: 
                if os.path.exists(path):
                    os.remove(path)
                print('JIT saving failed. saving weights only. \nReason: ', x.args[0])
                torch.save(to_save.state_dict(), 
                            path_pref.joinpath(f"model_{name}{now_for_file()}_{epoch}_state.pth")
                )      
