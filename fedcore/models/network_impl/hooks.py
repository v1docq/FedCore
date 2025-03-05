import os
from abc import abstractmethod, ABC
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Set
from functools import reduce
from operator import iadd

import torch

from numpy import diff, abs as npabs
from tqdm import tqdm

from fedcore.architecture.abstraction.accessor import Accessor
from fedcore.models.network_impl.decomposed_layers import IDecomposed
from fedcore.api.utils.data import DataLoaderHandler


def now_for_file():
    return datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

class BaseHook(ABC):
    _SUMMON_KEY: str
    _hook_place : str = 'post'

    def __init__(self, params, model):
        self.params : dict = params
        self.model : torch.nn.Module = model

    def __call__(self, epoch, **kws):
        trigger_result = self.trigger(epoch, kws)
        if trigger_result:
            self.action(epoch, trigger_result, kws)

    @abstractmethod
    def trigger(self, epoch, kws):
        pass
    
    @abstractmethod
    def action(self, epoch, trigger_result, kws):
        pass

    def _filter_kw(self):
        pass


class Saver(BaseHook):
    _SUMMON_KEY = 'save_each'

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
        
    def action(self, epoch, trigger_result, kw):
        name = kw.get('name', '') or self.params.get('name', '')
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


class Evaluator(BaseHook):
    _SUMMON_KEY = 'eval_each'

    def trigger(self, epoch, kws):
        return epoch % self.eval_each == 0 and self.val_loader is not None
    
    @torch.no_grad()
    def action(self, epoch, kws):
        self.model.eval()
        metrics = kws.get('metrics', {})
        custom_loss = kws.get('custom_loss', None)
        loss_sum = 0
        total_iterations = 0
        val_dataloader = DataLoaderHandler.check_convert(dataloader=val_dataloader,
                                                       mode=self.batch_type,
                                                       max_batches=self.calib_batch_limit,
                                                       enumerate=False)
        for batch in tqdm(val_dataloader, desc='Batch #'):
            total_iterations += 1
            inputs, targets = batch
            output = self.model(inputs.to(self.device))
            if custom_loss:
                model_loss = {key: val(self.model) for key, val in custom_loss.items()}
                model_loss["metric_loss"] = self.loss_fn(
                    output, targets.to(self.device)
                )
                quality_loss = reduce(iadd, [loss for loss in model_loss.values()])
                loss_sum += model_loss["metric_loss"].item()
            else:
                quality_loss = self.loss_fn(output, targets.to(self.device))
                loss_sum += quality_loss.item()
                model_loss = quality_loss
        avg_loss = loss_sum / total_iterations
        return avg_loss

class OptimizerRenewal(BaseHook):
    def trigger(self, epoch, kws):
        return getattr()
