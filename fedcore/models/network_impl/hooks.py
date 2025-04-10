import os
from abc import abstractmethod, ABC
from datetime import datetime
from enum import Enum
from functools import partial
from inspect import isclass
from pathlib import Path
import torch
from tqdm import tqdm

from fedcore.architecture.abstraction.accessor import Accessor
from fedcore.api.utils.data import DataLoaderHandler
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.models.network_modules.layers.special import EarlyStopping

VERBOSE = True


def now_for_file():
    return datetime.now().strftime("%m-%d-%Y_%H-%M-%S")


class BaseHook(ABC):
    _SUMMON_KEY: str
    _hook_place: str = 'post'

    def __init__(self, params, model):
        self.params: dict = params
        self.model: torch.nn.Module = model

    def __call__(self, epoch, **kws):
        trigger_result = self.trigger(epoch, kws)
        if VERBOSE and trigger_result:
            self._verbose(epoch)
        if trigger_result:
            self.action(epoch, kws)

    @abstractmethod
    def trigger(self, epoch, kws):
        pass

    @abstractmethod
    def action(self, epoch, kws):
        pass

    def _filter_kw(self):
        pass

    @classmethod
    def check_init(cls, d: dict):
        from fedot.core.operations.operation_parameters import OperationParameters
        if isinstance(d, OperationParameters):
            d = d.to_dict()
        summons = cls._SUMMON_KEY if not isinstance(cls._SUMMON_KEY, str) else (cls._SUMMON_KEY,)
        return any(d[summon] is not None for summon in summons if summon in d.keys())

    def __repr__(self):
        return self.__class__.__name__

    def _verbose(self, epoch):
        print(f'Triggered {repr(self)} at {epoch} epoch.')


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

    def action(self, epoch, kw):
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


class FitReport(BaseHook):
    _SUMMON_KEY = 'log_each'

    def __init__(self, params, model):
        super().__init__(params, model)
        self.log_interval = params.get('log_each', 1)

    def trigger(self, epoch, kw) -> bool:
        return epoch % self.log_interval == 0

    def action(self, epoch, kws):
        history = kws['history']
        (tr_e, train_loss) = history['train_loss'][-1]
        val_losses = history['val_loss']

        (va_e, val_loss) = history['val_loss'][-1] if val_losses else (None, None)
        # if val_loss:
        #     val_loss = torch.round(val_loss, decimals=4)
        # if train_loss:
        #     train_loss = torch.round(train_loss, decimals=4)

        # TODO Any number of custom losses
        print(f'Train # epoch: {tr_e}, value: {train_loss}')
        if va_e:
            print(f'Valid # epoch: {va_e}, value: {val_loss}')
        if len(history) <= 2:
            return
        print('Including:')
        for name, hist in history.items():
            if name in ('train_loss', 'val_loss'):
                continue
            if hist:
                epoch, val = hist[-1]
                print(f'\tCriterion `{name}`: {val}')

        # if custom_loss is not None:
        #     print(f"Epoch: {epoch}, Average loss {}, {}: {:.6f}, {}: {:.6f}, {}: {:.6f}".
        #           format(epoch, avg_loss, *custom_loss))
        # else:
        #     print("Epoch: {}, Average loss {}".format(epoch, avg_loss))


# class Scheduler(BaseHook):
#     _SUMMON_KEY = 'use_scheduler'

#     def __init__(self, params, model):
#         super().__init__(params, model)
#         self._init_scheduler(optimizer=params['optimizer'],
#                              steps_per_epoch=max(1, len(params['train_loader'])),
#                              epochs=params['epochs'],
#                              learning_rate=params['learning_rate'],
#                              )
#         self._init_early_stop(learning_params=params['learning_params'])

#     def _init_scheduler(self, optimizer, epochs, learning_rate, steps_per_epoch):
#         self.scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
#                                                  steps_per_epoch=steps_per_epoch,
#                                                  epochs=epochs,
#                                                  max_lr=learning_rate)

#     def _init_early_stop(self, learning_params):
#         self.early_stopping = EarlyStopping(**learning_params['use_early_stopping'])

#     def trigger(self, epoch, kw) -> bool:
#         return True

#     def action(self, epoch, kws):
#         self.scheduler.step()
#         self.early_stopping(loss=kws['train_loss'])
#         if not self.early_stopping.early_stop:
#             print('Updating learning rate to {}'.format(self.scheduler.get_last_lr()[0]))
#         else:
#             print("Early stopping")


class Evaluator(BaseHook):
    _SUMMON_KEY = 'eval_each'

    def __init__(self, params, model):
        super().__init__(params, model)
        self.eval_each = params.get('eval_each', 1)
        self.device = next(iter(self.model.parameters())).device

    def trigger(self, epoch, kws):
        return epoch % self.eval_each == 0 and kws['val_loader'] is not None

    @torch.no_grad()
    def action(self, epoch, kws):
        self.model.eval()
        criterion = kws['criterion']
        val_dataloader = kws['val_loader']
        loss_sum = 0
        val_dataloader = DataLoaderHandler.check_convert(dataloader=val_dataloader,
                                                         enumerate=False)

        for batch in tqdm(val_dataloader, desc='Batch #'):
            *inputs, targets = batch
            inputs = tuple(inputs_.to(self.device) for inputs_ in inputs if hasattr(inputs_, 'to'))
            output = self.model(*inputs)
            loss_sum += criterion(model_output=output,
                                  target=targets.to(self.device),
                                  epoch=epoch, stage='val')
        avg_loss = loss_sum / len(val_dataloader)
        history = kws['history']['val_loss']
        history.append((epoch, avg_loss))


class OptimizerGen(BaseHook):
    _check_field = '_structure_changed__'
    _hook_place = 'pre'

    def __init__(self, params, model):
        super().__init__(params, model)
        self.__gen = self.__get_optimizer_gen(
            self.params.get('optimizer', 'adam')
        )

    @classmethod
    def check_init(cls, d: dict):
        return True

    def trigger(self, epoch, kws):
        return epoch == 1 or getattr(self.model, self._check_field, False)

    def action(self, epoch, kws):
        kws['trainer_objects']['optimizer'] = self.__gen(self.model.parameters())
        if hasattr(self.model, self._check_field):
            delattr(self.model, self._check_field)

    def __get_optimizer_gen(self, opt_type, **kws):
        if isinstance(opt_type, partial):
            return partial(opt_type, lr=self.learning_rate, **kws)
        if isinstance(opt_type, str):
            opt_constructor = Optimizers[opt_type].value
        elif isclass(opt_type):
            opt_constructor = opt_type
        else:
            raise TypeError('Unknown type for optimizer is passed! Required: constructor, partial, or str')
        optimizer_gen = partial(opt_constructor, lr=self.params.get('learning_rate', 1e-3), **kws)
        return optimizer_gen


class SchedulerRenewal(BaseHook):
    _hook_place = 'pre'
    _SUMMON_KEY = ('scheduler_step_each', 'scheduler')

    def __init__(self, params, model):
        super().__init__(params, model)
        self.__gen = self.__get_scheduler_gen(
            self.params.get('sch_type', 'one_cycle')
        )
        self._mode = []

    def __get_scheduler_gen(self, sch_type, **kws):
        if isinstance(sch_type, partial):
            return partial(sch_type, **kws)
        if isinstance(sch_type, str):
            sch_constructor, remapping = Schedulers[sch_type].value
        elif isclass(sch_type):
            sch_constructor = sch_type
        else:
            raise TypeError('Unknown type for scheduler is passed! Required: constructor, partial, or str')
        # kws = {remapping[k]: v for k, v in self.params.to_dict().items() if k in remapping}
        # print(kws)
        scheduler_gen = partial(sch_constructor, **kws)
        return scheduler_gen

    def __renew(self, kws):
        return self.__gen(kws['trainer_objects']['optimizer'],
                          self.params.get('learning_rate', 1e-3),
                          self.params.get('epochs', 1))

    def trigger(self, epoch, kws):
        if kws['trainer_objects']['scheduler'] is None:
            kws['trainer_objects']['scheduler'] = self.__renew(kws)
        opt_changed = kws['trainer_objects']['scheduler'].optimizer is kws['trainer_objects']['optimizer']
        if epoch == 1 or opt_changed:
            self._mode.append('renew')
        if epoch % self.params.get('scheduler_step_each', 1) == 0:
            self._mode.append('step')
        return bool(self._mode)

    def action(self, epoch, kws):
        if 'renew' in self._mode:
            kws['trainer_objects']['scheduler'] = self.__renew(kws)
        if 'step' in self._mode:
            kws['trainer_objects']['scheduler'].step()
        self._mode.clear()


class LoggingHooks(Enum):
    evaluator = Evaluator
    fit_report = FitReport
    saver = Saver


class ModelLearningHooks(Enum):
    optimizer_gen = OptimizerGen
    scheduler_renewal = SchedulerRenewal


class Optimizers(Enum):
    adam = torch.optim.Adam
    adamw = torch.optim.AdamW
    rmsprop = torch.optim.RMSprop
    sgd = torch.optim.SGD
    adadelta = torch.optim.Adadelta


class Schedulers(Enum):
    one_cycle = (torch.optim.lr_scheduler.OneCycleLR,
                 {'learning_rate': 'max_lr', 'epochs': 'total_steps'})
