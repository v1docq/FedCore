from enum import Enum
import torch
from torch import nn, optim
from tqdm import tqdm
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.models.network_impl.hooks import BaseHook
from fedcore.repository.constanst_repository import PRUNER_WITHOUT_REQUIREMENTS, PRUNER_REQUIRED_REG


class ZeroShotPruner(BaseHook):
    _SUMMON_KEY = 'pruning'
    _hook_place = 'post'

    def __init__(self, params, model):
        super().__init__(params, model)
        self.optimizer_for_grad = params.get('optimizer_for_grad_acc', None)
        self.criterion_for_grad = params.get('criterion_for_grad', None)
        self.zeroshot_names = ["Magnitude", 'LAMP', 'Random']

    def _define_pruner_type(self, importance):
        zeroshot_cond_one = isinstance(importance, tuple(PRUNER_WITHOUT_REQUIREMENTS.values()))
        zeroshot_cond_two = any([str(importance).__contains__(x) for x in self.zeroshot_names])
        zeroshot_pruner = all([zeroshot_cond_one, zeroshot_cond_two])
        pruner_with_grads = all([not zeroshot_pruner, any([str(importance).__contains__('Taylor'),
                                                           str(importance).__contains__('Hessian')
                                                           ])])
        pruner_with_reg = all([not zeroshot_pruner, any([str(importance).__contains__('BNScale'),
                                                         str(importance).__contains__('Group')])])
        if pruner_with_reg:
            callback = 'RegPruner'
        elif pruner_with_grads:
            callback = 'GradPruner'
        elif zeroshot_pruner:
            callback = 'ZeroShotPruner'
        try:
            return callback
        except Exception:
            _ = 1

    def __call__(self, importance, **kws):
        callback_type = self._define_pruner_type(importance)
        trigger_result = self.trigger(callback_type, kws)
        if trigger_result:
            self.action(callback_type, kws)

    def _accumulate_grads(self, data, target):
        data, target = data.to(default_device()), target.to(default_device())
        data, target = data.to(torch.float32), target.to(torch.float32)
        out = self.model(data)
        if isinstance(self.criterion_for_grad, torch.nn.CrossEntropyLoss):  # classification task
            target = target.to(torch.int64)  # convert probalistic output to labels)
        try:
            loss = self.criterion_for_grad(out, target)
        except Exception:
            loss = self.criterion_for_grad(out, target)
        loss.backward()
        return loss

    def trigger(self, callback_type, kw) -> bool:
        return callback_type.__contains__('ZeroShot')

    def pruning_operation(self, kws):
        pruning_hist = []
        for i in range(kws['pruning_iterations']):
            potential_groups_to_prune = list(kws['pruner_cls'].step(interactive=True))
            for group in potential_groups_to_prune:
                dep, idxs = group[0]
                layer = dep.layer
                pruning_fn = dep.pruning_fn
                pruning_hist.append((layer, idxs, pruning_fn))
                group.prune()

    def action(self, callback, kws):
        pruner_metadata = kws['pruner_objects']
        self.pruning_operation(pruner_metadata)


class PrunerWithGrad(ZeroShotPruner):
    _SUMMON_KEY = 'pruning'
    _hook_place = 'post'

    def __init__(self, params, model):
        super().__init__(params, model)
        self.criterion_for_grad = params.get('criterion_for_grad', None)

    def trigger(self, callback_type, kw) -> bool:
        return callback_type.__contains__('GradPruner')

    def action(self, callback_type, kws):
        print(f"Gradients accumulation")
        print(f"==========================================")
        pruner_metadata = kws['pruner_objects']
        for i, (data, target) in enumerate(pruner_metadata['input_data'].features.val_dataloader):
            self._accumulate_grads(data, target)
        self.pruning_operation(pruner_metadata)


class PrunerWithReg(ZeroShotPruner):
    _SUMMON_KEY = 'pruning'
    _hook_place = 'post'

    def __init__(self, params, model):
        super().__init__(params, model)
        self.optimizer_for_grad = params.get('optimizer_for_grad_acc', None)
        self.criterion_for_grad = params.get('criterion_for_grad', None)
        if self.optimizer_for_grad is None:
            self.optimizer_for_grad = optim.Adam(self.model.parameters(),
                                                 lr=0.0001)

    def regularize_model_params(self, pruner, val_dataloader):
        pruner.update_regularizer()  # <== initialize regularizer. Define model groups for pruning
        val_batches = len(val_dataloader) - 1
        with tqdm(total=val_batches, desc='Pruning reg', ) as pbar:
            for i, (data, target) in enumerate(val_dataloader):
                if i != 0:
                    self.optimizer_for_grad.zero_grad()
                    loss = self._accumulate_grads(data, target)
                    pruner.regularize(self.model)  # after loss.backward()
                    self.optimizer_for_grad.step()  # <== for sparse training
                    pbar.update(1)
        return pruner

    def prune_after_reg(self, pruner, pruning_iter):
        pruning_hist = []
        for i in range(pruning_iter):
            potential_groups_to_prune = pruner.step(interactive=True)
            for group in potential_groups_to_prune:
                dep, idxs = group[0]
                layer = dep.layer
                pruning_fn = dep.pruning_fn
                pruning_hist.append((layer, idxs, pruning_fn))
                group.prune()

    def trigger(self, callback_type, kw) -> bool:
        return callback_type.__contains__('RegPruner')

    def action(self, callback_type, kws):
        pruner_metadata = kws['pruner_objects']
        pruner = pruner_metadata['pruner_cls']
        pruning_iter = pruner_metadata['pruning_iterations']
        val_dataloader = pruner_metadata['input_data'].features.val_dataloader
        pruner = self.regularize_model_params(pruner, val_dataloader)
        self.prune_after_reg(pruner, pruning_iter)


class PruningHooks(Enum):
    PRUNERWITHGRAD = PrunerWithGrad
    PRUNERWITHREG = PrunerWithReg
    ZEROSHOTPRUNER = ZeroShotPruner
