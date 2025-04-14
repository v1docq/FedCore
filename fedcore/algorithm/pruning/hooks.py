from enum import Enum

from tqdm import tqdm
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.models.network_impl.hooks import BaseHook
from fedcore.repository.constanst_repository import PRUNER_WITHOUT_REQUIREMENTS, PRUNER_REQUIRED_REG


class ZeroShotPruner(BaseHook):
    _SUMMON_KEY = 'pruning'
    _hook_place = 50

    def __init__(self, params, model):
        super().__init__(params, model)
        self.optimizer_for_grad = params.get('optimizer_for_grad_acc', None)
        self.criterion_for_grad = params.get('criterion_for_grad', None)
        self.zeroshot_names = ["Magnitude", 'LAMP', 'Random']

    def _define_pruner_type(self, importance):
        zeroshot_cond_one = isinstance(importance, tuple(PRUNER_WITHOUT_REQUIREMENTS.values()))
        zeroshot_cond_two = any([str(importance).__contains__(x) for x in self.zeroshot_names])
        zeroshot_pruner = all([zeroshot_cond_one, zeroshot_cond_two])
        pruner_with_grads = all([not zeroshot_pruner, str(importance).__contains__('Taylor')])
        pruner_with_reg = all([not zeroshot_pruner, not pruner_with_grads])
        if pruner_with_grads:
            callback = 'GradPruner'
        elif pruner_with_reg:
            callback = 'RegPruner'
        elif zeroshot_pruner:
            callback = 'ZeroShotPruner'
        return callback

    def __call__(self, importance, **kws):
        callback_type = self._define_pruner_type(importance)
        trigger_result = self.trigger(callback_type, kws)
        if trigger_result:
            self.action(callback_type, kws)

    def _accumulate_grads(self, data, target):
        data, target = data.to(default_device()), target.to(default_device())
        out = self.model(data)
        loss = self.criterion_for_grad(out, target)
        loss.backward()

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
                # valid_to_prune = kws['validator'].validate_pruned_layers(layer, pruning_fn)
                # if valid_to_prune:

    def action(self, callback, kws):
        pruner_metadata = kws['pruner_objects']
        self.pruning_operation(pruner_metadata)


class PrunerWithGrad(ZeroShotPruner):
    _SUMMON_KEY = 'pruning'
    _hook_place = 50

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
    _hook_place = 50

    def __init__(self, params, model):
        super().__init__(params, model)
        self.optimizer_for_grad = params.get('optimizer_for_grad_acc', None)
        self.criterion_for_grad = params.get('criterion_for_grad', None)

    def trigger(self, callback_type, kw) -> bool:
        return callback_type.__contains__('RegPruner')

    def action(self, callback_type, kws):
        pruner_metadata = kws['pruner_objects']
        pruner = pruner_metadata['pruner_cls']
        pruner.update_regularizer()
        # <== initialize regularizer
        with tqdm(total=len(pruner_metadata['input_data'].features.val_dataloader) - 1,
                  desc='Pruning reg',
                  ) as pbar:
            for i, (data, target) in enumerate(pruner_metadata['input_data'].features.val_dataloader):
                if i != 0:
                    # we using 1 batch as example of pruning quality
                    try:
                        self.optimizer_for_grad.zero_grad()
                        # pruner_metadata['optimizer_for_grad_acc'].zero_grad()
                        loss = self._accumulate_grads(data, target)  # after loss.backward()
                        pruner.regularize(self.model, loss)  # <== for sparse training
                        self.optimizer_for_grad.step()
                        # pruner_metadata['optimizer_for_grad_acc'].step()
                    except Exception as er:
                        print('Caught ex')
                    pbar.update(1)


class PruningHooks(Enum):
    PRUNERWITHGRAD = PrunerWithGrad
    PRUNERWITHREG = PrunerWithReg
    ZEROSHOTPRUNER = ZeroShotPruner
