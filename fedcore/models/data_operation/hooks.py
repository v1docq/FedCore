import torch
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.models.network_impl.hooks import BaseHook
from fedcore.repository.constanst_repository import PRUNER_WITHOUT_REQUIREMENTS, PRUNER_REQUIRED_REG
import torch.nn.utils.prune as prune
class PairwiseAugmentationHook(BaseHook):
    _SUMMON_KEY = 'pairwise'
    _hook_place = 50

    def __init__(self, params, model):
        super().__init__(params, model)
        self.optimizer_for_grad = params.get('optimizer_for_grad_acc', None)
        self.criterion_for_grad = params.get('criterion_for_grad', None)
        self.zeroshot_names = ["Magnitude", 'LAMP', 'Random']
        self.device = default_device()

    def _define_pruner_type(self, importance):
        zeroshot_cond_one = isinstance(importance, tuple(PRUNER_WITHOUT_REQUIREMENTS.values()))
        zeroshot_cond_two = any([str(importance).__contains__(x) for x in self.zeroshot_names])
        zeroshot_pruner = all([zeroshot_cond_one, zeroshot_cond_two])
        pruner_with_grads = all([not zeroshot_pruner, any([str(importance).__contains__('Taylor'),
                                                           str(importance).__contains__('Hessian')
                                                           ])])
        pruner_with_reg = all([not zeroshot_pruner, any([str(importance).__contains__('BNScale'),
                                                         str(importance).__contains__('Group')])])
        pruner_in_depth = str(importance).__contains__('depth')
        if pruner_in_depth:
            callback = 'DepthPruner'
        elif pruner_with_reg:
            callback = 'RegPruner'
        elif pruner_with_grads:
            callback = 'GradPruner'
        elif zeroshot_pruner:
            callback = 'ZeroShotPruner'
        return callback

    def __call__(self, importance, **kws):
        callback_type = self._define_pruner_type(importance)
        trigger_result = self.trigger(callback_type, kws)
        if trigger_result:
            self.action(callback_type, kws)

    def _accumulate_grads(self, data, target):
        data, target = data.to(self.device), target.to(self.device)
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
