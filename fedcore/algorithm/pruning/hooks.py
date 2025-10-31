import copy
import torch
from tqdm import tqdm
from fedcore.algorithm.pruning.pruning_validation import PruningValidator
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
from fedcore.models.network_impl.hooks import BaseHook
from fedcore.repository.constanst_repository import PRUNER_WITHOUT_REQUIREMENTS, PrunerImportances
import torch.nn.utils.prune as prune
from typing import TYPE_CHECKING
import torch_pruning as tp
if TYPE_CHECKING:
    from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
    


class ZeroShotPruner(BaseHook):
    _SUMMON_KEY = 'prune_each'
    _hook_place = 50

    def __init__(self, pruner: tp.BasePruner, pruning_iterations: int, prune_each: int):
        super().__init__()
        self.pruner = pruner
        self.pruning_iterations = pruning_iterations
        self.prune_each = prune_each
        

    def link_to_trainer(self, hookable_trainer: 'BaseNeuralModel'):
        super().link_to_trainer(hookable_trainer)
        self.criterion_for_pruner = hookable_trainer.criterion
        self.device = default_device()

    @classmethod
    def check_init(cls, d: dict):
        if (not super().check_init(d)):
            return False
        if cls != define_pruner_hook_type(d["importance"]):
            return False
        return True

    def _backward_propagation_step(self, data, target):
        data, target = data.to(self.device), target.to(self.device)
        data, target = data.to(torch.float32), target.to(torch.float32)
        out = self.model(data)
        if isinstance(self.criterion_for_pruner, torch.nn.CrossEntropyLoss):  # classification task
            target = target.to(torch.int64)  # convert probalistic output to labels)
        try:
            loss = self.criterion_for_pruner(out, target)
        except Exception:
            loss = self.criterion_for_pruner(out, target)
        loss.backward()
        return loss

    def trigger(self, epoch, kws) -> bool:
        if not self.prune_each:
            return False
        if self.prune_each != -1:
            return not epoch % self.prune_each
        else:
            return epoch == self.hookable_trainer.epochs

    def pruning_operation(self):
        pruning_hist = []
        for i in range(self.pruning_iterations):
            potential_groups_to_prune = self.pruner.step(interactive=True)
            for group in potential_groups_to_prune:
                dep, idxs = group[0]
                layer = dep.layer
                pruning_fn = dep.pruning_fn
                pruning_hist.append((layer, idxs, pruning_fn))
                group.prune()

    def action(self, epoch, kws):
        self.pruning_operation()
        PruningValidator.validate_pruned_layers(self.hookable_trainer.model)


class PrunerWithGrad(ZeroShotPruner):
    _SUMMON_KEY = 'prune_each'
    _hook_place = 50

    def link_to_trainer(self, hookable_trainer: 'BaseNeuralModel'):
        super().link_to_trainer(hookable_trainer)

    def action(self, epoch, kws):
        print(f"Gradients accumulation")
        print(f"==========================================")
        for i, (data, target) in enumerate(kws['val_loader']):
            self._backward_propagation_step(data, target)
        self.pruning_operation()
        PruningValidator.validate_pruned_layers(self.hookable_trainer.model)


class PrunerWithReg(ZeroShotPruner):
    _SUMMON_KEY = 'prune_each'
    _hook_place = 50
            
    def link_to_trainer(self, hookable_trainer: 'BaseNeuralModel'):
        super().link_to_trainer(hookable_trainer)

    def _regularize_model_params(self, pruner, val_dataloader):
        pruner.update_regularizer()  # <== initialize regularizer. Define model groups for pruning
        val_batches = len(val_dataloader) - 1
        with tqdm(total=val_batches, desc='Pruning reg', ) as pbar:
            for i, (data, target) in enumerate(val_dataloader):
                if i != 0:
                    self.hookable_trainer.optimizer.zero_grad()
                    loss = self._backward_propagation_step(data, target)
                    pruner.regularize(self.model) # after loss.backward()
                    self.hookable_trainer.optimizer.step()  # <== for sparse training
                    pbar.update(1)
        return pruner

    def _prune_after_reg(self, pruner: tp.BasePruner, pruning_iter):
        pruning_hist = []
        for i in range(pruning_iter):
            potential_groups_to_prune = pruner.step(interactive=True)
            for group in potential_groups_to_prune:
                dep, idxs = group[0]
                layer = dep.layer
                pruning_fn = dep.pruning_fn
                pruning_hist.append((layer, idxs, pruning_fn))
                group.prune()

    def action(self, epoch, kws):
        pruning_iter = self.pruning_iterations
        val_dataloader = kws['val_loader']
        pruner = self._regularize_model_params(self.pruner, val_dataloader)
        self._prune_after_reg(pruner, pruning_iter)
        PruningValidator.validate_pruned_layers(self.hookable_trainer.model)


class PrunerInDepth(ZeroShotPruner):
    _SUMMON_KEY = 'prune_each'
    _hook_place = 50

    _ACTIVATION_TI_REPLACE = [torch.nn.ReLU, torch.nn.GELU]
    _CONV_LAYER_TO_REPLACE = [torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d]

    def link_to_trainer(self, hookable_trainer: 'BaseNeuralModel'):
        super().link_to_trainer(hookable_trainer)
        self.pruning_ratio = 0.5
        self.activation_replace_hooks = {}
        self.conv_replace_hooks = {}

    class TorchModuleHook:
        def __init__(self, module, backward=False):
            self.output = None
            if backward == False:
                self.hook = module.register_forward_hook(self.hook_fn)
            else:
                self.hook = module.register_backward_hook(self.hook_fn)

        def hook_fn(self, module, input, output):  # for torch compability
            self.output = input[0]

        def close(self):
            self.hook.remove()

    def _collect_activation_val(self):
        model_blocks = list(self.model.named_modules())
        all_conv_layers = []
        for name, module in model_blocks:
            for conv in PrunerInDepth._ACTIVATION_TI_REPLACE:
                if type(module) == conv:
                    all_conv_layers.append(name)
            for act in PrunerInDepth._CONV_LAYER_TO_REPLACE:
                if type(module) == act:
                    self.activation_replace_hooks.update({name: PrunerInDepth.TorchModuleHook(module)})
                    self.conv_replace_hooks.update({name: all_conv_layers[-1]})

    def _connect_activation_and_layers(self, layers_entropy):
        layers_to_prune = []
        for key in self.activation_replace_hooks.keys():
            if key == 'relu':
                layers_to_prune.append('conv1')
                layers_entropy['conv1'] = layers_entropy.pop("relu")
                del self.conv_replace_hooks[key]
            else:
                # layer_num = key.split('.relu')[0].split('layer')[1]
                # name = key.replace('relu', 'conv')
                # name = name + layer_num
                name = self.conv_replace_hooks[key]
                layers_to_prune.append(name)
                layers_entropy[name] = layers_entropy.pop(key)
        return layers_entropy, layers_to_prune

    def _get_layer_magnitude(self, layers_to_prune, layers_entropy):
        layer_entro_magni = {}
        for key in layers_entropy.keys():
            if layers_entropy[key] == 0:
                layers_to_prune.remove(key)

        for name, module in self.model.named_modules():
            if name in layers_to_prune:
                non_zero_weights = torch.abs(module.weight)[module.weight != 0]
                if torch.numel(non_zero_weights) == 0:  # is_already_zero_weight_layer
                    layers_to_prune.remove(name)
                else:  # calculate layer "magnitude" with respect to activation entropy
                    layer_entro_magni[name] = layers_entropy[name] * torch.mean(non_zero_weights)

        total_layers_entro_magni = 0
        for key in layer_entro_magni.keys():
            total_layers_entro_magni += layer_entro_magni[key].item()
        return layer_entro_magni, total_layers_entro_magni

    def _iter_pruning(self, layers_to_prune, layer_entro_magni, total_layers_entro_magni):
        filtred_layers_to_prune = copy.deepcopy(layers_to_prune)
        left_amount = {}
        total_layers_weight_params = 0
        entropy_layer_head_expo = {}
        for name, module in self.model.named_modules():
            if name in layers_to_prune:
                total_layers_weight_params += torch.numel(module.weight[module.weight != 0])
        total_layers_weight_paras_to_prune = self.pruning_ratio * total_layers_weight_params
        while True:
            amout_changed = False
            total_entropy_layer_head_expo = 0
            entropy_magni_layer_head = {name: total_layers_entro_magni / (layer_entro_magni[name])
                                        for name, module in self.model.named_modules() if name in filtred_layers_to_prune}
            max_value_entropy_magni_layer_head = max(entropy_magni_layer_head.values())
            for name, module in self.model.named_modules():
                if name in filtred_layers_to_prune:
                    entropy_layer_head_expo.update({name: torch.exp(entropy_magni_layer_head[name] -
                                                               max_value_entropy_magni_layer_head).item()})

                    total_entropy_layer_head_expo += entropy_layer_head_expo[name]
            fix_prun_amount = {name: total_layers_weight_paras_to_prune * (entropy_layer_head_expo[name]
                                                                           / total_entropy_layer_head_expo)
                               for name, module in self.model.named_modules() if name in filtred_layers_to_prune}

            for name, module in self.model.named_modules():
                if name in filtred_layers_to_prune:
                    left_amount[name] = torch.numel(module.weight[module.weight != 0])
                    if left_amount[name] < fix_prun_amount[name]:
                        fix_prun_amount[name] = left_amount[name]
                        total_layers_weight_paras_to_prune -= left_amount[name]
                        total_layers_entro_magni -= layer_entro_magni[name]
                        filtred_layers_to_prune.remove(name)
                        amout_changed = True
            if not amout_changed:
                break
            return filtred_layers_to_prune, fix_prun_amount

    def action(self, epoch, kws):
        # Step 1. Initialise data for predict loop and entropy monitoring
        train_dataloader = kws['train_loader']
        # Step 2. Calculate the entropy for each layer
        layers_entropy, total_loss = self.eval_action_entropy(train_dataloader)
        # Step 3. Get list of layers to prune which connect with activation
        layers_entropy, layers_to_prune = self._connect_activation_and_layers(layers_entropy)
        # Step 4. Initialise data for predict loop and entropy monitoring
        self.iterative_prune(layers_entropy, layers_to_prune)
        PruningValidator.validate_pruned_layers(self.hookable_trainer.model)

    def iterative_prune(self, layers_entropy, layers_to_prune):

        layer_entro_magni, total_layers_entro_magni = self._get_layer_magnitude(layers_to_prune, layers_entropy)
        filtred_layers_to_prune, fix_prun_amount = self._iter_pruning(layers_to_prune,
                                                                      layer_entro_magni, total_layers_entro_magni)
        #implement depth prune using Identity blocks to replace conv and relu with low entropy?
        temp_model = copy.deepcopy(self.model)
        for name, module in temp_model.named_modules():
            if name in filtred_layers_to_prune:
                prune.l1_unstructured(module, name='weight', amount=int(fix_prun_amount[name]))

        #finetune?
        for name, module in temp_model.named_modules():
            if name in filtred_layers_to_prune:
                prune.remove(module, 'weight')
        _ = 1

    def eval_action_entropy(self, train_loader):
        def eval_on_train(train_loader):
            total_loss = 0
            with torch.no_grad():
                for data in tqdm(train_loader):
                    features, labels = data[0].to(self.device), data[1].to(self.device)
                    outputs = self.model(features)
                    loss = self.criterion_for_pruner(outputs, labels)
                    total_loss += loss.item()
            return total_loss

        def eval_entropy():
            entropy = {key: 0 for key in self.activation_replace_hooks.keys()}
            for key in self.activation_replace_hooks.keys():  # For different layers
                activation_vals = self.activation_replace_hooks[key].output  # get activation vals
                heaviside_tensor = torch.tensor(data=[0], dtype=torch.float32).to(self.device)  # create mask for 0 vals
                full_p_one = torch.heaviside(activation_vals, heaviside_tensor)  # apply heaviside mask
                p_one = torch.mean(full_p_one, dim=0)  # apply mean along samples dimension
                state = self.activation_replace_hooks[key].output > 0  # get mask for initial attention
                state = state.reshape(state.shape[0], state.shape[1], -1)  # reshape
                state_sum = torch.mean(state * 1.0, dim=[0, 2])  # activation val for each sample across all channels
                state_sum_num = torch.sum((state_sum != 0) * (state_sum != 1))  # looking for zero and one vals

                if state_sum_num != 0:
                    while len(p_one.shape) > 1:
                        # if args.model == 'Resnet18':
                        #     p_one = torch.mean(p_one, dim=1)
                        # elif args.model == 'Swin-T':
                        #     p_one = torch.mean(p_one, dim=0)
                        p_one = torch.mean(p_one, dim=1)
                    p_one = (p_one * (state_sum != 0) * (state_sum != 1) * 1.0)
                    log_entropy_act = torch.log2(torch.clamp(p_one, min=1e-5))
                    inverse_log_entropy_act = torch.log2(torch.clamp(1 - p_one, min=1e-5))
                    entropy_sum = torch.sum((p_one * log_entropy_act) + ((1 - p_one) * inverse_log_entropy_act))
                    entropy[key] -= entropy_sum / state_sum_num
                else:
                    entropy[key] -= 0
            return entropy

        self.model.eval()
        self._collect_activation_val()
        train_loss = eval_on_train(train_loader)
        train_entropy = eval_entropy()

        layers_entropy = {key: train_entropy[key] / len(train_loader) for key in self.activation_replace_hooks.keys()}
        total_loss = train_loss / len(train_loader)
        return layers_entropy, total_loss


def define_pruner_hook_type(importance_name: str) -> type[ZeroShotPruner]:
    zeroshot_pruner = PrunerImportances[importance_name] in tuple(PRUNER_WITHOUT_REQUIREMENTS.values())
    pruner_with_grads = not zeroshot_pruner and any(['taylor' in importance_name, 'hessian' in importance_name])
    pruner_with_reg = not zeroshot_pruner and any(['bn_scale' in importance_name,
                                                     'group' in importance_name])
    pruner_in_depth = 'depth' in importance_name
    callback = ZeroShotPruner
    if pruner_in_depth:
        callback = PrunerInDepth
    elif pruner_with_reg:
        callback = PrunerWithReg
    elif pruner_with_grads:
        callback = PrunerWithGrad
    elif zeroshot_pruner:
        callback = ZeroShotPruner
    return callback
