import copy
from enum import Enum
import torch
from torch import nn, optim
from tqdm import tqdm
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.models.network_impl.utils.hooks import BaseHook
from fedcore.repository.constant_repository import PRUNER_WITHOUT_REQUIREMENTS, PRUNER_REQUIRED_REG
import torch.nn.utils.prune as prune


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


class ZeroShotPruner(BaseHook):
    _SUMMON_KEY = 'pruning'
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


class PrunerInDepth(ZeroShotPruner):
    _SUMMON_KEY = 'pruning'
    _hook_place = 50

    def __init__(self, params, model):
        super().__init__(params, model)
        self.optimizer_for_grad = params.get('optimizer_for_grad_acc', None)
        self.criterion_for_grad = params.get('criterion_for_grad', None)
        self.pruning_ratio = 0.5
        if self.optimizer_for_grad is None:
            self.optimizer_for_grad = optim.Adam(self.model.parameters(),
                                                 lr=0.0001)
        self.activation_replace_hooks = {}
        self.conv_replace_hooks = {}
        self.activation_to_replace = [torch.nn.ReLU, torch.nn.GELU]
        self.conv_layer_to_replace = [torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d]

    def trigger(self, callback_type, kw) -> bool:
        return callback_type.__contains__('Depth')

    def _collect_activation_val(self):
        model_blocks = list(self.model.named_modules())
        all_conv_layers = []
        for name, module in model_blocks:
            for conv in self.conv_layer_to_replace:
                if type(module) == conv:
                    all_conv_layers.append(name)
            for act in self.activation_to_replace:
                if type(module) == act:
                    self.activation_replace_hooks.update({name: TorchModuleHook(module)})
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

    def action(self, callback, kws):
        # Step 1. Initialise data for predict loop and entropy monitoring
        pruner_metadata = kws['pruner_objects']
        train_dataloader = pruner_metadata['input_data'].features.train_dataloader
        # Step 2. Calculate the entropy for each layer
        layers_entropy, total_loss = self.eval_action_entropy(train_dataloader)
        # Step 3. Get list of layers to prune which connect with activation
        layers_entropy, layers_to_prune = self._connect_activation_and_layers(layers_entropy)
        # Step 4. Initialise data for predict loop and entropy monitoring
        self.iterative_prune(layers_entropy, layers_to_prune)

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
                    loss = self.criterion_for_grad(outputs, labels)
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


class PruningHooks(Enum):
    PRUNERWITHGRAD = PrunerWithGrad
    PRUNERWITHREG = PrunerWithReg
    ZEROSHOTPRUNER = ZeroShotPruner
    DEPTHPRUNER = PrunerInDepth
