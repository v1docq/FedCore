"""
Pruning hooks and utilities for FedCore.

This module provides several training-time hooks that orchestrate different
pruning strategies for PyTorch models:

- ZeroShotPruner
    Runs zero-shot (no gradient) pruning using importance criteria that do not
    require backprop (e.g., Magnitude, LAMP, Random).
- PrunerWithGrad
    Accumulates gradients on a validation set, then runs pruning methods that
    depend on gradient-based importance (e.g., Taylor/Hessian families).
- PrunerWithReg
    Performs a short sparse-training / regularization pass to build pruning
    masks (e.g., BN-scale, Group regularization), then applies pruning.
- PrunerInDepth
    Experimental depth-wise layer selection based on activation entropy and
    magnitude heuristics; prunes layers proportionally to an entropy-based
    allocation.

Integration
-----------
All classes inherit from :class:`fedcore.models.network_impl.hooks.BaseHook`
and are intended to be plugged into the training loop through FedCore's hook
system. Each hook decides whether to run at a given moment via ``trigger``,
and executes pruning in ``action``.

Notes
-----
* ``TorchModuleHook`` is a thin helper to capture module activations without
  changing layer code; it uses forward/backward hooks.
* This file does not modify model logic when hooks are inactive.
"""

import copy
from enum import Enum
import torch
from torch import nn, optim
from tqdm import tqdm
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.models.network_impl.hooks import BaseHook
from fedcore.repository.constanst_repository import PRUNER_WITHOUT_REQUIREMENTS, PRUNER_REQUIRED_REG
import torch.nn.utils.prune as prune


class TorchModuleHook:
    """
    Lightweight wrapper around PyTorch forward/backward hooks to capture tensors.

    Parameters
    ----------
    module : nn.Module
        Target module to attach the hook to.
    backward : bool, default=False
        If ``False``, registers a forward hook. If ``True``, registers a backward
        hook. (Backward hooks are deprecated in recent PyTorch versions; kept here
        for compatibility.)

    Attributes
    ----------
    output : torch.Tensor | None
        Captured tensor from the hook callback. For the forward hook here, we
        store ``input[0]`` (historical choice for downstream use).
    hook : torch.utils.hooks.RemovableHandle
        The handle returned by ``register_forward_hook`` / ``register_backward_hook``.
    """
    def __init__(self, module, backward=False):
        self.output = None
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        """Hook callback that stores the leading input tensor for later use."""
        self.output = input[0]

    def close(self):
        """Detach the hook from the module."""
        self.hook.remove()


class ZeroShotPruner(BaseHook):
    """
    Zero-shot pruning hook.

    Runs pruning algorithms that do not require gradients or extra passes
    (e.g., Magnitude, LAMP, Random). The actual pruning work is delegated to
    a pruner object provided via ``kws['pruner_objects']``.

    Expected ``kws['pruner_objects']`` schema
    -----------------------------------------
    pruner_cls : object
        Pruner instance exposing ``step(interactive=True)`` that yields groups to prune.
    pruning_iterations : int
        Number of ``step`` iterations to perform.
    input_data : Any
        Optional container for dataloaders if a derived hook needs them.

    Hook metadata
    -------------
    _SUMMON_KEY : 'pruning'
    _hook_place : 50
    """
    _SUMMON_KEY = 'pruning'
    _hook_place = 50

    def __init__(self, params, model):
        """
        Parameters
        ----------
        params : dict-like
            Hook configuration. Recognized keys:
              - 'optimizer_for_grad_acc': optional optimizer (unused here).
              - 'criterion_for_grad'   : optional loss (unused here).
        model : nn.Module
            Target model.
        """
        super().__init__(params, model)
        self.optimizer_for_grad = params.get('optimizer_for_grad_acc', None)
        self.criterion_for_grad = params.get('criterion_for_grad', None)
        self.zeroshot_names = ["Magnitude", 'LAMP', 'Random']
        self.device = default_device()

    def _define_pruner_type(self, importance):
        """
        Infer pruner category from an importance object or name.

        Returns
        -------
        str
            One of {'DepthPruner', 'RegPruner', 'GradPruner', 'ZeroShotPruner'}.
        """
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
        """Entry point used by the training loop to run the appropriate pruner."""
        callback_type = self._define_pruner_type(importance)
        trigger_result = self.trigger(callback_type, kws)
        if trigger_result:
            self.action(callback_type, kws)

    def _accumulate_grads(self, data, target):
        """
        Run a forward + backward pass to accumulate gradients (helper).

        Converts inputs/targets to float32 by default and applies the configured
        criterion. For classification with CrossEntropyLoss, targets are cast to
        int64.

        Returns
        -------
        torch.Tensor
            Loss tensor used for the backward pass.
        """
        data, target = data.to(self.device), target.to(self.device)
        data, target = data.to(torch.float32), target.to(torch.float32)
        out = self.model(data)
        if isinstance(self.criterion_for_grad, torch.nn.CrossEntropyLoss):  # classification task
            target = target.to(torch.int64)  # convert probabilistic output to labels
        try:
            loss = self.criterion_for_grad(out, target)
        except Exception:
            loss = self.criterion_for_grad(out, target)
        loss.backward()
        return loss

    def trigger(self, callback_type, kw) -> bool:
        """Return True if this hook should run for zero-shot pruning."""
        return callback_type.__contains__('ZeroShot')

    def pruning_operation(self, kws):
        """
        Core loop that asks the pruner for candidate groups and prunes them.

        Parameters
        ----------
        kws : dict
            Must contain 'pruner_cls' and 'pruning_iterations' as described in the
            class docstring.
        """
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
        """Execute zero-shot pruning."""
        pruner_metadata = kws['pruner_objects']
        self.pruning_operation(pruner_metadata)


class PrunerWithGrad(ZeroShotPruner):
    """
    Gradient-based pruning hook.

    Accumulates gradients on a held-out validation dataloader and then runs
    the provided pruner (which assumes gradient-informed importance).

    See also: :class:`ZeroShotPruner` for the pruner metadata layout.
    """
    _SUMMON_KEY = 'pruning'
    _hook_place = 50

    def __init__(self, params, model):
        """Initialize and read the loss function from params."""
        super().__init__(params, model)
        self.criterion_for_grad = params.get('criterion_for_grad', None)

    def trigger(self, callback_type, kw) -> bool:
        """Return True if this hook should run for grad-based pruning."""
        return callback_type.__contains__('GradPruner')

    def action(self, callback_type, kws):
        """
        Accumulate gradients on the validation set, then prune.

        Expects ``kws['pruner_objects']['input_data'].features.val_dataloader``.
        """
        print(f"Gradients accumulation")
        print(f"==========================================")
        pruner_metadata = kws['pruner_objects']
        for i, (data, target) in enumerate(pruner_metadata['input_data'].features.val_dataloader):
            self._accumulate_grads(data, target)
        self.pruning_operation(pruner_metadata)


class PrunerWithReg(ZeroShotPruner):
    """
    Regularization-based pruning hook.

    Performs a short optimization loop that updates a regularizer maintained by
    the pruner (e.g., BN-scale or group-lasso style), and afterwards applies the
    resulting masks to prune the model.
    """
    _SUMMON_KEY = 'pruning'
    _hook_place = 50

    def __init__(self, params, model):
        """
        Parameters
        ----------
        params : dict-like
            Reads 'optimizer_for_grad_acc' and 'criterion_for_grad'. If no optimizer
            is provided, uses ``optim.Adam(self.model.parameters(), lr=1e-4)``.
        """
        super().__init__(params, model)
        self.optimizer_for_grad = params.get('optimizer_for_grad_acc', None)
        self.criterion_for_grad = params.get('criterion_for_grad', None)
        if self.optimizer_for_grad is None:
            self.optimizer_for_grad = optim.Adam(self.model.parameters(),
                                                 lr=0.0001)

    def regularize_model_params(self, pruner, val_dataloader):
        """
        Run a brief sparse-training loop to drive the pruner's regularizer.

        Parameters
        ----------
        pruner : object
            Must implement ``update_regularizer()``, ``regularize(model)`` and
            ``step(interactive=True)`` API.
        val_dataloader : DataLoader
            Validation dataloader used to accumulate gradients.

        Returns
        -------
        object
            The same ``pruner`` instance after regularization updates.
        """
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
        """
        Apply pruning after regularization using the pruner's step interface.

        Parameters
        ----------
        pruner : object
            Pruner instance with ``step(interactive=True)``.
        pruning_iter : int
            Number of iterations to run.
        """
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
        """Return True if this hook should run for regularization-based pruning."""
        return callback_type.__contains__('RegPruner')

    def action(self, callback_type, kws):
        """Execute the regularize-then-prune sequence."""
        pruner_metadata = kws['pruner_objects']
        pruner = pruner_metadata['pruner_cls']
        pruning_iter = pruner_metadata['pruning_iterations']
        val_dataloader = pruner_metadata['input_data'].features.val_dataloader
        pruner = self.regularize_model_params(pruner, val_dataloader)
        self.prune_after_reg(pruner, pruning_iter)


class PrunerInDepth(ZeroShotPruner):
    """
    Entropy-based depth pruning hook (experimental).

    Measures activation entropy around selected nonlinearities, maps those to
    their preceding convolutional layers, and allocates a global pruning budget
    proportionally to an entropy × magnitude heuristic.

    The implementation uses temporary hooks to capture activations, computes
    per-layer entropies, decides a per-layer pruning amount, performs
    unstructured L1 pruning, and removes reparametrizations afterwards.
    """
    _SUMMON_KEY = 'pruning'
    _hook_place = 50

    def __init__(self, params, model):
        """
        Parameters
        ----------
        params : dict-like
            Reads optional optimizer/loss for gradient accumulation.
            Defaults to Adam(lr=1e-4) when optimizer is not provided.
        """
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
        """Return True if this hook should run for 'Depth' pruning."""
        return callback_type.__contains__('Depth')

    def _collect_activation_val(self):
        """
        Attach forward hooks to selected activations and remember their preceding convs.

        Populates:
        ----------
        self.activation_replace_hooks : Dict[str, TorchModuleHook]
        self.conv_replace_hooks       : Dict[str, str]  (activation_name -> conv_layer_name)
        """
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
        """
        Map activation entropy measurements to corresponding conv layers.

        Parameters
        ----------
        layers_entropy : Dict[str, torch.Tensor]
            Entropy per activation hook.

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], List[str]]
            Updated dict keyed by conv layer names and the list of layers to prune.
        """
        layers_to_prune = []
        for key in self.activation_replace_hooks.keys():
            if key == 'relu':
                layers_to_prune.append('conv1')
                layers_entropy['conv1'] = layers_entropy.pop("relu")
                del self.conv_replace_hooks[key]
            else:
                name = self.conv_replace_hooks[key]
                layers_to_prune.append(name)
                layers_entropy[name] = layers_entropy.pop(key)
        return layers_entropy, layers_to_prune

    def _get_layer_magnitude(self, layers_to_prune, layers_entropy):
        """
        Build entropy × magnitude scores and filter obviously empty layers.

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], float]
            Per-layer scores and total score sum.
        """
        layer_entro_magni = {}
        for key in layers_entropy.keys():
            if layers_entropy[key] == 0:
                layers_to_prune.remove(key)

        for name, module in self.model.named_modules():
            if name in layers_to_prune:
                non_zero_weights = torch.abs(module.weight)[module.weight != 0]
                if torch.numel(non_zero_weights) == 0:  # already zeroed
                    layers_to_prune.remove(name)
                else:
                    layer_entro_magni[name] = layers_entropy[name] * torch.mean(non_zero_weights)

        total_layers_entro_magni = 0
        for key in layer_entro_magni.keys():
            total_layers_entro_magni += layer_entro_magni[key].item()
        return layer_entro_magni, total_layers_entro_magni

    def _iter_pruning(self, layers_to_prune, layer_entro_magni, total_layers_entro_magni):
        """
        Allocate a global pruning budget across layers via softmax of inverted scores.

        Returns
        -------
        Tuple[List[str], Dict[str, float]]
            Filtered list of layers and the per-layer pruning amounts (counts).
        """
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
        """
        Full pipeline:
          1) collect activation stats on train dataloader,
          2) compute per-layer entropy,
          3) map to conv layers and run iterative pruning.
        """
        pruner_metadata = kws['pruner_objects']
        train_dataloader = pruner_metadata['input_data'].features.train_dataloader
        layers_entropy, total_loss = self.eval_action_entropy(train_dataloader)
        layers_entropy, layers_to_prune = self._connect_activation_and_layers(layers_entropy)
        self.iterative_prune(layers_entropy, layers_to_prune)

    def iterative_prune(self, layers_entropy, layers_to_prune):
        """Run per-layer allocation and perform unstructured L1 pruning."""
        layer_entro_magni, total_layers_entro_magni = self._get_layer_magnitude(layers_to_prune, layers_entropy)
        filtred_layers_to_prune, fix_prun_amount = self._iter_pruning(layers_to_prune,
                                                                      layer_entro_magni, total_layers_entro_magni)
        # implement depth prune using Identity blocks to replace conv and relu with low entropy?
        temp_model = copy.deepcopy(self.model)
        for name, module in temp_model.named_modules():
            if name in filtred_layers_to_prune:
                prune.l1_unstructured(module, name='weight', amount=int(fix_prun_amount[name]))

        # finetune?
        for name, module in temp_model.named_modules():
            if name in filtred_layers_to_prune:
                prune.remove(module, 'weight')
        _ = 1

    def eval_action_entropy(self, train_loader):
        """
        Evaluate model on the training loader and compute activation entropy.

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], float]
            Per-activation entropy and average training loss.
        """
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
    """
    Registry of available pruning hooks.

    Members
    -------
    PRUNERWITHGRAD : type[PrunerWithGrad]
        Gradient-based pruning.
    PRUNERWITHREG : type[PrunerWithReg]
        Regularization-based pruning.
    ZEROSHOTPRUNER : type[ZeroShotPruner]
        Zero-shot pruning (no gradients).
    DEPTHPRUNER : type[PrunerInDepth]
        Entropy/magnitude depth pruning (experimental).
    """
    PRUNERWITHGRAD = PrunerWithGrad
    PRUNERWITHREG = PrunerWithReg
    ZEROSHOTPRUNER = ZeroShotPruner
    DEPTHPRUNER = PrunerInDepth
