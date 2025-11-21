"""Hook implementations for structured pruning during training.

This module contains several pruning hooks that can be attached to
:class:`BaseNeuralModel`:

* :class:`ZeroShotPruner` – performs pruning based on a pre-configured
  :class:`torch_pruning.BasePruner` without any additional requirements.
* :class:`PrunerWithGrad` – accumulates gradients on a validation set before
  calling the underlying pruner (for gradient-based importance).
* :class:`PrunerWithReg` – performs sparse training (regularization) before
  pruning (for BN-scale / group-norm–based importance).
* :class:`PrunerInDepth` – entropy- and magnitude-based pruning for
  convolutional layers, using activations and weight statistics.

The helper :func:`define_pruner_hook_type` selects an appropriate hook class
based on the configured importance metric.
"""

import copy
import torch
from tqdm import tqdm
from fedcore.algorithm.pruning.pruning_validation import PruningValidator
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
from fedcore.models.network_impl.hooks import BaseHook
from fedcore.repository.constanst_repository import PRUNER_WITHOUT_REQUIREMENTS, PrunerImportances
import torch.nn.utils.prune as prune
from typing import TYPE_CHECKING, Union
import torch_pruning as tp
if TYPE_CHECKING:
    from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
    


class ZeroShotPruner(BaseHook):
    """Epoch-based pruning hook using a preconfigured Torch-Pruning pruner.

    This hook calls a :class:`torch_pruning.BasePruner` instance at fixed
    epochs (every ``prune_each`` epochs) and applies structural pruning in
    multiple iterations. It does not require any extra information beyond
    what the pruner already encodes (zero-shot pruning).

    Parameters
    ----------
    pruner : tp.BasePruner
        Initialized Torch-Pruning pruner that defines importance measure and
        dependency graph.
    pruning_iterations : int
        Number of iterations of ``pruner.step(interactive=True)`` to perform
        each time the hook is triggered.
    prune_each : int
        Epoch interval at which pruning should be performed (passed to
        :meth:`BaseHook.is_epoch_arrived_default`).

    Notes
    -----
    The hook also stores a reference to the trainer's loss criterion
    (``criterion_for_pruner``) to be reused in subclasses that depend on
    backward passes (e.g. :class:`PrunerWithGrad`, :class:`PrunerWithReg`).
    """
    _SUMMON_KEY = 'prune_each'
    HOOK_PLACE = 50

    def __init__(self, pruner: tp.BasePruner, pruning_iterations: int, prune_each: int):
        super().__init__()
        self.pruner = pruner
        self.pruning_iterations = pruning_iterations
        self.prune_each = prune_each
        

    def link_to_trainer(self, hookable_trainer: 'BaseNeuralModel'):
        """Attach the hook to a trainer and cache common resources.

        Parameters
        ----------
        hookable_trainer : BaseNeuralModel
            Trainer instance that provides ``criterion`` and device settings.
        """
        super().link_to_trainer(hookable_trainer)
        self.criterion_for_pruner = hookable_trainer.criterion
        self.device = default_device()

    @classmethod
    def check_init(cls, d: dict):
        """Check whether this particular hook type should be instantiated.

        Parameters
        ----------
        d : dict
            Hook configuration dictionary.

        Returns
        -------
        bool
            ``True`` if the base hook initialization passes and the importance
            name in config corresponds to this concrete pruner class,
            ``False`` otherwise.
        """
        if (not super().check_init(d)):
            return False
        if cls != define_pruner_hook_type(d["importance"]):
            return False
        return True

    def _backward_propagation_step(self, data, target):
        """Perform a single backward pass for pruning-related statistics.

        This helper method is used by gradient-/regularization-based pruners
        to accumulate gradients using the trainer's criterion.

        Parameters
        ----------
        data : torch.Tensor
            Batch of input features.
        target : torch.Tensor
            Batch of targets/labels.

        Returns
        -------
        torch.Tensor
            Computed loss value for the batch.
        """
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
        """Decide whether pruning should be performed at the current epoch.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        kws : dict
            Extra arguments from the trainer (unused here).

        Returns
        -------
        bool
            ``True`` if current epoch matches the ``prune_each`` schedule,
            ``False`` otherwise.
        """
        return self.is_epoch_arrived_default(epoch, self.prune_each)

    def pruning_operation(self):
        """Run configured number of pruning iterations with the pruner.

        For each iteration, calls ``pruner.step(interactive=True)`` to obtain
        groups of prune operations and then executes ``group.prune()`` for
        all proposed groups. A simple pruning history is collected
        (but not used further in the current implementation).
        """
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
        """Apply pruning and validate the resulting model.

        Parameters
        ----------
        epoch : int
            Current epoch (unused, present for hook API compatibility).
        kws : dict
            Extra arguments from the trainer (unused here).

        Notes
        -----
        Calls :meth:`pruning_operation` and then
        :meth:`PruningValidator.validate_pruned_layers` to ensure that
        pruning did not break the model structure.
        """
        self.pruning_operation()
        PruningValidator.validate_pruned_layers(self.hookable_trainer.model)


class PrunerWithGrad(ZeroShotPruner):
    """Pruning hook that accumulates gradients on a validation loader.

    Unlike :class:`ZeroShotPruner`, this variant performs backward passes
    over batches from ``val_loader`` before calling the underlying pruner.
    This is useful for importance metrics that rely on gradients
    (e.g. Taylor-based criteria).
    """
    _SUMMON_KEY = 'prune_each'
    HOOK_PLACE = 50

    def link_to_trainer(self, hookable_trainer: 'BaseNeuralModel'):
        """Attach the hook to a trainer (no extra state beyond base class)."""
        super().link_to_trainer(hookable_trainer)

    def action(self, epoch, kws):
        """Accumulate gradients on validation data, then prune and validate.

        Parameters
        ----------
        epoch : int
            Current epoch (unused, present for hook API compatibility).
        kws : dict
            Dictionary expected to contain ``'val_loader'`` with validation
            data loader.

        Notes
        -----
        For each batch in ``val_loader``, this method calls
        :meth:`_backward_propagation_step` to accumulate gradients, then
        invokes :meth:`pruning_operation` and validates pruned layers.
        """
        print(f"Gradients accumulation")
        print(f"==========================================")
        for i, (data, target) in enumerate(kws['val_loader']):
            self._backward_propagation_step(data, target)
        self.pruning_operation()
        PruningValidator.validate_pruned_layers(self.hookable_trainer.model)


class PrunerWithReg(ZeroShotPruner):
    """Pruning hook that performs sparse-training regularization before pruning.

    This class follows the sparse-training workflow suggested in Torch-Pruning
    examples: it first augments the loss with a regularizer via
    ``pruner.update_regularizer()`` and ``pruner.regularize(model)`` during
    training, and only then runs the structured pruning step.
    """
    _SUMMON_KEY = 'prune_each'
    HOOK_PLACE = 50
            
    def link_to_trainer(self, hookable_trainer: 'BaseNeuralModel'):
        """Attach the hook to a trainer (no extra state beyond base class)."""
        super().link_to_trainer(hookable_trainer)

    def _regularize_model_params(self, pruner: Union[tp.BNScalePruner, tp.GroupNormPruner], train_dataloader):
        """Run sparse regularization over training batches.

        Parameters
        ----------
        pruner : Union[tp.BNScalePruner, tp.GroupNormPruner]
            Torch-Pruning pruner that supports regularization-based sparse
            training (e.g. BNScale or GroupNorm-based pruners).
        train_dataloader :
            Data loader providing training batches ``(data, target)``.

        Returns
        -------
        tp.BasePruner
            The same pruner instance after performing regularization passes.

        Notes
        -----
        For each batch (except the first), gradients are zeroed, a backward
        pass is performed via :meth:`_backward_propagation_step`, and
        ``pruner.regularize(self.model)`` is called before optimizer step.
        """
        """Regularize params, that have small magnitude during training process  
        See https://github.com/VainF/Torch-Pruning?tab=readme-ov-file#sparse-training-optional
        """
        pruner.update_regularizer()  # <== initialize regularizer. Define model groups for pruning
        train_batches = len(train_dataloader) - 1
        with tqdm(total=train_batches, desc='Pruning reg', ) as pbar:
            for i, (data, target) in enumerate(train_dataloader):
                if i != 0:
                    self.hookable_trainer.optimizer.zero_grad()
                    loss = self._backward_propagation_step(data, target)
                    pruner.regularize(self.model) # after loss.backward()
                    self.hookable_trainer.optimizer.step()  # <== for sparse training
                    pbar.update(1)
        return pruner

    def _prune_after_reg(self, pruner: tp.BasePruner, pruning_iter):
        """Run pruning iterations after regularization has been applied.

        Parameters
        ----------
        pruner : tp.BasePruner
            Pruner instance with initialized dependency graph.
        pruning_iter : int
            Number of pruning iterations to perform.
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

    def action(self, epoch, kws):
        """Perform sparse-training regularization, then prune and validate.

        Parameters
        ----------
        epoch : int
            Current epoch (unused, present for hook API compatibility).
        kws : dict
            Dictionary expected to contain ``'train_loader'`` with training
            data loader.
        """
        pruning_iter = self.pruning_iterations
        train_dataloader = kws['train_loader']
        pruner = self._regularize_model_params(self.pruner, train_dataloader)
        self._prune_after_reg(pruner, pruning_iter)
        PruningValidator.validate_pruned_layers(self.hookable_trainer.model)


class PrunerInDepth(ZeroShotPruner):
    """Entropy- and magnitude-based pruning hook for convolutional networks.

    This hook estimates per-layer importance by combining activation entropy
    (measured on ReLU/GELU activations) and weight magnitudes of convolutional
    layers. It then distributes a global pruning budget across layers based on
    this importance and applies unstructured L1 pruning to selected weights.

    Notes
    -----
    This implementation is more experimental and tailored to architectures
    with ``Conv*`` + activation blocks. It internally:

    * collects activations via forward hooks (:class:`TorchModuleHook`);
    * computes per-layer entropy over activation patterns;
    * combines entropy and weight magnitudes into a layer importance score;
    * allocates a pruning budget and prunes weights using
      :func:`torch.nn.utils.prune.l1_unstructured`.
    """
    _SUMMON_KEY = 'prune_each'
    HOOK_PLACE = 50

    _ACTIVATION_TI_REPLACE = [torch.nn.ReLU, torch.nn.GELU]
    _CONV_LAYER_TO_REPLACE = [torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d]

    def link_to_trainer(self, hookable_trainer: 'BaseNeuralModel'):
        """Attach the hook to a trainer and initialize internal state."""
        super().link_to_trainer(hookable_trainer)
        self.pruning_ratio = 0.5
        self.activation_replace_hooks = {}
        self.conv_replace_hooks = {}

    class TorchModuleHook:
        """Simple wrapper around PyTorch forward/backward hooks.

        Parameters
        ----------
        module : torch.nn.Module
            Module to which the hook should be attached.
        backward : bool, optional
            If ``False`` (default), register a forward hook; if ``True``,
            register a backward hook.

        Attributes
        ----------
        output : torch.Tensor or None
            Captured input or output tensor (depending on hook_fn
            implementation).
        """

        def __init__(self, module, backward=False):
            self.output = None
            if backward == False:
                self.hook = module.register_forward_hook(self.hook_fn)
            else:
                self.hook = module.register_backward_hook(self.hook_fn)

        def hook_fn(self, module, input, output):  # for torch compability
            """Hook callback that stores the first input tensor."""
            self.output = input[0]

        def close(self):
            """Remove the underlying PyTorch hook."""
            self.hook.remove()

    def _collect_activation_val(self):
        """Register hooks on activation layers and map them to conv layers.

        This method:

        * collects names of all activations of types in
          :data:`_ACTIVATION_TI_REPLACE`,
        * registers :class:`TorchModuleHook` on them,
        * associates each activation with the last seen convolutional layer
          (from :data:`_CONV_LAYER_TO_REPLACE`) so that entropy estimates can
          be mapped back to conv layers.
        """
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
        """Map activation entropy values to corresponding conv layers.

        Parameters
        ----------
        layers_entropy : dict
            Mapping from activation layer names to entropy values.

        Returns
        -------
        tuple[dict, list[str]]
            Updated ``layers_entropy`` with conv layer names as keys and a
            list of conv layers selected for pruning.
        """
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
        """Combine entropy and weight magnitudes into layer importance scores.

        Parameters
        ----------
        layers_to_prune : list[str]
            Names of layers considered for pruning.
        layers_entropy : dict
            Mapping from layer name to entropy value.

        Returns
        -------
        tuple[dict, float]
            * ``layer_entro_magni`` – per-layer combined score
              (entropy × mean |weight|),
            * ``total_layers_entro_magni`` – sum of all scores.
        """
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
        """Allocate pruning budget across layers according to importance.

        Parameters
        ----------
        layers_to_prune : list[str]
            Names of layers considered for pruning.
        layer_entro_magni : dict
            Per-layer importance scores from :meth:`_get_layer_magnitude`.
        total_layers_entro_magni : float
            Sum of all importance scores.

        Returns
        -------
        tuple[list[str], dict[str, float]]
            Filtered list of layers to prune and per-layer pruning amounts
            (number of weights to prune).
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

    def action(self, epoch, kws):
        """Run the full entropy-based pruning pipeline and validate.

        Steps
        -----
        1. Collect activation values for monitored layers.
        2. Compute entropy for each activation layer.
        3. Map activation entropy to convolutional layers.
        4. Run :meth:`iterative_prune` to apply pruning.
        5. Validate pruned model via :class:`PruningValidator`.

        Parameters
        ----------
        epoch : int
            Current epoch (unused, present for hook API compatibility).
        kws : dict
            Dictionary expected to contain ``'train_loader'`` with training
            data loader.
        """
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
        """Execute entropy-based pruning for selected layers.

        Parameters
        ----------
        layers_entropy : dict
            Mapping from layer name to entropy value.
        layers_to_prune : list[str]
            Names of layers to be considered for pruning.

        Notes
        -----
        A temporary copy of the model is pruned using unstructured L1
        pruning on the ``weight`` parameter. Pruning masks are then removed.
        """
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
        """Evaluate loss and activation entropy over a training loader.

        Parameters
        ----------
        train_loader :
            Data loader that provides training batches.

        Returns
        -------
        tuple[dict, float]
            * ``layers_entropy`` – mapping from activation layer names to
              mean entropy values over the dataset;
            * ``total_loss`` – mean loss value over the dataset.
        """
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
    """Select an appropriate pruning hook class given an importance name.

    The choice is based on global mappings in
    :data:`PrunerImportances` and :data:`PRUNER_WITHOUT_REQUIREMENTS`:

    * zero-shot pruners – do not require gradients or regularization;
    * gradient-based pruners – names that contain ``"taylor"`` or ``"hessian"``;
    * regularization-based pruners – names that contain ``"bn_scale"`` or
      ``"group"``;
    * depth-based pruners – names that contain ``"depth"``.

    Parameters
    ----------
    importance_name : str
        Name of the importance metric used for pruning.

    Returns
    -------
    type[ZeroShotPruner]
        One of :class:`ZeroShotPruner`, :class:`PrunerWithGrad`,
        :class:`PrunerWithReg`, or :class:`PrunerInDepth`, depending on the
        importance name.
    """
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
