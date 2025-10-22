"""
High-level pruning orchestration for FedCore.

This module defines :class:`BasePruner`, a compression wrapper that:
  - prepares the model and trainer for pruning/finetuning,
  - instantiates a pruning backend (from ``PRUNERS``) and an importance
    criterion (from ``PRUNING_IMPORTANCE``),
  - runs hook-driven pruning flows (see ``fedcore.algorithm.pruning.hooks``),
  - validates pruned layers and optionally fine-tunes the result.

Notes
-----
* The original (pre-pruning) model is kept in ``self.model_before``.
* The working copy being pruned/fine-tuned is ``self.model_after``.
* Hook groups are provided by ``PruningHooks`` and attached at initialization.
"""

from copy import deepcopy
from itertools import chain
from fedot.core.data.data import InputData
from torch import nn, optim
import traceback

from fedcore.algorithm.base_compression_model import BaseCompressionModel
import torch_pruning as tp
from typing import Optional
from fedot.core.operations.operation_parameters import OperationParameters

from fedcore.algorithm.pruning.hooks import PruningHooks
from fedcore.algorithm.pruning.pruning_validation import PruningValidator
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel, BaseNeuralForecaster
from fedcore.models.network_impl.hooks import BaseHook
from fedcore.repository.constanst_repository import (
    PRUNERS,
    PRUNING_IMPORTANCE, TorchLossesConstant
)


class BasePruner(BaseCompressionModel):
    """Orchestrates model pruning and optional finetuning.

    The class builds a pruning pipeline around a copy of the input model.
    Importance criteria and pruner backends are configured via repository
    registries (``PRUNING_IMPORTANCE`` and ``PRUNERS``). Execution of concrete
    strategies is delegated to hook implementations (see ``PruningHooks``).

    Attributes
    ----------
    epochs : int
        Epochs for finetuning (if enabled).
    ft_params : dict
        Parameters passed into the internal trainer (loss/optimizer, etc.).
    optimizer : type
        Optimizer class for finetuning (default: ``optim.Adam``).
    learning_rate : float
        Learning rate used by the optimizer for gradients accumulation and FT.
    pruner_name : str
        Key in ``PRUNERS`` registry that selects pruning backend.
    importance_name : str
        Key in ``PRUNING_IMPORTANCE`` registry to choose importance criterion.
    pruning_ratio : float
        Global target sparsity ratio (0..1).
    pruning_iterations : int
        Number of iterative pruning steps.
    importance_norm : int | float
        Norm used by some importance implementations.
    importance_reduction : str
        Group reduction method ('mean', 'sum', ...).
    importance_normalize : str
        Normalization strategy ('mean', 'lamp', ...).
    importance : Any
        Prepared importance object/function used by the pruner.
    _hooks : list
        List of hook groups to attach (here: [``PruningHooks``]).
    model_before : nn.Module
        Original model (kept intact).
    model_after : nn.Module
        Working copy that is pruned and fine-tuned.
    data_batch_for_calib : torch.Tensor
        A calibration batch used for validator/shape checks.
    validator : PruningValidator
        Verifies layer compatibility and builds groups/ignore lists.
    ignored_layers : list
        Layers excluded from pruning.
    channel_groups : dict
        Grouping info for channel-wise pruning.
    optimizer_for_grad : torch.optim.Optimizer
        Optimizer used during regularization/gradient accumulation in hooks.
    pruner : object | None
        Concrete pruning backend instance (may be None for special flows).
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        """Initialize the pruning pipeline from operation parameters.

        Parameters
        ----------
        params : Optional[OperationParameters], default={}
            Expected keys include:
              - 'epochs' (int), 'finetune_params' (dict),
              - 'optimizer' (torch.optim class), 'lr' (float),
              - 'pruner_name' (str), 'importance' (str),
              - 'pruning_ratio' (float), 'pruning_iterations' (int),
              - 'importance_norm' (int/float),
              - 'importance_reduction' (str),
              - 'importance_normalize' (str).
        """
        super().__init__(params)
        # finetune params
        self.epochs = params.get("epochs", 5)
        self.ft_params = params.get("finetune_params", dict())
        self.optimizer = params.get("optimizer", optim.Adam)
        self.learning_rate = params.get("lr", 0.0001)

        # pruning gradients params
        finetune_params = params.get('finetune_params', dict())
        criterion_for_grad = TorchLossesConstant[finetune_params.get("criterion", 'cross_entropy')]

        self.ft_params.update({'criterion_for_grad': criterion_for_grad.value()})
        self.ft_params.update({'lr_for_grad': params.get("lr", 0.0001)})

        # pruning params
        self.pruner_name = params.get("pruner_name", "meta_pruner")
        self.importance_name = params.get("importance", "MagnitudeImportance")

        # pruning hyperparams
        self.pruning_ratio = params.get("pruning_ratio", 0.5)
        self.pruning_iterations = params.get("pruning_iterations", 1)
        self.importance_norm = params.get("importance_norm", 1)
        self.importance_reduction = params.get("importance_reduction", "mean")
        self.importance_normalize = params.get("importance_normalize", "mean")
        self.importance = PRUNING_IMPORTANCE[self.importance_name]
        if self.importance_name == 'lamp':
            self.importance_normalize = 'lamp'
        # importance criterion for parameter selections
        if self.importance_name == 'random':
            self.importance = self.importance()
        elif isinstance(self.importance, str):
            self.importance = self.importance
        else:
            self.importance = self.importance(group_reduction=self.importance_reduction,
                                              normalizer=self.importance_normalize)

        self._hooks = [PruningHooks]
        self._init_empty_object()

    def __repr__(self):
        """Return human-readable name of the configured pruner backend."""
        return self.pruner_name

    def _init_empty_object(self):
        """Initialize history containers and hook lists (internal helper)."""
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
        # add hooks
        self._on_epoch_end = []
        self._on_epoch_start = []

    def _init_hooks(self):
        """Instantiate and register pruning hooks according to ``_hooks``."""
        for hook_elem in chain(*self._hooks):
            hook: BaseHook = hook_elem.value
            hook = hook(self.ft_params, self.model_after)
            if hook._hook_place >= 0:
                self._on_epoch_end.append(hook)
            else:
                self._on_epoch_start.append(hook)

    def _init_model(self, input_data):
        """Prepare original model, trainer, and pruning backend.

        Steps
        -----
        1) Read the model from ``input_data.target`` (or its ``.model`` attr).
        2) Select the appropriate trainer (classification vs forecasting).
        3) Move base model to default device and deep-copy to ``model_after``.
        4) Choose pruning backend from ``PRUNERS`` by ``importance_name`` /
           ``pruner_name`` (some importance types map to special pruners).
        5) Run basic validation and build ignore lists / channel groups.
        6) Create an optimizer for gradient accumulation used by hooks.

        Parameters
        ----------
        input_data : InputData
            Data container with features/targets/dataloaders and task metadata.
        """
        print('Prepare original model for pruning'.center(80, '='))
        self.model_before = input_data.target
        if input_data.task.task_type.value.__contains__('forecasting'):
            self.trainer = BaseNeuralForecaster(self.ft_params)
        else:
            self.trainer = BaseNeuralModel(self.ft_params)
        if hasattr(self.model_before, 'model'):
            self.trainer.model = self.model_before.model
        self.model_before.to(default_device())
        self.model_after = deepcopy(self.model_before)
        print(f' Initialisation of {self.pruner_name} pruning agent '.center(80, '='))
        print(f' Pruning importance - {self.importance_name} '.center(80, '='))
        print(f' Pruning ratio - {self.pruning_ratio} '.center(80, '='))
        print(f' Pruning importance norm -  {self.importance_norm} '.center(80, '='))
        # Pruner initialization
        if self.importance_name.__contains__('activation'):
            self.pruner = None
        elif self.importance_name.__contains__('group'):
            self.pruner = PRUNERS["group_norm_pruner"]
        elif self.importance_name in ['bn_scale']:
            self.pruner = PRUNERS["batch_norm_pruner"]
        elif not self.importance_name in ['random', 'lamp', 'magnitude']:
            self.pruner = PRUNERS["growing_reg_pruner"]
        else:
            self.pruner = PRUNERS[self.pruner_name]
        self._check_before_prune(input_data)
        self.optimizer_for_grad = optim.Adam(self.model_after.parameters(),
                                             lr=self.ft_params['lr_for_grad'])
        self.ft_params['optimizer_for_grad_acc'] = self.optimizer_for_grad

    def _check_before_prune(self, input_data):
        """Build calibration batch and run structural validation/grouping.

        Parameters
        ----------
        input_data : InputData
            Source of dataloaders and task metadata.

        Side Effects
        ------------
        - Sets ``self.data_batch_for_calib`` used for param estimation.
        - Creates ``self.validator``, ``self.ignored_layers``, ``self.channel_groups``.
        """
        # list of tensors with dim size n_samples x n_channel x height x width
        batch_generator = (b for b in input_data.features.val_dataloader)
        # take first batch
        batch_list = next(batch_generator)
        self.data_batch_for_calib = batch_list[0].to(default_device())
        n_classes = input_data.task.task_params['forecast_length'] \
            if input_data.task.task_type.value.__contains__('forecasting') else input_data.features.num_classes
        self.validator = PruningValidator(model=self.model_after,
                                          output_dim=n_classes, input_dim=input_data.features.input_dim)
        self.ignored_layers = self.validator.filter_ignored_layers(self.model_after,
                                                                   str(self.model_after.__class__))
        self.channel_groups = self.validator.validate_channel_groups()

    def fit(self, input_data: InputData, finetune: bool = True):
        """Run pruning hooks and optionally fine-tune the pruned model.

        Parameters
        ----------
        input_data : InputData
            Training/validation data and task description.
        finetune : bool, default=True
            Whether to fine-tune the model after pruning.

        Returns
        -------
        nn.Module
            The pruned (and possibly fine-tuned) model.
        """
        try:
            self._init_model(input_data)
            self._init_hooks()
            if self.pruner is not None:  # if we use torch_pruning as backbone
                self.pruner = self.pruner(
                    self.model_after,
                    self.data_batch_for_calib,
                    # global_pruning=False,
                    importance=self.importance,
                    iterative_steps=self.pruning_iterations,
                    pruning_ratio=self.pruning_ratio,
                    ignored_layers=self.ignored_layers,
                    channel_groups=self.channel_groups,
                    round_to=None,
                    unwrapped_parameters=None,
                )
            self.pruner_objects = {'input_data': input_data,
                                'pruning_iterations': self.pruning_iterations,
                                'model_before_pruning': self.model_before,
                                'optimizer_for_grad_acc': self.optimizer_for_grad,
                                'pruner_cls': self.pruner}
            for hook in self._on_epoch_end:
                hook(importance=self.importance, pruner_objects=self.pruner_objects)
            if finetune:
                return self.finetune(finetune_object=self.model_after, finetune_data=input_data)
        except Exception as e:
            traceback.print_exc()
            self.model_after = self.model_before
        return self.model_after

    def finetune(self, finetune_object, finetune_data):
        """Validate pruned layers, estimate params, and fine-tune.

        Parameters
        ----------
        finetune_object : nn.Module
            Model after pruning to be fine-tuned.
        finetune_data : InputData
            Data for fine-tuning.

        Returns
        -------
        nn.Module
            Fine-tuned model.
        """
        validated_finetune_object = self.validator.validate_pruned_layers(finetune_object)
        self.trainer.model = validated_finetune_object
        print(f"==============After {self.importance_name} pruning=================")
        params_dict = self.estimate_params(example_batch=self.data_batch_for_calib,
                                           model_before=self.model_before,
                                           model_after=validated_finetune_object)
        print("==============Finetune pruned model=================")
        self.model_after = self.trainer.fit(finetune_data)
        return self.model_after

    def predict_for_fit(self, input_data: InputData, output_mode: str = 'fedcore'):
        """Return the model object used for fit-time predictions (compat shim)."""
        return self.model_after if output_mode == 'fedcore' else self.model_before

    def predict(self, input_data: InputData, output_mode: str = 'fedcore'):
        """Predict with the chosen (pruned or original) model.

        Parameters
        ----------
        input_data : InputData
            Input for the model's forward pass.
        output_mode : {'fedcore', ...}
            If 'fedcore' uses ``model_after``; otherwise uses ``model_before``.

        Returns
        -------
        Any
            Output of the underlying trainer's ``predict``.
        """
        if output_mode == 'fedcore':
            self.trainer.model = self.model_after
        else:
            self.trainer.model = self.model_before
        return self.trainer.predict(input_data, output_mode)
