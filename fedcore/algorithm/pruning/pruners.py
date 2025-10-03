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
from fedcore.tools.model_registry import ModelRegistry



class BasePruner(BaseCompressionModel):
    """Class responsible for Pruning model implementation.
    Example:
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)
        # finetune params
        self.epochs = params.get("epochs", 5)
        self.ft_params = params.get("finetune_params", dict())
        # self.ft_params = params.get("finetune_params", None)
        self.optimizer = params.get("optimizer", optim.Adam)
        self.learning_rate = params.get("lr", 0.0001)

        # pruning gradients params
        finetune_params = params.get('finetune_params', dict())
        # finetune_params = params.get('finetune_params', None)
        criterion_for_grad = TorchLossesConstant[finetune_params.get("criterion", 'cross_entropy')]

        self.ft_params.update({'criterion_for_grad': criterion_for_grad.value()})
        self.ft_params.update({'lr_for_grad': params.get("lr", 0.0001)})

        # self.ft_params['criterion_for_grad'] = criterion_for_grad.value()
        # self.ft_params['lr_for_grad'] = params.get("lr", 0.0001)

        # pruning params
        self.pruner_name = params.get("pruner_name", "meta_pruner")
        self.importance_name = params.get("importance", "magnitude")

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
        self._pruning_index = 0

    def __repr__(self):
        return self.pruner_name

    def _init_empty_object(self):
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
        # add hooks
        self._on_epoch_end = []
        self._on_epoch_start = []

    def _init_hooks(self):
        for hook_elem in chain(*self._hooks):
            hook: BaseHook = hook_elem.value
            hook = hook(self.ft_params, self.model_after)
            if hook._hook_place >= 0:
                self._on_epoch_end.append(hook)
            else:
                self._on_epoch_start.append(hook)

    def _init_model(self, input_data):
        print('Prepare original model for pruning'.center(80, '='))
        self.model_before = input_data.target
        if input_data.task.task_type.value.__contains__('forecasting'):
            self.trainer = BaseNeuralForecaster(self.ft_params)
        else:
            self.trainer = BaseNeuralModel(self.ft_params)
        if hasattr(self.model_before, 'model'):
            self.trainer.model = self.model_before.model
        self.model_before.to(default_device())
        self.model_after = self.model_before
        registry = ModelRegistry().get_instance(model=self.model_before)
        self._model_registry = registry
        self._pruning_index += 1
        metrics_before = {
            "stage": f"pruning_{self._pruning_index}",
            "is_processed": False,
        }
        ModelRegistry.register_changes(metrics=metrics_before)
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
        # list of tensors with dim size n_samples x n_channel x height x width
        batch_generator = (b for b in input_data.features.val_dataloader)
        # take first batch
        batch_dict = next(batch_generator)
        # Handle dict-like batches (e.g., LLM) by picking the first value or a specific key
        if isinstance(batch_dict, dict):
            if 'input_ids' in batch_dict:
                self.data_batch_for_calib = batch_dict['input_ids'].to(default_device())
            else:
                self.data_batch_for_calib = next(iter(batch_dict.values())).to(default_device())
        else:
            # legacy: assume batch is a list/tuple and pick first element
            self.data_batch_for_calib = batch_dict[0].to(default_device())
        n_classes = input_data.task.task_params['forecast_length'] \
            if input_data.task.task_type.value.__contains__('forecasting') else input_data.features.num_classes
        self.validator = PruningValidator(model=self.model_after,
                                          output_dim=n_classes, input_dim=input_data.features.input_dim)
        self.ignored_layers = self.validator.filter_ignored_layers(self.model_after,
                                                                   str(self.model_after.__class__))
        self.channel_groups = self.validator.validate_channel_groups()

    def fit(self, input_data: InputData, finetune: bool = True):
        try:
            self._init_model(input_data)
            self._init_hooks()
            if self.pruner is not None: # in case if we dont use torch_pruning as backbone
                self.pruner = self.pruner(
                    self.model_after,
                    self.data_batch_for_calib,
                    # global_pruning=False,  # If False, a uniform ratio will be assigned to different layers.
                    importance=self.importance,  # importance criterion for parameter selection
                    iterative_steps=self.pruning_iterations,  # the number of iterations to achieve target ratio
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
                result_model = self.finetune(finetune_object=self.model_after, finetune_data=input_data)
                # Record post-pruning state in registry
                metrics_after = {
                    "stage": f"pruning_{self._pruning_index}",
                    "is_processed": True,
                }
                ModelRegistry.register_changes(metrics=metrics_after)
                return result_model
        except Exception as e:
            traceback.print_exc()
            self.model_after = self.model_before
        return self.model_after

    def finetune(self, finetune_object, finetune_data):
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
        return self.model_after if output_mode == 'fedcore' else self.model_before

    def predict(self, input_data: InputData, output_mode: str = 'fedcore'):
        if output_mode == 'fedcore':
            self.trainer.model = self.model_after
        else:
            self.trainer.model = self.model_before
        return self.trainer.predict(input_data, output_mode)
