from copy import deepcopy
from itertools import chain
from fedot.core.data.data import InputData
from torch import nn, optim

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
    PRUNING_IMPORTANCE
)


class BasePruner(BaseCompressionModel):
    """Class responsible for Pruning model implementation.
    Example:
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)
        # finetune params
        self.epochs = params.get("epochs", 5)
        self.ft_params = params.get("finetune_params", None)
        self.optimizer = params.get("optimizer", optim.Adam)
        self.learning_rate = params.get("lr", 0.0001)

        # pruning gradients params
        self.ft_params['criterion_for_grad'] = params.get("loss", nn.CrossEntropyLoss())
        self.ft_params['lr_for_grad'] = params.get("lr", 0.0001)

        # pruning params
        self.pruner_name = params.get("pruner_name", "meta_pruner")
        self.importance_name = params.get("importance", "MagnitudeImportance")

        # pruning hyperparams
        self.pruning_ratio = params.get("pruning_ratio", 0.5)
        self.pruning_iterations = params.get("pruning_iterations", 2)
        self.importance_norm = params.get("importance_norm", 1)
        self.importance_reduction = params.get("importance_reduction", "mean")
        self.importance_normalize = params.get("importance_normalize", "mean")

        # importance criterion for parameter selections
        self.importance = PRUNING_IMPORTANCE[self.importance_name](
            group_reduction=self.importance_reduction,
            normalizer=self.importance_normalize,
        )

        self._hooks = [PruningHooks]
        self._init_empty_object()

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
            hook = hook(self.ft_params, self.model_after_pruning)
            if hook._hook_place == 'post':
                self._on_epoch_end.append(hook)
            else:
                self._on_epoch_start.append(hook)

    def _init_model(self, input_data):
        print('Prepare original model for pruning'.center(80, '='))
        self.model_before_pruning = input_data.target
        if input_data.task.task_type.value.__contains__('forecasting'):
            self.trainer = BaseNeuralForecaster(self.ft_params)
        else:
            self.trainer = BaseNeuralModel(self.ft_params)
        if hasattr(self.model_before_pruning, 'model'):
            self.trainer.model = self.model_before_pruning.model
        self.model_before_pruning.to(default_device())
        self.model_after_pruning = deepcopy(self.model_before_pruning)
        print(f' Initialisation of {self.pruner_name} pruning agent '.center(80, '='))
        print(f' Pruning importance - {self.importance_name} '.center(80, '='))
        print(f' Pruning ratio - {self.pruning_ratio} '.center(80, '='))
        print(f' Pruning importance norm -  {self.importance_norm} '.center(80, '='))
        # Pruner initialization
        self.pruner = PRUNERS[self.pruner_name]
        self._check_before_prune(input_data)
        self.optimizer_for_grad = optim.Adam(self.model_after_pruning.parameters(),
                                             lr=self.ft_params['lr_for_grad'])

    def _check_before_prune(self, input_data):
        # list of tensors with dim size n_samples x n_channel x height x width
        batch_generator = (b for b in input_data.features.val_dataloader)
        # take first batch
        batch_list = next(batch_generator)
        self.data_batch_for_calib = batch_list[0].to(default_device())
        n_classes = input_data.task.task_params['forecast_length'] \
            if input_data.task.task_type.value.__contains__('forecasting') else input_data.features.num_classes
        self.validator = PruningValidator(model=self.model_after_pruning,
                                          output_dim=n_classes, input_dim=input_data.features.input_dim)
        self.ignored_layers = self.validator.filter_ignored_layers(self.model_after_pruning,
                                                                   str(self.model_after_pruning.__class__))
        self.channel_groups = self.validator.validate_channel_groups()

    def fit(self, input_data: InputData):
        self._init_model(input_data)
        self._init_hooks()
        self.pruner = self.pruner(
            self.model_after_pruning,
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
                               'model_before_pruning': self.model_before_pruning,
                               'optimizer_for_grad_acc': self.optimizer_for_grad,
                               'pruner_cls': self.pruner}
        for hook in self._on_epoch_end:
            hook(importance=self.importance, pruner_objects=self.pruner_objects)
        return self.finetune(finetune_object=self.model_after_pruning, finetune_data=input_data)

    def finetune(self, finetune_object, finetune_data):
        base_macs, base_nparams = tp.utils.count_ops_and_params(self.model_before_pruning, self.data_batch_for_calib)
        print("==============After pruning=================")
        finetune_object = self.validator.validate_pruned_layers(finetune_object)
        macs, nparams = tp.utils.count_ops_and_params(finetune_object, self.data_batch_for_calib)
        print("Params: %.6f M => %.6f M" % (base_nparams / 1e6, nparams / 1e6))
        print("MACs: %.6f G => %.6f G" % (base_macs / 1e9, macs / 1e9))
        print("==============Finetune pruned model=================")
        self.trainer.model = finetune_object
        self.model_after_pruning = self.trainer.fit(finetune_data)
        return self.model_after_pruning

    def predict_for_fit(self, input_data: InputData, output_mode: str = 'fedcore'):
        return self.model_after_pruning if output_mode == 'fedcore' else self.model_before_pruning

    def predict(self, input_data: InputData, output_mode: str = 'fedcore'):
        if output_mode == 'fedcore':
            self.trainer.model = self.model_after_pruning
        else:
            self.trainer.model = self.model_before_pruning
        return self.trainer.predict(input_data, output_mode)
