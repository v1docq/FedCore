from fedot.core.data.data import InputData
import traceback

from fedcore.algorithm.base_compression_model import BaseCompressionModel

from fedcore.algorithm.pruning.hooks import PrunerWithGrad, PrunerWithReg, ZeroShotPruner, PrunerInDepth
from fedcore.algorithm.pruning.pruning_validation import PruningValidator
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
from fedcore.repository.constanst_repository import (
    PRUNERS,
    PRUNING_IMPORTANCE,
)
from torch_pruning.pruner.importance import Importance
from typing import Union



class BasePruner(BaseCompressionModel):
    """Class responsible for Pruning model implementation.
    Example:
    """

    DEFAULT_HOOKS: list[type['ZeroShotPruner']] = [PrunerWithGrad, PrunerWithReg, ZeroShotPruner, PrunerInDepth]

    def __init__(self, params: dict = {}): 
        super().__init__(params)

        # pruning params
        self.pruner_name = params.get("pruner_name", "meta_pruner")
        self.importance_name = params.get("importance", "MagnitudeImportance")

        # pruning hyperparams
        self.pruning_ratio = params.get("pruning_ratio", 0.5)
        self.prune_each = params.get("prune_each", -1)
        self.pruning_iterations = params.get("pruning_iterations", 1)
        self.importance_norm = params.get("importance_norm", 1)
        self.importance_reduction = params.get("importance_reduction", "mean")
        self.importance_normalize = params.get("importance_normalize", "mean")
        if self.importance_name == 'lamp':
            self.importance_normalize = 'lamp'
        self.importance = self._map_importance_name()

    def _map_importance_name(self) -> Union[str, Importance]:
        importance_type = PRUNING_IMPORTANCE[self.importance_name]
        importance = importance_type
        if self.importance_name == 'random':
            importance = importance_type()
        elif not isinstance(importance_type, str):
            importance = importance_type(group_reduction=self.importance_reduction,
                                              normalizer=self.importance_normalize)
        return importance

    def __repr__(self):
        return self.pruner_name

    def _init_trainer_model_before_model_after_and_incapsulate_hooks(self, input_data):
        print('Prepare original model for pruning'.center(80, '='))

        super()._init_model_before_model_after(input_data)
        self.pruner = self._init_pruner_with_model_after(input_data)

        if (self.pruner != None):
            pruner_hooks = BaseNeuralModel.filter_hooks_by_params(self.params, self.DEFAULT_HOOKS)
            pruner_hooks = [pruner_hook_type(self.pruner, self.pruning_iterations, self.prune_each) for pruner_hook_type in pruner_hooks]
        else:
            pruner_hooks = []

        super()._init_trainer_with_model_after(input_data, pruner_hooks)

        print(f' Initialisation of {self.pruner_name} pruning agent '.center(80, '='))
        print(f' Pruning importance - {self.importance_name} '.center(80, '='))
        print(f' Pruning ratio - {self.pruning_ratio} '.center(80, '='))
        print(f' Pruning importance norm -  {self.importance_norm} '.center(80, '='))
        
    def _setup_pruner_validation_params_from_model(self, input_data):
        # list of tensors with dim size n_samples x n_channel x height x width
        batch_generator = (b for b in input_data.features.val_dataloader)
        # take first batch
        batch_list = next(batch_generator)
        self.data_batch_for_calib = batch_list[0].to(self.device)
        n_classes = input_data.task.task_params['forecast_length'] \
            if input_data.task.task_type.value.__contains__('forecasting') else input_data.features.num_classes
        self.validator = PruningValidator(model=self.model_after,
                                          output_dim=n_classes, input_dim=input_data.features.input_dim)
        self.ignored_layers = self.validator.filter_ignored_layers(self.model_after,
                                                                   str(self.model_after.__class__))
        self.channel_groups = self.validator.validate_channel_groups()

    def _init_pruner_with_model_after(self, input_data):
        pruner = None

        if 'activation' in self.importance_name:
            return pruner
        
        if 'group' in self.importance_name:
            pruner_type = PRUNERS["group_norm_pruner"]
        elif self.importance_name in ['bn_scale']:
            pruner_type = PRUNERS["batch_norm_pruner"]
        elif not self.importance_name in ['random', 'lamp', 'magnitude']:
            pruner_type = PRUNERS["growing_reg_pruner"]
        else:
            pruner_type = PRUNERS[self.pruner_name]

        self._setup_pruner_validation_params_from_model(input_data)
        pruner = pruner_type(
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

        return pruner

    def fit(self, input_data: InputData):
        try:
            super()._prepare_trainer_and_model_to_fit(input_data)
            self.trainer.fit(input_data)

        except Exception as e:
            traceback.print_exc()
            self.model_after = self.model_before
        return self.model_after

    def predict_for_fit(self, input_data: InputData, output_mode: str = 'fedcore'):
        return self.model_after if output_mode == 'fedcore' else self.model_before

    def predict(self, input_data: InputData, output_mode: str = 'fedcore'):
        if output_mode == 'fedcore':
            self.trainer.model = self.model_after
        else:
            self.trainer.model = self.model_before
        return self.trainer.predict(input_data, output_mode)
