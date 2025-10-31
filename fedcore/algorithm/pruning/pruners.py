from copy import deepcopy
from fedot.core.data.data import InputData
import torch
import logging
import traceback
from transformers import Trainer

from fedcore.algorithm.base_compression_model import BaseCompressionModel
from fedot.core.operations.operation_parameters import OperationParameters


from fedcore.algorithm.pruning.hooks import PrunerWithGrad, PrunerWithReg, ZeroShotPruner, PrunerInDepth
from fedcore.algorithm.pruning.pruning_validation import PruningValidator
from fedcore.architecture.computational.devices import default_device
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel, BaseNeuralForecaster
from fedcore.models.network_impl.utils.trainer_factory import create_trainer_from_input_data
from fedcore.models.network_impl.utils.hooks import BaseHook
from fedcore.repository.constant_repository import (
    PRUNERS,
    PRUNING_IMPORTANCE,
)
import torch_pruning as tp
from torch_pruning.pruner.importance import Importance
from typing import Union
from fedcore.tools.registry.model_registry import ModelRegistry



class BasePruner(BaseCompressionModel):
    """Base class for model pruning implementations.

    This class extends :class:`BaseCompressionModel` and configures a pruning
    pipeline based on Torch-Pruning. It is responsible for:

    * mapping user-specified importance names to concrete importance objects;
    * initializing a pruning agent (from :data:`PRUNERS`) on top of the model;
    * setting up validation utilities
      (:class:`PruningValidator`) to compute ignored layers and channel groups;
    * registering pruning hooks
      (:class:`ZeroShotPruner`, :class:`PrunerWithGrad`,
      :class:`PrunerWithReg`, :class:`PrunerInDepth`) in the trainer.

    Parameters
    ----------
    params : dict, optional
        Configuration dictionary. Common keys include:

        * ``"pruner_name"`` – name of the pruner in :data:`PRUNERS`
          (default: ``"meta_pruner"``).
        * ``"importance"`` – name of the importance metric in
          :data:`PRUNING_IMPORTANCE` (default: ``"MagnitudeImportance"``).
        * ``"pruning_ratio"`` – target global pruning ratio (float, default
          ``0.5``).
        * ``"prune_each"`` – epoch interval for pruning hooks (default ``-1``,
          i.e. never triggered unless explicitly set).
        * ``"pruning_iterations"`` – number of iterative pruning steps used
          by the pruner (default ``1``).
        * ``"importance_norm"`` – norm order for importance aggregation
          (stored but not directly used here).
        * ``"importance_reduction"`` – reduction mode for group importance
          (passed as ``group_reduction`` to importance object, default
          ``"mean"``).
        * ``"importance_normalize"`` – normalization strategy for importance
          (passed as ``normalizer`` to importance object, default ``"mean"``).
          For ``importance_name == "lamp"`` this is overridden to ``"lamp"``.
        * Any additional keys supported by :class:`BaseCompressionModel` and
          :class:`BaseNeuralModel` (optimizer, device, etc.).
    """

    DEFAULT_HOOKS: list[type['ZeroShotPruner']] = [PrunerWithGrad, PrunerWithReg, ZeroShotPruner, PrunerInDepth]

    def __init__(self, params: dict = {}):
        super().__init__(params)

        # pruning params
        self.pruner_name = params.get("pruner_name", "meta_pruner")
        self.importance_name = params.get("importance", "magnitude")

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
        """Return a short string representation with the pruner name."""
        return self.pruner_name
    
    def _init_model(self, input_data): #TODO check new logic with torch.load, insert in _init_model_before_model... and delete it!
        self.logger.info('Prepare original model for pruning'.center(80, '='))
        model = input_data.target
        if isinstance(model, str):
            device = default_device()
            loaded = torch.load(model, map_location=device)
            if isinstance(loaded, dict) and "model" in loaded:
                model = loaded["model"]
            else:
                model = loaded
        self.model_before = model
        self.trainer = create_trainer_from_input_data(input_data, self.ft_params)

        if hasattr(self.model_before, 'model'):
            self.trainer.model = self.model_before.model
        else:
            self.trainer.model = self.model_before

        self.model_before.to(default_device())

        self._model_registry = ModelRegistry()
        self._pruning_index += 1

        if self._model_id_before:
            self._model_registry.update_metrics(
                fedcore_id=self._fedcore_id,
                model_id=self._model_id_before,
                metrics={},
                stage="before",
                mode=self.__class__.__name__
            )

        self.model_after = self.model_before

        self.logger.info(f' Initialisation of {self.pruner_name} pruning agent '.center(80, '='))
        self.logger.info(f' Pruning importance - {self.importance_name} '.center(80, '='))
        self.logger.info(f' Pruning ratio - {self.pruning_ratio} '.center(80, '='))
        self.logger.info(f' Pruning importance norm -  {self.importance_norm} '.center(80, '='))
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

    def _init_model_before_model_after_hooks_and_pruner(self, input_data):
        print('Prepare original model for pruning'.center(80, '='))

        self._init_model_before_model_after(input_data)
        self.pruner = self._init_pruner_with_model_after(input_data)

        if (self.pruner != None):
            pruner_hooks = BaseNeuralModel.filter_hooks_by_params(self.params, self.DEFAULT_HOOKS)
            pruner_hooks = [pruner_hook_type(self.pruner, self.pruning_iterations, self.prune_each) for pruner_hook_type in pruner_hooks]
        else:
            pruner_hooks = []

        self._init_trainer_with_model_after(input_data, pruner_hooks)        
        self.model_after.to(default_device())

        print(f' Initialisation of {self.pruner_name} pruning agent '.center(80, '='))
        print(f' Pruning importance - {self.importance_name} '.center(80, '='))
        print(f' Pruning ratio - {self.pruning_ratio} '.center(80, '='))
        print(f' Pruning importance norm -  {self.importance_norm} '.center(80, '='))
        
    def _setup_pruner_validation_params_from_model(self, input_data):
        # list of tensors with dim size n_samples x n_channel x height x width
        batch_generator = (b for b in input_data.val_dataloader)
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
            if input_data.task.task_type.value.__contains__('forecasting') else input_data.num_classes
        self.validator = PruningValidator(model=self.model_after,
                                          output_dim=n_classes, input_dim=input_data.input_dim)
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
            self._init_model_before_model_after_hooks_and_pruner(input_data)
            self.trainer.fit(input_data)

        except Exception as e:
            traceback.print_exc()
            self.model_after = self.model_before
        return self.model_after
    
    def fitNEWFROMREBASE(self, input_data: InputData, finetune: bool = True): #TODO merge logic with fit
        """Run training with pruning hooks and return the pruned model.

        This method prepares the trainer and models via
        :meth:`_prepare_trainer_and_model_to_fit` (which in turn calls
        :meth:`_init_trainer_model_before_model_after_and_incapsulate_hooks`),
        and then starts training via ``self.trainer.fit``.

        If any exception occurs during training, the error is printed with a
        traceback and ``model_after`` is set to ``model_before`` (no pruning
        is applied).

        Parameters
        ----------
        input_data : InputData
            Fedot input data used to train the model and drive pruning hooks.

        Returns
        -------
        Any
            Pruned (or original, in case of error) model instance stored in
            ``self.model_after``.
        """
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
                self.model_after = result_model
                if self._model_id_after:
                    self._model_registry.update_metrics(
                        fedcore_id=self._fedcore_id,
                        model_id=self._model_id_after,
                        metrics={},
                        stage="after",
                        mode=self.__class__.__name__
                    )

                return result_model
        except Exception as e:
            traceback.print_exc()
            self.model_after = self.model_before
        return self.model_after

    def finetune(self, finetune_object, finetune_data):
        validated_finetune_object = self.validator.validate_pruned_layers(finetune_object)
        self.trainer.model = validated_finetune_object
        self.logger.info(f"==============After {self.importance_name} pruning=================")
        params_dict = self.estimate_params(example_batch=self.data_batch_for_calib,
                                           model_before=self.model_before,
                                           model_after=validated_finetune_object)
        self.logger.info("==============Finetune pruned model=================")
        self.model_after = self.trainer.fit(finetune_data)
        return self.model_after

    def predict_for_fit(self, input_data: InputData, output_mode: str = 'fedcore'):
        """Return the model object (before or after pruning) after `fit`.

        This helper is used in FedCore pipelines that expect a model object
        instead of predictions right after training.

        Parameters
        ----------
        input_data : InputData
            Input data (unused, provided for API compatibility).
        output_mode : str, optional
            If ``"fedcore"`` (default), return the pruned model
            ``self.model_after``; otherwise return the original model
            ``self.model_before``.

        Returns
        -------
        Any
            The selected model object.
        """
        return self.model_after if output_mode == 'fedcore' else self.model_before

    def predict(self, input_data: InputData, output_mode: str = 'fedcore'):
        """Run prediction using either the pruned or original model.

        Parameters
        ----------
        input_data : InputData
            Data for inference.
        output_mode : str, optional
            If ``"fedcore"`` (default), use the pruned model
            ``self.model_after``; otherwise use the original model
            ``self.model_before``.

        Returns
        -------
        Any
            Result of :meth:`BaseNeuralModel.predict` for the selected model.
        """
        if output_mode == 'fedcore':
            self.trainer.model = self.model_after
        else:
            self.trainer.model = self.model_before
        return self.trainer.predict(input_data, output_mode)
