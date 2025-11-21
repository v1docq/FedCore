"""Base pruning model abstraction for FedCore.

This module defines :class:`BasePruner`, a compression model that wraps
a :class:`BaseNeuralModel` and configures structured pruning using
`torch-pruning`. It:

* selects importance criteria and pruner types based on configuration;
* initializes a pruning agent on top of the trained model;
* injects appropriate pruning hooks (zero-shot, gradient-based, reg-based,
  depth-based) into the training loop;
* exposes a unified `fit/predict` interface compatible with FedCore
  compression models.
"""

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
        """Instantiate or resolve an importance object from configuration.

        The mapping is based on :data:`PRUNING_IMPORTANCE` and the current
        ``importance_name``:

        * if the importance type is a string, it is returned as-is
          (e.g. for special cases handled by the pruner itself);
        * if ``importance_name == "random"``, the importance class is
          instantiated without arguments;
        * otherwise, a :class:`Importance`-like object is instantiated with
          ``group_reduction`` and ``normalizer`` taken from the corresponding
          config fields.

        Returns
        -------
        Union[str, Importance]
            Either a string identifier or an instantiated importance object
            compatible with Torch-Pruning pruners.
        """
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

    def _init_trainer_model_before_model_after_and_incapsulate_hooks(self, input_data):
        """Initialize models, pruner, and attach pruning hooks to the trainer.

        This method orchestrates the setup of pruning:

        1. Initializes ``model_before`` and ``model_after`` using
           :meth:`BaseCompressionModel._init_model_before_model_after`.
        2. Creates a pruning agent bound to ``model_after`` via
           :meth:`_init_pruner_with_model_after`.
        3. If a pruner is available, filters and instantiates suitable pruning
           hooks from :data:`DEFAULT_HOOKS` using
           :meth:`BaseNeuralModel.filter_hooks_by_params`.
        4. Initializes the trainer with ``model_after`` and the created hooks
           via :meth:`BaseCompressionModel._init_trainer_with_model_after`.
        5. Logs basic pruning configuration (importance, ratio, norms).

        Parameters
        ----------
        input_data :
            Object describing data/model configuration for the trainer
            (typically a Fedot experimenter or dataset wrapper).
        """
        print('Prepare original model for pruning'.center(80, '='))

        super()._init_model_before_model_after(input_data)
        self.pruner = self._init_pruner_with_model_after(input_data)

        if (self.pruner is not None):
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
        """Prepare calibration batch and pruning validation metadata.

        This method uses :class:`PruningValidator` to derive structures
        required by Torch-Pruning:

        * selects a calibration batch from ``input_data.features.val_dataloader``;
        * infers the number of output classes (or forecast length for
          forecasting tasks);
        * creates a :class:`PruningValidator` object bound to ``model_after``;
        * computes a list of layers to ignore during pruning;
        * computes valid channel groups.

        Parameters
        ----------
        input_data : InputData
            Fedot input data object that provides validation dataloader,
            task metadata and input dimensionality.
        """
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
        """Instantiate a Torch-Pruning pruner bound to ``model_after``.

        The choice of pruner type is based on ``importance_name``:

        * if ``"activation"`` is found in the importance name, no pruner is
          created here and ``None`` is returned (activation-based pruners
          are handled by other mechanisms);
        * if ``"group"`` is in the name – use ``"group_norm_pruner"`` from
          :data:`PRUNERS`;
        * if importance name is ``"bn_scale"`` – use ``"batch_norm_pruner"``;
        * if importance name is none of ``["random", "lamp", "magnitude"]`` –
          use ``"growing_reg_pruner"``;
        * otherwise – fall back to the pruner specified by ``self.pruner_name``.

        After choosing the type, this method calls
        :meth:`_setup_pruner_validation_params_from_model` and creates a
        pruner instance with:

        * ``model_after`` as the target model,
        * calibration batch,
        * chosen importance object,
        * target pruning ratio,
        * ignored layers and channel groups.

        Parameters
        ----------
        input_data : InputData
            Fedot input data object with features, task info and val loader.

        Returns
        -------
        Any or None
            Instantiated pruner or ``None`` if no pruner should be used.
        """
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
            super()._prepare_trainer_and_model_to_fit(input_data)
            self.trainer.fit(input_data)

        except Exception as e:
            traceback.print_exc()
            self.model_after = self.model_before
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
