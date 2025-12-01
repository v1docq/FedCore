"""Configuration templates for FedCore API and training pipeline.

This module defines a collection of typed configuration templates built on
top of :mod:`dataclasses`. They are used to describe:

* device and edge deployment settings;
* compute/distributed environment parameters;
* AutoML/FEDOT settings;
* neural model architecture and training hyperparameters;
* compression strategies (low-rank, pruning, quantization).

Key ideas
---------
* :class:`ConfigTemplate` provides a base class with type-checked fields and
  utility methods for nested access, validation and conversion to ``dict``.
* :class:`ExtendableConfigTemplate` allows dynamic attributes in addition to
  statically declared slots.
* Specific templates (e.g. :class:`NeuralModelConfigTemplate`,
  :class:`LowRankTemplate`, :class:`PruningTemplate`,
  :class:`QuantTemplate`) extend these base classes with domain-specific
  parameters that are later consumed by FedCore components.
"""

from dataclasses import dataclass
from enum import Enum
from functools import reduce
from inspect import signature, isclass
from numbers import Number
from pathlib import Path
from typing import (
    get_origin, get_args,   
    Any, Callable, Dict, Iterable, List, Literal, Optional, Union, 
)
import logging

from torch.ao.quantization.utils import _normalize_kwargs
from torch.nn import Module

from fedcore.repository.constant_repository import (
    FedotTaskEnum,
    Schedulers, 
    Optimizers, 
    # PEFTStrategies,
    SLRStrategiesEnum,
    TaskTypesEnum,
    TorchLossesConstant,
)
# Avoid importing NLP-specific templates here to prevent circular imports

__all__ = [
    'ConfigTemplate',
    'DeviceConfigTemplate',
    'EdgeConfigTemplate',
    'AutoMLConfigTemplate',
    'TrainingTemplate',
    'LearningConfigTemplate',
    'LowRankTemplate',
    'QuantizationTemplate',
    'FedotConfigTemplate',
    'PruningTemplate',
    'APIConfigTemplate',
    'get_nested',
    'LookUp',
]


def get_nested(root: object, k: str):
    """Resolve a dotted path to an attribute and return parent + last key.

    This helper allows convenient nested access/update of configuration
    objects using dotted keys like ``"trainer.optimizer.lr"``.

    Parameters
    ----------
    root : object
        Root configuration object.
    k : str
        Dotted key in the form ``"attr1.attr2. ... .attrN"``.

    Returns
    -------
    tuple[object, str]
        A pair ``(parent, last)`` where ``parent`` is the object containing
        the last attribute and ``last`` is the attribute name itself.

    """
    *path, last = k.split(".")
    return reduce(getattr, path, root), last


class MisconfigurationError(BaseException):
    """Aggregated configuration validation error.

    Instances of this error contain a list of underlying exceptions
    (typically :class:`TypeError` or :class:`ValueError`) raised during
    config field validation. The string representation concatenates all
    messages line by line.
    """

    def __init__(self, exs, *args):
        super().__init__(*args)
        self.exs = exs

    def __repr__(self):
        return '\n'.join([f'\t{str(x)}' for x in self.exs])

    def __str__(self):
        return self.__repr__()


@dataclass
class LookUp:
    """Marker wrapper for values inherited from a parent config.

    A field wrapped in :class:`LookUp` signals that its value should be taken
    from a higher-level (parent) configuration if not explicitly specified.

    Attributes
    ----------
    value : Any
        Default or placeholder value to be used when resolving from parent.
    """

    value: Any


@dataclass
class ConfigTemplate:
    """Base template for strongly typed configuration sections.

    This class provides:

    * a type-checking :meth:`check` method that validates values against
      type annotations (including :class:`Enum` and :data:`Literal`);
    * a custom ``__new__`` that returns ``(cls, normalized_kwargs)`` instead
      of allocating an instance, which is useful for further processing of
      raw parameters;
    * helper methods for nested access, updates and conversion to ``dict``.

    Notes
    -----
    Actual config instances are typically created in a higher-level factory
    that consumes the ``(cls, kwargs)`` pair returned by ``__new__`` and
    performs additional wiring (e.g., inheritance from parent configs).
    """

    __slots__ = tuple()

    @classmethod
    def get_default_name(cls):
        """Return a human-friendly name derived from the template class.

        By default this strips the ``"Template"`` suffix from the class name.
        """
        name = cls.__name__.split(".")[-1]
        if name.endswith("Template"):
            name = name[:-8]
        return name

    @classmethod
    def get_annotation(cls, key):
        """Return the type annotation for a given field name.

        Parameters
        ----------
        key : str
            Name of the field defined in ``__init__``.

        Returns
        -------
        Any
            Annotation object for the field (can be a type, :class:`Enum`,
            :data:`Union`, :data:`Literal`, etc.).
        """
        obj = cls if ConfigTemplate in cls.__bases__ else cls.__bases__[0]
        return signature(obj.__init__).parameters[key].annotation

    @classmethod
    def check(cls, key, val):
        """Validate value type for the given field name.

        The validation rules respect:

        * plain Python/typing types (``int``, ``float``, ``Callable``, etc.);
        * :class:`Enum` subclasses (value must be a valid member name);
        * :data:`Literal` annotations;
        * ``Union[...]`` – value is valid if it satisfies at least one
          of the union options.

        Parameters
        ----------
        key : str
            Field name to validate.
        val : Any
            Value to be checked.

        Raises
        ------
        MisconfigurationError
            If the value does not match any of the allowed types for the
            field.
        """
        # we don't check parental attr
        if key == '_parent':
            return   

        def _check_primal(annotation, key, val):
            if isclass(annotation):
                if issubclass(annotation, Enum):
                    if val is not None and not hasattr(annotation, val):
                        return ValueError(
                            f'`{val}` not supported as {key} at config {cls.__name__}. Options: {annotation._member_names_}')
                elif not isinstance(val, annotation):
                    return TypeError(f'`Passed `{val}` at config: {cls.__name__}, field: {key}. Expected: {annotation}')
            elif annotation is Callable and not hasattr(val, '__call__'):
                return TypeError(f'`Passed `{val}` at config: {cls.__name__}, field: {key}, is not callable!')
            elif get_origin(annotation) is Literal and not val in get_args(annotation):
                return ValueError(f'Passed value `{val}` at config {cls.__name__}. Supported: {get_args(annotation)}')
            return False

        def _check(annotation, key, val):   
            options = get_args(annotation) or (annotation,)
            exs = [_check_primal(option, key, val)
                               for option in options]
            if exs and all(exs):
                raise MisconfigurationError(exs)
        
        annotation = cls.get_annotation(key)
        _check(annotation, key, val)

    def __new__(cls, *args, **kwargs):
        """Normalize constructor arguments and return them without instantiation.

        Instead of allocating an instance of the template, this method
        returns a tuple ``(cls, normalized_kwargs)`` where
        ``normalized_kwargs`` contains:

        * keyword arguments filtered via :func:`_normalize_kwargs`
          (compatible with ``__init__`` signature);
        * positional arguments mapped to the corresponding parameter names.

        This behaviour allows a separate factory layer to decide when and how
        to actually instantiate config objects.

        Raises
        ------
        KeyError
            If an unknown field is passed in ``kwargs``.
        """
        """We don't need template instances themselves. Only normalized parameters"""
        allowed_parameters = _normalize_kwargs(cls.__init__, kwargs)
        for k in kwargs:
            if k not in allowed_parameters:
                raise KeyError(f'Unknown field `{k}` was passed into {cls.__name__}')
        sign_args = tuple(signature(cls.__init__).parameters)
        complemented_args = dict(zip(sign_args[1:],
                                     args))
        allowed_parameters.update(complemented_args)
        return cls, allowed_parameters
    
    def __repr__(self):
        """Return a multi-line representation with field names and values."""
        params_str = "\n".join(f"{k}: {getattr(self, k)}" for k in self.__slots__)
        return f"{self.get_default_name()}: \n{params_str}\n"

    def get_parent(self):
        """Return parent configuration object if present."""
        return getattr(self, "_parent")

    def update(self, d: dict):
        """Update configuration fields using dotted keys.

        Parameters
        ----------
        d : dict
            Mapping from dotted keys (see :func:`get_nested`) to new values.
        """
        for k, v in d.items():
            obj, attr = get_nested(self, k)
            obj.__setattr__(attr, v)

    def get(self, key, default=None):
        """Retrieve a nested attribute using a dotted key.

        Parameters
        ----------
        key : str
            Dotted path to attribute.
        default : Any, optional
            Default value if attribute is not found.

        Returns
        -------
        Any
            Value of the attribute or ``default``.
        """
        return getattr(*get_nested(self, key), default)

    def keys(self) -> Iterable:
        """Return iterable of field names excluding the parent link."""
        return tuple(slot for slot in self.__slots__ if slot != "_parent")

    def items(self) -> Iterable:
        """Iterate over ``(key, value)`` pairs for all declared fields."""
        return ((k, self[k]) for k in self.keys())

    def to_dict(self) -> dict:
        """Convert configuration subtree to a plain dictionary.

        Nested objects that implement :meth:`to_dict` are converted
        recursively.
        """
        ret = {}
        for k, v in self.items():
            if hasattr(v, "to_dict"):
                v = v.to_dict()
            ret[k] = v
        return ret

    @property
    def config(self):
        """Expose self as ``config`` for ergonomic access in higher-level code."""
        return self


class ExtendableConfigTemplate(ConfigTemplate):
    """Config template that allows dynamic attributes in addition to slots.

    Unlike :class:`ConfigTemplate`, this class does not enforce type
    checks for attributes that are not present in ``__slots__``. Such
    dynamic fields are stored in ``__dict__``, and :meth:`keys` returns
    both static and dynamic keys.

    Warning
    -------
    Dynamic attributes are not validated by :meth:`check`. Use with care.
    """

    @classmethod
    def check(cls, key, val):
        """Validate only fields that are part of ``__slots__``."""
        if key not in cls.__slots__:
            return
        super().check(key, val)

    def keys(self):
        """Return list of both static and dynamic field names."""
        return [
            *tuple(slot for slot in self.__slots__ if slot != "_parent"),
            *list(self.__dict__),
        ]


@dataclass
class DeviceConfigTemplate(ConfigTemplate):
    """Configuration of the primary training/inference device.

    Attributes
    ----------
    device : {'cuda', 'cpu', 'gpu'}
        Device identifier used by the training loop.
    inference : {'onnx'}
        Inference backend for exported models (currently only ``'onnx'``).
    """

    device: Literal["cuda", "cpu", "gpu"] = "cuda"
    inference: Literal["onnx"] = "onnx"


@dataclass
class EdgeConfigTemplate(ConfigTemplate):
    """Configuration of an edge device deployment.

    Typically mirrors :class:`DeviceConfigTemplate`, but may be extended
    in the future for specific edge runtimes.
    """

    device: Literal["cuda", "cpu", "gpu"] = "cuda"
    inference: Literal["onnx"] = "onnx"


@dataclass
class DistributedConfigTemplate(ConfigTemplate):
    """Distributed execution parameters (currently Dask-oriented).

    Attributes
    ----------
    processes : bool
        Whether to use processes instead of threads.
    n_workers : int
        Number of worker processes/threads.
    threads_per_worker : int
        Number of threads per worker.
    memory_limit : {'auto'}
        Memory limit per worker (``'auto'`` – let backend decide).
    """

    processes: bool = False
    n_workers: int = 1
    threads_per_worker: int = 4
    memory_limit: Literal['auto'] = 'auto'  ###


@dataclass
class ComputeConfigTemplate(ConfigTemplate):
    """General compute and storage configuration.

    Attributes
    ----------
    backend : dict, optional
        Settings for the underlying compute backend (Dask/Ray/etc.).
    distributed : DistributedConfigTemplate, optional
        Parameters for distributed execution.
    output_folder : str or Path
        Directory where experiment artifacts are stored.
    use_cache : bool
        Whether to reuse cached intermediate results when possible.
    automl_folder : str or Path
        Directory for AutoML-related artifacts.
    """

    backend: dict = None
    distributed: DistributedConfigTemplate = None
    output_folder: Union[str, Path] = './current_experiment_folder'
    use_cache: bool = True
    automl_folder: Union[str, Path] = './current_automl_folder'


@dataclass
class FedotConfigTemplate(ConfigTemplate):
    """Wrapper for FEDOT AutoML settings.

    Attributes
    ----------
    timeout : int or float
        Global timeout for AutoML run.
    pop_size : int
        Population size used in evolutionary optimization.
    early_stopping_iterations : int
        Maximum number of iterations without improvement.
    early_stopping_timeout : int
        Time-based early stopping threshold.
    with_tuning : bool
        Whether to perform hyperparameter tuning.
    problem : FedotTaskEnum, optional
        Task type (classification, regression, forecasting, etc.).
    task_params : TaskTypesEnum, optional
        Additional task-specific parameters.
    metric : Iterable[str], optional
        List of metric names used for evaluation.
    n_jobs : int
        Number of parallel jobs (threads/processes) FEDOT can use.
    initial_assumption : nn.Module or str or dict, optional
        Initial pipeline/model assumption.
    available_operations : Iterable[str], optional
        Whitelist of allowed operations.
    optimizer : Any, optional
        Custom optimizer object or configuration.
    """

    """Evth for Fedot"""
    timeout: Union[int, float] = 10.0
    pop_size: int = 5
    early_stopping_iterations: int = 10
    early_stopping_timeout: int = 10
    with_tuning: bool = False
    problem: FedotTaskEnum = None
    task_params: Optional[TaskTypesEnum] = None
    metric: Optional[Iterable[str]] = None  ###
    n_jobs: int = -1
    initial_assumption: Union[Module, str, dict] = None
    available_operations: Optional[Iterable[str]] = None
    optimizer: Optional[Any] = None


@dataclass
class AutoMLConfigTemplate(ConfigTemplate):
    """AutoML-related configuration for FedCore.

    This extends the FEDOT configuration with FedCore-specific options,
    such as mutation strategies and custom optimizers.

    Attributes
    ----------
    fedot_config : FedotConfigTemplate, optional
        Underlying FEDOT configuration.
    mutation_agent : {'random'}
        Mutation agent used in AutoML search.
    mutation_strategy : {'params_mutation_strategy'}
        Concrete strategy identifier.
    optimizer : Any, optional
        Custom optimizer object used by AutoML (excluding FedCoreEvoOptimizer).
    """

    """Extension for FedCore-specific treats"""
    fedot_config: FedotConfigTemplate = None

    mutation_agent: Literal['random'] = 'random'
    mutation_strategy: Literal['params_mutation_strategy'] = 'params_mutation_strategy'
    optimizer: Optional[Any] = None  ### TODO which optimizers may be used? anything except FedCoreEvoOptimizer


@dataclass
class ModelArchitectureConfigTemplate(ConfigTemplate):
    """Basic model architecture settings.

    Attributes
    ----------
    input_dim : int, optional
        Input dimensionality.
    output_dim : int, optional
        Output dimensionality.
    depth : int or dict
        Model depth or a more detailed structure description.
    custom_model_params : dict, optional
        Extra backend-specific architecture parameters.
    """
    input_dim: Union[None, int] = None
    output_dim: Union[None, int] = None
    depth: Union[int, dict] = 3
    custom_model_params: dict = None


@dataclass
class TrainingTemplate(ConfigTemplate):
    """Computational Node settings. May include hooks summon keys"""
    log_each: Optional[int] = LookUp(None)
    eval_each: Optional[int] = LookUp(None)
    save_each: Optional[int] = LookUp(None)
    epochs: int = 1
    optimizer: Optimizers = 'adam'
    scheduler: Optional[Schedulers] = None
    criterion: Union[TorchLossesConstant, Callable] = LookUp(None)  # TODO add additional check for those fields which represent
    custom_learning_params: dict = None
    custom_criterions: dict = None
    model_architecture: ModelArchitectureConfigTemplate = None


@dataclass
class LearningConfigTemplate(ConfigTemplate):
    """High-level learning strategy configuration.

    Attributes
    ----------
    learning_strategy : {'from_scratch', 'checkpoint'}
        How to initialize model weights (from scratch or from checkpoint).
    peft_strategy : PEFTStrategies
        Parameter-efficient fine-tuning strategy.
    criterion : Callable or TorchLossesConstant or LookUp
        Global loss function configuration.
    peft_strategy_params : NeuralModelConfigTemplate, optional
        Additional parameters for PEFT strategy.
    learning_strategy_params : NeuralModelConfigTemplate, optional
        Additional parameters for the learning strategy.
    """
    learning_strategy: Literal['from_scratch', 'checkpoint'] = 'from_scratch'
    criterion: Union[Callable, TorchLossesConstant] = LookUp(None)
    peft_strategy_params: TrainingTemplate = None
    learning_strategy_params: TrainingTemplate = None


@dataclass
class APIConfigTemplate(ExtendableConfigTemplate):
    """Top-level API configuration for FedCore.

    This template aggregates all major config sections used by the API:
    device, AutoML, learning, compute and solver settings. It is extendable,
    so additional attributes may be attached at runtime if needed.

    Attributes
    ----------
    device_config : DeviceConfigTemplate, optional
        Device/runtime settings.
    automl_config : AutoMLConfigTemplate, optional
        AutoML/FEDOT configuration.
    learning_config : LearningConfigTemplate, optional
        High-level learning strategy configuration.
    compute_config : ComputeConfigTemplate, optional
        Compute and storage settings.
    solver : Any, optional
        Custom solver/manager implementation.
    predicted_probs : Any, optional
        Flag or configuration for returning prediction probabilities.
    original_model : Any, optional
        Reference to an externally provided model instance.
    """

    """Extendable (!) instead of APIManager"""
    device_config: DeviceConfigTemplate = None
    automl_config: AutoMLConfigTemplate = None
    learning_config: LearningConfigTemplate = None
    compute_config: ComputeConfigTemplate = None
    # optimization_agent: Any = FedcoreEvoOptimizer
    solver: Optional[Any] = None
    predicted_probs: Optional[Any] = None
    original_model: Optional[Any] = None


@dataclass
class LowRankTemplate(TrainingTemplate):
    """Configuration for low-rank (SVD-based) compression.

    Attributes
    ----------
    strategy : SLRStrategiesEnum
        Low-rank strategy identifier (e.g. ``'quantile'``).
    rank_prune_each : int
        How often (in epochs) to apply rank pruning. ``-1`` disables it.
    custom_criterions : dict, optional
        Additional structure loss terms for low-rank models.
    compose_mode : {'one_layer', 'two_layers', 'three_layers'}, optional
        Mode used when composing decomposed layers.
    non_adaptive_threshold : float
        Threshold for non-adaptive rank pruning.
    finetune_params : TrainingTemplate, optional
        Fine-tuning parameters after compression.
    decomposer : {'svd', 'rsvd', 'cur', 'two_sided'}, optional
        Type of decomposer from tdecomp to use (default: 'svd').
    decomposing_mode : {'channel', 'spatial'}, optional
        Decomposition mode for weights (default: 'channel').
        'channel' mode reshapes weights along channel dimension.
        'spatial' mode reshapes weights along spatial dimensions.
    rank : int or float, optional
        Rank for decomposition. If None, will be estimated automatically.
        Can be int (absolute rank) or float (relative rank, 0-1).
    distortion_factor : float, optional
        Distortion factor for decomposer (default: 0.6). Must be in (0, 1].
    random_init : str, optional
        Random initialization method for randomized decomposers (default: 'normal').
    power : int, optional
        Power parameter for RandomizedSVD (default: 3).
    fedcore_id : str, optional
        FedCore model registry ID for model tracking and registration.
    """

    """Example of specific node template"""
    strategy: SLRStrategiesEnum = 'quantile'
    rank_prune_each: int = -1
    custom_criterions: dict = None  # {'norm_loss':{...},
    compose_mode: Optional[Literal['one_layer', 'two_layers', 'three_layers']] = None
    non_adaptive_threshold: float = .5
    finetune_params: TrainingTemplate = None
    decomposer: Optional[Literal['svd', 'rsvd', 'cur', 'two_sided']] = 'svd'
    decomposing_mode: Optional[Literal['channel', 'spatial']] = None
    rank: Optional[Union[int, float]] = None
    distortion_factor: float = 0.6
    random_init: str = 'normal'
    power: int = 3
    fedcore_id: Optional[str] = None


@dataclass
class PruningTemplate(TrainingTemplate):
    """Configuration for structured/unstructured pruning.

    Attributes
    ----------
    importance : str
        Importance criterion name (e.g. ``"magnitude"``, ``"lamp"``, etc.).
    importance_norm : int
        Norm used when aggregating importance scores.
    pruning_ratio : float
        Global ratio of parameters to prune.
    importance_reduction : str
        Reduction method across channels/layers (legacy field, may be dropped).
    importance_normalize : str
        Normalization strategy for importance scores (legacy field).
    pruning_iterations : int
        Number of iterative pruning steps (legacy field).
    finetune_params : TrainingTemplate, optional
        Fine-tuning parameters after pruning.
    prune_each : int
        Frequency (in epochs) to apply pruning; ``-1`` disables it.
    """

    """Example of specific node template"""
    importance: str = "magnitude" # main
    importance_norm: int = 1 # main
    pruning_ratio: float = 0.5 # main
    importance_reduction: str = 'max' # drop 
    importance_normalize: str = 'max' # drop
    pruning_iterations: int = 1 # drop
    finetune_params: TrainingTemplate = None
    
@dataclass
class QuantTemplate(TrainingTemplate):
    """Configuration for model quantization.

    Attributes
    ----------
    quant_type : str
        Quantization mode, one of :class:`QuantMode` values.
    allow_emb : bool
        Whether embedding layers can be quantized.
    allow_conv : bool
        Whether convolution layers can be quantized.
    quant_each : int
        Apply quantization hook every N epochs. ``-1`` disables periodic
        quantization. For QAT this usually marks the final conversion epoch.
    prepare_qat_after_epoch : int
        Epoch number at which quantization-aware training should be prepared
        via ``prepare_qat``. Must be less than ``quant_each`` for QAT.
    """

    """Example of specific node template"""
    quant_type: str = "dynamic" # dynamic, static, qat
    allow_emb: bool = False
    allow_conv: bool = True
    qat_params: TrainingTemplate = None
