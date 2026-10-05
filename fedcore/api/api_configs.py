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
* Specific templates (e.g. :class:`TrainingTemplate`,
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
from collections.abc import Mapping, Iterable as IterableABC, Callable as CallableABC
from types import UnionType
import logging
from math import isfinite

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
    'LoraTemplate',
    'LoRATemplate',
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


class MisconfigurationError(ValueError):
    """Aggregated configuration validation error.

    Instances of this error contain a list of underlying exceptions
    (typically :class:`TypeError` or :class:`ValueError`) raised during
    config field validation. The string representation concatenates all
    messages line by line.
    """

    def __init__(self, exs, *args, field=None):
        self.exs = list(exs) if isinstance(exs, (list, tuple)) else [exs]
        self.field = field
        self.fields = tuple(error.field for error in self.exs
                            if isinstance(error, MisconfigurationError) and error.field is not None)
        super().__init__(str(self), *args)

    def __repr__(self):
        return '\n'.join([f'\t{str(x)}' for x in self.exs])

    def __str__(self):
        return self.__repr__()


def matches_annotation(value, annotation):
    """Validate closed alternatives before examining their members."""
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation is Any:
        return True
    if annotation in (int, float) and isinstance(value, bool):
        return False
    if origin in (Union, UnionType):
        return any(matches_annotation(value, option) for option in args)
    if origin is Literal:
        return any(type(value) is type(option) and value == option for option in args)
    if annotation is None or annotation is type(None):
        return value is None
    if annotation is Callable or origin is CallableABC:
        return callable(value)
    if isclass(annotation) and issubclass(annotation, Enum):
        return (value is None or isinstance(value, annotation) or
                isinstance(value, str) and any(value in (member.name, member.value) for member in annotation))
    if origin is not None:
        if not isinstance(value, origin):
            return False
        if isinstance(value, Mapping) and args:
            return all(matches_annotation(k, args[0]) and matches_annotation(v, args[1])
                       for k, v in value.items())
        if isinstance(value, IterableABC) and args:
            if origin is tuple and len(args) > 1 and args[-1] is not Ellipsis:
                return len(value) == len(args) and all(matches_annotation(v, a) for v, a in zip(value, args))
            return all(matches_annotation(v, args[0]) for v in value)
        return True
    return isinstance(value, annotation) if isclass(annotation) else True


def validate_config(config):
    """Validate fields and dependent numeric settings without starting resources."""
    errors = []
    def visit(section, path=''):
        invalid = set()
        for key, value in section.items():
            field = f'{path}.{key}' if path else key
            try:
                section.check(key, value)
            except MisconfigurationError as error:
                errors.append(error)
                invalid.add(key)
            if isinstance(value, ConfigTemplate):
                visit(value, field)
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, ConfigTemplate):
                        visit(item, f'{field}[{i}]')
        if isinstance(section, DistributedConfigTemplate):
            for key in ('n_workers', 'threads_per_worker'):
                if key not in invalid and getattr(section, key) <= 0:
                    errors.append(MisconfigurationError(f'{path}.{key}: must be positive', field=key))
        if isinstance(section, TrainingTemplate) and 'epochs' not in invalid and section.epochs < 0:
            errors.append(MisconfigurationError(f'{path}.epochs: must be nonnegative', field='epochs'))
        if isinstance(section, LowRankTemplate):
            if 'distortion_factor' not in invalid and not 0 < section.distortion_factor <= 1:
                errors.append(MisconfigurationError(f'{path}.distortion_factor: expected (0, 1]', field='distortion_factor'))
            if 'rank' not in invalid and section.rank is not None and (section.rank <= 0 or isinstance(section.rank, float) and section.rank > 1):
                errors.append(MisconfigurationError(f'{path}.rank: expected positive integer or fraction in (0, 1]', field='rank'))
        if isinstance(section, PruningTemplate) and 'pruning_ratio' not in invalid and not 0 <= section.pruning_ratio <= 1:
            errors.append(MisconfigurationError(f'{path}.pruning_ratio: expected [0, 1]', field='pruning_ratio'))
        if isinstance(section, LoraTemplate):
            if invalid:
                return
            if section.epochs <= 0:
                errors.append(MisconfigurationError(f'{path}.epochs: LoRA requires a positive integer', field='epochs'))
            if section.lora_r <= 0 or any(v is not None and v <= 0 for v in (section.rank, section.r)):
                errors.append(MisconfigurationError(f'{path}.lora_r: rank must be positive', field='lora_r'))
            if section.rank is not None and section.r is not None and section.rank != section.r:
                errors.append(MisconfigurationError(f'{path}.rank: conflicts with r', field='rank'))
            if not isfinite(section.lora_alpha) or not 0 <= section.lora_dropout < 1:
                errors.append(MisconfigurationError(f'{path}: invalid LoRA alpha or dropout', field='lora_dropout'))
            if section.use_peft:
                errors.append(MisconfigurationError(f'{path}.use_peft: this operation supports local LoRA only', field='use_peft'))
            if section.target_layers is not None and section.lora_target_modules is not None and section.target_layers != section.lora_target_modules:
                errors.append(MisconfigurationError(f'{path}.target_layers: conflicting aliases', field='target_layers'))
    visit(config)
    if errors:
        raise MisconfigurationError(errors)
    from fedcore.repository.capabilities import validate_requested_modes
    validate_requested_modes(config.to_dict())
    return config


@dataclass(frozen=True)
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
        obj = getattr(cls, '__template__', cls)
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
        if key == '_parent':
            return
        annotation = cls.get_annotation(key)
        if not matches_annotation(val, annotation):
            raise MisconfigurationError(
                f'{cls.__name__}.{key}: expected {annotation}, got {val!r}', field=key)

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
        def convert(value):
            if isinstance(value, ConfigTemplate):
                return value.to_dict()
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(convert(item) for item in value)
            return value
        return {key: convert(value) for key, value in self.items()}

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
    initial_assumption: Optional[Union[Module, str, dict]] = None
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
    model_factory: Optional[Callable] = None
    model_factory_before: Optional[Callable] = None
    model_factory_after: Optional[Callable] = None


@dataclass
class LearningConfigTemplate(ExtendableConfigTemplate):
    """High-level learning strategy configuration.

    Attributes
    ----------
    learning_strategy : {'from_scratch', 'checkpoint'}
        How to initialize model weights (from scratch or from checkpoint).
    peft_strategy : PEFTStrategies
        Parameter-efficient fine-tuning strategy.
    criterion : Callable or TorchLossesConstant or LookUp
        Global loss function configuration.
    peft_strategy_params : TrainingTemplate, optional
        Additional parameters for PEFT strategy.
    learning_strategy_params : TrainingTemplate, optional
        Additional parameters for the learning strategy.
    """
    learning_strategy: Literal['from_scratch', 'checkpoint'] = 'from_scratch'
    criterion: Union[Callable, TorchLossesConstant] = LookUp(None)
    peft_strategy_params: Union[TrainingTemplate, List[TrainingTemplate]] = None
    learning_strategy_params: TrainingTemplate = None
    fedcore_id = None


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
class LoraTemplate(TrainingTemplate):
    """Local LoRA training; aliases normalize at the public API boundary."""
    lora_r: int = 8
    lora_alpha: Union[int, float] = 16.0
    lora_dropout: float = 0.0
    lora_target_modules: Optional[List[str]] = None
    lora_bias: Literal['none'] = 'none'
    use_peft: bool = False
    rank: Optional[int] = None
    r: Optional[int] = None
    target_layers: Optional[List[str]] = None
    lr: float = 0.001


LoRATemplate = LoraTemplate


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
    prune_each: int = -1
    importance: str = "magnitude" # main
    importance_norm: int = 1 # main
    pruning_ratio: float = 0.5 # main
    importance_reduction: str = 'max' # drop
    importance_normalize: str = 'max' # drop
    pruning_iterations: int = 1 # drop
    finetune_params: TrainingTemplate = None

@dataclass
class QuantizationTemplate(TrainingTemplate):
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
