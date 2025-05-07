from dataclasses import dataclass
from enum import Enum
from functools import reduce
from inspect import signature, isclass
from numbers import Number
from pathlib import Path
from typing import (
    get_origin, get_args,   
    Any, Callable, Dict, Iterable, Literal, Optional, Union, 
)
import logging

from torch.ao.quantization.utils import _normalize_kwargs
from torch.nn import Module

from fedcore.repository.constanst_repository import (
    FedotTaskEnum,
    Schedulers, 
    Optimizers, 
    PEFTStrategies,
    SLRStrategiesEnum,
    TaskTypesEnum,
    TorchLossesConstant,
)

__all__ = [
    'ConfigTemplate',
    'DeviceConfigTemplate',
    'EdgeConfigTemplate',
    'AutoMLConfigTemplate',
    'NeuralModelConfigTemplate',
    'LearningConfigTemplate',
    'APIConfigTemplate',
    'get_nested',
    'LookUp',
    'LookUp',
]


def get_nested(root: object, k: str):
    """Func to allow subcrtiption like config['x.y.x']"""
    *path, last = k.split('.')
    return reduce(getattr, path, root), last

class MisconfigurationError(BaseException):
    def __init__(self, exs, *args):
        super().__init__(*args)
        self.exs = exs

    def __repr__(self):
        return '\n'.join([f'\t{str(x)}' for x in self.exs])

    def __str__(self):
        return self.__repr__()


@dataclass
class LookUp:
    """Wrapping telling to fill it from parental config if not specified"""
    value: Any


@dataclass
class ConfigTemplate:
    """Fixed Structure Config"""
    __slots__ = tuple()

    @classmethod
    def get_default_name(cls):
        name = cls.__name__.split('.')[-1]
        if name.endswith('Template'):
            name = name[:-8]
        return name
    
    @classmethod
    def get_annotation(cls, key):
        obj = cls if ConfigTemplate in cls.__bases__ else cls.__bases__[0]
        return signature(obj.__init__).parameters[key].annotation

    @classmethod
    def check(cls, key, val):
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
        params_str = '\n'.join(
            f'{k}: {getattr(self, k)}' for k in self.__slots__
        )
        return f'{self.get_default_name()}: \n{params_str}\n'
    
    def get_parent(self):
        return getattr(self, '_parent')

    def update(self, d: dict):
        for k, v in d.items():
            obj, attr = get_nested(self, k)
            obj.__setattr__(attr, v)

    def get(self, key, default=None):
        return getattr(*get_nested(self, key), default)

    def keys(self) -> Iterable:
        return tuple(slot for slot in self.__slots__ if slot != '_parent')

    def items(self) -> Iterable:
        return (
            (k, self[k]) for k in self.keys()
        )

    def to_dict(self) -> dict:
        ret = {}
        for k, v in self.items():
            if hasattr(v, 'to_dict'):
                v = v.to_dict()
            ret[k] = v
        return ret

    @property
    def config(self):
        return self


class ExtendableConfigTemplate(ConfigTemplate):
    """Allows to dynamically add attributes.
    Warning: check behaviour of keys with newly added ones"""

    @classmethod
    def check(cls, key, val):
        if not key in cls.__slots__:
            return
        super().check(key, val)

    def keys(self):
        return [*tuple(slot for slot in self.__slots__ if slot != '_parent'), *list(self.__dict__)]


@dataclass
class DeviceConfigTemplate(ConfigTemplate):
    """Training device specification. TODO check fields"""
    device: Literal['cuda', 'cpu', 'gpu'] = 'cuda'
    inference: Literal['onnx'] = 'onnx'
    
@dataclass
class EdgeConfigTemplate(ConfigTemplate):
    """Edge device specification"""
    device: Literal['cuda', 'cpu', 'gpu'] = 'cuda'
    inference: Literal['onnx'] = 'onnx'


@dataclass
class DistributedConfigTemplate(ConfigTemplate):
    """Everything for Dask"""
    processes: bool = False
    n_workers: int = 1
    threads_per_worker: int = 4
    memory_limit: Literal['auto'] = 'auto'  ###


@dataclass
class ComputeConfigTemplate(ConfigTemplate):
    """How we learn, where we store"""
    backend: dict = None
    distributed: DistributedConfigTemplate = None
    output_folder: Union[str, Path] = './current_experiment_folder'
    use_cache: bool = True
    automl_folder: Union[str, Path] = './current_automl_folder'


@dataclass
class FedotConfigTemplate(ConfigTemplate):
    """Evth for Fedot"""
    timeout: float = 10.0
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
    """Extension for FedCore-specific treats"""
    fedot_config: FedotConfigTemplate = None

    mutation_agent: Literal['random'] = 'random'
    mutation_strategy: Literal['params_mutation_strategy'] = 'params_mutation_strategy'
    optimizer: Optional[Any] = None  ### TODO which optimizers may be used? anything except FedCoreEvoOptimizer


@dataclass
class NodeTemplate(ConfigTemplate):
    """Computational Node settings. May include hooks summon keys"""
    log_each: Optional[int] = LookUp(None)
    eval_each: Optional[int] = LookUp(None)
    save_each: Optional[int] = LookUp(None)
    epochs: int = 1
    optimizer: Optimizers = 'adam'
    scheduler: Optional[Schedulers] = None
    criterion: Union[TorchLossesConstant, Callable] = LookUp(None)  # TODO add additional check for those fields which represent


@dataclass
class ModelArchitectureConfigTemplate(ConfigTemplate):
    """Example of specific node template"""
    input_dim: Union[None, int] = None
    output_dim: Union[None, int] = None
    depth: Union[int, dict] = 3
    custom_model_params: dict = None


@dataclass
class NeuralModelConfigTemplate(NodeTemplate):
    """Additional learning settings. May be redundant"""
    custom_learning_params: dict = None
    custom_criterions: dict = None
    model_architecture: ModelArchitectureConfigTemplate = None


@dataclass
class LearningConfigTemplate(ConfigTemplate):
    """Copies previeous version od learning config"""
    learning_strategy: Literal['from_scratch', 'checkpoint'] = 'from_scratch'
    peft_strategy: PEFTStrategies = 'training'
    criterion: Union[Callable, TorchLossesConstant] = LookUp(None)
    peft_strategy_params: NeuralModelConfigTemplate = None
    learning_strategy_params: NeuralModelConfigTemplate = None


@dataclass
class APIConfigTemplate(ExtendableConfigTemplate):
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
class LowRankTemplate(NeuralModelConfigTemplate):
    """Example of specific node template"""
    strategy: SLRStrategiesEnum = 'quantile'
    rank_prune_each: int = -1
    custom_criterions: dict = None  # {'norm_loss':{...},
    non_adaptive_threshold: float = .5
    finetune_params: NeuralModelConfigTemplate = None


@dataclass
class PruningTemplate(NeuralModelConfigTemplate):
    """Example of specific node template"""
    importance: str = "magnitude" # main
    importance_norm: int = 1 # main
    pruning_ratio: float = 0.5 # main
    importance_reduction: str = 'max' # drop 
    importance_normalize: str = 'max' # drop
    pruning_iterations: int = 1 # drop
    finetune_params: NeuralModelConfigTemplate = None
    
@dataclass
class QuantTemplate(NeuralModelConfigTemplate):
    """Example of specific node template"""
    quant_type: str = "dynamic" # dynamic, static, qat
    allow_emb: bool = False
    allow_conv: bool = True
    qat_params: NeuralModelConfigTemplate = None