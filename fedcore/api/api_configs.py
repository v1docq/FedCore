from enum import Enum
from functools import reduce
from inspect import signature, isclass
from typing import get_origin, get_args, Literal, Optional, Union

from torch.ao.quantization.utils import _normalize_kwargs
from torch.nn import Module

from fedcore.repository.constanst_repository import Schedulers, Optimizers, PEFTStrategies


__all__ = [
    'ConfigTemplate',
    'DeviceConfigTemplate',
    'EdgeConfigTemplate',
    'AutoMLConfigTemplate',
    'NeuralModelConfigTemplate',
    'LearningConfigTemplate',
    'APIConfigTemplate',

    'get_nested',
]

def get_nested(root: object, k: str):
    *path, last = k.split('.')
    return reduce(getattr, path, root), last
    

class ConfigTemplate:
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
        annotation = cls.get_annotation(key)
        if isclass(annotation):
            if issubclass(annotation, Enum) and val is not None and not hasattr(annotation, val):
                raise ValueError(f'`{val}` not supported as {key}. Options: {annotation._member_names_}')
            elif not isinstance(val, annotation):
                raise TypeError(f'`Passed type {val.__class__}. Expected: {annotation}')
        if get_origin(annotation) is Literal and not val in get_args(annotation):
            raise ValueError(f'Passed value `{val}`. Supported: {get_args(annotation)}')

    def __new__(cls, *args, **kwargs):
        allowed_parameters = _normalize_kwargs(cls.__init__, kwargs)
        for k in kwargs:
            if k not in allowed_parameters:
                raise KeyError(f'Unknown field `{k}` was passed into {cls.__name__}')
        sign_args = tuple(signature(cls.__init__).parameters)
        complemented_args = dict(zip(sign_args[1:],
            args))
        allowed_parameters.update(complemented_args)
        return cls, allowed_parameters

class DeviceConfigTemplate(ConfigTemplate):
    # """Backend specifications"""
    def __init__(self,
        device: Literal['cuda', 'cpu', 'gpu'] = 'cuda',
        inference: Literal['onnx'] = 'onnx',
    ): pass
    
class EdgeConfigTemplate(ConfigTemplate):
    # """Backend specifications"""
    def __init__(self,
        device: Literal['cuda', 'cpu', 'gpu'] = 'cuda',
        inference: Literal['onnx'] = 'onnx',
    ): pass

class AutoMLConfigTemplate(ConfigTemplate):
    """ """
    def __init__(self,
        timeout: int = 10,
        pop_size: int = 5,  
        early_stopping_iterations: int = 10,
        early_stopping_timeout: int = 10,
        with_tuning: bool = False,
        n_jobs: int = -1,    
        initial_assumption: Union[Module, str] = None,
        optimizer: dict = {'optimisation_strategy':
                          {'mutation_agent': 'random',
                           'mutation_strategy': "params_mutation_strategy"},
                      'optimisation_agent': 'Fedcore'},
        optimization_agent: Literal['Fedcore'] = 'Fedcore',
        mutation_strategy: Literal['params_mutation_strategy'] = 'params_mutation_strategy',
    ): pass

class NeuralModelConfigTemplate(ConfigTemplate):
    def __init__(self, 
        epochs: int = 15, 
        learning_rate: float = 0.0001,
        optimizer: Optimizers = 'adam',
        scheduler: Optional[Schedulers] = None,
    ): pass 

class LearningConfigTemplate(ConfigTemplate):
    def __init__(self, 
        learning_strategy: Literal['from_scratch', ] = 'from_scratch',
        peft_strategy: PEFTStrategies = 'training',
        peft_strategy_params: str = '',
        learning_params: NeuralModelConfigTemplate = None,
    ): pass

class ComputeConfigTemplate(ConfigTemplate):
    def __init__(self,
        some_dict: dict = None
    ): pass

class APIConfigTemplate(ConfigTemplate):
    def __init__(self, 
        device_config: DeviceConfigTemplate = None,
        automl_config: AutoMLConfigTemplate = None,
        learning_config: LearningConfigTemplate = None,
        compute_config: ComputeConfigTemplate = None
    ): pass
