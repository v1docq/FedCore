from enum import Enum
from inspect import signature, isclass
from typing import get_args, get_origin, Literal, Optional, Union

from .api_configs import ConfigTemplate, get_nested


__all__ = [
    'ConfigFactory',
]

        
class ConfigFactory:
    registered_configs: dict = {}

    def __new__(*ars, **kwargs):
        raise Exception('ConfigFactory is a static class')

    @classmethod
    def from_template(cls, template: ConfigTemplate, name:str = None):
        template_cls, content = template
        if name is None:
            name = template_cls.get_default_name()
        cls.registered_configs[name] = template_cls
        slots = tuple(content)
        def __init__(self):
            for key, value in content.items():
                value, need_check = ConfigFactory._instantiate_default(name, key, value)
                if not need_check:
                    self.__class__.check(key, value)
                    setattr(self, key, value)
                else:
                    object.__setattr__(self, key, value)

        def __getitem__(self, key):
            val = getattr(self, key)
            if isinstance(val, Enum):
                return val.value
            return val

        def __new__(cls, *args, **kwargs):
            return object.__new__(cls, *args, **kwargs)
        
        def __setattr__(self: ConfigTemplate, key, value):
            self.check(key, value)
            ConfigTemplate.__setattr__(self, key, value)
        
        def __setitem__(self, key, value):
            orig_type = type(getattr(self, key))
            if not isinstance(value, orig_type): #isinstance or is?
                raise ValueError(f'Passing wrong argument of type {value.__class__.__name__}! Required: {orig_type}')
            setattr(self, key, value)

        def __repr__(self: ConfigTemplate):
            params_str = '\n'.join(
                f'{k}: {getattr(self, k)}' for k in self.__slots__
            )
            return f'{self.get_default_name()}: \n{params_str}\n'

        def update(self, d: dict):
            for k, v in d.items():
                obj, attr = get_nested(self, k)
                obj.__setattr__(attr, v)

        def get(self, key, default=None):
            return getattr(*get_nested(self, key), default)
        
        class_dict = {
            '__slots__': slots,
            '__init__': __init__,
            '__new__': __new__,
            '__getitem__': __getitem__,
            '__setitem__': __setitem__,
            '__setattr__': __setattr__,
            '__repr__': __repr__,
            'get': get,
            'update': update,
        }
        return type(name, (cls.registered_configs[name],), class_dict)
    
    @classmethod
    def _get_annotation(cls, config_name, key):
        template_cls = cls.registered_configs[config_name]
        return signature(template_cls.__init__).parameters[key].annotation
    
    @staticmethod
    def __is_optional(annotation):
        origin = get_origin(annotation)
        return ((origin is Union or origin is Literal) and 
           type(None) in get_args(annotation))         
    
    @classmethod
    def _instantiate_default(cls, config_name: str, k: str, v):
        if v is not None:
            return v, True
        # check if explicitly optional
        annotation = cls._get_annotation(config_name, k)
        if cls.__is_optional(annotation):
            return None, False
        # if another config, run recursively
        if isclass(annotation) and issubclass(annotation, ConfigTemplate):
            return ConfigFactory.from_template(annotation(), annotation.get_default_name())(), False
        # try instantiate empty object
        x = ValueError(f'Misconfiguration! detected class `{annotation}`\
                        has wrong argument passed and not explicitly optional.')
        # case union
        for arg in get_args(annotation):
            try:
                return arg(), False
            except:
                raise x
        # case type
        if isclass(annotation):
            try:
                return annotation(), False
            except:
                raise x
        raise x
