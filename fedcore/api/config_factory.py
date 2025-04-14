from enum import Enum
from inspect import signature, isclass
from typing import get_args, get_origin, Iterable, Literal, Optional, Union

from .api_configs import ConfigTemplate, ExtendableConfigTemplate, get_nested, LookUp


__all__ = [
    'ConfigFactory',
]

        
class ConfigFactory:
    registered_configs: dict = {}

    def __new__(*ars, **kwargs):
        raise Exception('ConfigFactory is a static class')

    @classmethod
    def from_template(cls, template: ConfigTemplate, name: str = None):
        template_cls, content = template
        if name is None:
            name = template_cls.get_default_name()
        cls.registered_configs[name] = template_cls
        slots = list(content) + ['_parent']

        def __init__(self, parent=None, **kwargs):
            object.__setattr__(self, '_parent', parent)
            content.update(kwargs)
            for key, value in content.items():
                value, need_check = ConfigFactory._instantiate_default(self, name, key, value)
                if need_check:
                    self.__class__.check(key, value)
                    setattr(self, key, value)
                else:
                    object.__setattr__(self, key, value)
            if not isinstance(self, ExtendableConfigTemplate):
                delattr(self, '__dict__')
            self._parent = None

        def __new__(cls, *args, **kwargs):
            return object.__new__(cls)
        
        def __getitem__(self, key):
            val = getattr(self, key)
            if isinstance(val, Enum):
                return val.value
            return val
        
        def __contains__(self, obj):
            return hasattr(self, str(obj))
            
        def __setattr__(self, key, value):
            self.check(key, value)
            ConfigTemplate.__setattr__(self, key, value)
                
        def __setitem__(self, key, value):
            orig_type = type(getattr(self, key))
            if not isinstance(value, orig_type): #isinstance or is?
                raise ValueError(f'Passing wrong argument of type {value.__class__.__name__}! Required: {orig_type}')
            setattr(self, key, value)
        
        class_dict = {
            '__slots__': slots,
            '__init__': __init__,
            '__new__': __new__,
            '__getitem__': __getitem__,
            '__setitem__': __setitem__,
            '__setattr__': __setattr__,
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
    def _instantiate_default(cls, self: ConfigTemplate, config_name: str, k: str, v):
        # check look-up
        if isinstance(v, LookUp):
            if self._parent:
                v = getattr(self._parent, k, None)
            else:
                v = v.value
        annotation = cls._get_annotation(config_name, k)
        is_config = isclass(annotation) and issubclass(annotation, ConfigTemplate)
        
        if v is not None and not is_config:
            return v, True
        # check if explicitly optional
        if cls.__is_optional(annotation):
            return None, False
        # if another config and not specified, run recursively
        if is_config and v is None:
            x = ConfigFactory.from_template(
                annotation(), annotation.get_default_name())(
                    parent=self
                ), False
            return x
        # specified
        if is_config and isinstance(v, tuple):
            x = ConfigFactory.from_template(v)(parent=self), False
            return x
        
        # try instantiate empty object
        x = ValueError(f'Misconfiguration! detected class `{annotation}`\n' +
                    f'has wrong argument passed at `{k}` and not explicitly optional.\n' +
                    f'Config: {config_name}')
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
