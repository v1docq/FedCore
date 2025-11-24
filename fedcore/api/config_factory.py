from enum import Enum
from inspect import signature, isclass
from typing import OrderedDict, Tuple, get_args, get_origin, Iterable, Literal, Optional, Union

from .api_configs import ConfigTemplate, ExtendableConfigTemplate, get_nested, LookUp, MisconfigurationError


__all__ = [
    'ConfigFactory',
]

        
class ConfigFactory:
    registered_configs: dict = {}

    def __new__(*ars, **kwargs):
        raise Exception('ConfigFactory is a static class')

    @classmethod
    def from_template(cls, template: Tuple[type, OrderedDict], class_name: str = None):
        template_cls, content = template #content - is dict of allowed params (epochs: 5 for example)
        if class_name is None:
            class_name = template_cls.get_default_name()
        cls.registered_configs[class_name] = template_cls
        slots = list(content) + ['_parent']

        def __init__(self, parent=None, **kwargs):
            object.__setattr__(self, '_parent', parent)
            content.update(kwargs)
            misconf_errors = []
            for key, value in content.items():
                try:
                    value, need_check = ConfigFactory._instantiate_default(self, class_name, key, value)
                    if need_check:
                        self.__class__.check(key, value) # ConfigTemplate#check
                        setattr(self, key, value) #May be overrided by child (<Something>Template) with additional check on the way of setattr
                    else:
                        object.__setattr__(self, key, value) #set directly self.key = value
                except MisconfigurationError as e:
                    misconf_errors.append(e)
                except Exception as x:
                    if not misconf_errors:
                        raise x
                    else:
                        continue
            if misconf_errors:
                raise MisconfigurationError(misconf_errors)
                    
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

        def __setattr__ex(self, key, value):
            if key in self.__slots__:
                self.check(key, value)
            elif not key in self.additional_features:
                self.additional_features.append(key)
                
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
        return type(class_name, (cls.registered_configs[class_name],), class_dict)
    
    @classmethod
    def _get_annotation(cls, config_class_name, key):
        """Returns annotation of key. For example:  
        __init__(self, smth: int)  

        annotation will be "int" for key=smth
        """
        template_cls = cls.registered_configs[config_class_name]
        return signature(template_cls.__init__).parameters[key].annotation
    
    @staticmethod
    def __is_optional(annotation):
        origin = get_origin(annotation)
        return ((origin is Union or origin is Literal) and 
           type(None) in get_args(annotation))         
    

    @classmethod
    def _instantiate_default(cls, self: ConfigTemplate, config_class_name: str, k: str, v):
        """Gets default params of config from parent, if value is <code>LookUp</code>
        or gets default value from <code>LookUp(value)</code>, or gets just default value
        """
        # check look-up
        if isinstance(v, LookUp):
            if self._parent:
                v = getattr(self._parent, k, None)
            else:
                v = v.value
        annotation = cls._get_annotation(config_class_name, k)
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
                    f'Config: {config_class_name}')
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
