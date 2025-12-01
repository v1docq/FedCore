"""Factory for building concrete configuration classes from templates.

This module provides :class:`ConfigFactory`, a helper that turns
``ConfigTemplate`` objects (returned as ``(TemplateClass, kwargs)`` from
``ConfigTemplate.__new__``) into concrete dataclass-like configuration
classes with:

* proper ``__init__`` that performs type checking and instantiates nested
  configs;
* support for ``LookUp`` fields (inherit values from a parent config);
* automatic handling of optional / nested ``ConfigTemplate`` fields.

Resulting classes are lightweight containers that can be used in the rest
of the FedCore API as strongly typed configuration objects.
"""

from enum import Enum
from inspect import signature, isclass
from typing import get_args, get_origin, Iterable, Literal, Optional, Union

from .api_configs import (
    ConfigTemplate,
    ExtendableConfigTemplate,
    get_nested,
    LookUp,
    MisconfigurationError,
)


__all__ = [
    "ConfigFactory",
]


class ConfigFactory:
    """Static factory for materializing configs from templates.

    The typical workflow is:

    1. A template class derived from :class:`ConfigTemplate` is called,
       which returns a tuple ``(TemplateClass, normalized_kwargs)`` instead
       of an instance.
    2. This tuple is passed to :meth:`ConfigFactory.from_template`, which
       creates a concrete subclass with appropriate ``__slots__`` and
       an ``__init__`` that:
       * resolves :class:`LookUp` values using the parent config;
       * instantiates nested :class:`ConfigTemplate`-based configs;
       * validates all values against type annotations (including enums
         and ``Literal`` types).
    3. The returned class can then be instantiated as a normal config
       object: ``ConfigCls(parent=..., **overrides)``.

    All created config classes are registered in :attr:`registered_configs`
    by name so that helper methods (like :meth:`_get_annotation`) can
    access their annotations.
    """

    #: Registry of template name → template class.
    registered_configs: dict = {}
    _listed_instantiation = {'peft_strategy_params'}

    def __new__(*ars, **kwargs):
        """Disable direct instantiation of :class:`ConfigFactory`.

        The factory is intended to be used only via its classmethods.
        """
        raise Exception("ConfigFactory is a static class")

    @classmethod
    def from_template(cls, template: ConfigTemplate, name: str = None):
        """Create a concrete config class from a template tuple.

        Parameters
        ----------
        template : ConfigTemplate
            The tuple returned by calling a template class derived from
            :class:`ConfigTemplate`, i.e. ``(TemplateClass, kwargs_dict)``.
        name : str, optional
            Name of the resulting config class. If not provided, the
            ``TemplateClass.get_default_name()`` is used.

        Returns
        -------
        type[ConfigTemplate]
            Dynamically created config class that:

            * inherits from the template class;
            * has slots for all template fields plus ``_parent``;
            * performs value instantiation and type validation in ``__init__``;
            * supports dictionary-style access via ``__getitem__`` /
              ``__setitem__``.
        """
        template_cls, content = template
        if name is None:
            name = template_cls.get_default_name()
        cls.registered_configs[name] = template_cls
        slots = list(content) + ["_parent"]

        def __init__(self, parent=None, **kwargs):
            object.__setattr__(self, "_parent", parent)
            content.update(kwargs)
            misconf_errors = []
            for key, value in content.items():
                try:
                    if key in cls._listed_instantiation and isinstance(value, (tuple, list)): # case for listings
                        enlisted_value = []
                        for item in value:
                            item, need_check = ConfigFactory._instantiate_default(self, name, key, item)
                            if need_check:
                                self.__class__.check(key, item)
                            enlisted_value.append(item)
                        object.__setattr__(self, key, enlisted_value)
                    else: # general case
                        value, need_check = ConfigFactory._instantiate_default(self, name, key, value)
                        if need_check:
                            self.__class__.check(key, value)
                            setattr(self, key, value)
                        else:
                            object.__setattr__(self, key, value)
                except MisconfigurationError as e:
                    misconf_errors.append(e)
                except Exception as x:
                    if not misconf_errors:
                        raise x
                    else:
                        continue
            if misconf_errors:
                raise MisconfigurationError(misconf_errors)

            # For non-extendable configs keep only slot-based storage.
            if not isinstance(self, ExtendableConfigTemplate):
                delattr(self, "__dict__")
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
            # NOTE: currently unused, kept for backwards compatibility.
            if key in self.__slots__:
                self.check(key, value)
            elif not key in self.additional_features:
                self.additional_features.append(key)

            ConfigTemplate.__setattr__(self, key, value)

        def __setitem__(self, key, value):
            orig_type = type(getattr(self, key))
            if not isinstance(value, orig_type):  # isinstance or is?
                raise ValueError(
                    f"Passing wrong argument of type {value.__class__.__name__}! "
                    f"Required: {orig_type}"
                )
            setattr(self, key, value)

        class_dict = {
            "__slots__": slots,
            "__init__": __init__,
            "__new__": __new__,
            "__getitem__": __getitem__,
            "__setitem__": __setitem__,
            "__setattr__": __setattr__,
        }
        return type(name, (cls.registered_configs[name],), class_dict)

    @classmethod
    def _get_annotation(cls, config_name, key):
        """Return type annotation for a field of a registered template.

        Parameters
        ----------
        config_name : str
            Name under which the template was registered (typically the
            same as the resulting config class name).
        key : str
            Field name to inspect.

        Returns
        -------
        Any
            Annotation object for the corresponding parameter in the
            template's ``__init__`` signature.
        """
        template_cls = cls.registered_configs[config_name]
        return signature(template_cls.__init__).parameters[key].annotation

    @staticmethod
    def __is_optional(annotation):
        """Check whether an annotation is explicitly optional.

        The method treats types of the form ``Union[..., None]`` and
        ``Literal[..., None]`` as optional.

        Parameters
        ----------
        annotation :
            Type annotation to inspect.

        Returns
        -------
        bool
            ``True`` if the annotation allows ``None``, ``False`` otherwise.
        """
        origin = get_origin(annotation)
        return ((origin is Union or origin is Literal) and 
           type(None) in get_args(annotation)) 

    @classmethod
    def _instantiate_default(cls, self: ConfigTemplate, config_name: str, k: str, v):
        """Resolve and instantiate default value for a field.

        This method implements several behaviours:

        * For :class:`LookUp` values, it tries to inherit from parent config
          (``self._parent``); otherwise uses the wrapped default.
        * For nested :class:`ConfigTemplate` subclasses, it recursively
          creates a new config instance using :meth:`from_template`.
        * For optional annotations (detected via :meth:`__is_optional`),
          returns ``None`` without further checks.
        * For non-config values, returns them as-is and asks caller to
          perform type checking.

        Parameters
        ----------
        self : ConfigTemplate
            Current config instance under construction.
        config_name : str
            Name of the config/template being instantiated.
        k : str
            Field name.
        v :
            Raw field value.

        Returns
        -------
        tuple[Any, bool]
            A pair ``(value, need_check)`` where:

            * ``value`` is the resolved value to assign to the field;
            * ``need_check`` indicates whether the caller should run
              :meth:`ConfigTemplate.check` on the value.

        Raises
        ------
        MisconfigurationError
            If the value cannot be instantiated or validated against the
            annotation and is not explicitly optional.
        """
        # check look-up
        if isinstance(v, LookUp):
            if self._parent:
                v = getattr(self._parent, k, None)
            else:
                v = v.value
        annotation = cls._get_annotation(config_name, k)
        is_union = get_origin(annotation) is Union
        if is_union:
            exceptions = []
            for arg in get_args(annotation):
                try:
                    x = cls._instantiate_default_one(self, arg, k, v)
                except Exception as exception:
                    exceptions.append(exception)
                else:
                    return x 
            raise MisconfigurationError(exceptions)
        else:
            return cls._instantiate_default_one(self, annotation, k, v, config_name)
                

    @classmethod
    def _instantiate_default_one(cls, self: ConfigTemplate, annotation, k: str, v, config_name: str = None):
        
        is_config = isclass(annotation) and issubclass(annotation, ConfigTemplate) 
        
        if v is not None and not is_config:
            return v, True
        # check if explicitly optional
        if cls.__is_optional(annotation):
            return None, False
        # if another config and not specified, run recursively
        if is_config and v is None:
            x = ConfigFactory.from_template(annotation(), annotation.get_default_name())(parent=self), False
            return x
        # specified
        if is_config and isinstance(v, tuple):
            x = ConfigFactory.from_template(v)(parent=self), False
            return x

        # try instantiate empty object
        x = ValueError(
            f"Misconfiguration! detected class `{annotation}`\n"
            f"has wrong argument passed at `{k}` and not explicitly optional.\n"
            f"Config: {config_name}"
        )
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
