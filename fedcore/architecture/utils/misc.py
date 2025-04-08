from enum import Enum
from functools import wraps
from typing import Optional, Any

from torch.ao.quantization.utils import _normalize_kwargs
from torch import Tensor
from torch.nn import Module


def default_value(val: Optional[Any], default_val: Any) -> Any:
    """Safely returns `val` or `default_val` if `val` is None."""
    return val if val is not None else default_val

def filter_kw(f):
    """Allows to pass kwargs with extra key-value pairs without raising an exception"""

    @wraps(f)
    def _wrapping(self, *args, **kwargs):
        new_kw = _normalize_kwargs(f, kwargs)
        f(self, *args, **new_kw)

    return _wrapping


def filter_params(f):
    """Allows to combine standard __init__ signatures with fedot-style `params`"""

    @wraps(f)
    def _wrapping(self, params={}):
        new_kw = _normalize_kwargs(f, params)
        f(self, **new_kw)

    return _wrapping


def filter_kw_universal(f):
    """Automatically switches between fedot-style and conventional init"""

    @wraps(f)
    def _wrapping(self, *args, **kwargs):
        if (len(args) == 1 and isinstance(args[0], dict) and not len(kwargs)):
            params = args[0]
            args = args[1:]
        elif 'params' in kwargs and len(kwargs) == 1:
            params = kwargs['params']
        else:
            params = kwargs
        new_kw = _normalize_kwargs(f, params)
        f(self, *args, **new_kw)

    return _wrapping


def _contiguous(t: Tensor):
    return t if t.is_contiguous() else t.contiguous()


def count_params(m: Module):
    c = 0
    for p in m.parameters():
        c += p.numel()
    return c


class EnumNoValue:
    def __init__(self, base: Enum):
        assert issubclass(base, Enum), 'Only Enums are supported'
        self.base = base

    def __getattr__(self, name):
        val = getattr(self.base, name)
        if isinstance(val, Enum):
            return val.value
        return val

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return name in self.base._member_map_
