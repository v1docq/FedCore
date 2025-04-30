import torch
from torch.ao.quantization.utils import _normalize_kwargs
from torch import Tensor
from torch.nn import Module
from functools import wraps


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
        if  (len(args) == 1 and isinstance(args[0], dict) and not len(kwargs)):
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


def clear_device_cache(cls):
    if cls.device == torch.device('mps'):
        torch.mps.empty_cache()
        print(f'{cls.device} cache cleaned during {cls} execution')
    elif cls.device == torch.device('cuda'):
        torch.cuda.empty_cache()
        print(f'{cls.device} cache cleaned during {cls} execution')
    