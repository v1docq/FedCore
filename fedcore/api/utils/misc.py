import torch
from torch.ao.quantization.utils import _normalize_kwargs
from torch import Tensor
from torch.nn import Module
from functools import wraps


def camel_to_snake(camel_case_string):
            import re
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_case_string)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

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


def extract_fitted_operation(solver, max_depth: int = 5):
    """Extract fitted operation from solver using recursive search.

    Args:
        solver: Fedot solver instance or any object containing fitted_operation
        max_depth: Maximum depth for recursive search (default: 5)

    Returns:
        Fitted operation (usually a trained model)

    Raises:
        AttributeError: If fitted_operation cannot be found
    """

    def _recursive_search(obj, target_attr: str, depth: int = 0, visited=None):
        if visited is None:
            visited = set()

        obj_id = id(obj)
        if obj_id in visited or depth > max_depth:
            return None
        visited.add(obj_id)

        if hasattr(obj, target_attr):
            return getattr(obj, target_attr)

        priority_attrs = ['operator', 'root_node', 'operation']

        for attr_name in priority_attrs:
            if hasattr(obj, attr_name):
                nested_obj = getattr(obj, attr_name)
                if nested_obj is not None and not isinstance(nested_obj, (str, int, float, bool)):
                    result = _recursive_search(nested_obj, target_attr, depth + 1, visited)
                    if result is not None:
                        return result

        for attr_name in dir(obj):
            if (attr_name.startswith('_') or
                    attr_name in priority_attrs or
                    attr_name in ['__class__', '__dict__']):
                continue

            nested_obj = getattr(obj, attr_name)
            if (nested_obj is not None and
                    not isinstance(nested_obj, (str, int, float, bool, list, dict, tuple)) and
                    not callable(nested_obj)):
                result = _recursive_search(nested_obj, target_attr, depth + 1, visited)
                if result is not None:
                    return result

        return None

    return _recursive_search(solver, 'fitted_operation')