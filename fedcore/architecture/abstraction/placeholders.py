from copy import deepcopy
from typing import Iterable

from torch.nn import Module


class PlaceHolder:
    __slots__: Iterable[str]

    def __init__(self, obj):
        for attr in self.__slots__:
            val_orig = getattr(obj, attr, None)
            if hasattr(val_orig, '__call__'):
                try:
                    val_ = deepcopy(val_orig())
                    val = lambda *args, **kwargs: val_
                except:
                    val = None
            else:
                val = deepcopy(val_orig)
            setattr(self, attr, val)
    
class ParameterPlaceHolder(PlaceHolder):
    __slots__ = ('shape', 'size', 'dtype', 'device')

    @staticmethod
    def set(m: Module, name: str, plhr: PlaceHolder):
        assert isinstance(m, Module), 'Only Torch Modules are supported!'
        d: dict = m.__dict__.get('_parameters', {})
        d.pop(name, None)
        setattr(m, name, plhr)
        d[name] = plhr