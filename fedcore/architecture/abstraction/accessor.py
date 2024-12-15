from functools import reduce

from torch.nn import Module


class Accessor:
    @classmethod
    def set_module(cls, m: Module, name: str, new: Module):
        if not name:
            return new
        *path, name = name.split('.')
        parent = reduce(getattr, path, m)
        setattr(parent, name, new)

    @classmethod
    def get_module(cls, m: Module, name: str):
        if not name:
            return m
        return reduce(getattr, name.split('.'), m)
