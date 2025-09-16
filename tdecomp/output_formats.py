import abc
import torch
from functools import reduce

# from tensorly.... mode_mmul


class _IDecompositionResult(abc.ABC):
    @abc.abstractmethod
    def compose(cls, *tensors):
        pass 

    def __init__(self, tensors):
        if isinstance(tensors, dict):
            self.tensors = tensors
        elif isinstance(tensors, (tuple, list)):
            self.tensors = dict(enumerate(tensors))


class LinearDecomposition(_IDecompositionResult):
    @classmethod
    def compose(cls, *tensors):
        return reduce(torch.matmul, tensors)


class ModalDecomposition(_IDecompositionResult):
    @classmethod
    def compose(cls, *tensors):
        pass
