import torch
import torch.nn
import torchvision.datasets
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt
from fedcore.tools.ruler import PerformanceEvaluator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DecomposedWeight(torch.Tensor):
    def __init__(self, base):
        self.base = base
        self.decompose()

    def __rmatmul__(self, other):
        x = other
        for matrix in self.matrices:
            x = x @ matrix
        print('after sigma')
        return x
    
    def __matmul__(self, other):
        x = self.matrices[0] @ self.matrices[-1]
        print('not inverse')
        return x @ other

    def decompose(self):
        print('DECOMP')
        U, S, Vh = torch.linalg.svd(self.base, full_matrices=False)
        self.U, self.S, self.Vh = U, S, Vh
        self.U *= self.S
        self.U.requires_grad = True
        self.Vh.requires_grad = True
        setattr(self, 'data', torch.nn.ParameterList([self.U, self.Vh]))
        setattr(self, 'matrices', [U, Vh])
        print(self.matrices)
    
    def __getitem__(self, key):
        x = self.matrices[0][key]
        return x @ self.matrices[-1]
    
    def __repr__(self):
        return 'pass'

class Decomposed:
    _EIGEN_METHODS = {
        '__init__',
        '_substitute_weight',
        '__getattribute__',
        '__setattr__'
    }
    def __init__(self, base: nn.Linear):
        self.base = base
        self._substitute_weight()
    
    def _substitute_weight(self):
        W = self.base.weight.data
        self.base.register_parameter('weight', None)
        # setattr(self.base, 'matrices', (DecomposedWeight(W)))
        setattr(self.base, 'weight', torch.nn.Parameter(DecomposedWeight(W)))
        # print(self.base.weight if hasattr(self.base, 'weight') else 'NO')

    def __repr__(self):
        return 'Decomposed ' + repr(self.base)
    
    def __getattr__(self, attr):
        print('getattribute')
        if not attr in self._EIGEN_METHODS:
            return self.base.__getattr__(attr)
        else:
            return self.__getattr__(attr)

    # def __setattr__(self, attr, val):
    #     print('setattribute')
    #     if not attr in self._EIGEN_METHODS:
    #         self.base.__setattr__(attr, val)
    #     else:
    #         super().__setattr__(attr, val)
