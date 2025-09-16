import torch

from functools import wraps, reduce, partial
from typing import *
from abc import ABC, abstractmethod


DIM_SUM_LIM = 1024
DIM_LIM = 2048

def _need_t(f):
    """Performs matrix transposition for maximal projection effect"""
    @wraps(f)
    def _wrapper(self: Decomposer, W: torch.Tensor, *args, **kwargs):
            m, n = W.size(-2), W.size(-1)
            _is_transposed = m >= n
            weight = W.t() if _is_transposed else W
            tns = f(self, weight, *args, **kwargs)
            return (
                tns if not _is_transposed
                else tuple(t.t() for t in reversed(tns))
            )
    return _wrapper

def _conditioning(f):
    """If conditioner is detected, apply W' = W @ C @ C^-1
    then U, S, Vh = Decompostion(W @ C)
    and Vh = Vh @ C^-1
    """
    @wraps(f)
    def _conditioned(self: "Decomposer", W: torch.Tensor, rank=None, conditioner=None, *args, **kwargs):
        if conditioner is None:
            conditioner = self._conditioner
        if conditioner is None:
            return f(self, W, rank, *args, **kwargs)
        if conditioner.ndim != 1:
            operation = torch.matmul
            inverse_operation = torch.linalg.pinv
        else: 
            operation = torch.mul
            inverse_operation = lambda x: 1 / x  
        W = operation(W, conditioner)
        inverse = inverse_operation(conditioner)
        *decomposition, Vh = f(self, W, rank, *args, **kwargs)
        Vh = operation(Vh, inverse)
        return *decomposition, Vh
    return _conditioned

class Decomposer(ABC):
    def __init__(self, rank: Union[int, float] = None, distortion_factor: float = 0.6, 
                 random_init: str = 'normal'):
        assert 0 < distortion_factor <= 1, 'distortion_factor must be in (0, 1]'
        self.distortion_factor = distortion_factor
        self.random_init = random_init
        self.rank = rank
        self._conditioner = None

    @_conditioning
    def decompose(self, W: torch.Tensor, rank=None, *args, **kwargs):
        if rank is None:
            rank = self.estimate_stable_rank(W)
        elif isinstance(rank, float):
            rank = max(1, int(rank * min(W.size())))
        if not self._is_big(W):
            return self._decompose(W, rank, *args, **kwargs)
        else:
            return self._decompose_big(W, rank, *args, **kwargs)
        
    def _is_big(self, W: torch.Tensor):
        return sum(W.size()) > DIM_SUM_LIM or any(d > DIM_LIM for d in W.size())
    
    def set_conditioner(self, conditioner):
        self._conditioner = conditioner
        
    @abstractmethod
    def _decompose(self, W, rank, *args, **kwargs):
        pass
    
    def _decompose_big(self, W, rank, *args, **kwargs):
        return self._decompose(W, rank, *args, **kwargs)
    
    def estimate_stable_rank(self, W):
        n_samples = max(W.shape)
        eps = self.distortion_factor
        min_num_samples = torch.ceil(4 * torch.log(torch.scalar_tensor(n_samples)) / (eps**2 / 2 - eps**3 / 3))
        return min(torch.round(min_num_samples), *W.size(), 1)
    
    def get_approximation_error(self, W, *result_matrices):
        approx = reduce(torch.matmul, result_matrices)
        return torch.linalg.norm(W - approx)
    
    def compose(self, *factors, **kwargs) -> torch.Tensor:
        nfactors = len(factors)
        if nfactors == 2:
            return factors[0] @ factors[1]
        elif nfactors == 3:
            U, S, Vh = factors
            return (U * S) @ Vh
        else:
            raise ValueError('Unknown type of decomposition!')
