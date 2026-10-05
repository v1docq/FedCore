"""
Wrapper module for tdecomp matrix decomposition API.
This module re-exports decomposers from tdecomp for backward compatibility.
"""

from typing import Dict, Type
from tdecomp.matrix.decomposer import (
    SVDDecomposition,
    RandomizedSVD as TdecompRandomizedSVD,
    TwoSidedRandomSVD,
    CURDecomposition,
    DECOMPOSERS as TDECOMP_DECOMPOSERS
)
from tdecomp._base import Decomposer

__all__ = [
    'SVDDecomposition',
    'RandomizedSVD',
    'TwoSidedRandomSVD',
    'CURDecomposition',
    'DECOMPOSERS',
    'Decomposer',
    'DecomposerType',
]

import torch
from enum import Enum
from functools import partial


class RandomizedSVD(TdecompRandomizedSVD):
    """Corrected range finder for tdecomp 0.2.18's rectangular projection defect.

    Uses matrix power iterations with QR stabilization. With no explicit rank,
    requests the full rank to preserve the historical layer recreation contract.
    Explicit rank is an approximation of the supplied matrix.
    """
    def decompose(self, tensor, rank=None, *args, **kwargs):
        if tensor.ndim != 2 or not torch.isfinite(tensor).all():
            raise ValueError('Randomized SVD requires a finite matrix')
        requested = self.rank if rank is None else rank
        maximum = min(tensor.shape)
        if requested is None:
            requested = maximum
        if isinstance(requested, bool):
            raise ValueError('rank must be an integer or fraction')
        if isinstance(requested, float):
            if not 0 < requested <= 1:raise ValueError('rank fraction must be in (0,1]')
            requested = max(1,int(requested*maximum))
        if not isinstance(requested,int) or requested < 1:
            raise ValueError('rank must be positive')
        requested = min(requested, maximum)
        samples = min(maximum, requested + 8)
        if self.random_init == 'normal':
            omega = torch.randn(tensor.shape[1],samples,device=tensor.device,dtype=tensor.dtype)
        elif self.random_init == 'uniform':
            omega = torch.rand(tensor.shape[1],samples,device=tensor.device,dtype=tensor.dtype)*2-1
        else:
            raise ValueError('Randomized SVD supports normal or uniform initialization')
        q, _ = torch.linalg.qr(tensor @ omega, mode='reduced')
        for _ in range(self.power):
            z, _ = torch.linalg.qr(tensor.mH @ q, mode='reduced')
            q, _ = torch.linalg.qr(tensor @ z, mode='reduced')
        u, s, vh = torch.linalg.svd(q.mH @ tensor, full_matrices=False)
        return (q @ u)[:,:requested], s[:requested], vh[:requested,:]


DECOMPOSERS: Dict[str, Type[Decomposer]] = {
    'svd': SVDDecomposition,
    'rsvd': RandomizedSVD,
    'cur': CURDecomposition,
    'two_sided': TwoSidedRandomSVD,
}


class DecomposerType(Enum):
    SVD = partial(SVDDecomposition)
    RSVD = partial(RandomizedSVD)
    CUR = partial(CURDecomposition)
    TWO_SIDED = partial(TwoSidedRandomSVD)
