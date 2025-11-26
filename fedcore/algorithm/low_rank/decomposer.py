"""
Wrapper module for tdecomp matrix decomposition API.
This module re-exports decomposers from tdecomp for backward compatibility.
"""

from typing import Dict, Type
from tdecomp.matrix.decomposer import (
    SVDDecomposition,
    RandomizedSVD, 
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
]

DECOMPOSERS: Dict[str, Type[Decomposer]] = {
    'svd': SVDDecomposition,
    'rsvd': RandomizedSVD,
    'cur': CURDecomposition,
    'two_sided': TwoSidedRandomSVD,
}