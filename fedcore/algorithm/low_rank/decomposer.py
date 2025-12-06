"""
Wrapper module for tdecomp matrix decomposition API.
This module re-exports decomposers from tdecomp for backward compatibility.
"""

from enum import Enum
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
    'Decomposer',
]

class DecomposerType(Enum):
    SVD = SVDDecomposition
    RSVD = RandomizedSVD
    CUR = CURDecomposition
    TWO_SIDED = TwoSidedRandomSVD

    @staticmethod
    def map_from_str(decomposer_type: str) -> 'DecomposerType':
        if (decomposer_type == "svd"):
            return DecomposerType.SVD
        elif (decomposer_type == "rsvd"):
            return DecomposerType.RSVD
        elif (decomposer_type == "cur"):
            return DecomposerType.CUR
        elif (decomposer_type == "two_sided"):
            return DecomposerType.TWO_SIDED
        raise ValueError(f"Unknown decomposer_type: {decomposer_type}")