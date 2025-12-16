"""
Wrapper module for tdecomp matrix decomposition API.
This module re-exports decomposers from tdecomp for backward compatibility.
"""

from enum import Enum
from tdecomp.matrix.decomposer import (
    SVDDecomposition,
    RandomizedSVD, 
    TwoSidedRandomSVD,
    CURDecomposition,
)
from tdecomp.matrix.decomposer import DECOMPOSERS
import inspect
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
    
def _get_all_decomposer_params() -> set:
    """Dynamically extract all unique parameters from all decomposer classes.

    Returns:
        set: Set of all parameter names across all decomposer implementations
    """
    all_params = set()
    for decomposer_cls in DECOMPOSERS.values():
        sig = inspect.signature(decomposer_cls.__init__)
        all_params.update(sig.parameters.keys())
    all_params.discard('self')
    return all_params


DECOMPOSER_PARAMS = _get_all_decomposer_params()

def extract_decomposer_params(decomposer_type: DecomposerType, params: dict) -> dict:
    """Extract decomposer parameters from operation parameters.

    Args:
        params: Operation parameters from config

    Returns:
        dict: Parameters for tdecomp decomposer (rank, distortion_factor, etc.)
    """
    filtered_params = {
        key: value for key, value in params.items()
        if key in DECOMPOSER_PARAMS
    }

    if 'rank' not in filtered_params:
        filtered_params['rank'] = None

    if decomposer_type == DecomposerType.SVD:
        filtered_params.pop('power', None)
        filtered_params.pop('random_init', None)
    elif decomposer_type == DecomposerType.TWO_SIDED:
        filtered_params.pop('power', None)
    elif decomposer_type == DecomposerType.CUR:
        filtered_params.pop('power', None)
        filtered_params.pop('random_init', None)

    return filtered_params

def init_decomposer_from_whole_params(params: dict) -> Decomposer:
    """Inits decomposer from params (dict), filtering params that doesn't applyable for that decomposer
    """
    decomposer_type = DecomposerType.map_from_str(params.get('decomposer', 'svd'))
    correct_params = extract_decomposer_params(decomposer_type, params)
    return decomposer_type.value(**correct_params)