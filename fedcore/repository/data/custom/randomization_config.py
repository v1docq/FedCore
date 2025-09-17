from tdecomp.matrix.decomposer import SVDDecomposition, RandomizedSVD, TwoSidedRandomSVD, CURDecomposition
from tdecomp.matrix.random_projections import RANDOM_GENS

FEDCORE_RANDOM_MATRIX_TYPES = list(RANDOM_GENS.keys())

FEDCORE_SAMPLING_METHODS = dict(svd=SVDDecomposition,
                                random_svd=RandomizedSVD,
                                # need to define type of random matrix from FEDCORE_RANDOM_MATRIX_TYPES
                                twosided_random_svd=TwoSidedRandomSVD,
                                # need to define type of random matrix from FEDCORE_RANDOM_MATRIX_TYPES
                                cur=CURDecomposition)