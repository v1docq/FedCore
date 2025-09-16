import cupy as cp
from py_boost.multioutput.sketching import *
from tdecomp.matrix.decomposer import SVDDecomposition, RandomizedSVD, TwoSidedRandomSVD, CURDecomposition
from tdecomp.matrix.random_projections import RANDOM_GENS

FEDCORE_RANDOM_MATRIX_TYPES = list(RANDOM_GENS.keys())

FEDCORE_SAMPLING_METHODS = dict(svd=SVDDecomposition,
                                random_svd=RandomizedSVD,
                                # need to define type of random matrix from FEDCORE_RANDOM_MATRIX_TYPES
                                twosided_random_svd=TwoSidedRandomSVD,
                                # need to define type of random matrix from FEDCORE_RANDOM_MATRIX_TYPES
                                cur=CURDecomposition)

PYBOOST_SAMPLING_METHODS = dict(filter=FilterSketch,  #
                                svd=SVDSketch,  # using approximation to truncate hessian and grad dims
                                random_sampling=RandomSamplingSketch,  # using "probabilities" obtain from columns
                                random_projection=RandomProjectionSketch
                                # using random matrix from Normal distribution/ Radamacher distribution
                                )


class FedcoreRandApproximation(GradSketch):
    """Sketch using Fedcore strategies"""

    def __init__(self, randomization_method: str = 'random_svd',
                 randomization_params: dict = {}):
        """

        Args:
            sample: int, subsample to speed up SVD fitting
            **svd_params: dict, SVD params, see cuml.TruncatedSVD docs
        """

        self.randomization_method = randomization_method
        self.randomization_params = randomization_params
        self.solver = FEDCORE_SAMPLING_METHODS[randomization_method]

    def _define_approximation_regime(self, tensor):
        max_num_rows = 10000
        is_matrix_big = any([tensor.shape[0] > max_num_rows, tensor.shape[1] > max_num_rows])
        if is_matrix_big:
            self.approximation = True
        else:
            self.approximation = False

    def _scheduler(self):
        self.counter += 1
        if self.counter % 20 == 0:
            return True
        else:
            return False

    def approximate(self, grad, hess):
        grad_approx = cp.asarray(self.solver.rsvd(tensor=grad.get(), approximation=self.approximation,
                                                  reg_type=self.regularisation, regularized_rank=self.rank,
                                                  return_svd=False,
                                                  sampling_regime='column_sampling').astype('float32'))
        if hess.shape[1] > 1:
            hess = cp.asarray(self.solver.rsvd(tensor=hess.get(), approximation=self.approximation,
                                               reg_type=self.regularisation, regularized_rank=self.rank,
                                               return_svd=False))
            hess_approx = cp.clip(hess, 0.01, None)
        else:
            hess_approx = hess
            # hess_approx = self.solver.random_projection @ hess.get()
            # hess_approx = cp.asarray(hess_approx.astype('float32'))
        self.rank = self.solver.regularized_rank
        return grad_approx, hess_approx

    def __call__(self, grad: cp.ndarray, hess: cp.ndarray):
        self._define_approximation_regime(grad)
        if self._scheduler():
            return self.apply_sketch(grad, hess)
        else:
            return grad, hess
