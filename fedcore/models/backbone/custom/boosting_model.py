from py_boost.multioutput.sketching import *
from py_boost import GradientBoosting
import cupy as cp

from fedcore.repository.data.custom.randomization_config import FEDCORE_SAMPLING_METHODS


class FedcoreBoostingModel(GradientBoosting):
    """

    """

    def __init__(self, **kwargs):
        self.industrial_strategy = kwargs['industrial_strategy '] if 'industrial_strategy ' in kwargs.keys() else {}
        if len(self.industrial_strategy) != 0:
            del kwargs['industrial_strategy']
        super().__init__(**kwargs)
        self._init_sketch_method()
        self.params['use_hess'] = self.use_hess
        self.params['multioutput_sketch'] = self.sketch_method

    def _init_sketch_method(self):
        self.sketch_params = self.industrial_strategy.get('sketch_params', {})
        self.sketch_outputs = self.industrial_strategy.get('sketch_outputs', 1)
        self.use_hess = self.industrial_strategy.get('use_hess', False)
        self.sketch_method = self.industrial_strategy.get('sketch_method', 'random_svd')
        self.sketch_method_dict = dict(filter=FilterSketch,
                                       svd=SVDSketch,
                                       random_sampling=RandomSamplingSketch,
                                       random_projection=RandomProjectionSketch,
                                       #random_svd=RandomSVD,
                                       )
        self.sketch_method = self.sketch_method_dict[self.sketch_method](self.sketch_outputs,
                                                                         **self.sketch_params)

    def __repr__(self):
        return "FedcoreBoosting"
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