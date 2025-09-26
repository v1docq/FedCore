from py_boost import GradientBoosting
import cupy as cp
from py_boost.callbacks.callback import Callback
from py_boost.gpu.tree import DepthwiseTreeBuilder
from py_boost.multioutput.sketching import FilterSketch, SVDSketch, RandomSamplingSketch, RandomProjectionSketch, \
    GradSketch

from fedcore.repository.data.custom.randomization_config import FEDCORE_SAMPLING_METHODS


class FedcoreBoostingModel(GradientBoosting):

    def __init__(self, **kwargs):
        self.industrial_strategy = kwargs.get('industrial_strategy', {})
        if len(self.industrial_strategy) != 0:
            del kwargs['industrial_strategy']
        super().__init__(**kwargs)
        self._init_sketch_method()
        self.params['use_hess'] = self.use_hess
        self.params['multioutput_sketch'] = self.sketch_method

    def _fit(self, builder: DepthwiseTreeBuilder, build_info: dict) -> None:
        history_cb = FedcoreGradHessHistory()
        try:
            from py_boost.callbacks.callback import CallbackPipeline
            existing = []
            if hasattr(self, 'callbacks') and getattr(self, 'callbacks') is not None:
                existing = list(self.callbacks.callbacks)
            self.callbacks = CallbackPipeline(history_cb, *existing)
        except Exception:
            pass
        finally:
            super()._fit(builder, build_info)

    def _init_sketch_method(self):
        self.sketch_params = self.industrial_strategy.get('sketch_params', {})
        self.sketch_outputs = self.industrial_strategy.get('sketch_outputs', 1)
        self.use_hess = self.industrial_strategy.get('use_hess', False)
        self.sketch_method = self.industrial_strategy.get('sketch_method', 'filter')
        self.history_period = int(self.industrial_strategy.get('history_period', 10))
        self.sketch_method_dict = dict(filter=FilterSketch,
                                       svd=SVDSketch,
                                       random_sampling=RandomSamplingSketch,
                                       random_projection=RandomProjectionSketch,
                                       #random_svd=RandomSVD,
                                       )
        self.sketch_method = self.sketch_method_dict[self.sketch_method](self.sketch_outputs,
                                                                         **self.sketch_params)
        self.multioutput_sketch = FedcoreRandApproximation(randomization_params=self.sketch_params,
                                                           history_period=self.history_period)

    def __repr__(self):
        return self.__class__.__name__


class FedcoreGradHessHistory(Callback):
    """Callback to accumulate gradients and hessians history during boosting."""

    def __init__(self):
        self.grad_snapshots = []
        self.hess_snapshots = []

    def before_train(self, build_info):
        self.grad_snapshots = []
        self.hess_snapshots = []

        build_info['fedcore_history'] = {
            'callback': self
        }
        model = build_info.get('model', None)
        if model is not None:
            setattr(model, '_fedcore_history_cb', self)

    def before_iteration(self, build_info):
        train = build_info['data']['train']
        grad: cp.ndarray = train.get('grad')
        hess: cp.ndarray = train.get('hess')

        if grad is None or hess is None:
            return

        it = build_info.get('num_iter', 0)
        model = build_info.get('model', None)
        if model is not None:
            setattr(model, '_fedcore_iter', it)
        self.grad_snapshots.append(grad.copy())
        self.hess_snapshots.append(hess.copy())

        model = build_info.get('model', None)
        period = int(getattr(model, 'history_period', 1)) if model is not None else 1

        if (it % period) == 0:
            if len(self.grad_snapshots) > 0:
                build_info['last_grad_history'] = cp.concatenate(self.grad_snapshots, axis=0)
                build_info['last_hess_history'] = cp.concatenate(self.hess_snapshots, axis=0)
            self.grad_snapshots = []
            self.hess_snapshots = []


class FedcoreRandApproximation(GradSketch):
    """Sketch using Fedcore strategies"""

    def __init__(self,
                 randomization_method: str = 'random_svd',
                 randomization_params: dict = {},
                 history_period: int = 10):
        self.randomization_method = randomization_method
        self.randomization_params = randomization_params
        self.solver = FEDCORE_SAMPLING_METHODS[randomization_method]
        self.history_period = history_period

    def _define_approximation_regime(self, tensor):
        max_num_rows = 10000
        is_matrix_big = any([tensor.shape[0] > max_num_rows, tensor.shape[1] > max_num_rows])
        if is_matrix_big:
            self.approximation = True
        else:
            self.approximation = False

    def _scheduler(self):
        self.counter += 1
        return (self.counter % self.history_period) == 0

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
