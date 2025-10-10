import ctypes
import inspect

from py_boost import GradientBoosting
import cupy as cp
from py_boost.gpu.tree import DepthwiseTreeBuilder
from py_boost.multioutput.sketching import FilterSketch, GradSketch

from fedcore.repository.data.custom.boosting_config import PYBOOST_SAMPLING_METHODS
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
        # from py_boost.callbacks.callback import CallbackPipeline
        # try:
        #     if hasattr(self, 'callbacks') and getattr(self, 'callbacks') is not None:
        #         existing = list(self.callbacks.callbacks)
        #         existing.append(self.sketch_method)
        #         self.callbacks = CallbackPipeline(*existing)
        # except Exception:
        #     pass
        # finally:
        super()._fit(builder, build_info)

    def _init_sketch_method(self):
        self.sketch_params = self.industrial_strategy.get('sketch_params', {})
        self.use_hess = self.industrial_strategy.get('use_hess', False)
        self.history_period = int(self.industrial_strategy.get('history_period', 10))
        method_key = self.industrial_strategy.get('sketch_method', 'fedcore')
        sampling_key = self.industrial_strategy.get('sampling_method', 'random_svd')

        if method_key == 'fedcore':
            self.sketch_method = FedcoreGradHessHistory(randomization_method=sampling_key,
                                                        history_period=self.history_period,
                                                        **self.sketch_params)
        else:
            self.sketch_method = PYBOOST_SAMPLING_METHODS.get(method_key, FilterSketch)(**self.sketch_params)

    def __repr__(self):
        return self.__class__.__name__


class FedcoreGradHessHistory(GradSketch):
    """Callback that accumulates grads/hess, schedules and applies Fedcore approximation."""

    def __init__(self,
                 randomization_method: str = 'random_svd',
                 randomization_params: dict = {},
                 history_period: int = 10):
        self.randomization_params = dict(randomization_params or {})
        self.solver = FEDCORE_SAMPLING_METHODS[randomization_method]
        self.history_period = int(history_period)

        self.approximation = False
        self._collect_history = False
        self._buf_grad, self._buf_hess = [], []
        self._hist_grad, self._hist_hess = None, None

    def before_train(self, build_info):
        self._collect_history = False
        self._buf_grad, self._buf_hess = [], []

    def before_iteration(self, build_info):
        train = build_info['data']['train']
        counter = build_info['num_iter'] + 1
        grad: cp.ndarray = train.get('grad')
        hess: cp.ndarray = train.get('hess')
        if grad is not None and hess is not None:
            self._buf_grad.append(grad.copy())
            self._buf_hess.append(hess.copy())

        # schedule decision for this iteration
        self._collect_history = self._scheduler(counter)
        if self._collect_history:
            try:
                self._hist_grad = cp.stack(self._buf_grad, axis=0)
                self._hist_hess = cp.stack(self._buf_hess, axis=0)
            except Exception:
                self._hist_grad, self._hist_hess = self._buf_grad[-1], self._buf_hess[-1]
            self._buf_grad, self._buf_hess = [], []  # clear buffers for next window
        else:
            self._hist_grad, self._hist_hess = None, None

    def _scheduler(self, counter: int) -> bool:
        if self.history_period <= 0:
            return False
        return (counter % self.history_period) == 0

    def get_indexers_from_decomposition(self, tensor: cp.ndarray, top_fraction: float):
        U, s, Vh = cp.linalg.svd(tensor, full_matrices=False)
        s_diag_root = cp.diag(cp.sqrt(s))

        row_norms = cp.linalg.norm(U @ s_diag_root, axis=1)
        k_row = max(1, int(len(row_norms) * top_fraction))
        row_indexer = cp.sort(cp.argsort(row_norms)[-k_row:]).astype(cp.uint64)

        col_norms = cp.linalg.norm(s_diag_root @ Vh, axis=0)
        k_col = max(1, int(len(col_norms) * top_fraction))
        col_indexer = cp.sort(cp.argsort(col_norms)[-k_col:]).astype(cp.uint64)

        return row_indexer, col_indexer

    def perform_historic_approximation(self, grad, hess):

        row_indexer, col_indexer = self.get_indexers_from_decomposition(grad, top_fraction=0.5)

        stack = inspect.stack()
        target_method = 'build_tree'

        for frame_info in stack[1:]:

            if frame_info.function == target_method:
                frame = frame_info.frame
                try:
                    frame.f_locals['row_indexer'] = row_indexer
                    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(1))
                    frame.f_locals['col_indexer'] = col_indexer
                    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(1))
                except Exception as _:
                    pass
                break

        grad_approx = grad
        hess_approx = hess
        return grad_approx, hess_approx

    def __call__(self, grad: cp.ndarray, hess: cp.ndarray):
        if self._collect_history:
            return self.perform_historic_approximation(grad, hess)
        return grad, hess
