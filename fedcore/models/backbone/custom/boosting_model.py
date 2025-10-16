import ctypes
import inspect
import logging

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
                 history_period: int = 10,
                 derivative_threshold: float = 0.01):
        self.randomization_params = dict(randomization_params or {})
        self.solver = FEDCORE_SAMPLING_METHODS[randomization_method]
        self.history_period = int(history_period)
        self.derivative_threshold = derivative_threshold
        self.logger = logging.getLogger(self.__class__.__name__)

        self.use_approximation = False
        self._hist_grad, self._hist_hess = None, None
        self._current_iteration = 0

    def before_train(self, build_info):
        self.use_approximation = False
        self._hist_grad, self._hist_hess = None, None
        self._current_iteration = 0

    def before_iteration(self, build_info):
        self._current_iteration = build_info['num_iter'] + 1
        train = build_info['data']['train']
        grad: cp.ndarray = train.get('grad')
        hess: cp.ndarray = train.get('hess')

        # accumulate to history when grads and hesses are available
        if grad is not None and hess is not None:
            self._update_history(grad, hess)
            # check if we should apply approximation based on history
            self.use_approximation = self._scheduler()

    def _update_history(self, grad: cp.ndarray, hess: cp.ndarray):
        if self._hist_grad is None or len(self._hist_grad) < self.history_period:
            self._hist_grad = cp.stack([grad.copy()] * self.history_period)
            self._hist_hess = cp.stack([hess.copy()] * self.history_period)
        else:
            # sliding window: roll and replace the oldest grad and hess
            self._hist_grad = cp.roll(self._hist_grad, -1, axis=0)
            self._hist_hess = cp.roll(self._hist_hess, -1, axis=0)
            self._hist_grad[-1] = grad.copy()
            self._hist_hess[-1] = hess.copy()

    def _gaussian_smooth(self, data: cp.ndarray, sigma: float = 1.0) -> cp.ndarray:
        if len(data) < 3:
            return data

        kernel_size = min(5, len(data))
        x = cp.arange(kernel_size) - (kernel_size - 1) // 2
        kernel = cp.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / cp.sum(kernel)
        smoothed = cp.convolve(data, kernel, mode='same')
        return smoothed

    def _scheduler(self) -> bool:
        # TODO: rewrite as dynamic observers
        if self._hist_grad is None or self._current_iteration < self.history_period:
            return False

        threshold = self.derivative_threshold
        grad_norms = cp.linalg.norm(self._hist_grad, axis=1)
        derivative = cp.gradient(self._gaussian_smooth(grad_norms))

        recent_deriv = cp.abs(derivative[-1])
        # also check if the average of recent derivatives is close to zero
        recent_window = min(3, len(derivative))
        avg_recent_deriv = cp.mean(cp.abs(derivative[-recent_window:]))

        self.logger.debug(f'recent_deriv: {recent_deriv}, avg_recent_deriv: {avg_recent_deriv}')
        return (recent_deriv < threshold and avg_recent_deriv < threshold)

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
                except Exception:
                    pass
                finally:
                    # unset this flag after approximation
                    self.use_approximation = False
                    break

        grad_approx = grad
        hess_approx = hess
        return grad_approx, hess_approx

    def __call__(self, grad: cp.ndarray, hess: cp.ndarray):
        if self.use_approximation:
            return self.perform_historic_approximation(grad, hess)
        return grad, hess
