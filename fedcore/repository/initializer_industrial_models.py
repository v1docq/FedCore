import pathlib
import fedot.core.repository.tasks as fedot_task
import fedot.core.repository.metrics_repository as metrics_repository
from fedot.api.api_utils.api_params_repository import ApiParamsRepository
from fedot.core.repository.pipeline_operation_repository import PipelineOperationRepository
from fedot.api.api_utils.assumptions.assumptions_handler import AssumptionsHandler
from fedot.core.data.merge.data_merger import DataMerger, ImageDataMerger
from fedot.core.operations.operation import Operation
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.optimisers.objective.data_objective_eval import PipelineObjectiveEvaluate
from fedot.api.api_utils.api_composer import ApiComposer
import fedot.utilities.define_metric_by_task as define_metric_by_task
from fedot.api.main import Fedot
from fedot.core.composer.gp_composer.gp_composer import GPComposer
import fedot.core.pipelines.verification as pipeline_verification
import fedot.core.pipelines.verification_rules as verification_rules
from fedcore.architecture.utils.paths import PROJECT_PATH
from fedcore.interfaces.search_space import get_fedcore_search_space
from fedcore.repository.fedcore_impl.abstract import (
    _fit_assumption_and_check_correctness,
    TaskCompression,
    _merge,
    _fit,
    predict_operation_fedcore,
    fedcore_preprocess_predicts,
    merge_fedcore_predicts,
    _get_default_fedcore_mutations, obtain_model_fedcore, divide_operations_fedcore, fit_fedcore,
    restore_pipeline_fedcore,
)
from fedcore.repository.fedcore_impl.data import (
    build_holdout_producer,
    build_fedcore_dataproducer,
)
from fedot.core.repository.metrics_repository import MetricsRepository
from fedcore.repository.fedcore_impl.metrics import MetricsRepository as FedcoreMetric, evaluate_objective_fedcore, MetricByTask as FedcoreMetricByTask

# FEDCORE_METRIC_REPO = FedcoreMetric()

_DEFAULT_ROOT_VALIDATOR = verification_rules.has_final_operation_as_model


def _has_fedcore_root_or_model(pipeline):
    operation = pipeline.root_node.operation.operation_type.split('/')[0]
    if operation in {'training_model', 'low_rank_model', 'pruning_model',
                     'quantization_model', 'distilation_model', 'lora_model'}:
        return
    return _DEFAULT_ROOT_VALIDATOR(pipeline)

FEDOT_METHOD_TO_REPLACE = {
    (verification_rules, 'has_final_operation_as_model'): _has_fedcore_root_or_model,
    (pipeline_verification, 'has_final_operation_as_model'): _has_fedcore_root_or_model,
    (pipeline_verification, 'common_rules'): [
        _has_fedcore_root_or_model if rule is _DEFAULT_ROOT_VALIDATOR else rule
        for rule in pipeline_verification.common_rules],
    #(Fedot, 'fit'),
    #(GPComposer, '_convert_opt_results_to_pipeline'),
    (PipelineObjectiveEvaluate, 'evaluate'): evaluate_objective_fedcore,
    (fedot_task, "TaskTypesEnum"): TaskCompression,
    (DataSourceSplitter, "build"): build_fedcore_dataproducer,

    (metrics_repository, "MetricsRepository"): FedcoreMetric,
    (metrics_repository.MetricsRepository, 'get_metric'): FedcoreMetric.get_metric,
    (metrics_repository.MetricsRepository, 'get_metric_class'): FedcoreMetric.get_metric_class,

    (define_metric_by_task, "MetricByTask"): FedcoreMetricByTask,
    (define_metric_by_task.MetricByTask, "compute_default_metric"): FedcoreMetricByTask.compute_default_metric,
    (define_metric_by_task.MetricByTask, "get_default_quality_metrics"): FedcoreMetricByTask.get_default_quality_metrics,

    (ApiParamsRepository, "_get_default_mutations"): _get_default_fedcore_mutations,
    (PipelineSearchSpace, "get_parameters_dict"): get_fedcore_search_space,
    (AssumptionsHandler, "fit_assumption_and_check_correctness"): _fit_assumption_and_check_correctness,
    (DataSourceSplitter, "_build_holdout_producer"): build_holdout_producer,
    (DataMerger, "merge"): _merge,
    (Operation, "fit"): _fit,
    (Operation, "_predict"): predict_operation_fedcore,
    (ImageDataMerger, "preprocess_predicts"): fedcore_preprocess_predicts,
    (ImageDataMerger, "merge_predicts"): merge_fedcore_predicts,
    (ApiComposer, 'obtain_model'): obtain_model_fedcore,
    (PipelineOperationRepository, 'divide_operations'): divide_operations_fedcore
}


# Descriptors are captured before any explicit adaptation, preserving static methods.
from copy import deepcopy
from inspect import getattr_static
from threading import RLock, get_ident

DEFAULT_METHODS = [getattr_static(owner, name) for owner, name in FEDOT_METHOD_TO_REPLACE]


class FedcoreModels:
    """Explicit reversible FEDOT adaptation. Nested scopes restore their exact parent state."""
    _lock = RLock()
    _stack = []

    def __init__(self):
        from importlib.resources import files
        resources = files('fedcore.repository.data')
        self.fedcore_data_operation_path = str(resources.joinpath('compression_data_operation_repository.json'))
        self.fedcore_model_path = str(resources.joinpath('compression_model_repository.json'))
        self._frames = []

    def _replace_operation(self, to_fedcore=True):
        for ((owner, name), replacement), original in zip(FEDOT_METHOD_TO_REPLACE.items(), DEFAULT_METHODS):
            setattr(owner, name, replacement if to_fedcore else original)

    def setup_repository(self):
        # FEDOT adaptation changes process-global objects. Hold the reentrant lock
        # for the entire scope so other FedCore scopes cannot interleave it.
        self._lock.acquire()
        try:
            methods = {(owner, name): getattr_static(owner, name) for owner, name in FEDOT_METHOD_TO_REPLACE}
            repositories = deepcopy(OperationTypesRepository.__repository_dict__)
            initialized = dict(OperationTypesRepository.__initialized_repositories__)
            frame = (self, methods, repositories, initialized, get_ident())
            self._stack.append(frame)
            self._frames.append(frame)
        except BaseException:
            self._lock.release()
            raise
        try:
            for kind, path in [('data_operation', self.fedcore_data_operation_path),
                               ('model', self.fedcore_model_path)]:
                OperationTypesRepository.__repository_dict__[kind] = {
                    'file': path, 'initialized_repo': None, 'default_tags': []}
                OperationTypesRepository.assign_repo(kind, path)
            self._replace_operation()
        except BaseException:
            self.setup_default_repository()
            raise
        return OperationTypesRepository

    def setup_default_repository(self):
        if not self._frames:
            return OperationTypesRepository
        frame = self._frames[-1]
        if frame[4] != get_ident():
            raise RuntimeError('FEDOT adaptation must be restored by its installing thread')
        if not self._stack or self._stack[-1] is not frame:
            raise RuntimeError('FEDOT adaptation scopes must close in reverse installation order')
        try:
            for (owner, name), original in frame[1].items():
                setattr(owner, name, original)
            OperationTypesRepository.__repository_dict__.clear()
            OperationTypesRepository.__repository_dict__.update(frame[2])
            OperationTypesRepository.__initialized_repositories__.clear()
            OperationTypesRepository.__initialized_repositories__.update(frame[3])
            self._stack.pop()
            self._frames.pop()
        finally:
            self._lock.release()
        return OperationTypesRepository

    def __enter__(self):
        self.setup_repository()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.setup_default_repository()
        return False
