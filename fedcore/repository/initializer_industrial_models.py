import pathlib
import fedot.core.repository.tasks as fedot_task
from fedot.core.repository.metrics_repository import (
    MetricsRepository as fedot_metric_repo,
)
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
from fedot.api.main import Fedot
from fedot.core.composer.gp_composer.gp_composer import GPComposer
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
from fedcore.repository.fedcore_impl.metrics import MetricsRepository as FedcoreMetric, evaluate_objective_fedcore

FEDCORE_METRIC_REPO = FedcoreMetric()

FEDOT_METHOD_TO_REPLACE = {
    #(Fedot, 'fit'),
    #(GPComposer, '_convert_opt_results_to_pipeline'),
    (PipelineObjectiveEvaluate, 'evaluate'): evaluate_objective_fedcore,
    (fedot_task, "TaskTypesEnum"): TaskCompression,
    (DataSourceSplitter, "build"): build_fedcore_dataproducer,
    (fedot_metric_repo, "_metrics_implementations"): FedcoreMetric._metrics_implementations,
    (fedot_metric_repo, "_metrics_classes"): FedcoreMetric._metrics_classes,
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

DEFAULT_METHODS = [
    getattr(class_impl[0], class_impl[1]) for class_impl in FEDOT_METHOD_TO_REPLACE
]


class FedcoreModels:
    def __init__(self):
        self.fedcore_data_operation_path = pathlib.Path(
            PROJECT_PATH,
            "fedcore",
            "repository",
            "data",
            "compression_data_operation_repository.json",
        )
        self.base_data_operation_path = pathlib.Path("data_operation_repository.json")

        self.fedcore_model_path = pathlib.Path(
            PROJECT_PATH,
            "fedcore",
            "repository",
            "data",
            "compression_model_repository.json",
        )
        self.base_model_path = pathlib.Path("model_repository.json")

    def _replace_operation(self, to_fedcore=True):
        for (cls, method_name), fedcore_method in FEDOT_METHOD_TO_REPLACE.items():
            new_method = fedcore_method if to_fedcore else getattr(cls, method_name)
            setattr(cls, method_name, new_method)

    def setup_repository(self):
        OperationTypesRepository.__repository_dict__.update(
            {
                "data_operation": {
                    "file": self.fedcore_data_operation_path,
                    "initialized_repo": True,
                    "default_tags": [],
                }
            }
        )

        OperationTypesRepository.assign_repo(
            "data_operation", self.fedcore_data_operation_path
        )

        OperationTypesRepository.__repository_dict__.update(
            {
                "model": {
                    "file": self.fedcore_data_operation_path,
                    "initialized_repo": True,
                    "default_tags": [],
                }
            }
        )
        OperationTypesRepository.assign_repo("model", self.fedcore_model_path)
        # replace mutations
        self._replace_operation(to_fedcore=True)
        return OperationTypesRepository

    def setup_default_repository(self):
        """
        Switching to fedot models.
        """
        OperationTypesRepository.__repository_dict__.update(
            {
                "data_operation": {
                    "file": self.base_data_operation_path,
                    "initialized_repo": None,
                    "default_tags": [
                        OperationTypesRepository.DEFAULT_DATA_OPERATION_TAGS
                    ],
                }
            }
        )
        OperationTypesRepository.assign_repo(
            "data_operation", self.base_data_operation_path
        )

        OperationTypesRepository.__repository_dict__.update(
            {
                "model": {
                    "file": self.base_model_path,
                    "initialized_repo": None,
                    "default_tags": [],
                }
            }
        )
        OperationTypesRepository.assign_repo("model", self.base_model_path)
        self._replace_operation(to_fedcore=False)
        return OperationTypesRepository

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Switching to fedot models.
        """
        OperationTypesRepository.__repository_dict__.update(
            {
                "data_operation": {
                    "file": self.base_data_operation_path,
                    "initialized_repo": None,
                    "default_tags": [
                        OperationTypesRepository.DEFAULT_DATA_OPERATION_TAGS
                    ],
                }
            }
        )
        OperationTypesRepository.assign_repo(
            "data_operation", self.base_data_operation_path
        )

        OperationTypesRepository.__repository_dict__.update(
            {
                "model": {
                    "file": self.base_model_path,
                    "initialized_repo": None,
                    "default_tags": [],
                }
            }
        )
        OperationTypesRepository.assign_repo("model", self.base_model_path)
