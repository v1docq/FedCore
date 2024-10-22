import pathlib
import fedot.core.repository.tasks as fedot_task
from fedot.core.repository.metrics_repository import (
    MetricsRepository as fedot_metric_repo,
)
from fedot.api.api_utils.api_params_repository import ApiParamsRepository

from fedot.api.api_utils.assumptions.assumptions_handler import AssumptionsHandler
from fedot.core.data.merge.data_merger import DataMerger, ImageDataMerger
from fedot.core.operations.operation import Operation
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.repository.operation_types_repository import OperationTypesRepository

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
    _get_default_fedcore_mutations,
)
from fedcore.repository.fedcore_impl.data import (
    build_holdout_producer,
    build_fedcore_dataproducer,
)
from fedcore.repository.fedcore_impl.metrics import MetricsRepository as FedcoreMetric

FEDCORE_METRIC_REPO = FedcoreMetric()

FEDOT_METHOD_TO_REPLACE = [
    (fedot_task, "TaskTypesEnum"),
    (DataSourceSplitter, "build"),
    (fedot_metric_repo, "_metrics_implementations"),
    (fedot_metric_repo, "_metrics_classes"),
    (ApiParamsRepository, "_get_default_mutations"),
    (PipelineSearchSpace, "get_parameters_dict"),
    (AssumptionsHandler, "fit_assumption_and_check_correctness"),
    (DataSourceSplitter, "_build_holdout_producer"),
    (DataMerger, "merge"),
    (Operation, "fit"),
    (Operation, "_predict"),
    (ImageDataMerger, "preprocess_predicts"),
    (ImageDataMerger, "merge_predicts"),
]

FEDCORE_REPLACE_METHODS = [
    TaskCompression,
    build_fedcore_dataproducer,
    FedcoreMetric._metrics_implementations,
    FedcoreMetric._metrics_classes,
    _get_default_fedcore_mutations,
    get_fedcore_search_space,
    _fit_assumption_and_check_correctness,
    build_holdout_producer,
    _merge,
    _fit,
    predict_operation_fedcore,
    fedcore_preprocess_predicts,
    merge_fedcore_predicts,
]
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
        if to_fedcore:
            method = FEDCORE_REPLACE_METHODS
        else:
            method = DEFAULT_METHODS
        for class_impl, method_to_replace in zip(FEDOT_METHOD_TO_REPLACE, method):
            setattr(class_impl[0], class_impl[1], method_to_replace)

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
