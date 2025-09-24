import traceback
from abc import abstractmethod

from typing import Dict
from fedot.core.repository.metrics_repository import (
    ClassificationMetricsEnum,
    RegressionMetricsEnum,
    MetricsEnum,
    MetricCallable, ComplexityMetricsEnum,
)
from fedcore.metrics.metric_impl import (Accuracy,
                                         F1,
                                        RMSE,
                                         Logloss,
                                         MAE,
                                         MAPE,
                                         MSE,
                                         MSLE,
                                         Precision,
                                         QualityMetric,
                                         R2,
                                         SMAPE)
from fedot.core.composer.metrics import (ROCAUC, ComplexityMetric, NodeNum, StructuralComplexity)
from typing import Union

from fedcore.metrics.cv_metrics import (
    LastLayer,
    IntermediateFeatures,
    IntermediateAttention,
    Latency,
    Throughput,
    CV_quality_metric,
)
from fedcore.repository.constant_repository import (
    DistilationMetricsEnum,
    InferenceMetricsEnum,
    CVMetricsEnum,
)
from typing import Callable, Iterable, Tuple

import numpy as np
from golem.core.optimisers.fitness import Fitness
from golem.core.optimisers.objective.objective import to_fitness
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline

DataSource = Callable[[], Iterable[Tuple[InputData, InputData]]]


def evaluate_objective_fedcore(self, graph: Pipeline) -> Fitness:
    # Seems like a workaround for situation when logger is lost
    #  when adapting and restoring it to/from OptGraph.
    graph.log = self._log

    graph_id = graph.root_node.descriptive_id
    self._log.debug(f'Pipeline {graph_id} fit started')

    folds_metrics = []
    for fold_id, (train_data, test_data) in enumerate(self._data_producer()):
        self._pipelines_cache = None # turn off cache
        try:
            prepared_pipeline = self.prepare_graph(graph, train_data, fold_id, self._eval_n_jobs)
            evaluated_fitness = self._objective(prepared_pipeline,
                                                reference_data=test_data,
                                                validation_blocks=self._validation_blocks)
        except Exception as exec_during_fit:
            self._log.warning(f'Exception - {exec_during_fit} during pipeline learning process. '
                              f'Skipping the graph: {graph_id}', raise_if_test=True)
            break
        evaluated_fitness = self._objective(prepared_pipeline,
                                            reference_data=test_data,
                                            validation_blocks=self._validation_blocks)
        if evaluated_fitness.valid:
            folds_metrics.append(evaluated_fitness.values)
        else:
            self._log.warning(f'Invalid fitness after objective evaluation. '
                              f'Skipping the graph: {graph_id}', raise_if_test=True)
    if folds_metrics:
        folds_metrics = tuple(np.mean(folds_metrics, axis=0))  # averages for each metric over folds
        self._log.debug(f'Pipeline {graph_id} with evaluated metrics: {folds_metrics}')
    else:
        folds_metrics = None
    return to_fitness(folds_metrics, self._objective.is_multi_objective)


class MetricsRepository:
    _metrics_implementations: Dict[MetricsEnum, MetricCallable] = {
        # classification
        ClassificationMetricsEnum.ROCAUC: ROCAUC.get_value,
        ClassificationMetricsEnum.ROCAUC_penalty: ROCAUC.get_value_with_penalty,
        ClassificationMetricsEnum.f1: F1.get_value,
        ClassificationMetricsEnum.precision: Precision.get_value,
        ClassificationMetricsEnum.accuracy: Accuracy.get_value,
        ClassificationMetricsEnum.logloss: Logloss.get_value,
        # regression
        RegressionMetricsEnum.MAE: MAE.get_value,
        RegressionMetricsEnum.MSE: MSE.get_value,
        RegressionMetricsEnum.MSLE: MSLE.get_value,
        RegressionMetricsEnum.MAPE: MAPE.get_value,
        RegressionMetricsEnum.SMAPE: SMAPE.get_value,
        RegressionMetricsEnum.RMSE: RMSE.get_value,
        #RegressionMetricsEnum.RMSE_penalty: RMSE.get_value_with_penalty,
        RegressionMetricsEnum.R2: R2.get_value,
        # Distilation metric
        DistilationMetricsEnum.last_layer: LastLayer.get_value,
        DistilationMetricsEnum.intermediate_layers_attention: IntermediateAttention.get_value,
        DistilationMetricsEnum.intermediate_layers_feature: IntermediateFeatures.get_value,
        # Inference metric
        InferenceMetricsEnum.latency: Latency.get_value,
        InferenceMetricsEnum.throughput: Throughput.get_value,
        ComplexityMetricsEnum.node_number: NodeNum.get_value,
        # CV metric
        CVMetricsEnum.cv_clf_metric: CV_quality_metric.get_value,

    }
    _metrics_classes = {}
    for metric_id, metric_func in _metrics_implementations.items():
        try:
            _metrics_classes.update({metric_id: getattr(metric_func, "__self__")})
        except Exception:
            continue

    @staticmethod
    def get_metric(metric_name: MetricsEnum) -> MetricCallable:
        return MetricsRepository._metrics_implementations[metric_name]

    @staticmethod
    def get_metric_class(
            metric_name: MetricsEnum,
    ) -> Union[QualityMetric, ComplexityMetric]:
        return MetricsRepository._metrics_classes[metric_name]
