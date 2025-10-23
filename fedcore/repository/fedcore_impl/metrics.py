import traceback
from abc import abstractmethod

from typing import Dict, List
from fedot.core.repository.metrics_repository import (
    ClassificationMetricsEnum,
    RegressionMetricsEnum,
    MetricsEnum,
    MetricCallable, ComplexityMetricsEnum,
)

from fedcore.repository.constanst_repository import DEFAULT_METRICS_BY_TASK, TaskTypesEnum

from fedcore.metrics.quality import QualityMetric, metric_factory

from fedot.core.composer.metrics import (ComplexityMetric, NodeNum, StructuralComplexity) # TODO include in fedcore.metrics module
from typing import Union

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


class MetricByTask:
    __metric_by_task = DEFAULT_METRICS_BY_TASK

    @staticmethod
    def get_default_quality_metrics(task_type: TaskTypesEnum) -> List[MetricsEnum]:
        return [MetricByTask.__metric_by_task.get(task_type)]

    @staticmethod
    def compute_default_metric(task_type: TaskTypesEnum, true: InputData, predicted,
                               round_up_to: int = 6) -> float:
        """Returns the value of metric defined by task"""
        metric_name = MetricByTask.get_default_quality_metrics(task_type)[0]
        metric = MetricsRepository.get_metric_class(metric_name)
        try:
            return round(metric.metric(reference=true, predicted=predicted), round_up_to)
        except ValueError:
            return metric.default_value


class MetricsRepository:
    @staticmethod
    def get_metric(metric_name: MetricsEnum) -> MetricCallable:
        return metric_factory(metric_name).get_value

    @staticmethod
    def get_metric_class(
            metric_name: MetricsEnum,
    ) -> Union[QualityMetric, ComplexityMetric]:
        return metric_factory(metric_name)
