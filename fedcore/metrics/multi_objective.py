from typing import Optional
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.tuning.optuna_tuner import OptunaTuner
from golem.visualisation.opt_viz_extra import OptHistoryExtraVisualizer

from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.tasks import Task, TaskTypesEnum

from fedcore.repository.constanst_repository import FEDOT_GENETIC_MULTI_STRATEGY, FEDOT_EVO_MULTI_STRATEGY, \
    InferenceMetricsEnum, CVMetricsEnum
from fedcore.repository.initializer_industrial_models import FedcoreModels
from fedcore.repository.model_repository import default_fedcore_availiable_operation


class MultiobjectiveCompression:
    def __init__(self, params: Optional[OperationParameters] = {}):

        self.multiobj_strategy = params.get('evo_strategy', 'spea2')
        self.genetic_scheme_type = params.get('genetic_strategy', 'parameter_free')
        self.task = params.get('task', 'pruning')
        self.with_composition = params.get('with_composition', False)

        self.multiobj_strategy = FEDOT_EVO_MULTI_STRATEGY[self.multiobj_strategy]
        self.genetic_scheme_type = FEDOT_GENETIC_MULTI_STRATEGY[self.genetic_scheme_type]

        self.pipeline_compressed_node = list(default_fedcore_availiable_operation(self.task))[0]
        self.pipeline_compressed = PipelineBuilder().add_node(self.pipeline_compressed_node).build()

        quality_metric = CVMetricsEnum.cv_clf_metric
        latency_metric = InferenceMetricsEnum.latency
        throughput_metric = InferenceMetricsEnum.throughput

        self.metrics = [quality_metric, latency_metric]
        self.timeout = 10

    def _results_visualization(self, history, composed_pipelines):
        visualiser = OptHistoryExtraVisualizer(history)
        visualiser.visualise_history()
        visualiser.pareto_gif_create()
        visualiser.boxplots_gif_create()
        for pipeline_evo_composed in composed_pipelines:
            pipeline_evo_composed.show()

    def _with_composistion(self):
        # the choice of the metric for the pipeline quality assessment during composition
        # composer_requirements = PipelineComposerRequirements(
        #     primary=self.available_model_types,
        #     secondary=self.available_model_types,
        #     timeout=self.timeout,
        #     num_of_generations=20
        # )
        # # the choice and initialisation of the GP search
        # params = GPAlgorithmParameters(
        #     selection_types=[self.multiobj_strategy],
        #     genetic_scheme_type=self.genetic_scheme_type,
        # )
        #
        # # Create composer and with required composer params
        # self.composer = (
        #     ComposerBuilder(task=Task(TaskTypesEnum.classification))
        #     .with_optimizer_params(params)
        #     .with_requirements(composer_requirements)
        #     .with_metrics(self.metrics)
        #     .build()
        # )
        # # the optimal pipeline generation by composition - the most time-consuming task
        # pipelines_evo_composed = self.composer.compose_pipeline(data=input_data)
        # pipelines_quality_metric = []
        #
        # for pipeline_num, pipeline_evo_composed in enumerate(pipelines_evo_composed):
        #
        #     tuner = (
        #         TunerBuilder(self.task)
        #         .with_tuner(OptunaTuner)
        #         .with_iterations(50)
        #         .with_metric(self.metrics[0], self.metrics[1])
        #         .build(input_data)
        #     )
        #     nodes = pipeline_evo_composed.nodes
        #     for node_index, node in enumerate(nodes):
        #         if isinstance(node, PipelineNode) and node.is_primary:
        #             pipeline_evo_composed = tuner.tune_node(pipeline_evo_composed, node_index)
        #
        #     pipeline_evo_composed.fit(input_data=input_data)
        #
        #     # the quality assessment for the obtained composite models
        #     roc_on_valid_evo_composed = calculate_validation_metric(pipeline_evo_composed,
        #                                                             dataset_to_validate)
        #
        #     pipelines_roc_auc.append(roc_on_valid_evo_composed)
        #     if len(pipelines_evo_composed) > 1:
        #         print(f'Composed ROC AUC of pipeline {pipeline_num + 1} is {round(roc_on_valid_evo_composed, 3)}')
        #
        #     else:
        #         print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')
        #
        #     if self.visualization:
        #         self._results_visualization(composed_pipelines=pipelines_evo_composed,
        #                                     history=self.composer.history)
        #
        # self.composer.history.to_csv()
            pass

    def evaluate(self, input_data):
        if self.with_composition:
            pipeline_compressed_tuned = self._with_composistion()
        else:
            tuner = (
                TunerBuilder(Task(TaskTypesEnum.classification))
                .with_tuner(OptunaTuner)
                .with_n_jobs(1)
                .with_iterations(10)
                .with_metric(self.metrics)
                .build(input_data)
            )
            pipeline_compressed_tuned = tuner.tune(self.pipeline_compressed)

        return pipeline_compressed_tuned
