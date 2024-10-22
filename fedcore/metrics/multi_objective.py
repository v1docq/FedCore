import os
from typing import List, Sequence, Tuple, Optional
from typing import Optional

from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.tasks import Task, TaskTypesEnum
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.tuning.optuna_tuner import OptunaTuner
from golem.visualisation.opt_viz_extra import OptHistoryExtraVisualizer
from matplotlib import pyplot as plt

from fedcore.repository.constanst_repository import (
    FEDOT_GENETIC_MULTI_STRATEGY,
    FEDOT_EVO_MULTI_STRATEGY,
    InferenceMetricsEnum,
    CVMetricsEnum,
)
from fedcore.repository.model_repository import default_fedcore_availiable_operation


def visualise_pareto(
    front: Sequence[Individual],
    objectives_numbers: Tuple[int, int] = (0, 1),
    objectives_names: Sequence[str] = ("ROC-AUC", "Complexity"),
    file_name: str = "result_pareto.png",
    show: bool = False,
    save: bool = True,
    folder: str = "../../tmp/pareto",
    generation_num: int = None,
    individuals: Sequence[Individual] = None,
    minmax_x: List[float] = None,
    minmax_y: List[float] = None,
):
    pareto_obj_first, pareto_obj_second = [], []
    for ind in front:
        fit_first = ind.fitness.values[objectives_numbers[0]]
        pareto_obj_first.append(abs(fit_first))
        fit_second = ind.fitness.values[objectives_numbers[1]]
        pareto_obj_second.append(abs(fit_second))

    fig, ax = plt.subplots()

    if individuals is not None:
        obj_first, obj_second = [], []
        for ind in individuals:
            fit_first = ind.fitness.values[objectives_numbers[0]]
            obj_first.append(abs(fit_first))
            fit_second = ind.fitness.values[objectives_numbers[1]]
            obj_second.append(abs(fit_second))
        ax.scatter(obj_first, obj_second, c="green")

    ax.scatter(pareto_obj_first, pareto_obj_second, c="red")
    plt.plot(pareto_obj_first, pareto_obj_second, color="r")

    if generation_num is not None:
        ax.set_title(f"Pareto frontier, Generation: {generation_num}", fontsize=15)
    else:
        ax.set_title("Pareto frontier", fontsize=15)
    plt.xlabel(objectives_names[0], fontsize=15)
    plt.ylabel(objectives_names[1], fontsize=15)

    if minmax_x is not None:
        plt.xlim(minmax_x[0], minmax_x[1])
    if minmax_y is not None:
        plt.ylim(minmax_y[0], minmax_y[1])
    fig.set_figwidth(8)
    fig.set_figheight(8)
    if save:
        if not os.path.isdir("../../tmp"):
            os.mkdir("../../tmp")
        if not os.path.isdir(f"{folder}"):
            os.mkdir(f"{folder}")

        path = f"{folder}/{file_name}"
        plt.savefig(path, bbox_inches="tight")
    if show:
        plt.show()

    plt.cla()
    plt.clf()
    plt.close("all")


class MultiobjectiveCompression:
    def __init__(self, params: Optional[OperationParameters] = {}):

        self.multiobj_strategy = params.get("evo_strategy", "spea2")
        self.genetic_scheme_type = params.get("genetic_strategy", "parameter_free")
        self.task = params.get("task", "pruning")
        self.with_composition = params.get("with_composition", False)

        self.multiobj_strategy = FEDOT_EVO_MULTI_STRATEGY[self.multiobj_strategy]
        self.genetic_scheme_type = FEDOT_GENETIC_MULTI_STRATEGY[
            self.genetic_scheme_type
        ]

        self.pipeline_compressed_node = list(
            default_fedcore_availiable_operation(self.task)
        )[0]
        self.pipeline_compressed = (
            PipelineBuilder().add_node(self.pipeline_compressed_node).build()
        )

        quality_metric = CVMetricsEnum.cv_clf_metric
        latency_metric = InferenceMetricsEnum.latency
        InferenceMetricsEnum.throughput

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
            pipeline_pareto_front = self._with_composistion()
        else:
            tuner = (
                TunerBuilder(Task(TaskTypesEnum.classification))
                .with_tuner(OptunaTuner)
                .with_n_jobs(1)
                .with_iterations(10)
                .with_timeout(60)
                .with_early_stopping_rounds(5)
                .with_metric(self.metrics)
                .build(input_data)
            )
            pipeline_pareto_front = tuner.tune(self.pipeline_compressed)
        visualise_pareto(
            front=pipeline_pareto_front,
            objectives_numbers=(0, 1),
            objectives_names=("Accuracy", "Latency"),
        )
        return pipeline_pareto_front
