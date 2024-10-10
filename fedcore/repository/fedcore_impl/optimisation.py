import itertools
from copy import deepcopy
from itertools import chain
from math import ceil
from random import choice, sample, random
from typing import Sequence, Optional
from typing import Tuple

from fedot.core.composer.gp_composer.specific_operators import boosting_mutation
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.hyperparams import ParametersChanger
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.tasks import Task
from fedot.core.repository.tasks import TaskTypesEnum
from golem.core.adapter import register_native
from golem.core.dag.graph import ReconnectType
from golem.core.dag.graph_node import GraphNode
from golem.core.dag.graph_utils import graph_has_cycle, distance_to_primary_level
from golem.core.dag.graph_utils import node_depth, nodes_from_layer
from golem.core.dag.verification_rules import ERROR_PREFIX
from golem.core.optimisers.advisor import RemoveType
from golem.core.optimisers.genetic.gp_operators import equivalent_subtree, replace_subtrees
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import get_mutation_prob
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.mutation import MutationTypesEnum
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.opt_node_factory import OptNodeFactory
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import AlgorithmParameters
from golem.core.optimisers.optimizer import GraphGenerationParams



from fedcore.repository.model_repository import default_fedcore_availiable_operation


class FedcoreMutations:
    def __init__(self, task_type):
        self.node_adapter = PipelineAdapter()
        self.task_type = Task(task_type)
        #self.excluded_mutation = EXCLUDED_OPERATION_MUTATION[self.task_type.task_type.value]
        # self.fedcore_data_operations = default_fedcore_availiable_operation(
        #     self.task_type.task_type.value)
        # self.excluded = [list(TEMPORARY_EXCLUDED[x].keys())
        #                  for x in TEMPORARY_EXCLUDED.keys()]
        # self.excluded = (list(itertools.chain(*self.excluded)))
        # self.excluded = self.excluded + self.excluded_mutation
        # self.industrial_data_operations = [
        #     operation for operation in self.industrial_data_operations if operation not in self.excluded]

    def transform_to_pipeline_node(self, node):
        return self.node_adapter._transform_to_pipeline_node(node)

    def transform_to_opt_node(self, node):
        return self.node_adapter._transform_to_opt_node(node)

    def parameter_change_mutation(self,
                                  pipeline: Pipeline, requirements, graph_gen_params, parameters, **kwargs) -> Pipeline:
        """
        This type of mutation is passed over all nodes and changes
        hyperparameters of the operations with probability - 'node mutation probability'
        which is initialised inside the function
        """
        node_mutation_probability = get_mutation_prob(mut_id=parameters.mutation_strength,
                                                      node=pipeline.root_node)
        for node in pipeline.nodes:
            lagged = node.operation.metadata.id in ('lagged', 'sparse_lagged', 'exog_ts')
            do_mutation = random() < (node_mutation_probability * (0.5 if lagged else 1))
            if do_mutation:
                operation_name = node.operation.operation_type
                current_params = node.parameters

                # Perform specific change for particular parameter
                changer = ParametersChanger(operation_name, current_params)
                try:
                    new_params = changer.get_new_operation_params()
                    if new_params is not None:
                        node.parameters = new_params
                except Exception as ex:
                    pipeline.log.error(ex)
        return pipeline

    def single_change(self,
                      graph: OptGraph,
                      requirements: GraphRequirements,
                      graph_gen_params: GraphGenerationParams,
                      parameters: AlgorithmParameters
                      ) -> OptGraph:
        """
        Change node between two sequential existing modes.

        :param graph: graph to mutate
        """
        node = choice(graph.nodes)
        new_node = graph_gen_params.node_factory.exchange_node(
            self.transform_to_opt_node(node))
        if not new_node:
            return graph
        graph.update_node(node, self.transform_to_pipeline_node(new_node))
        return graph
