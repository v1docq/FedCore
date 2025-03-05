from typing import Sequence
import logging
import pathlib
import timeit
from datetime import datetime
from typing import Optional, Tuple

from golem.core.optimisers.adaptive.operator_agent import RandomAgent
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.optimisers.populational_optimizer import _try_unfit_graph
from golem.core.optimisers.genetic.evaluation import MultiprocessingDispatcher
from golem.core.log import Log
from golem.core.optimisers.genetic.operators.operator import (
    EvaluationOperator,
    PopulationT,
)
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import ObjectiveFunction
from golem.core.optimisers.opt_history_objects.individual import GraphEvalResult
from golem.core.optimisers.timer import Timer
from golem.utilities.memory import MemoryAnalytics
from golem.utilities.utilities import determine_n_jobs
from joblib import wrap_non_picklable_objects, Parallel, parallel_backend
from pymonad.maybe import Maybe

from fedcore.repository.constanst_repository import (
    FEDCORE_MUTATION_STRATEGY,
    FEDCORE_GRAPH_VALIDATION,
)
from fedcore.repository.initializer_industrial_models import FedcoreModels


class FedcoreDispatcher(MultiprocessingDispatcher):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Evaluate individuals without valid fitness in parallel.
        self.n_jobs = determine_n_jobs(self._n_jobs, self.logger)
        self.parallel_backednd = Parallel(
            n_jobs=self.n_jobs, verbose=0, pre_dispatch="2 * n_jobs"
        )

    def dispatch(
        self, objective: ObjectiveFunction, timer: Optional[Timer] = None
    ) -> EvaluationOperator:
        """Return handler to this object that hides all details
        and allows only to evaluate population with provided objective."""
        super().dispatch(objective, timer)
        return self.evaluate_with_cache

    def _eval_at_least_one_ind(self, individuals):
        for single_ind in individuals:
            evaluation_result = self.fedcore_evaluate_single(
                self,
                graph=single_ind.graph,
                uid_of_individual=single_ind.uid,
                with_time_limit=False,
            )
            successful_evals = self.apply_evaluation_results(
                [single_ind], [evaluation_result]
            )
            if successful_evals:
                break

    def evaluate_population(self, individuals: PopulationT) -> PopulationT:
        individuals_to_evaluate, individuals_to_skip = (
            self.split_individuals_to_evaluate(individuals)
        )

        with parallel_backend(
            backend="dask",
            n_jobs=self.n_jobs,
            # ,scatter=[individuals_to_evaluate]
        ):
            evaluation_results = list(
                map(
                    lambda ind: self.fedcore_evaluate_single(
                        self,
                        graph=ind.graph,
                        uid_of_individual=ind.uid,
                        logs_initializer=Log().get_parameters(),
                    ),
                    individuals_to_evaluate,
                )
            )

            individuals_evaluated = self.apply_evaluation_results(
                individuals_to_evaluate, evaluation_results
            )
            # If there were no successful evals then try once again getting at least one,
            # even if time limit was reached
            successful_evals = individuals_evaluated + individuals_to_skip
            self.population_evaluation_info(
                evaluated_pop_size=len(successful_evals), pop_size=len(individuals)
            )
            successful_evals = Maybe(
                value=individuals, monoid=not successful_evals
            ).maybe(
                default_value=successful_evals,
                extraction_function=lambda individuals: self._eval_at_least_one_ind(
                    individuals
                ),
            )

            MemoryAnalytics.log(
                self.logger,
                additional_info="parallel evaluation of population",
                logging_level=logging.INFO,
            )
        return successful_evals

    # @delayed
    @wrap_non_picklable_objects
    def fedcore_evaluate_single(
        self,
        graph: OptGraph,
        uid_of_individual: str,
        with_time_limit: bool = True,
        cache_key: Optional[str] = None,
        logs_initializer: Optional[Tuple[int, pathlib.Path]] = None,
    ) -> GraphEvalResult:
        if self._n_jobs != 1:
            FedcoreModels().setup_repository()

        graph = self.evaluation_cache.get(cache_key, graph)

        if with_time_limit and self.timer.is_time_limit_reached():
            return None
        if logs_initializer is not None:
            # in case of multiprocessing run
            Log.setup_in_mp(*logs_initializer)

        adapted_evaluate = self._adapter.adapt_func(self._evaluate_graph)
        start_time = timeit.default_timer()
        fitness, graph = adapted_evaluate(graph)
        end_time = timeit.default_timer()
        eval_time_iso = datetime.now().isoformat()

        eval_res = GraphEvalResult(
            uid_of_individual=uid_of_individual,
            fitness=fitness,
            graph=graph,
            metadata={
                "computation_time_in_seconds": end_time - start_time,
                "evaluation_time_iso": eval_time_iso,
            },
        )
        return eval_res


class FedcoreEvoOptimizer(EvoGraphOptimizer):
    def __init__(
        self,
        objective: Objective,
        initial_graphs: Sequence[OptGraph],
        requirements: GraphRequirements,
        graph_generation_params: GraphGenerationParams,
        graph_optimizer_params: GPAlgorithmParameters,
    ):
        graph_optimizer_params.adaptive_mutation_type = RandomAgent(
            actions=graph_optimizer_params.mutation_types,
            probs=FEDCORE_MUTATION_STRATEGY["params_mutation_strategy"],
        )
        graph_generation_params.verifier._rules = FEDCORE_GRAPH_VALIDATION
        super().__init__(
            objective,
            initial_graphs,
            requirements,
            graph_generation_params,
            graph_optimizer_params,
        )
        self.operators.remove(self.crossover)
        self.requirements = requirements
        self.eval_dispatcher = FedcoreDispatcher(
            adapter=graph_generation_params.adapter,
            n_jobs=requirements.n_jobs,
            graph_cleanup_fn=_try_unfit_graph,
            delegate_evaluator=graph_generation_params.remote_evaluator,
        )
