import gc
import logging
import pathlib
import timeit
from datetime import datetime
from typing import Optional, Tuple
import dask
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
from joblib import wrap_non_picklable_objects
from pymonad.maybe import Maybe
from pymonad.either import Either
from fedcore.repository.initializer_industrial_models import FedcoreModels
from golem.core.optimisers.fitness import null_fitness


class FedcoreDispatcher(MultiprocessingDispatcher):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Evaluate individuals without valid fitness in parallel.
        self.n_jobs = determine_n_jobs(self._n_jobs, self.logger)

    def dispatch(
            self, objective: ObjectiveFunction, timer: Optional[Timer] = None
    ) -> EvaluationOperator:
        """Return handler to this object that hides all details
        and allows only to evaluate population with provided objective."""
        super().dispatch(objective, timer)
        print('@@@ eval with cache', self.evaluate_with_cache)
        return self.evaluate_with_cache

    def _multithread_eval(self, individuals_to_evaluate):
        log = Log().get_parameters()
        if self.is_singlethread_regime:
            evaluation_results = [self.fedcore_evaluate_single(self,
                                                               graph=ind.graph,
                                                               uid_of_individual=ind.uid,
                                                               logs_initializer=log)
                                  for ind in individuals_to_evaluate]
        else:
            evaluation_results = list(map(lambda ind:
                                          self.fedcore_evaluate_single(self,
                                                                       graph=ind.graph,
                                                                       uid_of_individual=ind.uid,
                                                                       logs_initializer=log),
                                          individuals_to_evaluate))
            evaluation_results = dask.compute(*evaluation_results)
        return evaluation_results

    def _singlethread_eval(self, individuals_to_evaluate):
        log = Log().get_parameters()
        evaluation_results = list(map(lambda ind:
                                      self.fedcore_evaluate_single(self,
                                                                   graph=ind.graph,
                                                                   uid_of_individual=ind.uid,
                                                                   logs_initializer=log),
                                      individuals_to_evaluate))
        evaluation_results = dask.compute(*evaluation_results)
        return evaluation_results

    def _evaluate_graph(self, domain_graph):
        try:
            fitness = self._objective_eval(domain_graph)

            if self._post_eval_callback:
                self._post_eval_callback(domain_graph)
            gc.collect()
        except Exception:
            self.logger.warning('Exception during graph fitness eval')
            fitness = null_fitness()

        return fitness, domain_graph

    def _eval_at_least_one(self, individuals):
        successful_evals = None
        for single_ind in individuals:
            try:
                evaluation_result = self.fedcore_evaluate_single(
                    self, graph=single_ind.graph, uid_of_individual=single_ind.uid, with_time_limit=False)
                successful_evals = self.apply_evaluation_results(
                    [single_ind], [evaluation_result])
                if successful_evals:
                    break
            except Exception:
                successful_evals = None
        return successful_evals

    def apply_evaluation_results(self, individuals: PopulationT, evaluation_results) -> PopulationT:
        """Applies results of evaluation to the evaluated population.
        Excludes individuals that weren't evaluated."""
        print('@@@ FEDC appl_eval_res:', individuals, evaluation_results)
        evaluation_results = {res.uid_of_individual: res for res in evaluation_results if res is not None}
        print('@@@', evaluation_results)
        individuals_evaluated = []
        for ind in individuals:
            if not ind.uid in evaluation_results: 
                continue
            eval_res = evaluation_results[ind.uid]
            # if not eval_res:
            #     continue
            ind.set_evaluation_result(eval_res)
            individuals_evaluated.append(ind)
        return individuals_evaluated

    def evaluate_population(self, individuals: PopulationT) -> PopulationT:
        print('@@@ FEDC eval pop', individuals)
        individuals_to_evaluate, individuals_to_skip = self.split_individuals_to_evaluate(
            individuals)

        # Evaluate individuals without valid fitness in parallel.
        self.n_jobs = determine_n_jobs(self._n_jobs, self.logger)
        self.is_singlethread_regime = self._n_jobs == 1
        eval_res = self._multithread_eval(individuals)
        print('@@@ FEDC eval res', eval_res)
        individuals_evaluated = self.apply_evaluation_results(individuals_to_evaluate, eval_res)
        print('@@@ FEDC individuals_evaluated', individuals_evaluated)

        # individuals_evaluated = Maybe(individuals,
        #                               monoid=[individuals, True]).then(
        #     lambda generation: self._multithread_eval(generation)). \
        #     then(lambda eval_res: self.apply_evaluation_results(individuals_to_evaluate, eval_res)).value

        successful_evals = individuals_evaluated + individuals_to_skip
        self.population_evaluation_info(evaluated_pop_size=len(successful_evals), pop_size=len(individuals))

        successful_evals = Either(successful_evals,
                                  monoid=[individuals_evaluated, not successful_evals]).either(
            left_function=lambda x: x,
            right_function=lambda y: self._eval_at_least_one(y))
        print('@@@ FEDC susccm val', successful_evals)

        MemoryAnalytics.log(self.logger, additional_info='parallel evaluation of population',
                            logging_level=logging.INFO)
        return successful_evals

    def eval_ind(self, graph, uid_of_individual):
        if self.is_singlethread_regime:
            return self.singlethread_eval_ind(graph, uid_of_individual)
        else:
            return self.multithread_eval_ind(graph, uid_of_individual)

    @dask.delayed
    def multithread_eval_ind(self, graph, uid_of_individual):
        return self.__eval_ind(graph, uid_of_individual)

    def singlethread_eval_ind(self, graph, uid_of_individual):
        return self.__eval_ind(graph, uid_of_individual)

    def __eval_ind(self, graph, uid_of_individual):
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
                'computation_time_in_seconds': end_time - start_time,
                'evaluation_time_iso': eval_time_iso})
        return eval_res

    @wrap_non_picklable_objects
    def fedcore_evaluate_single(self,
                                graph: OptGraph,
                                uid_of_individual: str,
                                with_time_limit: bool = True,
                                cache_key: Optional[str] = None,
                                logs_initializer: Optional[Tuple[int,
                                pathlib.Path]] = None) -> GraphEvalResult:
        if not self.is_singlethread_regime:
            FedcoreModels().setup_repository()
        graph = self.evaluation_cache.get(cache_key, graph)
        if logs_initializer is not None:
            # in case of multiprocessing run
            Log.setup_in_mp(*logs_initializer)
        eval_ind = self.eval_ind(graph, uid_of_individual)
        return eval_ind
