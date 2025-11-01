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
        def safe_evaluate_with_cache(individuals):
            if not individuals:
                self.logger.warning('Empty individuals list passed to evaluator')
                return []
            result = self.evaluate_with_cache(individuals)
            if result is None:
                self.logger.error('evaluate_with_cache returned None, returning empty list')
                return []
            return result
        return safe_evaluate_with_cache

    def _multithread_eval(self, individuals_to_evaluate):
        if not individuals_to_evaluate:
            self.logger.warning('_multithread_eval: No individuals to evaluate')
            return []
        
        self.logger.info(f'_multithread_eval: Evaluating {len(individuals_to_evaluate)} individuals, is_singlethread_regime={self.is_singlethread_regime}')
        log = Log().get_parameters()
        if self.is_singlethread_regime:
            evaluation_results = []
            for ind in individuals_to_evaluate:
                try:
                    result = self.fedcore_evaluate_single(self,
                                                         graph=ind.graph,
                                                         uid_of_individual=ind.uid,
                                                         logs_initializer=log)
                    evaluation_results.append(result)
                except Exception as e:
                    self.logger.error(f'_multithread_eval: Failed to evaluate {ind.uid} in single-thread mode: {e}')
                    evaluation_results.append(None)
        else:
            @dask.delayed
            def delayed_eval(ind, log_params):
                if log_params is not None:
                    Log.setup_in_mp(*log_params)
                if not self.is_singlethread_regime:
                    FedcoreModels().setup_repository()
                return self.__eval_ind(ind.graph, ind.uid)
            
            evaluation_results = [delayed_eval(ind, log) for ind in individuals_to_evaluate]
            
            if evaluation_results:
                try:
                    from fedcore.architecture.abstraction.decorators import DaskServer
                    
                    dask_server = None
                    try:

                        if hasattr(DaskServer, '_instances') and DaskServer._instances and DaskServer in DaskServer._instances:
                            dask_server = DaskServer._instances[DaskServer]
                        else:
                            dask_server = None
                    except Exception:
                        dask_server = None
                    
                    if dask_server is not None:
                        has_client_attr = hasattr(dask_server, 'client')
                        client_not_none = dask_server.client is not None if has_client_attr else False
                    
                    if dask_server is not None and has_client_attr and client_not_none:
                        futures = dask_server.client.compute(evaluation_results)
                        try:
                            from distributed import wait
                            wait(futures, timeout=300)  
                            evaluation_results = dask_server.client.gather(futures)
                            evaluation_results = list(evaluation_results) if evaluation_results else []
                            for idx, res in enumerate(evaluation_results):
                                if res is not None and (hasattr(res, 'key') or 'Delayed' in str(type(res))):
                                    self.logger.error(f'_multithread_eval: Result {idx} from gather is still Delayed')
                                    evaluation_results[idx] = None
                        except Exception as e:
                            self.logger.error(f'Error gathering dask futures: {e}')
                            for future in futures:
                                if not future.done():
                                    future.cancel()
                            raise
                    
                    if dask_server is None or not (hasattr(dask_server, 'client') and dask_server.client is not None):
                        try:
                            computed = dask.compute(*evaluation_results, scheduler='threads')
                            evaluation_results = list(computed) if computed else []
                            for idx, res in enumerate(evaluation_results):
                                if res is not None and (hasattr(res, 'key') or 'Delayed' in str(type(res))):
                                    self.logger.error(f'_multithread_eval: Result {idx} from compute is still Delayed')
                                    evaluation_results[idx] = None
                        except Exception:
                            raise
                    
                    gc.collect()
                except (ImportError, AttributeError, Exception) as e:
                    self.logger.warning(f'Error during dask computation, falling back to sequential: {e}')

                    evaluation_results = []
                    for ind in individuals_to_evaluate:
                        try:
                            result = self.fedcore_evaluate_single(self,
                                                                 graph=ind.graph,
                                                                 uid_of_individual=ind.uid,
                                                                 logs_initializer=log)
                            evaluation_results.append(result)
                        except Exception as eval_err:
                            self.logger.error(f'_multithread_eval: Failed to evaluate {ind.uid} in fallback: {eval_err}')
                            evaluation_results.append(None)
            else:
                evaluation_results = []
        
        for idx, result in enumerate(evaluation_results):
            if result is not None:
                if hasattr(result, 'key') or hasattr(type(result), '__name__') and 'Delayed' in type(result).__name__:
                    self.logger.error(f'_multithread_eval: Result {idx} is still a Delayed object, marking as None')
                    evaluation_results[idx] = None
        
        successful = sum(1 for r in evaluation_results if r is not None)
        failed = len(evaluation_results) - successful
        self.logger.info(f'_multithread_eval: {successful} succeeded, {failed} failed out of {len(individuals_to_evaluate)}')
        
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
        if not individuals:
            self.logger.warning('_eval_at_least_one: No individuals provided')
            return []
        successful_evals = None
        for single_ind in individuals:
            try:
                evaluation_result = self.fedcore_evaluate_single(
                    self, graph=single_ind.graph, uid_of_individual=single_ind.uid, with_time_limit=False)
                if evaluation_result is None:
                    continue
                successful_evals = self.apply_evaluation_results(
                    [single_ind], [evaluation_result])
                if successful_evals:
                    break
            except Exception:
                successful_evals = None
        return successful_evals if successful_evals is not None else []

    def apply_evaluation_results(self, individuals: PopulationT, evaluation_results) -> PopulationT:
        """Applies results of evaluation to the evaluated population.
        Excludes individuals that weren't evaluated."""
        if not evaluation_results:
            self.logger.warning(f'apply_evaluation_results: No evaluation results provided for {len(individuals)} individuals')
            return []
        
        valid_results = {}
        for res in evaluation_results:
            if res is None:
                continue
            if hasattr(res, 'key') or 'Delayed' in str(type(res)):
                self.logger.warning(f'apply_evaluation_results: Skipping Delayed object: {type(res)}')
                continue
            if not hasattr(res, 'uid_of_individual'):
                self.logger.warning(f'apply_evaluation_results: Result has no uid_of_individual: {type(res)}')
                continue
            valid_results[res.uid_of_individual] = res
        
        if not valid_results:
            self.logger.error(f'apply_evaluation_results: All {len(evaluation_results)} results were None or invalid!')
            return []
        
        self.logger.info(f'apply_evaluation_results: {len(valid_results)} valid results out of {len(evaluation_results)} total')
        individuals_evaluated = []
        failed_uids = []
        for ind in individuals:
            if ind.uid not in valid_results:
                failed_uids.append(ind.uid)
                continue
            eval_res = valid_results[ind.uid]
            ind.set_evaluation_result(eval_res)
            individuals_evaluated.append(ind)
        
        if failed_uids:
            self.logger.warning(f'apply_evaluation_results: Failed to match {len(failed_uids)} individuals: {failed_uids[:3]}...')
        
        return individuals_evaluated

    def evaluate_population(self, individuals: PopulationT) -> PopulationT:
        individuals_to_evaluate, individuals_to_skip = self.split_individuals_to_evaluate(
            individuals)
        
        self.logger.info(f'evaluate_population: {len(individuals)} total, {len(individuals_to_evaluate)} to evaluate, {len(individuals_to_skip)} to skip')

        # Evaluate individuals without valid fitness in parallel.
        self.n_jobs = determine_n_jobs(self._n_jobs, self.logger)
        self.is_singlethread_regime = self._n_jobs == 1
        eval_res = self._multithread_eval(individuals_to_evaluate)
        individuals_evaluated = self.apply_evaluation_results(individuals_to_evaluate, eval_res)
        self.logger.info(f'evaluate_population: {len(individuals_evaluated)} individuals successfully evaluated')

        successful_evals = individuals_evaluated + individuals_to_skip
        self.population_evaluation_info(evaluated_pop_size=len(successful_evals), pop_size=len(individuals))

        successful_evals = Either(successful_evals,
                                  monoid=[individuals_evaluated, not successful_evals]).either(
            left_function=lambda x: x,
            right_function=lambda y: self._eval_at_least_one(y))

        MemoryAnalytics.log(self.logger, additional_info='parallel evaluation of population',
                            logging_level=logging.INFO)
        
        if not self.is_singlethread_regime:
            gc.collect()
        
        if successful_evals is None:
            self.logger.error('evaluate_population returned None, returning empty list')
            return []
        
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