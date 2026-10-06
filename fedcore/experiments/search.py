"""Budgeted finite-space comparisons using existing GOLEM and optional Optuna.

This is a benchmark adapter for a frozen candidate space, not the original
FedcoreEvoOptimizer graph search. Hypervolume is a report statistic; GOLEM
selection uses SPEA2 and never a claimed hypervolume fitness.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, fields
from datetime import timedelta

import torch

from .protocol import CandidateSpec, ProtocolError


@dataclass(frozen=True)
class SearchConfig:
    method: str = "random"
    seed: int = 0
    max_evaluations: int = 20
    max_seconds: float = 60.0
    mutation_only: bool = True
    population_size: int = 4
    tpe_startup_trials: int = 2

    def __post_init__(self):
        if self.method not in ("random", "evolution", "bayesian", "exhaustive", "manual"):
            raise ProtocolError("Unknown finite-space comparison method")
        for key in ("seed", "max_evaluations", "population_size", "tpe_startup_trials"):
            value = getattr(self, key)
            if type(value) is not int or value < (0 if key == "seed" else 1):
                raise ProtocolError(f"Invalid {key}")
        if isinstance(self.max_seconds, bool) or not isinstance(self.max_seconds, (int, float)) or not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ProtocolError("Finite positive wall-time budget required")
        if type(self.mutation_only) is not bool:
            raise ProtocolError("mutation_only must be boolean")

    def to_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}


def _feasible(record, baseline, protocol):
    return (record.get("status") == "succeeded" and
            record.get("measurement", {}).get("status") == "succeeded" and
            record["validation"]["loss"] <= baseline["validation"]["loss"] + protocol.quality_tolerance and
            math.isfinite(record["validation"]["loss"]) and
            math.isfinite(record["measurement"]["metrics"][protocol.selection_cost]))


def _objectives(record, protocol):
    return (record["validation"]["loss"] / protocol.quality_scale,
            record["measurement"]["metrics"][protocol.selection_cost] / protocol.cost_scale)


def hypervolume_2d(points, reference):
    """Exact union of minimization rectangles at a fixed, finite reference."""
    if len(reference) != 2 or any(not math.isfinite(v) for v in reference):
        raise ProtocolError("Finite 2D reference required")
    points = tuple(tuple(point) for point in points)
    if any(len(point) != 2 or any(not math.isfinite(value) for value in point) for point in points):
        raise ProtocolError("Hypervolume points must be finite 2D vectors")
    clipped = sorted(set(point for point in points if point[0] < reference[0] and point[1] < reference[1]))
    area, previous_y = 0.0, reference[1]
    for x, y in clipped:
        if y < previous_y:
            area += (reference[0] - x) * (previous_y - y)
            previous_y = y
    return area


def archive_summary(records, baseline, protocol):
    from fedcore.metrics.pareto import ParetoMetrics
    feasible = [record for record in records if _feasible(record, baseline, protocol)]
    points = [_objectives(record, protocol) for record in feasible]
    mask = ParetoMetrics().pareto_metric_list(torch.tensor(points, dtype=torch.float64), maximise=False).tolist() if points else []
    archive = [record for record, keep in zip(feasible, mask) if keep]
    archive_points = [_objectives(record, protocol) for record in archive]
    volume = hypervolume_2d(archive_points, protocol.hypervolume_reference)
    contributions = {record["candidate_id"]: volume - hypervolume_2d(
        [point for index, point in enumerate(archive_points) if index != excluded], protocol.hypervolume_reference)
        for excluded, record in enumerate(archive)}
    return {"archive_ids": [record["candidate_id"] for record in archive],
            "feasible_ids": [record["candidate_id"] for record in feasible],
            "excluded_ids": [record["candidate_id"] for record in records if record not in feasible],
            "objectives": {record["candidate_id"]: list(_objectives(record, protocol)) for record in feasible},
            "hypervolume": volume, "candidate_hypervolume_contributions": contributions,
            "reference": list(protocol.hypervolume_reference),
            "directions": ["minimize quality loss", f"minimize {protocol.selection_cost}"],
            "scales": [protocol.quality_scale, protocol.cost_scale],
            "quality_tolerance": protocol.quality_tolerance,
            "scope": "nondominated among evaluated feasible candidates; no global optimality guarantee",
            "fitness_role": "validation", "hypervolume_role": "archive reporting; not GOLEM selection fitness"}


def cost_selection_comparison(records, baseline, protocol):
    """H3 comparison on one fixed feasible archive, with repeat dispersion."""
    archive_ids = set(archive_summary(records, baseline, protocol)["archive_ids"])
    feasible = [record for record in records if record["candidate_id"] in archive_ids]
    if not feasible:
        return {"status": "unavailable", "reason": "No measured feasible candidate"}
    size = min(feasible, key=lambda item: (item["measurement"]["metrics"]["file_bytes"], item["candidate_id"]))
    latency = min(feasible, key=lambda item: (item["measurement"]["metrics"]["latency_p50_ms"], item["candidate_id"]))
    def describe(record):
        values = record["measurement"]["raw_inference_ms"]
        return {"candidate_id": record["candidate_id"], "file_bytes": record["measurement"]["metrics"]["file_bytes"],
                "latency_p50_ms": record["measurement"]["metrics"]["latency_p50_ms"],
                "repeat_min_ms": min(values), "repeat_max_ms": max(values), "repeats": len(values),
                "dispersion_kind": "raw repeat range; not a confidence interval"}
    return {"status": "measured", "size_choice": describe(size), "latency_choice": describe(latency),
            "same_choice": size["candidate_id"] == latency["candidate_id"],
            "scope": "fixed evaluated feasible set; single-run observation does not establish H3"}


class _BudgetStop(BaseException):
    """Escapes optimizer objective exception wrappers; caught by this adapter."""


def run_search(space, config, evaluate, baseline, protocol, *, start_time=None, event=None):
    if not isinstance(config, SearchConfig) or any(not isinstance(item, CandidateSpec) for item in space):
        raise ProtocolError("Frozen CandidateSpec space and SearchConfig required")
    if not space or len({item.candidate_id for item in space}) != len(space):
        raise ProtocolError("Nonempty unique search space required")
    if any(item.method == "baseline" for item in space):
        raise ProtocolError("Baseline is already evaluated and charged outside the proposal space")
    start = time.perf_counter() if start_time is None else start_time
    rng = random.Random(config.seed)
    records, trace, result_cache = [baseline], [], {}
    stop_reason = "space_exhausted"
    emit = event or (lambda value: None)
    def elapsed():
        return time.perf_counter() - start
    def budget_exhausted():
        return len(records) >= config.max_evaluations or elapsed() >= config.max_seconds
    def execute(index, origin):
        nonlocal stop_reason
        if index in result_cache:
            trace.append({"event": "proposal_reused", "candidate_id": space[index].candidate_id,
                          "origin": origin, "elapsed_seconds": elapsed()})
            return result_cache[index]
        if budget_exhausted():
            stop_reason = "evaluation_budget" if len(records) >= config.max_evaluations else "time_budget"
            raise _BudgetStop()
        generated = {"event": "proposal", "candidate_id": space[index].candidate_id,
                     "candidate_index": index, "origin": origin, "elapsed_seconds": elapsed()}
        trace.append(generated)
        emit(generated)
        record = evaluate(space[index])
        records.append(record)
        result_cache[index] = record
        summary = archive_summary(records, baseline, protocol)
        observation = {"event": "evaluated", "candidate_id": space[index].candidate_id,
                       "status": record["status"], "evaluations": len(records),
                       "elapsed_seconds": elapsed(), "hypervolume": summary["hypervolume"],
                       "archive_ids": summary["archive_ids"], "cost_seconds": record["wall_seconds"]}
        trace.append(observation)
        emit(observation)
        return record
    engine = {"random": "python seeded random permutation", "exhaustive": "ordered enumeration",
              "manual": "prospectively supplied manual candidate order", "bayesian": "Optuna TPESampler",
              "evolution": "GOLEM EvoGraphOptimizer finite-space adapter"}[config.method]
    engine_status, engine_reason = "succeeded", None
    try:
        if config.method in ("random", "exhaustive", "manual"):
            indices = list(range(len(space)))
            if config.method == "random":
                rng.shuffle(indices)
            for index in indices:
                execute(index, config.method)
        elif config.method == "bayesian":
            import optuna
            sampler = optuna.samplers.TPESampler(seed=config.seed, n_startup_trials=config.tpe_startup_trials)
            study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
            # Each trial sees the same frozen categorical domain, no hidden training.
            max_trials = max(1, config.max_evaluations * 10)
            for _ in range(max_trials):
                if len(result_cache) == len(space):
                    break
                if budget_exhausted():
                    raise _BudgetStop()
                trial = study.ask()
                observations = len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)))
                index = trial.suggest_categorical("candidate_index", list(range(len(space))))
                trace.append({"event": "optuna_proposal", "trial": trial.number,
                              "startup_phase": observations < config.tpe_startup_trials,
                              "completed_observations_before_proposal": observations,
                              "candidate_id": space[index].candidate_id,
                              "repeated_candidate": index in result_cache,
                              "elapsed_seconds": elapsed()})
                record = execute(index, "Optuna TPE categorical proposal")
                if _feasible(record, baseline, protocol):
                    study.tell(trial, _objectives(record, protocol))
                else:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
            else:
                stop_reason = "proposal_attempt_limit"
        else:
            from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
            from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
            from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
            from golem.core.optimisers.genetic.operators.selection import SelectionTypesEnum
            from golem.core.optimisers.graph import OptGraph, OptNode
            from golem.core.optimisers.objective import Objective
            from golem.core.optimisers.optimization_parameters import GraphRequirements
            from golem.core.optimisers.optimizer import GraphGenerationParams
            def replace_candidate(graph, **kwargs):
                old = int(graph.nodes[0].parameters["candidate_index"])
                choices = [index for index in range(len(space)) if index != old]
                new = rng.choice(choices) if choices else old
                graph.nodes[0].parameters["candidate_index"] = new
                variation = {"event": "variation", "operator": "replace_candidate", "type": "mutation",
                             "parent_candidate_id": space[old].candidate_id, "candidate_id": space[new].candidate_id,
                             "elapsed_seconds": elapsed()}
                trace.append(variation)
                emit(variation)
                return graph
            def observed(graph):
                return execute(int(graph.nodes[0].parameters["candidate_index"]), "GOLEM objective evaluation")
            def objective_quality(graph):
                record = observed(graph)
                if not _feasible(record, baseline, protocol):
                    raise ValueError("Infeasible/failed candidate has no selection fitness")
                return _objectives(record, protocol)[0]
            def objective_cost(graph):
                return _objectives(observed(graph), protocol)[1]
            population_size = min(config.population_size, len(space))
            initial = list(range(len(space)))
            rng.shuffle(initial)
            graphs = [OptGraph([OptNode({"name": "candidate", "params": {"candidate_index": index}})])
                      for index in initial[:population_size]]
            objective = Objective({"validation_loss": objective_quality}, {protocol.selection_cost: objective_cost}, is_multi_objective=True)
            parameters = GPAlgorithmParameters(multi_objective=True, pop_size=population_size,
                max_pop_size=population_size, mutation_types=[replace_candidate], mutation_prob=1.0,
                crossover_types=[CrossoverTypesEnum.none] if config.mutation_only else [CrossoverTypesEnum.one_point],
                crossover_prob=0.0 if config.mutation_only else 0.5,
                selection_types=[SelectionTypesEnum.spea2], variable_mutation_num=False)
            requirements = GraphRequirements(num_of_generations=max(2, config.max_evaluations),
                timeout=timedelta(seconds=config.max_seconds), n_jobs=1, show_progress=False,
                keep_history=True, history_dir=None, agent_dir=None,
                max_depth=1, min_arity=0, max_arity=0, early_stopping_iterations=config.max_evaluations)
            generation = GraphGenerationParams(rules_for_constraint=(lambda graph: len(graph.nodes) == 1,),
                                               available_node_types=["candidate"])
            optimizer = EvoGraphOptimizer(objective, graphs, requirements, generation, parameters)
            original_stop = optimizer.stop_optimization
            optimizer.stop_optimization = lambda: budget_exhausted() or len(result_cache) == len(space) or original_stop()
            def callback(population, *args):
                selection = {"event": "golem_population", "selection": "SPEA2",
                             "candidate_ids": [space[int(ind.graph.nodes[0].parameters["candidate_index"])].candidate_id for ind in population],
                             "mutation_prob": optimizer.graph_optimizer_params.mutation_prob,
                             "crossover_prob": optimizer.graph_optimizer_params.crossover_prob,
                             "crossover_types": [str(item) for item in optimizer.graph_optimizer_params.crossover_types],
                             "elapsed_seconds": elapsed()}
                trace.append(selection)
                emit(selection)
            optimizer.set_iteration_callback(callback)
            random_state = random.getstate()
            import numpy as np
            numpy_state = np.random.get_state()
            try:
                random.seed(config.seed)
                np.random.seed(config.seed)
                optimizer.optimise(objective)
            finally:
                random.setstate(random_state)
                np.random.set_state(numpy_state)
            if len(result_cache) < len(space) and not budget_exhausted():
                stop_reason = "optimizer_stopped"
    except _BudgetStop:
        if stop_reason == "space_exhausted":
            stop_reason = "evaluation_budget" if len(records) >= config.max_evaluations else "time_budget"
    except ImportError as error:
        engine_status, engine_reason, stop_reason = "unsupported", str(error), "dependency_unavailable"
    except Exception as error:
        engine_status, engine_reason, stop_reason = "failed", f"{type(error).__name__}: {error}", "optimizer_failure"
    if budget_exhausted() and stop_reason == "space_exhausted":
        stop_reason = "evaluation_budget" if len(records) >= config.max_evaluations else "time_budget"
    summary = archive_summary(records, baseline, protocol)
    return {"status": engine_status, "reason": engine_reason, "configuration": config.to_dict(),
            "engine": engine, "domain": "frozen finite candidate space; distinct from FedcoreEvoOptimizer graph domain",
            "space": [item.to_dict() for item in space], "trace": trace, "stop_reason": stop_reason,
            "charged_evaluations": len(records), "baseline_evaluations": 1,
            "wall_seconds": elapsed(), "budget_scope": "runner preparation and supplied external preparation, baseline, calibration, training, export, measurement and failed attempts; final test reported separately",
            "overshoot_seconds": max(0.0, elapsed() - config.max_seconds),
            "budget_enforcement": "before each candidate; an in-flight indivisible operation may overrun and is reported",
            "archive": summary, "empirical_h4": "unconfirmed; requires prospectively designed repeated comparisons"}


def compare_search_runs(manifests, *, bootstrap_samples=1000, seed=0):
    """Paired pilot effect estimates, preserving every seed and failure."""
    if bootstrap_samples < 1:
        raise ProtocolError("Positive bootstrap count required")
    groups = {}
    for manifest in manifests:
        search = manifest.get("search")
        if search is None:
            raise ProtocolError("Search manifest required")
        config = search["configuration"]
        metadata = manifest["data"].get("metadata", {})
        identity = {key: metadata.get(key) for key in ("scenario", "dataset", "data_sha256", "target_sha256", "raw_source_sha256", "dataset_revision", "source")}
        key = (manifest["data"]["task"], canonical_protocol(identity), config["max_evaluations"], config["max_seconds"],
               canonical_protocol(manifest["protocol"]), canonical_protocol(manifest["environment"]))
        runs = groups.setdefault(key, {}).setdefault(config["method"], {})
        if config["seed"] in runs:
            raise ProtocolError("Duplicate method/seed run in one comparison group")
        runs[config["seed"]] = manifest
    comparisons = []
    rng = random.Random(seed)
    for group, methods in groups.items():
        reference = methods.get("random", {})
        for method, runs in methods.items():
            if method == "random":
                continue
            shared = sorted(reference.keys() & runs.keys())
            for item in shared:
                left, right = runs[item], reference[item]
                from .protocol import role_content_identity
                if role_content_identity(left["data"]["roles"]) != role_content_identity(right["data"]["roles"]) or left["baseline_training"]["state_sha256"] != right["baseline_training"]["state_sha256"]:
                    raise ProtocolError("Paired methods must share all data roles and the trained baseline within a seed")
                if any(record.get("cache_hit") for run in (left, right) for record in run["candidates"]):
                    raise ProtocolError("Cached computation cannot support a fair wall-time comparison")
            valid = [item for item in shared if runs[item]["search"]["status"] == "succeeded" and reference[item]["search"]["status"] == "succeeded"]
            differences = [runs[item]["selection"]["hypervolume"] - reference[item]["selection"]["hypervolume"] for item in valid]
            if differences:
                means = sorted(sum(rng.choice(differences) for _ in differences) / len(differences) for _ in range(bootstrap_samples))
                interval = [means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]]
            else:
                interval = None
            success = [run["search"]["status"] == "succeeded" and any(record["configuration"]["method"] != "baseline" and
                       record["candidate_id"] in run["selection"]["feasible_ids"] for record in run["candidates"]) for run in runs.values()]
            comparisons.append({"task": group[0], "dataset_identity": json_identity(group[1]),
                                "method": method, "reference": "random", "paired_seeds": valid,
                                "all_shared_seeds": shared,
                                "hypervolume_differences": differences,
                                "mean_difference": sum(differences) / len(differences) if differences else None,
                                "bootstrap_95_percentile_interval": interval,
                                "feasible_solution_probability": sum(success) / len(success) if success else None,
                                "all_run_statuses": {str(item): run["status"] for item, run in runs.items()},
                                "all_engine_statuses": {str(item): run["search"]["status"] for item, run in runs.items()},
                                "inference": "pilot descriptive estimate; repeat count/power not established"})
    return {"comparisons": comparisons, "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed,
            "claims": "H4 is unconfirmed; time/evaluation comparisons must be reported separately"}


def canonical_protocol(protocol):
    from .protocol import canonical_json
    return canonical_json({key: value for key, value in protocol.items() if key not in ("seed", "repeat_seeds")})


def json_identity(value):
    import json
    return json.loads(value)
