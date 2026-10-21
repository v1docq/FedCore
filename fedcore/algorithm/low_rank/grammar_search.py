"""Closed grammar operators wired into the existing FedcoreEvoOptimizer.

GOLEM owns reproduction, selection, inheritance and stopping. This adapter only
encodes choices, compiles proposals, charges attempts and calls heldout fitness.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
from datetime import timedelta
import hashlib
import json
import math
import random
import time

from .compression_grammar import (CompressionGenotype, GrammarSpace, MutationKind,
    compile_genotype, crossover_genotypes, generate_genotype, genotype_payload,
    mutate_genotype, normalize_genotype, validate_grammar_space)
from .grammar_execution import (CallbackFailure, NumericalFailure, ResourceFailure,
    StructuralInvalid)
from .plans import ValidationFailure, Violation


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class GrammarSearchConfig:
    seed: int = 0
    population_size: int = 4
    generations: int = 3
    max_evaluations: int = 12
    max_seconds: float = 60.0
    mutation_only: bool = False
    max_proposals: int | None = None

    def __post_init__(self):
        for key in ("seed", "population_size", "generations", "max_evaluations"):
            if type(getattr(self, key)) is not int or getattr(self, key) < (0 if key == "seed" else 1):
                raise ValueError(f"Invalid {key}")
        if (not _finite(self.max_seconds)
                or self.max_seconds <= 0 or type(self.mutation_only) is not bool):
            raise ValueError("Finite positive wall-time and explicit mutation_only are required")
        if self.max_proposals is not None and (type(self.max_proposals) is not int or self.max_proposals < 1):
            raise ValueError("max_proposals must be None or a positive integer")


@dataclass(frozen=True)
class GrammarFitness:
    """Finite minimization fitness from the caller's explicit heldout context."""
    quality: float
    cost: float
    quality_accepted: bool = True


class _BudgetStop(BaseException):
    pass


class _TraceCallbackStop(BaseException):
    def __init__(self, failure):
        self.failure = failure


@dataclass
class _SearchBudget:
    """Owned counters; reserving crossover's two children is atomic."""
    max_evaluations: int
    max_proposals: int | None
    deadline: float
    attempts: int = 0
    proposals: int = 0
    stopped_by: str | None = None

    def reason(self, now, *, include_proposals=True):
        if self.stopped_by is not None:
            return self.stopped_by
        if self.attempts >= self.max_evaluations:
            return "evaluation_budget"
        if now >= self.deadline:
            return "time_budget"
        if include_proposals and self.max_proposals is not None and self.proposals >= self.max_proposals:
            return "proposal_budget"
        return None

    def reserve_proposals(self, count, now):
        reason = self.reason(now)
        if reason is None and self.max_proposals is not None and self.proposals + count > self.max_proposals:
            reason = "proposal_budget"
        if reason is not None:
            self.stopped_by = reason
            raise _BudgetStop()
        first = self.proposals + 1
        self.proposals += count
        return tuple(range(first, self.proposals + 1))

    def charge_evaluation(self, now):
        # Reaching the proposal cap still permits assessment of already-created
        # children. Only creation of another proposal is forbidden by that cap.
        reason = self.reason(now, include_proposals=False)
        if reason is not None:
            self.stopped_by = reason
            raise _BudgetStop()
        self.attempts += 1
        return self.attempts


def _solution_id(genotype):
    payload = genotype_payload(genotype)
    payload["genes"].sort(key=lambda item: tuple(item["aliases"]))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False,
        separators=(",", ":")).encode()).hexdigest()


def grammar_graph(genotype):
    """One native GOLEM node carries only an immutable genotype."""
    from golem.core.optimisers.graph import OptGraph, OptNode
    return OptGraph([OptNode({"name": "compression_genotype", "params": {"genotype": genotype}})])


def graph_genotype(graph, space):
    """Checked native adapter boundary; no unknown params or topology allowed."""
    try:
        if (len(graph.nodes) != 1 or graph.nodes[0].nodes_from
                or set(graph.nodes[0].parameters) != {"genotype"}
                or graph.nodes[0].name != "compression_genotype"):
            return ValidationFailure((Violation("InvalidGrammarGraph", "graph", "Exactly one named genotype node with no parents or unrelated parameters is required"),))
        return normalize_genotype(graph.nodes[0].parameters["genotype"], space)
    except (AttributeError, TypeError, KeyError):
        return ValidationFailure((Violation("InvalidGrammarGraph", "graph", "A native one-node GOLEM grammar graph is required"),))


def run_grammar_search(space, config, evaluate, *, fitness_context, initial_genotypes=None,
                       absolute_deadline=None, trace_callback=None, phase_callback=None):
    """Run actual FedcoreEvoOptimizer with grammar mutation and crossover.

    ``evaluate(CheckedWorkflow)`` returns GrammarFitness or an explicit failure.
    Its model, data, evaluator and optional stores remain owned by the caller.
    Failed attempts are charged and receive GOLEM null fitness; finite quality
    rejections retain their real fitness and separate acceptance metadata.

    ``absolute_deadline`` uses ``time.perf_counter`` in the caller's process;
    it includes work before this entry point, including the imports below.
    ``trace_callback(event)`` receives each event immediately. Ordinary callback
    exceptions remain typed failures; a caller-owned BaseException can signal a
    fatal package stop. The default configuration preserves the old limits.
    """
    failure = validate_grammar_space(space)
    if failure is not None:
        return {"status": "structurally_invalid", "failure": failure, "charged_evaluations": 0, "trace": []}
    if (type(config) is not GrammarSearchConfig or not callable(evaluate)
            or type(fitness_context) is not str or not fitness_context):
        raise ValueError("Search config, callable evaluation and explicit heldout fitness context are required")
    if absolute_deadline is not None and not _finite(absolute_deadline):
        raise ValueError("absolute_deadline must be a finite perf_counter timestamp")
    if trace_callback is not None and not callable(trace_callback):
        raise ValueError("trace_callback must be callable or None")
    if phase_callback is not None and not callable(phase_callback):
        raise ValueError("phase_callback must be callable or None")
    from golem.core.adapter import IdentityAdapter
    from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
    from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
    from golem.core.optimisers.genetic.operators.selection import SelectionTypesEnum
    from golem.core.optimisers.objective import Objective
    from golem.core.optimisers.optimization_parameters import GraphRequirements
    from golem.core.optimisers.optimizer import GraphGenerationParams
    from fedcore.interfaces.fedcore_optimizer import FedcoreEvoOptimizer
    import numpy as np

    rng, started, trace, observations = random.Random(config.seed), time.perf_counter(), [], {}
    deadline = min(started + config.max_seconds, absolute_deadline) if absolute_deadline is not None else started + config.max_seconds
    budget = _SearchBudget(config.max_evaluations, config.max_proposals, deadline)
    measured_genotypes, proposal_metadata, latest_proposals, reads = {}, {}, {}, {}
    cache_hits, component_reads, repeated_failures = 0, 0, 0
    stop_reason, optimizer, trace_failure = "optimizer_stopped", None, None
    callback_working = True

    @contextmanager
    def phase(name, **metadata):
        first = time.perf_counter() if phase_callback is not None else None
        try:
            yield
        finally:
            if phase_callback is not None:
                phase_callback({"phase": name, "started_at": first,
                    "completed_at": time.perf_counter(), **metadata})

    def emit(event, **values):
        nonlocal callback_working
        timestamp = time.perf_counter()
        record = {"event": event, "timestamp": timestamp, "elapsed_seconds": timestamp - started, **values}
        trace.append(record)
        if trace_callback is not None and callback_working:
            try:
                trace_callback(dict(record))
            except Exception as error:
                callback_working = False
                raise _TraceCallbackStop(CallbackFailure(type(error).__name__, str(error)))
        return record

    def exhausted():
        return budget.reason(time.perf_counter()) is not None

    def decode(graph):
        genotype = graph_genotype(graph, space)
        if isinstance(genotype, ValidationFailure):
            raise ValueError("Structurally invalid grammar graph")
        return genotype

    def proposal(genotype, operator, parents, number, *, operator_started, kind=None):
        identity = _solution_id(genotype)
        metadata = {"proposal_id": number, "proposal_number": number,
            "canonical_solution_id": identity, "genotype": genotype_payload(genotype),
            "operator": operator, "parents": parents, "kind": kind,
            "started_at": operator_started, "ended_at": time.perf_counter()}
        proposal_metadata[number] = metadata
        latest_proposals[identity] = number
        with phase("proposal_graph_creation", proposal_id=number):
            graph = grammar_graph(genotype)
        # GOLEM copies graph objects during variation; this metadata follows
        # those copies without changing the checked one-node genotype payload.
        graph._compression_proposal_id = number
        emit("proposal", **metadata, created_proposals=budget.proposals)
        return graph

    def mutate(graph, **kwargs):
        operator_started = time.perf_counter()
        number, = budget.reserve_proposals(1, operator_started)
        parent = decode(graph)
        kind = rng.choice((MutationKind.CHOICE, MutationKind.SCOPE))
        child = mutate_genotype(parent, space, kind=kind, seed=rng.randrange(2**32))
        if isinstance(child, ValidationFailure):
            raise ValueError("Invalid closed-grammar mutation")
        emit("variation", operator="grammar_mutation", kind=kind.value,
             parent=genotype_payload(parent), child=genotype_payload(child))
        return proposal(child, "grammar_mutation", [_solution_id(parent)], number,
                        operator_started=operator_started, kind=kind.value)

    def cross(first, second, **kwargs):
        operator_started = time.perf_counter()
        numbers = budget.reserve_proposals(2, operator_started)
        left, right = decode(first), decode(second)
        children = crossover_genotypes(left, right, space, seed=rng.randrange(2**32))
        if isinstance(children, ValidationFailure):
            raise ValueError("Invalid closed-grammar crossover")
        emit("variation", operator="grammar_crossover",
            parents=[genotype_payload(left), genotype_payload(right)],
            children=[genotype_payload(child) for child in children])
        return tuple(proposal(child, "grammar_crossover", [_solution_id(left), _solution_id(right)],
                     number, operator_started=operator_started) for child, number in zip(children, numbers))

    def _observed(graph, component):
        nonlocal cache_hits, component_reads, repeated_failures
        genotype = decode(graph)
        with phase("candidate_compilation", component=component):
            workflow = compile_genotype(genotype, space)
        if isinstance(workflow, ValidationFailure):
            raise ValueError("Structurally invalid compilation")
        identity = workflow.fingerprint
        solution_id = _solution_id(genotype)
        number = getattr(graph, "_compression_proposal_id", latest_proposals.get(solution_id))
        if number not in proposal_metadata or proposal_metadata[number]["canonical_solution_id"] != solution_id:
            number = latest_proposals.get(solution_id)
        if number is None:
            raise ValueError("An assessed graph must originate from a recorded grammar proposal")
        first_read = number not in reads
        reads[number] = reads.get(number, 0) + 1
        component_reads += 1
        cached = identity in observations
        if first_read:
            successful_hit = cached and type(observations[identity]) is GrammarFitness
            cache_hits += int(successful_hit)
            repeated_failures += int(cached and not successful_hit)
            emit("cache_observation", proposal_id=number, canonical_solution_id=solution_id,
                 workflow_fingerprint=identity, cache_hit=successful_hit,
                 prior_observation=cached, repeated_failure=cached and not successful_hit,
                 charged_evaluations=budget.attempts, cache_hits=cache_hits,
                 repeated_failures=repeated_failures)
        emit("fitness_read", proposal_id=number, workflow_fingerprint=identity,
             component=component, technical_read=not first_read,
             component_reads=component_reads)
        if identity not in observations:
            attempt = budget.charge_evaluation(time.perf_counter())
            evaluation_started = time.perf_counter()
            emit("evaluation_started", proposal_id=number, canonical_solution_id=solution_id,
                 workflow_fingerprint=identity, genotype=genotype_payload(genotype),
                 attempt_id=attempt, charged_evaluations=attempt, started_at=evaluation_started)
            try:
                observation = evaluate(workflow)
            except Exception as error:
                observation = CallbackFailure(type(error).__name__, str(error))
            if type(observation) is GrammarFitness:
                if (any(not _finite(value) for value in (observation.quality, observation.cost))
                        or type(observation.quality_accepted) is not bool):
                    observation = NumericalFailure("NonfiniteFitness", "The heldout quality and cost must be finite")
            elif type(observation) not in (StructuralInvalid, NumericalFailure, ResourceFailure, CallbackFailure):
                observation = CallbackFailure("InvalidEvaluationResult", "Return GrammarFitness or a typed execution failure")
            observations[identity] = observation
            measured_genotypes[identity] = genotype
            evaluation_ended = time.perf_counter()
            emit("evaluated", workflow_fingerprint=identity, proposal_id=number,
                canonical_solution_id=solution_id, attempt_id=attempt,
                genotype=genotype_payload(genotype), outcome=type(observation).__name__,
                quality_accepted=observation.quality_accepted if type(observation) is GrammarFitness else None,
                fitness=[observation.quality, observation.cost] if type(observation) is GrammarFitness else None,
                failure=None if type(observation) is GrammarFitness else str(observation),
                failure_details=None if type(observation) is GrammarFitness else asdict(observation),
                charged_evaluations=attempt, started_at=evaluation_started, ended_at=evaluation_ended,
                completed_within_deadline=evaluation_ended <= deadline,
                deadline_overrun_seconds=max(0.0, evaluation_ended - deadline))
        observation = observations[identity]
        if type(observation) is not GrammarFitness:
            # GOLEM converts this objective exception to null fitness. There is
            # no synthetic low cost, zero loss or finite penalty for a failure.
            raise ValueError(f"Execution has no fitness: {type(observation).__name__}")
        return observation

    def observed(graph, component):
        with phase("objective_component", component=component):
            return _observed(graph, component)

    initial = tuple(generate_genotype(space, seed=config.seed + i) for i in range(config.population_size)) if initial_genotypes is None else tuple(initial_genotypes)
    if not initial or any(isinstance(normalize_genotype(item, space), ValidationFailure) for item in initial):
        return {"status": "structurally_invalid", "charged_evaluations": 0, "trace": []}
    status, reason, best = "succeeded", None, []
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(config.seed)
        np.random.seed(config.seed)
        emit("started", absolute_deadline=deadline, cache_entries=0,
             max_evaluations=config.max_evaluations, max_proposals=config.max_proposals)
        graphs = []
        for item in initial:
            operator_started = time.perf_counter()
            number, = budget.reserve_proposals(1, operator_started)
            graphs.append(proposal(normalize_genotype(item, space), "initial", [], number,
                                   operator_started=operator_started))
        objective = Objective({"heldout_quality": lambda graph: observed(graph, "quality").quality},
                              {"actual_cost": lambda graph: observed(graph, "cost").cost}, is_multi_objective=True)
        parameters = GPAlgorithmParameters(multi_objective=True, pop_size=config.population_size,
            max_pop_size=config.population_size, mutation_types=[mutate], mutation_prob=1.0,
            crossover_types=[CrossoverTypesEnum.none] if config.mutation_only else [cross],
            crossover_prob=0.0 if config.mutation_only else 1.0,
            selection_types=[SelectionTypesEnum.spea2], variable_mutation_num=False,
            max_num_of_operator_attempts=4)
        # Recompute here, after internal imports and initial graph preparation.
        # GOLEM's timer may start later; the absolute guard still runs before
        # every new assessment and every variation operator.
        remaining_seconds = deadline - time.perf_counter()
        if remaining_seconds <= 0:
            budget.stopped_by = "time_budget"
            raise _BudgetStop()
        requirements = GraphRequirements(num_of_generations=config.generations,
            timeout=timedelta(seconds=remaining_seconds), n_jobs=1, show_progress=False,
            keep_history=True, history_dir=None, agent_dir=None, max_depth=1, min_arity=0,
            max_arity=0, early_stopping_iterations=config.generations)
        generation = GraphGenerationParams(adapter=IdentityAdapter(),
            rules_for_constraint=(lambda graph: not isinstance(graph_genotype(graph, space), ValidationFailure),),
            available_node_types=["compression_genotype"])
        optimizer = FedcoreEvoOptimizer(objective, graphs, requirements, generation, parameters,
            compression_grammar_profile=True)
        original_stop = optimizer.stop_optimization
        optimizer.stop_optimization = lambda: exhausted() or original_stop()
        optimizer.set_iteration_callback(lambda population, *args: emit("population",
            size=len(population), generation=optimizer.current_generation_num,
            selection="SPEA2", engine="FedcoreEvoOptimizer"))
        with phase("optimizer_call"):
            best = [decode(graph) for graph in optimizer.optimise(objective)]
    except _BudgetStop:
        pass
    except _TraceCallbackStop as error:
        status, stop_reason = "failed", "trace_callback_failure"
        trace_failure = error.failure
        reason = f"{error.failure.code}: {error.failure.message}"
    except Exception as error:
        status = "no_valid_fitness" if observations and not any(type(item) is GrammarFitness for item in observations.values()) else "failed"
        stop_reason = "optimizer_failure"
        reason = f"{type(error).__name__}: {error}"
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    budget_reason = budget.reason(time.perf_counter())
    if trace_failure is None and budget_reason is not None:
        stop_reason = budget_reason
    elif trace_failure is None and optimizer is not None and status == "succeeded":
        if optimizer.current_generation_num >= config.generations:
            stop_reason = "generation_limit"
        elif getattr(optimizer.generations, "stagnation_iter_count", 0) >= config.generations:
            stop_reason = "stagnation"
        elif optimizer.timer.minutes_from_start * 60 >= remaining_seconds:
            stop_reason = "time_budget"
    finite = [(identity, observation) for identity, observation in observations.items()
              if type(observation) is GrammarFitness]
    if observations and not finite and trace_failure is None:
        status = "no_valid_fitness"
        reason = "All charged proposals failed before obtaining finite heldout fitness"
    # This is a reporting archive of observed finite fitness, including when a
    # budget interrupts GOLEM mid-generation. It is not another search loop.
    archive = [(identity, observation) for identity, observation in finite if not any(
        other.quality <= observation.quality and other.cost <= observation.cost
        and (other.quality < observation.quality or other.cost < observation.cost)
        for _, other in finite)]
    best = [measured_genotypes[identity] for identity, _ in archive]
    # A failed trace sink cannot be called again, but the in-memory terminal
    # record still preserves the typed failure and final counters.
    try:
        emit("stopped", stop_reason=stop_reason, status=status,
             charged_evaluations=budget.attempts, created_proposals=budget.proposals,
             cache_entries=len(observations), cache_hits=cache_hits,
             component_reads=component_reads, repeated_failures=repeated_failures,
             failure_details=asdict(trace_failure) if trace_failure is not None else None)
    except _TraceCallbackStop as error:
        trace_failure = error.failure
        status, stop_reason = "failed", "trace_callback_failure"
        reason = f"{error.failure.code}: {error.failure.message}"
        trace[-1].update(status=status, stop_reason=stop_reason,
                         failure_details=asdict(trace_failure))
    return {"status": status, "reason": reason, "engine": "FedcoreEvoOptimizer",
        "domain": "closed frozen-M0 compression grammar", "fitness_context": fitness_context,
        "best_genotypes": best, "charged_evaluations": budget.attempts, "trace": trace,
        "created_proposals": budget.proposals, "cache_hits": cache_hits,
        "repeated_failures": repeated_failures,
        "component_reads": component_reads, "cache_entries": len(observations),
        "trace_failure": asdict(trace_failure) if trace_failure is not None else None,
        "finite_archive": [{"workflow_fingerprint": identity, "quality": observation.quality,
            "cost": observation.cost, "quality_accepted": observation.quality_accepted,
            "genotype": genotype_payload(measured_genotypes[identity])} for identity, observation in archive],
        "best_role": "nondominated among evaluated finite fitness; quality acceptance is reported separately",
        "stop_reason": stop_reason, "wall_seconds": time.perf_counter() - started,
        "started_at": started, "ended_at": time.perf_counter(), "absolute_deadline": deadline,
        "budget_enforcement": "before each uncached callback; in-flight callbacks may overrun",
        "quality_rejection_count": sum(type(item) is GrammarFitness and not item.quality_accepted for item in observations.values()),
        "failure_count": sum(type(item) is not GrammarFitness for item in observations.values()),
        "utility_claim": "functional grammar integration only; empirical search benefit is untested"}
