"""Closed grammar operators wired into the existing FedcoreEvoOptimizer.

GOLEM owns reproduction, selection, inheritance and stopping. This adapter only
encodes choices, compiles proposals, charges attempts and calls heldout fitness.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
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

    def __post_init__(self):
        for key in ("seed", "population_size", "generations", "max_evaluations"):
            if type(getattr(self, key)) is not int or getattr(self, key) < (0 if key == "seed" else 1):
                raise ValueError(f"Invalid {key}")
        if (not _finite(self.max_seconds)
                or self.max_seconds <= 0 or type(self.mutation_only) is not bool):
            raise ValueError("Finite positive wall-time and explicit mutation_only are required")


@dataclass(frozen=True)
class GrammarFitness:
    """Finite minimization fitness from the caller's explicit heldout context."""
    quality: float
    cost: float
    quality_accepted: bool = True


class _BudgetStop(BaseException):
    pass


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


def run_grammar_search(space, config, evaluate, *, fitness_context, initial_genotypes=None):
    """Run actual FedcoreEvoOptimizer with grammar mutation and crossover.

    ``evaluate(CheckedWorkflow)`` returns GrammarFitness or an explicit failure.
    Its model, data, evaluator and optional stores remain owned by the caller.
    Failed attempts are charged and receive GOLEM null fitness; finite quality
    rejections retain their real fitness and separate acceptance metadata.
    """
    failure = validate_grammar_space(space)
    if failure is not None:
        return {"status": "structurally_invalid", "failure": failure, "charged_evaluations": 0, "trace": []}
    if (type(config) is not GrammarSearchConfig or not callable(evaluate)
            or type(fitness_context) is not str or not fitness_context):
        raise ValueError("Search config, callable evaluation and explicit heldout fitness context are required")
    from golem.core.adapter import IdentityAdapter
    from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
    from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
    from golem.core.optimisers.genetic.operators.selection import SelectionTypesEnum
    from golem.core.optimisers.objective import Objective
    from golem.core.optimisers.optimization_parameters import GraphRequirements
    from golem.core.optimisers.optimizer import GraphGenerationParams
    from fedcore.interfaces.fedcore_optimizer import FedcoreEvoOptimizer

    rng, started, trace, observations = random.Random(config.seed), time.perf_counter(), [], {}
    measured_genotypes = {}
    attempts, stop_reason = 0, "optimizer_stopped"
    def exhausted():
        return attempts >= config.max_evaluations or time.perf_counter() - started >= config.max_seconds
    def decode(graph):
        genotype = graph_genotype(graph, space)
        if isinstance(genotype, ValidationFailure):
            raise ValueError("Structurally invalid grammar graph")
        return genotype
    def mutate(graph, **kwargs):
        parent = decode(graph)
        kind = rng.choice((MutationKind.CHOICE, MutationKind.SCOPE))
        child = mutate_genotype(parent, space, kind=kind, seed=rng.randrange(2**32))
        if isinstance(child, ValidationFailure):
            raise ValueError("Invalid closed-grammar mutation")
        trace.append({"event": "variation", "operator": "grammar_mutation", "kind": kind.value,
                      "parent": genotype_payload(parent), "child": genotype_payload(child)})
        return grammar_graph(child)
    def cross(first, second, **kwargs):
        left, right = decode(first), decode(second)
        children = crossover_genotypes(left, right, space, seed=rng.randrange(2**32))
        if isinstance(children, ValidationFailure):
            raise ValueError("Invalid closed-grammar crossover")
        trace.append({"event": "variation", "operator": "grammar_crossover",
            "parents": [genotype_payload(left), genotype_payload(right)],
            "children": [genotype_payload(child) for child in children]})
        return tuple(grammar_graph(child) for child in children)
    def observed(graph):
        nonlocal attempts, stop_reason
        genotype = decode(graph)
        workflow = compile_genotype(genotype, space)
        if isinstance(workflow, ValidationFailure):
            raise ValueError("Structurally invalid compilation")
        identity = workflow.fingerprint
        if identity not in observations:
            if exhausted():
                stop_reason = "evaluation_budget" if attempts >= config.max_evaluations else "time_budget"
                raise _BudgetStop()
            attempts += 1
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
            trace.append({"event": "evaluated", "workflow_fingerprint": identity,
                "genotype": genotype_payload(genotype), "outcome": type(observation).__name__,
                "quality_accepted": observation.quality_accepted if type(observation) is GrammarFitness else None,
                "fitness": [observation.quality, observation.cost] if type(observation) is GrammarFitness else None,
                "failure": None if type(observation) is GrammarFitness else str(observation),
                "charged_evaluations": attempts})
        observation = observations[identity]
        if type(observation) is not GrammarFitness:
            # GOLEM converts this objective exception to null fitness. There is
            # no synthetic low cost, zero loss or finite penalty for a failure.
            raise ValueError(f"Execution has no fitness: {type(observation).__name__}")
        return observation

    initial = tuple(generate_genotype(space, seed=config.seed + i) for i in range(config.population_size)) if initial_genotypes is None else tuple(initial_genotypes)
    if not initial or any(isinstance(normalize_genotype(item, space), ValidationFailure) for item in initial):
        return {"status": "structurally_invalid", "charged_evaluations": 0, "trace": []}
    graphs = [grammar_graph(normalize_genotype(item, space)) for item in initial]
    objective = Objective({"heldout_quality": lambda graph: observed(graph).quality},
                          {"actual_cost": lambda graph: observed(graph).cost}, is_multi_objective=True)
    parameters = GPAlgorithmParameters(multi_objective=True, pop_size=config.population_size,
        max_pop_size=config.population_size, mutation_types=[mutate], mutation_prob=1.0,
        crossover_types=[CrossoverTypesEnum.none] if config.mutation_only else [cross],
        crossover_prob=0.0 if config.mutation_only else 1.0,
        selection_types=[SelectionTypesEnum.spea2], variable_mutation_num=False,
        max_num_of_operator_attempts=4)
    requirements = GraphRequirements(num_of_generations=config.generations,
        timeout=timedelta(seconds=config.max_seconds), n_jobs=1, show_progress=False,
        keep_history=True, history_dir=None, agent_dir=None, max_depth=1, min_arity=0,
        max_arity=0, early_stopping_iterations=config.generations)
    generation = GraphGenerationParams(adapter=IdentityAdapter(),
        rules_for_constraint=(lambda graph: not isinstance(graph_genotype(graph, space), ValidationFailure),),
        available_node_types=["compression_genotype"])
    optimizer = FedcoreEvoOptimizer(objective, graphs, requirements, generation, parameters,
        compression_grammar_profile=True)
    original_stop = optimizer.stop_optimization
    optimizer.stop_optimization = lambda: exhausted() or original_stop()
    optimizer.set_iteration_callback(lambda population, *args: trace.append({"event": "population",
        "size": len(population), "generation": optimizer.current_generation_num,
        "selection": "SPEA2", "engine": "FedcoreEvoOptimizer"}))
    status, reason, best = "succeeded", None, []
    import numpy as np
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(config.seed)
        np.random.seed(config.seed)
        best = [decode(graph) for graph in optimizer.optimise(objective)]
    except _BudgetStop:
        pass
    except Exception as error:
        status = "no_valid_fitness" if observations and not any(type(item) is GrammarFitness for item in observations.values()) else "failed"
        reason = f"{type(error).__name__}: {error}"
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    if exhausted():
        stop_reason = "evaluation_budget" if attempts >= config.max_evaluations else "time_budget"
    finite = [(identity, observation) for identity, observation in observations.items()
              if type(observation) is GrammarFitness]
    if observations and not finite:
        status = "no_valid_fitness"
        reason = "All charged proposals failed before obtaining finite heldout fitness"
    # This is a reporting archive of observed finite fitness, including when a
    # budget interrupts GOLEM mid-generation. It is not another search loop.
    archive = [(identity, observation) for identity, observation in finite if not any(
        other.quality <= observation.quality and other.cost <= observation.cost
        and (other.quality < observation.quality or other.cost < observation.cost)
        for _, other in finite)]
    best = [measured_genotypes[identity] for identity, _ in archive]
    return {"status": status, "reason": reason, "engine": "FedcoreEvoOptimizer",
        "domain": "closed frozen-M0 compression grammar", "fitness_context": fitness_context,
        "best_genotypes": best, "charged_evaluations": attempts, "trace": trace,
        "finite_archive": [{"workflow_fingerprint": identity, "quality": observation.quality,
            "cost": observation.cost, "quality_accepted": observation.quality_accepted,
            "genotype": genotype_payload(measured_genotypes[identity])} for identity, observation in archive],
        "best_role": "nondominated among evaluated finite fitness; quality acceptance is reported separately",
        "stop_reason": stop_reason, "wall_seconds": time.perf_counter() - started,
        "budget_enforcement": "before each uncached callback; in-flight callbacks may overrun",
        "quality_rejection_count": sum(type(item) is GrammarFitness and not item.quality_accepted for item in observations.values()),
        "failure_count": sum(type(item) is not GrammarFitness for item in observations.values()),
        "utility_claim": "functional grammar integration only; empirical search benefit is untested"}
