"""Deterministic shell tests: no training, solver work or wall-clock sleeps.

The script replaces only the optimiser effect boundary. Existing grammar tests
exercise actual GOLEM reproduction; these tests control the exact proposal and
metric-read sequence to verify budgets and the durable event stream.
"""
import builtins
import random
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from fedcore.algorithm.low_rank.compression_graph import CalibrationRef, ModelVersion
from fedcore.algorithm.low_rank.compression_grammar import (
    CompressionGenotype, GrammarSpace, ScopeDomain, ScopeGene, WeightedChoice,
    normalize_genotype)
from fedcore.algorithm.low_rank import grammar_search as search
from fedcore.algorithm.low_rank.plans import OperatorDescriptor


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class Graph:
    def __init__(self, genotype):
        self.genotype = genotype


@pytest.fixture
def scripted_search(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(search.time, "perf_counter", clock)
    monkeypatch.setattr(search, "grammar_graph", Graph)
    monkeypatch.setattr(search, "graph_genotype", lambda graph, space: normalize_genotype(graph.genotype, space))

    class Objective:
        def __init__(self, quality, cost, **kwargs):
            self.quality, self.cost = quality, cost

        def read(self, graph):
            return (next(iter(self.quality.values()))(graph),
                    next(iter(self.cost.values()))(graph))

    script = SimpleNamespace(run=lambda optimizer, objective: None, constructed=[])

    class Optimizer:
        def __init__(self, objective, graphs, requirements, generation, parameters, **kwargs):
            self.graphs, self.requirements, self.parameters = graphs, requirements, parameters
            self.current_generation_num = 0
            self.generations = SimpleNamespace(stagnation_iter_count=0)
            self.timer = SimpleNamespace(minutes_from_start=0)
            script.constructed.append(self)

        def stop_optimization(self):
            return False

        def set_iteration_callback(self, callback):
            self.callback = callback

        def optimise(self, objective):
            script.run(self, objective)
            return []

    definitions = {
        "golem.core.adapter": {"IdentityAdapter": SimpleNamespace},
        "golem.core.optimisers.genetic.gp_params": {"GPAlgorithmParameters": SimpleNamespace},
        "golem.core.optimisers.genetic.operators.crossover": {"CrossoverTypesEnum": SimpleNamespace(none="none")},
        "golem.core.optimisers.genetic.operators.selection": {"SelectionTypesEnum": SimpleNamespace(spea2="spea2")},
        "golem.core.optimisers.objective": {"Objective": Objective},
        "golem.core.optimisers.optimization_parameters": {"GraphRequirements": SimpleNamespace},
        "golem.core.optimisers.optimizer": {"GraphGenerationParams": SimpleNamespace},
        "fedcore.interfaces.fedcore_optimizer": {"FedcoreEvoOptimizer": Optimizer},
    }
    for name, attributes in definitions.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    alternatives = (WeightedChoice(1), WeightedChoice(2))
    space = GrammarSpace(ModelVersion("test", "a" * 64), CalibrationRef("b" * 64, (6, 4), "float32"),
        tuple(ScopeDomain(OperatorDescriptor(path, (4, 4), "Linear"), (path,), alternatives)
              for path in ("left", "right")))
    initial = tuple(CompressionGenotype((ScopeGene((path,), WeightedChoice(1)),)) for path in ("left", "right"))

    def run(evaluate=None, **kwargs):
        config = kwargs.pop("config", search.GrammarSearchConfig(population_size=2, max_evaluations=20))
        return search.run_grammar_search(space, config,
            evaluate or (lambda workflow: search.GrammarFitness(1.0, 2.0)),
            fitness_context="synthetic validation; no model execution", initial_genotypes=initial, **kwargs)

    return SimpleNamespace(clock=clock, script=script, run=run)


def test_crossover_reserves_both_children_before_operator_execution(scripted_search):
    fixture = scripted_search

    def script(optimizer, objective):
        for graph in optimizer.graphs:
            objective.read(graph)
        optimizer.parameters.crossover_types[0](*optimizer.graphs)

    fixture.script.run = script
    result = fixture.run(config=search.GrammarSearchConfig(population_size=2, max_evaluations=20, max_proposals=3))
    assert result["stop_reason"] == "proposal_budget"
    assert result["created_proposals"] == result["charged_evaluations"] == 2
    assert len([event for event in result["trace"] if event["event"] == "proposal"]) == 2
    assert not any(event.get("operator") == "grammar_crossover" for event in result["trace"])


@pytest.mark.parametrize("strict", [False, True])
def test_late_evaluation_is_charged_but_strict_mode_cannot_improve_archive(scripted_search, strict):
    fixture = scripted_search
    calls = []
    def evaluate(workflow):
        calls.append(workflow)
        if len(calls) == 2:
            fixture.clock.now = 200.0
            return search.GrammarFitness(0.01, 0.01)
        return search.GrammarFitness(1.0, 2.0)
    def script(optimizer, objective):
        objective.read(optimizer.graphs[0])
        try:
            objective.read(optimizer.graphs[1])
        except ValueError:
            assert strict
    fixture.script.run = script
    result = fixture.run(evaluate, absolute_deadline=110.0, strict_deadline=strict)
    assert result["charged_evaluations"] == 2
    assert len(result["finite_archive"]) == 1
    assert result["finite_archive"][0]["quality"] == (1.0 if strict else 0.01)
    assert result["late_evaluation_count"] == int(strict)
    assert len([event for event in result["trace"] if event["event"] == "deadline_excluded"]) == int(strict)


def test_cached_children_are_two_proposals_but_cost_reads_are_technical(scripted_search):
    fixture = scripted_search

    def script(optimizer, objective):
        for graph in optimizer.graphs:
            objective.read(graph)
        children = optimizer.parameters.crossover_types[0](*optimizer.graphs)
        for child in children:
            objective.read(child)

    fixture.script.run = script
    delivered = []
    result = fixture.run(config=search.GrammarSearchConfig(population_size=2, max_evaluations=20, max_proposals=4),
                         trace_callback=delivered.append)
    assert delivered == result["trace"]
    assert result["created_proposals"] == 4 and result["charged_evaluations"] == 2
    assert result["cache_hits"] == 2 and result["component_reads"] == 8
    assert result["stop_reason"] == "proposal_budget"
    proposals = [event for event in result["trace"] if event["event"] == "proposal"]
    assert [event["proposal_id"] for event in proposals] == [1, 2, 3, 4]
    assert [event["operator"] for event in proposals] == ["initial", "initial", "grammar_crossover", "grammar_crossover"]
    assert all(len(event["parents"]) == 2 for event in proposals[2:])
    observations = [event for event in result["trace"] if event["event"] == "cache_observation"]
    assert [event["cache_hit"] for event in observations] == [False, False, True, True]
    reads = [event for event in result["trace"] if event["event"] == "fitness_read"]
    assert sum(event["technical_read"] for event in reads) == 4
    assert result["trace"][0]["event"] == "started" and result["trace"][-1]["event"] == "stopped"


def test_mutation_is_charged_even_if_its_result_has_been_seen(scripted_search):
    fixture = scripted_search

    def script(optimizer, objective):
        for graph in optimizer.graphs:
            objective.read(graph)
        changed = optimizer.parameters.mutation_types[0](optimizer.graphs[0])
        objective.read(changed)

    fixture.script.run = script
    result = fixture.run(config=search.GrammarSearchConfig(population_size=2, max_evaluations=20, max_proposals=3))
    assert result["created_proposals"] == 3
    mutations = [event for event in result["trace"] if event["event"] == "proposal" and event["operator"] == "grammar_mutation"]
    assert len(mutations) == 1 and len(mutations[0]["parents"]) == 1
    assert mutations[0]["kind"] in ("scope", "choice")


def test_new_attempt_cap_is_charged_before_callback_and_not_twice_for_components(scripted_search):
    fixture = scripted_search
    fixture.script.run = lambda optimizer, objective: [objective.read(graph) for graph in optimizer.graphs]
    called = []
    result = fixture.run(evaluate=lambda workflow: called.append(workflow) or search.GrammarFitness(1, 2),
        config=search.GrammarSearchConfig(population_size=2, max_evaluations=1))
    assert len(called) == result["charged_evaluations"] == 1
    assert result["created_proposals"] == 2 and result["stop_reason"] == "evaluation_budget"
    assert len([event for event in result["trace"] if event["event"] == "evaluated"]) == 1


def test_absolute_deadline_includes_internal_imports(scripted_search, monkeypatch):
    fixture = scripted_search
    original_import = builtins.__import__

    def delayed_import(name, *args, **kwargs):
        if name == "fedcore.interfaces.fedcore_optimizer":
            fixture.clock.now += 3
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", delayed_import)
    result = fixture.run(absolute_deadline=102)
    assert result["stop_reason"] == "time_budget"
    assert result["created_proposals"] == result["charged_evaluations"] == 0
    assert not fixture.script.constructed
    assert result["absolute_deadline"] == 102


def test_expired_inflight_result_has_time_evidence_and_cannot_start_next_assessment(scripted_search):
    fixture = scripted_search
    fixture.script.run = lambda optimizer, objective: [objective.read(graph) for graph in optimizer.graphs]

    def evaluate(workflow):
        fixture.clock.now += 6
        return search.GrammarFitness(1, 2)

    result = fixture.run(evaluate=evaluate, absolute_deadline=105)
    assert result["stop_reason"] == "time_budget" and result["charged_evaluations"] == 1
    evaluated, = [event for event in result["trace"] if event["event"] == "evaluated"]
    assert evaluated["started_at"] == 100 and evaluated["ended_at"] == 106
    assert not evaluated["completed_within_deadline"] and evaluated["deadline_overrun_seconds"] == 1
    assert result["finite_archive"]  # The caller reconstructs the 30-second archive by completion time.
    assert fixture.script.constructed[0].requirements.timeout.total_seconds() == 5


def test_unknown_evaluation_error_remains_typed_and_never_finite(scripted_search):
    fixture = scripted_search
    fixture.script.run = lambda optimizer, objective: objective.read(optimizer.graphs[0])

    def evaluate(workflow):
        raise RuntimeError("unexpected evaluator failure")

    result = fixture.run(evaluate=evaluate)
    event, = [event for event in result["trace"] if event["event"] == "evaluated"]
    assert event["outcome"] == "CallbackFailure"
    assert event["failure_details"] == {"code": "RuntimeError", "message": "unexpected evaluator failure"}
    assert event["fitness"] is None and not result["finite_archive"]
    assert result["charged_evaluations"] == result["failure_count"] == 1


def test_repeated_failed_observations_are_not_reported_as_useful_cache_hits(scripted_search):
    fixture = scripted_search

    def script(optimizer, objective):
        for graph in optimizer.graphs:
            with pytest.raises(ValueError, match="no fitness"):
                objective.read(graph)
        for graph in optimizer.parameters.crossover_types[0](*optimizer.graphs):
            with pytest.raises(ValueError, match="no fitness"):
                objective.read(graph)

    fixture.script.run = script
    result = fixture.run(evaluate=lambda workflow: search.NumericalFailure("domain", "unsupported numeric input"),
        config=search.GrammarSearchConfig(population_size=2, max_evaluations=20, max_proposals=4))
    assert result["charged_evaluations"] == 2 and result["created_proposals"] == 4
    assert result["cache_hits"] == 0 and result["repeated_failures"] == 2
    cache_events = [event for event in result["trace"] if event["event"] == "cache_observation"]
    assert [event["prior_observation"] for event in cache_events] == [False, False, True, True]
    assert not any(event["cache_hit"] for event in cache_events)


def test_fatal_parent_signal_propagates_and_restores_global_rng(scripted_search):
    fixture = scripted_search
    fixture.script.run = lambda optimizer, objective: objective.read(optimizer.graphs[0])

    class PackageFatal(BaseException):
        pass

    def evaluate(workflow):
        random.random()
        np.random.random()
        raise PackageFatal("stop the package")

    python_before, numpy_before = random.getstate(), np.random.get_state()
    with pytest.raises(PackageFatal, match="stop the package"):
        fixture.run(evaluate=evaluate)
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0] and np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]


def test_trace_sink_failure_stays_explicit_and_prevents_evaluation(scripted_search):
    def sink(event):
        if event["event"] == "proposal":
            raise OSError("cannot persist ledger")

    result = scripted_search.run(trace_callback=sink)
    assert result["status"] == "failed" and result["stop_reason"] == "trace_callback_failure"
    assert result["trace_failure"] == {"code": "OSError", "message": "cannot persist ledger"}
    assert result["charged_evaluations"] == 0
    assert result["trace"][-1]["event"] == "stopped"


def test_real_golem_engine_preserves_proposal_identity_through_graph_copies():
    """Exercise real reproduction with a synthetic fitness and no model solver."""
    from copy import deepcopy
    from fedcore.algorithm.low_rank.compression_grammar import ASVDChoice, AffinePCAChoice

    choices = (WeightedChoice(1), WeightedChoice(2), ASVDChoice(1), AffinePCAChoice(1))
    space = GrammarSpace(ModelVersion("observability-real-engine", "a" * 64),
        CalibrationRef("b" * 64, (6, 4), "float32"),
        tuple(ScopeDomain(OperatorDescriptor(path, (4, 4), "Linear"), (path,), choices)
              for path in ("left", "right")))
    initial = tuple(CompressionGenotype((ScopeGene(("left",), choice),)) for choice in choices)
    graph = search.grammar_graph(initial[0])
    graph._compression_proposal_id = 17
    assert deepcopy(graph)._compression_proposal_id == 17
    calls, events = [], []

    def evaluate(workflow):
        calls.append(workflow)
        quality = sum(gene.choice.rank + {WeightedChoice: 0, ASVDChoice: .1,
                     AffinePCAChoice: .2}[type(gene.choice)] for gene in workflow.genotype.genes)
        return search.GrammarFitness(quality, 10 / quality)

    result = search.run_grammar_search(space,
        search.GrammarSearchConfig(seed=13, population_size=4, generations=2,
            max_evaluations=8, max_proposals=20, max_seconds=20), evaluate,
        fitness_context="synthetic validation; actual FedcoreEvoOptimizer reproduction",
        initial_genotypes=initial, trace_callback=events.append)
    assert result["status"] == "succeeded", result
    assert result["engine"] == "FedcoreEvoOptimizer"
    assert events == result["trace"]
    proposals = {event["proposal_id"]: event for event in events if event["event"] == "proposal"}
    assert len(proposals) == result["created_proposals"] <= 20
    assert len(calls) == result["charged_evaluations"] <= 8
    assert any(event["operator"] == "grammar_crossover" for event in proposals.values())
    assert any(event["operator"] == "grammar_mutation" for event in proposals.values())
    for event in events:
        if event["event"] == "evaluated":
            assert event["proposal_id"] in proposals
            assert event["canonical_solution_id"] == proposals[event["proposal_id"]]["canonical_solution_id"]
    assert result["cache_hits"] == sum(event["cache_hit"] for event in events if event["event"] == "cache_observation")


@pytest.mark.parametrize("value", (0, -1, True, 2.5))
def test_proposal_limit_validation_and_default_backward_compatibility(value):
    assert search.GrammarSearchConfig().max_proposals is None
    with pytest.raises(ValueError, match="max_proposals"):
        search.GrammarSearchConfig(max_proposals=value)
