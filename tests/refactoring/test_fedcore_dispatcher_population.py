"""Population conservation through the actual FedcoreDispatcher boundary."""
from types import SimpleNamespace

import pytest

from golem.core.adapter import IdentityAdapter
from golem.core.optimisers.fitness.multi_objective_fitness import MultiObjFitness
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.opt_history_objects.individual import GraphEvalResult, Individual

from fedcore.interfaces import fedcore_dispatcher as dispatcher_module
from fedcore.interfaces.fedcore_dispatcher import FedcoreDispatcher


def individual(name, *, valid=False):
    graph = OptGraph([OptNode(name)])
    item = Individual(graph)
    if valid:
        item.set_evaluation_result(MultiObjFitness((1., 2.)))
    return item


def result_for(item):
    return GraphEvalResult(item.uid, MultiObjFitness((.5, 3.)), item.graph,
                           {"source": "deterministic dispatcher fixture"})


@pytest.fixture
def dispatcher(monkeypatch):
    instance = FedcoreDispatcher(adapter=IdentityAdapter(), n_jobs=1)
    monkeypatch.setattr(dispatcher_module.MemoryAnalytics, "log", lambda *args, **kwargs: None)
    info = []
    monkeypatch.setattr(instance, "population_evaluation_info", lambda **kwargs: info.append(kwargs))
    return SimpleNamespace(instance=instance, info=info)


def assert_same_objects(actual, expected):
    assert len(actual) == len(expected)
    assert all(item is original for item, original in zip(actual, expected))


def test_all_already_valid_individuals_survive_without_re_evaluation(dispatcher, monkeypatch):
    instance = dispatcher.instance
    population = [individual(name, valid=True) for name in ("a", "b", "c", "d")]
    evaluated = []
    monkeypatch.setattr(instance, "_multithread_eval", lambda items: evaluated.append(list(items)) or [])
    monkeypatch.setattr(instance, "_eval_at_least_one", lambda items: pytest.fail("Valid individuals must not be retried"))
    fitness_before = [item.fitness for item in population]
    actual = instance.evaluate_population(population)
    assert_same_objects(actual, population)
    assert evaluated == [[]]
    assert all(item.fitness is fitness for item, fitness in zip(actual, fitness_before))
    assert dispatcher.info == [{"evaluated_pop_size": 4, "pop_size": 4}]


def test_mixed_population_preserves_both_partitions_uid_matching_and_order(dispatcher, monkeypatch):
    instance = dispatcher.instance
    skipped_b, skipped_a = individual("skip-b", valid=True), individual("skip-a", valid=True)
    new_a, new_b, failed = individual("new-a"), individual("new-b"), individual("failed")
    population = [skipped_b, new_a, skipped_a, new_b, failed]
    evaluated = []

    def evaluate(items):
        evaluated.append(list(items))
        # Results may arrive in another order; application is matched by UID.
        return [result_for(new_b), None, result_for(new_a)]

    monkeypatch.setattr(instance, "_multithread_eval", evaluate)
    monkeypatch.setattr(instance, "_eval_at_least_one", lambda items: pytest.fail("A nonempty valid population must not be retried"))
    actual = instance.evaluate_population(population)
    assert_same_objects(evaluated[0], [new_a, new_b, failed])
    # Preserve the upstream dispatcher's contract: newly evaluated, then
    # already valid; input order remains stable inside each partition.
    assert_same_objects(actual, [new_a, new_b, skipped_b, skipped_a])
    assert not failed.fitness.valid
    assert new_a.fitness.valid and new_b.fitness.valid
    assert dispatcher.info == [{"evaluated_pop_size": 4, "pop_size": 5}]


def test_all_new_evaluation_failures_do_not_discard_valid_skipped_members(dispatcher, monkeypatch):
    instance = dispatcher.instance
    failed, retained = individual("failed"), individual("retained", valid=True)
    monkeypatch.setattr(instance, "_multithread_eval", lambda items: [None])
    monkeypatch.setattr(instance, "_eval_at_least_one", lambda items: pytest.fail("A retained valid member already supplies fitness"))
    assert_same_objects(instance.evaluate_population([failed, retained]), [retained])


def test_all_new_successes_keep_input_order_without_fallback(dispatcher, monkeypatch):
    instance = dispatcher.instance
    population = [individual(name) for name in ("a", "b", "c")]
    monkeypatch.setattr(instance, "_multithread_eval", lambda items: [result_for(item) for item in reversed(items)])
    monkeypatch.setattr(instance, "_eval_at_least_one", lambda items: pytest.fail("Successful new evaluations need no retry"))
    assert_same_objects(instance.evaluate_population(population), population)


def test_all_new_failures_retry_original_individuals_until_first_success(dispatcher, monkeypatch):
    instance = dispatcher.instance
    population = [individual(name) for name in ("fails-again", "recovers", "not-retried")]
    monkeypatch.setattr(instance, "_multithread_eval", lambda items: [None] * len(items))
    retries = []

    def evaluate_single(owner, *, graph, uid_of_individual, with_time_limit):
        retries.append((uid_of_individual, graph, with_time_limit))
        return result_for(population[1]) if uid_of_individual == population[1].uid else None

    monkeypatch.setattr(instance, "fedcore_evaluate_single", evaluate_single)
    actual = instance.evaluate_population(population)
    assert_same_objects(actual, [population[1]])
    assert [uid for uid, _, _ in retries] == [population[0].uid, population[1].uid]
    assert all(not with_time_limit for _, _, with_time_limit in retries)
    assert retries[0][1] is population[0].graph and retries[1][1] is population[1].graph
    assert not population[0].fitness.valid and population[1].fitness.valid
    assert not population[2].fitness.valid


def test_unrecoverable_and_empty_populations_return_lists(dispatcher, monkeypatch):
    instance = dispatcher.instance
    population = [individual("fails")]
    monkeypatch.setattr(instance, "_multithread_eval", lambda items: [None] * len(items))
    retries = []
    monkeypatch.setattr(instance, "fedcore_evaluate_single", lambda owner, **kwargs: retries.append(kwargs["uid_of_individual"]))
    assert instance.evaluate_population(population) == []
    assert retries == [population[0].uid]
    assert instance.evaluate_population([]) == []
    assert retries == [population[0].uid]


@pytest.mark.parametrize("boundary", ("initial_evaluation", "fallback"))
def test_parent_budget_signal_propagates_through_dispatcher(dispatcher, monkeypatch, boundary):
    instance = dispatcher.instance
    population = [individual("budget-stop")]

    class ParentBudgetStop(BaseException):
        pass

    def stop(*args, **kwargs):
        raise ParentBudgetStop("original package allowance exhausted")

    monkeypatch.setattr(instance, "_multithread_eval",
        stop if boundary == "initial_evaluation" else lambda items: [None] * len(items))
    monkeypatch.setattr(instance, "fedcore_evaluate_single", stop)
    with pytest.raises(ParentBudgetStop, match="original package allowance"):
        instance.evaluate_population(population)


def test_real_g3_retains_later_generations_and_reaches_beyond_eight_finite_evaluations():
    """Real GOLEM/SPEA2, synthetic finite fitness, no model transformations."""
    from fedcore.algorithm.low_rank.compression_graph import CalibrationRef, ModelVersion
    from fedcore.algorithm.low_rank.compression_grammar import GrammarSpace, ScopeDomain
    from fedcore.algorithm.low_rank.grammar_search import GrammarFitness, GrammarSearchConfig, run_grammar_search
    from fedcore.algorithm.low_rank.plans import OperatorDescriptor
    from fedcore.experiments.compression_search_protocol import (
        INITIAL_GENOTYPES, TUNED_CATALOG, canonical_solution_id)

    space = GrammarSpace(ModelVersion("dispatcher-multigeneration-regression", "a" * 64),
        CalibrationRef("b" * 64, (6, 30), "float32"), (
            ScopeDomain(OperatorDescriptor("0", (128, 30), "Linear"), ("0",), TUNED_CATALOG.alternatives),
            ScopeDomain(OperatorDescriptor("2", (64, 128), "Linear"), ("2",), TUNED_CATALOG.alternatives)))
    calls = []

    def evaluate(workflow):
        identity = canonical_solution_id(workflow.genotype)
        calls.append(identity)
        quality = (int(identity[:12], 16) + 1) / (16**12 + 1)
        # Every distinct fixture outcome is finite and nondominated, so the
        # regression tests population flow rather than a quality preference.
        return GrammarFitness(quality, 1.0 - quality)

    outcome = run_grammar_search(space, GrammarSearchConfig(seed=11, population_size=4,
        generations=64, max_evaluations=12, max_proposals=96, max_seconds=30), evaluate,
        fitness_context="synthetic validation for dispatcher population conservation",
        initial_genotypes=INITIAL_GENOTYPES)
    assert outcome["status"] == "succeeded", outcome
    assert len(calls) == len(set(calls)) == outcome["charged_evaluations"] == 12
    assert outcome["stop_reason"] == "evaluation_budget"
    populations = [event for event in outcome["trace"] if event["event"] == "population"]
    assert any(event["generation"] >= 2 and event["size"] >= 4 for event in populations)
    assert all(event["size"] > 0 for event in populations)
