"""Mutation failure is a finite outcome, including through public evolution."""
from types import SimpleNamespace

import pytest
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.opt_history_objects.individual import Individual

from fedcore.interfaces.fedcore_optimizer import FedcoreEvoOptimizer


def individual(name):
    return Individual(OptGraph(OptNode(name)))


class Verifier:
    def __init__(self):
        self.checked = []

    def __call__(self, graph):
        self.checked.append(graph)
        return graph.nodes[0].name != 'invalid'


def optimizer(mutate):
    return SimpleNamespace(graph_generation_params=SimpleNamespace(verifier=Verifier()),
                           mutation=mutate, graph_generation_attempts=2, min_reproduce_attempt=3)


@pytest.mark.parametrize('result', [[], None, [None], ()])
def test_exhausted_mutation_keeps_population_without_verifying_placeholder(result):
    calls = []
    parent = individual('parent')
    subject = optimizer(lambda item: calls.append(item) or result)
    population = [parent]
    assert FedcoreEvoOptimizer._extend_population(subject, population, 2) == population
    assert len(calls) == 6
    assert subject.graph_generation_params.verifier.checked == []


def test_population_result_filters_invalid_and_duplicate_candidates_and_respects_cap():
    parent, good, extra, invalid = map(individual, ['parent', 'good', 'extra', 'invalid'])
    subject = optimizer(lambda item: [None, parent, invalid, good, extra])
    assert FedcoreEvoOptimizer._extend_population(subject, [parent], 2) == [parent, good]
    assert subject.graph_generation_params.verifier.checked == [invalid.graph, good.graph]


def test_empty_or_already_sufficient_population_never_requests_mutation():
    subject = optimizer(lambda item: pytest.fail('Unneeded mutation'))
    parent = individual('parent')
    assert FedcoreEvoOptimizer._extend_population(subject, [], 2) == []
    assert FedcoreEvoOptimizer._extend_population(subject, [parent], 1) == [parent]


def test_empty_results_then_individual_can_extend():
    parent, child = map(individual, ['parent', 'child'])
    results = iter([[], [], child])
    subject = optimizer(lambda item: next(results))
    assert FedcoreEvoOptimizer._extend_population(subject, [parent], 2) == [parent, child]


def test_public_fit_survives_all_unsuccessful_mutations(tmp_path, isolated_registry, monkeypatch):
    import distributed
    from golem.core.optimisers.genetic.operators.mutation import Mutation
    from fedcore.api.main import FedCore
    from tests.refactoring.test_core_contracts import FakeResource, configuration, dataset, identity_model

    monkeypatch.setattr(distributed, 'LocalCluster', FakeResource)
    monkeypatch.setattr(distributed, 'Client', FakeResource)
    calls = []
    monkeypatch.setattr(Mutation, '__call__', lambda self, item: calls.append(item) or [])
    model = identity_model()
    api = FedCore(configuration(model, tmp_path))
    settings = api.manager.automl_config.fedot_config
    settings.timeout, settings.pop_size, settings.n_jobs = .05, 2, 1
    settings.available_operations = ['training_model']
    api.manager.learning_config.peft_strategy_params.epochs = 0
    api.fit(dataset(model))
    assert calls
    assert api.predict(dataset(model)).predict.shape == (20, 2)


# Reuse the fixture that isolates the production singleton registry.
from tests.refactoring.test_core_contracts import isolated_registry
