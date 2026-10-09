"""Observe the actual public evolutionary fit, including initialization count."""
import json
from fedcore.interfaces.search_trace import SearchTrace
from tests.refactoring.test_core_contracts import (
    FakeResource, configuration, dataset, identity_model, isolated_registry,
)
from fedcore.api.main import FedCore


def test_trace_records_failed_batches_without_swallowing_error(tmp_path):
    trace = SearchTrace(tmp_path / 'trace.json')
    def fail(items):
        raise ValueError('expected')
    import pytest
    with pytest.raises(ValueError):
        trace.evaluator(fail)([])
    event = json.loads(trace.path.read_text())['events'][0]
    assert event['error_type'] == 'ValueError'
    assert event['elapsed_seconds'] >= 0


def test_public_evolution_trace_has_one_initialization(tmp_path, isolated_registry, monkeypatch):
    import distributed
    monkeypatch.setattr(distributed, 'LocalCluster', FakeResource)
    monkeypatch.setattr(distributed, 'Client', FakeResource)
    model = identity_model()
    api = FedCore(configuration(model, tmp_path))
    settings = api.manager.automl_config.fedot_config
    settings.timeout, settings.pop_size, settings.n_jobs = 0.05, 2, 1
    settings.available_operations = ['training_model']
    api.manager.learning_config.peft_strategy_params.epochs = 0
    trace_path = tmp_path / 'original_search.json'
    api.manager.automl_config.search_trace_path = str(trace_path)
    api.fit(dataset(model))
    trace = json.loads(trace_path.read_text(encoding='utf-8'))
    assert trace['engine'] == 'FedcoreEvoOptimizer'
    populations = [e for e in trace['events'] if e['event'] == 'population']
    assert sum(e['label'] in ('initial_assumptions', 'extended_initial_assumptions')
               for e in populations) == 1
    assert populations[-1]['label'] == 'final_choices'
    assert any(e['event'] == 'evaluation_batch' for e in trace['events'])
    assert all('variation' in i for e in populations for i in e['individuals'])
