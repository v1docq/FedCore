"""Public workflow store ownership, provenance and fixed-graph atomicity."""
import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.artifacts import StatisticsStore
from fedcore.algorithm.low_rank.compression_grammar import (
    ASVDChoice, AffinePCAChoice, CompressionGenotype, ScopeGene, WeightedChoice, compile_genotype)
from fedcore.algorithm.low_rank.grammar_execution import (
    ResourceFailure, StructuralInvalid, WorkflowSuccess, execute_workflow, prepare_grammar_space)
from fedcore.algorithm.low_rank.method_specs import ASVD


def fixture(dtype=torch.float32):
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(4, 4, dtype=dtype), nn.Tanh(), nn.Linear(4, 4, dtype=dtype)).eval()
    data = torch.randn(12, 4, dtype=dtype)+.3
    choices = (WeightedChoice(1), WeightedChoice(2), ASVDChoice(1, ASVD(alpha=.25)),
               ASVDChoice(2, ASVD(alpha=.5)), AffinePCAChoice(1))
    space = prepare_grammar_space(model, data, alternatives={path: choices for path in ("0", "2")})
    return model, data, choices, space


def workflow(space, choice, paths=("0", "2")):
    return compile_genotype(CompressionGenotype(tuple(ScopeGene((path,), choice) for path in paths)), space)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_public_workflow_reuses_rank_and_alpha_statistics_without_changing_outputs(dtype):
    model, data, choices, space = fixture(dtype)
    store = StatisticsStore(8192)
    try:
        for first, second in ((choices[0], choices[1]), (choices[2], choices[3])):
            warm = execute_workflow(model, data, workflow(space, first), store=store)
            refs = store.refs
            cold = execute_workflow(model, data, workflow(space, second))
            events = []
            cached = execute_workflow(model, data, workflow(space, second), store=store,
                phase_callback=events.append)
            assert all(type(value) is WorkflowSuccess for value in (warm, cold, cached))
            assert torch.equal(cold.model(data), cached.model(data))
            assert store.refs == refs
            observations = [event for event in events if event["phase"] == "observe" and event["state"] == "completed"]
            assert len(observations) == 2 and all(event["cache_hit"] for event in observations)
        assert len(store.refs) == 4  # two moments and two absolute-mean vectors
        refs, used = store.refs, store.used_bytes
        events = []
        afm = execute_workflow(model, data, workflow(space, choices[4]), store=store,
            phase_callback=events.append)
        assert type(afm) is WorkflowSuccess
        assert store.refs == refs and store.used_bytes == used
        assert sum(event["phase"] == "afm_whole_profile" and event["state"] == "completed" for event in events) == 2
        assert not any(event["phase"] == "observe" for event in events)
    finally:
        store.close()


@pytest.mark.parametrize("context", ["rng", "calibration", "source", "graph_version"])
def test_public_store_keeps_complete_provenance(context):
    model, data, choices, space = fixture()
    store = StatisticsStore(8192)
    try:
        assert type(execute_workflow(model, data, workflow(space, choices[0]), store=store)) is WorkflowSuccess
        old = {ref.key_digest for ref in store.refs}
        if context == "rng":
            torch.manual_seed(12345)
        elif context == "calibration":
            data = data+.1
        elif context == "source":
            with torch.no_grad():
                model[0].weight.add_(.01)
        version = "changed-version" if context == "graph_version" else space.source.graph_version
        new = prepare_grammar_space(model, data, alternatives={path: choices for path in ("0", "2")}, graph_version=version)
        assert type(execute_workflow(model, data, workflow(new, choices[0]), store=store)) is WorkflowSuccess
        assert len(store.refs) == 4
        assert old < {ref.key_digest for ref in store.refs}
    finally:
        store.close()


def test_later_scope_failure_keeps_only_earlier_committed_statistics(monkeypatch):
    import fedcore.algorithm.low_rank.grammar_execution as runtime
    from fedcore.algorithm.low_rank.graph_execution import CompressionGraphError
    from fedcore.algorithm.low_rank.plans import Violation
    model, data, choices, space = fixture()
    original = runtime.execute_fixed_graph
    def fail_second(model, data, graph, **kwargs):
        if graph.transform.steps[0].operator.path == "2":
            raise CompressionGraphError((Violation("StoreCapacityExceeded", "store", "injected second-scope refusal"),))
        return original(model, data, graph, **kwargs)
    monkeypatch.setattr(runtime, "execute_fixed_graph", fail_second)
    store = StatisticsStore(8192)
    try:
        result = execute_workflow(model, data, workflow(space, choices[0]), store=store)
        assert type(result) is ResourceFailure
        assert len(store.refs) == 1 and store.refs[0].provenance.path == "0"
        events, refs = [], store.refs
        retry = execute_workflow(model, data, workflow(space, choices[1], ("0",)),
            store=store, phase_callback=events.append)
        assert type(retry) is WorkflowSuccess and store.refs == refs
        assert any(event.get("cache_hit") for event in events if event["phase"] == "observe")
    finally:
        store.close()


def test_public_store_ownership_and_capacity_fail_before_observation(monkeypatch):
    import fedcore.algorithm.low_rank.staged_execution as stages
    model, data, choices, space = fixture()
    closed, observed = [], []
    original_close, original_observe = StatisticsStore.close, stages._weighted_observe
    def close(store):
        closed.append(store)
        return original_close(store)
    def observe(*args, **kwargs):
        observed.append(True)
        return original_observe(*args, **kwargs)
    monkeypatch.setattr(StatisticsStore, "close", close)
    monkeypatch.setattr(stages, "_weighted_observe", observe)
    caller = StatisticsStore(8192)
    try:
        assert type(execute_workflow(model, data, workflow(space, choices[0]), store=caller)) is WorkflowSuccess
        assert not closed
        assert type(execute_workflow(model, data, workflow(space, choices[0]))) is WorkflowSuccess
        assert len(closed) == 2 and caller not in closed
        empty = StatisticsStore(0)
        count = len(observed)
        assert type(execute_workflow(model, data, workflow(space, choices[0]), store=empty)) is ResourceFailure
        assert len(observed) == count and empty.used_bytes == 0
        empty.close()
        result = execute_workflow(model, data, workflow(space, choices[0]), store=empty)
        assert type(result) is StructuralInvalid and result.violations[0].code == "StoreClosed"
    finally:
        caller.close()


def test_failed_stage_keeps_started_interval_for_residual_accounting(monkeypatch):
    import fedcore.algorithm.low_rank.staged_execution as stages
    from fedcore.algorithm.low_rank.approximation import NumericalDomainViolation
    model, data, choices, space = fixture()
    def fail(*args, **kwargs):
        raise NumericalDomainViolation("injected solve refusal")
    monkeypatch.setattr(stages, "solve_weighted_prepared", fail)
    events = []
    result = execute_workflow(model, data, workflow(space, choices[0]), phase_callback=events.append)
    assert type(result).__name__ == "NumericalFailure"
    assert any(event["phase"] == "solve" and event["state"] == "started" for event in events)
    assert not any(event["phase"] == "solve" and event["state"] == "completed" for event in events)
