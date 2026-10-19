"""Executed fixed stages, provenance reuse and owned-store lifecycle."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.artifacts import ArtifactStoreError, StatisticsStore
from fedcore.algorithm.low_rank.compression_graph import Stage
from fedcore.algorithm.low_rank.execution import transform_weighted
from fedcore.algorithm.low_rank.graph_execution import CompressionGraphError, execute_fixed_graph, prepare_fixed_graph
from fedcore.algorithm.low_rank.method_execution import transform_method
from fedcore.algorithm.low_rank.method_specs import ASVD
from fedcore.algorithm.low_rank.plans import MetricPolicy


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def _rows(width=4):
    return torch.randn(11, width, dtype=torch.float64, generator=torch.Generator().manual_seed(42)) + .3


def _graph(model, data, recipe="weighted_svd", **options):
    return prepare_fixed_graph(model, data, recipe=recipe, executor="staged_fixed_recipe_v1", **options)


@pytest.mark.parametrize("recipe", ["weighted_svd", "asvd"])
@pytest.mark.parametrize("structure", [{"rank": 1}, {"rank_ratio": .5}, {"parameter_fraction": .8}])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stage_results_exactly_match_legacy_and_warm_cache(recipe, structure, dtype):
    model = nn.Sequential(nn.Linear(4, 4), nn.LeakyReLU(.3), nn.Linear(4, 3)).to(dtype)
    data, store = _rows().to(dtype), StatisticsStore(100000)
    before, original, rng = deepcopy(model.state_dict()), data.clone(), torch.get_rng_state().clone()
    spec = ASVD(alpha=.7, epsilon=.02)
    options = {"method_spec": spec} if recipe == "asvd" else {"policy": MetricPolicy(ridge=.02)}
    graph = _graph(model, data, recipe, batch_size=3, **structure, **options)
    cold = execute_fixed_graph(model, data, graph, store=store)
    warm = execute_fixed_graph(model, data, graph, store=store)
    legacy = (transform_weighted(model, data, policy=options["policy"], batch_size=3, **structure)
              if recipe == "weighted_svd" else transform_method(model, data, spec, batch_size=3, **structure))
    for name, value in legacy.model.state_dict().items():
        torch.testing.assert_close(cold.model.state_dict()[name], value, rtol=0, atol=0)
        torch.testing.assert_close(warm.model.state_dict()[name], value, rtol=0, atol=0)
    assert cold.evidence["observe_forward_calls"] == 8 and warm.evidence["observe_forward_calls"] == 0
    assert warm.evidence["validation_forward_calls"] == 0
    assert [stage["stage"] for stage in warm.evidence["actual_stages"]] == [node.stage.value for node in graph.nodes]
    assert all(stage["executed"] for stage in warm.evidence["actual_stages"])
    assert all(stage["cache_hit"] for stage in warm.evidence["actual_stages"] if stage["stage"] == Stage.OBSERVE.value)
    json.loads(json.dumps(warm.evidence, allow_nan=False))
    assert len(warm.statistic_refs) == 2 and torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(data, original, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(not layer._forward_pre_hooks for layer in model.modules())


@pytest.mark.parametrize("kind", ["conv1", "conv2", "grouped", "reflect"])
@pytest.mark.parametrize("representation", ["one_layer", "two_layers", "three_layers"])
def test_weighted_channel_conv_stage_parity(kind, representation):
    if kind == "conv1":
        model = nn.Conv1d(2, 3, 3, padding=2, dilation=2).double()
        data = torch.randn(5, 2, 9, dtype=torch.float64)
    else:
        model = nn.Conv2d(4, 6, (3, 2), padding=1, stride=(2, 1), groups=2 if kind == "grouped" else 1,
                          padding_mode="reflect" if kind == "reflect" else "zeros").double()
        data = torch.randn(5, 4, 8, 7, dtype=torch.float64)
    graph = _graph(model, data, rank=1, representation=representation, batch_size=2)
    store = StatisticsStore(100000)
    actual = execute_fixed_graph(model, data, graph, store=store)
    legacy = transform_weighted(model, data, rank=1, representation=representation, batch_size=2)
    torch.testing.assert_close(actual.model(data), legacy.model(data), rtol=0, atol=0)
    warm = execute_fixed_graph(model, data, graph, store=store)
    assert warm.evidence["observe_forward_calls"] == 0
    for value, reference in zip(warm.model.state_dict().values(), legacy.model.state_dict().values()):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


class Aliases(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4).double()
        self.b = self.a
    def forward(self, data):
        return self.a(data) + self.b(data * 2)


@pytest.mark.parametrize("recipe", ["weighted_svd", "asvd"])
def test_aliases_preserved_and_observation_count_includes_both_calls(recipe):
    model, data, store = Aliases(), _rows(), StatisticsStore(100000)
    result = execute_fixed_graph(model, data, _graph(model, data, recipe, rank=1), store=store)
    assert result.model.a is result.model.b
    ref = result.statistic_refs[0]
    snapshot = store.read(ref, expected=ref.provenance)
    assert snapshot.passports[0].count == 2 * len(data)


def test_alpha_reuses_absolute_statistics_but_rebuilds_metric(monkeypatch):
    from fedcore.algorithm.low_rank import staged_execution
    model, data, store = nn.Linear(4, 3).double(), _rows(), StatisticsStore(100000)
    first = _graph(model, data, "asvd", rank=1, method_spec=ASVD(alpha=.2))
    initial = execute_fixed_graph(model, data, first, store=store)
    second = _graph(model, data, "asvd", rank=1, method_spec=ASVD(alpha=.9))
    def no_collection(*args, **kwargs):
        pytest.fail("alpha-only change must reuse abs-stat")
    monkeypatch.setattr(staged_execution, "_collect_rows", no_collection)
    revised = execute_fixed_graph(model, data, second, store=store)
    assert initial.statistic_refs == revised.statistic_refs
    assert revised.evidence["observe_forward_calls"] == 0
    assert revised.legacy_result.evidence["layers"][0]["numerics"]["alpha"] == .9
    assert not torch.equal(initial.model.factor_matrix(), revised.model.factor_matrix())


def test_upstream_change_invalidates_current_student_downstream_statistic():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 3)).double()
    data, store = _rows(), StatisticsStore(100000)
    first = execute_fixed_graph(model, data, _graph(model, data, "asvd", rank=1, method_spec=ASVD(alpha=.2)), store=store)
    second = execute_fixed_graph(model, data, _graph(model, data, "asvd", rank=1, method_spec=ASVD(alpha=.8)), store=store)
    observations = [entry for entry in second.evidence["actual_stages"] if entry["stage"] == "observe"]
    assert observations[0]["cache_hit"] and not observations[1]["cache_hit"]
    assert first.statistic_refs[0] == second.statistic_refs[0]
    assert first.statistic_refs[1] != second.statistic_refs[1]
    with pytest.raises(ArtifactStoreError, match="StaleArtifact"):
        store.read(first.statistic_refs[1], expected=second.statistic_refs[1].provenance)


def test_owned_store_ingress_egress_capacity_and_close():
    model, data = nn.Linear(4, 3).double(), _rows()
    store = StatisticsStore(128)
    result = execute_fixed_graph(model, data, _graph(model, data, rank=1), store=store)
    ref = result.statistic_refs[0]
    original = store.read(ref, expected=ref.provenance)
    copied = StatisticsStore(128)
    copied_ref = copied.put(ref.provenance, original)
    original.tensors[0].zero_()
    exported = copied.read(copied_ref, expected=ref.provenance)
    assert bool(exported.tensors[0].abs().sum() > 0)
    exported.tensors[0].zero_()
    assert bool(copied.read(copied_ref, expected=ref.provenance).tensors[0].abs().sum() > 0)
    assert copied.used_bytes == 128
    with pytest.raises(ArtifactStoreError, match="StaleArtifact"):
        copied.read(copied_ref, expected=replace(ref.provenance, calibration_fingerprint="changed"))
    with pytest.raises(ArtifactStoreError, match="StaleArtifact"):
        StatisticsStore(128).put(replace(ref.provenance, calibration_fingerprint="changed"), exported)
    copied.close()
    assert copied.used_bytes == 0 and copied.refs == ()
    copied.close()
    with pytest.raises(ArtifactStoreError, match="StoreClosed"):
        copied.read(copied_ref, expected=ref.provenance)


def test_late_admission_failure_rolls_back_new_entries_and_keeps_old():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 3)).double()
    data, store = _rows(), StatisticsStore(63)
    initial = execute_fixed_graph(model, data, _graph(model, data, "asvd", rank=1, target_paths=("0",)), store=store)
    before = store.read(initial.statistic_refs[0], expected=initial.statistic_refs[0].provenance)
    with pytest.raises(CompressionGraphError, match="StoreCapacityExceeded"):
        execute_fixed_graph(model, data, _graph(model, data, "asvd", rank=1), store=store)
    assert store.used_bytes == 32 and len(store.refs) == 1
    after = store.read(initial.statistic_refs[0], expected=initial.statistic_refs[0].provenance)
    torch.testing.assert_close(before.tensors[0], after.tensors[0], rtol=0, atol=0)
    # Clear old entry: a miss inserted for layer 0 must also roll back when the
    # downstream admission exceeds capacity later in the same transaction.
    store.clear()
    with pytest.raises(CompressionGraphError, match="StoreCapacityExceeded"):
        execute_fixed_graph(model, data, _graph(model, data, "asvd", rank=1), store=store)
    assert store.used_bytes == 0 and store.refs == ()


@pytest.mark.parametrize("recipe", ["weighted_svd", "asvd"])
def test_effectful_observation_refused_without_source_mutation(recipe):
    model = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(4, 3)).double()
    data, store = _rows() - 3, StatisticsStore(100000)
    original, weights = data.clone(), deepcopy(model.state_dict())
    with pytest.raises(CompressionGraphError, match="UnsupportedObservationEffect"):
        execute_fixed_graph(model, data, _graph(model, data, recipe, rank=1), store=store)
    assert store.used_bytes == 0 and store.refs == ()
    torch.testing.assert_close(data, original, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
    assert all(not layer._forward_pre_hooks for layer in model.modules())


def test_legacy_store_rejected_and_staged_store_required():
    model, data = nn.Linear(4, 3).double(), _rows()
    with pytest.raises(CompressionGraphError, match="UnsupportedOptions"):
        execute_fixed_graph(model, data, prepare_fixed_graph(model, data, rank=1), store=StatisticsStore(1000))
    with pytest.raises(CompressionGraphError, match="InvalidArtifactStore"):
        execute_fixed_graph(model, data, _graph(model, data, rank=1))


def test_actual_metric_and_solve_execute_on_warm_hit(monkeypatch):
    from fedcore.algorithm.low_rank import staged_execution
    model, data, store = nn.Linear(4, 3).double(), _rows(), StatisticsStore(1000)
    graph = _graph(model, data, rank=1)
    execute_fixed_graph(model, data, graph, store=store)
    metric, solve = staged_execution.prepare_weighted_metric, staged_execution.solve_weighted_prepared
    calls = []
    def metric_call(*args, **kwargs):
        calls.append("metric")
        return metric(*args, **kwargs)
    def solve_call(*args, **kwargs):
        calls.append("solve")
        return solve(*args, **kwargs)
    monkeypatch.setattr(staged_execution, "prepare_weighted_metric", metric_call)
    monkeypatch.setattr(staged_execution, "solve_weighted_prepared", solve_call)
    result = execute_fixed_graph(model, data, graph, store=store)
    assert calls == ["metric", "solve"] and result.evidence["observe_forward_calls"] == 0


def test_combined_copy_and_solver_floor_refused_before_allocation(monkeypatch):
    from types import SimpleNamespace
    import psutil
    from fedcore.algorithm.low_rank import staged_execution
    model, data = nn.Linear(4, 3).double(), _rows()
    rss, reserve, solve = 1024**2, 64 * 1024**2, 2080
    copies = sum(p.numel()*p.element_size() for p in model.parameters()) + data.numel()*data.element_size()
    peak = rss + reserve + solve + copies // 2
    graph = _graph(model, data, rank=1, max_peak_bytes=peak, max_workspace_bytes=1024**2)
    monkeypatch.setattr(psutil, "Process", lambda _pid: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=rss)))
    def forbidden(*args, **kwargs):
        pytest.fail("Combined memory refusal must precede model copy and metric allocation")
    monkeypatch.setattr(staged_execution, "deepcopy", forbidden)
    monkeypatch.setattr(staged_execution, "prepare_weighted_metric", forbidden)
    with pytest.raises(CompressionGraphError, match="ResourceLimitExceeded"):
        execute_fixed_graph(model, data, graph, store=StatisticsStore(1000))


@pytest.mark.parametrize("recipe", ["weighted_svd", "asvd"])
def test_known_factorized_source_uses_actual_operator(recipe):
    data = _rows()
    source = transform_weighted(nn.Linear(4, 3).double(), data, rank=2).model
    graph = _graph(source, data, recipe, rank=1)
    result = execute_fixed_graph(source, data, graph, store=StatisticsStore(1000))
    legacy = transform_weighted(source, data, rank=1) if recipe == "weighted_svd" else transform_method(source, data, ASVD(), rank=1)
    torch.testing.assert_close(result.model.factor_matrix(), legacy.model.factor_matrix(), rtol=0, atol=0)


def test_asvd_reduction_workspace_refused_after_collection_before_abs(monkeypatch):
    from fedcore.algorithm.low_rank import staged_execution
    model = nn.Linear(4, 3)
    data = torch.ones(1000, 4)
    store = StatisticsStore(1000)
    graph = _graph(model, data, "asvd", rank=1, max_workspace_bytes=64 * 1024, batch_size=64)
    def forbidden(*args, **kwargs):
        pytest.fail("Float64 reduction must be refused before its allocations")
    monkeypatch.setattr(staged_execution, "channel_abs_statistics", forbidden)
    with pytest.raises(CompressionGraphError, match="ResourceLimitExceeded"):
        execute_fixed_graph(model, data, graph, store=store)
    assert store.used_bytes == 0 and store.refs == ()
    assert not model._forward_pre_hooks
