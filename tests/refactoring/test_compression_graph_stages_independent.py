"""Independent numerical, ownership and lifecycle witnesses for opt-in G2a."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import random
from types import SimpleNamespace

from hypothesis import settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule
import numpy as np
import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.artifacts import (
    ArtifactStoreError, StatisticRef, StatisticsKey, StatisticsSnapshot, StatisticsStore,
)
from fedcore.algorithm.low_rank.compression_graph import ArtifactKind
from fedcore.algorithm.low_rank.graph_execution import (
    CompressionGraphError, execute_fixed_graph, prepare_fixed_graph,
)
from fedcore.algorithm.low_rank.method_specs import ASVD
from fedcore.algorithm.low_rank.statistics import (
    StatisticsPassport, empty_second_moment, finish_second_moment,
    merge_second_moment, update_second_moment,
)


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _key(identifier=0, *, kind=ArtifactKind.INPUT_SECOND_MOMENT):
    return StatisticsKey("1" * 64, "independent-source-v1", "layer", ("layer",),
        hashlib.sha256(str(identifier).encode()).hexdigest(), kind, "float64", 2,
        "3" * 64, source_fingerprint="2" * 64)


def _snapshot(key, *, groups=1, requires_grad=False):
    tensors, passports = [], []
    for group in range(groups):
        tensors.append(torch.eye(2, dtype=torch.float64, requires_grad=requires_grad) * (group + 1))
        passports.append(StatisticsPassport(key.source_fingerprint, key.graph_version,
            key.model_fingerprint, key.path + f":input:group={group}", key.calibration_fingerprint,
            count=3, weight_sum=3., method_version=key.estimator_version))
    return StatisticsSnapshot(tuple(tensors), tuple(passports), key.statistic)


def _codes(error):
    return {violation.code for violation in error.violations}


class StoreLifecycleModel(RuleBasedStateMachine):
    """A two-slot reference dictionary; byte accounting is independent of store helpers."""
    def __init__(self):
        super().__init__()
        self.store = StatisticsStore(64)
        self.present = set()
        self.closed = False

    @rule(identifier=st.integers(0, 2))
    def put(self, identifier):
        key = _key(identifier)
        expected = ("StoreClosed" if self.closed else "DuplicateArtifact" if identifier in self.present
                    else "StoreCapacityExceeded" if len(self.present) == 2 else None)
        if expected:
            with pytest.raises(ArtifactStoreError) as caught:
                self.store.put(key, _snapshot(key))
            assert expected in _codes(caught.value)
        else:
            self.store.put(key, _snapshot(key))
            self.present.add(identifier)

    @rule(identifier=st.integers(0, 2))
    def read_and_mutate_export(self, identifier):
        key = _key(identifier)
        ref = StatisticRef(key.digest, key)
        if self.closed or identifier not in self.present:
            with pytest.raises(ArtifactStoreError) as caught:
                self.store.read(ref, expected=key)
            assert ("StoreClosed" if self.closed else "MissingArtifact") in _codes(caught.value)
        else:
            exported = self.store.read(ref, expected=key)
            exported.tensors[0].zero_()
            actual = self.store.read(ref, expected=key).tensors[0]
            torch.testing.assert_close(actual, torch.eye(2, dtype=torch.float64), rtol=0, atol=0)

    @rule()
    def clear(self):
        if self.closed:
            with pytest.raises(ArtifactStoreError) as caught:
                self.store.clear()
            assert "StoreClosed" in _codes(caught.value)
        else:
            self.store.clear()
            self.present.clear()

    @rule()
    def close(self):
        self.store.close()
        self.closed = True
        self.present.clear()

    @invariant()
    def exact_owned_bytes_and_membership(self):
        assert self.store.used_bytes == 32 * len(self.present)
        assert self.store.used_bytes <= 64
        assert {ref.provenance for ref in self.store.refs} == {_key(i) for i in self.present}


TestStoreLifecycle = StoreLifecycleModel.TestCase
TestStoreLifecycle.settings = settings(max_examples=12, stateful_step_count=16, deadline=None)


def test_store_copies_ingress_and_every_nested_export_tensor():
    key = _key()
    snapshot = _snapshot(key, groups=2, requires_grad=True)
    store = StatisticsStore(64)
    ref = store.put(key, snapshot)
    assert store.used_bytes == 64
    with torch.no_grad():
        for tensor in snapshot.tensors:
            tensor.add_(100)
    first = store.read(ref, expected=key)
    assert all(not tensor.requires_grad and tensor.grad_fn is None for tensor in first.tensors)
    assert first.tensors[0].untyped_storage().data_ptr() != first.tensors[1].untyped_storage().data_ptr()
    for tensor in first.tensors:
        tensor.resize_(100).zero_()
    second = store.read(ref, expected=key)
    torch.testing.assert_close(second.tensors[0], torch.eye(2, dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(second.tensors[1], 2 * torch.eye(2, dtype=torch.float64), rtol=0, atol=0)
    assert store.used_bytes == 64
    assert not hasattr(store, "transaction") and not hasattr(store, "lookup")


@pytest.mark.parametrize("field,value", [
    ("source_fingerprint", "4" * 64), ("model_fingerprint", "5" * 64),
    ("calibration_fingerprint", "6" * 64), ("graph_version", "changed-v2"),
    ("path", "other_layer"), ("coordinates", "rotated_input"),
    ("normalization", "sample_count"), ("input_dtype", "float32"),
    ("estimator_version", "changed_estimator"), ("mask", "observed_mask"),
    ("weights", "observed_weights"), ("rng_fingerprint", "7" * 64),
    ("statistic", ArtifactKind.CENTERED_INPUT_COVARIANCE),
])
def test_saved_ref_cannot_be_reinterpreted_as_different_provenance(field, value):
    key = _key()
    store = StatisticsStore(64)
    ref = store.put(key, _snapshot(key))
    with pytest.raises(ArtifactStoreError) as caught:
        store.read(ref, expected=replace(key, **{field: value}))
    assert "StaleArtifact" in _codes(caught.value)
    assert store.used_bytes == 32 and store.refs == (ref,)


@pytest.mark.parametrize("field,value", [
    ("source_fingerprint", "4" * 64), ("model_fingerprint", "5" * 64),
    ("calibration_fingerprint", "6" * 64), ("graph_version", "changed-v2"),
    ("path", "another_layer"),
])
def test_public_put_cannot_relabel_a_snapshot_from_another_source(field, value):
    key = _key()
    snapshot = _snapshot(key)
    store = StatisticsStore(64)
    ref = store.put(key, snapshot)
    changed = replace(key, **{field: value})
    if field == "path":
        changed = replace(changed, aliases=(value,))
    with pytest.raises(ArtifactStoreError) as caught:
        store.put(changed, snapshot)
    assert _codes(caught.value) & {"InvalidSnapshot", "StaleArtifact"}
    assert store.used_bytes == 32 and store.refs == (ref,)


def _inputs():
    generator = torch.Generator().manual_seed(741)
    return torch.randn((13, 4), generator=generator, dtype=torch.float64) * torch.tensor([.2, 2., 1., 4.]) + .7


def _model():
    model = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 3)).double()
    generator = torch.Generator().manual_seed(13)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=torch.float64))
    return model


def _prepare(model, inputs, recipe="weighted_svd", **options):
    return prepare_fixed_graph(model, inputs, recipe=recipe, rank=1, batch_size=3,
        graph_version="independent-pure-v1", executor="staged_fixed_recipe_v1", **options)


@pytest.mark.parametrize("recipe,mode", [("weighted_svd", None), ("asvd", "abs_mean"), ("asvd", "abs_max")])
def test_actual_observation_matches_direct_matrix_or_vector_reduction(recipe, mode):
    model = nn.Linear(4, 3).double()
    inputs = _inputs()
    options = {"method_spec": ASVD(abs_mode=mode)} if mode else {}
    graph = _prepare(model, inputs, recipe, target_paths=("",), **options)
    store = StatisticsStore(1024)
    result = execute_fixed_graph(model, inputs, graph, store=store)
    ref = result.statistic_refs[0]
    snapshot = store.read(ref, expected=ref.provenance)
    if recipe == "weighted_svd":
        expected = inputs.T @ inputs / len(inputs)
        passport = snapshot.passports[0]
        parts = [update_second_moment(empty_second_moment(4, replace(passport, count=0, weight_sum=0.)), chunk)
                 for chunk in (inputs[:2], inputs[2:8], inputs[8:])]
        merged = finish_second_moment(merge_second_moment(merge_second_moment(parts[0], parts[1]), parts[2]))
        torch.testing.assert_close(merged.matrix, expected, rtol=2e-14, atol=2e-14)
    else:
        expected = inputs.abs().mean(0) if mode == "abs_mean" else inputs.abs().amax(0)
    torch.testing.assert_close(snapshot.tensors[0], expected, rtol=2e-14, atol=2e-14)
    assert snapshot.passports[0].count == len(inputs)
    assert snapshot.passports[0].weight_sum == len(inputs)
    assert ref.provenance.mask == ref.provenance.weights == "none"
    assert ref.provenance.accumulator_dtype == "float64"
    trace = result.evidence["actual_stages"]
    assert [row["node"] for row in trace] == [node.key for node in graph.nodes]
    assert [row["semantic_kind"] for row in trace] == [node.outputs[0].kind.value for node in graph.nodes]
    assert all(row["executed"] for row in trace)
    assert all(row["tensor_bytes"] > 0 for row in trace)
    json.dumps(result.evidence, allow_nan=False)


class _AuditedLinear(nn.Module):
    calls = []  # External test instrumentation does not affect model values.

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 3).double()

    def forward(self, inputs):
        type(self).calls.append(1)
        return self.layer(inputs)


@pytest.mark.parametrize("recipe", ["weighted_svd", "asvd"])
def test_warm_cache_skips_real_observation_and_matches_legacy(recipe, monkeypatch):
    from fedcore.algorithm.low_rank.execution import transform_weighted
    from fedcore.algorithm.low_rank.method_execution import transform_method
    import fedcore.algorithm.low_rank.graph_execution as execution
    import fedcore.algorithm.low_rank.method_execution as methods

    model, inputs = _AuditedLinear(), _inputs()
    graph = _prepare(model, inputs, recipe, target_paths=("layer",))
    store = StatisticsStore(1024)
    _AuditedLinear.calls.clear()
    first = execute_fixed_graph(model, inputs, graph, store=store)
    assert len(_AuditedLinear.calls) == first.evidence["observe_forward_calls"] + first.evidence["validation_forward_calls"]

    def forbidden_delegate(*args, **kwargs):
        raise AssertionError("Staged executor delegated the whole numerical recipe")

    with monkeypatch.context() as guard:
        guard.setattr(execution, "transform_weighted", forbidden_delegate)
        guard.setattr(methods, "transform_method", forbidden_delegate)
        _AuditedLinear.calls.clear()
        warm = execute_fixed_graph(model, inputs, graph, store=store)
        assert warm.evidence["observe_forward_calls"] == 0
        assert len(_AuditedLinear.calls) == warm.evidence["validation_forward_calls"]
    legacy = (transform_weighted(model, inputs.clone(), rank=1, batch_size=3, target_paths=("layer",))
              if recipe == "weighted_svd" else transform_method(model, inputs.clone(), ASVD(), rank=1, batch_size=3, target_paths=("layer",)))
    torch.testing.assert_close(first.model(inputs), legacy.model(inputs), rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(warm.model(inputs), legacy.model(inputs), rtol=1e-11, atol=1e-11)
    assert store.used_bytes == (4 * 4 * 8 if recipe == "weighted_svd" else 4 * 8)
    json.dumps(warm.evidence, allow_nan=False)


def test_alpha_reuses_first_statistic_but_recollects_after_changed_upstream():
    from fedcore.algorithm.low_rank.method_execution import transform_method

    model, inputs = _model(), _inputs()
    store = StatisticsStore(4096)
    first = execute_fixed_graph(model, inputs, _prepare(model, inputs, "asvd", method_spec=ASVD(alpha=.1)), store=store)
    second = execute_fixed_graph(model, inputs, _prepare(model, inputs, "asvd", method_spec=ASVD(alpha=1.7)), store=store)
    observe = [row for row in second.evidence["actual_stages"] if row["stage"] == "observe"]
    assert [row["cache_hit"] for row in observe] == [True, False]
    assert first.statistic_refs[0] == second.statistic_refs[0]
    assert first.statistic_refs[1].provenance.model_fingerprint != second.statistic_refs[1].provenance.model_fingerprint
    legacy = transform_method(model, inputs.clone(), ASVD(alpha=1.7), rank=1, batch_size=3)
    torch.testing.assert_close(second.model(inputs), legacy.model(inputs), rtol=1e-10, atol=1e-10)
    with pytest.raises(ArtifactStoreError) as caught:
        store.read(first.statistic_refs[1], expected=second.statistic_refs[1].provenance)
    assert "StaleArtifact" in _codes(caught.value)


class _EffectModel(nn.Module):
    def __init__(self, effect, fail=False):
        super().__init__()
        self.layer = nn.Linear(4, 3).double()
        self.register_buffer("counter", torch.zeros((), dtype=torch.float64))
        self.scale = 1.
        self.effect, self.fail = effect, fail

    def forward(self, inputs):
        if self.effect == "python_rng":
            random.random()
        elif self.effect == "numpy_rng":
            np.random.random()
        elif self.effect == "torch_rng":
            torch.rand(())
        elif self.effect == "buffer":
            self.counter.add_(1)
        elif self.effect == "scalar":
            self.scale += 1
        elif self.effect == "mode":
            self.train()
        elif self.effect == "calibration":
            inputs.add_(1)
        output = self.layer(inputs)
        if self.fail:
            raise RuntimeError("independent pure-forward failure")
        return output


@pytest.mark.parametrize("recipe", ["weighted_svd", "asvd"])
@pytest.mark.parametrize("effect,fail", [
    ("python_rng", False), ("numpy_rng", False), ("torch_rng", False),
    ("buffer", False), ("scalar", False), ("mode", False),
    ("calibration", False), ("buffer", True), ("none", True),
])
def test_observation_effect_or_failure_rolls_back_and_preserves_caller(monkeypatch, recipe, effect, fail):
    import fedcore.algorithm.low_rank.staged_execution as stages

    model, inputs = _EffectModel(effect, fail), _inputs()
    source, data = deepcopy(model.state_dict()), inputs.clone()
    modes = tuple(layer.training for layer in model.modules())
    graph = _prepare(model, inputs, recipe, target_paths=("layer",))
    store = StatisticsStore(4096)
    existing = store.put(_key(), _snapshot(_key()))
    copies = []

    def record_copy(value):
        copied = deepcopy(value)
        copies.append(copied)
        return copied

    monkeypatch.setattr(stages, "deepcopy", record_copy)
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    expected_error = RuntimeError if effect == "none" else CompressionGraphError
    with pytest.raises(expected_error) as caught:
        execute_fixed_graph(model, inputs, graph, store=store)
    if effect != "none":
        assert "UnsupportedObservationEffect" in _codes(caught.value)
    assert store.used_bytes == 32 and store.refs == (existing,)
    assert random.getstate() == python_rng and torch.equal(torch.get_rng_state(), torch_rng)
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == numpy_rng[0] and actual_numpy[2:] == numpy_rng[2:]
    np.testing.assert_array_equal(actual_numpy[1], numpy_rng[1])
    assert torch.equal(inputs, data) and model.scale == 1.
    assert tuple(layer.training for layer in model.modules()) == modes
    assert copies and all(not layer._forward_pre_hooks and not layer._forward_hooks for copied in copies for layer in copied.modules())
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, source[name])


@pytest.mark.parametrize("recipe", ["weighted_svd", "asvd"])
def test_downstream_solve_failure_does_not_publish_partial_statistics(monkeypatch, recipe):
    import fedcore.algorithm.low_rank.staged_execution as stages

    model, inputs = _model(), _inputs()
    graph = _prepare(model, inputs, recipe)
    store = StatisticsStore(4096)
    existing = store.put(_key(), _snapshot(_key()))
    name = "solve_weighted_prepared" if recipe == "weighted_svd" else "_solve_input_scales"
    original = getattr(stages, name)
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("independent downstream solve failure")
        return original(*args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(stages, name, fail_second)
        with pytest.raises(RuntimeError, match="independent downstream solve failure"):
            execute_fixed_graph(model, inputs, graph, store=store)
    assert len(calls) == 2 and store.used_bytes == 32 and store.refs == (existing,)
    execute_fixed_graph(model, inputs, graph, store=store)
    assert len(store.refs) == 3


@pytest.mark.parametrize("recipe", ["weighted_svd", "asvd"])
def test_store_admission_refuses_before_observation(monkeypatch, recipe):
    import fedcore.algorithm.low_rank.staged_execution as stages

    model, inputs = nn.Linear(4, 3).double(), _inputs()
    graph = _prepare(model, inputs, recipe)
    required = 128 if recipe == "weighted_svd" else 32
    store = StatisticsStore(required - 1)

    def forbidden_observe(*args, **kwargs):
        raise AssertionError("Store admission occurred after observation")

    monkeypatch.setattr(stages, "_pure_observe", forbidden_observe)
    with pytest.raises(CompressionGraphError) as caught:
        execute_fixed_graph(model, inputs, graph, store=store)
    assert "StoreCapacityExceeded" in _codes(caught.value)
    assert store.used_bytes == 0 and not store.refs


@pytest.mark.parametrize("operation", ["read", "put"])
def test_snapshot_peak_refusal_precedes_clone_and_keeps_store_atomic(monkeypatch, operation):
    import psutil

    key = _key()
    store = StatisticsStore(64)
    ref = store.put(key, _snapshot(key))
    next_key, snapshot = _key(1), _snapshot(_key(1))
    rss, reserve, copied_bytes = 8 * 1024**2, 64 * 1024**2, 32
    monkeypatch.setattr(psutil.Process, "memory_info", lambda self: SimpleNamespace(rss=rss))

    def forbidden_clone(*args, **kwargs):
        raise AssertionError("Snapshot copied before peak admission")

    monkeypatch.setattr(torch.Tensor, "clone", forbidden_clone)
    with pytest.raises(ArtifactStoreError) as caught:
        if operation == "read":
            store.read(ref, expected=key, max_peak_bytes=rss + reserve + copied_bytes - 1)
        else:
            store.put(next_key, snapshot, max_peak_bytes=rss + reserve + copied_bytes - 1)
    assert "ResourceLimitExceeded" in _codes(caught.value)
    assert store.used_bytes == 32 and store.refs == (ref,)


def test_copies_and_solver_workspace_are_admitted_together_before_allocation(monkeypatch):
    import psutil
    import fedcore.algorithm.low_rank.staged_execution as stages

    model, inputs = nn.Linear(4, 3).double(), _inputs()
    copied_bytes = 15 * 8 + 13 * 4 * 8  # Work model and independent calibration.
    solver_bytes = 8 * (8 * 4 * 4 + 8 * 3 * 4 + 4 * 3 * 3)
    rss, reserve = 8 * 1024**2, 64 * 1024**2
    limit = rss + reserve + solver_bytes + copied_bytes // 2
    assert max(copied_bytes, solver_bytes) < limit - rss - reserve < copied_bytes + solver_bytes
    graph = _prepare(model, inputs, max_peak_bytes=limit, max_workspace_bytes=1024**2)
    store = StatisticsStore(4096)
    monkeypatch.setattr(psutil.Process, "memory_info", lambda self: SimpleNamespace(rss=rss))

    def forbidden_allocation(*args, **kwargs):
        raise AssertionError("Copies or metric allocation preceded their combined admission")

    monkeypatch.setattr(stages, "deepcopy", forbidden_allocation)
    monkeypatch.setattr(stages, "prepare_weighted_metric", forbidden_allocation)
    monkeypatch.setattr(torch.Tensor, "clone", forbidden_allocation)
    with pytest.raises(CompressionGraphError) as caught:
        execute_fixed_graph(model, inputs, graph, store=store)
    assert "ResourceLimitExceeded" in _codes(caught.value)
    assert store.used_bytes == 0 and not store.refs


@pytest.mark.parametrize("recipe", ["weighted_svd", "asvd"])
def test_workspace_refusal_precedes_model_copy_and_observation(monkeypatch, recipe):
    import fedcore.algorithm.low_rank.staged_execution as stages

    model, inputs = nn.Linear(4, 3).double(), _inputs()
    graph = _prepare(model, inputs, recipe, max_workspace_bytes=4 * 8)
    store = StatisticsStore(4096)

    def forbidden(*args, **kwargs):
        raise AssertionError("Workspace refusal happened after model copy or observation")

    monkeypatch.setattr(stages, "deepcopy", forbidden)
    monkeypatch.setattr(stages, "_pure_observe", forbidden)
    with pytest.raises(CompressionGraphError) as caught:
        execute_fixed_graph(model, inputs, graph, store=store)
    assert "ResourceLimitExceeded" in _codes(caught.value)
    assert store.used_bytes == 0 and not store.refs


def test_second_tensor_allocation_failure_cannot_publish_partial_ingress(monkeypatch):
    store = StatisticsStore(128)
    existing = store.put(_key(), _snapshot(_key()))
    key, snapshot = _key(1), _snapshot(_key(1), groups=2)
    original_clone = torch.Tensor.clone
    calls = []

    def fail_second(tensor, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise MemoryError("independent snapshot allocation failure")
        return original_clone(tensor, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(torch.Tensor, "clone", fail_second)
        with pytest.raises(ArtifactStoreError) as caught:
            store.put(key, snapshot)
    assert "ResourceLimitExceeded" in _codes(caught.value)
    assert len(calls) == 2 and store.used_bytes == 32 and store.refs == (existing,)
    store.put(key, snapshot)
    assert store.used_bytes == 96 and len(store.refs) == 2


def test_asvd_alpha_changes_the_selected_axis_while_reusing_the_same_abs_snapshot():
    model = nn.Linear(2, 2, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.diag(torch.tensor([2., 1.], dtype=torch.float64)))
    inputs = torch.tensor([[1., 3.], [1., -3.], [-1., 3.], [-1., -3.]], dtype=torch.float64)
    store = StatisticsStore(64)
    first = execute_fixed_graph(model, inputs, _prepare(model, inputs, "asvd", method_spec=ASVD(alpha=.5)), store=store)
    second = execute_fixed_graph(model, inputs, _prepare(model, inputs, "asvd", method_spec=ASVD(alpha=1.)), store=store)
    # abs(X)=(1,3): singular values of W*sqrt(S) are (2,sqrt(3)),
    # while those of W*S are (2,3). The retained axes therefore swap.
    expected_first = inputs @ torch.diag(torch.tensor([2., 0.], dtype=torch.float64))
    expected_second = inputs @ torch.diag(torch.tensor([0., 1.], dtype=torch.float64))
    torch.testing.assert_close(first.model(inputs), expected_first, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(second.model(inputs), expected_second, rtol=1e-12, atol=1e-12)
    assert first.statistic_refs == second.statistic_refs
    assert second.evidence["observe_forward_calls"] == 0 and store.used_bytes == 16


@pytest.mark.parametrize("stream", ["python", "numpy", "torch_cpu"])
def test_initial_rng_state_is_part_of_provenance_even_for_pure_builtin_forward(stream):
    model, inputs = nn.Linear(4, 3).double(), _inputs()
    graph = _prepare(model, inputs)
    store = StatisticsStore(4096)
    saved_python, saved_numpy, saved_torch = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        random.seed(701)
        np.random.seed(701)
        torch.default_generator.manual_seed(701)
        first = execute_fixed_graph(model, inputs, graph, store=store)
        if stream == "python":
            random.seed(702)
        elif stream == "numpy":
            np.random.seed(702)
        else:
            torch.default_generator.manual_seed(702)
        second = execute_fixed_graph(model, inputs, graph, store=store)
        assert first.statistic_refs[0].provenance.rng_fingerprint != second.statistic_refs[0].provenance.rng_fingerprint
        assert not second.evidence["actual_stages"][0]["cache_hit"]
        torch.testing.assert_close(first.model(inputs), second.model(inputs), rtol=0, atol=0)
    finally:
        random.setstate(saved_python)
        np.random.set_state(saved_numpy)
        torch.set_rng_state(saved_torch)


@pytest.mark.parametrize("mode", ["legacy_with_store", "staged_without_store"])
def test_executor_and_store_are_explicit_additive_capabilities(mode):
    model, inputs = nn.Linear(4, 3).double(), _inputs()
    if mode == "legacy_with_store":
        graph = prepare_fixed_graph(model, inputs, rank=1)
        assert graph.executor == "checked_whole_recipe_v1"
        store, expected = StatisticsStore(4096), "UnsupportedOptions"
    else:
        graph = _prepare(model, inputs)
        store, expected = None, "InvalidArtifactStore"
    with pytest.raises(CompressionGraphError) as caught:
        execute_fixed_graph(model, inputs, graph, store=store)
    assert expected in _codes(caught.value)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_asvd_reduction_admission_accounts_for_rows_and_float64_temporaries(monkeypatch, dtype):
    import fedcore.algorithm.low_rank.staged_execution as stages

    model = nn.Linear(4, 3, dtype=dtype)
    inputs = torch.linspace(-3., 3., 101 * 4, dtype=dtype).reshape(101, 4)
    workspace = 2 * inputs.numel() * inputs.element_size()
    assert workspace >= 8 * (8 * 4 * 4 + 8 * 3 * 4 + 4 * 3 * 3)
    graph = _prepare(model, inputs, "asvd", max_workspace_bytes=workspace)
    store = StatisticsStore(4096)

    def forbidden_reduction(*args, **kwargs):
        raise AssertionError("ASVD cast/abs/product allocation preceded its Nd workspace admission")

    monkeypatch.setattr(stages, "channel_abs_statistics", forbidden_reduction)
    with pytest.raises(CompressionGraphError) as caught:
        execute_fixed_graph(model, inputs, graph, store=store)
    assert "ResourceLimitExceeded" in _codes(caught.value)
    assert store.used_bytes == 0 and not store.refs
