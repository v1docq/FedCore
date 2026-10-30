"""Exact fixed-recipe contracts, pre-effect refusal and legacy parity."""
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from enum import Enum
import random
from types import SimpleNamespace

from hypothesis import given, strategies as st
import numpy as np
import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.compression_graph import (
    ArtifactKind, CalibrationRef, ModelVersion, Recipe, RuntimeOptions,
    SourceMode, Stage, compile_fixed_graph, graph_fingerprint, validate_fixed_graph,
)
from fedcore.algorithm.low_rank.execution import transform_weighted
from fedcore.algorithm.low_rank.graph_execution import (
    CompressionGraphError, execute_fixed_graph, prepare_fixed_graph,
)
from fedcore.algorithm.low_rank.method_execution import transform_method
from fedcore.algorithm.low_rank.method_specs import ASVD
from fedcore.algorithm.low_rank.plans import (
    FixedRank, Goal, MetricPolicy, OperatorDescriptor, ResourceLimits,
    TransformRequest, ValidationFailure, plan_transform,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def _input(width=4):
    return torch.randn(9, width, generator=torch.Generator().manual_seed(813), dtype=torch.float64) + .4


def _model():
    return nn.Sequential(nn.Linear(4, 4), nn.LeakyReLU(.2), nn.Linear(4, 3)).double()


def _seal(graph):
    return replace(graph, fingerprint=graph_fingerprint(graph))


def _forbid(*args, **kwargs):
    pytest.fail("An invalid/preparation-only graph reached a runtime effect")


@pytest.mark.parametrize("recipe", ["weighted_svd", "asvd"])
@pytest.mark.parametrize("structure", [{"rank": 1}, {"rank_ratio": .5}, {"parameter_fraction": .8}])
def test_exact_legacy_weights_and_source_preservation(recipe, structure):
    model, data = _model(), _input()
    before, observations, rng = deepcopy(model.state_dict()), data.clone(), torch.get_rng_state().clone()
    spec = ASVD(alpha=.7, epsilon=.01)
    extra = {"method_spec": spec} if recipe == "asvd" else {"policy": MetricPolicy(ridge=.01)}
    graph = prepare_fixed_graph(model, data, recipe=recipe, batch_size=3, **structure, **extra)
    result = execute_fixed_graph(model, data, graph)
    legacy = (transform_method(model, data, spec, batch_size=3, **structure) if recipe == "asvd"
              else transform_weighted(model, data, policy=extra["policy"], batch_size=3, **structure))
    assert result.model is not model and torch.equal(rng, torch.get_rng_state())
    assert result.graph is graph and result.evidence["executor"] == "checked_whole_recipe_v1"
    assert result.evidence["stage_trace_kind"] == "declared_inline_whole_recipe_stages"
    for name, value in result.model.state_dict().items():
        torch.testing.assert_close(value, legacy.model.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(result.model(data), legacy.model(data), rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    torch.testing.assert_close(data, observations, rtol=0, atol=0)
    assert all(not layer._forward_pre_hooks for layer in model.modules())
    assert all(not layer._forward_pre_hooks for layer in result.model.modules())


@pytest.mark.parametrize("ranks", [{"0": 1, "2": 2}, {"0": 2, "2": 1}])
def test_asvd_per_path_ranks_and_current_student_versions(ranks):
    model, data = _model(), _input()
    spec = ASVD(abs_mode="abs_max", alpha=.3)
    graph = prepare_fixed_graph(model, data, recipe="asvd", method_spec=spec,
                                ranks=ranks, target_paths=("0", "2"))
    assert graph.source_mode is SourceMode.CURRENT_STUDENT
    observe = [node for node in graph.nodes if node.stage is Stage.OBSERVE]
    assert observe[0].inputs[0].artifact == "model:initial"
    assert observe[1].inputs[0].artifact == "model:accepted=1"
    assert observe[1].outputs[0].model_version.endswith(":accepted=1")
    assert all(node.outputs[0].kind is ArtifactKind.INPUT_ABS_MAX for node in observe)
    result = execute_fixed_graph(model, data, graph)
    legacy = transform_method(model, data, spec, ranks=ranks, target_paths=("0", "2"))
    torch.testing.assert_close(result.model(data), legacy.model(data), rtol=0, atol=0)
    assert result.legacy_result.evidence["layers"][1]["graph_version"].endswith(":accepted=1")


def test_weighted_observes_original_snapshot_and_applies_once():
    graph = prepare_fixed_graph(_model(), _input(), rank=1)
    assert graph.source_mode is SourceMode.FROZEN_SNAPSHOT
    observe = [node for node in graph.nodes if node.stage is Stage.OBSERVE]
    assert all(node.inputs[0].artifact == "model:initial" for node in observe)
    assert all(node.outputs[0].kind is ArtifactKind.INPUT_SECOND_MOMENT for node in observe)
    assert sum(node.stage is Stage.APPLY for node in graph.nodes) == 1


@pytest.mark.parametrize("recipe", ["weighted_svd", "asvd"])
def test_symbolic_output_versions_distinguish_recipe_branches(recipe):
    model, data = _model(), _input()
    first = prepare_fixed_graph(model, data, recipe=recipe, rank=1)
    second = prepare_fixed_graph(model, data, recipe=recipe, rank=2)
    other_data = prepare_fixed_graph(model, data + .2, recipe=recipe, rank=1)
    versions = [graph.nodes[-1].outputs[0].model_version for graph in (first, second, other_data)]
    assert len(set(versions)) == 3
    assert first.external_artifacts[0] == second.external_artifacts[0]


def test_prepare_is_read_only_and_graph_contains_no_runtime_objects(monkeypatch):
    from fedcore.algorithm.low_rank import execution
    model, data = _model(), _input()
    monkeypatch.setattr(execution, "deepcopy", _forbid)
    monkeypatch.setattr(execution, "solve_weighted", _forbid)
    monkeypatch.setattr(nn.Module, "register_forward_pre_hook", _forbid)
    graph = prepare_fixed_graph(model, data, rank=1)
    assert graph == prepare_fixed_graph(model, data, rank=1)
    def inspect(value):
        assert not isinstance(value, (torch.Tensor, nn.Module, dict, list)) and not callable(value)
        if is_dataclass(value):
            assert type(value).__dataclass_params__.frozen
            for field in fields(value):
                inspect(getattr(value, field.name))
        elif isinstance(value, tuple):
            for item in value:
                inspect(item)
        else:
            assert value is None or type(value) in (str, int, bool, float) or isinstance(value, Enum)
    inspect(graph)


@pytest.mark.parametrize("mutation", ["missing", "wrong_statistic", "wrong_version", "wrong_mode", "wrong_rank", "bad_hash"])
def test_tampered_graph_refused_before_copy_hooks_or_solver(monkeypatch, mutation):
    from fedcore.algorithm.low_rank import execution
    model, data = _model(), _input()
    graph = prepare_fixed_graph(model, data, rank=1)
    nodes = list(graph.nodes)
    if mutation == "missing":
        graph = replace(graph, nodes=tuple(node for node in nodes if node.stage is not Stage.METRIC))
    elif mutation == "wrong_statistic":
        node = nodes[0]
        nodes[0] = replace(node, outputs=(replace(node.outputs[0], kind=ArtifactKind.CENTERED_INPUT_COVARIANCE),))
        graph = replace(graph, nodes=tuple(nodes))
    elif mutation == "wrong_version":
        nodes[0] = replace(nodes[0], outputs=(replace(nodes[0].outputs[0], model_version="other-model"),))
        graph = replace(graph, nodes=tuple(nodes))
    elif mutation == "wrong_mode":
        graph = replace(graph, source_mode=SourceMode.CURRENT_STUDENT)
    elif mutation == "wrong_rank":
        steps = graph.transform.steps
        graph = replace(graph, transform=replace(graph.transform,
            steps=(replace(steps[0], rank=2), *steps[1:])))
    else:
        graph = replace(graph, fingerprint="0" * 64)
    if mutation != "bad_hash":
        graph = _seal(graph)  # Recomputing the digest never legitimizes the DAG.
    monkeypatch.setattr(execution, "deepcopy", _forbid)
    monkeypatch.setattr(execution, "solve_weighted", _forbid)
    monkeypatch.setattr(nn.Module, "register_forward_pre_hook", _forbid)
    with pytest.raises(CompressionGraphError) as error:
        execute_fixed_graph(model, data, graph)
    assert error.value.violations[0].code


@pytest.mark.parametrize("change", ["weights", "upstream_config", "data"])
def test_stale_source_rejected_without_effects(monkeypatch, change):
    from fedcore.algorithm.low_rank import execution
    model, data = _model(), _input()
    graph = prepare_fixed_graph(model, data, rank=1)
    if change == "weights":
        with torch.no_grad():
            model[0].weight.add_(.125)
    elif change == "upstream_config":
        model[1].negative_slope = .8
    else:
        data = data + .125
    monkeypatch.setattr(execution, "deepcopy", _forbid)
    monkeypatch.setattr(nn.Module, "register_forward_pre_hook", _forbid)
    with pytest.raises(CompressionGraphError) as error:
        execute_fixed_graph(model, data, graph)
    assert error.value.violations[0].code == ("StaleCalibration" if change == "data" else "StaleModelVersion")


@pytest.mark.parametrize("kind", ["conv1", "conv2", "grouped", "reflect"])
@pytest.mark.parametrize("representation", ["one_layer", "two_layers", "three_layers"])
def test_weighted_conv_capabilities_match_legacy(kind, representation):
    if kind == "conv1":
        model, data = nn.Conv1d(2, 3, 3, padding=2, dilation=2).double(), torch.randn(4, 2, 9, dtype=torch.float64)
    else:
        model = nn.Conv2d(4, 6, (3, 2), stride=(2, 1), padding=1,
            groups=2 if kind == "grouped" else 1,
            padding_mode="reflect" if kind == "reflect" else "zeros").double()
        data = torch.randn(4, 4, 8, 7, dtype=torch.float64)
    graph = prepare_fixed_graph(model, data, rank=1, representation=representation)
    result = execute_fixed_graph(model, data, graph)
    legacy = transform_weighted(model, data, rank=1, representation=representation)
    torch.testing.assert_close(result.model(data), legacy.model(data), rtol=0, atol=0)


class AliasModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4).double()
        self.b = self.a
    def forward(self, value):
        return self.a(value) + self.b(2 * value)


@pytest.mark.parametrize("recipe", ["weighted_svd", "asvd"])
def test_alias_scope_and_execution_preserve_shared_module(recipe):
    model, data = AliasModel(), _input()
    graph = prepare_fixed_graph(model, data, recipe=recipe, rank=1)
    assert len(graph.scopes) == 1 and graph.scopes[0].aliases == ("a", "b")
    apply = [node for node in graph.nodes if node.stage is Stage.APPLY]
    assert apply[0].writes == ("a", "b")
    result = execute_fixed_graph(model, data, graph)
    assert result.model.a is result.model.b and model.a is model.b


class FailingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplace = nn.ReLU(inplace=True)
        self.layer = nn.Linear(4, 3).double()
    def forward(self, value):
        self.layer(self.inplace(value))
        random.random()
        np.random.random()
        torch.rand(1)
        raise RuntimeError("deliberate forward failure")


@pytest.mark.parametrize("recipe", ["weighted_svd", "asvd"])
def test_forward_failure_cleans_hooks_and_preserves_rng(monkeypatch, recipe):
    from fedcore.algorithm.low_rank import execution, method_execution
    model, data = FailingModel(), _input()
    graph = prepare_fixed_graph(model, data, recipe=recipe, rank=1)
    clones = []
    def capture(value):
        clone = deepcopy(value)
        clones.append(clone)
        return clone
    monkeypatch.setattr(execution if recipe == "weighted_svd" else method_execution, "deepcopy", capture)
    rng, before = torch.get_rng_state().clone(), deepcopy(model.state_dict())
    observations = data.clone()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    with pytest.raises(RuntimeError, match="deliberate"):
        execute_fixed_graph(model, data, graph)
    assert torch.equal(rng, torch.get_rng_state())
    assert random.getstate() == python_rng
    np.testing.assert_equal(np.random.get_state(), numpy_rng)
    torch.testing.assert_close(data, observations, rtol=0, atol=0)
    assert clones and all(not layer._forward_pre_hooks for clone in clones for layer in clone.modules())
    assert all(not layer._forward_pre_hooks for layer in model.modules())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("recipe", ["weighted_svd", "asvd"])
def test_success_owns_inplace_calibration_and_records_copy_preflight(recipe):
    model = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(4, 3)).double()
    data = _input() - 2
    observations, source = data.clone(), deepcopy(model.state_dict())
    graph = prepare_fixed_graph(model, data, recipe=recipe, rank=1)
    result = execute_fixed_graph(model, data, graph)
    # Legacy collection is allowed to alter its private input; start it from
    # the same original observations when comparing numerical behavior.
    legacy_data = observations.clone()
    legacy = (transform_weighted(model, legacy_data, rank=1) if recipe == "weighted_svd"
              else transform_method(model, legacy_data, ASVD(), rank=1))
    for name, value in result.model.state_dict().items():
        torch.testing.assert_close(value, legacy.model.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(data, observations, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, source[name], rtol=0, atol=0)
    copy = result.evidence["calibration_copy"]
    assert copy["copy_bytes"] == data.numel() * data.element_size()
    assert copy["required_floor_bytes"] == copy["copy_bytes"] + copy["process_rss_before_bytes"] + copy["runtime_reserve_bytes"]
    assert copy["required_floor_bytes"] <= copy["limit_bytes"] == graph.options.max_peak_bytes


@pytest.mark.parametrize("recipe", ["weighted_svd", "asvd"])
def test_calibration_copy_peak_refused_before_clone(monkeypatch, recipe):
    import psutil
    model, data = _model(), _input()
    graph = prepare_fixed_graph(model, data, recipe=recipe, rank=1)
    monkeypatch.setattr(psutil, "Process", lambda _pid: SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=graph.options.max_peak_bytes)))
    monkeypatch.setattr(torch.Tensor, "clone", _forbid)
    with pytest.raises(CompressionGraphError) as error:
        execute_fixed_graph(model, data, graph)
    assert error.value.violations[0].code == "ResourceLimitExceeded"
    assert error.value.violations[0].path == "calibration.copy"


@pytest.mark.parametrize("recipe", ["weighted_svd", "asvd"])
@pytest.mark.parametrize("failure", ["python_memory", "torch_cpu_allocator"])
def test_calibration_clone_oom_is_structured_and_preserves_source(monkeypatch, recipe, failure):
    model, data = _model(), _input()
    source, observations = deepcopy(model.state_dict()), data.clone()
    graph = prepare_fixed_graph(model, data, recipe=recipe, rank=1)
    def oom(_tensor, *args, **kwargs):
        if failure == "python_memory":
            raise MemoryError("deliberate clone failure")
        raise RuntimeError("DefaultCPUAllocator: not enough memory: you tried to allocate 288 bytes")
    monkeypatch.setattr(torch.Tensor, "clone", oom)
    with pytest.raises(CompressionGraphError) as error:
        execute_fixed_graph(model, data, graph)
    assert error.value.violations[0].code == "ResourceLimitExceeded"
    torch.testing.assert_close(data, observations, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, source[name], rtol=0, atol=0)


@pytest.mark.parametrize("recipe", ["weighted_svd", "asvd"])
def test_backend_oom_is_structured_without_retry_or_calibration_mutation(monkeypatch, recipe):
    from fedcore.algorithm.low_rank import graph_execution, method_execution
    model, data = _model(), _input()
    observations = data.clone()
    graph = prepare_fixed_graph(model, data, recipe=recipe, rank=1)
    calls = []
    def oom(_model, working, *args, **kwargs):
        calls.append(working)
        working.zero_()
        raise MemoryError("deliberate backend failure")
    monkeypatch.setattr(graph_execution if recipe == "weighted_svd" else method_execution,
                        "transform_weighted" if recipe == "weighted_svd" else "transform_method", oom)
    with pytest.raises(CompressionGraphError) as error:
        execute_fixed_graph(model, data, graph)
    assert error.value.violations[0].code == "ResourceLimitExceeded"
    assert error.value.violations[0].path == "runtime" and len(calls) == 1
    torch.testing.assert_close(data, observations, rtol=0, atol=0)


class RandomForwardModel(AliasModel):
    def forward(self, value):
        return super().forward(value) + random.random() + float(np.random.random()) + torch.rand(1)


@pytest.mark.parametrize("recipe", ["weighted_svd", "asvd"])
def test_success_preserves_python_numpy_and_torch_rng(recipe):
    model, data = RandomForwardModel(), _input()
    graph = prepare_fixed_graph(model, data, recipe=recipe, rank=1)
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    execute_fixed_graph(model, data, graph)
    assert random.getstate() == python_rng and torch.equal(torch_rng, torch.get_rng_state())
    np.testing.assert_equal(np.random.get_state(), numpy_rng)


@pytest.mark.parametrize("field,value", [("schema_version", True), ("schema_version", 1.),
    ("request", "bad"), ("resource_limits", "bad")])
def test_malformed_nested_plan_is_structured_failure(field, value):
    graph = prepare_fixed_graph(_model(), _input(), rank=1)
    graph = replace(graph, transform=replace(graph.transform, **{field: value}))
    assert isinstance(validate_fixed_graph(graph), ValidationFailure)


def test_tensor_payload_cannot_enter_metadata_graph():
    graph = prepare_fixed_graph(_model(), _input(), rank=1)
    request = replace(graph.transform.request, metric_policy=MetricPolicy(ridge=torch.ones(1)))
    graph = replace(graph, transform=replace(graph.transform, request=request))
    failure = validate_fixed_graph(graph)
    assert failure.violations[0].code == "LiveResource"


def test_numeric_metadata_overflow_has_structured_failure():
    graph = prepare_fixed_graph(_model(), _input(), rank=1)
    request = replace(graph.transform.request, metric_policy=MetricPolicy(ridge=10**1000))
    failure = validate_fixed_graph(replace(graph, transform=replace(graph.transform, request=request)))
    assert failure.violations[0].code == "InvalidTransform"


@pytest.mark.parametrize("recipe", ["weighted_svd", "asvd"])
def test_nested_operator_writes_refused_during_preparation(monkeypatch, recipe):
    from fedcore.algorithm.low_rank import execution
    model = nn.Linear(4, 4).double()
    model.add_module("nested", nn.Linear(4, 4).double())
    monkeypatch.setattr(execution, "deepcopy", _forbid)
    with pytest.raises(CompressionGraphError) as error:
        prepare_fixed_graph(model, _input(), recipe=recipe, rank=1)
    assert error.value.violations[0].code == "TopologyConflict"


def test_forged_alias_scope_rejected_before_effects(monkeypatch):
    from fedcore.algorithm.low_rank import execution
    model, data = AliasModel(), _input()
    graph = prepare_fixed_graph(model, data, rank=1)
    forged = compile_fixed_graph(graph.recipe, graph.transform, graph.source, graph.calibration,
        graph.options, module_aliases=(("a", "invented_alias"),))
    assert validate_fixed_graph(forged) is None
    monkeypatch.setattr(execution, "deepcopy", _forbid)
    with pytest.raises(CompressionGraphError) as error:
        execute_fixed_graph(model, data, forged)
    assert error.value.violations[0].code == "CapabilityMismatch"


@given(rows=st.integers(1, 20), columns=st.integers(1, 20))
def test_pure_compile_deterministic_for_valid_operator_shapes(rows, columns):
    request = TransformRequest(Goal.REPLACE_OPERATOR, "input_second_moment", "svd", FixedRank(1),
                               representation="two_layers")
    plan = plan_transform(request, (OperatorDescriptor("layer", (rows, columns), "Linear"),), ResourceLimits(12 * 1024**3))
    args = (Recipe.WEIGHTED_SVD, plan, ModelVersion("declared-v1", "1" * 64),
            CalibrationRef("2" * 64, (4, columns), "float64"))
    graph = compile_fixed_graph(*args)
    assert not isinstance(graph, ValidationFailure)
    assert graph == compile_fixed_graph(*args) and validate_fixed_graph(graph) is None
    assert [node.stage for node in graph.nodes] == [Stage.OBSERVE, Stage.METRIC, Stage.SOLVE, Stage.BUILD, Stage.APPLY]


def test_new_facade_preserves_trainer_boundary():
    from fedcore.algorithm.low_rank.low_rank_opt import LowRankModel
    # Registry persistence has separate integration tests. This boundary test
    # records registrations without filesystem/device initialization.
    wrapper = LowRankModel.__new__(LowRankModel)
    wrapper._model_id_before = wrapper._model_id_after = None
    wrapper._model_before_cached = wrapper._model_after_cached = None
    wrapper._fedcore_id = "graph-test"
    registered = []
    def register(**values):
        registered.append(values)
        return str(len(registered))
    wrapper._registry = SimpleNamespace(register_model=register)
    model, data = _model(), _input()
    wrapper.trainer = SimpleNamespace(model=model)
    graph = wrapper.plan_compression_graph(model, data, rank=1)
    assert wrapper.trainer.model is model
    result = wrapper.compress_graph(model, data, graph)
    assert wrapper.model_before is model and wrapper.model_after is result.model
    assert wrapper.trainer.model is result.model and wrapper.trainer._fedcore_requires_optimizer_rebuild
    assert [value["model"] for value in registered] == [model, result.model]
