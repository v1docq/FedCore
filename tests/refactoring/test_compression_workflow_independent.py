"""Independent contract witnesses for the fixed whole-recipe workflow boundary."""
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.compression_graph import (
    ArtifactKind, Recipe, SourceMode, compile_fixed_graph, graph_fingerprint,
)
from fedcore.algorithm.low_rank.graph_execution import (
    CompressionGraphError, execute_fixed_graph, prepare_fixed_graph,
)
from fedcore.algorithm.low_rank.method_specs import ASVD
from fedcore.algorithm.low_rank.plans import MetricPolicy


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _case():
    generator = torch.Generator().manual_seed(912)
    model = nn.Sequential(nn.Linear(4, 4), nn.LeakyReLU(.12), nn.Linear(4, 3)).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=torch.float64))
    inputs = torch.randn((11, 4), generator=generator, dtype=torch.float64)
    return model, inputs


def _prepare(model, inputs, recipe="weighted_svd", **options):
    return prepare_fixed_graph(model, inputs, recipe=recipe, target_paths=("0", "2"),
                               rank=1, graph_version="independent-g1-v1", **options)


def _reseal(graph):
    # A digest is deliberately recomputed: checksum rejection cannot establish
    # that the semantic constraints below have actually been enforced.
    return replace(graph, fingerprint=graph_fingerprint(graph))


def _forbid_effects(monkeypatch):
    import fedcore.algorithm.low_rank.execution as weighted
    import fedcore.algorithm.low_rank.graph_execution as graph_execution
    import fedcore.algorithm.low_rank.method_execution as methods

    def forbidden(*args, **kwargs):
        raise AssertionError("Static refusal happened after a copy, hook or recipe execution")

    monkeypatch.setattr(weighted, "deepcopy", forbidden)
    monkeypatch.setattr(methods, "deepcopy", forbidden)
    monkeypatch.setattr(graph_execution, "transform_weighted", forbidden)
    monkeypatch.setattr(methods, "transform_method", forbidden)
    monkeypatch.setattr(nn.Module, "register_forward_pre_hook", forbidden)
    monkeypatch.setattr(nn.Module, "register_forward_hook", forbidden)


def _mutate(graph, mutation):
    nodes = list(graph.nodes)
    if mutation == "missing_apply":
        return replace(graph, nodes=graph.nodes[:-1])
    if mutation == "reordered_independent_blocks":
        # This remains a topologically valid DAG: each weighted block has its
        # producers, and both replacements exist before the final batch apply.
        return replace(graph, nodes=graph.nodes[4:8] + graph.nodes[:4] + graph.nodes[8:])
    if mutation == "wrong_statistics_with_matching_consumer":
        wrong = ArtifactKind.CENTERED_INPUT_COVARIANCE
        nodes[0] = replace(nodes[0], outputs=(replace(nodes[0].outputs[0], kind=wrong),))
        nodes[1] = replace(nodes[1], inputs=(replace(nodes[1].inputs[0], kind=wrong),))
    elif mutation == "foreign_model_version":
        nodes[0] = replace(nodes[0], outputs=(replace(nodes[0].outputs[0], model_version="foreign:initial"),))
    elif mutation == "wrong_data_reference":
        nodes[0] = replace(nodes[0], inputs=(nodes[0].inputs[0], replace(nodes[0].inputs[1], artifact="model:initial")))
    elif mutation == "wrong_write_scope":
        nodes[-1] = replace(nodes[-1], writes=("0",))
    elif mutation == "current_student_reads_old_source":
        nodes[5] = replace(nodes[5], inputs=(replace(nodes[5].inputs[0], artifact="model:initial"), nodes[5].inputs[1]))
    elif mutation == "wrong_mode":
        return replace(graph, source_mode=SourceMode.CURRENT_STUDENT)
    elif mutation == "ignored_solver":
        return replace(graph, transform=replace(graph.transform,
            request=replace(graph.transform.request, solver="an_unexecuted_solver")))
    else:
        raise AssertionError(mutation)
    return replace(graph, nodes=tuple(nodes))


@pytest.mark.parametrize("recipe,mutation", [
    ("weighted_svd", "missing_apply"),
    ("weighted_svd", "reordered_independent_blocks"),
    ("weighted_svd", "wrong_statistics_with_matching_consumer"),
    ("weighted_svd", "foreign_model_version"),
    ("weighted_svd", "wrong_data_reference"),
    ("weighted_svd", "wrong_write_scope"),
    ("weighted_svd", "wrong_mode"),
    ("asvd", "current_student_reads_old_source"),
    ("asvd", "ignored_solver"),
])
def test_resealed_semantic_mutations_refuse_before_any_execution(monkeypatch, recipe, mutation):
    model, inputs = _case()
    graph = _prepare(model, inputs, recipe)
    changed = _reseal(_mutate(graph, mutation))
    assert changed.fingerprint != graph.fingerprint
    _forbid_effects(monkeypatch)
    with pytest.raises(CompressionGraphError) as caught:
        execute_fixed_graph(model, inputs, changed)
    assert caught.value.violations


@pytest.mark.parametrize("aliases", [(), (("0", "a_path_that_does_not_exist"),)])
def test_recompiled_alias_scope_must_match_real_python_identity(monkeypatch, aliases):
    layer = nn.Linear(4, 4).double()
    model = nn.Sequential(layer, nn.Tanh(), layer)
    inputs = torch.eye(4, dtype=torch.float64)
    assert model[0] is model[2]
    graph = prepare_fixed_graph(model, inputs, target_paths=("0",), rank=1)
    changed = compile_fixed_graph(graph.recipe, graph.transform, graph.source,
        graph.calibration, graph.options, method_spec=graph.method_spec, module_aliases=aliases)
    # Both candidates are internally canonical and carry a fresh valid digest;
    # only comparison with the actual model can reveal the scope mismatch.
    assert changed.module_aliases == aliases
    _forbid_effects(monkeypatch)
    with pytest.raises(CompressionGraphError) as caught:
        execute_fixed_graph(model, inputs, changed)
    assert caught.value.violations


@pytest.mark.parametrize("mutation,expected_code", [
    ("weight", "StaleModelVersion"),
    ("nonweight_config", "StaleModelVersion"),
    ("calibration", "StaleCalibration"),
])
def test_same_shape_runtime_changes_are_stale_before_effects(monkeypatch, mutation, expected_code):
    model, inputs = _case()
    graph = _prepare(model, inputs)
    shapes = tuple(tuple(parameter.shape) for parameter in model.parameters())
    if mutation == "weight":
        with torch.no_grad():
            model[0].weight[0, 0].add_(.1)
    elif mutation == "nonweight_config":
        model[1].negative_slope = .43
    else:
        inputs = inputs.clone()
        inputs[0, 0] += .1
    assert tuple(tuple(parameter.shape) for parameter in model.parameters()) == shapes
    _forbid_effects(monkeypatch)
    with pytest.raises(CompressionGraphError) as caught:
        execute_fixed_graph(model, inputs, graph)
    assert expected_code in {violation.code for violation in caught.value.violations}


@pytest.mark.parametrize("recipe", ["weighted_svd", "asvd"])
def test_runtime_dispatch_passes_exact_declared_parameters(monkeypatch, recipe):
    import fedcore.algorithm.low_rank.graph_execution as execution
    import fedcore.algorithm.low_rank.method_execution as methods

    model, inputs = _case()
    common = dict(batch_size=3, max_workspace_bytes=32 * 1024**2,
                  max_peak_bytes=11 * 1024**3, graph_version="dispatch-protocol-v1")
    if recipe == "weighted_svd":
        specific = dict(rank_ratio=.5, policy=MetricPolicy(ridge=.03, rcond=1e-8), representation="one_layer")
        graph = prepare_fixed_graph(model, inputs, target_paths=("0", "2"), **common, **specific)
    else:
        specific = dict(ranks={"0": 1, "2": 2}, parameter_budget=40, tensor_byte_budget=320)
        graph = prepare_fixed_graph(model, inputs, recipe=recipe, method_spec=ASVD(alpha=.7, epsilon=.01),
                                    target_paths=("0", "2"), **common, **specific)
    calls = []
    sentinel = SimpleNamespace(model=object(), evidence={"recorded": True})

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(execution, "transform_weighted", record)
    monkeypatch.setattr(methods, "transform_method", record)
    result = execute_fixed_graph(model, inputs, graph)
    assert result.legacy_result is sentinel and result.model is sentinel.model
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] is model and args[1] is not inputs
    assert args[1].untyped_storage().data_ptr() != inputs.untyped_storage().data_ptr()
    torch.testing.assert_close(args[1], inputs, rtol=0, atol=0)
    assert len(args) == (2 if recipe == "weighted_svd" else 3)
    if recipe == "asvd":
        assert args[2] == ASVD(alpha=.7, epsilon=.01)
    assert kwargs == {**common, **specific, "target_paths": ("0", "2")}


@pytest.mark.parametrize("recipe", ["weighted_svd", "asvd"])
def test_public_facade_matches_real_legacy_and_preserves_owned_state(recipe):
    from fedcore.algorithm.low_rank.execution import transform_weighted
    from fedcore.algorithm.low_rank.low_rank_opt import LowRankModel
    from fedcore.algorithm.low_rank.method_execution import transform_method

    model, inputs = _case()
    model.train()
    model[0].eval()
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, .25)
    parameters = tuple(model.parameters())
    gradients = tuple(parameter.grad for parameter in parameters)
    state = deepcopy(model.state_dict())
    modes = tuple(module.training for module in model.modules())
    owner = LowRankModel({"device": "cpu"})
    owner.trainer = SimpleNamespace(model=model)
    common = dict(rank=1, target_paths=("0", "2"), batch_size=3, graph_version="parity-protocol-v1")
    spec = ASVD(alpha=.7, epsilon=.01)
    options = {"recipe": recipe, **common}
    if recipe == "asvd":
        options["method_spec"] = spec
    graph = owner.plan_compression_graph(model, inputs, **options)
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    result = owner.compress_graph(model, inputs, graph)
    assert random.getstate() == python_rng
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == numpy_rng[0] and actual_numpy[2:] == numpy_rng[2:]
    np.testing.assert_array_equal(actual_numpy[1], numpy_rng[1])
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    assert tuple(model.parameters()) == parameters
    assert all(parameter.grad is gradient for parameter, gradient in zip(parameters, gradients))
    assert all(torch.equal(parameter.grad, torch.full_like(parameter, .25)) for parameter in parameters)
    assert tuple(module.training for module in model.modules()) == modes
    for name, value in model.state_dict().items():
        assert torch.equal(value, state[name])
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
    assert owner.model_before is model and owner.model_after is result.model
    assert owner.trainer.model is result.model and owner.trainer._fedcore_requires_optimizer_rebuild
    assert result.model is not model and result.model._fedcore_requires_optimizer_rebuild
    expected = (transform_weighted(model, inputs, **common) if recipe == "weighted_svd"
                else transform_method(model, inputs, spec, **common))
    torch.testing.assert_close(result.model(inputs), expected.model(inputs), rtol=1e-11, atol=1e-11)
    evidence = result.legacy_result.evidence
    unique = {id(parameter): parameter for parameter in result.model.parameters()}.values()
    assert evidence["parameters_after"] == sum(parameter.numel() for parameter in unique)
    assert evidence["optimizer_rebuild_required"] is True
    json.dumps(result.evidence, allow_nan=False)


def test_graph_recursively_contains_only_frozen_metadata():
    model, inputs = _case()
    graph = _prepare(model, inputs, "asvd")

    def check(value):
        if is_dataclass(value):
            assert type(value).__dataclass_params__.frozen
            for field in fields(value):
                check(getattr(value, field.name))
        elif isinstance(value, tuple):
            for item in value:
                check(item)
        else:
            assert value is None or isinstance(value, (str, bool, int, float))

    check(graph)
    assert graph.recipe is Recipe.ASVD


class _RandomCalibrationModel(nn.Module):
    def __init__(self, fail):
        super().__init__()
        self.layer = nn.Linear(3, 3).double()
        self.fail = fail

    def forward(self, inputs):
        # Eval mode does not suppress randomness implemented by a user module.
        noise = .01 * (random.random() + np.random.random() + torch.rand(()))
        output = self.layer(inputs + noise)
        if self.fail:
            raise RuntimeError("independent calibration failure")
        return output


@pytest.mark.parametrize("recipe", ["weighted_svd", "asvd"])
@pytest.mark.parametrize("fail", [False, True])
def test_user_forward_preserves_all_cpu_rng_on_success_and_failure(monkeypatch, recipe, fail):
    import fedcore.algorithm.low_rank.execution as weighted
    import fedcore.algorithm.low_rank.method_execution as methods

    model = _RandomCalibrationModel(fail)
    inputs = torch.eye(3, dtype=torch.float64)
    graph = prepare_fixed_graph(model, inputs, recipe=recipe, rank=1,
                                target_paths=("layer",), graph_version="random-forward-v1")
    clones = []

    def capture_copy(value):
        clone = deepcopy(value)
        clones.append(clone)
        return clone

    monkeypatch.setattr(weighted if recipe == "weighted_svd" else methods, "deepcopy", capture_copy)
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    if fail:
        with pytest.raises(RuntimeError, match="independent calibration failure"):
            execute_fixed_graph(model, inputs, graph)
    else:
        execute_fixed_graph(model, inputs, graph)
    assert random.getstate() == python_rng
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == numpy_rng[0] and actual_numpy[2:] == numpy_rng[2:]
    np.testing.assert_array_equal(actual_numpy[1], numpy_rng[1])
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    assert clones
    assert all(not module._forward_pre_hooks and not module._forward_hooks
               for clone in clones for module in clone.modules())


class _AfterCaptureFailure(nn.Module):
    def __init__(self, fail):
        super().__init__()
        self.fail = fail

    def forward(self, inputs):
        if self.fail:
            raise RuntimeError("independent inplace calibration failure")
        return inputs


@pytest.mark.parametrize("recipe", ["weighted_svd", "asvd"])
@pytest.mark.parametrize("fail", [False, True])
def test_inplace_forward_preserves_original_calibration_bytes(recipe, fail):
    model = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(4, 3), _AfterCaptureFailure(fail)).double()
    inputs = torch.tensor([[-1., 2., -3., 4.], [5., -6., 7., -8.], [-2., -1., 3., 1.]],
                          dtype=torch.float64)
    # A noncontiguous calibration view must be protected just like an owned
    # contiguous tensor; its backing storage remains owned by the caller.
    inputs = inputs.T.contiguous().T
    before_bytes = inputs.contiguous().view(torch.uint8).clone()
    graph = prepare_fixed_graph(model, inputs, recipe=recipe, rank=1,
                                target_paths=("1",), batch_size=2)
    if fail:
        with pytest.raises(RuntimeError, match="independent inplace calibration failure"):
            execute_fixed_graph(model, inputs, graph)
    else:
        result = execute_fixed_graph(model, inputs, graph)
        copy_evidence = result.evidence["calibration_copy"]
        assert copy_evidence["copy_bytes"] == inputs.numel() * inputs.element_size()
        assert copy_evidence["required_floor_bytes"] <= graph.options.max_peak_bytes
    assert torch.equal(inputs.contiguous().view(torch.uint8), before_bytes)


def test_working_calibration_bytes_are_budgeted_before_clone(monkeypatch):
    import psutil

    model, inputs = _case()
    rss = 8 * 1024**2
    reserve = 64 * 1024**2
    copied_bytes = inputs.numel() * inputs.element_size()
    # The model plan itself is tiny and fits this limit. The new owned input
    # copy misses its declared peak by exactly one byte.
    limit = rss + reserve + copied_bytes - 1
    graph = _prepare(model, inputs, max_peak_bytes=limit, max_workspace_bytes=32 * 1024**2)
    monkeypatch.setattr(psutil.Process, "memory_info", lambda self: SimpleNamespace(rss=rss))

    def forbidden_clone(*args, **kwargs):
        raise AssertionError("Calibration storage was copied before its memory preflight")

    monkeypatch.setattr(torch.Tensor, "clone", forbidden_clone)
    _forbid_effects(monkeypatch)
    with pytest.raises(CompressionGraphError) as caught:
        execute_fixed_graph(model, inputs, graph)
    assert "ResourceLimitExceeded" in {violation.code for violation in caught.value.violations}
