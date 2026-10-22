"""Independent numerical, storage and ownership witnesses for G5 composition.

The oracles use NumPy algebra and registered backing storages. They do not call
the production solver, composition validator, inventory or packed decoder.
"""
from dataclasses import fields, is_dataclass, replace
import random

from hypothesis import given, settings, strategies as st
import numpy as np
import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.compression_composition import (
    AddResidual, CheckedComposition, CompositionRequest, CompositionRecipe,
    CompositionSuccess, LowRank, MaterializeDense, PruneMask, QuantizeRemapped,
    execute_composition, prepare_composition,
)
from fedcore.algorithm.low_rank.dobi_remapping import RemappedLinear
from fedcore.algorithm.low_rank.plans import ValidationFailure
from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear, ResidualLinear
from fedcore.tools.registry.checkpoint_manager import CheckpointManager


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _model(weight=None, *, bias=True):
    weight = np.array([[3., 1., -2., .5], [1., -2., .25, 4.],
                       [2., .5, 1., -1.]]) if weight is None else np.asarray(weight)
    with torch.random.fork_rng(devices=[]):
        model = nn.Sequential(nn.Linear(weight.shape[1], weight.shape[0],
                                        bias=bias, dtype=torch.float64))
    with torch.no_grad():
        model[0].weight.copy_(torch.as_tensor(weight, dtype=torch.float64))
        if bias:
            model[0].bias.copy_(torch.arange(weight.shape[0], dtype=torch.float64) / 7)
    return model.eval()


def _request(*steps, **kwargs):
    return CompositionRequest((CompositionRecipe("0", tuple(steps)),), **kwargs)


def _prepare(model, *steps, **kwargs):
    result = prepare_composition(model, _request(*steps, **kwargs))
    assert type(result) is CheckedComposition, result
    return result


def _execute(model, *steps, **kwargs):
    result = execute_composition(model, _prepare(model, *steps, **kwargs))
    assert type(result) is CompositionSuccess, result
    return result


def _failure(value):
    assert type(value) is ValidationFailure, value
    assert value.violations
    assert all(item.code and item.path and item.message for item in value.violations)
    return {item.code for item in value.violations}


def _numpy_weight(layer):
    if type(layer) is nn.Linear:
        return layer.weight.detach().numpy().copy()
    if type(layer) is FactorizedLinear:
        return layer.left.weight.detach().numpy() @ layer.right.weight.detach().numpy()
    if type(layer) is RemappedLinear:
        # Decode the named integer codes, component scales and long-side tail
        # directly, without calling decoded_factors or materializing helpers.
        left = layer.left_codes.numpy().astype(np.float64) * layer.left_scale.numpy()[None, :]
        right = layer.right_codes.numpy().astype(np.float64) * layer.right_scale.numpy()[:, None]
        tail = layer.tail.numpy().astype(np.float64)
        if layer.out_features >= layer.in_features:
            left = np.concatenate((left, tail), axis=0)
        else:
            right = np.concatenate((right, tail), axis=1)
        return left @ right
    if type(layer) is ResidualLinear:
        return _numpy_weight(layer.base) + _numpy_weight(layer.residual)
    raise AssertionError(type(layer))


def _truncated(weight, rank):
    left, singular, right = np.linalg.svd(weight, full_matrices=False)
    return (left[:, :rank] * singular[:rank]) @ right[:rank]


def _physical_bytes(model):
    storages = {}
    for module in model.modules():
        for tensor in (*module._parameters.values(), *module._buffers.values()):
            if tensor is not None:
                storage = tensor.untyped_storage()
                storages[(str(tensor.device), storage._cdata)] = storage.nbytes()
    return sum(storages.values())


def _owned_state(model):
    tensors = []
    for path, module in model.named_modules(remove_duplicate=False):
        for kind, collection in (("parameter", module._parameters), ("buffer", module._buffers)):
            for name, value in collection.items():
                if value is None:
                    continue
                storage = value.untyped_storage()
                raw = torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))
                tensors.append((path, kind, name, id(value), storage._cdata,
                                tuple(value.shape), tuple(value.stride()), value.storage_offset(),
                                value.requires_grad, raw.clone()))
    return (tensors, tuple((path, id(module), module.training) for path, module in
                           model.named_modules(remove_duplicate=False)),
            random.getstate(), np.random.get_state(), torch.get_rng_state().clone())


def _assert_owned_state(model, before):
    after = _owned_state(model)
    assert len(after[0]) == len(before[0])
    for current, original in zip(after[0], before[0]):
        assert current[:-1] == original[:-1]
        assert torch.equal(current[-1], original[-1])
    assert after[1:3] == before[1:3]
    assert after[3][0] == before[3][0]
    assert np.array_equal(after[3][1], before[3][1])
    assert after[3][2:] == before[3][2:]
    assert torch.equal(after[4], before[4])


def _forbid_effects(monkeypatch):
    import fedcore.algorithm.low_rank.compression_composition as composition

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid composition reached copying, factorization or forward")

    monkeypatch.setattr(torch.linalg, "svd", forbidden)
    monkeypatch.setattr(nn.Linear, "forward", forbidden)
    if hasattr(composition, "clone_preserving_storages"):
        monkeypatch.setattr(composition, "clone_preserving_storages", forbidden)


def test_noncommuting_rank_and_mask_follow_declared_order_and_explicit_materialization():
    model = _model(np.diag([2., 1.]), bias=False)
    before = _owned_state(model)
    rank_then_mask = _execute(model, LowRank(1), MaterializeDense(), PruneMask((0, 1, 2)))
    mask_then_rank = _execute(model, PruneMask((0, 1, 2)), LowRank(1))
    np.testing.assert_allclose(_numpy_weight(rank_then_mask.model[0]), np.zeros((2, 2)), atol=1e-14)
    np.testing.assert_allclose(_numpy_weight(mask_then_rank.model[0]), np.diag([0., 1.]), atol=1e-14)
    assert type(rank_then_mask.model[0]) is nn.Linear
    assert type(mask_then_rank.model[0]) is FactorizedLinear
    assert rank_then_mask.fingerprint != mask_then_rank.fingerprint
    assert len(rank_then_mask.history) == 3 and len(mask_then_rank.history) == 2
    assert [(item.recipe_index, item.step_index, item.path) for item in rank_then_mask.history] == [
        (0, 0, "0"), (0, 1, "0"), (0, 2, "0")]
    _assert_owned_state(model, before)


@given(indices=st.lists(st.integers(0, 11), unique=True, max_size=12).map(lambda xs: tuple(sorted(xs))))
@settings(max_examples=18, deadline=None)
def test_each_declared_mask_zero_is_exact_and_other_weights_and_bias_are_preserved(indices):
    model = _model()
    result = _execute(model, PruneMask(indices))
    expected = model[0].weight.detach().numpy().copy()
    expected.reshape(-1)[list(indices)] = 0
    np.testing.assert_array_equal(_numpy_weight(result.model[0]), expected)
    torch.testing.assert_close(result.model[0].bias, model[0].bias, rtol=0, atol=0)
    # This mask has dense storage; a high zero count alone is not a byte saving.
    assert _physical_bytes(result.model) == _physical_bytes(model)


def test_low_rank_does_not_silently_claim_a_preceding_element_zero_pattern():
    model = _model([[3., 1.], [2., 4.]], bias=False)
    result = _execute(model, PruneMask((0,)), LowRank(1))
    masked = model[0].weight.detach().numpy().copy()
    masked[0, 0] = 0
    np.testing.assert_allclose(_numpy_weight(result.model[0]), _truncated(masked, 1), atol=2e-14)
    assert abs(_numpy_weight(result.model[0])[0, 0]) > .1
    assert result.model[0].rank == 1


@pytest.mark.parametrize("last", [PruneMask((0,)), LowRank(1)])
def test_factor_to_dense_only_operation_requires_explicit_materialization(monkeypatch, last):
    model = _model()
    before = _owned_state(model)
    _forbid_effects(monkeypatch)
    _failure(prepare_composition(model, _request(LowRank(1), last)))
    _assert_owned_state(model, before)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_packed_factor_runtime_has_independent_decode_and_structural_rank_bound(rank):
    model = _model()
    result = _execute(model, LowRank(rank), QuantizeRemapped(
        tail_dtype="float64", scale_dtype="float64", compute_dtype="float64"))
    layer = result.model[0]
    assert type(layer) is RemappedLinear and layer.rank == rank
    expected = _numpy_weight(layer)
    assert np.linalg.matrix_rank(expected, tol=1e-10) <= rank
    inputs = np.array([[1., -2., .5, 3.], [.25, 1., 2., -.5]])
    prediction = result.model(torch.as_tensor(inputs, dtype=torch.float64)).detach().numpy()
    np.testing.assert_allclose(prediction, inputs @ expected.T + model[0].bias.detach().numpy(),
                               rtol=3e-14, atol=3e-14)
    # Two 3-by-rank int8 blocks, one rank-wide float64 tail, two scales,
    # and the original three-element float64 bias are all physically stored.
    assert _physical_bytes(result.model) == 2 * 3 * rank + rank * 8 + 2 * rank * 8 + 3 * 8


@pytest.mark.parametrize("steps", [
    (LowRank(1),),
    (LowRank(1), QuantizeRemapped(tail_dtype="float64", scale_dtype="float64", compute_dtype="float64")),
    (LowRank(1), QuantizeRemapped(tail_dtype="float64", scale_dtype="float64", compute_dtype="float64"),
     AddResidual(1)),
])
def test_serialized_artifact_executes_and_exact_budget_matches_its_reloaded_storage(steps):
    model = _model()
    initial = _execute(model, *steps)
    actual_bytes = _physical_bytes(initial.model)
    assert isinstance(initial.checkpoint_bytes, bytes) and initial.checkpoint_bytes
    loaded = CheckpointManager("", auto_cleanup=False).deserialize_from_bytes(initial.checkpoint_bytes)
    assert _physical_bytes(loaded) == actual_bytes
    inputs = torch.tensor([[1., -2., .5, 3.], [.25, 1., 2., -.5]], dtype=torch.float64)
    torch.testing.assert_close(loaded(inputs), initial.model(inputs), rtol=1e-13, atol=1e-13)
    assert sum(item.nbytes for item in initial.inventory) == actual_bytes
    exact = _execute(model, *steps, max_storage_bytes=actual_bytes)
    assert _physical_bytes(exact.model) == actual_bytes
    checked = prepare_composition(model, _request(*steps, max_storage_bytes=actual_bytes - 1))
    if type(checked) is ValidationFailure:
        _failure(checked)
    else:
        _failure(execute_composition(model, checked))


def test_residual_corrects_packed_base_toward_original_and_dense_merge_rebudgets():
    model = _model()
    packed_steps = (LowRank(1), QuantizeRemapped(
        tail_dtype="float64", scale_dtype="float64", compute_dtype="float64"))
    base = _execute(model, *packed_steps)
    kept = _execute(model, *packed_steps, AddResidual(1))
    merged = _execute(model, *packed_steps, AddResidual(1), MaterializeDense())
    baseline = _numpy_weight(base.model[0])
    expected = baseline + _truncated(model[0].weight.detach().numpy() - baseline, 1)
    assert type(kept.model[0]) is ResidualLinear
    assert type(kept.model[0].base) is RemappedLinear
    assert type(merged.model[0]) is nn.Linear
    np.testing.assert_allclose(_numpy_weight(kept.model[0]), expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(_numpy_weight(merged.model[0]), expected, rtol=2e-13, atol=2e-13)
    assert _physical_bytes(kept.model) == _physical_bytes(base.model) + (3 + 4) * 8
    assert _physical_bytes(merged.model) == _physical_bytes(model) > _physical_bytes(kept.model)
    request = _request(*packed_steps, AddResidual(1), MaterializeDense(),
                       max_storage_bytes=_physical_bytes(kept.model))
    checked = prepare_composition(model, request)
    _failure(checked if type(checked) is ValidationFailure else execute_composition(model, checked))


def test_unchanged_reference_coordinate_maps_and_indices_are_in_the_complete_artifact_budget():
    from fedcore.algorithm.low_rank.coupled_layers import ReferenceAttention

    with torch.random.fork_rng(devices=[]):
        model = nn.Sequential(nn.Linear(4, 4, dtype=torch.float64),
                              ReferenceAttention(4, 4, 2, 4, dtype=torch.float64)).eval()
    integer_bytes = sum(buffer.untyped_storage().nbytes() for buffer in model[1].buffers()
                        if not buffer.is_floating_point())
    assert integer_bytes > 0
    unchanged_bytes = _physical_bytes(model[1])
    result = _execute(model, LowRank(1))
    expected = (4 + 4) * 8 + 4 * 8 + unchanged_bytes
    assert _physical_bytes(result.model) == result.storage_bytes == expected
    assert sum(item.nbytes for item in result.inventory) == expected
    assert torch.equal(result.model[1].head_map, model[1].head_map)
    assert torch.equal(result.model[1].group_coordinates, model[1].group_coordinates)
    exact = _execute(model, LowRank(1), max_storage_bytes=expected)
    inputs = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4) / 16
    loaded = CheckpointManager("", auto_cleanup=False).deserialize_from_bytes(exact.checkpoint_bytes)
    torch.testing.assert_close(loaded(inputs), exact.model(inputs), rtol=1e-13, atol=1e-13)
    assert _physical_bytes(loaded) == expected
    assert "BudgetExceeded" in _failure(prepare_composition(
        model, _request(LowRank(1), max_storage_bytes=expected - 1)))


def test_unchanged_shared_basis_has_one_physical_cost_and_one_reloaded_parameter():
    from fedcore.algorithm.low_rank.structured_layers import SharedBasisLinear

    with torch.random.fork_rng(devices=[]):
        basis = nn.Parameter(torch.tensor([[1., 2., -1., .5]], dtype=torch.float64))
        model = nn.Sequential(nn.Linear(4, 4, dtype=torch.float64),
                              SharedBasisLinear(4, 4, 1, basis=basis),
                              SharedBasisLinear(4, 4, 1, basis=basis)).eval()
    untouched = _physical_bytes(nn.Sequential(model[1], model[2]))
    # One four-element basis, two four-element lefts and two biases.
    assert untouched == (4 + 2 * 4 + 2 * 4) * 8
    result = _execute(model, LowRank(1))
    expected = (4 + 4 + 4) * 8 + untouched
    assert result.model[1].right is result.model[2].right
    assert result.storage_bytes == _physical_bytes(result.model) == expected
    exact = _execute(model, LowRank(1), max_storage_bytes=expected)
    loaded = CheckpointManager("", auto_cleanup=False).deserialize_from_bytes(exact.checkpoint_bytes)
    assert loaded[1].right is loaded[2].right
    assert _physical_bytes(loaded) == expected
    inputs = torch.arange(12, dtype=torch.float64).reshape(3, 4) / 8
    torch.testing.assert_close(loaded(inputs), exact.model(inputs), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("steps", [
    (PruneMask((0, 3)),), (LowRank(1),),
    (LowRank(1), QuantizeRemapped(compute_dtype="float64"), AddResidual(1)),
])
def test_preparation_and_execution_preserve_source_storage_modes_and_all_rng_states(steps):
    model = _model()
    model.train()
    model[0].eval()
    before = _owned_state(model)
    checked = _prepare(model, *steps)
    _assert_owned_state(model, before)
    result = execute_composition(model, checked)
    assert type(result) is CompositionSuccess, result
    _assert_owned_state(model, before)
    source_ids = {tensor.untyped_storage()._cdata for tensor in model.parameters()}
    assert all(tensor.untyped_storage()._cdata not in source_ids for tensor in result.model.parameters())


@pytest.mark.parametrize("change", ["weight", "bias", "dimensions", "forward", "hook"])
def test_source_or_runtime_change_is_rejected_before_copy_solver_or_forward(monkeypatch, change):
    model = _model()
    checked = _prepare(model, LowRank(1))
    if change == "weight":
        with torch.no_grad():
            model[0].weight[0, 0] += .25
    elif change == "bias":
        with torch.no_grad():
            model[0].bias[0] += .25
    elif change == "dimensions":
        model[0].in_features += 1
    elif change == "forward":
        model[0].forward = lambda value: value
    else:
        model[0].register_forward_pre_hook(lambda module, args: None)
    before = _owned_state(model)
    _forbid_effects(monkeypatch)
    _failure(execute_composition(model, checked))
    _assert_owned_state(model, before)


@pytest.mark.parametrize("method", ["parameters", "buffers", "named_modules"])
def test_instance_traversal_overrides_are_rejected_without_invoking_the_callback(monkeypatch, method):
    model = _model()
    target = model if method == "named_modules" else model[0]
    original, calls = getattr(target, method), []

    def callback(*args, **kwargs):
        calls.append(method)
        return original(*args, **kwargs)

    setattr(target, method, callback)
    _forbid_effects(monkeypatch)
    _failure(prepare_composition(model, _request(LowRank(1))))
    assert calls == []


@pytest.mark.parametrize("location", ["scope", "nested_under_primitive"])
def test_unsupported_subclass_state_callback_is_never_invoked_by_checkpoint_preflight(monkeypatch, location):
    calls = []

    class StatefulLinear(nn.Linear):
        def _save_to_state_dict(self, destination, prefix, keep_vars):
            calls.append("save")
            return super()._save_to_state_dict(destination, prefix, keep_vars)

    with torch.random.fork_rng(devices=[]):
        model = nn.Sequential(nn.Linear(4, 3, dtype=torch.float64))
        if location == "scope":
            model[0] = StatefulLinear(4, 3, dtype=torch.float64)
        else:
            model[0].add_module("unexpected", StatefulLinear(4, 3, dtype=torch.float64))
    _forbid_effects(monkeypatch)
    _failure(prepare_composition(model, _request(LowRank(1))))
    assert calls == []


def test_unused_source_backing_bytes_cannot_be_silently_dropped_by_the_checkpoint(monkeypatch):
    model = _model()
    backing = torch.arange(24, dtype=torch.float64)
    weight = backing[3:15].reshape(3, 4)
    weight.copy_(model[0].weight.detach())
    model[0].weight = nn.Parameter(weight)
    before = _owned_state(model)
    _forbid_effects(monkeypatch)
    assert "UnsupportedExport" in _failure(prepare_composition(model, _request(LowRank(1))))
    _assert_owned_state(model, before)


def test_nonpersistent_source_buffer_is_explicitly_refused_before_transform(monkeypatch):
    model = _model()
    model[0].register_buffer("indices", torch.tensor([1, 2, 0]), persistent=False)
    before = _owned_state(model)
    _forbid_effects(monkeypatch)
    assert "UnsupportedExport" in _failure(prepare_composition(model, _request(LowRank(1))))
    _assert_owned_state(model, before)


@pytest.mark.parametrize("composition_request", [
    CompositionRequest(()), CompositionRequest((CompositionRecipe("0", ()),)),
    _request(LowRank(0)), _request(LowRank(True)), _request(LowRank(4)),
    _request(PruneMask((-1,))), _request(PruneMask((12,))),
    _request(LowRank(1), max_storage_bytes=0),
    _request(LowRank(1), max_storage_bytes=True),
    _request(LowRank(1), export_format="opaque_torchao"),
    _request(LowRank(1), QuantizeRemapped(compute_dtype="float32")),
])
def test_invalid_or_unsupported_configuration_refuses_before_any_effect(monkeypatch, composition_request):
    model = _model()
    before = _owned_state(model)
    _forbid_effects(monkeypatch)
    _failure(prepare_composition(model, composition_request))
    _assert_owned_state(model, before)


@pytest.mark.parametrize("kind", ["physical_sparse", "opaque_quantized", "custom_forward"])
def test_unsupported_operator_runtime_cannot_be_relabelled_as_dense(monkeypatch, kind):
    model = _model()
    if kind == "physical_sparse":
        model[0].weight = nn.Parameter(model[0].weight.detach().to_sparse(), requires_grad=False)
    elif kind == "opaque_quantized":
        model[0] = nn.quantized.dynamic.Linear(4, 3)
    else:
        class CustomLinear(nn.Linear):
            def forward(self, inputs):
                return super().forward(inputs) + 1

        with torch.random.fork_rng(devices=[]):
            model[0] = CustomLinear(4, 3, dtype=torch.float64)
    _forbid_effects(monkeypatch)
    _failure(prepare_composition(model, _request(LowRank(1))))


def test_checked_composition_contains_only_frozen_metadata_and_rejects_forged_order(monkeypatch):
    model = _model()
    checked = _prepare(model, LowRank(1), MaterializeDense(), PruneMask((0,)))

    def metadata(value):
        if is_dataclass(value):
            assert type(value).__dataclass_params__.frozen
            for field in fields(value):
                metadata(getattr(value, field.name))
        elif isinstance(value, tuple):
            for item in value:
                metadata(item)
        else:
            assert value is None or isinstance(value, (str, bool, int, float))

    metadata(checked)
    # A fingerprint alone cannot authorize a request with a different order.
    changed = replace(checked, request=_request(PruneMask((0,)), LowRank(1), MaterializeDense()))
    _forbid_effects(monkeypatch)
    _failure(execute_composition(model, changed))
