"""Independent numerical and ownership witnesses for finite G4 group execution.

The numerical oracles use NumPy matrix operations, never the production solver,
collector, fingerprint validator, or grouped workflow validator.
"""
from dataclasses import fields, is_dataclass, replace
import json
import random
from types import SimpleNamespace

from hypothesis import given, settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule
import numpy as np
import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.artifacts import ArtifactStoreError, StatisticsStore
from fedcore.algorithm.low_rank.compression_graph import SourceMode
from fedcore.algorithm.low_rank.grouped_workflow import (
    CompressionRegion, ConsumerCoordinates, CoordinateAxis, GroupRequest,
    grouped_fingerprint,
)
from fedcore.algorithm.low_rank.grouped_execution import (
    GroupedWorkflowError, execute_grouped_workflow, prepare_grouped_workflow,
)
from fedcore.algorithm.low_rank.method_specs import BasisSharing, DRONE, SVDLLMV1


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _case(seed=912):
    generator = torch.Generator().manual_seed(seed)
    # Constructor randomness must not interfere with the RNG ownership witness.
    with torch.random.fork_rng(devices=[]):
        model = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 4),
                              nn.Sigmoid(), nn.Linear(4, 3)).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator,
                                        dtype=torch.float64) * .8)
    inputs = torch.randn((13, 4), generator=generator, dtype=torch.float64)
    inputs = inputs * torch.tensor([.2, 2., 1., 4.]) + .3
    return model, inputs


def _basis_case(seed=93):
    model, inputs = _case(seed)
    # The group explicitly declares that these equal-width inputs use one
    # coordinate atlas. This is a caller assumption for generic Linear graphs.
    return model, inputs


def _axis(space="features", width=4, *, indices=None):
    return CoordinateAxis(space, tuple(range(width)) if indices is None else indices)


def _region(model, paths, *, key="region", input_axes=None):
    modules = dict(model.named_modules())
    coordinates = []
    for index, path in enumerate(paths):
        layer = modules[path]
        width = layer.in_features if hasattr(layer, "in_features") else layer.d_model
        outputs = layer.out_features if hasattr(layer, "out_features") else layer.d_model
        incoming = _axis(width=width) if input_axes is None else input_axes[index]
        coordinates.append(ConsumerCoordinates(path, incoming, _axis(path + ":output", outputs)))
    return CompressionRegion(key, tuple(paths), tuple(coordinates))


def _sequential_requests(model, spec, *, paths=("0", "2", "4"), rank=1):
    return tuple(GroupRequest("step-" + path, _region(model, (path,), key="region-" + path),
                              (path,), spec, rank) for path in paths)


def _basis_requests(model, *, rank=1, region=None, paths=("0", "2")):
    region = _region(model, paths) if region is None else region
    return (GroupRequest("shared", region, tuple(paths), BasisSharing(), rank),)


def _prepare(model, inputs, requests, **kwargs):
    return prepare_grouped_workflow(model, inputs, tuple(requests), batch_size=5,
                                    graph_version="independent-g4-v1", **kwargs)


def _execute(model, inputs, workflow, *, capacity=1024 * 1024):
    return execute_grouped_workflow(model, inputs, workflow, store=StatisticsStore(capacity))


def _numpy_weight(layer):
    if hasattr(layer, "factor_matrix"):
        return layer.factor_matrix().detach().numpy().copy()
    if hasattr(layer, "right") and isinstance(layer.right, nn.Linear):
        return (layer.left.weight @ layer.right.weight).detach().numpy().copy()
    if hasattr(layer, "right") and isinstance(layer.right, nn.Parameter):
        return (layer.left @ layer.right).detach().numpy().copy()
    return layer.weight.detach().numpy().copy()


def _truncated(matrix, rank):
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    return (left[:, :rank] * singular[:rank]) @ right[:rank]


def _weighted_operator(weight, rows, rank):
    moment = rows.T @ rows / len(rows)
    return _weighted_by_moment(weight, moment, rank)


def _weighted_by_moment(weight, moment, rank):
    values, vectors = np.linalg.eigh(moment)
    cutoff = np.finfo(np.float64).eps * len(values) * max(0., values.max())
    keep = values > cutoff
    root = (vectors[:, keep] * np.sqrt(values[keep])) @ vectors[:, keep].T
    inverse = (vectors[:, keep] / np.sqrt(values[keep])) @ vectors[:, keep].T
    return _truncated(weight @ root, rank) @ inverse


def _oracle_fit(weight, current_rows, initial_rows, rank, spec):
    if type(spec) is DRONE:
        # Best rank-k output on the observed support, pulled back with X^+.
        x = current_rows.T
        return _truncated(weight @ x, rank) @ np.linalg.pinv(x)
    assert type(spec) is SVDLLMV1 and spec.ridge == 0
    initial = _weighted_operator(weight, initial_rows, rank)
    _, _, right = np.linalg.svd(initial, full_matrices=False)
    right = right[:rank]
    design = current_rows @ right.T
    target = current_rows @ weight.T
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    return coefficients.T @ right


def _dense_rows(model, inputs):
    rows, current = {}, inputs.detach().numpy().copy()
    for path, layer in model.named_children():
        if type(layer) is nn.Linear:
            rows[path] = current.copy()
            current = current @ _numpy_weight(layer).T + layer.bias.detach().numpy()
        elif type(layer) is nn.Tanh:
            current = np.tanh(current)
        elif type(layer) is nn.Sigmoid:
            current = 1. / (1. + np.exp(-current))
        else:
            raise AssertionError(type(layer))
    return rows


def _manual_sequential(model, inputs, spec, rank=1, *, frozen=False):
    initial_rows = _dense_rows(model, inputs)
    current, operators, observed = inputs.detach().numpy().copy(), {}, {}
    for path, layer in model.named_children():
        if type(layer) is nn.Linear:
            observed[path] = current.copy()
            data = initial_rows[path] if frozen else current
            operator = _oracle_fit(_numpy_weight(layer), data, initial_rows[path], rank, spec)
            operators[path] = operator
            current = current @ operator.T + layer.bias.detach().numpy()
        elif type(layer) is nn.Tanh:
            current = np.tanh(current)
        elif type(layer) is nn.Sigmoid:
            current = 1. / (1. + np.exp(-current))
        else:
            raise AssertionError(type(layer))
    return operators, current, observed


def _rng_state():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


def _assert_rng(before):
    python_state, numpy_state, torch_state = before
    assert random.getstate() == python_state
    actual = np.random.get_state()
    assert actual[0] == numpy_state[0] and actual[2:] == numpy_state[2:]
    np.testing.assert_array_equal(actual[1], numpy_state[1])
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def _owned_state(model, inputs):
    tensors = tuple(model.parameters()) + tuple(model.buffers())
    return SimpleNamespace(
        modules=tuple(model.modules()),
        tensors=tensors,
        values=tuple(tensor.detach().clone() for tensor in tensors),
        pointers=tuple(tensor.untyped_storage().data_ptr() for tensor in tensors),
        modes=tuple(module.training for module in model.modules()),
        gradients=tuple(parameter.grad for parameter in model.parameters()),
        gradient_values=tuple(None if parameter.grad is None else parameter.grad.clone()
                              for parameter in model.parameters()),
        data=inputs.clone(), data_pointer=inputs.untyped_storage().data_ptr(), rng=_rng_state())


def _assert_owned_state(model, inputs, before):
    assert tuple(model.modules()) == before.modules
    actual_tensors = tuple(model.parameters()) + tuple(model.buffers())
    assert all(left is right for left, right in zip(actual_tensors, before.tensors))
    assert len(actual_tensors) == len(before.tensors)
    for tensor, value, pointer in zip(actual_tensors, before.values, before.pointers):
        torch.testing.assert_close(tensor, value, rtol=0, atol=0)
        assert tensor.untyped_storage().data_ptr() == pointer
    assert tuple(module.training for module in model.modules()) == before.modes
    for parameter, gradient, value in zip(model.parameters(), before.gradients,
                                          before.gradient_values):
        assert parameter.grad is gradient
        if value is not None:
            torch.testing.assert_close(parameter.grad, value, rtol=0, atol=0)
    assert all(not module._forward_pre_hooks and not module._forward_hooks
               for module in model.modules())
    assert inputs.untyped_storage().data_ptr() == before.data_pointer
    torch.testing.assert_close(inputs, before.data, rtol=0, atol=0)
    _assert_rng(before.rng)


def _codes(error):
    return {violation.code for violation in error.violations}


def _facade(model):
    from fedcore.algorithm.low_rank.low_rank_opt import LowRankModel

    owner = LowRankModel.__new__(LowRankModel)
    owner._model_id_before = owner._model_id_after = None
    owner._model_before_cached = owner._model_after_cached = None
    owner._fedcore_id = "independent-g4-facade"
    registrations = []

    def register(**options):
        registrations.append(options)
        return options["stage"]

    owner._registry = SimpleNamespace(register_model=register)
    owner.trainer = SimpleNamespace(model=model)
    return owner, registrations


@pytest.mark.parametrize("spec", [DRONE(), SVDLLMV1()])
def test_current_student_matches_manual_loop_and_differs_from_frozen_batch(spec):
    from fedcore.algorithm.low_rank.method_execution import transform_method

    model, inputs = _case()
    workflow = _prepare(model, inputs, _sequential_requests(model, spec))
    actual = _execute(model, inputs, workflow)
    operators, expected, rows = _manual_sequential(model, inputs, spec)
    frozen_operators, frozen, _ = _manual_sequential(model, inputs, spec, frozen=True)
    assert np.linalg.norm(expected - frozen) > 1e-3
    assert np.linalg.norm(operators["2"] - frozen_operators["2"]) > 1e-3
    assert np.linalg.norm(rows["2"] - _dense_rows(model, inputs)["2"]) > 1e-3
    for path, operator in operators.items():
        np.testing.assert_allclose(_numpy_weight(dict(actual.model.named_modules())[path]),
                                   operator, rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(actual.model(inputs).detach().numpy(), expected,
                               rtol=2e-9, atol=2e-9)
    legacy = transform_method(model, inputs, spec, rank=1, target_paths=("0", "2", "4"),
                              batch_size=5, graph_version="independent-g4-v1")
    torch.testing.assert_close(actual.model(inputs), legacy.model(inputs),
                               rtol=2e-9, atol=2e-9)
    json.dumps(actual.evidence, allow_nan=False)


@pytest.mark.parametrize("mode", [SourceMode.CURRENT_STUDENT, SourceMode.FROZEN_SNAPSHOT])
def test_shared_basis_has_stacked_objective_physical_identity_and_reload(tmp_path, mode):
    from fedcore.tools.registry.checkpoint_manager import CheckpointManager

    model, inputs = _basis_case()
    requests = _basis_requests(model)
    workflow = _prepare(model, inputs, requests, source_mode=mode,
                         parameter_budget=35, tensor_byte_budget=35 * 8)
    result = _execute(model, inputs, workflow)
    initial_rows = _dense_rows(model, inputs)
    moment = sum(rows.T @ rows / len(rows) for path, rows in initial_rows.items()
                 if path in ("0", "2")) / 2
    stacked = np.concatenate([_numpy_weight(model[0]), _numpy_weight(model[2])])
    expected = _weighted_by_moment(stacked, moment, 1)
    np.testing.assert_allclose(_numpy_weight(result.model[0]), expected[:4], rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(_numpy_weight(result.model[2]), expected[4:], rtol=2e-9, atol=2e-9)
    assert result.model[0].right is result.model[2].right
    assert (result.model[0].right.untyped_storage().data_ptr()
            == result.model[2].right.untyped_storage().data_ptr())
    assert sum(parameter.numel() for parameter in result.model.parameters()) == 35
    assert sum(parameter.numel() * parameter.element_size()
               for parameter in result.model.parameters()) == 35 * 8
    manager = CheckpointManager(str(tmp_path), auto_cleanup=False)
    checkpoint = tmp_path / "grouped.pt"
    manager.save_to_file(manager.serialize_to_bytes(result.model), str(checkpoint))
    loaded = manager.load_from_file(str(checkpoint))
    assert loaded[0].right is loaded[2].right
    assert loaded[0].right is not result.model[0].right
    assert sum(parameter.numel() for parameter in loaded.parameters()) == 35
    torch.testing.assert_close(loaded(inputs), result.model(inputs), rtol=1e-11, atol=1e-11)
    # Equality of tensor values alone would miss a duplicated physical basis.
    with torch.no_grad():
        loaded[0].right[0, 0].add_(.125)
    assert loaded[2].right[0, 0] == loaded[0].right[0, 0]


def test_later_unrelated_region_accepts_existing_shared_basis():
    model, inputs = _basis_case()
    requests = _basis_requests(model) + _sequential_requests(model, DRONE(), paths=("4",))
    workflow = _prepare(model, inputs, requests)
    result = _execute(model, inputs, workflow)
    assert result.model[0].right is result.model[2].right
    initial_rows = _dense_rows(model, inputs)
    moment = sum(initial_rows[path].T @ initial_rows[path] / len(inputs)
                 for path in ("0", "2")) / 2
    stacked = _weighted_by_moment(np.concatenate([_numpy_weight(model[0]),
                                                 _numpy_weight(model[2])]), moment, 1)
    first = np.tanh(inputs.numpy() @ stacked[:4].T + model[0].bias.detach().numpy())
    second = first @ stacked[4:].T + model[2].bias.detach().numpy()
    current = 1. / (1. + np.exp(-second))
    expected_final = _oracle_fit(_numpy_weight(model[4]), current, initial_rows["4"], 1, DRONE())
    np.testing.assert_allclose(_numpy_weight(result.model[4]), expected_final,
                               rtol=2e-9, atol=2e-9)


@pytest.mark.parametrize("limit,value", [("parameter_budget", 34), ("tensor_byte_budget", 35 * 8 - 1)])
def test_one_below_unique_shared_budget_is_atomic(limit, value):
    model, inputs = _basis_case()
    store = StatisticsStore(1024 * 1024)
    before = _owned_state(model, inputs)
    with pytest.raises(GroupedWorkflowError):
        workflow = _prepare(model, inputs, _basis_requests(model), **{limit: value})
        execute_grouped_workflow(model, inputs, workflow, store=store)
    assert store.refs == () and store.used_bytes == 0
    _assert_owned_state(model, inputs, before)


@pytest.mark.parametrize("spec", [BasisSharing(), DRONE(), SVDLLMV1()])
def test_finite_huge_observations_report_numerical_failure_and_roll_back(spec):
    model, _ = _case()
    inputs = torch.full((13, 4), 1e200, dtype=torch.float64)
    assert bool(torch.isfinite(inputs).all())
    # The input domain is finite, but its uncentered second moment cannot be
    # represented in float64. This must not leak the collector's raw ValueError.
    requests = (_basis_requests(model) if type(spec) is BasisSharing
                else _sequential_requests(model, spec))
    workflow = _prepare(model, inputs, requests)
    store = StatisticsStore(1024 * 1024)
    before = _owned_state(model, inputs)
    with pytest.raises(GroupedWorkflowError) as caught:
        execute_grouped_workflow(model, inputs, workflow, store=store)
    assert "NumericalFailure" in _codes(caught.value)
    assert store.refs == () and store.used_bytes == 0
    _assert_owned_state(model, inputs, before)


def test_shared_budget_counts_nonpersistent_buffer_backing_tail():
    model, inputs = _basis_case()
    backing = torch.arange(41, dtype=torch.float64)
    model.register_buffer("scratch", backing[:1], persistent=False)
    assert model.scratch.numel() == 1 and model.scratch.untyped_storage().nbytes() == 41 * 8
    requests = _basis_requests(model)
    exact = _prepare(model, inputs, requests, parameter_budget=35 + 41,
                      tensor_byte_budget=(35 + 41) * 8)
    result = _execute(model, inputs, exact)
    tensors = tuple(result.model.parameters()) + tuple(result.model.buffers())
    storages = {tensor.untyped_storage()._cdata: tensor.untyped_storage() for tensor in tensors}
    assert sum(storage.nbytes() for storage in storages.values()) == (35 + 41) * 8
    assert result.model.scratch.numel() == 1
    assert result.model.scratch.untyped_storage().nbytes() == 41 * 8
    before = _owned_state(model, inputs)
    with pytest.raises(GroupedWorkflowError):
        insufficient = _prepare(model, inputs, requests, parameter_budget=35 + 40)
        _execute(model, inputs, insufficient)
    _assert_owned_state(model, inputs, before)


def _forbid_effects(monkeypatch):
    import fedcore.algorithm.low_rank.grouped_execution as grouped
    import fedcore.algorithm.low_rank.method_execution as methods

    def forbidden(*args, **kwargs):
        raise AssertionError("Static group rejection occurred after an execution effect")

    monkeypatch.setattr(grouped, "clone_preserving_storages", forbidden)
    monkeypatch.setattr(methods, "deepcopy", forbidden)
    monkeypatch.setattr(nn.Module, "register_forward_pre_hook", forbidden)
    monkeypatch.setattr(nn.Module, "register_forward_hook", forbidden)
    monkeypatch.setattr(torch.linalg, "svd", forbidden)


@pytest.mark.parametrize("mutation", ["partial", "wrong_basis", "permuted_coordinates", "missing_coordinates"])
def test_declared_group_closure_and_coordinate_identity_are_checked_before_copy(monkeypatch, mutation):
    model, inputs = _basis_case()
    region = _region(model, ("0", "2"))
    paths = ("0", "2")
    if mutation == "partial":
        paths = ("0",)
    elif mutation == "missing_coordinates":
        region = replace(region, coordinates=region.coordinates[:-1])
    else:
        coordinates = list(region.coordinates)
        wrong = (_axis("other_basis") if mutation == "wrong_basis"
                 else _axis(indices=(1, 0, 2, 3)))
        coordinates[1] = replace(coordinates[1], input=wrong)
        region = replace(region, coordinates=tuple(coordinates))
    _forbid_effects(monkeypatch)
    with pytest.raises(GroupedWorkflowError) as caught:
        _prepare(model, inputs, _basis_requests(model, region=region, paths=paths))
    assert caught.value.violations


@settings(max_examples=10, deadline=None)
@given(st.permutations((0, 1, 2, 3)).filter(lambda indices: tuple(indices) != (0, 1, 2, 3)))
def test_equal_dimension_different_coordinate_order_never_forms_a_shared_basis(indices):
    model, inputs = _basis_case()
    region = _region(model, ("0", "2"), input_axes=(_axis(), _axis(indices=tuple(indices))))
    with pytest.raises(GroupedWorkflowError):
        _prepare(model, inputs, _basis_requests(model, region=region))


def test_conflicting_write_regions_reject_before_copy(monkeypatch):
    model, inputs = _basis_case()
    requests = _basis_requests(model) + _sequential_requests(model, DRONE(), paths=("2",))
    _forbid_effects(monkeypatch)
    with pytest.raises(GroupedWorkflowError) as caught:
        _prepare(model, inputs, requests)
    assert caught.value.violations


@pytest.mark.parametrize("spec", [DRONE(), SVDLLMV1()])
def test_sequential_profiles_reject_frozen_statistics_mode_before_copy(monkeypatch, spec):
    model, inputs = _case()
    _forbid_effects(monkeypatch)
    with pytest.raises(GroupedWorkflowError):
        _prepare(model, inputs, _sequential_requests(model, spec), source_mode=SourceMode.FROZEN_SNAPSHOT)


@pytest.mark.parametrize("spec", [DRONE(), SVDLLMV1(), BasisSharing()])
def test_public_facade_publishes_only_independent_success_and_preserves_owned_state(spec):
    model, inputs = _case()
    model.train()
    model[0].eval()
    model.register_buffer("nonpersistent_observation", torch.tensor([3., 8.]), persistent=False)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, .25)
    requests = (_basis_requests(model) if type(spec) is BasisSharing
                else _sequential_requests(model, spec))
    owner, registrations = _facade(model)
    before = _owned_state(model, inputs)
    workflow = owner.prepare_compression_groups(model, inputs, requests,
        batch_size=5, graph_version="independent-g4-v1")
    _assert_owned_state(model, inputs, before)
    result = owner.compress_groups(model, inputs, workflow, store=StatisticsStore(1024 * 1024))
    _assert_owned_state(model, inputs, before)
    assert result.model is not model
    assert owner.model_before is model and owner.model_after is result.model
    assert owner.transform_result is result and owner.trainer.model is result.model
    assert [(item["stage"], item["model"]) for item in registrations] == [("before", model), ("after", result.model)]
    assert owner.trainer._fedcore_requires_optimizer_rebuild
    assert result.model._fedcore_requires_optimizer_rebuild
    assert tuple(module.training for module in result.model.modules())[:1] == (True,)
    assert not result.model[0].training
    assert all(actual.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
               for actual in result.model.parameters() for source in model.parameters())
    assert (result.model.nonpersistent_observation.untyped_storage().data_ptr()
            != model.nonpersistent_observation.untyped_storage().data_ptr())


class _RandomSequential(nn.Sequential):
    def forward(self, inputs):
        noise = .01 * (random.random() + np.random.random() + torch.rand(()))
        return super().forward(inputs + noise)


def test_random_forward_refusal_preserves_python_numpy_torch_rng():
    dense, inputs = _case()
    model = _RandomSequential(*dense.children())
    before = _owned_state(model, inputs)
    workflow = _prepare(model, inputs, _sequential_requests(model, DRONE()))
    with pytest.raises(GroupedWorkflowError) as caught:
        _execute(model, inputs, workflow)
    assert "UnsupportedObservationEffect" in _codes(caught.value)
    _assert_owned_state(model, inputs, before)


class _FailAfterFirstAccepted(nn.Sequential):
    def forward(self, inputs):
        from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear
        # The first observation succeeds on the dense copy; the second must see
        # the accepted replacement and fail before publishing the private copy.
        if isinstance(self[0], FactorizedLinear):
            raise RuntimeError("independent second-region observation failure")
        return super().forward(inputs)


def test_later_failure_rolls_back_store_and_facade_publication():
    dense, inputs = _case()
    model = _FailAfterFirstAccepted(*dense.children())
    model.register_buffer("scratch", torch.tensor([5.]), persistent=False)
    model.train()
    model[2].eval()
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, .125)
    owner, registrations = _facade(model)
    prior = nn.Identity()
    # Seed prior publication without invoking its unrelated disk registry.
    owner._model_before_cached = owner._model_after_cached = prior
    owner.transform_result = prior
    workflow = _prepare(model, inputs, _sequential_requests(model, DRONE()))
    store = StatisticsStore(1024 * 1024)
    before = _owned_state(model, inputs)
    with pytest.raises((GroupedWorkflowError, RuntimeError),
                       match="independent second-region observation failure"):
        owner.compress_groups(model, inputs, workflow, store=store)
    _assert_owned_state(model, inputs, before)
    assert store.refs == () and store.used_bytes == 0
    assert owner.model_before is prior and owner.model_after is prior
    assert owner.transform_result is prior and owner.trainer.model is model
    assert registrations == []
    assert not hasattr(owner.trainer, "_fedcore_requires_optimizer_rebuild")


def test_later_failure_preserves_preexisting_store_entries():
    dense, inputs = _case()
    model = _FailAfterFirstAccepted(*dense.children())
    store = StatisticsStore(1024 * 1024)
    prefix = _prepare(model, inputs, _sequential_requests(model, DRONE(), paths=("0",)))
    first = execute_grouped_workflow(model, inputs, prefix, store=store)
    refs, used = store.refs, store.used_bytes
    saved = {ref.key_digest: store.read(ref, expected=ref.provenance).tensors[0].clone()
             for ref in refs}
    workflow = _prepare(model, inputs, _sequential_requests(model, DRONE()))
    with pytest.raises((GroupedWorkflowError, RuntimeError),
                       match="independent second-region observation failure"):
        execute_grouped_workflow(model, inputs, workflow, store=store)
    assert store.refs == refs == first.statistic_refs and store.used_bytes == used
    for ref in refs:
        torch.testing.assert_close(store.read(ref, expected=ref.provenance).tensors[0],
                                   saved[ref.key_digest], rtol=0, atol=0)


@pytest.mark.parametrize("change", ["weight", "data", "nonpersistent_buffer"])
def test_same_shape_source_and_calibration_changes_reject_before_copy(monkeypatch, change):
    model, inputs = _case()
    model.register_buffer("scratch", torch.tensor([5.]), persistent=False)
    workflow = _prepare(model, inputs, _sequential_requests(model, DRONE()))
    with torch.no_grad():
        if change == "weight":
            model[0].weight[0, 0].add_(.125)
        elif change == "data":
            inputs[0, 0].add_(.125)
        else:
            model.scratch.add_(.125)
    before = _owned_state(model, inputs)
    _forbid_effects(monkeypatch)
    with pytest.raises(GroupedWorkflowError):
        _execute(model, inputs, workflow)
    _assert_owned_state(model, inputs, before)


def test_backing_storage_aliases_reject_before_copy(monkeypatch):
    model, inputs = _case()
    base = torch.arange(32, dtype=torch.float64)
    model[0].weight = nn.Parameter(base[:16].reshape(4, 4))
    model[2].weight = nn.Parameter(base[8:24].reshape(4, 4))
    assert model[0].weight is not model[2].weight
    assert (model[0].weight.untyped_storage().data_ptr()
            == model[2].weight.untyped_storage().data_ptr())
    _forbid_effects(monkeypatch)
    with pytest.raises(GroupedWorkflowError):
        _prepare(model, inputs, _sequential_requests(model, DRONE()))


class SourcePlanLifecycle(RuleBasedStateMachine):
    """A saved plan is valid exactly for the scalar source/data state it read."""

    def __init__(self):
        super().__init__()
        self.model, self.inputs = _case()
        self.weight_value = float(self.model[0].weight[0, 0])
        self.data_value = float(self.inputs[0, 0])
        self.prepared_state = None
        self.workflow = None
        self.prepare()

    @rule()
    def prepare(self):
        self.workflow = _prepare(self.model, self.inputs,
            _sequential_requests(self.model, DRONE(), paths=("0",)))
        self.prepared_state = self.weight_value, self.data_value

    @rule(delta=st.sampled_from((-.25, .125, .5)))
    def change_weight(self, delta):
        self.weight_value += delta
        with torch.no_grad():
            self.model[0].weight[0, 0].fill_(self.weight_value)

    @rule(delta=st.sampled_from((-.25, .125, .5)))
    def change_data(self, delta):
        self.data_value += delta
        self.inputs[0, 0].fill_(self.data_value)

    @rule()
    def execute_saved_plan(self):
        store = StatisticsStore(1024 * 1024)
        before = _owned_state(self.model, self.inputs)
        if self.prepared_state != (self.weight_value, self.data_value):
            with pytest.raises(GroupedWorkflowError):
                execute_grouped_workflow(self.model, self.inputs, self.workflow, store=store)
            assert store.used_bytes == 0 and store.refs == ()
        else:
            result = execute_grouped_workflow(self.model, self.inputs, self.workflow, store=store)
            expected = _oracle_fit(_numpy_weight(self.model[0]), self.inputs.numpy(),
                                   self.inputs.numpy(), 1, DRONE())
            np.testing.assert_allclose(_numpy_weight(result.model[0]), expected,
                                       rtol=2e-9, atol=2e-9)
            assert result.model is not self.model
        _assert_owned_state(self.model, self.inputs, before)

    @invariant()
    def original_parameters_and_calibration_follow_only_explicit_commands(self):
        assert float(self.model[0].weight[0, 0]) == self.weight_value
        assert float(self.inputs[0, 0]) == self.data_value
        assert type(self.model[0]) is nn.Linear
        assert not self.model[0]._forward_pre_hooks


TestSourcePlanLifecycle = SourcePlanLifecycle.TestCase
TestSourcePlanLifecycle.settings = settings(max_examples=8, stateful_step_count=8, deadline=None)


def test_two_disjoint_basis_groups_keep_both_physical_shared_parameters():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 4),
                              nn.Tanh(), nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 3)).double()
    inputs = torch.arange(52, dtype=torch.float64).reshape(13, 4) / 17 - .4
    requests = _basis_requests(model, paths=("0", "2"))
    second = _basis_requests(model, region=_region(model, ("4", "6"), key="second-region"),
                              paths=("4", "6"))[0]
    requests += (replace(second, key="second-shared"),)
    workflow = _prepare(model, inputs, requests)
    result = _execute(model, inputs, workflow)
    assert result.model[0].right is result.model[2].right
    assert result.model[4].right is result.model[6].right
    assert result.model[0].right is not result.model[4].right
    assert sum(parameter.numel() for parameter in result.model.parameters()) == 38


def _reseal(workflow):
    return replace(workflow, fingerprint=grouped_fingerprint(workflow))


@pytest.mark.parametrize("mutation", ["coordinates", "adapter", "reads", "writes", "rank", "method", "objective", "write_paths"])
def test_resealed_semantic_forgery_is_rejected_before_execution(monkeypatch, mutation):
    model, inputs = _case()
    workflow = _prepare(model, inputs, _sequential_requests(model, DRONE()))
    groups = list(workflow.groups)
    first = groups[0]
    if mutation == "coordinates":
        bindings = first.request.region.coordinates
        wrong = replace(bindings[0], input=_axis("false-input", width=3))
        request = replace(first.request, region=replace(first.request.region, coordinates=(wrong,)))
        first = replace(first, request=request)
    elif mutation == "rank":
        first = replace(first, request=replace(first.request, rank=2))
    elif mutation == "method":
        first = replace(first, request=replace(first.request, method_spec=SVDLLMV1()))
    elif mutation == "objective":
        first = replace(first, transform=replace(first.transform,
            request=replace(first.transform.request, objective="forged_objective")))
    elif mutation == "adapter":
        first = replace(first, adapter="a_forged_adapter")
    elif mutation == "write_paths":
        first = replace(first, write_paths=("2",))
    else:
        first = replace(first, **{mutation: ()})
    groups[0] = first
    changed = _reseal(replace(workflow, groups=tuple(groups)))
    assert changed.fingerprint != workflow.fingerprint
    _forbid_effects(monkeypatch)
    with pytest.raises(GroupedWorkflowError) as caught:
        _execute(model, inputs, changed)
    assert caught.value.violations


def test_workflow_contains_frozen_metadata_and_no_runtime_resources():
    model, inputs = _case()
    workflow = _prepare(model, inputs, _sequential_requests(model, SVDLLMV1()))

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

    check(workflow)


@pytest.mark.parametrize("spec,copies", [(DRONE(), 1), (SVDLLMV1(), 2)])
def test_statistics_store_exact_capacity_and_one_byte_below_are_transactional(spec, copies):
    model, inputs = _case()
    workflow = _prepare(model, inputs, _sequential_requests(model, spec))
    required = 3 * 4 * 4 * 8 * copies
    exact = StatisticsStore(required)
    result = execute_grouped_workflow(model, inputs, workflow, store=exact)
    assert exact.used_bytes == required and len(result.statistic_refs) == 3 * copies
    before = _owned_state(model, inputs)
    insufficient = StatisticsStore(required - 1)
    with pytest.raises(ArtifactStoreError):
        execute_grouped_workflow(model, inputs, workflow, store=insufficient)
    assert insufficient.refs == () and insufficient.used_bytes == 0
    _assert_owned_state(model, inputs, before)


@pytest.mark.parametrize("spec", [DRONE(), SVDLLMV1()])
def test_each_accepted_version_stores_its_actual_independently_computed_moment(spec):
    model, inputs = _case()
    workflow = _prepare(model, inputs, _sequential_requests(model, spec))
    store = StatisticsStore(1024 * 1024)
    result = execute_grouped_workflow(model, inputs, workflow, store=store)
    _, _, current_rows = _manual_sequential(model, inputs, spec)
    initial_rows = _dense_rows(model, inputs)
    current_keys = []
    for ref in result.statistic_refs:
        key = ref.provenance
        initial = key.graph_version.endswith(":initialization=M0")
        rows = (initial_rows if initial else current_rows)[key.path]
        expected = rows.T @ rows / len(rows)
        snapshot = store.read(ref, expected=key)
        np.testing.assert_allclose(snapshot.tensors[0].numpy(), expected, rtol=3e-9, atol=3e-9)
        assert key.source_fingerprint == workflow.source.fingerprint
        assert key.calibration_fingerprint == workflow.calibration.fingerprint
        if not initial:
            step = ("0", "2", "4").index(key.path)
            assert key.graph_version.endswith(f":accepted={step}")
            assert workflow.fingerprint in key.graph_version
            current_keys.append(key)
    assert len(current_keys) == 3
    assert len({key.model_fingerprint for key in current_keys}) == 3
    stages = [(event["group"], event["stage"]) for event in result.evidence["trace"]]
    assert stages == [("step-" + path, stage) for path in ("0", "2", "4")
                      for stage in ("observe", "solve", "apply", "advance")]
    advances = [event for event in result.evidence["trace"] if event["stage"] == "advance"]
    for index, event in enumerate(advances, start=1):
        assert event["accepted"] == index
        assert event["source"]["graph_version"].endswith(f":accepted={index}")
        assert event["predecessor"]["fingerprint"] != event["source"]["fingerprint"]
    # A real ref cannot be relabelled as statistics of the next accepted state.
    with pytest.raises(ArtifactStoreError) as caught:
        store.read(result.statistic_refs[0], expected=current_keys[-1])
    assert "StaleArtifact" in _codes(caught.value)


@pytest.mark.parametrize("spec", [DRONE(), SVDLLMV1()])
def test_statistic_cache_hit_still_observes_ephemeral_current_rows(monkeypatch, spec):
    import fedcore.algorithm.low_rank.grouped_execution as grouped

    model, inputs = _case()
    workflow = _prepare(model, inputs, _sequential_requests(model, spec))
    store = StatisticsStore(1024 * 1024)
    first = execute_grouped_workflow(model, inputs, workflow, store=store)
    original, calls = grouped._collect_rows, []

    def record(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(grouped, "_collect_rows", record)
    second = execute_grouped_workflow(model, inputs, workflow, store=store)
    assert second.statistic_refs == first.statistic_refs
    assert len(calls) == 3
    observations = [event for event in second.evidence["trace"] if event["stage"] == "observe"]
    assert len(observations) == 3
    assert all(all(event["statistic_cache_hits"]) and event["forwards"] > 0
               and event["rows_cached"] is False for event in observations)
    torch.testing.assert_close(first.model(inputs), second.model(inputs), rtol=1e-11, atol=1e-11)


def test_schedule_is_finite_and_store_has_one_active_owner_before_copy(monkeypatch):
    model, inputs = _case()
    requests = _sequential_requests(model, DRONE())
    with pytest.raises(GroupedWorkflowError) as caught:
        _prepare(model, inputs, (requests[0],) * 65)
    assert "InvalidSchedule" in _codes(caught.value)
    workflow = _prepare(model, inputs, requests)
    store = StatisticsStore(1024 * 1024)
    _forbid_effects(monkeypatch)
    with store._transaction():
        with pytest.raises(ArtifactStoreError) as caught:
            execute_grouped_workflow(model, inputs, workflow, store=store)
        assert "StoreBusy" in _codes(caught.value)
    assert store.refs == () and store.used_bytes == 0


def _attention_case():
    from fedcore.algorithm.low_rank.coupled_layers import ReferenceAttention
    from fedcore.algorithm.low_rank.coupled_specs import FlatQueryKey

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(103)
        model = nn.Sequential(ReferenceAttention(4, 4, 2, 4, dtype=torch.float64))
    generator = torch.Generator().manual_seed(51)
    inputs = torch.randn((3, 5, 4), generator=generator, dtype=torch.float64)
    axis = _axis("residual")
    region = CompressionRegion("attention-region", ("0",), (ConsumerCoordinates("0", axis, axis),))
    request = GroupRequest("attention-qk", region, ("0",), FlatQueryKey(coordinates=(2, 3)), 2)
    return model, inputs, (request,)


def _attention_oracle(layer, inputs):
    """Original-frequency pair selection, GQA expansion and causal attention."""
    x = inputs.detach().numpy()
    b, t, _ = x.shape

    def projection(module):
        return x @ module.weight.detach().numpy().T + module.bias.detach().numpy()

    q = projection(layer.q_proj).reshape(b, t, 4, 4).transpose(0, 2, 1, 3)[..., [2, 3]]
    k = projection(layer.k_proj).reshape(b, t, 2, 4).transpose(0, 2, 1, 3)[..., [2, 3]]
    angle = np.arange(t) * layer.rope_base ** (-2. / 4)
    cosine, sine = np.cos(angle)[None, None, :], np.sin(angle)[None, None, :]

    def rotate(states):
        even, odd = states[..., 0].copy(), states[..., 1].copy()
        return np.stack((even * cosine - odd * sine, even * sine + odd * cosine), axis=-1)

    q, k = rotate(q), rotate(k)
    mapping = layer.head_map.numpy()
    logits = q @ k[:, mapping].transpose(0, 1, 3, 2) * layer.logits_scale
    causal = np.arange(t)[None, :] <= np.arange(t)[:, None]
    masked = np.where(causal, logits, -np.inf)
    probabilities = np.exp(masked - masked.max(-1, keepdims=True))
    probabilities /= probabilities.sum(-1, keepdims=True)
    values = projection(layer.v_proj).reshape(b, t, 2, 4).transpose(0, 2, 1, 3)
    context = (probabilities @ values[:, mapping]).transpose(0, 2, 1, 3).reshape(b, t, 16)
    output = context @ layer.o_proj.weight.detach().numpy().T + layer.o_proj.bias.detach().numpy()
    return logits, output


def test_reference_adapter_preserves_actual_rope_gqa_and_external_residual_axis(tmp_path):
    from fedcore.tools.registry.checkpoint_manager import CheckpointManager

    model, inputs, requests = _attention_case()
    workflow = _prepare(model, inputs, requests)
    result = _execute(model, inputs, workflow)
    layer = result.model[0]
    assert layer.d_model == model[0].d_model == 4
    assert layer.qk_dim == 2 and layer.head_dim == model[0].head_dim == 4
    assert layer.logits_scale == model[0].logits_scale
    assert layer.head_map.tolist() == model[0].head_map.tolist() == [0, 0, 1, 1]
    assert layer.group_coordinates.tolist() == [[2, 3], [2, 3]]
    expected_logits, expected_output = _attention_oracle(model[0], inputs)
    np.testing.assert_allclose(layer.attention(inputs).logits.detach().numpy(), expected_logits,
                               rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(result.model(inputs).detach().numpy(), expected_output,
                               rtol=2e-10, atol=2e-10)
    manager = CheckpointManager(str(tmp_path), auto_cleanup=False)
    path = tmp_path / "attention.pt"
    manager.save_to_file(manager.serialize_to_bytes(result.model), str(path))
    loaded = manager.load_from_file(str(path))
    assert loaded[0].group_coordinates.tolist() == [[2, 3], [2, 3]]
    assert loaded[0].head_map.tolist() == [0, 0, 1, 1]
    torch.testing.assert_close(loaded(inputs), result.model(inputs), rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("mutation", ["head_map", "rope_coordinates", "valid_foreign_basis", "rope_base"])
def test_resealed_reference_atlas_forgery_is_checked_against_actual_model_before_copy(monkeypatch, mutation):
    model, inputs, requests = _attention_case()
    workflow = _prepare(model, inputs, requests)
    group = workflow.groups[0]
    actual = group.reference_coordinates[0]
    if mutation == "head_map":
        changed = replace(actual, head_map=(0, 1, 0, 1))
    elif mutation == "rope_coordinates":
        changed = replace(actual, group_coordinates=((0, 2, 1, 3),) * 2)
    elif mutation == "valid_foreign_basis":
        changed = replace(actual, group_coordinates=((2, 3, 0, 1),) * 2)
    else:
        changed = replace(actual, rope_base=20000.)
    forged = _reseal(replace(workflow, groups=(replace(group, reference_coordinates=(changed,)),)))
    _forbid_effects(monkeypatch)
    with pytest.raises(GroupedWorkflowError):
        _execute(model, inputs, forged)


@pytest.mark.parametrize("mutation", ["broken_rope_pair", "bad_gqa_map"])
def test_invalid_actual_reference_configuration_refuses_before_copy(monkeypatch, mutation):
    model, inputs, requests = _attention_case()
    if mutation == "broken_rope_pair":
        model[0].group_coordinates[0].copy_(torch.tensor([0, 2, 1, 3]))
    else:
        model[0].head_map.copy_(torch.tensor([0, 0, 0, 1]))
    _forbid_effects(monkeypatch)
    with pytest.raises(GroupedWorkflowError):
        _prepare(model, inputs, requests)
