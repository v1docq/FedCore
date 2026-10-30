"""Independent score, storage and finite-allocation witnesses for G2b."""
from dataclasses import asdict, fields, is_dataclass, replace
from fractions import Fraction
from itertools import product
import json
import random

from hypothesis import given, settings, strategies as st
import numpy as np
import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.candidate_storage import (
    CandidateStorageError, capture_inventory, clone_preserving_storages,
    inventory_cost, inventory_fingerprint,
)
from fedcore.algorithm.low_rank.artifacts import StatisticsStore
from fedcore.algorithm.low_rank.graph_execution import (
    CompressionGraphError, execute_fixed_graph, prepare_fixed_graph,
)
from fedcore.algorithm.low_rank.method_specs import ASVD
from fedcore.algorithm.low_rank.allocation import (
    Allocation, BudgetInfeasible, CandidateOption, CandidateTable, ScoreObjective,
    StorageCost, allocate,
)
from fedcore.algorithm.low_rank.candidate_graph import (
    CandidateGraphError, CandidateRecipe, allocate_candidate_catalog,
    apply_candidate_allocation, prepare_candidate_catalog, verify_candidate_export,
)
from fedcore.algorithm.low_rank.plans import MetricPolicy


PEAK = 12 * 1024**3


def _codes(error):
    return {violation.code for violation in error.violations}


def _cartesian_oracle(tables, fixed, budget, unit):
    """Enumerate choices without either production allocator/cost helper."""
    ordered = sorted(tables, key=lambda table: table.table_id)
    feasible, minimum = [], None
    for choices in product(*(table.options for table in ordered)):
        physical = {}
        for record in (*fixed, *(record for option in choices for record in option.storages)):
            signature = (record.numel, record.bytes_per_element)
            assert physical.setdefault(record.storage_id, signature) == signature
        cost = sum(elements * (width if unit == "bytes" else 1)
                   for elements, width in physical.values())
        minimum = cost if minimum is None else min(minimum, cost)
        if cost <= budget:
            ids = tuple(option.option_id for option in choices)
            score = sum((Fraction(option.score) for option in choices), Fraction())
            feasible.append(((score, cost, ids), tuple((table.table_id, option.option_id)
                             for table, option in zip(ordered, choices))))
    return (min(feasible) if feasible else None), minimum


@st.composite
def _finite_tables(draw):
    count = draw(st.integers(1, 3))
    sizes = draw(st.lists(st.integers(1, 7), min_size=6, max_size=6))
    widths = draw(st.lists(st.sampled_from([1, 2, 4, 8]), min_size=6, max_size=6))
    pool = tuple(StorageCost(f"storage:{index}", size, width, category)
                 for index, (size, width, category) in enumerate(zip(sizes, widths,
                     ("bias", "buffer_map", "factor", "scale", "residual", "shared_basis"))))
    tables = []
    for index in range(count):
        options = []
        for choice in range(draw(st.integers(2, 3))):
            indices = draw(st.sets(st.integers(0, 5), max_size=4))
            score = draw(st.integers(-20, 20)) / 2
            options.append(CandidateOption(f"option:{choice}", (choice + 1,), score,
                                           tuple(pool[item] for item in sorted(indices))))
        tables.append(CandidateTable(f"table:{index}", tuple(options), (f"layer:{index}",)))
    return tuple(tables), pool[:1], draw(st.integers(0, 160))


@settings(max_examples=35, deadline=None)
@given(case=_finite_tables())
@pytest.mark.parametrize("unit", ["parameters", "bytes"])
def test_allocator_matches_independent_cartesian_model_with_shared_payloads(case, unit):
    tables, fixed, budget = case
    expected, minimum = _cartesian_oracle(tables, fixed, budget, unit)
    objective = ScoreObjective("independent_local_reconstruction")
    actual = allocate(tables, budget, objective=objective, fixed_storages=fixed, unit=unit)
    if expected is None:
        assert isinstance(actual, BudgetInfeasible)
        assert actual.minimum_cost == minimum and actual.unit == unit
    else:
        assert isinstance(actual, Allocation)
        (score, cost, _ids), selected = expected
        assert actual.total_cost == cost <= budget
        assert tuple((item.table_id, item.option.option_id) for item in actual.selected) == selected
        assert Fraction.from_float(actual.objective_score) == score
        reversed_tables = tuple(replace(table, options=tuple(reversed(table.options)))
                                for table in reversed(tables))
        reordered = allocate(reversed_tables, budget, objective=objective, fixed_storages=fixed, unit=unit)
        assert reordered.selected == actual.selected


class _Views(nn.Module):
    def __init__(self):
        super().__init__()
        backing = torch.arange(40, dtype=torch.float64)
        self.weight = nn.Parameter(backing[5:17:2].reshape(2, 3))
        self.register_buffer("window", backing[3:8].requires_grad_(), persistent=False)
        self.register_buffer("window_alias", self.window, persistent=False)
        self.register_buffer("map", torch.arange(4, dtype=torch.int64))


def test_inventory_counts_full_backing_and_all_buffers_in_both_units():
    model = _Views()
    inventory = capture_inventory(model)
    # Six trainable elements view a 40-element backing, plus four int64 map cells.
    assert sum(record.parameter_numel for record in inventory) == 6
    assert inventory_cost(inventory, "parameters") == 44
    assert inventory_cost(inventory, "bytes") == 352
    assert {view.path for record in inventory for view in record.views} == {
        "weight", "window", "window_alias", "map",
    }
    assert next(view for record in inventory for view in record.views
                if view.path == "window").persistent is False
    # A hidden cell outside every registered view remains physical payload.
    initial = inventory_fingerprint(inventory)
    raw = torch.empty(0, dtype=torch.float64).set_(model.weight.untyped_storage(), 0, (40,), (1,))
    raw[-1] = -123
    assert inventory_fingerprint(capture_inventory(model)) != initial


def test_owned_clone_preserves_views_object_aliases_and_buffer_grad_flags():
    source = _Views()
    copied = clone_preserving_storages(source, max_peak_bytes=PEAK)
    assert capture_inventory(copied) == capture_inventory(source)
    assert copied.window is copied.window_alias
    assert copied.window.requires_grad and copied.weight.requires_grad
    assert copied.weight.stride() == (6, 2) and copied.weight.storage_offset() == 5
    assert copied.window.storage_offset() == 3
    assert copied.weight.untyped_storage()._cdata == copied.window.untyped_storage()._cdata
    assert copied.weight.untyped_storage()._cdata != source.weight.untyped_storage()._cdata
    with torch.no_grad():
        copied.window[2] = -17
    assert copied.weight[0, 0].item() == -17
    assert source.weight[0, 0].item() == 5


@pytest.mark.parametrize("container", ["nested_value", "dict_key"])
def test_python_container_alias_is_preserved_or_explicitly_outside_copy_capability(container):
    model = nn.Sequential(nn.Linear(2, 2, bias=False, dtype=torch.float64))
    model.weight_links = ({"nested": [model[0].weight]} if container == "nested_value"
                          else {model[0].weight: "tensor-key"})
    try:
        work = clone_preserving_storages(model, max_peak_bytes=PEAK)
    except CandidateStorageError as error:
        assert _codes(error) & {"UnsupportedStorage", "UnsupportedTensorReference"}
    else:
        # A copied hidden Parameter must not diverge from the registered slot.
        linked = work.weight_links["nested"][0] if container == "nested_value" else next(iter(work.weight_links))
        assert linked is work[0].weight
        with torch.no_grad():
            work[0].weight.add_(1)
        torch.testing.assert_close(linked, work[0].weight, rtol=0, atol=0)


@pytest.mark.parametrize("change", ["value", "alias", "grad", "persistence"])
def test_nonpersistent_buffer_changes_are_part_of_cost_and_source_context(change):
    model = _Views()
    original = inventory_fingerprint(capture_inventory(model))
    original_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with torch.no_grad():
        if change == "value":
            # Cell 3 lies outside the Parameter view, so state_dict is unchanged.
            model.window[0] += 1
        elif change == "alias":
            model.window_alias = model.window.clone()
        elif change == "grad":
            model.window.requires_grad_(False)
        else:
            model._non_persistent_buffers_set.remove("window")
    assert inventory_fingerprint(capture_inventory(model)) != original
    if change != "persistence":
        assert model.state_dict().keys() == original_state.keys()
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, original_state[name], rtol=0, atol=0)


@pytest.mark.parametrize("mixed", [False, True])
def test_element_budget_refuses_ambiguous_backing_but_bytes_remain_exact(mixed):
    model = nn.Module()
    backing = torch.zeros(10 if not mixed else 12, dtype=torch.uint8)
    floats = torch.empty(0, dtype=torch.float32).set_(backing.untyped_storage(), 0, (2,), (1,))
    model.register_buffer("float_view", floats)
    if mixed:
        model.register_buffer("byte_view", backing)
    inventory = capture_inventory(model)
    assert inventory_cost(inventory, "bytes") == backing.numel()
    with pytest.raises(CandidateStorageError) as caught:
        inventory_cost(inventory, "parameters")
    assert "AmbiguousElementCost" in _codes(caught.value)


def test_storage_copy_preflight_refuses_before_python_or_tensor_copy(monkeypatch):
    import fedcore.algorithm.low_rank.candidate_storage as storage_module
    model = _Views()
    monkeypatch.setattr(storage_module, "deepcopy", lambda *_args, **_kwargs: pytest.fail("deepcopy before admission"))
    with pytest.raises(CandidateStorageError) as caught:
        clone_preserving_storages(model, max_peak_bytes=1)
    assert "ResourceLimitExceeded" in _codes(caught.value)


@pytest.mark.parametrize("lazy", ["conjugate", "negative"])
def test_lazy_tensor_views_have_an_explicit_storage_capability_refusal(lazy):
    model = nn.Module()
    tensor = (torch.tensor([1 + 2j]).conj() if lazy == "conjugate"
              else torch.tensor([1., 2.])._neg_view())
    model.register_buffer("view", tensor)
    with pytest.raises(CandidateStorageError) as caught:
        clone_preserving_storages(model, max_peak_bytes=PEAK)
    assert "UnsupportedStorage" in _codes(caught.value)


class _NonpersistentEffect(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(3, 3, bias=False, dtype=torch.float64))
        self.register_buffer("observations", torch.zeros(1), persistent=False)

    def forward(self, value):
        self.observations.add_(1)
        return super().forward(value)


@pytest.mark.parametrize("recipe", ["weighted_svd", "asvd"])
def test_nonpersistent_observation_effect_refuses_without_publishing_statistics(recipe):
    model = _NonpersistentEffect()
    calibration = torch.tensor([[1., 2., 3.], [-1., 0., 2.]], dtype=torch.float64)
    original = capture_inventory(model)
    store = StatisticsStore(1024)
    graph = prepare_fixed_graph(model, calibration, recipe=recipe, rank=1,
        method_spec=ASVD() if recipe == "asvd" else None,
        target_paths=("0",), graph_version="independent-nonpersistent-effect",
        executor="staged_fixed_recipe_v1")
    with pytest.raises(CompressionGraphError) as caught:
        execute_fixed_graph(model, calibration, graph, store=store)
    assert "UnsupportedObservationEffect" in _codes(caught.value)
    assert store.used_bytes == 0 and store.refs == ()
    assert capture_inventory(model) == original


def _diagonal(values):
    model = nn.Sequential(nn.Linear(len(values), len(values), bias=False, dtype=torch.float64))
    with torch.no_grad():
        model[0].weight.copy_(torch.diag(torch.tensor(values, dtype=torch.float64)))
    return model


def _local_error(source, replacement, data):
    with torch.no_grad():
        return (source(data) - replacement(data)).square().sum().item() / len(data)


def test_asvd_and_weighted_alternatives_share_one_actual_output_score():
    model = _diagonal([4., 1.])
    data = torch.tensor([[1., 3.], [1., -3.], [-1., 3.], [-1., -3.]], dtype=torch.float64)
    store = StatisticsStore(4096)
    catalog = prepare_candidate_catalog(model, data, alternatives={"0": (
        CandidateRecipe("weighted_svd", 1), CandidateRecipe("asvd", 1, method_spec=ASVD(alpha=.5)),
    )}, store=store)
    assert len(catalog.tables) == 1 and len(catalog.tables[0].candidates) == 2
    assert catalog.context.regularization == "none"
    assert catalog.context.normalization == "weight_sum"
    assert catalog.context.semantics == "frozen_snapshot_surrogate"
    candidates = catalog.tables[0].candidates
    assert candidates[0].score_ref == candidates[1].score_ref
    # Both operators are diag(4,0). ASVD's native diagonal objective is 3,
    # whereas the shared output-error objective on these rows is exactly 9.
    for candidate in candidates:
        assert candidate.score == pytest.approx(9., rel=1e-12, abs=1e-12)
        assert candidate.metric_ref != candidate.score_ref
    plan = allocate_candidate_catalog(catalog, 32, unit="bytes")
    applied = apply_candidate_allocation(model, data, plan, store=store)
    assert _local_error(model, applied.model, data) == pytest.approx(9., abs=1e-10)
    assert applied.evidence["local_score_is_whole_model_quality"] is False
    assert applied.evidence["whole_model_evaluation"]["status"] == "required_not_run"
    json.dumps(applied.evidence, allow_nan=False)


def test_alpha_reuses_abs_statistics_but_metric_identity_is_scoped_to_its_graph():
    model = _diagonal([4., 1.])
    data = torch.tensor([[1., 3.], [1., -3.], [-1., 3.], [-1., -3.]], dtype=torch.float64)
    store = StatisticsStore(4096)
    catalog = prepare_candidate_catalog(model, data, alternatives={"0": (
        CandidateRecipe("asvd", 1, method_spec=ASVD(alpha=.5)),
        CandidateRecipe("asvd", 1, method_spec=ASVD(alpha=2.)),
    )}, store=store)
    first, second = catalog.tables[0].candidates
    assert first.statistic_ref == second.statistic_ref
    assert (first.graph.fingerprint, first.metric_ref) != (second.graph.fingerprint, second.metric_ref)
    # alpha=.5 retains the first coordinate; alpha=2 retains the second one.
    assert first.score == pytest.approx(9., rel=1e-12)
    assert second.score == pytest.approx(16., rel=1e-12)


def test_common_score_matches_direct_outputs_with_correlated_data_and_regularized_solver():
    model = nn.Sequential(nn.Linear(3, 2, dtype=torch.float64))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[3., 1., 2.], [-2., 1., 4.]], dtype=torch.float64))
        model[0].bias.copy_(torch.tensor([.25, -.75], dtype=torch.float64))
    data = torch.tensor([[1., 2., 3.], [-1., 2., 0.], [2., 3., 1.], [0., 1., 2.]], dtype=torch.float64)
    store = StatisticsStore(4096)
    catalog = prepare_candidate_catalog(model, data, alternatives={"0": (
        CandidateRecipe("weighted_svd", 1, policy=MetricPolicy(ridge=3.)),
        CandidateRecipe("asvd", 1),
    )}, store=store)
    table = catalog.tables[0]
    snapshot = store.read(table.score_ref, expected=table.score_ref.provenance)
    torch.testing.assert_close(snapshot.tensors[0], data.T @ data / len(data), rtol=1e-13, atol=1e-13)
    assert snapshot.tensors[0][0, 1].item() != 0
    for candidate in table.candidates:
        replay = execute_fixed_graph(model, data, candidate.graph, store=store)
        assert candidate.score == pytest.approx(_local_error(model, replay.model, data), rel=1e-11, abs=1e-11)


def test_grouped_conv_common_score_normalizes_each_group_over_real_spatial_patches():
    model = nn.Sequential(nn.Conv1d(4, 4, 1, groups=2, dtype=torch.float64))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[[3.], [1.]], [[-1.], [2.]],
                                          [[2.], [1.]], [[1.], [-4.]]], dtype=torch.float64))
        model[0].bias.copy_(torch.tensor([.1, .2, .3, .4], dtype=torch.float64))
    data = torch.arange(24, dtype=torch.float64).reshape(2, 4, 3) / 10 - 1
    store = StatisticsStore(4096)
    catalog = prepare_candidate_catalog(model, data, alternatives={"0": (CandidateRecipe("weighted_svd", 1),)}, store=store)
    table, candidate = catalog.tables[0], catalog.tables[0].candidates[0]
    snapshot = store.read(table.score_ref, expected=table.score_ref.provenance)
    assert len(snapshot.tensors) == 2
    assert all(passport.count == 6 and passport.weight_sum == 6. for passport in snapshot.passports)
    for group, moment in enumerate(snapshot.tensors):
        rows = data[:, group * 2:(group + 1) * 2, :].permute(0, 2, 1).reshape(-1, 2)
        torch.testing.assert_close(moment, rows.T @ rows / 6, rtol=1e-13, atol=1e-13)
    replay = execute_fixed_graph(model, data, candidate.graph, store=store)
    with torch.no_grad():
        error_per_patch = (model(data) - replay.model(data)).square().sum().item() / 6
    assert candidate.score == pytest.approx(error_per_patch, rel=1e-11, abs=1e-11)


def test_weighted_rank_family_is_resolved_in_metric_space_not_canonical_prefix():
    model = _diagonal([10., 1., .1])
    data = torch.tensor([[.1, 10., 1.], [.1, -10., -1.],
                         [-.1, 10., -1.], [-.1, -10., 1.]], dtype=torch.float64)
    store = StatisticsStore(4096)
    catalog = prepare_candidate_catalog(model, data, alternatives={"0": (
        CandidateRecipe("weighted_svd", 1), CandidateRecipe("weighted_svd", 2),
    )}, store=store)
    scores = {candidate.recipe.rank: candidate.score for candidate in catalog.tables[0].candidates}
    assert scores[1] == pytest.approx(1.01, rel=1e-11)
    assert scores[2] == pytest.approx(.01, rel=1e-11)
    plan = allocate_candidate_catalog(catalog, 48, unit="bytes")
    actual = apply_candidate_allocation(model, data, plan, store=store)
    with torch.no_grad():
        actual_matrix = actual.model(torch.eye(3, dtype=torch.float64)).T
    torch.testing.assert_close(actual_matrix, torch.diag(torch.tensor([0., 1., 0.], dtype=torch.float64)),
                               rtol=1e-11, atol=1e-11)
    assert _local_error(model, actual.model, data) == pytest.approx(1.01, rel=1e-11)
    canonical_prefix = _diagonal([10., 0., 0.])
    assert _local_error(model, canonical_prefix, data) == pytest.approx(100.01, rel=1e-11)


@pytest.mark.parametrize("representation,elements", [("one_layer", 16), ("two_layers", 11), ("three_layers", 12)])
def test_cost_follows_actual_export_representation_including_bias_and_singular_scales(representation, elements):
    model = nn.Sequential(nn.Linear(3, 4, dtype=torch.float64))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[3., 0., 0.], [0., 2., 0.], [0., 0., 1.], [1., 1., 1.]], dtype=torch.float64))
        model[0].bias.copy_(torch.tensor([.1, .2, .3, .4], dtype=torch.float64))
    data, store = _data3(), StatisticsStore(4096)
    catalog = prepare_candidate_catalog(model, data, alternatives={"0": (
        CandidateRecipe("weighted_svd", 1, representation=representation),
    )}, store=store)
    plan = allocate_candidate_catalog(catalog, elements, unit="parameters")
    result = apply_candidate_allocation(model, data, plan, store=store)
    assert _physical_cost(result.model, "parameters") == elements
    exported = verify_candidate_export(result)
    assert exported.evidence["actual_cost"] == elements
    assert exported.evidence["tensor_bytes"] == elements * 8
    assert exported.model[0].bias.numel() == 4
    torch.testing.assert_close(exported.model[0].bias, model[0].bias, rtol=0, atol=0)


class _TwoTargets(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(3, 3, dtype=torch.float64)
        self.b = nn.Linear(3, 2, dtype=torch.float64)
        self.register_buffer("map", torch.arange(5, dtype=torch.int16))
        self.register_buffer("map_alias", self.map, persistent=False)
        with torch.no_grad():
            self.a.weight.copy_(torch.tensor([[3., 0., 0.], [0., 2., 0.], [0., 0., 1.]], dtype=torch.float64))
            self.a.bias.copy_(torch.tensor([.1, .2, .3], dtype=torch.float64))
            self.b.weight.copy_(torch.tensor([[1., 2., 0.], [0., 1., 3.]], dtype=torch.float64))
            self.b.bias.copy_(torch.tensor([.4, .5], dtype=torch.float64))

    def forward(self, value):
        return self.b(torch.relu(self.a(value)))


def _data3():
    return torch.tensor([[1., 2., 3.], [-1., 2., 1.], [2., -3., 1.], [-2., -1., 3.]], dtype=torch.float64)


def _physical_cost(model, unit):
    # Count the actual registered backing stores directly, without inventory helpers.
    seen = {}
    for module in model.modules():
        for tensor in (*module._parameters.values(), *module._buffers.values()):
            if tensor is not None:
                storage = tensor.untyped_storage()
                seen[storage._cdata] = (storage.nbytes(), tensor.element_size())
    return sum(size if unit == "bytes" else size // width for size, width in seen.values())


@pytest.mark.parametrize("unit,budget,expected", [("parameters", 21, 21), ("bytes", 138, 138)])
def test_combined_apply_and_real_export_count_fixed_buffers_bias_and_shared_aliases_once(unit, budget, expected):
    model, data = _TwoTargets(), _data3()
    original, data_before = capture_inventory(model), data.clone()
    modes = tuple(module.training for module in model.modules())
    store = StatisticsStore(16384)
    catalog = prepare_candidate_catalog(model, data, alternatives={
        "a": (CandidateRecipe("weighted_svd", 1), CandidateRecipe("weighted_svd", 2)),
        "b": (CandidateRecipe("asvd", 1), CandidateRecipe("asvd", 2)),
    }, store=store)
    infeasible = allocate_candidate_catalog(catalog, budget - 1, unit=unit)
    assert isinstance(infeasible, BudgetInfeasible) and infeasible.minimum_cost == expected
    plan = allocate_candidate_catalog(catalog, budget, unit=unit)
    assert plan.allocation.total_cost == expected
    actual = apply_candidate_allocation(model, data, plan, store=store)
    assert _physical_cost(actual.model, unit) == expected
    assert actual.model.map is actual.model.map_alias
    assert actual.model.map.untyped_storage()._cdata != model.map.untyped_storage()._cdata
    exported = verify_candidate_export(actual)
    assert exported.evidence["status"] == "verified"
    assert exported.evidence["actual_cost"] == expected
    assert exported.evidence["tensor_bytes"] == 138
    assert exported.evidence["serialized_stream_bytes"] > 138
    assert _physical_cost(exported.model, unit) == expected
    assert exported.model.map is exported.model.map_alias
    torch.testing.assert_close(exported.model(data), actual.model(data), rtol=0, atol=0)
    assert capture_inventory(model) == original
    assert tuple(module.training for module in model.modules()) == modes
    torch.testing.assert_close(data, data_before, rtol=0, atol=0)


def test_downstream_asvd_table_uses_explicit_original_frozen_inputs():
    model, data = _TwoTargets(), _data3()
    store = StatisticsStore(16384)
    catalog = prepare_candidate_catalog(model, data, alternatives={
        "a": (CandidateRecipe("asvd", 1),), "b": (CandidateRecipe("asvd", 1),),
    }, store=store)
    b_table = next(table for table in catalog.tables if table.path == "b")
    snapshot = store.read(b_table.score_ref, expected=b_table.score_ref.provenance)
    with torch.no_grad():
        original_inputs = torch.relu(model.a(data))
        expected_moment = original_inputs.T @ original_inputs / len(data)
    torch.testing.assert_close(snapshot.tensors[0], expected_moment, rtol=1e-12, atol=1e-12)
    assert b_table.score_ref.provenance.model_fingerprint == catalog.context.source.fingerprint
    assert b_table.candidates[0].statistic_ref.provenance.model_fingerprint == catalog.context.source.fingerprint
    plan = allocate_candidate_catalog(catalog, 138, unit="bytes")
    actual = apply_candidate_allocation(model, data, plan, store=store)
    assert b_table.candidates[0].score == pytest.approx(
        _local_error(model.b, actual.model.b, original_inputs), rel=1e-10, abs=1e-12)
    with torch.no_grad():
        changed_inputs = torch.relu(actual.model.a(data))
    assert not torch.allclose(original_inputs, changed_inputs)
    assert catalog.context.semantics == "frozen_snapshot_surrogate"


@pytest.mark.parametrize("change", ["upstream", "nonpersistent_value", "storage_alias", "calibration"])
def test_apply_rejects_changed_full_source_context_before_any_candidate_replay(change, monkeypatch):
    import fedcore.algorithm.low_rank.candidate_graph as candidate_module
    model, data = _TwoTargets(), _data3()
    model.register_buffer("secret", torch.tensor([7.], dtype=torch.float64), persistent=False)
    store = StatisticsStore(16384)
    catalog = prepare_candidate_catalog(model, data, alternatives={"b": (CandidateRecipe("asvd", 1),)}, store=store)
    plan = allocate_candidate_catalog(catalog, 10000, unit="bytes")
    with torch.no_grad():
        if change == "upstream":
            model.a.weight[0, 0] += 1
        elif change == "nonpersistent_value":
            model.secret.add_(1)
        elif change == "storage_alias":
            model.map_alias = model.map.clone()
        else:
            data[0, 0] += 1
    before, refs = capture_inventory(model), store.refs
    monkeypatch.setattr(candidate_module, "execute_fixed_graph", lambda *_args, **_kwargs: pytest.fail("replay before source validation"))
    with pytest.raises(CandidateGraphError) as caught:
        apply_candidate_allocation(model, data, plan, store=store)
    assert "StaleSource" in _codes(caught.value)
    assert capture_inventory(model) == before and store.refs == refs


def test_resealed_catalog_cannot_reinterpret_abs_statistic_as_weighted_moment(monkeypatch):
    import fedcore.algorithm.low_rank.candidate_graph as candidate_module
    model, data = _diagonal([4., 1.]), torch.tensor([[1., 3.], [-1., 3.]], dtype=torch.float64)
    store = StatisticsStore(4096)
    catalog = prepare_candidate_catalog(model, data, alternatives={"0": (
        CandidateRecipe("weighted_svd", 1), CandidateRecipe("asvd", 1),
    )}, store=store)
    table = catalog.tables[0]
    weighted, asvd = table.candidates
    forged_weighted = replace(weighted, statistic_ref=asvd.statistic_ref)
    forged = replace(catalog, tables=(replace(table, candidates=(forged_weighted,)),), fingerprint="")
    forged = replace(forged, fingerprint=candidate_module._digest(forged))
    monkeypatch.setattr(candidate_module, "execute_fixed_graph", lambda *_args, **_kwargs: pytest.fail("replay before semantic-ref validation"))
    with pytest.raises(CandidateGraphError) as caught:
        plan = allocate_candidate_catalog(forged, 1000, unit="bytes")
        apply_candidate_allocation(model, data, plan, store=store)
    assert _codes(caught.value) & {"IncompatibleProvenance", "InvalidCandidateBinding", "InvalidArtifactRef"}


@pytest.mark.parametrize("invalid_inventory", [(None,), ("storage-id-without-description",)])
def test_resealed_candidate_inventory_has_a_structured_shape_failure(invalid_inventory):
    import fedcore.algorithm.low_rank.candidate_graph as candidate_module
    model, data = _diagonal([4., 1.]), torch.tensor([[1., 3.], [-1., 3.]], dtype=torch.float64)
    store = StatisticsStore(4096)
    catalog = prepare_candidate_catalog(model, data, alternatives={"0": (CandidateRecipe("weighted_svd", 1),)}, store=store)
    table = catalog.tables[0]
    forged_candidate = replace(table.candidates[0], storages=invalid_inventory)
    forged = replace(catalog, tables=(replace(table, candidates=(forged_candidate,)),), fingerprint="")
    forged = replace(forged, fingerprint=candidate_module._digest(forged))
    with pytest.raises(CandidateGraphError) as caught:
        allocate_candidate_catalog(forged, 1000, unit="bytes")
    assert _codes(caught.value) & {"InvalidCatalog", "InvalidCandidateBinding", "InvalidStorage", "InvalidInventory"}


@pytest.mark.parametrize("field,value", [("graph_version", 123), ("graph_version", None), ("rng_fingerprint", 123)])
def test_resealed_basic_context_field_type_errors_are_structured(field, value):
    import fedcore.algorithm.low_rank.candidate_graph as candidate_module
    model, data = _diagonal([4., 1.]), torch.tensor([[1., 3.], [-1., 3.]], dtype=torch.float64)
    store = StatisticsStore(4096)
    catalog = prepare_candidate_catalog(model, data, alternatives={"0": (CandidateRecipe("weighted_svd", 1),)}, store=store)
    forged = replace(catalog, context=replace(catalog.context, **{field: value}), fingerprint="")
    forged = replace(forged, fingerprint=candidate_module._digest(forged))
    with pytest.raises(CandidateGraphError) as caught:
        allocate_candidate_catalog(forged, 1000, unit="bytes")
    assert _codes(caught.value) & {"InvalidContext", "InvalidSourceContext", "IncompatibleProvenance"}


def test_unserializable_export_is_a_structured_failure_and_preserves_prepared_model():
    model, data = _diagonal([4., 1.]), torch.tensor([[1., 3.], [-1., 3.]], dtype=torch.float64)
    store = StatisticsStore(4096)
    catalog = prepare_candidate_catalog(model, data, alternatives={"0": (CandidateRecipe("weighted_svd", 1),)}, store=store)
    plan = allocate_candidate_catalog(catalog, 32, unit="bytes")
    actual = apply_candidate_allocation(model, data, plan, store=store)
    actual.model.unserializable_callback = lambda value: value
    before = capture_inventory(actual.model)
    with pytest.raises(CandidateGraphError) as caught:
        verify_candidate_export(actual)
    assert "UnsupportedExport" in _codes(caught.value)
    assert capture_inventory(actual.model) == before


def test_export_refuses_same_value_alias_topology_change_even_with_same_payload_cost(monkeypatch):
    import fedcore.algorithm.low_rank.candidate_graph as candidate_module
    model, data = _TwoTargets(), _data3()
    store = StatisticsStore(16384)
    catalog = prepare_candidate_catalog(model, data, alternatives={"a": (CandidateRecipe("weighted_svd", 1),)}, store=store)
    plan = allocate_candidate_catalog(catalog, 10000, unit="bytes")
    actual = apply_candidate_allocation(model, data, plan, store=store)
    original_load = candidate_module.torch.load

    def changed_load(*args, **kwargs):
        restored = original_load(*args, **kwargs)
        # Retain the same physical storage and payload size but change one view's layout.
        restored.map_alias = restored.map.view(1, 5)
        return restored

    monkeypatch.setattr(candidate_module.torch, "load", changed_load)
    with pytest.raises(CandidateGraphError) as caught:
        verify_candidate_export(actual)
    assert "ExportStateMismatch" in _codes(caught.value)


def _assert_metadata(value):
    assert not isinstance(value, (torch.Tensor, nn.Module))
    if is_dataclass(value):
        for field in fields(value):
            _assert_metadata(getattr(value, field.name))
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_metadata(key)
            _assert_metadata(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_metadata(item)


def test_catalog_is_metadata_only_and_read_snapshots_cannot_poison_replay():
    model, data = _diagonal([4., 1.]), torch.tensor([[1., 3.], [-1., 3.]], dtype=torch.float64)
    store = StatisticsStore(4096)
    catalog = prepare_candidate_catalog(model, data, alternatives={"0": (CandidateRecipe("asvd", 1),)}, store=store)
    _assert_metadata(catalog)
    json.dumps(asdict(catalog), allow_nan=False)
    candidate = catalog.tables[0].candidates[0]
    for ref in (candidate.statistic_ref, candidate.score_ref):
        public = store.read(ref, expected=ref.provenance)
        for tensor in public.tensors:
            tensor.fill_(float("nan"))
    plan = allocate_candidate_catalog(catalog, 32, unit="bytes")
    _assert_metadata(plan)
    actual = apply_candidate_allocation(model, data, plan, store=store)
    assert _local_error(model, actual.model, data) == pytest.approx(9., abs=1e-10)
    assert all(item["observe_forward_calls"] == 0 for item in actual.evidence["selected"])


def _rng_state():
    numpy_state = np.random.get_state()
    return random.getstate(), (numpy_state[0], numpy_state[1].copy(), *numpy_state[2:]), torch.get_rng_state().clone()


def _assert_rng_equal(before):
    after = _rng_state()
    assert after[0] == before[0]
    assert after[1][0] == before[1][0] and after[1][2:] == before[1][2:]
    np.testing.assert_array_equal(after[1][1], before[1][1])
    torch.testing.assert_close(after[2], before[2], rtol=0, atol=0)


@pytest.mark.parametrize("failure", [False, True])
def test_batch_apply_preserves_source_calibration_hooks_modes_and_cpu_rng_on_all_exits(failure, monkeypatch):
    import fedcore.algorithm.low_rank.candidate_graph as candidate_module
    model, data = _TwoTargets(), _data3()
    model.train()
    model.b.eval()
    store = StatisticsStore(16384)
    catalog = prepare_candidate_catalog(model, data, alternatives={
        "a": (CandidateRecipe("weighted_svd", 1),), "b": (CandidateRecipe("asvd", 1),),
    }, store=store)
    plan = allocate_candidate_catalog(catalog, 138, unit="bytes")
    rng, before, input_before, refs = _rng_state(), capture_inventory(model), data.clone(), store.refs
    mode_before = tuple(module.training for module in model.modules())
    hooks_before = tuple((len(module._forward_pre_hooks), len(module._forward_hooks)) for module in model.modules())
    if failure:
        execute, calls = candidate_module.execute_fixed_graph, []

        def fail_second(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                random.random()
                np.random.random()
                torch.rand(1)
                raise RuntimeError("independent selected replay failed")
            return execute(*args, **kwargs)

        monkeypatch.setattr(candidate_module, "execute_fixed_graph", fail_second)
        with pytest.raises(RuntimeError, match="independent selected replay failed"):
            apply_candidate_allocation(model, data, plan, store=store)
    else:
        actual = apply_candidate_allocation(model, data, plan, store=store)
        assert actual.model is not model
    _assert_rng_equal(rng)
    assert capture_inventory(model) == before and store.refs == refs
    assert tuple(module.training for module in model.modules()) == mode_before
    assert tuple((len(module._forward_pre_hooks), len(module._forward_hooks)) for module in model.modules()) == hooks_before
    torch.testing.assert_close(data, input_before, rtol=0, atol=0)


def test_export_rechecks_budget_on_restored_storages(monkeypatch):
    import fedcore.algorithm.low_rank.candidate_graph as candidate_module
    model, data = _diagonal([4., 1.]), torch.tensor([[1., 3.], [-1., 3.]], dtype=torch.float64)
    store = StatisticsStore(4096)
    catalog = prepare_candidate_catalog(model, data, alternatives={"0": (CandidateRecipe("weighted_svd", 1),)}, store=store)
    plan = allocate_candidate_catalog(catalog, 32, unit="bytes")
    actual = apply_candidate_allocation(model, data, plan, store=store)
    original_load = candidate_module.torch.load

    def inflated_load(*args, **kwargs):
        restored = original_load(*args, **kwargs)
        restored.register_buffer("export_added_payload", torch.zeros(1, dtype=torch.float64))
        return restored

    monkeypatch.setattr(candidate_module.torch, "load", inflated_load)
    with pytest.raises(CandidateGraphError) as caught:
        verify_candidate_export(actual)
    assert "BudgetInfeasible" in _codes(caught.value)


@pytest.mark.parametrize("unit", ["parameters", "bytes"])
def test_graph_catalog_allocation_matches_cartesian_oracle_at_every_cost_boundary(unit):
    model, data, store = _TwoTargets(), _data3(), StatisticsStore(16384)
    catalog = prepare_candidate_catalog(model, data, alternatives={
        "a": (CandidateRecipe("weighted_svd", 1), CandidateRecipe("asvd", 2)),
        "b": (CandidateRecipe("asvd", 1), CandidateRecipe("weighted_svd", 2)),
    }, store=store)

    def records(inventory):
        return tuple(StorageCost(record.storage_id, record.nbytes if unit == "bytes" else record.elements,
                                 1, "independent_physical_payload") for record in inventory)

    fixed = records(catalog.fixed_storages)
    tables = tuple(CandidateTable(table.table_id, tuple(CandidateOption(candidate.option_id,
        (candidate.recipe.rank,), candidate.score, records(candidate.storages))
        for candidate in table.candidates)) for table in catalog.tables)
    costs = {sum(record.numel for record in fixed) + sum(record.numel for candidate in choices for record in candidate.storages)
             for choices in product(*(table.options for table in tables))}
    # Independent scopes have no candidate sharing; the map is one fixed storage.
    assert sum(record.numel for record in fixed) == (10 if unit == "bytes" else 5)
    for budget in sorted({0, *(max(0, cost + delta) for cost in costs for delta in (-1, 0, 1))}):
        expected, minimum = _cartesian_oracle(tables, fixed, budget, unit)
        actual = allocate_candidate_catalog(catalog, budget, unit=unit)
        if expected is None:
            assert isinstance(actual, BudgetInfeasible) and actual.minimum_cost == minimum
        else:
            (score, cost, _ids), choices = expected
            assert actual.allocation.total_cost == cost <= budget
            assert tuple((selection.table_id, selection.option.option_id)
                         for selection in actual.allocation.selected) == choices
            assert actual.allocation.objective_score == pytest.approx(float(score), rel=1e-14, abs=1e-14)
