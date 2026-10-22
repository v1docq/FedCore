"""Closed regions, actual sequential stages and compatibility with P2/P3."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.artifacts import ArtifactStoreError, StatisticsStore
from fedcore.algorithm.low_rank.compression_graph import SourceMode
from fedcore.algorithm.low_rank.coupled_layers import ReferenceAttention
from fedcore.algorithm.low_rank.coupled_specs import FlatQueryKey
from fedcore.algorithm.low_rank.grouped_execution import execute_grouped_workflow, prepare_grouped_workflow
from fedcore.algorithm.low_rank.grouped_workflow import (
    CompressionRegion, ConsumerCoordinates, CoordinateAxis, GroupRequest,
    GroupedWorkflowError, grouped_fingerprint, validate_grouped_workflow,
)
from fedcore.algorithm.low_rank.method_execution import transform_method
from fedcore.algorithm.low_rank.method_specs import BasisSharing, DRONE, SVDLLMV1
from fedcore.algorithm.low_rank.plans import ValidationFailure, _metadata_only


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def _data(width=4):
    return torch.randn(13, width, generator=torch.Generator().manual_seed(37), dtype=torch.float64) + .1


def _request(key, paths, spec, *, rank=1, width=4, outputs=None, input_space="features"):
    outputs = (width,) * len(paths) if outputs is None else outputs
    coordinates = tuple(ConsumerCoordinates(path, CoordinateAxis(input_space, tuple(range(width))),
        CoordinateAxis(key + ":output:" + path, tuple(range(output)))) for path, output in zip(paths, outputs))
    region = CompressionRegion(key + ":region", paths, coordinates)
    return GroupRequest(key, region, paths, spec, rank)


def _sequential():
    torch.manual_seed(71)
    return nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 4), nn.Sigmoid(), nn.Linear(4, 4)).double()


@pytest.mark.parametrize("spec", [DRONE(), SVDLLMV1(ridge=.01)])
def test_current_stages_match_supported_legacy_profile_and_real_snapshots(spec):
    model, data = _sequential(), _data()
    before = deepcopy(model.state_dict())
    requests = tuple(_request("group" + path, (path,), spec) for path in ("0", "2", "4"))
    workflow = prepare_grouped_workflow(model, data, requests, batch_size=4)
    assert _metadata_only(workflow) and validate_grouped_workflow(workflow) == workflow
    rng = torch.get_rng_state().clone()
    store = StatisticsStore(10000)
    result = execute_grouped_workflow(model, data, workflow, store=store)
    legacy = transform_method(model, data, spec, rank=1, target_paths=("0", "2", "4"), batch_size=4)
    torch.testing.assert_close(result.model(data), legacy.model(data), rtol=1e-11, atol=1e-11)
    assert torch.equal(rng, torch.get_rng_state())
    assert result.evidence["accepted_versions"] == 3
    assert [event["stage"] for event in result.evidence["trace"]] == ["observe", "solve", "apply", "advance"] * 3
    observations = [event for event in result.evidence["trace"] if event["stage"] == "observe"]
    assert len({event["model_fingerprint"] for event in observations}) == 3
    for path, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[path], rtol=0, atol=0)
    refs = [ref for ref in result.statistic_refs if ":initialization=" not in ref.provenance.graph_version]
    for ref, event in zip(refs, observations):
        assert ref.provenance.model_fingerprint == event["model_fingerprint"]
        assert store.read(ref, expected=ref.provenance).passports[0].predecessor_version == event["model_fingerprint"]
    warm = execute_grouped_workflow(model, data, workflow, store=store)
    assert warm.statistic_refs == result.statistic_refs
    assert all(event["forwards"] > 0 and all(event["statistic_cache_hits"])
        for event in warm.evidence["trace"] if event["stage"] == "observe")
    assert all(not module._forward_pre_hooks for module in model.modules())
    json.dumps(result.evidence, allow_nan=False)


class _Parallel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 2).double()
        self.b = nn.Linear(4, 3).double()
        self.c = nn.Linear(4, 2).double()
        self.d = nn.Linear(4, 3).double()

    def forward(self, data):
        return torch.cat((self.a(data), self.b(data), self.c(data), self.d(data)), -1)


@pytest.mark.parametrize("mode", [SourceMode.FROZEN_SNAPSHOT, SourceMode.CURRENT_STUDENT])
def test_two_closed_basis_regions_keep_physical_sharing_and_old_numerics(mode):
    model, data = _Parallel(), _data()
    requests = (_request("ab", ("a", "b"), BasisSharing(), rank=3, outputs=(2, 3)),
                _request("cd", ("c", "d"), BasisSharing(), rank=2, outputs=(2, 3)))
    plan = prepare_grouped_workflow(model, data, requests, source_mode=mode)
    result = execute_grouped_workflow(model, data, plan, store=StatisticsStore(10000))
    assert result.model.a.right is result.model.b.right
    assert result.model.c.right is result.model.d.right
    assert result.model.a.right is not result.model.c.right
    expected_ab = transform_method(model, data, BasisSharing(), rank=3, target_paths=("a", "b"))
    torch.testing.assert_close(result.model.a(data), expected_ab.model.a(data), rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(result.model.b(data), expected_ab.model.b(data), rtol=1e-11, atol=1e-11)
    assert result.evidence["groups"][0]["numerics"]["coordinate_evidence"] == "declared_assumption"
    observations = [event for event in result.evidence["trace"] if event["stage"] == "observe"]
    if mode is SourceMode.FROZEN_SNAPSHOT:
        assert len({event["model_fingerprint"] for event in observations}) == 1
        assert result.evidence["accepted_versions"] == 1
    else:
        assert len({event["model_fingerprint"] for event in observations}) == 2


@pytest.mark.parametrize("scale,spec", [
    (1., BasisSharing(metric_mode="equal_moments")),
    (2., BasisSharing(metric_mode="proportional_moments", proportional_scales=(1., 4.))),
])
def test_shared_metric_relations_preserve_legacy_results_and_physical_basis(scale, spec):
    torch.manual_seed(163)
    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Identity(),
                          nn.Linear(4, 3, bias=False)).double()
    with torch.no_grad():
        model[0].weight.copy_(scale * torch.eye(4, dtype=torch.float64))
    data = _data()
    request = _request("related", ("0", "2"), spec, rank=2, outputs=(4, 3))
    workflow = prepare_grouped_workflow(model, data, (request,), batch_size=4)
    result = execute_grouped_workflow(model, data, workflow, store=StatisticsStore(10000))
    legacy = transform_method(model, data, spec, rank=2, target_paths=("0", "2"), batch_size=4)
    assert result.model[0].right is result.model[2].right
    for name, tensor in legacy.model.state_dict().items():
        torch.testing.assert_close(result.model.state_dict()[name], tensor, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(result.model(data), legacy.model(data), rtol=1e-11, atol=1e-11)
    assert result.evidence["groups"][0]["numerics"]["metric_mode"] == spec.metric_mode


@pytest.mark.parametrize("case", ["subset", "coordinate_space", "coordinate_order", "overlap", "repeat", "batch_sequential"])
def test_invalid_complete_group_contract_refuses_before_any_effect(monkeypatch, case):
    from fedcore.algorithm.low_rank import grouped_execution
    model, data = _Parallel(), _data()
    request = _request("ab", ("a", "b"), BasisSharing(), outputs=(2, 3))
    requests, mode = (request,), SourceMode.CURRENT_STUDENT
    if case == "subset":
        requests = (replace(request, target_paths=("a",)),)
    elif case in ("coordinate_space", "coordinate_order"):
        bindings = request.region.coordinates
        axis = CoordinateAxis("other" if case == "coordinate_space" else "features",
            (0, 1, 2, 3) if case == "coordinate_space" else (3, 2, 1, 0))
        requests = (replace(request, region=replace(request.region,
            coordinates=(bindings[0], replace(bindings[1], input=axis)))),)
    elif case == "overlap":
        requests = (request, _request("ac", ("a", "c"), BasisSharing(), outputs=(2, 2)))
    elif case == "repeat":
        requests = (request, request)
    else:
        requests = (_request("a", ("a",), DRONE(), outputs=(2,)),)
        mode = SourceMode.FROZEN_SNAPSHOT
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid metadata reached model copying or observation")
    monkeypatch.setattr(grouped_execution, "clone_preserving_storages", forbidden)
    monkeypatch.setattr(nn.Module, "register_forward_pre_hook", forbidden)
    with pytest.raises(GroupedWorkflowError):
        prepare_grouped_workflow(model, data, requests, source_mode=mode)


def test_resealed_tampered_plan_does_not_authorize_new_transform(monkeypatch):
    from fedcore.algorithm.low_rank import grouped_execution
    model, data = _sequential(), _data()
    plan = prepare_grouped_workflow(model, data, (_request("first", ("0",), DRONE()),))
    bad_transform = replace(plan.groups[0].transform,
        request=replace(plan.groups[0].transform.request, objective="unrelated"))
    bad = replace(plan, groups=(replace(plan.groups[0], transform=bad_transform),))
    bad = replace(bad, fingerprint=grouped_fingerprint(bad))
    monkeypatch.setattr(grouped_execution, "clone_preserving_storages", lambda *a, **k: pytest.fail("Copied forged plan"))
    with pytest.raises(GroupedWorkflowError) as caught:
        execute_grouped_workflow(model, data, bad, store=StatisticsStore(10000))
    assert tuple(violation.code for violation in caught.value.violations) == ("InvalidTransform",)


@pytest.mark.parametrize("spec,capacity", [(DRONE(), 383), (SVDLLMV1(), 767)])
def test_last_statistic_admission_failure_rolls_back_entire_job(spec, capacity):
    model, data = _sequential(), _data()
    requests = tuple(_request(path, (path,), spec) for path in ("0", "2", "4"))
    plan = prepare_grouped_workflow(model, data, requests)
    store = StatisticsStore(capacity)
    before = deepcopy(model.state_dict())
    with pytest.raises(ArtifactStoreError, match="StoreCapacityExceeded"):
        execute_grouped_workflow(model, data, plan, store=store)
    assert store.used_bytes == 0 and not store.refs
    for path, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[path], rtol=0, atol=0)


def test_full_backing_storage_and_nonpersistent_buffer_provenance():
    model, data = _sequential(), _data()
    owner = torch.arange(100, dtype=torch.float64)
    model.register_buffer("scratch", owner[:2], persistent=False)
    request = _request("first", ("0",), DRONE())
    plan = prepare_grouped_workflow(model, data, (request,))
    assert any(record.nbytes == 800 and not record.views[0].persistent for record in plan.source_inventory)
    owner[-1] = -1
    with pytest.raises(GroupedWorkflowError, match="StaleSource"):
        execute_grouped_workflow(model, data, plan, store=StatisticsStore(10000))


def test_final_physical_budget_failure_rolls_back_statistics():
    model, data = _sequential(), _data()
    model.register_buffer("scratch", torch.zeros(1000), persistent=False)
    plan = prepare_grouped_workflow(model, data, (_request("first", ("0",), DRONE()),), tensor_byte_budget=500)
    store = StatisticsStore(10000)
    with pytest.raises(GroupedWorkflowError, match="BudgetInfeasible"):
        execute_grouped_workflow(model, data, plan, store=store)
    assert not store.refs and store.used_bytes == 0


def test_reference_qk_adapter_preserves_actual_rope_gqa_atlas_and_legacy_output():
    model = nn.Sequential(ReferenceAttention(4, 4, 2, 4, head_map=(0, 1, 0, 1), dtype=torch.float64)).eval()
    data = _data().reshape(1, 13, 4)
    axis = CoordinateAxis("reference:0:residual", (0, 1, 2, 3))
    region = CompressionRegion("qk", ("0",), (ConsumerCoordinates("0", axis, axis),))
    spec = FlatQueryKey(coordinates=(2, 3))
    plan = prepare_grouped_workflow(model, data, (GroupRequest("qk", region, ("0",), spec, 2),))
    assert plan.groups[0].reference_coordinates[0].head_map == (0, 1, 0, 1)
    result = execute_grouped_workflow(model, data, plan, store=StatisticsStore(0))
    legacy = transform_method(model, data, spec, rank=2, target_paths=("0",))
    torch.testing.assert_close(result.model(data), legacy.model(data), rtol=0, atol=0)
    assert result.model[0].group_coordinates.tolist() == [[2, 3], [2, 3]]
    assert not result.statistic_refs
    assert [event["stage"] for event in result.evidence["trace"]] == ["checked_whole_recipe", "apply", "advance"]
    assert result.model[0].d_model == 4 and result.model[0].logits_scale == model[0].logits_scale


@pytest.mark.parametrize("field,value", [("source_mode", "current_student"), ("schema_version", True), ("groups", ())])
def test_static_validator_returns_structured_failure(field, value):
    model, data = _sequential(), _data()
    plan = prepare_grouped_workflow(model, data, (_request("first", ("0",), DRONE()),))
    failure = validate_grouped_workflow(replace(plan, **{field: value}))
    assert type(failure) is ValidationFailure and failure.violations[0].code and failure.violations[0].path
