"""Finite alternatives: common scores, exact budgets, replay and export."""
from copy import deepcopy
from dataclasses import replace
from itertools import product
import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.allocation import BudgetInfeasible, exact_unique_cost
from fedcore.algorithm.low_rank.artifacts import StatisticsStore
from fedcore.algorithm.low_rank.candidate_graph import (
    CandidateGraphError, CandidateRecipe, allocate_candidate_catalog, apply_candidate_allocation,
    prepare_candidate_catalog, verify_candidate_export, _digest,
)
from fedcore.algorithm.low_rank.candidate_storage import (
    capture_inventory, inventory_cost, project_storages,
)
from fedcore.algorithm.low_rank.method_specs import ASVD
from fedcore.algorithm.low_rank.plans import _metadata_only
from fedcore.algorithm.low_rank.topology import module_paths


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def _rows(width=3):
    return torch.randn(13, width, generator=torch.Generator().manual_seed(51), dtype=torch.float64)


def _specs(*ranks):
    return tuple(CandidateRecipe("weighted_svd", rank) for rank in ranks)


def _catalog(model, data, alternatives):
    store = StatisticsStore(1024**2)
    return prepare_candidate_catalog(model, data, alternatives=alternatives, store=store), store


def test_common_score_compares_methods_in_one_table_and_surrogate_is_explicit():
    model = nn.Linear(2, 2, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.diag(torch.tensor([4., 1.], dtype=torch.float64)))
    data = torch.tensor([[1., 3.], [1., -3.], [-1., 3.], [-1., -3.]], dtype=torch.float64)
    catalog, store = _catalog(model, data, {"": (CandidateRecipe("weighted_svd", 1), CandidateRecipe("asvd", 1))})
    assert len(catalog.tables) == 1
    assert all(math.isclose(candidate.score, 9., abs_tol=1e-12) for candidate in catalog.tables[0].candidates)
    assert catalog.context.regularization == "none" and _metadata_only(catalog)
    plan = allocate_candidate_catalog(catalog, 10000)
    result = apply_candidate_allocation(model, data, plan, store=store)
    assert result.evidence["semantics"] == "frozen_snapshot_surrogate"
    assert result.evidence["whole_model_evaluation"]["status"] == "required_not_run"
    assert not result.evidence["local_score_is_whole_model_quality"]
    assert json.loads(json.dumps(result.evidence, allow_nan=False))["actual_cost"] == plan.allocation.total_cost


def test_rank_family_is_solved_in_original_metric_not_truncated_returned_operator():
    model = nn.Linear(3, 3, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.diag(torch.tensor([10., 1., .1], dtype=torch.float64)))
    rows = torch.diag(torch.tensor([math.sqrt(.03), math.sqrt(300), math.sqrt(3)], dtype=torch.float64))
    data = torch.cat((rows, -rows))
    catalog, _store = _catalog(model, data, {"": _specs(1, 2)})
    scores = {candidate.recipe.rank: candidate.score for candidate in catalog.tables[0].candidates}
    assert scores[1] == pytest.approx(1.01, abs=1e-12) and scores[2] == pytest.approx(.01, abs=1e-12)
    assert scores[1] < 100.01  # Canonical rank-1 truncation of rank-2 A has this wrong error.


@pytest.mark.parametrize("unit", ["bytes", "parameters"])
def test_small_allocation_matches_independent_cartesian_search(unit):
    model = nn.Sequential(nn.Linear(3, 3), nn.Tanh(), nn.Linear(3, 2)).double()
    data = _rows()
    catalog, store = _catalog(model, data, {"0": _specs(1, 2, 3), "2": _specs(1, 2)})
    costs = []
    choices = list(product(*(table.candidates for table in catalog.tables)))
    fixed = project_storages(catalog.fixed_storages, unit)
    for choice in choices:
        costs.append(exact_unique_cost((*fixed, *(storage for candidate in choice for storage in project_storages(candidate.storages, unit))), unit))
    for budget in sorted({0, min(costs)-1, *costs, max(costs)+1}):
        eligible = [(math.fsum(candidate.score for candidate in choice), cost,
                     tuple(candidate.option_id for candidate in choice))
                    for choice, cost in zip(choices, costs) if cost <= budget]
        plan = allocate_candidate_catalog(catalog, budget, unit=unit)
        if not eligible:
            assert isinstance(plan, BudgetInfeasible) and plan.minimum_cost == min(costs)
        else:
            expected = min(eligible)
            assert plan.allocation.objective_score == expected[0] and plan.allocation.total_cost == expected[1]
            assert tuple(selection.option.option_id for selection in plan.allocation.selected) == expected[2]
            result = apply_candidate_allocation(model, data, plan, store=store)
            assert result.evidence["actual_cost"] == plan.allocation.total_cost <= budget


class BackingBuffers(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(3, 3).double()
        self.b = nn.Linear(3, 2).double()
        backing = torch.arange(8, dtype=torch.float64)
        self.register_buffer("scale", backing[1:5])
        self.register_buffer("scale_alias", backing[2:4], persistent=False)
    def forward(self, value):
        return self.b(torch.tanh(self.a(value))) * self.scale[0]


@pytest.mark.parametrize("unit,budget", [("bytes", 192), ("parameters", 24)])
def test_bias_untouched_full_backing_and_shared_buffers_count_once_after_export(unit, budget):
    model, data = BackingBuffers(), _rows()
    catalog, store = _catalog(model, data, {"a": _specs(1), "b": _specs(1)})
    assert len(catalog.fixed_storages) == 1 and catalog.fixed_storages[0].nbytes == 64
    plan = allocate_candidate_catalog(catalog, budget, unit=unit)
    result = apply_candidate_allocation(model, data, plan, store=store)
    receipt = verify_candidate_export(result)
    assert receipt.evidence["tensor_bytes"] == 192 and result.evidence["actual_parameter_numel"] == 16
    assert inventory_cost(capture_inventory(receipt.model), unit) == budget
    assert receipt.model.scale.untyped_storage()._cdata == receipt.model.scale_alias.untyped_storage()._cdata
    torch.testing.assert_close(receipt.model(data), result.model(data), rtol=0, atol=0)


class AliasNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(3, 3).double()
        self.b = self.a
    def forward(self, value):
        return self.a(value) + self.b(value * 2)


def test_one_alias_table_preserves_sharing_and_two_alias_tables_are_rejected():
    model, data = AliasNetwork(), _rows()
    catalog, store = _catalog(model, data, {"a": _specs(1, 2)})
    plan = allocate_candidate_catalog(catalog, 1000)
    result = apply_candidate_allocation(model, data, plan, store=store)
    assert result.model.a is result.model.b
    with pytest.raises(CandidateGraphError, match="OverlappingScopes"):
        prepare_candidate_catalog(model, data, alternatives={"a": _specs(1), "b": _specs(1)}, store=store)


@pytest.mark.parametrize("change", ["weights", "calibration", "nonpersistent", "alias_layout"])
def test_source_changes_rejected_before_replay(monkeypatch, change):
    from fedcore.algorithm.low_rank import candidate_graph
    model, data = BackingBuffers(), _rows()
    catalog, store = _catalog(model, data, {"a": _specs(1)})
    plan = allocate_candidate_catalog(catalog, 10000)
    if change == "weights":
        with torch.no_grad():
            model.a.weight.add_(.1)
    elif change == "calibration":
        data = data + .1
    elif change == "nonpersistent":
        model.scale_alias.add_(.1)
    else:
        model.scale_alias = model.scale_alias.clone()
    def forbidden(*args, **kwargs):
        pytest.fail("Stale source must be rejected before candidate replay")
    monkeypatch.setattr(candidate_graph, "execute_fixed_graph", forbidden)
    with pytest.raises(CandidateGraphError, match="StaleSource"):
        apply_candidate_allocation(model, data, plan, store=store)


def test_rehashed_wrong_statistic_or_candidate_cost_cannot_be_applied():
    model, data = nn.Linear(3, 3).double(), _rows()
    catalog, store = _catalog(model, data, {"": (CandidateRecipe("weighted_svd", 1), CandidateRecipe("asvd", 1))})
    table = catalog.tables[0]
    weighted, absolute = table.candidates
    forged = replace(weighted, statistic_ref=absolute.statistic_ref)
    changed = replace(catalog, tables=(replace(table, candidates=(forged, absolute)),), fingerprint="")
    changed = replace(changed, fingerprint=_digest(changed))
    with pytest.raises(CandidateGraphError, match="InvalidCandidateBinding"):
        allocate_candidate_catalog(changed, 10000)
    costs = (replace(weighted.storages[0], content_digest="0" * 64), *weighted.storages[1:])
    forged = replace(weighted, storages=costs)
    changed = replace(catalog, tables=(replace(table, candidates=(forged,)),), fingerprint="")
    changed = replace(changed, fingerprint=_digest(changed))
    plan = allocate_candidate_catalog(changed, 10000)
    with pytest.raises(CandidateGraphError, match="CandidateCostMismatch"):
        apply_candidate_allocation(model, data, plan, store=store)


@pytest.mark.parametrize("invalid", ["model", "data", "cap", "rank"])
def test_invalid_inputs_rejected_and_valid_source_preserved(invalid):
    model, data, store = nn.Linear(3, 2).double(), _rows(), StatisticsStore(10000)
    options = {"alternatives": {"": _specs(1)}, "store": store}
    if invalid == "model":
        model = "invalid"
    elif invalid == "data":
        data = "invalid"
    elif invalid == "cap":
        options["max_candidates"] = 0
    else:
        options["alternatives"] = {"": _specs(99)}
    with pytest.raises(CandidateGraphError):
        prepare_candidate_catalog(model, data, **options)
    assert store.used_bytes == 0


def test_candidate_boundary_preserves_source_calibration_rng_and_facade_ownership():
    from fedcore.algorithm.low_rank.low_rank_opt import LowRankModel
    facade = LowRankModel.__new__(LowRankModel)
    facade._model_id_before = facade._model_id_after = None
    facade._model_before_cached = facade._model_after_cached = None
    facade._fedcore_id = "candidate-test"
    facade._registry = SimpleNamespace(register_model=lambda **options: options["stage"])
    model, data, store = nn.Linear(3, 2).double(), _rows(), StatisticsStore(10000)
    facade.trainer = SimpleNamespace(model=model)
    before, calibration, rng = deepcopy(model.state_dict()), data.clone(), torch.get_rng_state().clone()
    catalog = facade.prepare_candidate_catalog(model, data, alternatives={"": _specs(1)}, store=store)
    result = facade.compress_candidates(model, data, allocate_candidate_catalog(catalog, 1000), store=store)
    assert facade.trainer.model is result.model and facade.trainer._fedcore_requires_optimizer_rebuild
    assert torch.equal(rng, torch.get_rng_state()) and not model._forward_pre_hooks
    torch.testing.assert_close(data, calibration, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
