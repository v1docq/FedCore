"""Integer feasibility, manually counted storage, and exhaustive selection laws."""
from dataclasses import replace
from itertools import product
import math
import pytest

from fedcore.algorithm.low_rank.allocation import (
    Allocation, BudgetInfeasible, CandidateOption, CandidateTable, ScoreObjective,
    StorageCost, allocate, exact_unique_cost, exhaustive_allocate,
)


def option(prefix, rank, score, elements):
    return CandidateOption(f"r{rank}", (rank,), score, (StorageCost(f"{prefix}:{rank}", elements),))


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
@pytest.mark.parametrize("budget", [0, 7, 11, 15, 20, 100])
def test_dynamic_program_equals_small_manual_exhaustive_reference(direction, budget):
    tables = (CandidateTable("a", (option("a", 1, 8., 3), option("a", 2, 3., 6), option("a", 4, 0., 10))),
              CandidateTable("b", (option("b", 1, 10., 2), option("b", 2, 2., 5), option("b", 4, 0., 9))))
    fixed = (StorageCost("untouched_bias", 2),)
    objective = ScoreObjective("local_output_error", direction)
    feasible = []
    for left, right in product(tables[0].options, tables[1].options):
        cost = 2 + left.storages[0].numel + right.storages[0].numel
        score = math.fsum((left.score, right.score))
        if cost <= budget:
            feasible.append((score if direction == "minimize" else -score, cost, (left.option_id, right.option_id)))
    result = allocate(tables, budget, objective=objective, fixed_storages=fixed)
    oracle = exhaustive_allocate(tables, budget, objective=objective, fixed_storages=fixed)
    if not feasible:
        assert isinstance(result, BudgetInfeasible) and result.minimum_cost == 7
        assert result == oracle
    else:
        expected = min(feasible)
        assert isinstance(result, Allocation)
        assert result.total_cost == expected[1] <= budget
        assert tuple(selection.option.option_id for selection in result.selected) == expected[2]
        assert result.selected == oracle.selected and result.objective_score == oracle.objective_score


def test_fixed_asvd_aafm_tables_count_bias_maps_residual_and_precision():
    # W[6,8], rank=2: L=12, R=16, bias=6; named maps=8 and residual=3.
    storages = (StorageCost("left", 12, 2), StorageCost("right", 16, 2), StorageCost("bias", 6, 4, "bias"),
                StorageCost("scales", 8, 4, "map"), StorageCost("residual", 3, 8, "residual"))
    assert exact_unique_cost(storages) == 45
    assert exact_unique_cost(storages, "bytes") == 12 * 2 + 16 * 2 + 6 * 4 + 8 * 4 + 3 * 8
    assert exact_unique_cost(storages + (storages[0],)) == 45


def test_basis_sharing_is_counted_once_and_selection_uses_exact_shared_cost():
    basis = StorageCost("shared_basis", 12, 4, "basis")
    a = CandidateTable("a", (CandidateOption("shared", (2,), 1., (basis, StorageCost("a_head", 4))),
                              CandidateOption("private", (1,), 4., (StorageCost("a_private", 9),))))
    b = CandidateTable("b", (CandidateOption("shared", (2,), 1., (basis, StorageCost("b_head", 6))),
                              CandidateOption("private", (1,), 4., (StorageCost("b_private", 10),))))
    fixed = (StorageCost("untouched", 7), StorageCost("bias", 3, category="bias"))
    result = allocate((b, a), 32, objective=ScoreObjective("calibration_error"), fixed_storages=fixed)
    assert result.algorithm == "exhaustive"
    assert tuple(item.option.option_id for item in result.selected) == ("shared", "shared")
    assert result.total_cost == 12 + 4 + 6 + 7 + 3 == 32
    refusal = allocate((a, b), 28, objective=ScoreObjective("calibration_error"), fixed_storages=fixed)
    assert isinstance(refusal, BudgetInfeasible) and refusal.minimum_cost == 29


def test_hardware_grid_and_coupled_ranks_remain_admissible_and_ties_deterministic():
    table = CandidateTable("coupled", (CandidateOption("z", (4, 2), 1., (StorageCost("z", 20),)),
                                      CandidateOption("a", (2, 4), 1., (StorageCost("a", 20),))), ("attn.q", "attn.k"), (4, 4), (2, 2))
    result = allocate((table,), 20, objective=ScoreObjective("local_error"))
    assert result.selected[0].option.option_id == "a"
    assert result.selected[0].option.ranks == (2, 4)
    reversed_table = replace(table, options=tuple(reversed(table.options)))
    assert allocate((reversed_table,), 20, objective=ScoreObjective("local_error")) == result
    with pytest.raises(ValueError, match="shape"):
        allocate((replace(table, maximum_ranks=(2, 2)),), 100, objective=ScoreObjective("error"))
    with pytest.raises(ValueError, match="hardware grid"):
        allocate((replace(table, rank_multiples=(4, 4)),), 100, objective=ScoreObjective("error"))


def test_score_rounding_cannot_prune_the_optimal_prefix():
    tables = (CandidateTable("first", (option("first", 1, 1e16, 1),)),
              CandidateTable("middle", (CandidateOption("a", (1,), 1., (StorageCost("middle_a", 1),)),
                                        CandidateOption("z", (1,), 0., (StorageCost("middle_z", 1),)))),
              CandidateTable("third", (option("third", 1, -1e16, 1),)))
    result = allocate(tables, 3, objective=ScoreObjective("signed_surrogate"))
    oracle = exhaustive_allocate(tables, 3, objective=ScoreObjective("signed_surrogate"))
    assert result.selected == oracle.selected
    assert result.selected[1].option.option_id == "z"
    assert result.objective_score == 0


@pytest.mark.parametrize("budget", [True, -1, 2.5, float("nan")])
def test_invalid_budget_never_silently_rounds_or_increases(budget):
    with pytest.raises(ValueError):
        allocate((CandidateTable("x", (option("x", 1, 1., 5),)),), budget, objective=ScoreObjective("error"))


def test_shared_identity_conflict_and_overlapping_targets_refuse():
    with pytest.raises(ValueError, match="Conflicting"):
        exact_unique_cost((StorageCost("same", 3), StorageCost("same", 4)))
    tables = (CandidateTable("a", (option("a", 1, 1., 2),), ("proj",)), CandidateTable("b", (option("b", 1, 1., 2),), ("proj",)))
    with pytest.raises(ValueError, match="grouped"):
        allocate(tables, 10, objective=ScoreObjective("error"))
    with pytest.raises(ValueError, match="direction"):
        allocate((), 10, objective=ScoreObjective("", "unknown"))
