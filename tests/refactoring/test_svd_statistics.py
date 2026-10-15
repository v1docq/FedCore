"""Independent batch references and provenance/memory lifecycle laws."""
from dataclasses import replace
import pytest
import torch

from fedcore.algorithm.low_rank.statistics import (
    MemoryPlan, ResourceLimitExceeded, StatisticsPassport, StaleStatistics,
    absolute_statistic, empty_centered_moments, empty_second_moment,
    finish_centered_moments, finish_second_moment, finish_within_group_covariance,
    frequency_statistic, individual_gradient_squares, merge_centered_moments,
    merge_second_moment, normalized_observations, plan_statistics_memory,
    update_centered_moments, update_second_moment, validate_passport,
)


def passport(**changes):
    return replace(StatisticsPassport("ckpt", "graph-1", "upstream-1", "before_bias", "cal-1"), **changes)


def test_stream_merge_matches_weighted_batch_and_does_not_mutate():
    x = torch.tensor([[1., -2., 3.], [4., 2., -1.], [-3., 7., 2.], [2., -1., -4.]], dtype=torch.float64)
    w = torch.tensor([.1, 0., 3.25, .7], dtype=torch.float64)
    empty = empty_second_moment(3, passport())
    whole = update_second_moment(empty, x, w)
    parts = [update_second_moment(empty, x[start:end], w[start:end]) for start, end in ((0, 1), (1, 3), (3, 4))]
    merged = merge_second_moment(merge_second_moment(parts[0], parts[1]), parts[2])
    regrouped = merge_second_moment(parts[0], merge_second_moment(parts[1], parts[2]))
    expected = torch.einsum("n,ni,nj->ij", w, x, x) / w.sum()
    for state in (whole, merged, regrouped):
        result = finish_second_moment(state)
        torch.testing.assert_close(result.matrix, expected, atol=1e-12, rtol=1e-12)
        assert result.passport.count == 4
        assert result.passport.weight_sum == pytest.approx(float(w.sum()), rel=result.passport.weight_sum_rtol,
                                                          abs=result.passport.weight_sum_atol)
    assert empty.count == 0 and empty.weight_sum == 0
    assert torch.count_nonzero(empty.gram) == 0


def test_centered_and_uncentered_schemas_have_different_meaning():
    x = torch.tensor([[4., 1.], [4., 3.]], dtype=torch.float64)
    centered = finish_centered_moments(update_centered_moments(empty_centered_moments(2, passport()), x))
    uncentered = finish_second_moment(update_second_moment(empty_second_moment(2, passport()), x))
    torch.testing.assert_close(centered.covariance, torch.tensor([[0., 0.], [0., 1.]], dtype=torch.float64))
    torch.testing.assert_close(uncentered.matrix, centered.covariance + torch.outer(centered.mean, centered.mean))
    # Zero mean does not mean zero uncentered moment.
    symmetric = torch.tensor([[3., -2.], [-3., 2.]], dtype=torch.float64)
    result = finish_second_moment(update_second_moment(empty_second_moment(2, passport()), symmetric))
    assert result.matrix.trace() == 13


def test_centered_chan_merge_stays_accurate_with_large_mean():
    x = torch.tensor([[1e9 + 1, -1e9 + 2], [1e9 - 1, -1e9 - 2], [1e9 + 3, -1e9 + 1]], dtype=torch.float64)
    w = torch.tensor([1., 1., 2.], dtype=torch.float64)
    empty = empty_centered_moments(2, passport())
    whole = update_centered_moments(empty, x, w)
    merged = merge_centered_moments(update_centered_moments(empty, x[:2], w[:2]), update_centered_moments(empty, x[2:], w[2:]))
    mean = (x * w[:, None]).sum(0) / w.sum()
    expected = torch.einsum("n,ni,nj->ij", w, x - mean, x - mean) / w.sum()
    torch.testing.assert_close(finish_centered_moments(whole).covariance, expected, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(finish_centered_moments(merged).covariance, expected, atol=1e-7, rtol=1e-7)
    assert merged.count == 3


def test_group_mean_and_individual_gradient_cancellation_do_not_corrupt_statistics():
    empty = empty_centered_moments(2, passport())
    a = torch.tensor([[5., 3.], [5., 3.]], dtype=torch.float64)
    groups = (("plus", update_centered_moments(empty, a)), ("minus", update_centered_moments(empty, -a)))
    within = finish_within_group_covariance(groups)
    torch.testing.assert_close(within.covariance, torch.zeros((2, 2), dtype=torch.float64))
    pooled = finish_centered_moments(update_centered_moments(empty, torch.cat((a, -a))))
    assert pooled.covariance.trace() == 34
    gradients = torch.tensor([[3., -2.], [-3., 2.]], dtype=torch.float64)
    fisher = individual_gradient_squares(gradients, passport(observation_point="per_example_gradient"), loss_id="ce", labels_id="labels-1")
    torch.testing.assert_close(fisher.mean_square, torch.tensor([9., 4.], dtype=torch.float64))
    torch.testing.assert_close(absolute_statistic(gradients, passport()).mean_absolute, torch.tensor([3., 2.], dtype=torch.float64))
    normalized = normalized_observations(gradients, passport())
    assert normalized.second_moment.trace() == pytest.approx(1.)
    assert frequency_statistic(torch.tensor([[1., 0.], [1., 2.]]), passport()).frequencies.tolist() == [1., 1.]


@pytest.mark.parametrize("weights", [torch.tensor([-1., 2.]), torch.tensor([float("nan"), 1.]), torch.tensor([float("inf"), 1.]), torch.tensor([True, False])])
def test_invalid_weights_are_rejected(weights):
    state = empty_second_moment(2, passport())
    with pytest.raises(ValueError):
        update_second_moment(state, torch.eye(2), weights)
    assert state.count == 0


def test_zero_mass_has_explicit_refusal_but_exact_count():
    state = update_second_moment(empty_second_moment(2, passport()), torch.eye(2), torch.zeros(2))
    assert state.count == 2 and state.weight_sum == 0
    with pytest.raises(ValueError, match="zero-mass"):
        finish_second_moment(state)
    with pytest.raises(ValueError, match="test/validation"):
        empty_second_moment(2, passport(data_role="test"))


def test_cache_lifecycle_requires_rebuild_after_predecessor_change():
    old = finish_second_moment(update_second_moment(empty_second_moment(2, passport()), torch.eye(2)))
    assert validate_passport(old.passport, passport()) is None
    current = passport(graph_version="graph-2", predecessor_version="upstream-2")
    stale = validate_passport(old.passport, current)
    assert isinstance(stale, StaleStatistics)
    assert set(stale.mismatches) == {"graph_version", "predecessor_version"}
    rebuilt = finish_second_moment(update_second_moment(empty_second_moment(2, current), 2 * torch.eye(2)))
    assert validate_passport(rebuilt.passport, current) is None
    with pytest.raises(ValueError, match="StaleStatistics"):
        merge_second_moment(empty_second_moment(2, passport()), empty_second_moment(2, current))


def test_memory_plan_accounts_live_buffers_and_refuses_before_allocating(monkeypatch):
    monkeypatch.setattr(torch, "zeros", lambda *args, **kwargs: pytest.fail("Memory planner allocated a tensor"))
    plan = plan_statistics_memory(8, 12, groups=2, model_bytes=1000, replacement_bytes=300,
                                  forward_bytes=500, runtime_overhead_bytes=200)
    assert isinstance(plan, MemoryPlan)
    assert plan.eigensolve_bytes >= 6 * 2 * 8 * 8 * 8
    assert plan.peak_bytes >= 1000 + 300 + 200 + plan.persistent_bytes + plan.eigensolve_bytes
    refusal = plan_statistics_memory(8, 12, groups=2, model_bytes=1000, replacement_bytes=300,
                                     forward_bytes=500, runtime_overhead_bytes=200, max_peak_bytes=plan.peak_bytes - 1)
    assert isinstance(refusal, ResourceLimitExceeded)
    assert refusal.required_bytes == plan.peak_bytes


def test_rectangular_operator_factorization_is_part_of_memory_peak():
    small = plan_statistics_memory(4, 8, operator_rows=4)
    tall = plan_statistics_memory(4, 8, operator_rows=400)
    assert tall.factorization_bytes > 8 * 400 * 4 * 8
    assert tall.peak_bytes > small.peak_bytes


def test_passport_cannot_hold_live_resources_and_frequency_accepts_count_dtype():
    with pytest.raises(ValueError, match="string identifiers"):
        empty_second_moment(2, passport(checkpoint_id=torch.ones(2)))
    torch.testing.assert_close(frequency_statistic(torch.tensor([[1, 0], [3, 2]]), passport()).frequencies,
                               torch.tensor([2., 1.], dtype=torch.float64))


@pytest.mark.parametrize("scale", [1e308, 1e-308, 5e-324])
def test_unit_observation_moment_is_scale_invariant_at_finite_extremes(scale):
    rows = torch.tensor([[scale, scale]], dtype=torch.float64)
    before = rows.clone()
    result = normalized_observations(rows, passport())
    # Each unit row is (1/sqrt(2), 1/sqrt(2)), irrespective of scale.
    expected = torch.full((2, 2), .5, dtype=torch.float64)
    torch.testing.assert_close(result.second_moment, expected, atol=1e-15, rtol=1e-15)
    assert result.second_moment.trace() == pytest.approx(1., abs=1e-15)
    assert result.passport.count == 1 and result.passport.weight_sum == 1
    assert torch.equal(rows, before)


def test_unit_observation_moment_handles_mixed_large_and_tiny_rows():
    rows = torch.tensor([[1e308, -1e308], [1e-308, 1e-308]], dtype=torch.float64)
    result = normalized_observations(rows, passport(), torch.tensor([1., 3.], dtype=torch.float64))
    # Weighted outer products of two orthogonal unit directions.
    expected = torch.tensor([[.5, .25], [.25, .5]], dtype=torch.float64)
    torch.testing.assert_close(result.second_moment, expected, atol=1e-15, rtol=1e-15)
    with pytest.raises(ValueError, match="Zero rows"):
        normalized_observations(torch.zeros((1, 2), dtype=torch.float64), passport())


@pytest.mark.parametrize("rows", [torch.tensor([[1 + 2j]], dtype=torch.complex128),
                                torch.tensor([[float("inf")]], dtype=torch.float64),
                                torch.tensor([[float("nan")]], dtype=torch.float64)])
def test_vector_statistics_validate_original_observation_domain(rows):
    with pytest.raises(ValueError, match="finite float32/float64"):
        absolute_statistic(rows, passport())
    with pytest.raises(ValueError, match="finite float32/float64"):
        normalized_observations(rows, passport())


@pytest.mark.parametrize("changes", [{"weight_sum": complex(1, 1)}, {"weight_sum": float("inf")},
                                    {"weight_sum_rtol": float("nan")}, {"count": True}])
def test_invalid_passport_numeric_metadata_is_rejected(changes):
    with pytest.raises(ValueError):
        empty_second_moment(2, passport(**changes))
