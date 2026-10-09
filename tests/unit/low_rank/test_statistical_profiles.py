"""Independent objectives, sklearn/numpy references and explicit boundaries."""
from dataclasses import replace
import io
import json

import numpy as np
import pytest
import torch
from sklearn.covariance import ledoit_wolf

from fedcore.algorithm.low_rank.allocation import BudgetInfeasible
from fedcore.algorithm.low_rank.approximation import NumericalDomainViolation
from fedcore.algorithm.low_rank.plans import MetricPolicy
from fedcore.algorithm.low_rank.statistics import (
    StatisticsPassport, empty_centered_moments, finish_centered_moments,
    finish_within_group_covariance, update_centered_moments, validate_passport,
)
from fedcore.algorithm.low_rank.statistical_profiles import (
    ChannelAbsStats, GradientCollectionMetadata, LatencyProfile,
    LatencyProfileKey, RankMeasurement, channel_abs_statistics,
    fwsvd_statistics, ledoit_wolf_metric, select_latency_rank,
    solve_affine_pca, solve_asvd, solve_flar, solve_fwsvd,
)


def passport(**updates):
    return replace(StatisticsPassport("checkpoint", "graph", "predecessor", "input", "calibration"), **updates)


def centered(rows):
    return finish_centered_moments(update_centered_moments(empty_centered_moments(rows.shape[1], passport()), rows))


@pytest.mark.parametrize("mode,expected", [("abs_mean", [3.5, 2., 0.]), ("abs_max", [4., 2., 0.])])
def test_absolute_statistics_mask_and_weight_support(mode, expected):
    rows = torch.tensor([[2., -2., 0.], [-4., 2., 0.], [900., 800., 700.], [600., 500., 400.]])
    stats = channel_abs_statistics(rows, passport(mask_id="mask", weight_id="weights"), mode=mode,
                                   weights=torch.tensor([1., 3., 1., 0.]), mask=torch.tensor([True, True, False, True]))
    torch.testing.assert_close(stats.values, torch.tensor(expected, dtype=torch.float64))
    assert stats.passport.count == 3 and stats.passport.weight_sum == 4
    assert stats.mode == mode


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["abs_mean", "abs_max"])
def test_asvd_objective_matches_independent_sum_and_weighted_singular_tail(dtype, mode):
    x = torch.tensor([[2., 1., 0.], [-1., 3., 0.], [3., -2., 0.]], dtype=dtype)
    w = torch.tensor([[5., 2., 8.], [1., 3., -3.]], dtype=dtype)
    stats = channel_abs_statistics(x, passport(), mode=mode)
    result = solve_asvd(w, stats, 1, alpha=.7, epsilon=.03)
    scales = (stats.values.numpy() + .03) ** .7
    independent_sum = sum(scales[j] ** 2 * sum(float(w[i, j] - result.approximation[i, j]) ** 2
                                               for i in range(w.shape[0])) for j in range(w.shape[1]))
    reference_singular = np.linalg.svd(w.double().numpy() * scales, compute_uv=False)
    assert result.factors.weighted_error_squared == pytest.approx(independent_sum, rel=3e-7, abs=1e-12)
    assert independent_sum == pytest.approx(float(sum(reference_singular[1:] ** 2)), rel=3e-7)
    torch.testing.assert_close(result.left @ result.right, result.approximation)
    assert result.manifest["channel_correlations"] == "not_estimated"
    assert result.manifest["epsilon"] == .03 and result.manifest["alpha"] == .7
    json.dumps(result.manifest, allow_nan=False)


def test_asvd_alpha_zero_is_isotropic_and_zero_policies_are_visible():
    w = torch.tensor([[4., 0., 2.], [0., 3., 0.]], dtype=torch.float64)
    stats = ChannelAbsStats(torch.tensor([1., 1., 0.], dtype=torch.float64), passport(count=2, weight_sum=2.), "abs_mean")
    ordinary = solve_asvd(w, stats, 1, alpha=0)
    u, s, vt = np.linalg.svd(w.numpy(), full_matrices=False)
    np.testing.assert_allclose(ordinary.approximation.numpy(), u[:, :1] @ np.diag(s[:1]) @ vt[:1], rtol=1e-12, atol=1e-12)
    assert ordinary.manifest["scales"] == [1., 1., 1.]
    support = solve_asvd(w, stats, 1)
    assert support.approximation[0, 2] == 0 and support.factors.moment_rank == 2
    keep = solve_asvd(w, stats, 1, zero_policy="preserve_nullspace")
    assert keep.approximation[0, 2] == pytest.approx(2.)
    with pytest.raises(NumericalDomainViolation, match="Zero channel"):
        solve_asvd(w, stats, 1, zero_policy="reject")
    unaligned = w.clone()
    unaligned[1, 2] = 1
    with pytest.raises(NumericalDomainViolation, match="exceeds"):
        solve_asvd(unaligned, stats, 1, zero_policy="preserve_nullspace")
    torch.testing.assert_close(solve_asvd(w, stats, 2).approximation, w, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("alpha,epsilon", [(-1., 0.), (float("nan"), 0.), (.5, -.1), (.5, True)])
def test_asvd_rejects_invalid_power_and_hidden_regularization(alpha, epsilon):
    stats = channel_abs_statistics(torch.eye(2), passport())
    with pytest.raises(NumericalDomainViolation):
        solve_asvd(torch.eye(2), stats, 1, alpha=alpha, epsilon=epsilon)


def test_correlated_activation_mse_is_not_asvd_diagonal_objective():
    x = torch.tensor([[1., 1.], [-1., -1.]], dtype=torch.float64)
    w = torch.tensor([[1., -1.], [0., 0.]], dtype=torch.float64)
    result = solve_asvd(w, channel_abs_statistics(x, passport()), 1)
    # The diagonal estimator itself contains no off-diagonal covariance.
    residual = torch.tensor([[1., -1.], [0., 0.]], dtype=torch.float64)
    assert (x @ residual.T).square().mean() == 0
    assert (residual * torch.tensor(result.manifest["scales"])).square().sum() == 2


def test_fwsvd_paper_rows_become_input_columns_and_sqrt_aggregation():
    w = torch.diag(torch.tensor([4., 1.], dtype=torch.float64))
    squares = torch.tensor([[.25, 8.], [.75, 1.]], dtype=torch.float64)
    stats = fwsvd_statistics(squares, passport(count=2, weight_sum=2.), GradientCollectionMetadata("loss", "labels"))
    result = solve_fwsvd(w, stats, 1)
    # SUM over output gives [1,9]; sqrt produces [1,3]. Using raw weights
    # instead of sqrt would retain the second axis and fail this reference.
    torch.testing.assert_close(result.approximation, torch.diag(torch.tensor([4., 0.], dtype=torch.float64)))
    assert result.manifest["importance"] == [1., 9.]
    independent = sum(float(squares[:, j].sum()) * float((w[:, j] - result.approximation[:, j]).square().sum()) for j in range(2))
    assert independent == result.factors.weighted_error_squared == 9
    assert result.manifest["elementwise_fisher_optimum"] is False
    assert result.manifest["full_fisher"] is False and result.manifest["hessian"] is False
    json.dumps(result.manifest, allow_nan=False)


@pytest.mark.parametrize("shape", [(3, 3), (2, 4), (4, 2)])
def test_affine_projection_with_nonzero_mean_and_original_bias_matches_numpy(shape):
    rng = np.random.default_rng(9)
    x = rng.normal(size=(39, shape[1])) + np.arange(shape[1]) + 2
    w, b = rng.normal(size=shape), rng.normal(size=shape[0])
    y = x @ w.T + b
    wt, bt = torch.tensor(w), torch.tensor(b)
    result = solve_affine_pca(wt, bt, centered(torch.tensor(y)), 1)
    _, vectors = np.linalg.eigh(np.cov(y, rowvar=False, bias=True))
    p = vectors[:, -1:] @ vectors[:, -1:].T
    expected_y = (y - y.mean(0)) @ p + y.mean(0)
    actual_y = x @ result.approximation.numpy().T + result.bias.numpy()
    np.testing.assert_allclose(actual_y, expected_y, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(result.bias.numpy(), p @ b + (np.eye(shape[0]) - p) @ y.mean(0), rtol=1e-11, atol=1e-11)
    direct_error = ((y - actual_y) ** 2).sum() / len(x)
    assert result.factors.weighted_error_squared == pytest.approx(direct_error, rel=1e-11, abs=1e-11)
    full = solve_affine_pca(wt, bt, centered(torch.tensor(y)), min(shape))
    torch.testing.assert_close(full.approximation, wt, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(full.bias, bt, atol=0, rtol=0)


def test_affine_pca_handles_equal_eigenvalues_and_zero_scatter():
    w = torch.eye(3, dtype=torch.float64)
    rows = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]], dtype=torch.float64)
    result = solve_affine_pca(w, None, centered(rows), 1)
    assert result.factors.weighted_error_squared == pytest.approx(.5)
    assert torch.linalg.matrix_rank(result.approximation) == 1
    zero = solve_affine_pca(w, None, centered(torch.ones(3, 3, dtype=torch.float64)), 1)
    assert zero.factors.weighted_error_squared == 0 and zero.bias is not None
    torch.testing.assert_close(torch.ones(3, 3, dtype=torch.float64) @ zero.approximation.T + zero.bias,
                               torch.ones(3, 3, dtype=torch.float64))


def test_bolaco_within_group_subspace_excludes_between_means_and_keeps_weights():
    positive = torch.tensor([[10., -2.], [10., 2.]], dtype=torch.float64)
    negative = torch.tensor([[-10., -1.], [-10., 1.], [-10., -1.], [-10., 1.]], dtype=torch.float64)
    states = tuple((name, update_centered_moments(empty_centered_moments(2, passport()), rows))
                   for name, rows in (("positive", positive), ("negative", negative)))
    groups = finish_within_group_covariance(states)
    w = torch.eye(2, dtype=torch.float64)
    grouped = solve_affine_pca(w, None, groups, 1)
    total = solve_affine_pca(w, None, centered(torch.cat((positive, negative))), 1)
    torch.testing.assert_close(grouped.approximation, torch.diag(torch.tensor([0., 1.], dtype=torch.float64)))
    torch.testing.assert_close(total.approximation, torch.diag(torch.tensor([1., 0.], dtype=torch.float64)))
    assert grouped.manifest["group_weights"] == [1/3, 2/3]
    assert grouped.manifest["group_means"] == [[10., 0.], [-10., 0.]]
    assert grouped.factors.weighted_error_squared == 0
    # One shared bias cannot reconstruct both means, and no total-MSE claim
    # is made by the within-group centered objective.
    actual = torch.cat((positive, negative)) @ grouped.approximation.T + grouped.bias
    assert (actual - torch.cat((positive, negative))).square().sum() > 100
    assert grouped.manifest["between_group_scatter"] == "excluded"
    json.dumps(grouped.manifest, allow_nan=False)


@pytest.mark.parametrize("rows", [
    [[2., 3.]], [[1.], [2.], [6.]], [[3., 4.], [3., 4.]],
    [[1., 2., 4.], [3., 4., 5.], [8., -2., 6.], [4., 7., 2.]],
])
def test_ledoit_wolf_matches_independent_sklearn_centered_ml_reference(rows):
    x = np.asarray(rows, dtype=np.float64)
    metric = ledoit_wolf_metric(torch.tensor(x), passport())
    reference, alpha = ledoit_wolf(x)
    np.testing.assert_allclose(metric.covariance.numpy(), reference, rtol=1e-12, atol=1e-12)
    assert metric.shrinkage == pytest.approx(alpha, rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(metric.mean.numpy(), x.mean(0), atol=1e-12)
    assert metric.formula_version == "ledoit_wolf_2004_centered_ml_v1"


def test_flar_centered_error_translation_invariance_and_zero_variance_support():
    x = torch.tensor([[1., 0., 3.], [-2., 1., -1.], [3., 3., -2.], [4., 1., 1.]], dtype=torch.float64)
    w = torch.tensor([[3., 1., 1.], [1., 2., -1.]], dtype=torch.float64)
    metric = ledoit_wolf_metric(x, passport())
    shifted = ledoit_wolf_metric(x + torch.tensor([30., -17., 11.]), passport())
    torch.testing.assert_close(metric.covariance, shifted.covariance, atol=1e-12, rtol=1e-12)
    result = solve_flar(w, metric, 1)
    difference = w - result.approximation
    centered_objective = torch.trace(difference @ metric.covariance @ difference.T).item()
    assert result.factors.weighted_error_squared == pytest.approx(centered_objective, rel=1e-12)
    assert result.manifest["uncentered_output_mse"] is False
    assert result.manifest["literal_printed_alpha"] is False
    zero = ledoit_wolf_metric(torch.ones((4, 3), dtype=torch.float64), passport())
    support = solve_flar(w, zero, 1)
    assert support.factors.moment_rank == 0 and support.manifest["zero_variance"] is True
    assert support.factors.weighted_error_squared == 0
    ridge = solve_flar(w, zero, 1, policy=MetricPolicy(ridge=.1))
    assert ridge.factors.moment_rank == 3 and ridge.manifest["ridge"] == .1
    json.dumps(result.manifest, allow_nan=False)


def test_profiles_roundtrip_and_statistics_staleness_are_explicit():
    x = torch.tensor([[1., 2.], [3., 1.], [4., -1.]], dtype=torch.float64)
    result = solve_flar(torch.eye(2, dtype=torch.float64), ledoit_wolf_metric(x, passport()), 1)
    buffer = io.BytesIO()
    torch.save({"left": result.left, "right": result.right, "manifest": result.manifest}, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=True)
    torch.testing.assert_close(restored["left"] @ restored["right"], result.approximation)
    assert restored["manifest"] == result.manifest
    stale = validate_passport(passport(), passport(predecessor_version="changed"))
    assert stale.mismatches == ("predecessor_version",)


def latency_profile():
    key = LatencyProfileKey((8, 8), 4, "float32", "torch", "2.0", "cpu:model", "graph-v1", "timing-v1")
    # Both quality and latency are deliberately nonmonotone in rank.
    rows = (RankMeasurement(1, .9, 2.), RankMeasurement(2, .7, 1.),
            RankMeasurement(4, .95, 1.5, (5., 4., 6.), "exported-artifact"))
    return LatencyProfile(key, (1, 2, 4), rows, "test-raw-timings")


def test_latency_finite_grid_uses_actual_cost_and_measurement_can_refute_prediction():
    profile = latency_profile()
    chosen = select_latency_rank(profile, profile.key, maximum_parameters=70, bias_elements=8,
                                 maximum_latency_ms=3.)
    assert chosen.measurement.rank == 1 and chosen.parameter_cost == 24
    assert chosen.manifest["algorithm"] == "finite_grid_enumeration"
    assert chosen.manifest["latency_evidence"] == "prediction"
    # Rank4 predicted 1.5 but the actual artifact measured median 5ms.
    actual = select_latency_rank(profile, profile.key, maximum_parameters=72, bias_elements=8)
    assert actual.measurement.rank == 4 and actual.parameter_cost == 72
    assert actual.manifest["selected_latency_ms"] == 5.
    assert actual.manifest["raw_samples_ms"] == [5., 4., 6.]
    assert actual.manifest["latency_evidence"] == "final_artifact_measurement"
    too_small = select_latency_rank(profile, profile.key, maximum_parameters=23, bias_elements=8)
    assert isinstance(too_small, BudgetInfeasible)
    json.dumps(actual.manifest, allow_nan=False)


@pytest.mark.parametrize("field,value", [("shape", (8, 7)), ("batch", 8), ("dtype", "float64"),
                                         ("runtime", "onnx"), ("runtime_version", "other"),
                                         ("device", "other"), ("graph_version", "other"), ("source_version", "other")])
def test_latency_key_invalidates_every_domain_change(field, value):
    profile = latency_profile()
    with pytest.raises(NumericalDomainViolation, match="Stale latency"):
        select_latency_rank(profile, replace(profile.key, **{field: value}), maximum_parameters=100)


def test_calibration_refuses_test_split_and_nonfinite_inputs():
    with pytest.raises(ValueError, match="test/validation"):
        channel_abs_statistics(torch.eye(2), passport(data_role="test"))
    with pytest.raises(ValueError, match="finite"):
        ledoit_wolf_metric(torch.tensor([[1., float("nan")]]), passport())
