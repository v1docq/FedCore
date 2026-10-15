"""Output-error identities and independent weighted-SVD numerical evidence."""
import pytest
import json
import torch
from torch import nn
from torch.nn import functional as F

from fedcore.algorithm.low_rank.approximation import MetricPolicy, NumericalDomainViolation, solve_weighted
from fedcore.algorithm.low_rank.statistics import StatisticsPassport, empty_second_moment, finish_second_moment, update_second_moment


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_direct_output_error_equals_quadratic_objective(dtype):
    generator = torch.Generator().manual_seed(82)
    x = torch.randn((37, 5), dtype=dtype, generator=generator) * torch.tensor([1., 2., .1, 3., 1.], dtype=dtype)
    w = torch.randn((4, 5), dtype=dtype, generator=generator)
    c = x.double().T @ x.double() / len(x)
    result = solve_weighted(w, c, 2)
    difference = w.double() - result.approximation.double()
    direct = ((x.double() @ difference.T).square().sum() / len(x)).item()
    quadratic = torch.trace(difference @ c @ difference.T).item()
    assert result.weighted_error_squared == pytest.approx(direct, abs=1e-10, rel=1e-10)
    assert direct == pytest.approx(quadratic, abs=1e-10, rel=1e-10)
    assert result.actual_rank <= 2
    torch.testing.assert_close((result.u * result.s) @ result.vh, result.approximation)
    assert result.factor_semantics == "canonical_svd_of_returned_operator"


def test_isotropic_metric_reduces_to_ordinary_truncated_svd():
    w = torch.tensor([[5., 1., 2.], [1., 4., -2.], [0., 1., 3.]], dtype=torch.float64)
    u, s, vh = torch.linalg.svd(w)
    ordinary = (u[:, :2] * s[:2]) @ vh[:2]
    result = solve_weighted(w, 7 * torch.eye(3, dtype=torch.float64), 2)
    torch.testing.assert_close(result.approximation, ordinary, atol=1e-12, rtol=1e-12)
    assert result.weighted_error_squared == pytest.approx(7 * s[-1].square().item(), rel=1e-12)


def test_uses_metric_root_instead_of_second_moment_itself():
    # W*C would select the second axis, whereas W*sqrt(C) selects the first.
    w = torch.diag(torch.tensor([4., 1.], dtype=torch.float64))
    c = torch.diag(torch.tensor([1., 9.], dtype=torch.float64))
    result = solve_weighted(w, c, 1)
    torch.testing.assert_close(result.approximation, torch.diag(torch.tensor([4., 0.], dtype=torch.float64)))
    assert result.weighted_error_squared == 9


@pytest.mark.parametrize("shape", [(3, 3), (2, 4), (4, 2)])
@pytest.mark.parametrize("policy", [MetricPolicy(), MetricPolicy(ridge=.01), MetricPolicy(nullspace_policy="preserve_nullspace")])
def test_full_admissible_rank_preserves_unobserved_operator(shape, policy):
    generator = torch.Generator().manual_seed(191)
    w = torch.randn(shape, dtype=torch.float64, generator=generator)
    c = torch.zeros((shape[1], shape[1]), dtype=torch.float64)
    c[0, 0] = 3
    result = solve_weighted(w, c, min(shape), policy)
    torch.testing.assert_close(result.approximation, w, atol=1e-12, rtol=1e-12)
    assert result.nullspace_policy == "preserve_operator_at_full_rank"


def test_support_only_and_preserve_nullspace_are_explicit_and_rank_checked():
    w = torch.diag(torch.tensor([4., 3., 2.], dtype=torch.float64))
    c = torch.diag(torch.tensor([9., 1., 0.], dtype=torch.float64))
    result = solve_weighted(w, c, 1)
    torch.testing.assert_close(result.approximation, torch.diag(torch.tensor([4., 0., 0.], dtype=torch.float64)))
    assert result.moment_rank == 2 and result.nullspace_policy == "support_only"
    with pytest.raises(NumericalDomainViolation, match="null-space action exceeds"):
        solve_weighted(w, c, 1, MetricPolicy(nullspace_policy="preserve_nullspace"))
    # Invisible action can be retained when it lies in the same output subspace.
    aligned = torch.tensor([[4., 0., 2.], [0., 0., 0.]], dtype=torch.float64)
    kept = solve_weighted(aligned, c, 1, MetricPolicy(nullspace_policy="preserve_nullspace"))
    torch.testing.assert_close(kept.approximation, aligned)
    assert kept.actual_rank == 1


def test_ridge_changes_named_objective_and_cutoff_is_reported():
    w = torch.diag(torch.tensor([1., 2.], dtype=torch.float64))
    c = torch.diag(torch.tensor([10., 0.], dtype=torch.float64))
    ridge = solve_weighted(w, c, 1, MetricPolicy(ridge=10))
    assert ridge.objective == "input_second_moment_ridge" and ridge.ridge == 10
    torch.testing.assert_close(ridge.approximation, torch.diag(torch.tensor([0., 2.], dtype=torch.float64)))
    thresholded = solve_weighted(w, torch.diag(torch.tensor([1., 1e-14], dtype=torch.float64)), 1, MetricPolicy(rcond=1e-10))
    assert thresholded.moment_rank == 1
    assert thresholded.effective_cutoff == 1e-10
    assert thresholded.discarded_metric_mass == 1e-14


def test_zero_metric_has_explicit_support_and_full_rank_restoration():
    w = torch.tensor([[2., 1.], [1., 3.]], dtype=torch.float64)
    zero = solve_weighted(w, torch.zeros_like(w), 1)
    assert zero.actual_rank == 0 and zero.moment_rank == 0
    assert zero.condition_number is None and zero.weighted_error_squared == 0
    torch.testing.assert_close(zero.approximation, torch.zeros_like(w))
    torch.testing.assert_close(solve_weighted(w, torch.zeros_like(w), 2).approximation, w)


def test_checked_core_matches_independent_experimental_oracle():
    from fedcore.experiments.math_checks import weighted_svd
    generator = torch.Generator().manual_seed(16)
    for ridge in (0., .01):
        w = torch.randn((4, 6), dtype=torch.float64, generator=generator)
        x = torch.randn((19, 6), dtype=torch.float64, generator=generator)
        c = x.T @ x / len(x)
        production = solve_weighted(w, c, 2, MetricPolicy(ridge=ridge))
        oracle = weighted_svd(w, c, 2, ridge=ridge)
        torch.testing.assert_close(production.approximation, oracle["approximation"], atol=1e-11, rtol=1e-11)
        assert production.weighted_error_squared == pytest.approx(oracle["weighted_error_squared"], rel=1e-11)


@pytest.mark.parametrize("groups", [1, 2])
def test_conv_patch_output_error_includes_stride_dilation_padding_and_groups(groups):
    generator = torch.Generator().manual_seed(18)
    layer = nn.Conv2d(4, 6, (2, 3), groups=groups, stride=(2, 1), dilation=(1, 2), padding=(1, 2)).double()
    x = torch.randn((2, 4, 7, 8), dtype=torch.float64, generator=generator)
    matrices = layer.weight.detach().reshape(groups, 6 // groups, -1)
    patches = F.unfold(x, (2, 3), dilation=(1, 2), padding=(1, 2), stride=(2, 1)).reshape(2, groups, matrices.shape[-1], -1)
    approximations, error = [], 0.
    for group in range(groups):
        rows = patches[:, group].transpose(1, 2).reshape(-1, matrices.shape[-1])
        passport = StatisticsPassport("ckpt", "graph", "previous", "input_patches_before_bias", "cal")
        moment = finish_second_moment(update_second_moment(empty_second_moment(rows.shape[1], passport), rows))
        result = solve_weighted(matrices[group], moment, 1)
        approximations.append(result.approximation)
        error += result.weighted_error_squared
    weight = torch.stack(approximations).reshape_as(layer.weight)
    direct = (layer(x) - F.conv2d(x, weight, layer.bias, layer.stride, layer.padding, layer.dilation, groups)).square().sum()
    assert direct.item() == pytest.approx(error * patches.shape[0] * patches.shape[-1], rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("bad", [torch.tensor([[1., 2.], [0., 1.]]), torch.diag(torch.tensor([1., -1.])), torch.full((2, 2), float("nan")), torch.ones(3, 3), torch.ones(2, 2, dtype=torch.float16)])
def test_invalid_metric_refuses_without_mutating_operator(bad):
    w = torch.eye(2, dtype=torch.float64)
    original = w.clone()
    with pytest.raises(NumericalDomainViolation):
        solve_weighted(w, bad, 1)
    assert torch.equal(w, original)


@pytest.mark.parametrize("policy", [MetricPolicy(ridge=-1), MetricPolicy(ridge=True), MetricPolicy(rcond=float("inf")), MetricPolicy(nullspace_policy="implicit")])
def test_invalid_policy_is_not_hidden_regularization(policy):
    with pytest.raises(NumericalDomainViolation):
        solve_weighted(torch.eye(2), torch.eye(2), 1, policy)


def test_large_finite_metric_does_not_overflow_symmetrization():
    weight = torch.eye(2, dtype=torch.float64)
    moment = torch.eye(2, dtype=torch.float64) * 1e308
    result = solve_weighted(weight, moment, 1)
    assert result.weighted_error_squared == pytest.approx(1e308, rel=1e-12)
    assert result.actual_rank == 1 and result.moment_rank == 2
    assert torch.isfinite(result.approximation).all()
    # The evidence is suitable for strict JSON, without Infinity/NaN literals.
    diagnostics = {name: getattr(result, name) for name in result.__dataclass_fields__
                   if name not in ("u", "s", "vh", "approximation")}
    json.dumps(diagnostics, allow_nan=False)


@pytest.mark.parametrize("metric_scale,policy,stage", [
    (1., MetricPolicy(), "diagnostics"),
    (1e308, MetricPolicy(), "weighted operator"),
    (1e308, MetricPolicy(ridge=1e308), "regularized metric"),
])
def test_finite_inputs_that_overflow_float64_domain_fail_explicitly(metric_scale, policy, stage):
    weight = torch.eye(2, dtype=torch.float64) * 1e200
    moment = torch.eye(2, dtype=torch.float64) * metric_scale
    before = weight.clone()
    with pytest.raises(NumericalDomainViolation, match=stage):
        solve_weighted(weight, moment, 1, policy)
    assert torch.equal(weight, before)


def test_numerical_solver_failure_is_a_domain_refusal(monkeypatch):
    def fail_svd(*args, **kwargs):
        raise torch.linalg.LinAlgError("test convergence failure")
    monkeypatch.setattr(torch.linalg, "svd", fail_svd)
    with pytest.raises(NumericalDomainViolation, match="SVD could not converge"):
        solve_weighted(torch.eye(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64), 1)


@pytest.mark.parametrize("needs_symmetrization", [False, True])
def test_minimum_subnormal_metric_preserves_support_and_correct_rank_choice(needs_symmetrization):
    weight = torch.diag(torch.tensor([2., 1.], dtype=torch.float64))
    moment = torch.diag(torch.tensor([5e-324, 1e-323], dtype=torch.float64))
    if needs_symmetrization:
        # A tolerated asymmetry whose mean rounds to zero; diagonal values
        # must survive the actual symmetrization as well as the exact bypass.
        moment[0, 1] = 5e-324
    before = moment.clone()
    result = solve_weighted(weight, moment, 1)
    expected = torch.diag(torch.tensor([2., 0.], dtype=torch.float64))
    torch.testing.assert_close(result.approximation, expected, atol=1e-15, rtol=1e-15)
    assert result.moment_rank == 2
    residual = weight - result.approximation
    # Multiplying the quadratic form directly avoids relying on squared norms
    # near underflow. Default approximate-comparison tolerances would hide this.
    direct_error = float(torch.trace(residual @ moment @ residual.T))
    wrong_residual = weight - torch.diag(torch.tensor([0., 1.], dtype=torch.float64))
    wrong_error = float(torch.trace(wrong_residual @ moment @ wrong_residual.T))
    assert direct_error == 1e-323
    assert wrong_error == 2e-323 and direct_error < wrong_error
    assert result.weighted_error_squared == direct_error
    assert torch.equal(moment, before)
