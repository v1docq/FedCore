"""Explicit CPU Linear statistical objectives, without runners or registries.

All operators use PyTorch's W[out, in] convention. Statistics and solves own
detached float64 values; installed factors retain W's float32/float64 dtype.
The diagonal profiles allocate O(out*in + in) space, never an in-by-in
diagonal matrix. PCA and Ledoit--Wolf require dense covariance/eigensolve
space O(d**2), with O(n*d) detached calibration rows for the latter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
import statistics as python_statistics

import torch

from .allocation import (BudgetInfeasible, CandidateOption, CandidateTable,
                         ScoreObjective, StorageCost, allocate)
from .approximation import (ApproximationResult, NumericalDomainViolation,
                            _finite, _matrix, _rank, _svd, solve_weighted)
from .plans import MetricPolicy
from .statistics import (CenteredMoments, StatisticsPassport,
                         WithinGroupCovariance, _passport, _rows_weights,
                         empty_centered_moments, finish_centered_moments,
                         update_centered_moments)


ASVD_SOURCE = "https://arxiv.org/abs/2312.05821"
FWSVD_SOURCE = "https://arxiv.org/html/2207.00112v1#S4"
AFM_SOURCE = "https://cs.nju.edu.cn/wujx/paper/AAAI2023_AFM.pdf"
BOLACO_SOURCE = "https://arxiv.org/html/2405.10616v2#S3.SS1"
FLAR_SOURCE = "https://openaccess.thecvf.com/content/CVPR2025W/MAI/papers/Thoma_FLAR-SVD_Fast_and_Latency-Aware_Singular_Value_Decomposition_for_Model_Compression_CVPRW_2025_paper.pdf"
LEDOIT_WOLF_REFERENCE = "https://github.com/scikit-learn/scikit-learn/blob/1.5.2/sklearn/covariance/_shrunk_covariance.py"


def _cpu_matrix(value, name):
    _matrix(value, name)
    if value.device.type != "cpu":
        raise NumericalDomainViolation("The initial statistical profile requires CPU tensors")


def _number(value, name, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise NumericalDomainViolation(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")


def _rank_request(matrix, rank):
    _cpu_matrix(matrix, "weight")
    if type(rank) is not int or not 1 <= rank <= min(matrix.shape):
        raise NumericalDomainViolation("rank must fit both operator axes")


def _selected_rows(rows, passport, weights=None, mask=None):
    _passport(passport)
    x, w, _ = _rows_weights(rows, weights)
    if x.device.type != "cpu":
        raise NumericalDomainViolation("The initial statistical profile requires CPU observations")
    if mask is not None:
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != (len(x),) or mask.device != x.device:
            raise NumericalDomainViolation("mask must contain one CPU boolean per observation")
        x, w = x[mask], w[mask]
    if not len(x) or float(w.sum()) <= 0:
        raise NumericalDomainViolation("At least one positive-mass calibration observation is required")
    return x, w, replace(passport, count=len(x), weight_sum=float(w.sum()))


@dataclass(frozen=True)
class ChannelAbsStats:
    values: torch.Tensor
    passport: StatisticsPassport
    mode: str
    axis: str = "Linear.weight_input_columns"


def channel_abs_statistics(rows, passport, *, mode="abs_mean", weights=None, mask=None):
    """Reduce absolute activations; a zero-weight row is absent from abs_max.

    Masking selects observations before reduction. abs_mean uses supplied
    nonnegative weights; abs_max uses their positive support, not magnitudes.
    The passport's mask_id/weight_id identify the caller's selection policies.
    """
    if mode not in ("abs_mean", "abs_max"):
        raise NumericalDomainViolation("ASVD statistic must be abs_mean or abs_max")
    x, w, finished = _selected_rows(rows, passport, weights, mask)
    values = ((x.abs() * (w / finished.weight_sum).unsqueeze(1)).sum(0)
              if mode == "abs_mean" else x[w > 0].abs().amax(0))
    return ChannelAbsStats(_finite(values, "channel absolute statistics"), finished, mode)


@dataclass(frozen=True)
class GradientCollectionMetadata:
    loss_id: str
    labels_id: str
    example_unit: str = "batch_item"
    reduction: str = "mean"
    masking: str = "none"
    model_mode: str = "eval"


def validate_gradient_metadata(metadata):
    if not isinstance(metadata, GradientCollectionMetadata):
        raise NumericalDomainViolation("Explicit gradient collection metadata required")
    if any(not isinstance(getattr(metadata, name), str) or not getattr(metadata, name)
           for name in ("loss_id", "labels_id", "example_unit", "masking")):
        raise NumericalDomainViolation("Loss, observed labels, example unit and masking require identifiers")
    if metadata.example_unit != "batch_item" or metadata.reduction not in ("mean", "sum") or metadata.model_mode not in ("eval", "train"):
        raise NumericalDomainViolation("Supported collection uses batch_item, mean/sum reduction and eval/train mode")


@dataclass(frozen=True)
class EmpiricalGradientSquares:
    mean_square: torch.Tensor  # [out, in], individual squares before expectation
    passport: StatisticsPassport
    metadata: GradientCollectionMetadata
    collection_manifest: dict
    estimator: str = "observed_label_empirical_gradient_squares"


def fwsvd_statistics(mean_square, passport, metadata, *, collection_manifest=None):
    _cpu_matrix(mean_square, "individual gradient mean squares")
    _passport(passport)
    validate_gradient_metadata(metadata)
    if bool((mean_square < 0).any()) or passport.count <= 0 or passport.weight_sum <= 0:
        raise NumericalDomainViolation("Gradient squares require nonnegative values and positive observation mass")
    try:
        diagnostics = dict(collection_manifest or {})
        json.dumps(diagnostics, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise NumericalDomainViolation("Gradient collection diagnostics must be finite JSON data") from error
    return EmpiricalGradientSquares(mean_square.detach().double().clone(), passport, metadata, diagnostics)


@dataclass(frozen=True)
class StatisticalApproximation:
    factors: ApproximationResult
    bias: torch.Tensor | None
    manifest: dict

    @property
    def left(self):
        """Second Linear weight [out, rank]; first Linear weight is right."""
        return self.factors.u * self.factors.s

    @property
    def right(self):
        return self.factors.vh

    @property
    def approximation(self):
        return self.factors.approximation


def _canonical(matrix, approximation, rank, *, metric_rank, error_fn, objective,
               nullspace_policy, condition_number=None):
    u, s, vh = _svd(approximation, "returned statistical operator")
    u, s, vh = u[:, :rank].to(matrix), s[:rank].to(matrix), vh[:rank].to(matrix)
    installed = _finite((u * s) @ vh, "installed statistical operator")
    actual_rank = _rank(installed)
    if actual_rank > rank:
        raise NumericalDomainViolation("output precision violates the requested rank")
    error = float(error_fn(installed.double()))
    if not math.isfinite(error) or error < 0:
        raise NumericalDomainViolation("Statistical objective exceeds the finite numerical domain")
    return ApproximationResult(u, s, vh, installed, rank, actual_rank, metric_rank,
                               error, 0., 0., nullspace_policy, condition_number,
                               0., 0., objective=objective)


def _solve_input_scales(matrix, scales, rank, *, objective, zero_policy):
    """SVD of W*S with vector broadcasts and an explicit support inverse."""
    _rank_request(matrix, rank)
    if (not isinstance(scales, torch.Tensor) or scales.shape != (matrix.shape[1],)
            or scales.device.type != "cpu" or not bool(torch.isfinite(scales).all())
            or bool((scales < 0).any())):
        raise NumericalDomainViolation("One finite nonnegative scale per input column required")
    if zero_policy not in ("support_only", "preserve_nullspace", "error", "reject"):
        raise NumericalDomainViolation("Unknown zero-channel policy")
    support = scales > 0
    if zero_policy in ("error", "reject") and not bool(support.all()):
        raise NumericalDomainViolation("Zero channel rejected by the declared policy")
    with torch.no_grad():
        w, scales = matrix.detach().double(), scales.detach().double()
        if rank == min(matrix.shape):
            approximation, policy = w.clone(), "preserve_operator_at_full_rank"
        else:
            u, s, vh = _svd(_finite(w * scales, "diagonally scaled operator"), "diagonally scaled operator")
            inverse = torch.zeros_like(scales)
            inverse[support] = scales[support].reciprocal()
            approximation = _finite(((u[:, :rank] * s[:rank]) @ vh[:rank]) * inverse,
                                    "inverse-scaled operator")
            policy = zero_policy
            if zero_policy == "preserve_nullspace":
                approximation[:, ~support] = w[:, ~support]
                if _rank(approximation) > rank:
                    raise NumericalDomainViolation("Preserving zero-channel action exceeds the requested rank")
        # Diagnostics concern the diagonal *metric*, whose eigenvalues are s².
        retained = scales[support]
        condition = float((retained.max() / retained.min()).square()) if retained.numel() else None
        if condition is not None and not math.isfinite(condition):
            raise NumericalDomainViolation("Scale condition number exceeds the finite numerical domain")
        return _canonical(matrix, approximation, rank, metric_rank=int(support.sum()),
                          error_fn=lambda a: _finite((w - a) * scales, "diagonal residual").square().sum(),
                          objective=objective, nullspace_policy=policy, condition_number=condition)


def solve_asvd(matrix, stats, rank, *, alpha=.5, epsilon=0., zero_policy="support_only"):
    """Minimize sum_j (abs_stat_j + epsilon)**(2*alpha) ||W[:,j]-A[:,j]||².

    alpha=0 deliberately gives all scales one, including zero channels (0**0).
    epsilon is added before the power and changes the declared objective.
    """
    if not isinstance(stats, ChannelAbsStats) or stats.mode not in ("abs_mean", "abs_max"):
        raise NumericalDomainViolation("ChannelAbsStats required; Fisher is a separate estimator")
    _passport(stats.passport)
    _number(alpha, "alpha")
    _number(epsilon, "epsilon")
    if (not isinstance(stats.values, torch.Tensor) or stats.values.dtype not in (torch.float32, torch.float64)
            or stats.values.device.type != "cpu" or stats.values.shape != (matrix.shape[1],)
            or not bool(torch.isfinite(stats.values).all()) or bool((stats.values < 0).any())):
        raise NumericalDomainViolation("Absolute statistics require one finite nonnegative value per input feature")
    values = stats.values.detach().double()
    scales = _finite((values + epsilon).pow(alpha), "ASVD scales")
    factors = _solve_input_scales(matrix, scales, rank, objective="asvd_diagonal", zero_policy=zero_policy)
    return StatisticalApproximation(factors, None, {
        "method": "asvd", "objective": "diagonal_input_column_error", "source": ASVD_SOURCE,
        "statistic": stats.mode, "alpha": float(alpha), "epsilon": float(epsilon),
        "epsilon_placement": "before_power", "zero_policy": zero_policy,
        "zero_channels": (values == 0).nonzero().flatten().tolist(),
        "scales": scales.tolist(), "statistic_values": values.tolist(),
        "statistics_passport": asdict(stats.passport), "channel_correlations": "not_estimated",
        "abs_max_weights": "positive_support_only", "full_rank_policy": factors.nullspace_policy})


def solve_fwsvd(matrix, stats, rank, *, epsilon=0., zero_policy="support_only"):
    """Solve the paper's coarsened diagonal objective, not elementwise Fisher.

    Paper W[in,out] row sums become sum(dim=0) for torch W[out,in].
    The scale is sqrt(sum_out E[g_oi²] + epsilon), never sum of gradients.
    """
    if not isinstance(stats, EmpiricalGradientSquares):
        raise NumericalDomainViolation("EmpiricalGradientSquares required; abs statistics are not Fisher")
    _cpu_matrix(stats.mean_square, "empirical gradient squares")
    _passport(stats.passport)
    validate_gradient_metadata(stats.metadata)
    if stats.mean_square.shape != matrix.shape or bool((stats.mean_square < 0).any()):
        raise NumericalDomainViolation("Individual gradient squares must match W[out,in]")
    _number(epsilon, "epsilon")
    importance = _finite(stats.mean_square.sum(dim=0), "Fisher input-column aggregation")
    scales = _finite((importance + epsilon).sqrt(), "Fisher scales")
    factors = _solve_input_scales(matrix, scales, rank, objective="fwsvd_coarsened_diagonal", zero_policy=zero_policy)
    return StatisticalApproximation(factors, None, {
        "method": "fwsvd", "source": FWSVD_SOURCE, "estimator": stats.estimator,
        "aggregation": "sum_over_output_axis_dim0", "paper_weight_convention": "W[in,out]",
        "weight_convention": "Linear.weight[out,in]", "objective": "coarsened_input_diagonal_error",
        "elementwise_fisher_optimum": False, "full_fisher": False, "hessian": False,
        "epsilon": float(epsilon), "epsilon_placement": "after_aggregation_before_sqrt",
        "zero_policy": zero_policy, "zero_channels": (importance == 0).nonzero().flatten().tolist(),
        "importance": importance.tolist(), "scales": scales.tolist(),
        "gradient_metadata": asdict(stats.metadata), "statistics_passport": asdict(stats.passport),
        "collection": stats.collection_manifest})


def _checked_covariance(covariance, dimension):
    _cpu_matrix(covariance, "centered covariance")
    if covariance.shape != (dimension, dimension):
        raise NumericalDomainViolation("Centered covariance must match the output dimension")
    c = covariance.detach().double()
    tolerance = torch.finfo(torch.float64).eps * dimension * max(1., float(c.abs().max())) * 16
    if not torch.allclose(c, c.T, atol=tolerance, rtol=0):
        raise NumericalDomainViolation("Centered covariance must be symmetric")
    try:
        values, vectors = torch.linalg.eigh(c if torch.equal(c, c.T) else c*.5 + c.T*.5)
    except torch.linalg.LinAlgError as error:
        raise NumericalDomainViolation("Centered covariance eigensolve failed") from error
    if not bool(torch.isfinite(values).all()) or float(values.min()) < -tolerance:
        raise NumericalDomainViolation("Centered covariance must be finite positive semidefinite")
    return c, values.clamp_min(0), _finite(vectors, "PCA vectors")


def solve_affine_pca(matrix, bias, moments, rank):
    """Output PCA: W_hat=P W, b_hat=P b+(I-P) mean_output.

    WithinGroupCovariance selects a common subspace using within-group scatter;
    its one shared intercept uses the weighted global mean. The reported
    within-group objective excludes between-group mean error and therefore is
    explicitly not the total output MSE of that shared affine operator.
    """
    _rank_request(matrix, rank)
    if not isinstance(moments, (CenteredMoments, WithinGroupCovariance)):
        raise NumericalDomainViolation("Centered output moments or named within-group moments required")
    _passport(moments.passport)
    out = matrix.shape[0]
    if isinstance(moments, CenteredMoments):
        mean, objective = moments.mean, "affine_output_centered_pca"
        grouping = {}
    else:
        if (not moments.group_moments or len(moments.group_moments) != len(moments.group_weights)
                or len(moments.group_ids) != len(moments.group_weights)
                or len(set(moments.group_ids)) != len(moments.group_ids)
                or any(not isinstance(key, str) or not key for key in moments.group_ids)):
            raise NumericalDomainViolation("Within-group moments require unique named groups and weights")
        if any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in moments.group_weights) or not math.isclose(math.fsum(moments.group_weights), 1., rel_tol=1e-12, abs_tol=1e-12):
            raise NumericalDomainViolation("Within-group weights must be positive and sum to one")
        mean = sum((item.mean * weight for item, weight in zip(moments.group_moments, moments.group_weights)), torch.zeros(out, dtype=torch.float64))
        objective = "affine_output_within_group_pca"
        grouping = {"group_ids": list(moments.group_ids), "group_weights": list(moments.group_weights),
                    "group_means": [item.mean.tolist() for item in moments.group_moments],
                    "between_group_scatter": "excluded", "shared_intercept": "weighted_global_mean",
                    "covariance_normalization": "within_group_weight_sum",
                    "pooling_variant": "declared_weighted_ml_within_group_covariance",
                    "literal_equal_group_unbiased_paper_estimator": False}
    if not isinstance(mean, torch.Tensor) or mean.shape != (out,) or not bool(torch.isfinite(mean).all()):
        raise NumericalDomainViolation("One finite output mean per output feature required")
    if bias is not None and (not isinstance(bias, torch.Tensor) or bias.shape != (out,) or bias.dtype != matrix.dtype or bias.device != matrix.device or not bool(torch.isfinite(bias).all())):
        raise NumericalDomainViolation("bias must match output shape, weight precision and device")
    with torch.no_grad():
        c, values, vectors = _checked_covariance(moments.covariance, out)
        basis = vectors[:, -rank:]
        projector = basis @ basis.T
        w = matrix.detach().double()
        original_bias = torch.zeros(out, dtype=torch.float64) if bias is None else bias.detach().double()
        # The maximum admissible rank preserves even unobserved operator action.
        full = rank == min(matrix.shape)
        approximation = w.clone() if full else projector @ w
        projected_bias = original_bias.clone() if full else projector @ original_bias + mean.double() - projector @ mean.double()
        output_bias = _finite(projected_bias.to(matrix), "affine output bias")
        error = 0. if full else float(values[:-rank].sum())
        factors = _canonical(matrix, approximation, rank, metric_rank=int((values > 0).sum()),
                             error_fn=lambda _: error, objective=objective,
                             nullspace_policy="preserve_operator_at_full_rank" if full else "output_projection")
        return StatisticalApproximation(factors, output_bias, {
            "method": "bolaco" if isinstance(moments, WithinGroupCovariance) else "afm",
            "objective": objective, "source": BOLACO_SOURCE if grouping else AFM_SOURCE, "centering": "output_mean",
            "mean": mean.double().tolist(), "eigenvalues": values.tolist(),
            "statistics_passport": asdict(moments.passport),
            "expected_error_semantics": "centered_subspace_reconstruction_error",
            "pointwise_output_guarantee": False, "bias_added_if_needed": bias is None and output_bias is not None,
            "full_rank_policy": factors.nullspace_policy, **grouping})


@dataclass(frozen=True)
class CenteredShrinkageMetric:
    mean: torch.Tensor
    covariance: torch.Tensor
    empirical_covariance: torch.Tensor
    shrinkage: float
    target_variance: float
    passport: StatisticsPassport
    formula_version: str = "ledoit_wolf_2004_centered_ml_v1"
    reference: str = LEDOIT_WOLF_REFERENCE


def ledoit_wolf_metric(rows, passport, *, mask=None):
    """Verified unweighted centered ML estimator (division by n, not n-1).

    beta = sum_i ||x_i x_i.T-C||_F²/n²;
    delta = ||C-tr(C)/d*I||_F²; alpha=min(beta/delta,1).
    The norm expansion avoids n*d*d outer-product buffers. Nonnegative clamps
    only remove roundoff in mathematically nonnegative beta and delta; they are
    not ridge regularization. Zero variance remains the zero PSD metric.
    """
    x, _, finished = _selected_rows(rows, passport, mask=mask)
    state = update_centered_moments(empty_centered_moments(x.shape[1], passport), x)
    centered = finish_centered_moments(state)
    z = x - centered.mean
    c = _finite(centered.covariance, "empirical centered covariance")
    n, dimension = x.shape
    target = float(c.diagonal().mean())
    squared_covariance = float(c.square().sum())
    fourth_sum = float(z.square().sum(1).square().sum())
    beta = max(0., (fourth_sum / n - squared_covariance) / n)
    deviation = c.clone()
    deviation.diagonal().sub_(target)
    delta = float(deviation.square().sum())
    if not all(math.isfinite(v) for v in (target, beta, delta, fourth_sum, squared_covariance)):
        raise NumericalDomainViolation("Ledoit-Wolf fourth moments exceed the finite float64 domain")
    alpha = 0. if dimension == 1 or delta == 0 else min(beta / delta, 1.)
    shrunk = c * (1. - alpha)
    shrunk.diagonal().add_(alpha * target)
    return CenteredShrinkageMetric(centered.mean, _finite(shrunk, "shrunk covariance"), c.clone(),
                                   float(alpha), target, finished)


def solve_flar(matrix, metric, rank, *, policy=MetricPolicy()):
    """Centered/shrunk local objective, using the existing PSD checked solve.

    The affine intercept is retained by the caller. Centered error excludes
    the mean-input residual and is not uncentered output MSE. A zero covariance
    uses the explicit support/ridge policy; shrinkage alone does not promise SPD.
    """
    _rank_request(matrix, rank)
    if not isinstance(metric, CenteredShrinkageMetric):
        raise NumericalDomainViolation("Verified CenteredShrinkageMetric required")
    _passport(metric.passport)
    if metric.formula_version != "ledoit_wolf_2004_centered_ml_v1":
        raise NumericalDomainViolation("Unsupported or literal FLAR printed shrinkage formula")
    result = solve_weighted(matrix, metric.covariance, rank, policy)
    result = replace(result, objective="flar_centered_ledoit_wolf_ridge" if policy.ridge else "flar_centered_ledoit_wolf")
    return StatisticalApproximation(result, None, {
        "method": "flar", "source": FLAR_SOURCE, "objective": result.objective,
        "centering": "input_mean", "mean": metric.mean.tolist(),
        "shrinkage": metric.shrinkage, "target_variance": metric.target_variance,
        "formula_version": metric.formula_version, "formula_reference": metric.reference,
        "literal_printed_alpha": False, "uncentered_output_mse": False,
        "statistics_passport": asdict(metric.passport), "zero_variance": metric.target_variance == 0,
        "ridge": policy.ridge, "rcond": policy.rcond, "nullspace_policy": policy.nullspace_policy,
        "effective_metric_rank": result.moment_rank})


@dataclass(frozen=True)
class LatencyProfileKey:
    shape: tuple[int, int]
    batch: int
    dtype: str
    runtime: str
    runtime_version: str
    device: str
    graph_version: str
    source_version: str


@dataclass(frozen=True)
class RankMeasurement:
    rank: int
    quality: float
    predicted_latency_ms: float | None = None
    measured_samples_ms: tuple[float, ...] = ()
    artifact_id: str = ""

    @property
    def latency_ms(self):
        return (float(python_statistics.median(self.measured_samples_ms))
                if self.measured_samples_ms else self.predicted_latency_ms)


@dataclass(frozen=True)
class LatencyProfile:
    key: LatencyProfileKey
    rank_grid: tuple[int, ...]
    measurements: tuple[RankMeasurement, ...]
    source_id: str


@dataclass(frozen=True)
class LatencyRankSelection:
    measurement: RankMeasurement
    parameter_cost: int
    manifest: dict


def select_latency_rank(profile, expected_key, *, maximum_parameters,
                        bias_elements=0, maximum_latency_ms=None, minimum_quality=None,
                        score_direction="maximize"):
    """Finite grid enumeration with existing exact storage-budget allocation.

    No rank-to-quality or rank-to-latency monotonicity is assumed. Predictions
    retain their domain key; raw final-artifact measurements take precedence
    and can refute them. This function performs no hardware measurement.
    """
    if not isinstance(profile, LatencyProfile) or not isinstance(expected_key, LatencyProfileKey) or profile.key != expected_key:
        raise NumericalDomainViolation("Stale latency profile: shape/batch/precision/runtime/device/graph/source mismatch")
    key = profile.key
    if (not isinstance(key.shape, tuple) or len(key.shape) != 2 or any(type(v) is not int or v <= 0 for v in key.shape)
            or type(key.batch) is not int or key.batch <= 0
            or key.dtype not in ("float32", "float64")
            or any(not isinstance(getattr(key, name), str) or not getattr(key, name) for name in ("runtime", "runtime_version", "device", "graph_version", "source_version"))
            or not isinstance(profile.source_id, str) or not profile.source_id):
        raise NumericalDomainViolation("Latency profile requires complete finite validity-domain metadata")
    if type(maximum_parameters) is not int or maximum_parameters < 0 or type(bias_elements) is not int or bias_elements < 0:
        raise NumericalDomainViolation("Parameter budget and bias elements must be nonnegative integers")
    if score_direction not in ("maximize", "minimize"):
        raise NumericalDomainViolation("Quality score direction must be maximize or minimize")
    if maximum_latency_ms is not None:
        _number(maximum_latency_ms, "maximum latency", positive=True)
    if minimum_quality is not None and (type(minimum_quality) not in (int, float) or not math.isfinite(minimum_quality)):
        raise NumericalDomainViolation("Minimum quality must be finite")
    if (not profile.rank_grid or len(set(profile.rank_grid)) != len(profile.rank_grid)
            or any(type(r) is not int or not 1 <= r <= min(key.shape) for r in profile.rank_grid)
            or len({item.rank for item in profile.measurements}) != len(profile.measurements)
            or set(item.rank for item in profile.measurements) != set(profile.rank_grid)):
        raise NumericalDomainViolation("Every admissible integer grid rank requires exactly one finite evaluation")
    options, records = [], {}
    for item in profile.measurements:
        if (not isinstance(item, RankMeasurement) or type(item.quality) not in (int, float) or not math.isfinite(item.quality)
                or not isinstance(item.measured_samples_ms, tuple) or not isinstance(item.artifact_id, str)):
            raise NumericalDomainViolation("Malformed finite rank evaluation")
        if item.predicted_latency_ms is not None:
            _number(item.predicted_latency_ms, "predicted latency", positive=True)
        for sample in item.measured_samples_ms:
            _number(sample, "measured latency", positive=True)
        if item.latency_ms is None or (item.measured_samples_ms and not item.artifact_id):
            raise NumericalDomainViolation("Prediction or identified final-artifact samples required")
        if maximum_latency_ms is not None and item.latency_ms > maximum_latency_ms:
            continue
        if minimum_quality is not None and item.quality < minimum_quality:
            continue
        option_id = f"rank_{item.rank}"
        options.append(CandidateOption(option_id, (item.rank,), item.quality,
                                      (StorageCost(option_id, item.rank * sum(key.shape)),)))
        records[option_id] = item
    if not options:
        return BudgetInfeasible(maximum_parameters, min(profile.rank_grid) * sum(key.shape) + bias_elements,
                                "parameters", "No finite-grid candidate satisfies the latency/quality constraints")
    allocation = allocate((CandidateTable("flar_grid", tuple(options), maximum_ranks=(min(key.shape),)),),
                          maximum_parameters, objective=ScoreObjective("finite_evaluated_quality", score_direction),
                          fixed_storages=(StorageCost("original_bias", bias_elements, category="bias"),))
    if isinstance(allocation, BudgetInfeasible):
        return allocation
    selected = records[allocation.selected[0].option.option_id]
    return LatencyRankSelection(selected, allocation.total_cost, {
        "algorithm": "finite_grid_enumeration", "monotonicity_assumed": False,
        "validity_key": asdict(key), "source_id": profile.source_id,
        "rank_grid": list(profile.rank_grid), "evaluated_candidates": len(profile.measurements),
        "quality_semantics": "finite_supplied_model_evaluation", "quality_direction": score_direction,
        "selected_latency_ms": selected.latency_ms,
        "latency_evidence": "final_artifact_measurement" if selected.measured_samples_ms else "prediction",
        "raw_samples_ms": list(selected.measured_samples_ms), "artifact_id": selected.artifact_id,
        "predicted_latency_ms": selected.predicted_latency_ms, "parameter_cost": allocation.total_cost})
