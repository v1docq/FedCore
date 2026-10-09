"""Pure, explicitly typed calibration reductions and their provenance.

Rows are observations. Accumulators own detached float64 tensors, contain no
modules/hooks, and updates never mutate their input states. A shell owns data
selection, forward/backward passes and cleanup. Empty/zero-mass states cannot
be finalized. Counts are exact integers; weighted sums use float64 arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import torch


@dataclass(frozen=True)
class StatisticsPassport:
    checkpoint_id: str
    graph_version: str
    predecessor_version: str
    observation_point: str
    data_id: str
    data_role: str = "calibration"
    mask_id: str = ""
    weight_id: str = ""
    normalization: str = "weight_sum"
    count: int = 0
    weight_sum: float = 0.0
    dtype: str = "float64"
    method_version: str = "1"
    weight_sum_rtol: float = 1e-12
    weight_sum_atol: float = 1e-12


@dataclass(frozen=True)
class StaleStatistics:
    reason: str
    mismatches: tuple[str, ...] = ()


def validate_passport(actual, expected):
    """Conservative cache reuse: all provenance must match, even graph version.

    Counts/mass describe the result, not the reuse key. A predecessor change
    therefore invalidates a previously collected input moment before reuse.
    """
    names = ("checkpoint_id", "graph_version", "predecessor_version", "observation_point",
             "data_id", "data_role", "mask_id", "weight_id", "normalization", "dtype", "method_version",
             "weight_sum_rtol", "weight_sum_atol")
    mismatches = tuple(name for name in names if getattr(actual, name) != getattr(expected, name))
    return StaleStatistics("Incompatible statistics provenance", mismatches) if mismatches else None


def _passport(passport):
    if not isinstance(passport, StatisticsPassport):
        raise ValueError("StatisticsPassport required")
    names = ("checkpoint_id", "graph_version", "predecessor_version", "observation_point", "data_id",
             "data_role", "mask_id", "weight_id", "normalization", "dtype", "method_version")
    if any(not isinstance(getattr(passport, name), str) for name in names):
        raise ValueError("Passport provenance requires string identifiers, never live resources")
    if passport.data_role not in ("calibration", "train"):
        raise ValueError("test/validation data cannot calibrate a statistic")
    if passport.dtype != "float64" or passport.normalization != "weight_sum":
        raise ValueError("Exact statistics use float64 and explicit weight_sum normalization")
    if type(passport.count) is not int or passport.count < 0:
        raise ValueError("Observation count must be a nonnegative integer")
    if type(passport.weight_sum) not in (int, float) or not math.isfinite(passport.weight_sum) or passport.weight_sum < 0:
        raise ValueError("Observation mass must be finite and nonnegative")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
           for value in (passport.weight_sum_rtol, passport.weight_sum_atol)):
        raise ValueError("Weight-sum comparison tolerances must be finite and nonnegative")


def _rows_weights(rows, weights=None):
    if (not isinstance(rows, torch.Tensor) or rows.ndim != 2 or rows.dtype not in (torch.float32, torch.float64)
            or rows.shape[1] == 0 or not bool(torch.isfinite(rows).all())):
        raise ValueError("Observations require finite float32/float64 rows")
    x = rows.detach().to(torch.float64)
    if weights is None:
        w = torch.ones(len(x), dtype=torch.float64, device=x.device)
    else:
        if (not isinstance(weights, torch.Tensor) or weights.ndim != 1 or len(weights) != len(x)
                or weights.dtype == torch.bool or weights.is_complex()):
            raise ValueError("One real weight per observation required")
        w = weights.detach().to(device=x.device, dtype=torch.float64)
        if not bool(torch.isfinite(w).all()) or bool((w < 0).any()):
            raise ValueError("Weights must be finite and nonnegative")
    mass = float(w.sum())
    if not math.isfinite(mass):
        raise ValueError("Weight sum must be finite")
    return x, w, mass


def _compatible(left, right):
    stale = validate_passport(left.passport, right.passport)
    if stale:
        raise ValueError(f"StaleStatistics: {','.join(stale.mismatches)}")
    if left.gram.shape != right.gram.shape or left.gram.device != right.gram.device:
        raise ValueError("Accumulator dimensions/devices must agree")


def _finished_passport(state):
    if state.weight_sum <= 0 or not math.isfinite(state.weight_sum):
        raise ValueError("Cannot finalize empty or zero-mass statistics")
    return replace(state.passport, count=state.count, weight_sum=state.weight_sum)


@dataclass(frozen=True)
class SecondMomentAccumulator:
    passport: StatisticsPassport
    gram: torch.Tensor
    count: int = 0
    weight_sum: float = 0.0


@dataclass(frozen=True)
class InputSecondMoment:
    matrix: torch.Tensor
    passport: StatisticsPassport


def empty_second_moment(dimension, passport, device="cpu"):
    _passport(passport)
    if type(dimension) is not int or dimension <= 0:
        raise ValueError("Feature dimension must be a positive integer")
    return SecondMomentAccumulator(passport, torch.zeros((dimension, dimension), dtype=torch.float64, device=device))


def update_second_moment(state, rows, weights=None):
    x, w, mass = _rows_weights(rows, weights)
    if state.gram.shape != (x.shape[1], x.shape[1]) or state.gram.device != x.device:
        raise ValueError("Observation dimension/device does not match the accumulator")
    gram = state.gram + x.T @ (x * w.unsqueeze(1))
    if not bool(torch.isfinite(gram).all()) or not math.isfinite(state.weight_sum + mass):
        raise ValueError("Statistics overflow")
    return SecondMomentAccumulator(state.passport, gram, state.count + len(x), state.weight_sum + mass)


def merge_second_moment(left, right):
    _compatible(left, right)
    gram = left.gram + right.gram
    mass = left.weight_sum + right.weight_sum
    if not bool(torch.isfinite(gram).all()) or not math.isfinite(mass):
        raise ValueError("Statistics overflow")
    return SecondMomentAccumulator(left.passport, gram, left.count + right.count, mass)


def finish_second_moment(state):
    return InputSecondMoment(state.gram / state.weight_sum if state.weight_sum > 0 else state.gram.clone(), _finished_passport(state))


@dataclass(frozen=True)
class CenteredAccumulator:
    passport: StatisticsPassport
    mean: torch.Tensor
    gram: torch.Tensor  # centered scatter, not an uncentered moment
    count: int = 0
    weight_sum: float = 0.0


@dataclass(frozen=True)
class CenteredMoments:
    mean: torch.Tensor
    covariance: torch.Tensor
    passport: StatisticsPassport


def empty_centered_moments(dimension, passport, device="cpu"):
    base = empty_second_moment(dimension, passport, device)
    return CenteredAccumulator(passport, torch.zeros(dimension, dtype=torch.float64, device=device), base.gram)


def merge_centered_moments(left, right):
    """Weighted Chan merge; avoids cancellation in E[x²]-E[x]²."""
    _compatible(left, right)
    mass = left.weight_sum + right.weight_sum
    if not math.isfinite(mass):
        raise ValueError("Statistics overflow")
    delta = right.mean - left.mean
    if mass == 0:
        mean, scatter = left.mean.clone(), left.gram + right.gram
    else:
        mean = left.mean + delta * (right.weight_sum / mass)
        # Stable form avoids multiplying two large masses first.
        scatter = left.gram + right.gram + torch.outer(delta, delta) * (left.weight_sum * (right.weight_sum / mass))
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(scatter).all()):
        raise ValueError("Statistics overflow")
    return CenteredAccumulator(left.passport, mean, scatter, left.count + right.count, mass)


def update_centered_moments(state, rows, weights=None):
    x, w, mass = _rows_weights(rows, weights)
    if state.mean.shape != (x.shape[1],) or state.mean.device != x.device:
        raise ValueError("Observation dimension/device does not match the accumulator")
    if mass:
        mean = (x * (w / mass).unsqueeze(1)).sum(0)
        centered = x - mean
        scatter = centered.T @ (centered * w.unsqueeze(1))
    else:
        mean, scatter = state.mean.new_zeros(state.mean.shape), state.gram.new_zeros(state.gram.shape)
    return merge_centered_moments(state, CenteredAccumulator(state.passport, mean, scatter, len(x), mass))


def finish_centered_moments(state):
    passport = _finished_passport(state)
    return CenteredMoments(state.mean.clone(), state.gram / state.weight_sum, passport)


# Distinct schema obligations for later methods. These are statistical values,
# not implementations or registrations of the corresponding approximation.
@dataclass(frozen=True)
class WithinGroupCovariance:
    group_ids: tuple[str, ...]
    group_weights: tuple[float, ...]
    group_moments: tuple[CenteredMoments, ...]
    covariance: torch.Tensor
    passport: StatisticsPassport


def finish_within_group_covariance(groups):
    """Average within-group covariance, explicitly excluding between-group means."""
    if not groups:
        raise ValueError("At least one named group is required")
    ids, moments = zip(*groups)
    if len(set(ids)) != len(ids):
        raise ValueError("Group identifiers must be unique")
    finished = tuple(finish_centered_moments(state) for state in moments)
    for state in moments[1:]:
        _compatible(moments[0], state)
    mass = math.fsum(item.passport.weight_sum for item in finished)
    weights = tuple(item.passport.weight_sum / mass for item in finished)
    covariance = sum((item.covariance * weight for item, weight in zip(finished, weights)), torch.zeros_like(finished[0].covariance))
    passport = replace(finished[0].passport, count=sum(item.passport.count for item in finished), weight_sum=mass)
    return WithinGroupCovariance(tuple(ids), weights, finished, covariance, passport)


@dataclass(frozen=True)
class AbsStatistic:
    mean_absolute: torch.Tensor
    passport: StatisticsPassport


@dataclass(frozen=True)
class Frequencies:
    frequencies: torch.Tensor
    passport: StatisticsPassport


@dataclass(frozen=True)
class IndividualGradientSquares:
    mean_square: torch.Tensor
    passport: StatisticsPassport
    loss_id: str
    labels_id: str


@dataclass(frozen=True)
class NormalizedObservations:
    second_moment: torch.Tensor
    passport: StatisticsPassport
    normalization: str = "per_observation_unit_l2"


def _vector_mean(rows, passport, weights=None):
    _passport(passport)
    x, w, mass = _rows_weights(rows, weights)
    if mass <= 0:
        raise ValueError("Cannot finalize empty or zero-mass statistics")
    value = (x * (w / mass).unsqueeze(1)).sum(0)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Statistics overflow")
    return value, replace(passport, count=len(x), weight_sum=mass)


def absolute_statistic(rows, passport, weights=None):
    # Validate the original dtype before abs can turn complex rows into real.
    x, _, _ = _rows_weights(rows, weights)
    value, finished = _vector_mean(x.abs(), passport, weights)
    return AbsStatistic(value, finished)


def individual_gradient_squares(gradient_rows, passport, *, loss_id, labels_id, weights=None):
    # Square each observation before aggregation: g/-g cannot cancel.
    _matrix_rows, _, _ = _rows_weights(gradient_rows, weights)
    value, finished = _vector_mean(_matrix_rows.square(), passport, weights)
    return IndividualGradientSquares(value, finished, loss_id, labels_id)


def frequency_statistic(indicator_rows, passport, weights=None):
    if not isinstance(indicator_rows, torch.Tensor) or indicator_rows.ndim != 2 or indicator_rows.dtype == torch.bool or indicator_rows.is_complex():
        raise ValueError("Frequency observations require real count rows")
    if bool((indicator_rows < 0).any()) or not bool((indicator_rows == indicator_rows.round()).all()):
        raise ValueError("Frequency observations must be nonnegative integer counts")
    value, finished = _vector_mean(indicator_rows.to(torch.float64), passport, weights)
    return Frequencies(value, finished)


def normalized_observations(rows, passport, weights=None):
    x, w, _ = _rows_weights(rows, weights)
    # Computing sum(x*x) directly can overflow or underflow for finite rows.
    # Normalize bounded ratios instead; the largest ratio has magnitude one.
    scales = x.abs().amax(dim=1, keepdim=True)
    if bool((scales == 0).any()):
        raise ValueError("Zero rows do not have a unit-L2 normalization")
    scaled = x / scales
    norms = torch.linalg.vector_norm(scaled, dim=1, keepdim=True)
    if not bool(torch.isfinite(norms).all()) or bool((norms == 0).any()):
        raise ValueError("Finite nonzero row norms are required for unit-L2 normalization")
    state = update_second_moment(empty_second_moment(x.shape[1], passport, x.device), scaled / norms, w)
    result = finish_second_moment(state)
    return NormalizedObservations(result.matrix, result.passport)


@dataclass(frozen=True)
class MemoryPlan:
    """Conservative named live-buffer estimate, not a process RSS guarantee.

    model_bytes includes the original and work copy supplied by the shell.
    replacement_bytes includes already prepared factors. Numerical workspace
    reserves six dense FP64 matrices per group, plus eigenvalues. Framework/
    allocator overhead and model-forward activations are explicit caller inputs.
    """
    model_bytes: int
    persistent_bytes: int
    collection_bytes: int
    transfer_bytes: int
    eigensolve_bytes: int
    factorization_bytes: int
    replacement_bytes: int
    forward_bytes: int
    runtime_overhead_bytes: int

    @property
    def peak_bytes(self):
        base = self.model_bytes + self.persistent_bytes + self.replacement_bytes + self.runtime_overhead_bytes
        return base + max(self.collection_bytes + self.transfer_bytes + self.forward_bytes,
                          self.eigensolve_bytes + self.factorization_bytes)


@dataclass(frozen=True)
class ResourceLimitExceeded:
    required_bytes: int
    limit_bytes: int
    phase: str = "statistics_and_eigensolve"


def plan_statistics_memory(dimension, observation_rows, *, groups=1, model_bytes=0,
                           replacement_bytes=0, forward_bytes=0, runtime_overhead_bytes=0,
                           max_peak_bytes=None, operator_rows=0):
    """Plan before allocation; refuse an infeasible declared live-buffer peak.

    Uses FP64 even for FP32 observations. Collection reserves input conversion,
    a weighted row copy, a new Gram and the still-live previous Gram. Eigensolve
    reserves roots/pseudoinverse/vectors/workspace alongside persistent moments.
    ``operator_rows`` is the output dimension of one group. It reserves promoted
    W, WF, the approximation and rectangular-SVD intermediates/workspace. Zero
    means only the statistics phase is described, not an operator SVD limit.
    """
    values = (dimension, observation_rows, groups, model_bytes, replacement_bytes, forward_bytes, runtime_overhead_bytes, operator_rows)
    if any(type(value) is not int or value < 0 for value in values) or min(dimension, groups) <= 0:
        raise ValueError("Memory dimensions/costs must be nonnegative integers; dimension/groups positive")
    if max_peak_bytes is not None and (type(max_peak_bytes) is not int or max_peak_bytes <= 0):
        raise ValueError("Memory limit must be a positive integer")
    matrix = groups * dimension * dimension * 8
    rows = groups * observation_rows * dimension * 8
    minimum = min(operator_rows, dimension)
    factorization = groups * 8 * (8 * operator_rows * dimension + 4 * minimum * minimum
                                   + 2 * (operator_rows + dimension) * minimum + minimum)
    plan = MemoryPlan(model_bytes, matrix, 2 * matrix + 2 * rows + groups * observation_rows * 8,
                      rows, 6 * matrix + groups * dimension * 8, factorization,
                      replacement_bytes, forward_bytes, runtime_overhead_bytes)
    if max_peak_bytes is not None and plan.peak_bytes > max_peak_bytes:
        return ResourceLimitExceeded(plan.peak_bytes, max_peak_bytes)
    return plan
