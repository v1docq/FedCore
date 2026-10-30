"""Detached CPU algebra for ESPACE, PELA, LASER and centered CTR fields.

ESPACE follows arXiv:2410.05437v1, equations 6--10. GO is a bound;
NL additionally assumes first-order loss linearization. Normalized GO/NL
are calibration heuristics, not exact task-loss minimizers. All six criteria
use observation means (not means of unequal batch means).
"""
from dataclasses import dataclass, replace
import math
import torch

from .approximation import NumericalDomainViolation, _finite
from .statistics import (StatisticsPassport, _passport, empty_centered_moments,
                         update_centered_moments, finish_centered_moments)
from .structured_profiles import FactorPair, _matrix
from .projection_specs import ESPACE

ESPACE_SOURCE = "https://arxiv.org/html/2410.05437v1#S3.SS3"
PELA_SOURCE = "https://arxiv.org/html/2310.10700v2#S3"
LASER_SOURCE = "https://arxiv.org/html/2312.13558v1#S4"
CTR_SOURCE = "https://arxiv.org/html/2405.18146v2#S3"


def _cpu_matrix(value, name):
    _matrix(value, name)
    if value.device.type != "cpu":
        raise NumericalDomainViolation("P3 algebra requires CPU tensors")
    return value.detach().double().clone()


def _normalized(rows, zero_policy):
    norms = torch.linalg.vector_norm(rows, dim=1)
    zero = norms == 0
    if zero_policy == "error" and bool(zero.any()):
        raise NumericalDomainViolation("A normalized criterion encountered a zero vector")
    return rows / torch.where(zero, torch.ones_like(norms), norms)[:, None], int(zero.sum())


@dataclass(frozen=True)
class ESPACEStatistics:
    matrix: torch.Tensor
    criterion: str
    passport: StatisticsPassport
    zero_vectors: int
    objective_kind: str
    constant: float | None
    collection_manifest: dict


def espace_statistics(rows, passport, spec=ESPACE(), *, weight=None, gradients=None,
                      collection_manifest=None):
    """Construct one named criterion without an N-by-d-by-d temporary.

    GO averages uniformly over actual output rows of W[out,in]. Normalized
    GO normalizes each row separately before forming W.T@W/out. NL gradients
    are paired activation gradients, never gradients of averaged examples.
    Zero-contribution normalization preserves the original denominator N.
    """
    _passport(passport)
    if not isinstance(spec, ESPACE):
        raise NumericalDomainViolation("An ESPACE description is required")
    x = _cpu_matrix(rows, "activation rows")
    n, d = x.shape
    zeros = 0
    criterion = spec.criterion
    xn = x
    if criterion in ("nmse", "go_normalized", "nl_normalized"):
        xn, zeros = _normalized(x, spec.zero_policy)
    cx = xn.T @ xn / n
    kind, constant = "exact_input_squared_error", float(torch.trace(cx))
    if criterion.startswith("go_"):
        w = _cpu_matrix(weight, "W[out,in]")
        if w.shape[1] != d:
            raise NumericalDomainViolation("GO weights and input dimension disagree")
        if criterion == "go_normalized":
            w, count = _normalized(w, spec.zero_policy)
            zeros += count
            kind = "normalized_calibration_heuristic"
            constant = None
        else:
            kind = "gemm_output_mse_upper_bound"
            constant = 2 * float(x.square().sum(1).mean() * w.square().sum(1).mean())
        a = w.T @ w / len(w)
        h = cx @ a + a @ cx
    elif criterion.startswith("nl_"):
        g = _cpu_matrix(gradients, "paired activation gradient rows")
        if g.shape != x.shape:
            raise NumericalDomainViolation("Every activation row requires one gradient row")
        if criterion == "nl_normalized":
            g, count = _normalized(g, spec.zero_policy)
            zeros += count
            kind, constant = "normalized_calibration_heuristic", None
        else:
            kind = "first_order_network_loss_upper_bound"
            constant = 2 * float((x.square().sum(1) * g.square().sum(1)).mean())
        cross = xn.T @ ((xn * g).sum(1)[:, None] * g) / n
        h = cross + cross.T
    else:
        h = cx
        if criterion == "nmse":
            kind = "exact_normalized_input_squared_error_with_declared_zero_policy"
    stamp = replace(passport, count=n, weight_sum=float(n),
                    method_version="espace_2410.05437v1_" + criterion)
    return ESPACEStatistics(_finite((h + h.T) / 2, "ESPACE criterion"), criterion,
        stamp, zeros, kind, constant, dict(collection_manifest or {}))


@dataclass(frozen=True)
class ProjectorFactors:
    basis: torch.Tensor  # [in, width]
    eigenvalues: torch.Tensor
    criterion: str
    diagnostics: dict

    @property
    def width(self):
        return self.basis.shape[1]


def select_espace_projector(stats, rank):
    """Ky Fan trace maximizer: eigenvalues descending algebraically, not abs."""
    if not isinstance(stats, ESPACEStatistics):
        raise NumericalDomainViolation("Typed ESPACE statistics required")
    h = _cpu_matrix(stats.matrix, "symmetric ESPACE criterion")
    if h.shape[0] != h.shape[1] or not torch.allclose(h, h.T, rtol=1e-12, atol=1e-12):
        raise NumericalDomainViolation("ESPACE criterion must be symmetric square")
    if type(rank) is not int or not 1 <= rank <= len(h):
        raise NumericalDomainViolation("Projector width must fit the activation dimension")
    values, vectors = torch.linalg.eigh(h)
    order = torch.arange(len(h)-1, len(h)-rank-1, -1)
    basis, selected = vectors[:, order].contiguous(), values[order].contiguous()
    retained = float(selected.sum())
    objective = None if stats.constant is None else stats.constant-retained
    return ProjectorFactors(basis, selected, stats.criterion, {
        "criterion": stats.criterion, "objective_kind": stats.objective_kind,
        "selection": "largest_algebraic_eigenvalues", "actual_width": rank,
        "retained_trace": retained, "criterion_objective": objective,
        "zero_vectors": stats.zero_vectors, "source": ESPACE_SOURCE,
        "empirical_estimator":"observation_mean","zero_denominator_policy":"declared_by_spec",
        "normalized_go_nl_exact_objective": False,
        "network_loss_linearization": stats.criterion.startswith("nl_"),
    })


def solve_espace(weight, stats, rank):
    w = _cpu_matrix(weight, "W[out,in]")
    result = select_espace_projector(stats, rank)
    if w.shape[1] != result.basis.shape[0]:
        raise NumericalDomainViolation("Weight and projector input dimensions disagree")
    p = result.basis.to(weight)
    return FactorPair((weight.detach() @ p).clone(), p.T.contiguous(), result.diagnostics)


def pela_rank(shape, kappa):
    if len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape) or type(kappa) not in (int,float) or not math.isfinite(kappa) or kappa <= 0:
        raise NumericalDomainViolation("PELA requires positive dimensions and kappa")
    m, n = shape
    rank = math.floor(m*n/(kappa*(m+n)))
    if not 1 <= rank <= min(m,n):
        raise NumericalDomainViolation("PELA formula yields an invalid rank; no silent clamping")
    return rank


def pela_regularizer(left, right, reference, *, infinity_norm="max_element",
                     variant="published_subtract_epsilon", epsilon=0.0):
    """Differentiable literal penalty; subtraction changes value, not gradient."""
    for value, name in ((left,"left"),(right,"right"),(reference,"reference")):
        _matrix(value, name)
    if left.shape[1] != right.shape[0] or (left.shape[0],right.shape[1]) != reference.shape:
        raise NumericalDomainViolation("PELA factor dimensions disagree")
    if type(epsilon) not in (int,float) or not math.isfinite(epsilon) or epsilon < 0:
        raise NumericalDomainViolation("epsilon must be finite and nonnegative")
    residual = (left @ right - reference.detach()).abs()
    if infinity_norm == "max_element":
        norm = residual.amax()
    elif infinity_norm == "max_row_sum":
        norm = residual.sum(1).amax()
    else:
        raise NumericalDomainViolation("Unknown infinity norm")
    value = norm-epsilon
    if variant == "hinge_variant":
        return value.clamp_min(0)
    if variant != "published_subtract_epsilon":
        raise NumericalDomainViolation("Unknown PELA regularizer")
    return value


def laser_rank(shape, rho):
    if type(rho) not in (int,float) or not math.isfinite(rho) or not 0 <= rho < 1:
        raise NumericalDomainViolation("LASER rho must be in [0,1)")
    return math.floor(rho*min(shape))


@dataclass(frozen=True)
class CTRFieldFactors:
    coefficients: torch.Tensor  # [vocabulary, rank]
    basis: torch.Tensor  # [embedding_dim, rank]
    mean: torch.Tensor
    affine_correction: torch.Tensor  # (I-UU.T) mu
    passport: StatisticsPassport
    weighting: str

    def projected_table(self):
        return self.coefficients @ self.basis.T + self.mean


def solve_ctr_field(table, rank, passport, *, frequencies=None, weighting="uniform_rows"):
    rows = _cpu_matrix(table, "embedding table")
    _passport(passport)
    if type(rank) is not int or not 1 <= rank <= min(rows.shape):
        raise NumericalDomainViolation("CTR field rank must fit vocabulary and embedding dimensions")
    if weighting == "uniform_rows":
        if frequencies is not None:
            raise NumericalDomainViolation("Uniform-row PCA cannot silently use frequencies")
        weights = None
    elif weighting == "frequency":
        if (not isinstance(frequencies,torch.Tensor) or frequencies.shape != (len(rows),)
                or frequencies.device.type != "cpu" or frequencies.is_complex()
                or not bool(torch.isfinite(frequencies).all()) or bool((frequencies < 0).any())
                or float(frequencies.sum()) <= 0):
            raise NumericalDomainViolation("Frequency PCA requires positive-mass nonnegative category frequencies")
        weights = frequencies
    else:
        raise NumericalDomainViolation("Unknown CTR PCA weighting")
    moments = finish_centered_moments(update_centered_moments(
        empty_centered_moments(rows.shape[1], passport), rows, weights))
    _, vectors = torch.linalg.eigh(moments.covariance)
    u = vectors[:, -rank:].flip(1).contiguous()
    coefficients = (rows-moments.mean) @ u
    correction = moments.mean-u@(u.T@moments.mean)
    return CTRFieldFactors(coefficients.to(table),u.to(table),moments.mean.to(table),
                           correction.to(table),moments.passport,weighting)


def absorb_ctr_linear_blocks(weights, biases, fields):
    """Return per-field compact blocks and ONE bias equal to the block sum."""
    if not weights or len(weights) != len(fields) or len(biases) != len(fields):
        raise NumericalDomainViolation("One weight and optional bias per field required")
    compact, intercept = [], None
    for weight, bias, field in zip(weights,biases,fields):
        _matrix(weight,"CTR linear block")
        if weight.shape[1] != len(field.mean):
            raise NumericalDomainViolation("CTR field and linear block widths disagree")
        value = weight @ field.mean
        if bias is not None:
            if bias.shape != (len(weight),):
                raise NumericalDomainViolation("CTR linear bias dimension disagrees")
            value = value+bias
        if intercept is not None and intercept.shape != value.shape:
            raise NumericalDomainViolation("CTR block outputs must share a dimension for addition")
        intercept = value if intercept is None else intercept+value
        compact.append(weight @ field.basis)
    return tuple(compact), intercept
