"""Checked weighted operator approximation for the first production profile.

W has shape [out, in], and C is the *uncentered* input second moment.
The independent experimental oracle deliberately does not import this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from .plans import MetricPolicy


class NumericalDomainViolation(ValueError):
    """A dependent numerical step cannot satisfy the declared contract."""


@dataclass(frozen=True)
class ApproximationResult:
    """Canonical factors of the returned operator, not of W C^(1/2).

    In particular the spectrum describes the returned operator, never trained
    coefficients or the weighted singular spectrum used to choose it.
    """
    u: torch.Tensor
    s: torch.Tensor
    vh: torch.Tensor
    approximation: torch.Tensor
    requested_rank: int
    actual_rank: int
    moment_rank: int
    weighted_error_squared: float
    ridge: float
    rcond: float | None
    nullspace_policy: str
    condition_number: float | None
    effective_cutoff: float
    discarded_metric_mass: float
    objective: str = "input_second_moment"
    solve_dtype: str = "float64"
    factor_semantics: str = "canonical_svd_of_returned_operator"


def _matrix(value, name):
    if (not isinstance(value, torch.Tensor) or value.ndim != 2 or not value.numel()
            or value.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(value).all())):
        raise NumericalDomainViolation(f"{name} requires a nonempty finite float32/float64 matrix")


def _nonnegative(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _finite(value, stage):
    if not bool(torch.isfinite(value).all()):
        raise NumericalDomainViolation(f"{stage} exceeds the finite float64/output-precision numerical domain")
    return value


def _svd(value, stage):
    _finite(value, stage)
    try:
        factors = torch.linalg.svd(value, full_matrices=False)
    except torch.linalg.LinAlgError as error:
        raise NumericalDomainViolation(f"{stage} SVD could not converge within the numerical domain") from error
    for factor in factors:
        _finite(factor, stage + " SVD factors")
    return factors


def _rank(value):
    _finite(value, "rank-check operator")
    try:
        return int(torch.linalg.matrix_rank(value))
    except torch.linalg.LinAlgError as error:
        raise NumericalDomainViolation("operator rank could not be determined within the numerical domain") from error


def solve_weighted(matrix, moment, rank, policy=MetricPolicy()):
    """Minimize ||(W-A) F||²_F for C=F Fᵀ and rank(A)<=rank.

    ``ridge`` changes C to C+ridge*I. ``rcond`` explicitly removes eigenvalues
    at or below rcond*max(eigenvalue); the reported objective uses this effective
    support. No hidden regularization is applied. Preserve-nullspace adds the
    action W(I-P) and rejects a rank overflow. At full admissible rank, W itself
    is preserved under every policy, including its unobserved action.

    All solves use float64; factors and the returned operator use W's original
    dtype/device. Inputs are detached and never mutated. The output factors are
    re-canonicalized after undoing the metric; they are not factors of W F.
    Finite inputs can still exceed the representable range of intermediate
    products or diagnostics; these fail explicitly with NumericalDomainViolation.
    """
    _matrix(matrix, "matrix")
    # Accept only the explicitly named second-moment result, not centered data.
    from .statistics import InputSecondMoment
    if isinstance(moment, InputSecondMoment):
        moment = moment.matrix
    _matrix(moment, "moment")
    if moment.shape != (matrix.shape[1], matrix.shape[1]):
        raise NumericalDomainViolation("moment must match the input dimension")
    if type(rank) is not int or not 1 <= rank <= min(matrix.shape):
        raise NumericalDomainViolation("rank must fit both operator axes")
    if not isinstance(policy, MetricPolicy) or not _nonnegative(policy.ridge):
        raise NumericalDomainViolation("ridge must be finite and nonnegative")
    if policy.rcond is not None and not _nonnegative(policy.rcond):
        raise NumericalDomainViolation("rcond must be finite and nonnegative")
    if policy.nullspace_policy not in ("support_only", "preserve_nullspace"):
        raise NumericalDomainViolation("explicit null-space policy required")
    with torch.no_grad():
        w = matrix.detach().to(torch.float64)
        c = moment.detach().to(device=w.device, dtype=torch.float64)
        tolerance = torch.finfo(c.dtype).eps * len(c) * max(1., float(c.abs().max())) * 16
        if not torch.allclose(c, c.T, atol=tolerance, rtol=0):
            raise NumericalDomainViolation("moment must be symmetric")
        # Exact symmetry needs no arithmetic: halving minimum subnormals would
        # otherwise erase a represented positive eigenvalue. When averaging is
        # needed, half plus half avoids overflow and diagonal entries stay exact.
        if torch.equal(c, c.T):
            symmetric = c
        else:
            symmetric = c * .5 + c.T * .5
            symmetric.diagonal().copy_(c.diagonal())
        try:
            eigenvalues, vectors = torch.linalg.eigh(symmetric)
        except torch.linalg.LinAlgError as error:
            raise NumericalDomainViolation("metric eigensolve could not converge within the numerical domain") from error
        _finite(eigenvalues, "metric eigenvalues")
        _finite(vectors, "metric eigenvectors")
        if float(eigenvalues.min()) < -tolerance:
            raise NumericalDomainViolation("moment must be positive semidefinite")
        eigenvalues = eigenvalues.clamp_min(0) + policy.ridge
        _finite(eigenvalues, "regularized metric eigenvalues")
        cutoff = (policy.rcond if policy.rcond is not None else torch.finfo(c.dtype).eps * len(c)) * float(eigenvalues.max())
        if not math.isfinite(cutoff):
            raise NumericalDomainViolation("metric cutoff exceeds the finite float64 numerical domain")
        support = eigenvalues > cutoff
        supported_values = torch.where(support, eigenvalues, 0.)
        root = (vectors * supported_values.sqrt()) @ vectors.T
        _finite(root, "metric root")
        inverse_values = torch.zeros_like(eigenvalues)
        inverse_values[support] = eigenvalues[support].rsqrt()
        inverse_root = (vectors * inverse_values) @ vectors.T
        _finite(inverse_root, "metric inverse root")
        if rank == min(w.shape):
            approximation = w.clone()
            effective_nullspace_policy = "preserve_operator_at_full_rank"
        else:
            u, s, vh = _svd(w @ root, "weighted operator")
            approximation = (u[:, :rank] * s[:rank]) @ vh[:rank] @ inverse_root
            _finite(approximation, "inverse-metric operator")
            effective_nullspace_policy = policy.nullspace_policy
            if policy.nullspace_policy == "preserve_nullspace":
                null_vectors = vectors[:, ~support]
                approximation = approximation + (w @ null_vectors) @ null_vectors.T
                if _rank(approximation) > rank:
                    raise NumericalDomainViolation("preserving null-space action exceeds the requested rank")
        u, s, vh = _svd(approximation, "returned operator")
        u, s, vh = (u[:, :rank].to(matrix), s[:rank].to(matrix), vh[:rank].to(matrix))
        for factor in (u, s, vh):
            _finite(factor, "output-precision factors")
        # Reconstruct what will actually be installed, including output rounding.
        installed = (u * s) @ vh
        _finite(installed, "installed operator")
        actual_rank = _rank(installed)
        if actual_rank > rank:
            raise NumericalDomainViolation("output precision violates the requested rank")
        residual = _finite((w - installed.double()) @ root, "weighted residual")
        error = float(residual.square().sum())
        retained = eigenvalues[support]
        condition_number = float(retained.max() / retained.min()) if retained.numel() else None
        discarded_mass = float(eigenvalues[~support].sum())
        if not math.isfinite(error) or not math.isfinite(discarded_mass) or (condition_number is not None and not math.isfinite(condition_number)):
            raise NumericalDomainViolation("weighted objective/metric diagnostics exceed the finite float64 numerical domain")
        return ApproximationResult(
            u, s, vh, installed, rank, actual_rank, int(support.sum()), error,
            float(policy.ridge), policy.rcond, effective_nullspace_policy,
            condition_number, cutoff, discarded_mass,
            "input_second_moment_ridge" if policy.ridge else "input_second_moment")
