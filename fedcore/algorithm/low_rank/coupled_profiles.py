"""Pure numerical objectives for the bounded coupled graph transforms.

Pinned sources: FLAT 2505.23966v1, section 4.1 and appendices A/C;
MoDeGPT 2408.09632v3, algorithms 1/2 and theorem 3.
Q/K pair constraints and joint GQA product fitting are explicitly reconstructed
extensions; the FLAT v1 paper does not provide a Q/K algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from .approximation import NumericalDomainViolation
from .structured_profiles import FactorPair, linear_least_squares_fit


def _matrix(value, name):
    if (not isinstance(value, torch.Tensor) or value.ndim != 2 or not value.numel()
            or value.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(value).all())):
        raise NumericalDomainViolation(f"{name} requires a finite nonempty float32/float64 matrix")


def _rank(rank, width):
    if type(rank) is not int or not 1 <= rank <= width:
        raise NumericalDomainViolation("rank must be a positive integer fitting the inner width")


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise NumericalDomainViolation(f"{name} exceeds the finite numerical domain")
    return value


def flat_value_basis(rows, rank):
    """Uncentered Value PCA: min_S ||Y-YSS^T||F², S^T S=I."""
    _matrix(rows, "Value observations"); _rank(rank, rows.shape[1])
    values = rows.detach().double()
    _, vectors = torch.linalg.eigh(_finite(values.T @ values, "Value Gram"))
    basis = vectors[:, -rank:]
    error = _finite((values - values @ basis @ basis.T).square().sum(), "Value projection residual")
    return basis.to(rows), {"objective": "uncentered_value_projection", "projection_error_squared": float(error)}


def mode_product_factors(left, right, inputs_rows, rank):
    """Thin weighted-product fit without materializing left @ right.

For W=AB and row observations X, two QR decompositions yield a small core
R_A R_Z^T with Z=XB^T. Its SVD gives the optimal observed output rank-r
approximation. An explicit LS solve maps its right observation factor back to
input coordinates; the input nullspace is discarded for truncated ranks.
At the full original inner width the original factors are returned exactly,
including every input-nullspace direction.
"""
    for value, name in ((left, "left product factor"), (right, "right product factor"), (inputs_rows, "product inputs")):
        _matrix(value, name)
    if left.shape[1] != right.shape[0] or inputs_rows.shape[1] != right.shape[1]:
        raise NumericalDomainViolation("Product/observation axes disagree")
    _rank(rank, right.shape[0])
    a, b, x = (value.detach().double() for value in (left, right, inputs_rows))
    if rank == right.shape[0]:
        return FactorPair(left.detach().clone(), right.detach().clone(),
                          {"objective": "weighted_value_output_product", "weighted_error_squared": 0.0,
                           "product_materialized": False, "full_width_policy": "preserve_original_factors"})
    z = _finite(x @ b.T, "weighted right product")
    qa, ra = torch.linalg.qr(a, mode="reduced")
    qz, rz = torch.linalg.qr(z, mode="reduced")
    core = _finite(ra @ rz.T, "thin product core")
    u, s, vh = torch.linalg.svd(core, full_matrices=False)
    actual = min(rank, len(s))
    root = s[:actual].clamp_min(0).sqrt()
    factor_left = (qa @ u[:, :actual]) * root
    target_rows = qz @ (root[:, None] * vh[:actual]).T
    # This reuses the fixed-design primitive also used by MoDeGPT Type-I.
    fitted = linear_least_squares_fit(x, target_rows)
    factor_right = fitted.coefficients.double()
    # Requested physical width is preserved when calibration support is thin.
    if actual < rank:
        factor_left = torch.cat((factor_left, a.new_zeros((a.shape[0], rank - actual))), 1)
        factor_right = torch.cat((factor_right, b.new_zeros((rank - actual, b.shape[1]))), 0)
    l = _finite(factor_left.to(left), "installed left product factor")
    r = _finite(factor_right.to(right), "installed right product factor")
    # Frobenius error in thin coordinates: Qz rz A^T versus X R^T L^T.
    # No [output,input] product is needed, nor a dense output observation matrix.
    new_observed = x @ r.double().T
    old_norm = (ra @ rz.T).square().sum()
    new_norm = torch.trace((new_observed.T @ new_observed) @ (l.double().T @ l.double()))
    cross = torch.trace((a.T @ l.double()) @ (new_observed.T @ z))
    error = _finite((old_norm + new_norm - 2 * cross).clamp_min(0), "product residual")
    return FactorPair(l, r, {"objective": "weighted_value_output_product", "weighted_error_squared": float(error),
        "spectral_tail_squared": float(s[actual:].square().sum()), "product_materialized": False,
        "core_shape": list(core.shape), "nullspace_policy": "support_only", "requested_rank": rank})


@dataclass(frozen=True)
class MLPChannelFit:
    indices: tuple[int, ...]
    down_weight: torch.Tensor
    diagnostics: dict


def mlp_channel_fit(features_rows, down_weight, rank, *, indices=(), family="modegpt",
                    leverage_ridge=1.0, ridge=0.0, rcond=None):
    """Select actual post-activation channels; fit only the free Down map."""
    _matrix(features_rows, "post-activation features"); _matrix(down_weight, "Down weight")
    _rank(rank, features_rows.shape[1])
    if down_weight.shape[1] != features_rows.shape[1] or family not in ("flat", "modegpt"):
        raise NumericalDomainViolation("MLP family/features/Down axes disagree")
    h = features_rows.detach().double(); w = down_weight.detach().double()
    lam = 1.0 if family == "flat" else leverage_ridge
    if type(lam) not in (int, float) or not 0 < lam < float("inf"):
        raise NumericalDomainViolation("Leverage regularization must be finite and positive")
    gram = _finite(h.T @ h, "post-activation Gram")
    regularized = _finite(gram + lam * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device), "leverage regularized Gram")
    scores = _finite(torch.linalg.solve(regularized, gram).diagonal(), "leverage scores")
    if indices:
        if (not isinstance(indices, tuple) or len(indices) != rank or len(set(indices)) != rank
                or any(type(i) is not int or not 0 <= i < h.shape[1] for i in indices)):
            raise NumericalDomainViolation("Selected MLP indices must be distinct and fit the current channels")
        chosen = indices
    else:
        chosen = tuple(sorted(torch.argsort(scores, descending=True, stable=True)[:rank].tolist()))
    design = h[:, chosen]
    target = _finite(h @ w.T, "fixed MLP target")
    fit = linear_least_squares_fit(design, target, rcond=rcond, ridge=ridge)
    # Full-channel identity should preserve unseen directions as well as samples.
    if rank == h.shape[1] and not ridge:
        installed = w[:, chosen].to(down_weight)
    else:
        installed = fit.coefficients.to(down_weight)
    old = w[:, chosen]
    baseline = float(_finite((design @ old.T - target).square().sum(), "baseline MLP residual"))
    residual = float(_finite((design @ installed.double().T - target).square().sum(), "installed MLP residual"))
    objective = residual + ridge * float(installed.double().square().sum())
    if not ridge and rcond is None and residual > baseline + 1e-7 * max(1.0, baseline, float(target.square().sum())):
        raise NumericalDomainViolation("Installed exact LS unexpectedly worsens the fixed-design residual")
    return MLPChannelFit(chosen, installed, {"family": family,
        "objective": fit.objective, "selection_objective": "ridge_leverage_post_activation_gram",
        "selection_ridge": float(lam), "ridge": float(ridge), "rcond": rcond,
        "residual_squared": residual, "baseline_residual_squared": baseline,
        "penalized_objective": objective, "design_rank": fit.design_rank,
        "indices": list(chosen), "target": "current_post_activation_times_current_down",
        "full_width_policy": "preserve_original_down" if rank == h.shape[1] and not ridge else "fixed_design_ls"})


def select_query_key_coordinates(query_rows, key_rows, rank, *, family, rope=True, coordinates=()):
    """Family-specific coordinate scores; pair closure is an extra constraint.

MoDe's square-root correlation column norms equal sqrt(diag(C)); their
product supplies Type-II's CR score. FLAT's energy score is a reconstructed
baseline rather than a claim about the FLAT v1 paper.
"""
    _matrix(query_rows, "rotated queries"); _matrix(key_rows, "rotated keys")
    if query_rows.shape[1] != key_rows.shape[1] or family not in ("flat", "modegpt"):
        raise NumericalDomainViolation("Q/K coordinates and method family disagree")
    width = query_rows.shape[1]; _rank(rank, width)
    if rope and (width % 2 or rank % 2):
        raise NumericalDomainViolation("Q/K rank must retain complete interleaved RoPE pairs")
    qdiag = _finite(query_rows.detach().double().square().sum(0), "Q correlation diagonal")
    kdiag = _finite(key_rows.detach().double().square().sum(0), "K correlation diagonal")
    scores = _finite(qdiag + kdiag if family == "flat" else qdiag.sqrt() * kdiag.sqrt(), "Q/K scores")
    if coordinates:
        if (not isinstance(coordinates, tuple) or len(coordinates) != rank or len(set(coordinates)) != rank
                or any(type(i) is not int or not 0 <= i < width for i in coordinates)):
            raise NumericalDomainViolation("Selected Q/K coordinates must fit the current head width")
        chosen = coordinates
        if rope and any(chosen[i] % 2 or chosen[i + 1] != chosen[i] + 1 for i in range(0, rank, 2)):
            raise NumericalDomainViolation("Selected Q/K coordinates split or reorder a RoPE pair")
    elif rope:
        pair_scores = scores.reshape(-1, 2).sum(1)
        pairs = torch.argsort(pair_scores, descending=True, stable=True)[:rank // 2].sort().values.tolist()
        chosen = tuple(i for pair in pairs for i in (2 * pair, 2 * pair + 1))
    else:
        chosen = tuple(sorted(torch.argsort(scores, descending=True, stable=True)[:rank].tolist()))
    return chosen, {"objective": "reconstructed_qk_energy" if family == "flat" else "pair_constrained_type_ii_cr_norm",
                    "scores": scores.tolist(), "coordinates": list(chosen), "rope_pair_constraint": rope}


def validate_rope_intertwining(basis, original_frequencies, reduced_frequencies, *, atol=1e-10):
    """Check the generator identity G S=S G', sufficient for every position.

The runtime exports coordinate selectors only. This pure check demonstrates
why an arbitrary PCA rotation cannot be installed before interleaved RoPE.
"""
    _matrix(basis, "RoPE basis")
    if basis.shape[0] % 2 or basis.shape[1] % 2:
        raise NumericalDomainViolation("RoPE bases require complete coordinate pairs")
    def generator(freq, width):
        if (not isinstance(freq, torch.Tensor) or freq.ndim != 1 or len(freq) * 2 != width
                or not bool(torch.isfinite(freq).all()) or bool((freq <= 0).any())):
            raise NumericalDomainViolation("Finite positive RoPE frequencies required")
        result = basis.new_zeros((width, width), dtype=torch.float64)
        ids = torch.arange(len(freq), device=basis.device) * 2
        result[ids, ids + 1] = -freq.to(result); result[ids + 1, ids] = freq.to(result)
        return result
    s = basis.detach().double()
    if not torch.allclose(s.T @ s, torch.eye(s.shape[1], dtype=s.dtype, device=s.device), atol=atol, rtol=atol):
        raise NumericalDomainViolation("RoPE basis must have orthonormal columns")
    g = generator(original_frequencies, s.shape[0]); reduced = generator(reduced_frequencies, s.shape[1])
    if not torch.allclose(g @ s, s @ reduced, atol=atol, rtol=atol):
        raise NumericalDomainViolation("RoPE basis does not satisfy R_t S = S R'_t for all positions")
    return True
