"""Numerical evidence for PETRA hypotheses, separate from quality claims.

All matrices are finite float32/float64 tensors. Activation rows are observations;
the moment is E[x x^T], without centering. No test observations are used to fit it.
"""
from __future__ import annotations

from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F


def _finite_matrix(value, name):
    if (not isinstance(value, torch.Tensor) or value.ndim != 2 or
            value.dtype not in (torch.float32, torch.float64) or
            not value.numel() or not torch.isfinite(value).all()):
        raise ValueError(f"{name} requires a nonempty finite float32/float64 matrix")


def relative_error(reference, approximation):
    """JSON-safe relative norm, distinguishing a zero denominator from zero error."""
    if reference.shape != approximation.shape or not torch.isfinite(reference).all() or not torch.isfinite(approximation).all():
        raise ValueError("Aligned finite tensors are required")
    numerator = float(torch.linalg.vector_norm((approximation - reference).to(torch.float64)))
    denominator = float(torch.linalg.vector_norm(reference.to(torch.float64)))
    return {"absolute": numerator, "denominator": denominator,
            "relative": numerator / denominator if denominator else (0.0 if numerator == 0 else None),
            "status": "measured" if denominator else ("zero_reference_exact" if numerator == 0 else "undefined_zero_reference")}


def second_moment(activation_rows):
    _finite_matrix(activation_rows, "activation_rows")
    x = activation_rows.detach().to(torch.float64)
    return x.T @ x / len(x)


def truncated_svd(matrix, rank):
    _finite_matrix(matrix, "matrix")
    if type(rank) is not int or not 1 <= rank <= min(matrix.shape):
        raise ValueError("rank must fit both matrix axes")
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vh[:rank]


def weighted_svd(matrix, moment, rank, *, ridge=0.0, rcond=None):
    """Minimize ||(W-A)M^(1/2)||_F at rank <= r for PSD M.

    A=(WM^(1/2))_r M^(+1/2). Null-space action is zero for singular M;
    it is unidentifiable from these activations. Full rank returns W explicitly,
    preserving its null-space action. A positive ridge changes the objective and
    is recorded, never introduced silently. Float64 eigensolve avoids casting a
    calibration matrix down before detecting its conditioning.
    """
    _finite_matrix(matrix, "matrix")
    _finite_matrix(moment, "moment")
    if moment.shape != (matrix.shape[1], matrix.shape[1]):
        raise ValueError("moment must match the input dimension")
    if type(rank) is not int or not 1 <= rank <= min(matrix.shape):
        raise ValueError("rank must fit both matrix axes")
    if isinstance(ridge, bool) or not isinstance(ridge, (int, float)) or not math.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    if rcond is not None and (isinstance(rcond, bool) or not isinstance(rcond, (int, float)) or not math.isfinite(rcond) or rcond < 0):
        raise ValueError("rcond must be finite and nonnegative")
    w = matrix.detach().to(torch.float64)
    m = moment.detach().to(device=w.device, dtype=torch.float64)
    tolerance = torch.finfo(m.dtype).eps * max(m.shape) * max(1., float(m.abs().max())) * 16
    if not torch.allclose(m, m.T, atol=tolerance, rtol=0):
        raise ValueError("moment must be symmetric")
    eigenvalues, vectors = torch.linalg.eigh((m + m.T) / 2)
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError("moment must be positive semidefinite")
    eigenvalues = eigenvalues.clamp_min(0) + ridge
    cutoff = (rcond if rcond is not None else torch.finfo(m.dtype).eps * len(m)) * float(eigenvalues.max())
    positive = eigenvalues > cutoff
    root = (vectors * eigenvalues.sqrt()) @ vectors.T
    inverse_values = torch.zeros_like(eigenvalues)
    inverse_values[positive] = eigenvalues[positive].rsqrt()
    inverse_root = (vectors * inverse_values) @ vectors.T
    approximation = w.clone() if rank == min(w.shape) else truncated_svd(w @ root, rank) @ inverse_root
    retained = eigenvalues[positive]
    return {"approximation": approximation.to(matrix), "requested_rank": rank,
            "actual_rank": int(torch.linalg.matrix_rank(approximation)),
            "moment_rank": int(positive.sum()), "ridge": float(ridge), "rcond": rcond,
            "condition_number": float(retained.max() / retained.min()) if retained.numel() else None,
            "nullspace_policy": "preserve_operator_at_full_rank_else_zero",
            "weighted_error_squared": float(((w - approximation) @ root).square().sum())}


def layer_matrices(layer):
    """Use the actual composed trained operator, never trainable S as spectrum."""
    from fedcore.models.network_impl.decomposed_layers import IDecomposed
    if isinstance(layer, IDecomposed):
        matrix = layer.factor_matrix().detach()
    elif type(layer) is nn.Linear:
        matrix = layer.weight.detach()
    elif type(layer) in (nn.Conv1d, nn.Conv2d):
        matrix = layer.weight.detach().reshape(layer.groups, layer.out_channels // layer.groups, -1)
    else:
        raise ValueError("Only checked Linear/Conv1d/Conv2d layers are supported")
    return matrix.unsqueeze(0) if matrix.ndim == 2 else matrix


def activation_rows(layer, inputs):
    """Extract actual convolution patches; groups have independent moments."""
    if isinstance(layer, nn.Linear):
        if inputs.shape[-1] != layer.in_features:
            raise ValueError("Linear input dimension mismatch")
        return (inputs.reshape(-1, layer.in_features),)
    if not isinstance(layer, (nn.Conv1d, nn.Conv2d)) or isinstance(layer.padding, str):
        raise ValueError("Convolution requires explicit numeric padding")
    one_dimensional = isinstance(layer, nn.Conv1d)
    x = inputs.unsqueeze(-2) if one_dimensional else inputs
    kernel = (1, layer.kernel_size[0]) if one_dimensional else layer.kernel_size
    stride = (1, layer.stride[0]) if one_dimensional else layer.stride
    dilation = (1, layer.dilation[0]) if one_dimensional else layer.dilation
    padding = (0, layer.padding[0]) if one_dimensional else layer.padding
    if layer.padding_mode != "zeros":
        x = F.pad(x, (padding[1], padding[1], padding[0], padding[0]), mode=layer.padding_mode)
        padding = (0, 0)
    patches = F.unfold(x, kernel, dilation=dilation, padding=padding, stride=stride)
    group_patches = patches.reshape(len(x), layer.groups, -1, patches.shape[-1])
    return tuple(group_patches[:, group].transpose(1, 2).reshape(-1, group_patches.shape[2]) for group in range(layer.groups))


def approximate_layer(layer, calibration_inputs, rank, *, weighted=False, ridge=0.0, calibration_moments=None):
    """Return independent canonical FedCore factors, equal storage at equal rank."""
    from fedcore.models.network_impl.decomposed_layers import DecomposedLinear, DecomposedConv1d, DecomposedConv2d, IDecomposed
    constructors = {nn.Linear: DecomposedLinear, nn.Conv1d: DecomposedConv1d, nn.Conv2d: DecomposedConv2d}
    if not isinstance(layer, IDecomposed) and type(layer) not in constructors:
        raise ValueError("Only checked Linear/Conv1d/Conv2d layers are supported")
    result = deepcopy(layer) if isinstance(layer, IDecomposed) else constructors[type(layer)](layer, decomposer="svd")
    matrices = layer_matrices(layer)
    if calibration_moments is not None:
        if not weighted or len(calibration_moments) != len(matrices):
            raise ValueError("One calibration moment per group is required for weighted approximation")
        moments = calibration_moments
    else:
        moments = tuple(second_moment(x) for x in activation_rows(layer, calibration_inputs)) if weighted else (None,) * len(matrices)
    factors, diagnostics = [], []
    for matrix, moment in zip(matrices, moments):
        if weighted:
            evidence = weighted_svd(matrix, moment, rank, ridge=ridge)
            approximation = evidence.pop("approximation")
        else:
            approximation = truncated_svd(matrix, rank)
            evidence = {"requested_rank": rank, "actual_rank": int(torch.linalg.matrix_rank(approximation)), "ridge": 0.0}
        u, s, vh = torch.linalg.svd(approximation, full_matrices=False)
        factors.append((u[:, :rank], s[:rank], vh[:rank]))
        diagnostics.append(evidence)
    grouped = matrices.shape[0] > 1 or isinstance(layer, (nn.Conv1d, nn.Conv2d))
    result.set_U_S_Vh(*(torch.stack([part[i] for part in factors]) if grouped else factors[0][i] for i in range(3)))
    return result, diagnostics


def orthogonal_penalty(model, *, normalization, coefficient):
    """Explicit experimental /r or /r²; production OrthogonalLoss is untouched."""
    from fedcore.models.network_impl.decomposed_layers import IDecomposed
    if normalization not in ("rank", "rank_squared"):
        raise ValueError("normalization must be rank or rank_squared")
    if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)) or not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("coefficient must be finite and nonnegative")
    values = []
    for layer in model.modules():
        if isinstance(layer, IDecomposed) and layer.U is not None:
            u, _, vh = layer.get_U_S_Vh()
            rank = u.shape[-1]
            identity = torch.eye(rank, device=u.device, dtype=u.dtype)
            numerator = ((u.transpose(-2, -1) @ u - identity).square().sum(dim=(-2, -1)) +
                         (vh @ vh.transpose(-2, -1) - identity).square().sum(dim=(-2, -1))).mean()
            values.append(numerator / rank ** (1 if normalization == "rank" else 2))
    if values:
        return coefficient * torch.stack(values).mean()
    parameter = next(model.parameters(), None)
    return parameter.sum() * 0 if parameter is not None else torch.tensor(0.)


def factor_diagnostics(model):
    from fedcore.models.network_impl.decomposed_layers import IDecomposed
    records = []
    for name, layer in model.named_modules():
        if isinstance(layer, IDecomposed) and layer.U is not None:
            u, s, vh = layer.get_U_S_Vh()
            matrices = layer_matrices(layer)
            records.append({"layer": name, "stored_rank": u.shape[-1],
                            "actual_rank": torch.linalg.matrix_rank(matrices).tolist(),
                            "operator_spectrum": torch.linalg.svdvals(matrices).tolist(),
                            "coefficient_l1": float(s.detach().abs().sum()) if s is not None else None,
                            "coefficient_l2": float(s.detach().norm()) if s is not None else None,
                            "coefficient_status": "separate_trainable_coefficients" if s is not None else "absorbed_into_factors",
                            "u_norm": float(u.detach().norm()), "vh_norm": float(vh.detach().norm()),
                            "orthogonality_per_rank": float(orthogonal_penalty(layer, normalization="rank", coefficient=1.).detach())})
    return records


def matched_rank_policy_audit(layer, inputs, rank):
    """Invert scalar criteria for a single operator and verify obtained rank.

    Group-specific thresholds cannot silently become one shared threshold. A
    grouped operator is therefore reported unsupported for this exact matching
    subexperiment; its weighted/ordinary matched-rank experiment still runs.
    """
    from fedcore.algorithm.low_rank.rank_pruning import rank_threshold_pruning_in_place
    from fedcore.models.network_impl.decomposed_layers import DecomposedLinear, DecomposedConv1d, DecomposedConv2d, IDecomposed
    matrices = layer_matrices(layer)
    if len(matrices) != 1:
        return {"status": "unsupported", "reason": "One common policy threshold cannot promise equal rank for distinct group spectra"}
    matrix = matrices[0]
    if type(rank) is not int or not 1 <= rank <= min(matrix.shape):
        raise ValueError("rank must fit the actual operator")
    spectrum = torch.linalg.svdvals(matrix).double()
    expected = truncated_svd(matrix, rank)
    constructors = {nn.Linear: DecomposedLinear, nn.Conv1d: DecomposedConv1d, nn.Conv2d: DecomposedConv2d}
    records = []
    for strategy in ("quantile", "explained_variance", "absolute_sum", "energy"):
        if strategy == "quantile":
            threshold = (rank - .5) / len(spectrum)
        else:
            mass = spectrum.square() if strategy == "explained_variance" else spectrum if strategy == "absolute_sum" else spectrum.softmax(0)
            if mass.sum() == 0 or mass[rank - 1] == 0:
                records.append({"strategy": strategy, "status": "unmatchable_zero_mass", "requested_rank": rank})
                continue
            cumulative = mass.cumsum(0) / mass.sum()
            previous = cumulative[rank - 2] if rank > 1 else 0.
            threshold = 1. if rank == len(spectrum) else float((previous + cumulative[rank - 1]) / 2)
        candidate = deepcopy(layer) if isinstance(layer, IDecomposed) else constructors[type(layer)](layer, decomposer="svd")
        rank_threshold_pruning_in_place(candidate, threshold=threshold, strategy=strategy, round_to_times=1)
        obtained = candidate.S.shape[-1]
        records.append({"strategy": strategy, "threshold": threshold, "requested_rank": rank, "obtained_rank": obtained,
                        "status": "matched" if obtained == rank else "rank_mismatch",
                        "operator_error_from_common_svd": relative_error(expected, layer_matrices(candidate)[0]),
                        "factor_parameters": sum(p.numel() for p in candidate.parameters())})
    return {"status": "completed", "records": records,
            "interpretation": "Matched criteria retain the same operator-SVD prefix; thresholds mean different quantities"}
