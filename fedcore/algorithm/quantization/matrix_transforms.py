"""Pure small-matrix kernels for LPLR and the CPU scalar CALDERA profile."""
from dataclasses import dataclass
import math
import torch

from fedcore.algorithm.low_rank.structured_profiles import linear_least_squares_fit, balanced_factors
from fedcore.algorithm.low_rank.approximation import prepare_weighted_metric, solve_weighted_prepared
from fedcore.algorithm.low_rank.plans import MetricPolicy
from .graph_specs import Method
from .tensor_formats import PackedTensor, Grid, Rounding, pack_codes, quantize_codes
from .portable_layers import PortableQuantizedLinear, PortableQuantizedFactorLinear, QuantizedSumLinear


def _matrix(w):
    if (not isinstance(w, torch.Tensor) or w.ndim != 2 or w.device.type != "cpu"
            or w.dtype not in (torch.float32, torch.float64) or not w.numel() or not bool(torch.isfinite(w).all())):
        raise ValueError("Finite CPU matrix required")


@dataclass(frozen=True)
class LPLRResult:
    left: PackedTensor
    right: PackedTensor
    projection: torch.Tensor
    fitted_right: torch.Tensor
    diagnostics: dict


def lplr(weight, spec, generator):
    """Z=W S, Q(Z), C=Q(Z)^dagger W, Q(C); m is actual factor width."""
    _matrix(weight)
    if spec.method not in (Method.LPLR, Method.LPLR_SVD):
        raise ValueError("LPLR method required")
    if any(s.grid is not Grid.ENDPOINT or s.rounding is not Rounding.STOCHASTIC for s in (spec.left, spec.right)):
        raise ValueError("LPLR requires the declared 2^B endpoint stochastic grid")
    m, w = spec.rank, weight.detach().double()
    if type(m) is not int or not 1 <= m <= min(w.shape):
        raise ValueError("Factor width must fit matrix axes")
    if spec.method is Method.LPLR:
        projection = torch.randn((w.shape[1], m), dtype=torch.float64, generator=generator)/math.sqrt(m)
        left = w @ projection
    else:
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        projection = vh[:m].T
        left = u[:, :m]*s[:m]
    qleft = PackedTensor.from_tensor(left, spec.left, generator=generator)
    fit = linear_least_squares_fit(qleft.decode(), w, rcond=spec.rcond, ridge=spec.ridge)
    right = fit.coefficients.T.contiguous()
    qright = PackedTensor.from_tensor(right, spec.right, generator=generator)
    k = spec.comparison_rank if spec.comparison_rank is not None else m
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    dsvd = (u[:, :k]*s[:k])@vh[:k]
    return LPLRResult(qleft, qright, projection, right,
        {"method": spec.method.value, "factor_width_m": m, "comparison_rank_k": k,
         "regression_design": "decoded_quantized_left", "design_rank": fit.design_rank,
         "rcond": spec.rcond, "ridge": spec.ridge, "seed": spec.seed,
         "weight_frobenius_squared": float((w-qleft.decode()@qright.decode()).square().sum()),
         "dsvd_reference_squared": float((w-dsvd).square().sum())})


def _factor_layer(left, right, bias):
    layer = PortableQuantizedFactorLinear(right.shape[1], left.shape[0], left.shape[1], bias is not None,
        left_spec=left.spec, right_spec=right.spec)
    layer.left, layer.right = left, right
    if bias is not None:
        layer.bias = bias.detach().double().clone()
    return layer


def input_second_moment(rows, width):
    _matrix(rows)
    if rows.shape[1] != width or len(rows) == 0:
        raise ValueError("Calibration rows must use the original input coordinates")
    x = rows.detach().double()
    return x.T@x/len(x)


def scalar_ldl(moment, rcond):
    """Unpivoted PSD LDL; zero pivots are support removal, never hidden ridge."""
    _matrix(moment)
    if moment.shape[0] != moment.shape[1] or not torch.allclose(moment, moment.T, atol=1e-12, rtol=1e-12):
        raise ValueError("Symmetric moment required")
    h = moment.double()
    tolerance = max(rcond, torch.finfo(h.dtype).eps*len(h)*16)*float(h.abs().max())
    if float(torch.linalg.eigvalsh(h).min()) < -tolerance:
        raise ValueError("LDLQ requires a PSD metric")
    l, d = torch.eye(len(h), dtype=torch.float64), torch.zeros(len(h), dtype=torch.float64)
    for j in range(len(h)):
        pivot = h[j,j] - (l[j,:j].square()*d[:j]).sum()
        if pivot < -tolerance:
            raise ValueError("Numerical negative LDL pivot")
        residual = h[j+1:,j] - (l[j+1:,:j]*l[j,:j]*d[:j]).sum(1)
        if pivot > tolerance:
            d[j] = pivot
            l[j+1:,j] = residual/pivot
        elif residual.numel() and float(residual.abs().max()) > tolerance:
            raise ValueError("Unsupported unstable singular LDL pivot")
    return l, d


def scalar_ldlq(weight, moment, tensor_spec, *, generator, rcond=1e-10):
    """Reverse-column feedback for H=L D L^T: q_j=Q(w_j+sum e_i L_ij)."""
    _matrix(weight)
    l, d = scalar_ldl(moment, rcond)
    w = weight.detach().double()
    baseline = PackedTensor.from_tensor(w, tensor_spec, generator=generator)
    # Column processing requires stable per-tensor or per-output-row scales.
    if tensor_spec.axis not in (None, 0):
        raise ValueError("Scalar LDLQ supports tensor or output-channel scales")
    decoded = torch.zeros_like(w)
    signed = torch.zeros_like(w, dtype=torch.int64)
    feedback = []
    for j in reversed(range(w.shape[1])):
        correction = (w[:,j+1:]-decoded[:,j+1:])@l[j+1:,j]
        adjusted = (w[:,j]+correction).unsqueeze(1)
        codes, _ = quantize_codes(adjusted, tensor_spec, generator=generator, scale=baseline.scale)
        signed[:,j] = codes[:,0]
        scale = baseline.scale[0] if tensor_spec.axis is None else baseline.scale
        values = codes[:,0].double()
        if tensor_spec.grid is Grid.ENDPOINT:
            values -= (2**tensor_spec.bits-1)/2
        decoded[:,j] = values*scale
        feedback.append(float(correction.square().sum()))
    baseline.packed = pack_codes(signed + ((2**(tensor_spec.bits-1)-1) if tensor_spec.grid is Grid.SYMMETRIC else 0), tensor_spec.bits)
    baseline.validate_state(baseline.state_dict())
    return baseline, {"ldl_diagonal": d.tolist(), "feedback_squared": feedback,
                      "column_order": "reverse", "metric": "explicit_psd_no_hidden_ridge"}


def caldera(weight, bias, spec, rows, generator):
    """Finite scalar LDLQ/weighted-regression alternation, retaining best Q+LR."""
    _matrix(weight)
    w = weight.detach().double()
    h = input_second_moment(rows, w.shape[1])
    metric = prepare_weighted_metric(h, MetricPolicy(ridge=spec.ridge, rcond=spec.rcond))
    # LDLQ sees exactly the same supported/ridged objective as the regressions.
    effective_h = metric.root@metric.root.T
    q = PackedTensor.from_tensor(w, spec.weight, generator=generator)
    def quantized_factors(residual):
        pair = balanced_factors(solve_weighted_prepared(residual, metric, spec.rank))
        return (PackedTensor.from_tensor(pair.left, spec.left, generator=generator),
                PackedTensor.from_tensor(pair.right, spec.right, generator=generator))
    left, right = quantized_factors(w-q.decode())
    def score(q, left, right):
        return float(((w-q.decode()-left.decode()@right.decode())@metric.root).square().sum())
    best_score = score(q,left,right)
    best = (q,left,right)
    initial = best_score
    history, feedbacks = [best_score], []
    for _ in range(spec.outer_iterations):
        q, details = scalar_ldlq(w-left.decode()@right.decode(), effective_h, spec.weight, generator=generator, rcond=spec.rcond)
        feedbacks.append(details)
        candidate_score = score(q,left,right)
        history.append(candidate_score)
        if candidate_score < best_score:
            best_score, best = candidate_score, (q,left,right)
        residual = w-q.decode()
        # Weighted initialization reuses the existing prepared SVD kernel.
        left, right = quantized_factors(residual)
        candidate_score = score(q,left,right)
        history.append(candidate_score)
        if candidate_score < best_score:
            best_score, best = candidate_score, (q,left,right)
        for _ in range(spec.inner_iterations):
            target = residual@metric.root
            fit = linear_least_squares_fit(left.decode(), target, rcond=spec.rcond)
            r = fit.coefficients.T@metric.inverse_root
            right = PackedTensor.from_tensor(r.contiguous(), spec.right, generator=generator)
            fit = linear_least_squares_fit((right.decode()@metric.root).T, target.T, rcond=spec.rcond)
            left = PackedTensor.from_tensor(fit.coefficients, spec.left, generator=generator)
            candidate_score = score(q,left,right)
            history.append(candidate_score)
            if candidate_score < best_score:
                best_score, best = candidate_score, (q,left,right)
    q,left,right = best
    base = PortableQuantizedLinear(w.shape[1], w.shape[0], bias is not None, spec=spec.weight)
    base.weight = q
    if bias is not None:
        base.bias = bias.detach().double().clone()
    return QuantizedSumLinear(base, _factor_layer(left,right,None)), {
        "method": "caldera_cpu_scalar", "initial_error": initial, "best_error": best_score,
        "candidate_errors": history, "ldlq": feedbacks, "outer_iterations": spec.outer_iterations,
        "inner_iterations": spec.inner_iterations, "metric": "input_second_moment",
        "ridge": spec.ridge, "rcond": spec.rcond, "metric_support": int(metric.support.sum()),
        "profile": "scalar_uniform_no_quip_e8", "bias_count": int(bias is not None)}


def build_method(weight, bias, spec, rows, generator):
    if spec.method in (Method.LPLR, Method.LPLR_SVD):
        result = lplr(weight, spec, generator)
        return _factor_layer(result.left, result.right, bias), result.diagnostics
    if spec.method is Method.CALDERA:
        return caldera(weight, bias, spec, rows, generator)
    if spec.method is Method.TENDER:
        from .tender import TenderLinear
        return TenderLinear.calibrate(weight, bias, rows, weight_bits=spec.weight.bits,
            activation_bits=spec.activation_bits, accumulator_bits=spec.accumulator_bits), {"method": spec.method.value}
    raise ValueError("Unsupported method")
