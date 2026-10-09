"""Checked algebra for the explicitly supported structured SVD profiles.

Operators use W[out, in]; observation matrices use rows unless the argument is
named ``inputs_columns`` (DRONE's convention). Solves are detached float64 and
results return to the operator dtype/device. There are no models, loaders,
optimizers or hidden calibration objectives in this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import torch

from .approximation import NumericalDomainViolation, solve_weighted
from .plans import MetricPolicy


@dataclass(frozen=True)
class FactorPair:
    """Two factors of the actual operator, rather than its weighted image."""

    left: torch.Tensor
    right: torch.Tensor
    diagnostics: Mapping = field(default_factory=dict)

    @property
    def rank(self):
        return self.right.shape[0]

    @property
    def parameter_elements(self):
        return self.left.numel() + self.right.numel()

    def matrix(self):
        return self.left @ self.right


@dataclass(frozen=True)
class LinearLeastSquaresFit:
    coefficients: torch.Tensor  # [target features, design features]
    residual_squared: float
    penalized_objective: float
    design_rank: int
    rcond: float | None
    ridge: float
    condition_number: float | None
    objective: str


@dataclass(frozen=True)
class SharedBasisFactors:
    lefts: tuple[torch.Tensor, ...]
    right: torch.Tensor
    diagnostics: Mapping

    @property
    def parameter_elements(self):
        return self.right.numel() + sum(left.numel() for left in self.lefts)


@dataclass(frozen=True)
class GroupReduceFactors:
    token_to_group: torch.Tensor
    token_to_local: torch.Tensor
    group_tokens: tuple[torch.Tensor, ...]
    factors: tuple[FactorPair, ...]
    diagnostics: Mapping

    @property
    def parameter_elements(self):
        return sum(pair.parameter_elements for pair in self.factors)

    @property
    def map_bytes(self):
        # The portable table persists these two vocabulary maps only.
        return sum(t.numel() * t.element_size() for t in
                   (self.token_to_group, self.token_to_local))


def _matrix(value, name, *, allow_empty=False):
    if (not isinstance(value, torch.Tensor) or value.ndim != 2
            or (not allow_empty and not value.numel())
            or value.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(value).all())):
        raise NumericalDomainViolation(f"{name} requires a finite float32/float64 matrix")


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise NumericalDomainViolation(f"{name} exceeds the finite numerical domain")
    return value


def _number(value, name, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise NumericalDomainViolation(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")


def _requested_rank(rank, shape):
    if type(rank) is not int or not 1 <= rank <= min(shape):
        raise NumericalDomainViolation("rank must be a positive integer fitting the operator axes")


def _svd(value, name):
    _finite(value, name)
    try:
        result = torch.linalg.svd(value, full_matrices=False)
    except torch.linalg.LinAlgError as error:
        raise NumericalDomainViolation(f"{name} SVD failed to converge") from error
    for tensor in result:
        _finite(tensor, name + " SVD factors")
    return result


def _scalar(value, name):
    result = float(value)
    if not math.isfinite(result):
        raise NumericalDomainViolation(f"{name} exceeds the finite numerical domain")
    return result


def _pair(left, right, reference, diagnostics):
    left = _finite(left.to(reference), "left output factor")
    right = _finite(right.to(reference), "right output factor")
    _finite(left @ right, "installed factor product")
    return FactorPair(left, right, dict(diagnostics))


def balanced_factors(result, *, method=None):
    """Turn an existing ApproximationResult into a portable two-factor pair."""
    root = result.s.clamp_min(0).sqrt()
    diagnostics = {name: getattr(result, name) for name in result.__dataclass_fields__
                   if name not in ("u", "s", "vh", "approximation")}
    if method is not None:
        diagnostics["method"] = method
    return _pair(result.u * root, root[:, None] * result.vh,
                 result.u, diagnostics)


def concatenate_factor_pairs(first: FactorPair, second: FactorPair):
    """Represent L1 R1 + L2 R2 exactly; bias belongs to the caller once."""
    for pair in (first, second):
        _matrix(pair.left, "left factor")
        _matrix(pair.right, "right factor")
        if pair.left.shape[1] != pair.right.shape[0]:
            raise NumericalDomainViolation("factor inner axes disagree")
    if (first.left.shape[0] != second.left.shape[0]
            or first.right.shape[1] != second.right.shape[1]
            or first.left.dtype != second.left.dtype
            or first.right.dtype != second.right.dtype
            or first.left.device != second.left.device
            or first.right.device != second.right.device):
        raise NumericalDomainViolation("concatenation requires compatible shape, dtype and device")
    return FactorPair(torch.cat((first.left, second.left), 1),
                      torch.cat((first.right, second.right), 0),
                      {"method": "exact_residual_concatenation", "bias_count": 1,
                       "main_rank": first.rank, "residual_rank": second.rank})


def eora_factors(weight, base_weight, moment, rank, policy=MetricPolicy()):
    """Analytically approximate W-W_base in the explicitly supplied metric.

    The EoRA v1 eigenspace formula is DeltaW Q sqrt(Lambda), followed by
    unprojection. The supplied PSD Gram is explicit: callers must identify an
    author's averaged-activation Gram versus an uncentered second moment.
    ``base_weight`` is an explicitly supplied numerical view, never obtained by
    asking a packed base module to materialize its weights here.
    Source: https://arxiv.org/html/2410.21271v1 (Algorithm 1).
    """
    _matrix(weight, "weight")
    _matrix(base_weight, "explicit numerical base weight")
    if base_weight.shape != weight.shape:
        raise NumericalDomainViolation("base weight shape must match weight")
    residual = _finite(weight.detach() - base_weight.detach().to(weight), "EoRA residual")
    result = balanced_factors(solve_weighted(residual, moment, rank, policy), method="eora_psd_support")
    return FactorPair(result.left, result.right,
                      {**result.diagnostics, "source_version": "2410.21271v1",
                       "base_policy": "unchanged_explicit_numerical_view",
                       "recovery": "analytic_no_gradient_updates"})


def eora_author_gram_update(gram,inputs_rows,*,calibration_samples,batch_samples=1):
    """Explicit fixed-N forgetting update in the released NVlabs EoRA hook.

    ``inputs_rows`` flatten the B sample/token axes, but ``batch_samples`` is
    B, not the flattened row count. N is the configured total calibration sample
    count and remains fixed: G_new=N/(N+B) G_old + X^T X/N. Thus this state is
    order/batch dependent and must not be called an empirical second moment or
    merged as a sample-count accumulator. This reproduces the mathematical
    recurrence in float64, not the author's fp32 rounding/negative-eigenvalue
    replacement. Source: NVlabs/EoRA, EoRA/eora.py, llama_sequential_eigen hook,
    https://github.com/NVlabs/EoRA/blob/main/EoRA/eora.py (checked 2026-10-09).
    """
    _matrix(gram,"previous EoRA Gram")
    _matrix(inputs_rows,"EoRA batch rows")
    if gram.shape!=(inputs_rows.shape[1],inputs_rows.shape[1]):
        raise NumericalDomainViolation("EoRA Gram must match batch input dimension")
    if (type(calibration_samples) is not int or calibration_samples<1
            or type(batch_samples) is not int or batch_samples<1 or batch_samples>calibration_samples
            or len(inputs_rows)%batch_samples):
        raise NumericalDomainViolation("positive fixed N and compatible sample/token batch axes required")
    previous=gram.detach().double()
    batch=inputs_rows.detach().to(device=gram.device,dtype=torch.float64)
    updated=_finite(previous*(calibration_samples/(calibration_samples+batch_samples))
                    +(batch.T@batch)/calibration_samples,"author EoRA Gram update")
    return updated,{"gram_update_version":"nvlabs_fixed_n_forgetting_2026_10_09",
                    "calibration_samples":calibration_samples,"batch_samples":batch_samples,
                    "input_rows":len(inputs_rows),"mergeable":False,"order_dependent":True,
                    "solve_dtype":"float64","author_rounding_reproduced":False,
                    "source_url":"https://github.com/NVlabs/EoRA/blob/main/EoRA/eora.py"}


def mixed_rank_metric(weight, inputs_rows, *, objective, zero_policy="error"):
    """Return the chosen ViT metric without conflating two different targets.

    ``author_mean_input`` uses the representative mean input (a rank-one
    surrogate); ``normalized_output`` uses mean(xx^T / ||Wx||^2), the exact
    normalized per-observation output objective. Bias is excluded explicitly.
    Source: https://arxiv.org/html/2402.06004v1, Section 3.
    """
    _matrix(weight, "weight")
    _matrix(inputs_rows, "inputs_rows")
    if inputs_rows.shape[1] != weight.shape[1]:
        raise NumericalDomainViolation("input axis must match weight")
    if zero_policy not in ("error", "skip"):
        raise NumericalDomainViolation("zero_policy must be error or skip")
    x = inputs_rows.detach().to(device=weight.device, dtype=torch.float64)
    if objective == "author_mean_input":
        x = x.mean(0, keepdim=True)
    elif objective != "normalized_output":
        raise NumericalDomainViolation("explicit mixed-rank objective required")
    denominator = _finite((x @ weight.detach().double().T).square().sum(1), "output denominators")
    valid = denominator > 0
    if not bool(valid.all()) and zero_policy == "error":
        raise NumericalDomainViolation("normalized output has a zero denominator")
    if not bool(valid.any()):
        raise NumericalDomainViolation("no nonzero output observations remain")
    scaled = x[valid] / denominator[valid].sqrt()[:, None]
    metric = _finite(scaled.T @ scaled / len(scaled), "mixed-rank metric")
    return metric, {"objective": objective, "zero_policy": zero_policy,
                    "observations": len(x), "retained_observations": len(scaled),
                    "source_version": "2402.06004v1", "bias_in_objective": False}


def drone_factors(weight, inputs_columns, rank, *, rcond=None):
    """Thin-support DRONE factors; no W, M or X is needed by their forward.

    Z=Sigma_W V_W^T U_X Sigma_X and Z_k=U_Z Sigma_Z V_Z^T.
    L=U_W U_Z sqrt(Sigma_Z), R=sqrt(Sigma_Z) V_Z^T Sigma_X^-1 U_X^T.
    This cancels W V_W Sigma_W^-1 analytically instead of inverting W.
    Source: NeurIPS 2021 DRONE, Theorem 1, equations 4-5.
    """
    _matrix(weight, "weight")
    _matrix(inputs_columns, "inputs_columns")
    _requested_rank(rank, weight.shape)
    if inputs_columns.shape[0] != weight.shape[1]:
        raise NumericalDomainViolation("DRONE inputs_columns must have shape [in, observations]")
    if rcond is not None:
        _number(rcond, "rcond")
    w, x = weight.detach().double(), inputs_columns.detach().to(device=weight.device, dtype=torch.float64)
    with torch.no_grad():
        uw, sw, vhw = _svd(w, "DRONE weight")
        ux, sx, _ = _svd(x, "DRONE inputs")
        relative_w = rcond if rcond is not None else torch.finfo(w.dtype).eps * max(w.shape)
        relative_x = rcond if rcond is not None else torch.finfo(x.dtype).eps * max(x.shape)
        cutoff_w, cutoff_x = _scalar(relative_w * float(sw.max()),"weight cutoff"), _scalar(relative_x * float(sx.max()),"input cutoff")
        kw, kx = sw > cutoff_w, sx > cutoff_x
        rw, rx = int(kw.sum()), int(kx.sum())
        z = (sw[kw, None] * vhw[kw]) @ (ux[:, kx] * sx[kx])
        left = w.new_zeros((w.shape[0], rank))
        right = w.new_zeros((rank, w.shape[1]))
        tail = 0.
        if rw and rx:
            uz, sz, vhz = _svd(z, "DRONE thin joint matrix")
            k = min(rank, len(sz))
            root = sz[:k].sqrt()
            left[:, :k] = (uw[:, kw] @ uz[:, :k]) * root
            right[:k] = (root[:, None] * vhz[:k] / sx[kx]) @ ux[:, kx].T
            tail = _scalar(sz[k:].square().sum(), "DRONE joint spectral tail")
        pair = _pair(left, right, weight, {})
        residual = _finite((w - pair.matrix().double()) @ x, "DRONE observed residual")
        return FactorPair(pair.left, pair.right,
                          {"method": "drone_thin_support", "source_version": "neurips2021",
                           "requested_rank": rank, "actual_rank": int(torch.linalg.matrix_rank(pair.matrix())),
                           "weight_support": rw, "input_support": rx, "joint_shape": list(z.shape),
                           "weight_cutoff": cutoff_w, "input_cutoff": cutoff_x, "rcond": rcond,
                           "weighted_error_squared": _scalar(residual.square().sum(), "DRONE error"),
                           "effective_spectral_tail_squared": tail,
                           "input_condition_number": _scalar(sx[kx].max()/sx[kx].min(), "input condition") if rx else None,
                           "nullspace_policy": "support_only", "factor_semantics": "actual_operator"})


def linear_least_squares_fit(features_rows, targets_rows, *, rcond=None, ridge=0.):
    """Solve min_C ||H C^T-Y||_F^2 + ridge ||C||_F^2 by thin SVD.

    The shared primitive accepts explicit H/Y, including selected-channel MLP
    designs. ``rcond`` truncates the design support; ridge names a different
    objective. Inputs, their storage, and caller model parameters are unchanged.
    """
    _matrix(features_rows, "features_rows")
    _matrix(targets_rows, "targets_rows")
    if len(features_rows) != len(targets_rows):
        raise NumericalDomainViolation("design and target observations must match")
    _number(ridge, "ridge")
    if rcond is not None:
        _number(rcond, "rcond")
    h = features_rows.detach().double()
    y = targets_rows.detach().to(device=h.device, dtype=torch.float64)
    with torch.no_grad():
        u, s, vh = _svd(h, "least-squares design")
        cutoff = (rcond if rcond is not None else torch.finfo(h.dtype).eps * max(h.shape)) * float(s.max())
        supported = s > cutoff
        inverse = torch.zeros_like(s)
        if ridge:
            inverse[supported] = s[supported] / _finite(s[supported].square() + ridge, "ridge spectrum")
        else:
            inverse[supported] = s[supported].reciprocal()
        c = _finite(((vh.T * inverse) @ u.T @ y).T, "least-squares coefficients").to(targets_rows)
        residual = _scalar((h @ c.double().T-y).square().sum(), "least-squares residual")
        objective = _scalar(residual + ridge * c.double().square().sum(), "least-squares objective")
        retained = s[supported]
        return LinearLeastSquaresFit(c, residual, objective, int(supported.sum()), rcond, float(ridge),
                                     _scalar(retained.max()/retained.min(), "design condition") if len(retained) else None,
                                     ("linear_least_squares_restricted_support_ridge" if rcond is not None else "linear_least_squares_ridge") if ridge else
                                     ("linear_least_squares_restricted_support" if rcond is not None else "linear_least_squares"))


def svdllm_v1_fit(weight, right, inputs_rows, *, rcond=None, ridge=0.):
    """Fit A against W X' with fixed B, never against an old global output.

    Source: https://arxiv.org/html/2403.07378v1, layer-wise parameter update.
    """
    _matrix(weight, "weight")
    _matrix(right, "fixed right factor")
    _matrix(inputs_rows, "current inputs_rows")
    if right.shape[1] != weight.shape[1] or inputs_rows.shape[1] != weight.shape[1]:
        raise NumericalDomainViolation("SVD-LLM v1 input axes disagree")
    x = inputs_rows.detach().to(device=weight.device,dtype=torch.float64)
    design=x @ right.detach().to(device=weight.device,dtype=torch.float64).T
    target=x @ weight.detach().double().T
    fit = linear_least_squares_fit(design,target,rcond=rcond,ridge=ridge)
    installed=fit.coefficients.to(weight)
    error=_scalar((design@installed.double().T-target).square().sum(),"installed v1 residual")
    return FactorPair(installed, right.detach().clone().to(weight),
                      {"method": "svdllm_v1_local_ls", "source_version": "2403.07378v1",
                       "objective": fit.objective, "target": "current_weight_times_current_input",
                       "residual_squared": error, "solve_residual_squared":fit.residual_squared,"ridge": fit.ridge,
                       "rcond": fit.rcond, "design_rank": fit.design_rank,
                       "condition_number": fit.condition_number, "right_factor_policy": "unchanged"})


def _metric_support(weight, moment, policy):
    _matrix(moment, "moment")
    if moment.shape != (weight.shape[1], weight.shape[1]):
        raise NumericalDomainViolation("moment must match input axis")
    if not isinstance(policy, MetricPolicy):
        raise NumericalDomainViolation("MetricPolicy required")
    _number(policy.ridge, "ridge")
    if policy.rcond is not None:
        _number(policy.rcond, "rcond")
    if policy.nullspace_policy not in ("support_only", "preserve_nullspace"):
        raise NumericalDomainViolation("explicit nullspace policy required")
    c = moment.detach().to(device=weight.device, dtype=torch.float64)
    tolerance = torch.finfo(c.dtype).eps * len(c) * max(1., float(c.abs().max())) * 16
    if not torch.allclose(c, c.T, atol=tolerance, rtol=0):
        raise NumericalDomainViolation("moment must be symmetric")
    symmetric = c if torch.equal(c,c.T) else c*.5+c.T*.5
    if symmetric is not c:
        symmetric.diagonal().copy_(c.diagonal())
    try:
        values, vectors = torch.linalg.eigh(symmetric)
    except torch.linalg.LinAlgError as error:
        raise NumericalDomainViolation("metric eigensolve failed to converge") from error
    _finite(values, "metric eigenvalues")
    if float(values.min()) < -tolerance:
        raise NumericalDomainViolation("moment must be positive semidefinite")
    values = _finite(values.clamp_min(0) + policy.ridge, "regularized eigenvalues")
    cutoff = (policy.rcond if policy.rcond is not None else torch.finfo(c.dtype).eps*len(c))*float(values.max())
    if not math.isfinite(cutoff):
        raise NumericalDomainViolation("metric cutoff exceeds the finite numerical domain")
    support = values > cutoff
    return vectors, values, support, cutoff


def svdllm_v2_factors(weight, moment, rank, policy=MetricPolicy()):
    """Corrected V2 support formula, explicitly different from Algorithm 2.

    D=W Q+ sqrt(Lambda+)=U Sigma V^T;
    A=U_r sqrt(Sigma_r), B=sqrt(Sigma_r) V_r^T Lambda+^-1/2 Q+^T.
    Algorithm 2 v1 omits V^T and uses the wrong inverse power; Theorem 3.1
    supplies the reconstruction. Full admissible rank preserves W explicitly.
    Source: https://arxiv.org/html/2503.12340v1 (Theorem 3.1).
    """
    _matrix(weight, "weight")
    _requested_rank(rank, weight.shape)
    with torch.no_grad():
        q, values, support, cutoff = _metric_support(weight, moment, policy)
        roots = values[support].sqrt()
        w = weight.detach().double()
        d = _finite((w @ q[:, support]) * roots, "V2 weighted support image")
        left, right = w.new_zeros((w.shape[0],rank)), w.new_zeros((rank,w.shape[1]))
        tail = 0.
        if bool(support.any()):
            u, s, vh = _svd(d, "V2 support image")
            k = min(rank,len(s))
            root_s = s[:k].sqrt()
            left[:, :k] = u[:, :k]*root_s
            right[:k] = (root_s[:,None]*vh[:k]/roots) @ q[:,support].T
            tail = _scalar(s[k:].square().sum(), "V2 spectral tail")
        null_policy = policy.nullspace_policy
        if rank == min(w.shape) or policy.nullspace_policy == "preserve_nullspace":
            operator = w.clone() if rank == min(w.shape) else left@right+(w@q[:,~support])@q[:,~support].T
            if int(torch.linalg.matrix_rank(operator)) > rank:
                raise NumericalDomainViolation("preserving null-space action exceeds requested rank")
            u,s,vh = _svd(operator, "V2 preserved operator")
            left,right = u[:,:rank]*s[:rank].sqrt(), s[:rank,None].sqrt()*vh[:rank]
            if rank == min(w.shape):
                null_policy = "preserve_operator_at_full_rank"
        pair = _pair(left,right,weight,{})
        error = _scalar(((w-pair.matrix().double())@q[:,support]*roots).square().sum(), "V2 error")
        return FactorPair(pair.left,pair.right,
                          {"method":"svdllm_v2_corrected_support", "source_version":"2503.12340v1",
                           "formula":"theorem_3_1_corrected_v_transpose_inverse_square_root",
                           "requested_rank":rank, "actual_rank":int(torch.linalg.matrix_rank(pair.matrix())),
                           "moment_rank":int(support.sum()), "effective_cutoff":cutoff,
                           "ridge":float(policy.ridge), "rcond":policy.rcond, "nullspace_policy":null_policy,
                           "spectral_tail_squared":tail,"weighted_error_squared":error,
                           "discarded_metric_mass":_scalar(values[~support].sum(),"discarded mass"),
                           "objective":"input_second_moment_ridge" if policy.ridge else "input_second_moment"})


def inverse_log_allocation(errors: Sequence[float], removal_fraction: float, *, normalization=1.):
    """V2 inverse-log removal ratios on the explicit positive domain ell>1.

    Errors are raw (unsquared) Frobenius tails; normalization divides them.
    Refuse ell<=1, crossing 1, nonfinite scores and ratios outside [0,1].
    No clipping or automatic loss scaling changes the published heuristic.
    These continuous ratios do not themselves certify an integer cost budget.
    """
    _number(normalization,"normalization",positive=True)
    _number(removal_fraction,"removal_fraction")
    if removal_fraction>1 or not errors:
        raise NumericalDomainViolation("nonempty errors and removal_fraction in [0,1] required")
    losses=[]
    for error in errors:
        _number(error,"raw truncation error")
        ell=error/normalization
        if not math.isfinite(ell) or ell<=1:
            raise NumericalDomainViolation("published inverse-log profile requires every normalized ell>1")
        losses.append(ell)
    scores=[1/math.log(ell) for ell in losses]
    total=math.fsum(scores)
    ratios=tuple(len(scores)*removal_fraction*score/total for score in scores)
    if any(not math.isfinite(ratio) or not 0<=ratio<=1 for ratio in ratios):
        raise NumericalDomainViolation("inverse-log allocation leaves the admissible removal domain")
    return ratios, {"method":"svdllm_v2_inverse_log_positive_domain", "raw_errors":list(errors),
                    "normalization":normalization,"normalized_errors":losses,
                    "inverse_log_scores":scores,"removal_fractions":list(ratios),
                    "integer_budget_verified":False,"source_version":"2503.12340v1"}


def basis_sharing_factors(weights, moment, rank, *, metric_scales=None, policy=MetricPolicy()):
    """Minimize sum_i c_i ||(W_i-L_i R) C^1/2||² with one shared R.

    Stack sqrt(c_i) W_i along OUTPUT axes; split the fitted left factor and
    divide its blocks by sqrt(c_i). Arbitrary individual moments are not
    represented as an exact objective. Source: https://arxiv.org/html/2410.03765v1.
    """
    if not weights:
        raise NumericalDomainViolation("basis sharing needs at least one operator")
    for weight in weights:
        _matrix(weight,"group weight")
    first=weights[0]
    if any(w.shape[1]!=first.shape[1] or w.dtype!=first.dtype or w.device!=first.device for w in weights):
        raise NumericalDomainViolation("shared basis requires compatible input axes, dtype and device")
    scales=tuple(1. for _ in weights) if metric_scales is None else tuple(metric_scales)
    if len(scales)!=len(weights):
        raise NumericalDomainViolation("one positive metric scale per operator required")
    for scale in scales:
        _number(scale,"metric scale",positive=True)
    stacked=_finite(torch.cat([w.detach()*math.sqrt(c) for w,c in zip(weights,scales)],0),"stacked weights")
    result=svdllm_v2_factors(stacked,moment,rank,policy)
    blocks=result.left.split([len(w) for w in weights])
    lefts=tuple(_finite(block/math.sqrt(c),"shared coefficients") for block,c in zip(blocks,scales))
    return SharedBasisFactors(lefts,result.right,
                              {**result.diagnostics,"method":"basis_sharing_common_or_proportional_metric",
                               "metric_scales":list(scales),"objective":"sum_individual_proportional_metric_errors",
                               "shared_input_dimension":first.shape[1],"output_dimensions":[len(w) for w in weights],
                               "parameter_elements":rank*(first.shape[1]+sum(len(w) for w in weights)),
                               "source_version":"2410.03765v1","cross_depth_projection_cache":False})


def _groups(group_ids, vocabulary_size, group_count, device):
    if (not isinstance(group_ids,torch.Tensor) or group_ids.ndim!=1 or len(group_ids)!=vocabulary_size
            or group_ids.dtype not in (torch.int32,torch.int64) or type(group_count) is not int or group_count<1):
        raise NumericalDomainViolation("one integer group id per vocabulary token required")
    ids=group_ids.detach().to(device=device,dtype=torch.int64).clone()
    if bool(((ids<0)|(ids>=group_count)).any()):
        raise NumericalDomainViolation("token group id outside declared groups")
    return ids


def groupreduce_factors(weight, frequencies, group_ids, ranks, *, zero_policy="project"):
    """Frequency-weighted SVD in explicit vocabulary groups.

    Zero-frequency rows use their projection onto the group's chosen basis,
    without inventing pseudocounts. ``error`` rejects zeros instead. Empty groups
    require rank 0 and store no factors. Frequencies retain their caller units.
    Source: NeurIPS 2018 GroupReduce, equations 2-4.
    """
    _matrix(weight,"embedding weight")
    if (not isinstance(frequencies,torch.Tensor) or frequencies.ndim!=1 or len(frequencies)!=len(weight)
            or frequencies.dtype not in (torch.float32,torch.float64) or not bool(torch.isfinite(frequencies).all())
            or bool((frequencies<0).any())):
        raise NumericalDomainViolation("one finite nonnegative token frequency required")
    if zero_policy not in ("project","error"):
        raise NumericalDomainViolation("zero_policy must be project or error")
    if zero_policy=="error" and bool((frequencies==0).any()):
        raise NumericalDomainViolation("zero frequency rejected by declared policy")
    if not ranks:
        raise NumericalDomainViolation("explicit group ranks required")
    ids=_groups(group_ids,len(weight),len(ranks),weight.device)
    local=torch.empty_like(ids)
    tokens=[]; factors=[]; errors=[]
    w=weight.detach().double(); q=frequencies.detach().to(device=weight.device,dtype=torch.float64)
    with torch.no_grad():
        for group,rank in enumerate(ranks):
            selected=torch.nonzero(ids==group,as_tuple=False).flatten()
            n=len(selected)
            if type(rank) is not int or (rank!=0 if n==0 else not 1<=rank<=min(n,weight.shape[1])):
                raise NumericalDomainViolation("each nonempty group rank must fit N_p/D; empty group rank must be zero")
            local[selected]=torch.arange(n,device=ids.device)
            if n:
                _,s,vh=_svd(q[selected,None].sqrt()*w[selected],"GroupReduce weighted rows")
                right=vh[:rank]
                left=w[selected]@right.T
                pair=_pair(left,right,weight,{"group":group,"rank":rank})
                error=_scalar((q[selected,None]*(w[selected]-pair.matrix().double()).square()).sum(),"group error")
            else:
                pair=FactorPair(weight.new_empty((0,0)),weight.new_empty((0,weight.shape[1])),{"group":group,"rank":0})
                error=0.
            tokens.append(selected); factors.append(pair); errors.append(error)
    result=GroupReduceFactors(ids,local,tuple(tokens),tuple(factors),{})
    return GroupReduceFactors(ids,local,result.group_tokens,result.factors,
                              {"method":"groupreduce_frequency_weighted", "source_version":"neurips2018",
                               "zero_policy":zero_policy,"group_sizes":[len(t) for t in tokens],"ranks":list(ranks),
                               "group_weighted_errors_squared":errors,"weighted_error_squared":math.fsum(errors),
                               "parameter_elements":result.parameter_elements,"map_bytes":result.map_bytes,
                               "frequency_semantics":"caller_supplied_calibration_counts_or_probabilities"})


def frequency_groups(frequencies, group_count):
    """Deterministic equal-count frequency initialization, with token-id ties."""
    if (not isinstance(frequencies,torch.Tensor) or frequencies.ndim!=1 or not frequencies.numel()
            or frequencies.dtype not in (torch.float32,torch.float64) or not bool(torch.isfinite(frequencies).all())
            or bool((frequencies<0).any()) or type(group_count) is not int or not 1<=group_count<=len(frequencies)):
        raise NumericalDomainViolation("finite nonnegative frequencies and valid group_count required")
    order=torch.argsort(frequencies,descending=True,stable=True)
    result=torch.empty(len(order),device=order.device,dtype=torch.int64)
    for group,tokens in enumerate(torch.tensor_split(order,group_count)):
        result[tokens]=group
    return result


def groupreduce_transfer(weight, frequencies, state: GroupReduceFactors, *, max_transfers,
                         parameter_budget, rank_fractions=None):
    """One bounded refinement pass, with exact refit/cost after each move.

    Proposals use improvement in weighted row projection error; equal gains use
    token id then group id. Each accepted move is refitted and must not increase
    the total objective. Fixed-rank caps or explicit fractions determine new
    integer ranks after each change. A nonempty source is required, avoiding
    silent deletion of a rank allocation. This is a bounded deterministic
    GroupReduce variant, not a claim of globally optimal alternating clustering.
    """
    if type(max_transfers) is not int or max_transfers<0 or type(parameter_budget) is not int or parameter_budget<0:
        raise NumericalDomainViolation("nonnegative integer transfer limit and parameter budget required")
    caps=tuple(pair.rank for pair in state.factors)
    if rank_fractions is not None:
        rank_fractions=tuple(rank_fractions)
        if len(rank_fractions)!=len(caps) or any(type(f) not in (int,float) or not math.isfinite(f) or not 0<f<=1 for f in rank_fractions):
            raise NumericalDomainViolation("one rank fraction in (0,1] per group required")
    if state.parameter_elements>parameter_budget:
        raise NumericalDomainViolation("initial GroupReduce factors exceed parameter budget")
    current=groupreduce_factors(weight,frequencies,state.token_to_group,caps,
                               zero_policy=state.diagnostics.get("zero_policy","project"))
    moves=[]
    for _ in range(max_transfers):
        proposals=[]
        w=weight.detach().double()
        q=frequencies.detach().to(device=w.device,dtype=torch.float64)
        for token in range(len(w)):
            source=int(current.token_to_group[token])
            if len(current.group_tokens[source])<=1:
                continue
            source_basis=current.factors[source].right.double()
            old=float(q[token]*(w[token]-w[token]@source_basis.T@source_basis).square().sum())
            for target,pair in enumerate(current.factors):
                if target==source or pair.rank==0:
                    continue
                basis=pair.right.double()
                new=float(q[token]*(w[token]-w[token]@basis.T@basis).square().sum())
                if old>new:
                    proposals.append((-(old-new),token,target))
        accepted=False
        for negative_gain,token,target in sorted(proposals):
            ids=current.token_to_group.clone(); source=int(ids[token]); ids[token]=target
            sizes=[int((ids==g).sum()) for g in range(len(caps))]
            ranks=tuple((max(1,math.ceil(f*min(n,weight.shape[1]))) if n else 0)
                        for f,n in zip(rank_fractions,sizes)) if rank_fractions is not None else tuple(min(cap,n,weight.shape[1]) for cap,n in zip(caps,sizes))
            candidate=groupreduce_factors(weight,frequencies,ids,ranks,zero_policy=current.diagnostics["zero_policy"])
            if (candidate.parameter_elements<=parameter_budget
                    and candidate.diagnostics["weighted_error_squared"]<=current.diagnostics["weighted_error_squared"]):
                moves.append({"token":token,"source":source,"target":target,"proposed_gain":-negative_gain,
                              "group_sizes":sizes,"ranks":list(ranks),"parameter_elements":candidate.parameter_elements,
                              "weighted_error_squared":candidate.diagnostics["weighted_error_squared"]})
                current=candidate; accepted=True; break
        if not accepted:
            break
    return GroupReduceFactors(current.token_to_group,current.token_to_local,current.group_tokens,current.factors,
                              {**current.diagnostics,"refinement":"bounded_deterministic_refit",
                               "max_transfers":max_transfers,"transfers":moves,"parameter_budget":parameter_budget})
