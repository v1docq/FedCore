"""Bounded HASSLE-free and LoSparse mechanisms over existing FedCore primitives.

Independent implementation of arXiv:2502.00899v1 and 2306.11222v2. This CPU
profile checks the mechanisms, not the papers' LLM results or GPU throughput.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

from fedcore.algorithm.low_rank.approximation import solve_weighted
from fedcore.algorithm.low_rank.structured_profiles import balanced_factors
from fedcore.algorithm.low_rank.statistics import (
    StatisticsPassport, empty_second_moment, update_second_moment, finish_second_moment,
)
from .layers import PackedNMLinear, IndexedResidualLinear, _factors
from .specs import HASSLEFreeSpec, LoSparseSpec, NMStructure, SparseDomainError, plan_linear


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise SparseDomainError("numerical_domain", f"Nonfinite {name}")
    return value


def _linear_plan(layer, spec):
    if type(layer) is not nn.Linear:
        raise SparseDomainError("unsupported_operator", "An exact torch.nn.Linear is required")
    plan = plan_linear(layer.in_features, layer.out_features, spec,
                       dtype=str(layer.weight.dtype).split(".")[-1],
                       device=str(layer.weight.device), bias=layer.bias is not None)
    if (layer.weight.layout != torch.strided or layer.weight.shape != (layer.out_features, layer.in_features)
            or not bool(torch.isfinite(layer.weight).all())
            or (layer.bias is not None and (layer.bias.layout != torch.strided
                or layer.bias.shape != (layer.out_features,) or layer.bias.dtype != layer.weight.dtype
                or layer.bias.device != layer.weight.device or not bool(torch.isfinite(layer.bias).all())))):
        raise SparseDomainError("invalid_state", "Linear weight and bias must be finite ordinary tensors")
    return plan


def quadratic_objective(error, metric):
    """tr(E H E.T), with PyTorch E[out,in]; keeps autograd for factor tests."""
    return ((error @ metric) * error).sum()


def scale_coordinates(weight, metric):
    """W'=W D, H'=D^-1 H D^-1; undo both S and A by division by D."""
    scale = metric.diagonal().sqrt()
    if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise SparseDomainError("singular_metric", "Coordinate scaling needs a positive metric diagonal")
    return weight * scale, metric / scale[:, None] / scale[None, :], scale


@dataclass(frozen=True)
class SparseGPTResult:
    values: torch.Tensor
    indices: torch.Tensor
    objective: float
    algorithm: str = "sparsegpt_sequential_full_cholesky"


def sparsegpt_nm(weight, metric, structure: NMStructure):
    """Small full-Gram SparseGPT, one sequential block covering all columns.

    Uses the upper Cholesky factor of H^-1, group-wise OBS saliency and the
    sequential error feedback of SparseGPT (Frantar/Alistarh, Algorithm 1).
    Positive-definite H is required; there is no hidden damping/dead-column
    modification. This is not a globally optimal support search.
    """
    if (not isinstance(weight, torch.Tensor) or weight.ndim != 2
            or weight.dtype not in (torch.float32, torch.float64) or weight.device.type != "cpu"
            or not isinstance(metric, torch.Tensor) or metric.shape != (weight.shape[1], weight.shape[1])
            or metric.dtype not in (torch.float32, torch.float64) or metric.device.type != "cpu"):
        raise SparseDomainError("invalid_state", "SparseGPT requires CPU weight[out,in] and a matching full metric")
    plan_linear(weight.shape[1], weight.shape[0], structure,
                dtype=str(weight.dtype).split(".")[-1], bias=False)
    _finite(weight, "SparseGPT weight")
    _finite(metric, "SparseGPT metric")
    h = metric.detach().double()
    if not torch.allclose(h, h.T, atol=1e-12 * max(1., float(h.abs().max())), rtol=1e-12):
        raise SparseDomainError("invalid_metric", "SparseGPT metric must be symmetric")
    try:
        upper = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    except torch.linalg.LinAlgError as error:
        raise SparseDomainError("singular_metric", "SparseGPT requires a positive definite regularized metric") from error
    _finite(upper, "inverse-metric Cholesky")
    work = weight.detach().double().clone()
    kept = torch.ones_like(work, dtype=torch.bool)
    result = torch.zeros_like(work)
    for column in range(work.shape[1]):
        if column % structure.m == 0:
            saliency = work[:, column:column + structure.m].square() / upper.diagonal()[column:column + structure.m].square()
            # Stable ordering makes ties reproducible across calls; n means kept
            # slots, whereas SparseGPT's prunen argument means removed slots.
            removed = saliency.argsort(dim=1, stable=True)[:, :structure.m - structure.n]
            kept[:, column:column + structure.m].scatter_(1, removed, False)
        current = work[:, column].clone()
        quantized = torch.where(kept[:, column], current, 0.)
        result[:, column] = quantized
        error = (current - quantized) / upper[column, column]
        work[:, column:] -= error[:, None] * upper[column, column:][None, :]
    _finite(result, "SparseGPT output")
    indices = torch.arange(work.shape[1])[None, :].expand_as(work)[kept].reshape(
        work.shape[0], work.shape[1] // structure.m, structure.n)
    values = result.gather(1, indices.reshape(work.shape[0], -1)).reshape_as(indices).to(weight)
    installed = result.new_zeros(result.shape).scatter(1, indices.reshape(work.shape[0], -1),
                                                       values.double().reshape(work.shape[0], -1))
    objective = float(quadratic_objective(weight.detach().double() - installed, h))
    if not math.isfinite(objective):
        raise SparseDomainError("numerical_domain", "SparseGPT objective overflow")
    return SparseGPTResult(values, indices, objective)


def _dense_sparse(result, shape):
    # Temporary only during the local optimizer; never registered in a module.
    return result.values.new_zeros(shape).scatter(1, result.indices.reshape(shape[0], -1),
                                                result.values.reshape(shape[0], -1))


@dataclass(frozen=True)
class HASSLEEvidence:
    best_objective: float
    output_error_squared: float
    sparse_objectives: tuple[float, ...]
    factor_objectives: tuple[float, ...]
    normalization: str
    damping_rule: str
    damping_value: float
    observation_count: int
    scale: tuple[float, ...]
    seed: int
    method: str = "hassle_free"
    source: str = "arXiv:2502.00899v1"
    solver: str = "sparsegpt_sequential_full_cholesky_and_adam"
    kernel: str = "portable_cpu"
    limitations: tuple[str, ...] = ("bounded_cpu_full_gram", "no_global_optimum_guarantee", "no_acceleration_claim")


def hassle_free(layer, calibration, spec: HASSLEFreeSpec):
    """Alternate full-metric SparseGPT and Adam; keep the best checked iterate."""
    if not isinstance(spec, HASSLEFreeSpec):
        raise SparseDomainError("invalid_specification", "HASSLEFreeSpec required")
    _linear_plan(layer, spec)
    if (not isinstance(calibration, torch.Tensor) or calibration.ndim < 2
            or calibration.shape[-1] != layer.in_features or calibration.numel() == 0
            or calibration.dtype not in (torch.float32, torch.float64) or calibration.device.type != "cpu"
            or not bool(torch.isfinite(calibration).all())):
        raise SparseDomainError("invalid_calibration", "Finite CPU calibration rows must match Linear input width")
    rows = calibration.detach().reshape(-1, layer.in_features)
    passport = StatisticsPassport("local_linear", "local", "provided_current_inputs", "linear_input", "provided_tensor")
    state = update_second_moment(empty_second_moment(layer.in_features, passport), rows)
    moment = finish_second_moment(state)
    h0 = moment.matrix if spec.normalization == "mean" else moment.matrix * moment.passport.weight_sum
    damping_base = h0.diagonal().mean() if spec.damping_rule == "mean_diagonal" else h0.trace()
    damping = float(damping_base) * spec.damping
    h = h0 + torch.eye(layer.in_features, dtype=torch.float64) * damping
    _finite(h, "regularized Gram")
    w = layer.weight.detach().double()
    if spec.coordinate_scaling:
        scaled_w, scaled_h, scale = scale_coordinates(w, h)
    else:
        scaled_w, scaled_h, scale = w, h, torch.ones(layer.in_features, dtype=torch.float64)
    rank = spec.structure.rank
    generator = torch.Generator(device="cpu").manual_seed(spec.seed)
    left = nn.Parameter(torch.zeros((layer.out_features, rank), dtype=torch.float64))
    right = nn.Parameter(torch.randn((rank, layer.in_features), generator=generator, dtype=torch.float64)
                         / math.sqrt(layer.in_features))
    optimizer = torch.optim.Adam((left, right), lr=spec.learning_rate) if rank and spec.inner_steps else None
    best = None
    sparse_history, factor_history = [], []
    for iteration in range(spec.outer_steps):
        target = scaled_w - (left @ right).detach()
        packed = sparsegpt_nm(target, scaled_h, spec.structure)
        sparse = _dense_sparse(packed, target.shape)
        sparse_value = float(quadratic_objective(scaled_w - sparse - (left @ right).detach(), scaled_h))
        sparse_history.append(sparse_value)

        def retain(value):
            nonlocal best
            if not math.isfinite(value):
                raise SparseDomainError("numerical_domain", "HASSLE-free objective overflow")
            if best is None or value < best[0]:
                best = (value, packed.indices.clone(), sparse.clone(), left.detach().clone(), right.detach().clone())

        retain(sparse_value)
        if optimizer is not None:
            for group in optimizer.param_groups:
                group["lr"] = spec.learning_rate / (iteration + 1)
            for _ in range(spec.inner_steps):
                optimizer.zero_grad()
                loss = quadratic_objective(scaled_w - sparse - left @ right, scaled_h)
                _finite(loss, "HASSLE-free factor loss")
                loss.backward()
                _finite(left.grad, "left factor gradient")
                _finite(right.grad, "right factor gradient")
                optimizer.step()
                with torch.no_grad():
                    value = float(quadratic_objective(scaled_w - sparse - left @ right, scaled_h))
                factor_history.append(value)
                retain(value)
    _, indices, sparse, best_left, best_right = best
    # Undo the input-coordinate scale by DIVISION, for S and the right factor.
    sparse, best_right = sparse / scale, best_right / scale
    values = sparse.gather(1, indices.reshape(layer.out_features, -1)).reshape_as(indices).to(layer.weight)
    result = PackedNMLinear.from_tensors(values, indices, best_left.to(layer.weight), best_right.to(layer.weight),
                                         in_features=layer.in_features, bias=layer.bias, m=spec.structure.m)
    installed = _dense_sparse(SparseGPTResult(result.values, result.indices, 0.), w.shape).double()
    if result.factors is not None:
        installed += result.factors.left.weight.detach().double() @ result.factors.right.weight.detach().double()
    error = w - installed
    evidence = HASSLEEvidence(float(quadratic_objective(error, h)),
                             float((rows.double() @ error.T).square().sum()), tuple(sparse_history),
                             tuple(factor_history), spec.normalization, spec.damping_rule, damping,
                             len(rows), tuple(float(v) for v in scale), spec.seed)
    result.train(layer.training)
    return result, evidence


def cubic_residual_count(step, out_features, spec: LoSparseSpec):
    """Integer upper rounding of a cubic schedule, exact at both endpoints."""
    if type(step) is not int or not 0 <= step <= spec.total_steps:
        raise SparseDomainError("invalid_transition", "LoSparse step must fit the declared training budget")
    end = spec.total_steps if spec.final_step is None else spec.final_step
    if step <= spec.warmup_steps:
        return out_features
    if step >= end:
        return spec.residual_rows
    progress = (step - spec.warmup_steps) / (end - spec.warmup_steps)
    return min(out_features, max(spec.residual_rows,
               math.ceil(spec.residual_rows + (out_features - spec.residual_rows) * (1 - progress) ** 3)))


class LoSparseTrainingLinear(nn.Module):
    """Explicit dense training state; finalize before compact checkpoint/export.

    With reentry, a shadow residual retains inactive values and a straight-through
    mask provides inactive gradients; Adam state is retained. Without reentry,
    removed rows/gradients/moments are zeroed and support can only shrink.
    The dense training residual is float64; packing returns the source precision.
    """

    def __init__(self, layer, spec):
        super().__init__()
        if not isinstance(spec, LoSparseSpec):
            raise SparseDomainError("invalid_specification", "LoSparseSpec required")
        _linear_plan(layer, spec)
        self.in_features, self.out_features, self.rank = layer.in_features, layer.out_features, spec.rank
        self.spec = spec
        self.factors = _factors(self.in_features, self.out_features, spec.rank,
                                dtype=layer.weight.dtype, device=layer.weight.device)
        if spec.rank:
            identity = torch.eye(self.in_features, dtype=torch.float64)
            pair = balanced_factors(solve_weighted(layer.weight, identity, spec.rank), method="losparse_init")
            with torch.no_grad():
                self.factors.left.weight.copy_(pair.left)
                self.factors.right.weight.copy_(pair.right)
            # Use the installed (possibly rounded) factors.  A double residual
            # reconstructs the original float32 weight without retaining W.
            approximation = (self.factors.left.weight.detach().double()
                             @ self.factors.right.weight.detach().double())
        else:
            approximation = torch.zeros_like(layer.weight, dtype=torch.float64)
        self.residual = nn.Parameter((layer.weight.detach().double() - approximation) / spec.initial_scale)
        # Rank zero has no factor from which to recover the final precision.
        self.output_dtype = layer.weight.dtype
        self.bias = nn.Parameter(layer.bias.detach().clone()) if layer.bias is not None else None
        self.register_buffer("support", torch.ones(self.out_features, dtype=torch.bool, device=layer.weight.device))
        self.register_buffer("importance_ema", torch.zeros_like(layer.weight))
        self.register_buffer("completed_steps", torch.tensor(0, dtype=torch.int64, device=layer.weight.device))
        self.train(layer.training)

    def forward(self, x):
        if bool(self.support.all()) and int(self.completed_steps) < self.spec.total_steps:
            effective = self.residual * self.spec.initial_scale
            if self.factors is not None:
                effective = effective + (self.factors.left.weight.double()
                                         @ self.factors.right.weight.double())
            return F.linear(x, effective.to(x.dtype), self.bias)
        # This order, including the cast AFTER scaling, is shared with packing.
        result = (x.new_zeros((*x.shape[:-1], self.out_features))
                  if self.factors is None else self.factors(x))
        indices = torch.nonzero(self.support, as_tuple=False).flatten()
        if indices.numel():
            rows = (self.residual.index_select(0, indices) * self.spec.initial_scale).to(x.dtype)
            result = result.index_add(-1, indices, F.linear(x, rows))
        else:
            # Keep a real zero gradient after the last row is removed; the
            # schedule callback still observes an actual differentiable loss.
            result = result + (self.residual.sum() * 0).to(x.dtype)
        if self.spec.allow_reentry:
            # Exactly zero forward contribution; inactive rows keep the full
            # straight-through gradient so their shadow values may re-enter.
            inactive = (~self.support)[:, None]
            shadow = ((self.residual - self.residual.detach()) * inactive
                      * self.spec.initial_scale).to(x.dtype)
            result = result + F.linear(x, shadow)
        return result if self.bias is None else result + self.bias


def prepare_losparse(layer, spec: LoSparseSpec):
    return LoSparseTrainingLinear(layer, spec)


class LoSparseTrainingCallback:
    """Context-managed objective for existing runner.train_model hooks.

    loss_fn(model,x,y) supplies the task loss; this object owns no trainer,
    optimizer, loader, forward hooks or independent data selection.
    """

    def __init__(self, modules, loss_fn):
        self.modules = (modules,) if isinstance(modules, LoSparseTrainingLinear) else tuple(modules)
        if (not self.modules or any(not isinstance(m, LoSparseTrainingLinear) for m in self.modules)
                or len({id(m) for m in self.modules}) != len(self.modules) or not callable(loss_fn)):
            raise SparseDomainError("invalid_specification", "Distinct LoSparse training modules and a task loss required")
        self.loss_fn = loss_fn
        self.active = False
        self.step_pending = False

    @property
    def evidence(self):
        return {"method": "losparse", "completed_steps": [int(m.completed_steps) for m in self.modules],
                "support": [torch.nonzero(m.support, as_tuple=False).flatten().tolist() for m in self.modules],
                "allow_reentry": [m.spec.allow_reentry for m in self.modules],
                "optimizer_inactive_state": ["retained_shadow" if m.spec.allow_reentry else "zeroed" for m in self.modules],
                "dense_training_residual_bytes": sum(m.residual.numel() * m.residual.element_size() for m in self.modules),
                "dense_training_ema_bytes": sum(m.importance_ema.numel() * m.importance_ema.element_size() for m in self.modules),
                "finalization_required": True}

    def __enter__(self):
        if self.active:
            raise SparseDomainError("invalid_transition", "LoSparse callback is already active")
        self.active = True
        self.step_pending = False
        return self

    def __exit__(self, *exc):
        self.active = False
        self.step_pending = False

    def __call__(self, model, x, y):
        if not self.active:
            raise SparseDomainError("invalid_transition", "Enter the LoSparse objective context before training")
        live = {id(m) for m in model.modules()}
        if any(id(m) not in live for m in self.modules):
            raise SparseDomainError("stale_training_module", "LoSparse target is absent from the live model")
        if any(int(m.completed_steps) >= m.spec.total_steps for m in self.modules):
            raise SparseDomainError("invalid_transition", "LoSparse training budget is exhausted")
        return self.loss_fn(model, x, y)

    def before_step(self, model, optimizer):
        # Validate all targets before updating any EMA.
        if self.step_pending:
            raise SparseDomainError("invalid_transition", "A LoSparse optimizer step is already pending")
        sensitivities = []
        for module in self.modules:
            if module.residual.grad is None or not bool(torch.isfinite(module.residual.grad).all()):
                raise SparseDomainError("missing_residual_gradient", "LoSparse requires a finite real residual gradient")
            if not self.active or int(module.completed_steps) >= module.spec.total_steps:
                raise SparseDomainError("invalid_transition", "LoSparse update is outside the active declared budget")
            sensitivities.append(_finite((module.residual.detach() * module.residual.grad).abs(), "residual sensitivity"))
        with torch.no_grad():
            for module, sensitivity in zip(self.modules, sensitivities):
                module.importance_ema.mul_(module.spec.ema_beta).add_(sensitivity, alpha=1 - module.spec.ema_beta)
                if not module.spec.allow_reentry:
                    module.residual.grad.mul_(module.support[:, None])
        self.step_pending = True

    def after_step(self, model, optimizer):
        if not self.active or not self.step_pending or any(int(m.completed_steps) >= m.spec.total_steps for m in self.modules):
            raise SparseDomainError("invalid_transition", "LoSparse update is outside the active declared budget")
        for module in self.modules:
            _finite(module.residual, "updated residual")
            if module.factors is not None:
                _finite(module.factors.left.weight, "updated left factor")
                _finite(module.factors.right.weight, "updated right factor")
        with torch.no_grad():
            for module in self.modules:
                step = int(module.completed_steps) + 1
                count = cubic_residual_count(step, module.out_features, module.spec)
                end = module.spec.total_steps if module.spec.final_step is None else module.spec.final_step
                if step % module.spec.delta_t == 0 or step in (end, module.spec.total_steps):
                    scores = module.importance_ema.mean(dim=1)
                    if not module.spec.allow_reentry:
                        scores = scores.masked_fill(~module.support, -torch.inf)
                    chosen = scores.argsort(descending=True, stable=True)[:count]
                    module.support.zero_().scatter_(0, chosen, True)
                if not module.spec.allow_reentry:
                    mask = module.support[:, None]
                    module.residual.mul_(mask)
                    for state in optimizer.state.get(module.residual, {}).values():
                        if isinstance(state, torch.Tensor) and state.shape == module.residual.shape:
                            state.mul_(mask)
                module.completed_steps.fill_(step)
        self.step_pending = False


def finalize_losparse(module, *, require_complete=True):
    """Copy only saved support rows; the returned module owns no dense S/EMA."""
    if not isinstance(module, LoSparseTrainingLinear):
        raise SparseDomainError("unsupported_operator", "LoSparseTrainingLinear required")
    if require_complete and (int(module.completed_steps) != module.spec.total_steps
            or int(module.support.sum()) != module.spec.residual_rows):
        raise SparseDomainError("incomplete_training", "Complete the declared LoSparse schedule before finalization")
    indices = torch.nonzero(module.support, as_tuple=False).flatten()
    if module.factors is None:
        left = module.residual.new_empty((module.out_features, 0), dtype=module.output_dtype)
        right = module.residual.new_empty((0, module.in_features), dtype=module.output_dtype)
    else:
        left, right = module.factors.left.weight.detach(), module.factors.right.weight.detach()
    result = IndexedResidualLinear.from_tensors(left, right,
            (module.residual.detach().index_select(0, indices) * module.spec.initial_scale).to(right.dtype),
            indices, bias=module.bias)
    result.train(module.training)
    return result
