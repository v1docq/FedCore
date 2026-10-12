"""Bounded CPU RankDyna adaptation with real structural rank removal.

Primary source: Hua et al., Findings of EMNLP 2023, ACL final version,
2023.findings-emnlp.621, sections 3.2--3.5, equations 5--12, algorithm 1.
This small observed-label implementation does not reproduce the paper's
Transformer benchmarks. Free U/S/V training preserves neither orthogonality
nor an ordered singular spectrum; retained width is only a rank upper bound.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import math

import torch
from torch import nn
from torch.nn import functional as F

from .rankdyna_specs import (RankDyna, RankDynaID, initial_rankdyna_state,
                            update_rankdyna_ema, advance_rankdyna)
from .structured_layers import FactorizedLinear
from .structured_profiles import FactorPair
from .topology import inspect_topology, module_paths, replace_modules_atomically


RANKDYNA_SOURCE = "https://aclanthology.org/2023.findings-emnlp.621.pdf"
RANKDYNA_VERSION = "2023.findings-emnlp.621_acl_final"


class RankDynaProfileError(ValueError):
    """The bounded profile cannot preserve its training/export contract."""


def _finite(tensor, name):
    if not bool(torch.isfinite(tensor).all()):
        raise RankDynaProfileError(f"{name} must be finite")


def _selected(model, paths, spec, *, finite=False):
    if not isinstance(model, nn.Module) or not isinstance(spec, RankDyna):
        raise RankDynaProfileError("An owned model and RankDyna description are required")
    if not isinstance(paths, (tuple, list)):
        raise RankDynaProfileError("A sequence of explicit selected paths is required")
    paths = tuple(paths)
    if not paths or any(type(path) is not str for path in paths) or len(set(paths)) != len(paths):
        raise RankDynaProfileError("Distinct explicit selected paths are required")
    modules = module_paths(model)
    topology = inspect_topology(model)
    selected = set(paths)
    if any(any(path in selected for path in aliases) for aliases in topology.module_aliases):
        raise RankDynaProfileError("Selected module aliases are unsupported")
    def affected(parameter_path):
        owner = parameter_path.rpartition(".")[0]
        return any(path == "" or owner == path or owner.startswith(path+".") for path in paths)
    if (any(any(affected(name) for name in aliases) for aliases in topology.parameter_aliases)
            or any(any(affected(name) for name in aliases.parameter_paths) for aliases in topology.storage_aliases)):
        raise RankDynaProfileError("Selected tied Parameters and shared storage views are unsupported")
    for name, parameter in model.named_parameters():
        if parameter.device.type != "cpu" or parameter.dtype not in (torch.float32, torch.float64):
            raise RankDynaProfileError(f"Owned CPU float32/float64 model required: {name}")
    if any(buffer.device.type != "cpu" or
           (buffer.is_floating_point() and buffer.dtype not in (torch.float32, torch.float64))
           for buffer in model.buffers()):
        raise RankDynaProfileError("Owned model buffers must use CPU and supported floating dtypes")
    layers = {}
    for path in paths:
        layer = modules.get(path)
        if type(layer) is not nn.Linear or layer._modules:
            raise RankDynaProfileError(f"Selected path {path!r} must be an exact nn.Linear")
        if layer._forward_hooks or layer._forward_pre_hooks or layer._backward_hooks:
            raise RankDynaProfileError("Selected Linear hooks cannot be silently discarded")
        if layer.weight.shape != (layer.out_features, layer.in_features) or min(layer.weight.shape) < 1:
            raise RankDynaProfileError("Selected Linear dimensions are inconsistent")
        if finite:
            _finite(layer.weight, "initial weight")
            if layer.bias is not None:
                _finite(layer.bias, "initial bias")
        layers[path] = layer
    return layers


def estimate_rankdyna_bytes(model, paths, spec):
    """Reserve named tensor bounds from shapes before SVD/factor allocation.

    Native LAPACK scratch is a conservative reservation, not an allocator's
    hard RSS guarantee. The caller also owns whole-model copies, activations,
    objective outputs, Python metadata and optimizer object overhead.
    """
    layers = _selected(model, paths, spec)
    factor_bytes = bias_bytes = group_count = 0
    svd_input = svd_output = largest_weight = largest_factors = 0
    for layer in layers.values():
        m, n = layer.weight.shape
        rank = min(m, n)
        itemsize = layer.weight.element_size()
        factors = rank*(m+n+1)*itemsize
        matrix = m*n*itemsize
        factor_bytes += factors
        if spec.freeze == "factors_and_bias" and layer.bias is not None:
            bias_bytes += layer.bias.numel()*itemsize
        group_count += rank
        svd_input = max(svd_input, matrix)
        svd_output = max(svd_output, factors)
        largest_weight = max(largest_weight, matrix)
        largest_factors = max(largest_factors, factors)
    trained = factor_bytes+bias_bytes
    return {
        "factor_parameters_bytes": factor_bytes,
        "trainable_bias_bytes": bias_bytes,
        "factor_gradients_bytes": trained,
        "adam_moments_bytes": 3*trained,  # includes optional AMSGrad maximum
        "adam_steps_bytes": 8*(3*len(layers)+sum(
            layer.bias is not None for layer in layers.values()) if spec.freeze == "factors_and_bias" else 3*len(layers)),
        "svd_input_copy_bytes": svd_input,
        "svd_outputs_bytes": svd_output,
        "svd_workspace_reserve_bytes": 16*(largest_weight+largest_factors),
        "importance_temporaries_bytes": largest_factors+group_count*8,
        # Old/new Parameters, gradients and two Adam moments coexist while a
        # complete replacement plan is prepared. AMSGrad adds a third moment.
        "removal_copy_bytes": 5*factor_bytes,
        "export_copy_bytes": 3*factor_bytes,
    }


class RankDynaLinear(nn.Module):
    """Transient trainable sum of rank-one groups; never a portable checkpoint."""

    @property
    def weight(self):
        from .learned_layers import UnsupportedLearnedAccessError
        raise UnsupportedLearnedAccessError('P4 requires module calls; functional dense weight access is unsupported')

    def __init__(self, layer, path):
        super().__init__()
        with torch.no_grad():
            u, s, vh = torch.linalg.svd(layer.weight.detach().clone(), full_matrices=False)
        for tensor in (u, s, vh):
            _finite(tensor, "SVD initialization")
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.u = nn.Parameter(u.contiguous().clone())
        self.s = nn.Parameter(s.contiguous().clone())
        self.vh = nn.Parameter(vh.contiguous().clone())
        self.bias = None if layer.bias is None else nn.Parameter(layer.bias.detach().clone())
        self.active_ids = tuple(RankDynaID(path, i) for i in range(len(s)))
        self.train(layer.training)

    @property
    def rank(self):
        return len(self.active_ids)

    def forward(self, inputs):
        return F.linear(F.linear(inputs, self.vh)*self.s, self.u, self.bias)


def _forward(model, inputs):
    if isinstance(inputs, Mapping):
        return model(**inputs)
    if isinstance(inputs, tuple):
        return model(*inputs)
    return model(inputs)


def _ids_json(identities):
    return [[identity.path, identity.initial_column] for identity in identities]


def _optimizer_step(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RankDynaProfileError("Adam step state must be scalar")
        value = value.item()
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise RankDynaProfileError("Adam step state must be finite and nonnegative")
    return float(value)


def _optimizer_bytes(optimizer):
    return sum(value.numel()*value.element_size() for state in optimizer.state.values()
               for value in state.values() if isinstance(value, torch.Tensor))


class RankDynaObjective:
    """Own EMA/removal state; the caller owns batches and actual Adam steps.

    Required order: loss.backward(), before_step(), optimizer.step(),
    after_step(). The same optimizer stays live across structural reductions.
    """

    def __init__(self, model, paths, spec, reservations):
        self.model = model
        self.paths = tuple(paths)
        self.spec = spec
        self.reservations = dict(reservations)
        modules = module_paths(model)
        self.state = initial_rankdyna_state(tuple(
            (path, modules[path].rank, modules[path].in_features+modules[path].out_features)
            for path in self.paths))
        self.history = []
        self._pending = None
        self._finalized = False
        self._export_record = None
        self._trainable_names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
        self._optimizer_state_bytes = 0
        self._peak_optimizer_state_bytes = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def _layers(self, model):
        if model is not self.model or self._finalized:
            raise RankDynaProfileError("Use the live owned RankDyna model before finalization")
        modules = module_paths(model)
        layers = {}
        active = set(self.state.active_ids)
        for path in self.paths:
            layer = modules.get(path)
            if (type(layer) is not RankDynaLinear
                    or layer.active_ids != tuple(i for i in self.state.active_ids if i.path == path)
                    or layer.u.shape != (layer.out_features, layer.rank)
                    or layer.s.shape != (layer.rank,)
                    or layer.vh.shape != (layer.rank, layer.in_features)
                    or not set(layer.active_ids) <= active):
                raise RankDynaProfileError("RankDyna factor topology or active IDs changed")
            layers[path] = layer
        return layers

    def _optimizer(self, model, optimizer):
        self._layers(model)
        if type(optimizer) not in (torch.optim.Adam, torch.optim.AdamW):
            raise RankDynaProfileError("Structural state migration supports exact Adam/AdamW optimizers")
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        if tuple(name for name, p in model.named_parameters() if p.requires_grad) != self._trainable_names:
            raise RankDynaProfileError("Declared factor/bias training boundary changed")
        actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise RankDynaProfileError("Optimizer has stale, duplicate, missing or unauthorized Parameters")
        for layer in self._layers(model).values():
            if not all(p.requires_grad for p in (layer.u, layer.s, layer.vh)):
                raise RankDynaProfileError("Every live U/S/V factor must remain trainable")
        return tuple(p for group in optimizer.param_groups for p in group["params"])

    def __call__(self, model, inputs, labels):
        self._layers(model)
        if self._pending is not None or self.state.completed_updates >= self.spec.updates:
            raise RankDynaProfileError("Finish the pending step or the declared horizon first")
        output = _forward(model, inputs)
        if not isinstance(output, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise RankDynaProfileError("A tensor output and observed tensor labels are required")
        if self.spec.loss == "mse":
            if output.shape != labels.shape:
                raise RankDynaProfileError("MSE labels must match output shape without broadcasting")
            loss = F.mse_loss(output, labels, reduction=self.spec.reduction)
        else:
            loss = F.cross_entropy(output, labels, reduction=self.spec.reduction)
        if loss.ndim != 0:
            raise RankDynaProfileError("The RankDyna objective must be scalar")
        _finite(loss, "observed-label loss")
        return loss

    @property
    def evidence(self):
        record = {
            "method": "rankdyna", "source": RANKDYNA_SOURCE,
            "source_version": RANKDYNA_VERSION,
            "source_doi": "10.18653/v1/2023.findings-emnlp.621",
            "author_repository": None,
            "implementation": "bounded_cpu_observed_label_adaptation",
            "benchmark_reproduction": False, "spec": asdict(self.spec),
            "initial_groups": self.state.initial_groups,
            "paper_target_groups": math.ceil(self.spec.target_fraction*self.state.initial_groups),
            "active_groups": len(self.state.groups),
            "active_ids": _ids_json(self.state.active_ids),
            "removed_ids": _ids_json(self.state.removed_ids),
            "completed_updates": self.state.completed_updates,
            "ema": [[g.identity.path, g.identity.initial_column, g.ema] for g in self.state.groups],
            "importance": "sum_abs_theta_times_gradient_for_U_S_V",
            "ema_initialization": "zero_without_bias_correction",
            "observation_timing": "after_backward_before_optimizer_step",
            "removal_timing": "after_optimizer_step",
            "removal": "irreversible_physical_column_row_shrink",
            "tie_break": "path_then_initial_column_ascending",
            "schedule": "boundary_consistent_cubic_prose_variant",
            "schedule_note": "Eq12_printed_t_minus_t0_minus_t1_does_not_match_phase_boundaries",
            "parameter_budget_extension": self.spec.parameter_budget is not None,
            "parameter_budget_scope": "selected_exported_factors_excluding_bias",
            "parameter_budget_selection": "greedy_lowest_ema_removal_not_knapsack_optimal",
            "orthogonality_preserved": False, "ordered_singular_spectrum": False,
            "rank_interpretation": "retained_group_width_upper_bound",
            "resource_reservation": dict(self.reservations),
            "optimizer_state_bytes": self._optimizer_state_bytes,
            "peak_optimizer_state_bytes": self._peak_optimizer_state_bytes,
            "history": list(self.history),
            "finalized": self._finalized,
        }
        if self._export_record is not None:
            record["export"] = dict(self._export_record)
        return record

    def before_step(self, model, optimizer):
        if self._pending is not None or self.state.completed_updates >= self.spec.updates:
            raise RankDynaProfileError("Exactly one before_step is allowed per declared update")
        parameters = self._optimizer(model, optimizer)
        for parameter in parameters:
            if parameter.grad is None:
                raise RankDynaProfileError("Every declared trainable Parameter needs a backward gradient")
            _finite(parameter, "trainable Parameter")
            _finite(parameter.grad, "backward gradient")
        scores = []
        with torch.no_grad():
            for layer in self._layers(model).values():
                # Elementwise absolute values precede group summation. Taking
                # abs after a dot product would incorrectly permit cancellation.
                importance = ((layer.u*layer.u.grad).abs().sum(0, dtype=torch.float64)
                              +(layer.s*layer.s.grad).abs().double()
                              +(layer.vh*layer.vh.grad).abs().sum(1, dtype=torch.float64))
                _finite(importance, "rank-one group importance")
                scores.extend((identity, float(score)) for identity, score in zip(layer.active_ids, importance))
        observed = update_rankdyna_ema(self.state, tuple(scores), self.spec.beta)
        steps = tuple((parameter, _optimizer_step(optimizer.state.get(parameter, {}).get("step", 0)))
                      for parameter in parameters)
        self._pending = (observed, optimizer, steps)
        self.state = observed

    def after_step(self, model, optimizer):
        parameters = self._optimizer(model, optimizer)
        layers = self._layers(model)
        if self._pending is None or self._pending[1] is not optimizer:
            raise RankDynaProfileError("after_step requires the matching before_step and optimizer")
        observed, _, steps = self._pending
        for parameter, old_step in steps:
            new_step = _optimizer_step(optimizer.state.get(parameter, {}).get("step", 0))
            if new_step != old_step+1:
                raise RankDynaProfileError("Exactly one actual Adam step must occur between hooks")
        for parameter in parameters:
            _finite(parameter, "updated Parameter")
        self._peak_optimizer_state_bytes = max(self._peak_optimizer_state_bytes, _optimizer_bytes(optimizer))
        removal = advance_rankdyna(observed, self.spec)
        keep_ids = set(removal.state.active_ids)
        replacements = {}
        assignments = []
        # Prepare and validate the complete state migration before changing any
        # module, optimizer group, or optimizer state mapping.
        for layer in layers.values():
            kept = tuple(i for i in layer.active_ids if i in keep_ids)
            if kept == layer.active_ids:
                continue
            positions = torch.tensor([j for j, i in enumerate(layer.active_ids) if i in keep_ids], dtype=torch.long)
            for name, axis in (("u", 1), ("s", 0), ("vh", 0)):
                old = getattr(layer, name)
                new = nn.Parameter(old.detach().index_select(axis, positions).contiguous(), requires_grad=True)
                if old.grad is not None:
                    new.grad = old.grad.detach().index_select(axis, positions).contiguous()
                state = {}
                for key, value in optimizer.state.get(old, {}).items():
                    if isinstance(value, torch.Tensor):
                        if key == "step" and value.numel() == 1:
                            state[key] = value.detach().clone()
                        elif key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq") and value.shape == old.shape:
                            state[key] = value.detach().index_select(axis, positions).contiguous()
                        else:
                            raise RankDynaProfileError(f"Unsupported Adam tensor state {key!r}")
                    elif key == "step" and type(value) in (int, float):
                        state[key] = value
                    else:
                        raise RankDynaProfileError(f"Unsupported Adam state {key!r}")
                replacements[old] = (new, state)
                assignments.append((layer, name, new))
        groups = [[replacements[p][0] if p in replacements else p for p in group["params"]]
                  for group in optimizer.param_groups]
        for layer, name, new in assignments:
            layer._parameters[name] = new
        for group, new_parameters in zip(optimizer.param_groups, groups):
            group["params"] = new_parameters
        for old, (new, state) in replacements.items():
            optimizer.state.pop(old, None)
            optimizer.state[new] = state
        for layer in layers.values():
            layer.active_ids = tuple(i for i in layer.active_ids if i in keep_ids)
        self.state = removal.state
        self._optimizer_state_bytes = _optimizer_bytes(optimizer)
        self._pending = None
        self.history.append({
            "step": self.state.completed_updates,
            "scheduled": removal.scheduled,
            "scheduled_group_budget": removal.scheduled_group_budget,
            "scheduled_parameter_budget": removal.scheduled_parameter_budget,
            "active_groups": len(self.state.groups),
            "exported_factor_elements": sum(g.exported_elements for g in self.state.groups),
            "ranks": {path: sum(i.path == path for i in self.state.active_ids) for path in self.paths},
            "removed_ids": _ids_json(removal.removed_ids),
            "optimizer_migrated_parameters": len(replacements),
        })
        self._optimizer(model, optimizer)

    def finalize(self, model, calibration, batch_size):
        layers = self._layers(model)
        if self._pending is not None or self.state.completed_updates != self.spec.updates:
            raise RankDynaProfileError("Finalize only after all declared actual updates and removals")
        if type(batch_size) is not int or batch_size < 1:
            raise RankDynaProfileError("A positive export verification batch size is required")
        if (not isinstance(calibration, torch.Tensor) or calibration.device.type != "cpu"
                or calibration.ndim < 1 or len(calibration) == 0):
            raise RankDynaProfileError("Nonempty CPU tensor calibration is required for export verification")
        replacements = {}
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.spec.seed)
            for path, layer in layers.items():
                if layer.rank:
                    pair = FactorPair((layer.u.detach()*layer.s.detach()[None, :]).contiguous(),
                                      layer.vh.detach().clone(), {"method": "rankdyna"})
                    compact = FactorizedLinear.from_factors(pair, layer.bias)
                else:
                    # Root supplies the portable, bias-only rank-zero operator.
                    from .learned_layers import ZeroLinear
                    compact = ZeroLinear(layer.in_features, layer.out_features, layer.bias is not None,
                                         device=layer.u.device, dtype=layer.u.dtype)
                    if layer.bias is not None:
                        compact.bias.copy_(layer.bias)
                compact.requires_grad_(False)
                compact.eval()
                replacements[path] = compact
        old_modes = {module: module.training for module in model.modules()}
        model.eval()
        max_error = 0.0
        # Only a single pair of prediction batches is retained. Temporary swaps
        # are restored even if numerical verification rejects the export.
        try:
            with torch.no_grad():
                for batch in calibration.split(batch_size):
                    reference = _forward(model, batch)
                    if not isinstance(reference, torch.Tensor):
                        raise RankDynaProfileError("Export verification requires tensor model outputs")
                    candidate = replace_modules_atomically(model, replacements)
                    try:
                        actual = _forward(candidate, batch)
                        if not isinstance(actual, torch.Tensor) or actual.shape != reference.shape:
                            raise RankDynaProfileError("Export changed the model output contract")
                        _finite(reference, "pre-export output")
                        _finite(actual, "post-export output")
                        error = float((reference-actual).abs().max()) if reference.numel() else 0.0
                        scale = max(1.0, float(reference.abs().max())) if reference.numel() else 1.0
                        tolerance = 128*torch.finfo(reference.dtype).eps*scale
                        if error > tolerance:
                            raise RankDynaProfileError("Absorbing S failed numerical output preservation")
                        max_error = max(max_error, error)
                    finally:
                        replace_modules_atomically(candidate, layers)
        finally:
            for module, training in old_modes.items():
                module.training = training
        self.model = replace_modules_atomically(model, replacements)
        self._finalized = True
        self._export_record = {
            "representation": "FactorizedLinear_or_ZeroLinear",
            "train_s_absorbed_into_left": True,
            "max_abs_output_error": max_error,
            "verification_examples": len(calibration),
            "verification_batch_size": batch_size,
            "operator_preservation": "algebraic_with_dtype_scaled_numerical_check",
            "rank_upper_bounds": {path: layer.rank for path, layer in layers.items()},
            "exported_factor_elements": sum(g.exported_elements for g in self.state.groups),
            "optimizer_reusable": False,
        }
        return dict(self._export_record)


def prepare_rankdyna(model, paths, spec):
    """Replace only an OWNED CPU model; the caller must clone its source first.

    All tensor allocations and SVDs finish before the complete module write
    set is committed. No full masks are retained, and no removed group returns.
    """
    layers = _selected(model, paths, spec, finite=True)
    reservations = estimate_rankdyna_bytes(model, tuple(layers), spec)
    replacements = {path: RankDynaLinear(layer, path) for path, layer in layers.items()}
    model = replace_modules_atomically(model, replacements)
    model.requires_grad_(False)
    for layer in replacements.values():
        for parameter in (layer.u, layer.s, layer.vh):
            parameter.requires_grad_(True)
        if spec.freeze == "factors_and_bias" and layer.bias is not None:
            layer.bias.requires_grad_(True)
    return RankDynaObjective(model, tuple(layers), spec, reservations)
