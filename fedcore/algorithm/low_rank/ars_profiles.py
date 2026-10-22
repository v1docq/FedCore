"""Bounded CPU ARS search and physical export; no training runner or optimizer.

Primary source: Gao et al., NAACL 2024, 2024.naacl-long.13, equations 7--9,
Algorithm 1 and Appendix B. This is a vanilla-SVD, observed-label implementation
of that mechanism, not a reproduction of the paper's language-model results.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .approximation import NumericalDomainViolation
from .ars_specs import ARS
from .learned_layers import ZeroLinear
from .projection_recovery import observed_loss
from .structured_layers import FactorizedLinear
from .structured_profiles import FactorPair
from .topology import inspect_topology, module_paths, replace_modules_atomically


ARS_SOURCE = {
    "paper": "https://aclanthology.org/2024.naacl-long.13/",
    "pdf": "https://aclanthology.org/2024.naacl-long.13.pdf",
    "source_version": "NAACL-2024-2024.naacl-long.13",
    "doi": "10.18653/v1/2024.naacl-long.13",
    "locations": ["equation_7", "equation_8", "equation_9", "algorithm_1", "appendix_B"],
    "implementation_version": "fedcore_bounded_ars_v1",
    "author_code_revision": None,
    "author_code_status": "no_author_ARS_repository_link_in_paper_or_anthology",
}


def _finite(tensor, name):
    if not isinstance(tensor, torch.Tensor) or not bool(torch.isfinite(tensor).all()):
        raise NumericalDomainViolation(f"ARS {name} must be finite")
    return tensor


def ars_temperature(spec, completed_updates):
    """Linear schedule on optimizer updates; a single update uses the start."""
    if type(completed_updates) is not int or not 0 <= completed_updates <= spec.updates:
        raise NumericalDomainViolation("ARS temperature needs an in-range update count")
    fraction = min(completed_updates, spec.updates - 1) / max(spec.updates - 1, 1)
    return spec.temperature_start * (1 - fraction) + spec.temperature_end * fraction


def gumbel_sigmoid_mask(logits, *, temperature, generator, noise="paper_single_gumbel",
                        bias=0.0, stochastic=True):
    """Hard 0/1 forward, sigmoid backward, using only the explicit CPU generator.

    The paper prints one Gumbel in equation 9. A difference of independent
    Gumbels is the separately named binary-Concrete variant, not a silent fix.
    Deterministic export uses no noise and retains ties at zero.
    """
    _finite(logits, "logits")
    if logits.device.type != "cpu" or logits.dtype not in (torch.float32, torch.float64):
        raise NumericalDomainViolation("ARS masks require CPU float32/float64 logits")
    if (type(temperature) not in (int, float) or not math.isfinite(temperature)
            or temperature <= 0 or type(bias) not in (int, float) or not math.isfinite(bias)
            or noise not in ("paper_single_gumbel", "binary_gumbel_difference")
            or type(stochastic) is not bool or not isinstance(generator, torch.Generator)
            or str(generator.device) != "cpu"):
        raise NumericalDomainViolation("ARS requires explicit finite mask controls and a CPU generator")
    perturbation = 0.0
    if stochastic:
        def draw():
            uniform = torch.rand(logits.shape, dtype=logits.dtype, device="cpu", generator=generator)
            epsilon = torch.finfo(logits.dtype).eps
            return -torch.log(-torch.log(uniform.clamp(epsilon, 1 - epsilon)))
        perturbation = draw()
        if noise == "binary_gumbel_difference":
            perturbation = perturbation - draw()
    soft = torch.sigmoid((logits + bias + perturbation) / temperature)
    hard = (soft >= 0.5).to(soft.dtype)
    # Parenthesize the exact zero-valued correction to keep the forward binary.
    return hard + (soft - soft.detach())


def _targets(model, paths, spec):
    if type(spec) is not ARS or not isinstance(model, nn.Module):
        raise NumericalDomainViolation("ARS requires a model and a validated ARS description")
    paths = tuple(paths)
    if not paths or any(type(path) is not str for path in paths) or len(set(paths)) != len(paths):
        raise NumericalDomainViolation("ARS needs distinct explicit Linear paths in sequence order")
    modules = module_paths(model)
    topology = inspect_topology(model)
    if any(path not in modules or type(modules[path]) is not nn.Linear for path in paths):
        raise NumericalDomainViolation("ARS targets must be exact standard nn.Linear modules")
    target_ids = {id(modules[path]) for path in paths}
    if any(any(id(modules[path]) in target_ids for path in aliases) for aliases in topology.module_aliases):
        raise NumericalDomainViolation("ARS first profile refuses selected module aliases")
    def selected_parameter(path):
        return path.rpartition(".")[0] in paths
    if (any(any(selected_parameter(path) for path in aliases) for aliases in topology.parameter_aliases)
            or any(any(selected_parameter(path) for path in alias.parameter_paths)
                   for alias in topology.storage_aliases)):
        raise NumericalDomainViolation("ARS first profile refuses tied or shared-storage selected parameters")
    for tensor in (*model.parameters(), *model.buffers()):
        if tensor.device.type != "cpu" or tensor.layout != torch.strided:
            raise NumericalDomainViolation("ARS first profile requires an owned strided CPU model")
        if tensor.is_floating_point() and tensor.dtype not in (torch.float32, torch.float64):
            raise NumericalDomainViolation("ARS first profile requires float32/float64 model tensors")
        _finite(tensor, "model state")
    targets = tuple(modules[path] for path in paths)
    if any(layer.weight.dtype not in (torch.float32, torch.float64) or not layer.weight.numel()
           or layer.weight.shape != (layer.out_features, layer.in_features) for layer in targets):
        raise NumericalDomainViolation("ARS requires nonempty, dimension-consistent Linear weights")
    total = topology.unique_parameter_numel
    immutable = total - sum(layer.weight.numel() for layer in targets)
    if spec.retained_ratio * total < immutable:
        raise NumericalDomainViolation("ARS parameter target is below the immutable parameters and biases")
    return paths, targets, total, immutable


def estimate_ars_bytes(model, paths, spec):
    """Named tensor lower bounds before allocating factors, GRU or SVD state.

    LAPACK workspaces, activations, allocator overhead and optimizer state are
    deliberately unestimated. The interpreter must budget those separately.
    """
    paths, targets, _, _ = _targets(model, paths, spec)
    ranks = [min(layer.weight.shape) for layer in targets]
    factor_bytes = sum((layer.in_features + layer.out_features) * rank * layer.weight.element_size()
                       for layer, rank in zip(targets, ranks))
    bias_bytes = sum(0 if layer.bias is None else layer.bias.numel() * layer.bias.element_size()
                     for layer in targets)
    spectrum_bytes = sum(rank * layer.weight.element_size() for layer, rank in zip(targets, ranks))
    hidden, inputs = spec.hidden_size, spec.input_size
    controller_elements = 2 * (3 * hidden * inputs + 3 * hidden * hidden + 6 * hidden)
    controller_elements += 4 * hidden + sum(rank * (2 * hidden + 1) for rank in ranks)
    controller_bytes = controller_elements * targets[0].weight.element_size()
    return {
        "factor_parameters_bytes": factor_bytes + bias_bytes,
        "singular_value_buffers_bytes": spectrum_bytes,
        "fixed_hypernetwork_input_bytes": len(paths) * inputs * targets[0].weight.element_size(),
        "hypernetwork_parameters_bytes": controller_bytes,
        "structural_gradients_bytes_lower_bound": controller_bytes if spec.structural_trainable else 0,
        "weight_gradients_bytes_lower_bound": factor_bytes + bias_bytes if spec.weight_trainable else 0,
        "mask_work_bytes_lower_bound": 4 * spectrum_bytes,
        "svd_input_outputs_bytes_lower_bound": 8 * sum(layer.weight.numel()
            + (layer.in_features + layer.out_features + 1) * rank for layer, rank in zip(targets, ranks)),
    }


class ARSHypernetwork(nn.Module):
    """One shared bidirectional GRU over the declared sequence of operations."""

    def __init__(self, ranks, spec, dtype, generator):
        super().__init__()
        self.gru = nn.GRU(spec.input_size, spec.hidden_size, bidirectional=True, dtype=dtype)
        self.norm = nn.LayerNorm(2 * spec.hidden_size, dtype=dtype)
        self.activation = nn.GELU()
        self.heads = nn.ModuleList([nn.Linear(2 * spec.hidden_size, rank, dtype=dtype) for rank in ranks])
        self.register_buffer("z", torch.randn((len(ranks), 1, spec.input_size), dtype=dtype,
                                              generator=generator))

    def forward(self):
        sequence, _ = self.gru(self.z)
        features = self.activation(self.norm(sequence[:, 0]))
        return tuple(head(features[index]) for index, head in enumerate(self.heads))


class ARSMaskedLinear(nn.Module):
    """Full-sized search factors. Binary masks are transient objective inputs."""

    @property
    def weight(self):
        from .learned_layers import UnsupportedLearnedAccessError
        raise UnsupportedLearnedAccessError('P4 requires module calls; functional dense weight access is unsupported')

    def __init__(self, layer):
        super().__init__()
        self.in_features, self.out_features = layer.in_features, layer.out_features
        try:
            u, s, vh = torch.linalg.svd(layer.weight.detach().double(), full_matrices=False)
        except torch.linalg.LinAlgError as error:
            raise NumericalDomainViolation("ARS initialization SVD failed to converge") from error
        self.left = nn.Parameter(_finite((u * s).to(layer.weight), "left factors").clone())
        self.right = nn.Parameter(_finite(vh.to(layer.weight), "right factors").clone())
        self.register_buffer("singular_values", _finite(s.to(layer.weight), "singular values").clone())
        self.bias = None if layer.bias is None else nn.Parameter(layer.bias.detach().clone())
        self._mask = None

    def forward(self, inputs):
        hidden = F.linear(inputs, self.right)
        if self._mask is not None:
            hidden = hidden * self._mask.to(hidden)
        return F.linear(hidden, self.left, self.bias)


def _export(layer, indices):
    if not indices:
        result = ZeroLinear(layer.in_features, layer.out_features, layer.bias is not None,
                            dtype=layer.left.dtype, device=layer.left.device)
        if layer.bias is not None:
            with torch.no_grad():
                result.bias.copy_(layer.bias)
        return result
    index = torch.tensor(indices, dtype=torch.int64, device=layer.left.device)
    pair = FactorPair(layer.left.detach().index_select(1, index).clone(),
                      layer.right.detach().index_select(0, index).clone())
    return FactorizedLinear.from_factors(pair, layer.bias)


class ARSObjective:
    """Owned search state; the shared executor owns all optimizer operations."""

    def __init__(self, model, paths, layers, hypernetwork, generator, spec, total, immutable, estimate):
        self.model, self.paths, self.layers = model, paths, layers
        self.hypernetwork, self.generator, self.spec = hypernetwork, generator, spec
        self.total, self.immutable = total, immutable
        self.completed_updates = self.calls = 0
        self.active = self.finalized = False
        self._pending = None
        self.evidence = {
            "objective": "ars_observed_loss_parameter_budget_singular_alignment",
            "source": deepcopy(ARS_SOURCE), "loss": spec.loss, "reduction": spec.reduction,
            "requested_updates": spec.updates, "completed_updates": 0, "seed": spec.seed,
            "noise": spec.noise, "temperature_schedule": "linear_by_completed_optimizer_update",
            "temperature_start": spec.temperature_start, "temperature_end": spec.temperature_end,
            "mask_bias": spec.mask_bias, "fixed_z": True,
            "hypernetwork": {"kind": "bidirectional_GRU_LayerNorm_GELU_linear_heads",
                             "input_size": spec.input_size, "hidden_size": spec.hidden_size,
                             "operation_sequence": list(paths)},
            "structural_trainable": spec.structural_trainable,
            "weight_trainable": spec.weight_trainable,
            "weight_training_scope": "selected_SVD_factors_and_biases" if spec.weight_trainable else "frozen",
            "alignment_spectrum": "initial_SVD_singular_values_even_if_weight_factors_train",
            "budget_scope": "whole_model_parameter_elements_including_immutable_parameters_and_biases",
            "original_parameter_elements": total, "immutable_parameter_elements": immutable,
            "retained_ratio": spec.retained_ratio, "target_parameter_elements": spec.retained_ratio * total,
            "budget_weight": spec.budget_weight, "alignment_weight": spec.alignment_weight,
            "budget_is_hard_constraint": False, "export_policy": spec.export_policy,
            "final_decision": "noise_free_logits_plus_bias_nonnegative",
            "search_factor_shape": "full_SVD_rank_until_physical_export",
            "training_tensor_bytes_lower_bounds": estimate, "history": [],
            "profile_limits": ["CPU_float32_float64", "exact_nonaliased_Linear_targets",
                               "vanilla_SVD", "observed_labels", "no_language_model_reproduction_claim"],
            "paper_deviations": ["whole_model_budget_counts_immutable_parameters_and_biases",
                                 "deterministic_noise_free_final_decision"],
        }
        if spec.export_policy == "selected_indices":
            self.evidence["paper_deviations"].append("selected_indices_export_instead_of_paper_prefix")
        if spec.weight_trainable:
            self.evidence["paper_deviations"].append("selected_factors_and_biases_trainable")
        if spec.noise != "paper_single_gumbel":
            self.evidence["paper_deviations"].append("binary_Gumbel_difference_instead_of_equation_9_single_Gumbel")
        if (spec.input_size, spec.hidden_size) != (32, 64):
            self.evidence["paper_deviations"].append("custom_hypernetwork_dimensions")
        if spec.temperature_start != 0.4 or spec.temperature_end != 0.4:
            self.evidence["paper_deviations"].append("custom_temperature_schedule")
        self.evidence["trainable_names"] = [name for name, p in model.named_parameters() if p.requires_grad]
        self.evidence["trainable_parameter_elements"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def __enter__(self):
        if self.active or self.finalized:
            raise NumericalDomainViolation("ARS objective cannot be nested or reopened after export")
        self.active = True
        return self

    def __exit__(self, *exc):
        for layer in self.layers:
            layer._mask = None
        self._pending = None
        self.active = False
        self.evidence["objective_calls"] = self.calls
        return False

    def _validate_model(self, model):
        if model is not self.model:
            raise NumericalDomainViolation("ARS objective belongs to its owned prepared model")
        modules = module_paths(model)
        if any(modules.get(path) is not layer for path, layer in zip(self.paths, self.layers)):
            raise NumericalDomainViolation("ARS search topology changed before finalization")
        if self.layers[0]._modules.get("_ars_hypernetwork") is not self.hypernetwork:
            raise NumericalDomainViolation("ARS registered hypernetwork changed before finalization")
        if {name for name, p in model.named_parameters() if p.requires_grad} != set(self.evidence["trainable_names"]):
            raise NumericalDomainViolation("ARS declared parameter trainability changed during search")

    def masks(self, *, stochastic):
        temperature = ars_temperature(self.spec, self.completed_updates)
        return tuple(gumbel_sigmoid_mask(logits, temperature=temperature, generator=self.generator,
                                        noise=self.spec.noise, bias=self.spec.mask_bias,
                                        stochastic=stochastic) for logits in self.hypernetwork())

    def __call__(self, model, inputs, labels):
        self._validate_model(model)
        if not self.active or self.finalized or self.completed_updates >= self.spec.updates or self._pending is not None:
            raise NumericalDomainViolation("ARS objective requires its active update context and one call per step")
        masks = self.masks(stochastic=True)
        reference = masks[0]
        cost = reference.new_tensor(float(self.immutable))
        alignment = reference.new_zeros(())
        for layer, mask in zip(self.layers, masks):
            cost = cost + (layer.in_features + layer.out_features) * mask.sum()
            count = int(mask.detach().sum())
            prefix = (torch.arange(mask.numel(), device=mask.device) < count).to(mask)
            alignment = alignment + ((mask - prefix) * layer.singular_values.to(mask)).square().sum()
        alignment = alignment / len(masks)
        target = self.spec.retained_ratio * self.total
        budget = torch.log(torch.maximum(cost, cost.new_tensor(target)) / target)
        try:
            for layer, mask in zip(self.layers, masks):
                layer._mask = mask
            task = observed_loss(model(inputs), labels, self.spec.loss, self.spec.reduction)
        finally:
            for layer in self.layers:
                layer._mask = None
        loss = _finite(task + self.spec.budget_weight * budget + self.spec.alignment_weight * alignment, "scalar objective")
        if loss.ndim != 0 or not loss.requires_grad:
            raise NumericalDomainViolation("ARS objective must be a differentiable finite scalar")
        self.calls += 1
        self._pending = {"update": self.completed_updates + 1,
            "temperature": ars_temperature(self.spec, self.completed_updates),
            "task_loss": float(task.detach()), "budget_loss": float(budget.detach()),
            "alignment_loss": float(alignment.detach()), "total_loss": float(loss.detach()),
            "search_parameter_elements": float(cost.detach()),
            "selected_ranks": [int(mask.detach().sum()) for mask in masks]}
        return loss

    def before_step(self, model, optimizer):
        self._validate_model(model)
        if not self.active or self._pending is None or not isinstance(optimizer, torch.optim.Optimizer):
            raise NumericalDomainViolation("ARS before_step requires a pending loss and live optimizer")
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise NumericalDomainViolation("ARS optimizer has stale, duplicate or unauthorized Parameters")
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients:
            raise NumericalDomainViolation("ARS optimizer step must follow backward")
        for gradient in gradients:
            _finite(gradient, "gradients")

    def after_step(self, model, optimizer):
        self._validate_model(model)
        if not self.active or self._pending is None:
            raise NumericalDomainViolation("ARS after_step requires exactly one pending update")
        for parameter in model.parameters():
            _finite(parameter, "updated parameters")
        self.completed_updates += 1
        self.evidence["completed_updates"] = self.completed_updates
        self.evidence["history"].append(self._pending)
        self._pending = None

    def finalize(self, model, calibration=None, batch_size=16):
        """Select deterministically and compact the exact selected component axes.

        No calibration data is consumed or optimized here. Prefix export is an
        explicit alternative and reports its actual factor-operator difference.
        """
        self._validate_model(model)
        if self.active or self.finalized or self.completed_updates != self.spec.updates:
            raise NumericalDomainViolation("ARS export requires all declared updates and a closed objective context")
        if type(batch_size) is not int or batch_size < 1:
            raise NumericalDomainViolation("ARS finalize batch_size must be positive")
        replacements, decisions = {}, []
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            masks = self.masks(stochastic=False)
            for path, layer, mask in zip(self.paths, self.layers, masks):
                for parameter in (layer.left, layer.right, layer.bias):
                    if parameter is not None:
                        _finite(parameter, "export factors and bias")
                selected = tuple(torch.nonzero(mask, as_tuple=False).flatten().tolist())
                prefix = tuple(range(len(selected)))
                exported = selected if self.spec.export_policy == "selected_indices" else prefix
                selected_operator = layer.left[:, selected] @ layer.right[list(selected), :]
                prefix_operator = layer.left[:, prefix] @ layer.right[list(prefix), :]
                difference = _finite((selected_operator.double() - prefix_operator.double()).square().sum(),
                                     "mask-prefix operator discrepancy")
                replacement = _export(layer, exported).eval().requires_grad_(False)
                replacements[path] = replacement
                decisions.append({"path": path, "selected_indices": list(selected),
                    "exported_indices": list(exported), "selected_rank": len(selected),
                    "physical_rank": len(exported), "representation": type(replacement).__name__,
                    "selected_to_prefix_operator_squared_frobenius": float(difference),
                    "export_changes_selected_operator": exported != selected and float(difference) != 0.0,
                    "physical_parameter_elements": sum(p.numel() for p in replacement.parameters())})
        self.model = replace_modules_atomically(model, replacements)
        self.model.eval().requires_grad_(False)
        self.finalized = True
        actual = sum(p.numel() for p in self.model.parameters())
        self.evidence.update({"final_decisions": decisions, "final_parameter_elements": actual,
            "final_retained_ratio": actual / self.total,
            "final_budget_satisfied": actual <= self.spec.retained_ratio * self.total,
            "hypernetwork_removed": not any(isinstance(module, ARSHypernetwork) for module in self.model.modules()),
            "final_calibration_consumed": False, "optimizer_state_exported": False})
        return deepcopy(self.evidence)


def prepare_ars(model, paths, spec):
    """Mutate only an owned private model; return its objective and live model.

    All validation and factor/controller construction precede atomic replacement.
    The executor must rebind ``model = objective.model`` both after preparation
    and finalization, since the selected Linear can itself be the root model.
    One controller belongs to the first masked layer, keeping Sequential's
    forward graph unchanged and registering its parameters exactly once.
    """
    paths, targets, total, immutable = _targets(model, paths, spec)
    estimate = estimate_ars_bytes(model, paths, spec)
    generator = torch.Generator(device="cpu").manual_seed(spec.seed)
    with torch.random.fork_rng(devices=[]):
        # Seed only the CPU default generator used by module constructors;
        # torch.manual_seed would also alter ambient CUDA generators.
        torch.random.default_generator.manual_seed(spec.seed)
        layers = tuple(ARSMaskedLinear(layer) for layer in targets)
        hypernetwork = ARSHypernetwork([layer.singular_values.numel() for layer in layers], spec,
                                      layers[0].left.dtype, generator)
    hypernetwork.requires_grad_(spec.structural_trainable)
    for layer in layers:
        layer.requires_grad_(spec.weight_trainable)
    layers[0].add_module("_ars_hypernetwork", hypernetwork)
    prepared = replace_modules_atomically(model, dict(zip(paths, layers)))
    # Freeze the rest only after all preparation succeeds. Reapply the two
    # independent boundaries because the controller is a child of one target.
    prepared.requires_grad_(False)
    for layer in layers:
        for name in ("left", "right", "bias"):
            parameter = getattr(layer, name)
            if parameter is not None:
                parameter.requires_grad_(spec.weight_trainable)
    hypernetwork.requires_grad_(spec.structural_trainable)
    return ARSObjective(prepared, paths, layers, hypernetwork, generator, spec, total, immutable, estimate)
