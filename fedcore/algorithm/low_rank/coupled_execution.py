"""Bounded architecture adapter and preparation used by the method interpreter.

All capability and write-set checks precede collection. This module prepares
replacements only; the existing topology boundary applies them atomically.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from .coupled_layers import ReferenceAttention, ReferenceMLP, _coordinates
from .coupled_specs import (COUPLED_SPEC_TYPES, FlatValueOutput, MoDeValueOutput,
                            FlatMLPChannels, MoDeMLPChannels, FlatQueryKey, MoDeQueryKey)
from .coupled_profiles import (flat_value_basis, mode_product_factors, mlp_channel_fit,
                               select_query_key_coordinates)
from .plans import (TransformPlan, TransformRequest, StepSpec, OperatorDescriptor,
                    Goal, FixedRank, ResourceLimits)
from .topology import module_paths, validate_parameter_topology, TopologyError


class CoupledProfileError(ValueError):
    """The explicit reference architecture cannot execute this request."""


_VO = (FlatValueOutput, MoDeValueOutput)
_MLP = (FlatMLPChannels, MoDeMLPChannels)
_QK = (FlatQueryKey, MoDeQueryKey)


class ArchitectureAdapter:
    """Exact reference types only; no architecture inferred from attribute names."""
    runtime = "reference_cpu_interleaved_rope_v1"

    def validate(self, layer, spec, rank):
        expected = ReferenceMLP if type(spec) in _MLP else ReferenceAttention
        if type(layer) is not expected:
            raise CoupledProfileError("Coupled profile supports the exact declared reference module type only")
        if (type(layer.d_model) is not int or layer.d_model < 1 or layer._parameters):
            raise CoupledProfileError("Invalid reference dimension or undeclared direct Parameter")
        tensors = tuple(layer.parameters()) + tuple(layer.buffers())
        if any(v.device.type != "cpu" for v in tensors):
            raise CoupledProfileError("Reference coupled runtime requires CPU tensors")
        if any(v.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(v).all()) for v in layer.parameters()):
            raise CoupledProfileError("Reference coupled parameters require finite float32/float64 values")
        if len({v.dtype for v in layer.parameters()}) != 1:
            raise CoupledProfileError("Reference coupled parameters must use a single floating dtype")
        if type(rank) is not int or rank < 1:
            raise CoupledProfileError("Coupled profiles require an explicit positive integer rank")
        if expected is ReferenceAttention:
            dimensions = (layer.num_heads, layer.num_kv_heads, layer.head_dim, layer.qk_dim, layer.value_dim)
            if (any(type(v) is not int or v < 1 for v in dimensions)
                    or layer.qk_dim > layer.head_dim or layer.value_dim > layer.head_dim
                    or type(layer.rope) is not bool or type(layer.causal) is not bool
                    or (layer.rope and layer.head_dim % 2)
                    or type(layer.rope_base) not in (float, int) or not math.isfinite(layer.rope_base) or layer.rope_base <= 1
                    or type(layer.logits_scale) not in (float, int) or not math.isfinite(layer.logits_scale) or layer.logits_scale <= 0
                    or set(layer._modules) != {"q_proj", "k_proj", "v_proj", "o_proj"}
                    or set(layer._buffers) != {"head_map", "group_coordinates"}):
                raise CoupledProfileError("Unsupported reference attention constructor or runtime configuration")
            config = layer.representation_config()
            # Constructor-level validation is pure: do not allocate a temporary model.
            if (layer.head_map.dtype != torch.long or layer.head_map.shape != (layer.num_heads,)
                    or layer.group_coordinates.dtype != torch.long
                    or layer.group_coordinates.shape != (layer.num_kv_heads, layer.qk_dim)):
                raise CoupledProfileError("Invalid stored head map or coordinate metadata")
            mapping = config["head_map"]
            if (layer.num_heads % layer.num_kv_heads or any(type(v) is not int or not 0 <= v < layer.num_kv_heads for v in mapping)
                    or any(mapping.count(g) != layer.num_heads // layer.num_kv_heads for g in range(layer.num_kv_heads))):
                raise CoupledProfileError("Incompatible shared KV head map")
            for values in config["group_coordinates"]:
                _coordinates(values, layer.head_dim, layer.qk_dim, rope=layer.rope)
            width = layer.value_dim if type(spec) in _VO else layer.qk_dim
            if rank > width:
                raise CoupledProfileError("Requested attention rank exceeds its current physical width")
            shapes = ((layer.q_proj, layer.num_heads * layer.qk_dim, layer.d_model),
                      (layer.k_proj, layer.num_kv_heads * layer.qk_dim, layer.d_model),
                      (layer.v_proj, layer.num_kv_heads * layer.value_dim, layer.d_model),
                      (layer.o_proj, layer.d_model, layer.num_heads * layer.value_dim))
            if type(spec) in _QK:
                if layer.rope and rank % 2:
                    raise CoupledProfileError("Q/K rank must preserve complete RoPE pairs")
                if spec.coordinates:
                    _coordinates(spec.coordinates, layer.qk_dim, rank, rope=layer.rope)
        else:
            width = layer.intermediate_size
            if (type(width) is not int or width < 1 or type(layer.gated) is not bool
                    or layer._buffers or set(layer._modules) != ({"up_proj", "gate_proj", "down_proj"} if layer.gated else {"up_proj", "down_proj"})):
                raise CoupledProfileError("Unsupported reference MLP constructor configuration")
            if rank > width:
                raise CoupledProfileError("Requested MLP width exceeds its current physical width")
            shapes = [(layer.up_proj, width, layer.d_model), (layer.down_proj, layer.d_model, width)]
            if layer.gated:
                shapes.append((layer.gate_proj, width, layer.d_model))
            elif layer.gate_proj is not None:
                raise CoupledProfileError("Ordinary reference MLP must not contain a Gate projection")
            if (layer.activation not in ("silu", "gelu", "relu") or len(layer.channel_indices) != width
                    or len(set(layer.channel_indices)) != width):
                raise CoupledProfileError("Unsupported activation or invalid original channel metadata")
            if spec.indices and (len(spec.indices) != rank or max(spec.indices) >= width):
                raise CoupledProfileError("Gate/Up selection must use the same valid current channel indices")
        for projection, outputs, inputs in shapes:
            if (type(projection) is not nn.Linear or projection.weight.shape != (outputs, inputs)
                    or projection.in_features != inputs or projection.out_features != outputs
                    or (projection.bias is not None and projection.bias.shape != (outputs,))):
                raise CoupledProfileError("Projection shapes disagree with the declared coupled graph")
        if len({projection.bias is not None for projection, _, _ in shapes}) != 1:
            raise CoupledProfileError("The portable reference profile requires consistent bias presence")
        return width


def plan_coupled_profile(model, spec, *, rank=None, rank_ratio=None, parameter_fraction=None,
                         ranks=None, target_paths=None, max_peak_bytes=12 * 1024**3):
    """Read capabilities only; no copy, hooks, forward pass or RNG consumption."""
    if type(spec) not in COUPLED_SPEC_TYPES or not isinstance(model, nn.Module):
        raise CoupledProfileError("A validated coupled specification and torch Module are required")
    if any(v is not None for v in (rank_ratio, parameter_fraction, ranks)):
        raise CoupledProfileError("The bounded coupled profile requires one explicit integer rank")
    if (not isinstance(target_paths, (tuple, list)) or not target_paths
            or any(type(path) is not str for path in target_paths) or len(set(target_paths)) != len(target_paths)):
        raise CoupledProfileError("Select distinct explicit whole-module coupled target paths")
    if type(max_peak_bytes) is not int or not 0 < max_peak_bytes <= 12 * 1024**3:
        raise CoupledProfileError("Declare a CPU peak limit of at most 12 GiB")
    paths = module_paths(model)
    if any(path not in paths for path in target_paths):
        raise CoupledProfileError("Unknown coupled target path")
    if any(a != b and (a == "" or b.startswith(a + ".")) for a in target_paths for b in target_paths):
        raise CoupledProfileError("Coupled replacement paths may not contain one another")
    topology = validate_parameter_topology(model, target_paths)
    if topology.storage_aliases or topology.parameter_aliases:
        raise TopologyError("Reference coupled profiles reject tied Parameters and shared storage views")
    adapter = ArchitectureAdapter(); steps = []
    for path in target_paths:
        layer = paths[path]; width = adapter.validate(layer, spec, rank)
        storage_bytes = sum(v.numel() * v.element_size() for v in (*layer.parameters(), *layer.buffers()))
        if 3 * storage_bytes + 8 * (12 * (width * width + layer.d_model ** 2) + 8 * layer.d_model * width) > max_peak_bytes:
            raise CoupledProfileError("ResourceLimitExceeded: coupled graph preparation lower bound")
        steps.append(StepSpec(OperatorDescriptor(path, (layer.d_model, width),
            "reference_mlp" if type(layer) is ReferenceMLP else "reference_attention",
            str(next(layer.parameters()).dtype).removeprefix("torch."),
            bias_elements=sum(p.numel() for name, p in layer.named_parameters() if name.endswith("bias"))),
            rank, FixedRank(rank), True))
    return TransformPlan(TransformRequest(Goal.COUPLED_GRAPH_TRANSFORM, spec.variant,
        "bounded_reference_coupled", FixedRank(rank), representation="reference_compact"),
        tuple(steps), ResourceLimits(max_peak_bytes))


def _kwargs_for_batch(kwargs, start, end, total):
    result = {}
    if not isinstance(kwargs, dict) or set(kwargs) - {"padding_mask", "attention_mask", "positions"}:
        raise CoupledProfileError("Calibration kwargs support only declared boolean masks and position IDs")
    for name, value in kwargs.items():
        if value is not None and (not isinstance(value, torch.Tensor) or value.device.type != "cpu"):
            raise CoupledProfileError("Calibration mask/position arguments must be CPU tensors")
        if name in ("padding_mask", "positions") and value is not None:
            if value.ndim != 2 or len(value) != total:
                raise CoupledProfileError("Calibration masks/positions require the full calibration batch axis")
            result[name] = value[start:end]
        else:
            result[name] = value
    return result


def _capture(model, calibration, targets, *, batch_size, workspace, kwargs):
    if (not isinstance(calibration, torch.Tensor) or calibration.device.type != "cpu"
            or calibration.dtype not in (torch.float32, torch.float64) or calibration.ndim < 2
            or not calibration.numel() or not bool(torch.isfinite(calibration).all())):
        raise CoupledProfileError("Separate finite nonempty CPU float calibration observations are required")
    if type(batch_size) is not int or batch_size < 1 or type(workspace) is not int or workspace < 1:
        raise CoupledProfileError("Positive batch and workspace bounds are required")
    captured = {path: [] for path in targets}; occupied = calibration.numel() * calibration.element_size()
    if occupied > workspace:
        raise CoupledProfileError("ResourceLimitExceeded: coupled calibration exceeds workspace")
    handles = []; modes = [(module, module.training) for module in model.modules()]
    def hook_for(path):
        def hook(layer, args, call_kwargs):
            nonlocal occupied
            if len(args) != 1 or not isinstance(args[0], torch.Tensor):
                raise CoupledProfileError("Reference calibration requires one positional tensor input")
            if set(call_kwargs) - {"padding_mask", "attention_mask", "positions"}:
                raise CoupledProfileError("Caching and undeclared runtime arguments are not calibration observations")
            x = args[0]
            if x.dtype != next(layer.parameters()).dtype or x.device.type != "cpu" or x.shape[-1] != layer.d_model:
                raise CoupledProfileError("Observed input shape/dtype must match the reference target")
            if type(layer) is ReferenceAttention:
                if x.ndim != 3:
                    raise CoupledProfileError("Reference attention requires a batch/token/feature calibration layout")
                b, t, _ = x.shape
                # Dense logits, masked logits, softmax and masks coexist. Reserve
                # a conservative constant factor before calling target.forward.
                runtime_bytes = (16 * b * layer.num_heads * t * t * x.element_size()
                    + 8 * b * t * t + 16 * b * t * x.element_size()
                    * (layer.d_model + layer.num_heads * layer.qk_dim + layer.num_kv_heads * layer.value_dim))
            else:
                runtime_bytes = 16 * x.numel() * x.element_size() + 16 * (x.numel() // layer.d_model) * layer.intermediate_size * x.element_size()
            extra = sum(v.numel() * v.element_size() for v in call_kwargs.values() if isinstance(v, torch.Tensor))
            rows = x.numel() // layer.d_model
            derived_width = (layer.intermediate_size + layer.d_model if type(layer) is ReferenceMLP else
                layer.d_model + layer.num_heads * (layer.qk_dim + layer.value_dim))
            # Solvers concatenate observations and use float64 work arrays.
            # Reserve all-sample feature/QR storage, not just each small batch.
            occupied += x.numel() * x.element_size() + extra + 16 * rows * derived_width * 8
            if occupied + runtime_bytes > workspace:
                raise CoupledProfileError("ResourceLimitExceeded: captured observations and reference forward workspace")
            if not bool(torch.isfinite(x).all()):
                raise CoupledProfileError("Nonfinite current graph target observations")
            captured[path].append((x.detach().clone(), {key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                                                        for key, value in call_kwargs.items()}))
        return hook
    try:
        model.eval()
        for path, layer in targets.items():
            handles.append(layer.register_forward_pre_hook(hook_for(path), with_kwargs=True))
        with torch.no_grad():
            for start in range(0, len(calibration), batch_size):
                end = min(start + batch_size, len(calibration))
                model(calibration[start:end], **_kwargs_for_batch(kwargs, start, end, len(calibration)))
    finally:
        for handle in handles:
            handle.remove()
        for module, training in modes:
            module.training = training
    if any(not values for values in captured.values()):
        raise CoupledProfileError("Every selected reference module must be executed by calibration")
    return captured


def _new_attention(layer, *, value_dim=None, qk_dim=None, coordinates=None):
    config = layer.representation_config(); config.pop("layer_type"); config.pop("version")
    if value_dim is not None:
        config["value_dim"] = value_dim
    if qk_dim is not None:
        config["qk_dim"] = qk_dim; config["group_coordinates"] = coordinates
    return ReferenceAttention(**config, dtype=layer.q_proj.weight.dtype)


def _copy_projection(target, source):
    target.load_state_dict(source.state_dict())
    for name, param in target.named_parameters():
        param.requires_grad_(dict(source.named_parameters())[name].requires_grad)


def _masked_rows(value, kwargs):
    # Query padding is excluded from local statistics; key masks still affect attention.
    mask = kwargs.get("padding_mask")
    if mask is None:
        return value.reshape(-1, value.shape[-1])
    return value[mask]


def _prepare_vo(layer, observations, spec, rank):
    result = _new_attention(layer, value_dim=rank)
    _copy_projection(result.q_proj, layer.q_proj); _copy_projection(result.k_proj, layer.k_proj)
    if layer.o_proj.bias is not None:
        result.o_proj.bias.copy_(layer.o_proj.bias)
    group_records = []; mapping = layer.head_map.tolist(); width = layer.value_dim
    for group in range(layer.num_kv_heads):
        heads = [h for h, g in enumerate(mapping) if g == group]
        vweight = layer.v_proj.weight[group * width:(group + 1) * width]
        vbias = None if layer.v_proj.bias is None else layer.v_proj.bias[group * width:(group + 1) * width]
        outputs = [layer.o_proj.weight[:, head * width:(head + 1) * width] for head in heads]
        if type(spec) is FlatValueOutput:
            rows = []
            for x, kwargs in observations:
                if spec.observation_point == "post_softmax":
                    response = layer.attention(x, **kwargs)
                    rows.extend(_masked_rows(response.contexts[:, head], kwargs) for head in heads)
                else:
                    projected = torch.nn.functional.linear(x, vweight, vbias)
                    rows.append(_masked_rows(projected, kwargs))
            basis, details = flat_value_basis(torch.cat(rows), rank)
            new_v = basis.T @ vweight; new_bias = None if vbias is None else basis.T @ vbias
            new_outputs = [weight @ basis for weight in outputs]
            details.update({"basis": basis.tolist(), "observation_point": spec.observation_point,
                            "source_version": "2505.23966v1", "variant_status": "explicit_observation_convention"})
        else:
            rows = torch.cat([_masked_rows(x, kwargs) for x, kwargs in observations])
            right = vweight
            if vbias is not None:
                rows = torch.cat((rows, rows.new_ones((len(rows), 1))), 1)
                right = torch.cat((vweight, vbias[:, None]), 1)
            pair = mode_product_factors(torch.cat(outputs, 0), right, rows, rank)
            new_v = pair.right[:, :layer.d_model]
            new_bias = None if vbias is None else pair.right[:, -1]
            new_outputs = list(pair.left.split(layer.d_model, dim=0)); details = dict(pair.diagnostics)
            details.update({"source_version": "2408.09632v3",
                "variant_status": "reconstructed_joint_gqa_product" if len(heads) > 1 else "type_iii_thin_product",
                "observation_point": "input_before_attention", "bias_policy": "augmented_input"})
        result.v_proj.weight[group * rank:(group + 1) * rank].copy_(new_v)
        if new_bias is not None:
            result.v_proj.bias[group * rank:(group + 1) * rank].copy_(new_bias)
        for head, weight in zip(heads, new_outputs):
            result.o_proj.weight[:, head * rank:(head + 1) * rank].copy_(weight)
        details.update({"kv_group": group, "query_heads": heads, "rank": rank, "padding": 0})
        group_records.append(details)
    return result, {"groups": group_records, "group_ranks": [rank] * layer.num_kv_heads,
                    "cache_layout": "batch_kvgroup_tokens_compact_width", "value_cache_width": rank}


def _prepare_mlp(layer, observations, spec, rank):
    features = torch.cat([layer.features(x).reshape(-1, layer.intermediate_size) for x, _ in observations])
    family = "flat" if type(spec) is FlatMLPChannels else "modegpt"
    fitted = mlp_channel_fit(features, layer.down_proj.weight, rank, indices=spec.indices,
        family=family, leverage_ridge=getattr(spec, "leverage_ridge", 1.0), ridge=spec.ridge, rcond=spec.rcond)
    indices = fitted.indices
    result = ReferenceMLP(layer.d_model, rank, gated=layer.gated, activation=layer.activation,
        bias=layer.up_proj.bias is not None, channel_indices=[layer.channel_indices[i] for i in indices],
        dtype=layer.up_proj.weight.dtype)
    for source, target in ((layer.up_proj, result.up_proj), (layer.gate_proj, result.gate_proj)):
        if source is not None:
            target.weight.copy_(source.weight[list(indices)])
            if source.bias is not None:
                target.bias.copy_(source.bias[list(indices)])
    result.down_proj.weight.copy_(fitted.down_weight)
    if layer.down_proj.bias is not None:
        result.down_proj.bias.copy_(layer.down_proj.bias)
    details = dict(fitted.diagnostics)
    details.update({"source_version": "2505.23966v1" if family == "flat" else "2408.09632v3",
        "variant_status": "corrected_pseudoinverse_ls" if family == "flat" else "type_i_fixed_design_ls",
        "gate_indices": list(indices) if layer.gated else None, "up_indices": list(indices),
        "nonlinearity": layer.activation, "nonlinearity_policy": "original_elementwise_on_selected_channels"})
    return result, details


def _prepare_qk(layer, observations, spec, rank):
    family = "flat" if type(spec) is FlatQueryKey else "modegpt"
    mapping = layer.head_map.tolist(); groups = []; records = []; width = layer.qk_dim
    projected = [(layer.project_query_key(x, positions=kwargs.get("positions")), kwargs) for x, kwargs in observations]
    for group in range(layer.num_kv_heads):
        heads = [head for head, g in enumerate(mapping) if g == group]
        queries = []; keys = []
        for (q, k), kwargs in projected:
            # Pool per-group correlations with equal multiplicity of shared keys.
            queries.extend(_masked_rows(q[:, head], kwargs) for head in heads)
            keys.extend(_masked_rows(k[:, group], kwargs) for _ in heads)
        chosen, details = select_query_key_coordinates(torch.cat(queries), torch.cat(keys), rank,
            family=family, rope=layer.rope, coordinates=spec.coordinates)
        groups.append(chosen)
        details.update({"kv_group": group, "query_heads": heads,
            "original_coordinates": layer.group_coordinates[group, list(chosen)].tolist(),
            "source_version": "2505.23966v1" if family == "flat" else "2408.09632v3",
            "variant_status": "reconstructed_qk_pair_energy" if family == "flat" else "reconstructed_type_ii_rope_gqa"})
        records.append(details)
    absolute = [layer.group_coordinates[group, list(chosen)].tolist() for group, chosen in enumerate(groups)]
    result = _new_attention(layer, qk_dim=rank, coordinates=absolute)
    _copy_projection(result.v_proj, layer.v_proj); _copy_projection(result.o_proj, layer.o_proj)
    for head, group in enumerate(mapping):
        selected = [head * width + i for i in groups[group]]
        result.q_proj.weight[head * rank:(head + 1) * rank].copy_(layer.q_proj.weight[selected])
        if layer.q_proj.bias is not None:
            result.q_proj.bias[head * rank:(head + 1) * rank].copy_(layer.q_proj.bias[selected])
    for group, chosen in enumerate(groups):
        selected = [group * width + i for i in chosen]
        result.k_proj.weight[group * rank:(group + 1) * rank].copy_(layer.k_proj.weight[selected])
        if layer.k_proj.bias is not None:
            result.k_proj.bias[group * rank:(group + 1) * rank].copy_(layer.k_proj.bias[selected])
    logits_error = 0.0; output_error = 0.0
    for x, kwargs in observations:
        before = layer.attention(x, **kwargs); after = result.attention(x, **kwargs)
        logits_error += float((before.logits - after.logits).double().square().sum())
        output_error += float((before.output - after.output).double().square().sum())
    return result, {"groups": records, "qk_cache_width": rank, "logits_scale": layer.logits_scale,
        "original_head_dim": layer.head_dim, "rope_base": layer.rope_base,
        "group_coordinates": absolute, "local_logits_error_squared": logits_error,
        "local_output_error_squared": output_error, "error_semantics": "local_calibration_only"}


def prepare_coupled_replacements(model, calibration, spec, plan, *, batch_size=16,
                                  max_workspace_bytes=256 * 1024**2, calibration_kwargs=None):
    """Prepare a complete replacement mapping without modifying the input graph.

The caller owns model copying, job lifecycle, atomic apply, budgets and export.
The local fork_rng also makes this preparation safe to call directly.
"""
    if not isinstance(plan, TransformPlan) or plan.request.goal is not Goal.COUPLED_GRAPH_TRANSFORM:
        raise CoupledProfileError("A validated coupled TransformPlan is required")
    # Revalidate stale/tampered plans and all target capabilities before hooks.
    checked = plan_coupled_profile(model, spec, rank=plan.request.structure.rank,
        target_paths=[step.operator.path for step in plan.steps], max_peak_bytes=plan.resource_limits.max_peak_bytes)
    if checked != plan:
        raise CoupledProfileError("Coupled plan no longer matches the current graph")
    paths = module_paths(model); targets = {step.operator.path: paths[step.operator.path] for step in plan.steps}
    # Conservative scratch bound includes correlations and captured output axes.
    scratch = max(8 * (16 * (step.operator.shape[1] ** 2 + step.operator.shape[0] ** 2)
                       + 8 * step.operator.shape[0] * step.operator.shape[1]) for step in plan.steps)
    if scratch > max_workspace_bytes:
        raise CoupledProfileError("ResourceLimitExceeded: coupled solve scratch exceeds workspace")
    replacements = {}; records = []
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        captured = _capture(model, calibration, targets, batch_size=batch_size,
                            workspace=max_workspace_bytes - scratch, kwargs=calibration_kwargs or {})
        identities = {}
        for step in plan.steps:
            path = step.operator.path; layer = targets[path]; rank = step.rank
            if id(layer) in identities:
                result, details = identities[id(layer)]
            else:
                if type(spec) in _VO:
                    result, details = _prepare_vo(layer, captured[path], spec, rank)
                elif type(spec) in _MLP:
                    result, details = _prepare_mlp(layer, captured[path], spec, rank)
                else:
                    result, details = _prepare_qk(layer, captured[path], spec, rank)
                if type(spec) not in _QK:
                    local_error = 0.0
                    for x, kwargs in captured[path]:
                        residual = layer(x, **kwargs).double() - result(x, **kwargs).double()
                        squared = residual.square().sum()
                        if not bool(torch.isfinite(squared)):
                            raise CoupledProfileError("Nonfinite local coupled output error")
                        local_error += float(squared)
                    if not math.isfinite(local_error):
                        raise CoupledProfileError("Nonfinite accumulated local coupled output error")
                    details["local_output_error_squared"] = local_error
                    details["error_semantics"] = "local_calibration_current_module_only"
                result.train(layer.training)
                source_modules = dict(layer.named_modules())
                for name, module in result.named_modules():
                    module.training = source_modules[name].training
                # Preserve declared trainability for every corresponding projection.
                source_parameters = dict(layer.named_parameters())
                for name, parameter in result.named_parameters():
                    parameter.requires_grad_(source_parameters[name].requires_grad)
                if any(not bool(torch.isfinite(p).all()) for p in result.parameters()):
                    raise CoupledProfileError("Coupled replacement contains nonfinite installed parameters")
                identities[id(layer)] = result, details
            replacements[path] = result
            records.append({"path": path, "rank": rank, "runtime": ArchitectureAdapter.runtime,
                "variant": spec.variant, "diagnostics": details,
                "representation": result.representation_config(),
                "parameter_elements": sum(p.numel() for p in result.parameters()),
                "parameter_bytes": sum(p.numel() * p.element_size() for p in result.parameters()),
                "buffers_bytes": sum(b.numel() * b.element_size() for b in result.buffers())})
    return replacements, records
