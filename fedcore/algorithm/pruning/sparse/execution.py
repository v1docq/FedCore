"""Immutable selection plan and atomic sparse-profile interpreter."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
import json
import torch
from torch import nn

from fedcore.algorithm.low_rank.topology import (module_paths,
    validate_parameter_topology, replace_modules_atomically)
from .specs import HASSLEFreeSpec, LoSparseSpec, NMStructure, SparseDomainError, plan_linear, reject_method

METHODS = ("hassle_free", "losparse", "sola", "flexigpt", "prunenet", "moe_i2", "slice_mlp", "partition_mlp")


@dataclass(frozen=True)
class ChannelsSpec:
    channels: tuple[int, ...]

    def __post_init__(self):
        if (not isinstance(self.channels, tuple) or any(type(i) is not int or i < 0 for i in self.channels)
                or len(set(self.channels)) != len(self.channels)):
            raise SparseDomainError('invalid_specification', 'Distinct immutable nonnegative channel coordinates required')


def parse_sparse_spec(method, payload):
    """Strict data-only configuration parsing, shared by runner and service."""
    reject_method(method)
    if not isinstance(payload, Mapping):
        raise SparseDomainError("invalid_specification", "Sparse options must be an object")
    factories = {"hassle_free": HASSLEFreeSpec, "losparse": LoSparseSpec}
    if method in ('slice_mlp', 'partition_mlp'):
        factories[method] = ChannelsSpec
    if method == "sola":
        from .channels import SoLASpec
        factories[method] = SoLASpec
    elif method == "flexigpt":
        from . import architectures
        factories[method] = architectures.FlexiGPTSpec
    elif method == "prunenet":
        from .policies import PruneNetSpec
        factories[method] = PruneNetSpec
    elif method == "moe_i2":
        from .moe import MoECompressionSpec
        factories[method] = MoECompressionSpec
    if method not in factories:
        raise SparseDomainError("unsupported_method", "No declared immutable specification for this profile")
    cls = factories[method]
    values = dict(payload)
    if set(values) - set(cls.__dataclass_fields__):
        raise SparseDomainError("invalid_specification", "Unknown sparse method options")
    if method == "hassle_free" and isinstance(values.get("structure"), Mapping):
        structure = dict(values["structure"])
        if set(structure) - set(NMStructure.__dataclass_fields__):
            raise SparseDomainError("invalid_specification", "Unknown N:M options")
        values["structure"] = NMStructure(**structure)
    for name, field in cls.__dataclass_fields__.items():
        if isinstance(values.get(name), list):
            values[name] = tuple(tuple(x) if isinstance(x, list) else x for x in values[name])
    try:
        return cls(**values)
    except (TypeError, ValueError) as error:
        if isinstance(error, SparseDomainError):
            raise
        raise SparseDomainError("invalid_specification", str(error)) from error


@dataclass(frozen=True)
class SparseExecutionPlan:
    method: str
    specification_json: str
    target_paths: tuple[str, ...]
    layer_plans: tuple = ()
    version: int = 1


@dataclass(frozen=True)
class SparseTransformResult:
    model: nn.Module
    evidence: dict
    plan: SparseExecutionPlan


def plan_sparse(model, method, spec, *, target_paths=None):
    reject_method(method)  # must precede traversal, RNG, copies and data access
    if method not in METHODS or not isinstance(model, nn.Module):
        raise SparseDomainError("unsupported_method", "Explicit supported sparse method and module required")
    if not is_dataclass(spec) or not getattr(type(spec), "__dataclass_params__").frozen:
        raise SparseDomainError("invalid_specification", "An immutable method specification is required")
    parsed = parse_sparse_spec(method, asdict(spec))
    if type(parsed) is not type(spec):
        raise SparseDomainError('invalid_specification', 'Method and specification must match')
    specification_json = json.dumps(asdict(spec), allow_nan=False, sort_keys=True)
    modules = module_paths(model)
    if target_paths is None:
        paths = tuple(path for path, module in modules.items() if type(module) is nn.Linear) if method in ("hassle_free", "losparse") else ("",)
    else:
        if (not isinstance(target_paths, (tuple, list)) or not target_paths
                or any(not isinstance(p, str) for p in target_paths) or len(set(target_paths)) != len(target_paths)):
            raise SparseDomainError("invalid_targets", "Distinct explicit module paths required")
        paths = tuple(target_paths)
    if not paths or any(path not in modules for path in paths):
        raise SparseDomainError("invalid_targets", "No eligible target or unknown module path")
    if any(a != b and (a == "" or b.startswith(a+".")) for a in paths for b in paths):
        raise SparseDomainError("invalid_targets", "Ancestor and descendant targets cannot overlap")
    topology = validate_parameter_topology(model, paths)
    if topology.parameter_aliases or topology.storage_aliases:
        raise SparseDomainError("unsupported_topology", "Sparse transformations require independent target parameter storage")
    plans = []
    if method in ("hassle_free", "losparse"):
        expected = HASSLEFreeSpec if method == "hassle_free" else LoSparseSpec
        if type(spec) is not expected:
            raise SparseDomainError("invalid_specification", "Method and specification must match")
        for path in paths:
            layer = modules[path]
            if type(layer) is not nn.Linear:
                raise SparseDomainError("unsupported_architecture", "Linear profile requires exact nn.Linear targets")
            if tuple(layer.weight.shape) != (layer.out_features, layer.in_features) or (
                    layer.bias is not None and tuple(layer.bias.shape) != (layer.out_features,)):
                raise SparseDomainError('invalid_state', 'Linear parameter shapes conflict with declared dimensions')
            for value in (layer.weight, layer.bias):
                if value is not None and (value.layout != torch.strided or value.device != layer.weight.device
                        or value.dtype != layer.weight.dtype or not bool(torch.isfinite(value).all())):
                    raise SparseDomainError('invalid_state', 'Linear parameters require finite ordinary tensors with matching precision')
            plans.append(plan_linear(layer.in_features, layer.out_features, spec,
                dtype=str(layer.weight.dtype).removeprefix("torch."), device=str(layer.weight.device),
                bias=layer.bias is not None, storage_prefix=path or "linear"))
    return SparseExecutionPlan(method, specification_json, paths, tuple(plans))


def _inputs(model, paths, calibration):
    if not isinstance(calibration, torch.Tensor) or not calibration.numel() or not bool(torch.isfinite(calibration).all()):
        raise SparseDomainError("missing_calibration", "Finite tensor calibration inputs required")
    captures, handles = {}, []
    modules = module_paths(model)
    for path in paths:
        def capture(module, args, path=path):
            if len(args) != 1 or not isinstance(args[0], torch.Tensor):
                raise SparseDomainError("unsupported_input", "Sparse targets require one tensor input")
            captures.setdefault(path, []).append(args[0].detach().clone())
        handles.append(modules[path].register_forward_pre_hook(capture))
    try:
        with torch.no_grad():
            try:
                model(calibration)
            except SparseDomainError:
                raise
            except (RuntimeError, ValueError, TypeError) as error:
                raise SparseDomainError('unsupported_input', 'Calibration does not match the declared graph') from error
    finally:
        for handle in handles:
            handle.remove()
    if set(captures) != set(paths):
        raise SparseDomainError("inactive_target", "Every declared target must execute on calibration inputs")
    return {path: torch.cat(values, dim=0) for path, values in captures.items()}


def transform_sparse(model, method, spec, *, calibration=None, target_paths=None, **options):
    plan = plan_sparse(model, method, spec, target_paths=target_paths)
    allowed = {'train'} if method == 'losparse' else set()
    if set(options) - allowed:
        raise SparseDomainError('invalid_specification', 'Unknown execution options for this sparse method')
    if method == "losparse" and "train" not in options:
        raise SparseDomainError("missing_training", "LoSparse requires an explicit training callback before finalization")
    original_modes = {path: layer.training for path, layer in module_paths(model).items()}
    local = deepcopy(model).eval()
    modules = module_paths(local)
    needs_inputs = method in ("sola", "flexigpt") or (method == 'moe_i2' and spec.ranks is not None)
    captures = _inputs(local, plan.target_paths, calibration) if needs_inputs else {}
    replacements, records = {}, []
    if method == "losparse":
        from .linear import prepare_losparse, finalize_losparse
        prepared = {path: prepare_losparse(modules[path], spec) for path in plan.target_paths}
        local = replace_modules_atomically(local, prepared)
        options["train"](local, tuple(prepared.values()), spec)
        replacements = {path: finalize_losparse(layer) for path, layer in prepared.items()}
        records = [{"path": path, "method_evidence": {"training_steps": spec.total_steps,
                    "residual_rows": spec.residual_rows}} for path in plan.target_paths]
    # Prepare every candidate first; publish a complete independent graph once.
    for path in (() if method == "losparse" else plan.target_paths):
        layer = modules[path]
        if method == "hassle_free":
            from .linear import hassle_free
            # Later targets see the already transformed predecessors. The
            # caller's model remains untouched until the independent result.
            current = _inputs(local, (path,), calibration)[path]
            replacement, evidence = hassle_free(module_paths(local)[path], current, spec)
            local = replace_modules_atomically(local, {path: replacement})
        elif method == "sola":
            from .channels import sola
            replacement, evidence = sola(layer, captures[path], spec)
        elif method in ("slice_mlp", "partition_mlp"):
            from .channels import slice_mlp, partition_mlp
            replacement, evidence = (slice_mlp if method == "slice_mlp" else partition_mlp)(layer, spec.channels)
        elif method == "flexigpt":
            from . import architectures
            replacement, evidence = getattr(architectures, method)(layer, captures[path], spec, **options)
        elif method == "prunenet":
            from .policies import prunenet
            replacement, evidence = prunenet(layer, spec)
        else:
            from .moe import moe_i2
            replacement, evidence = moe_i2(layer, captures.get(path), spec, **options)
        replacements[path] = replacement
        records.append({"path": path, "method_evidence": asdict(evidence) if is_dataclass(evidence) else evidence})
    transformed = local if method == 'hassle_free' else replace_modules_atomically(local, replacements)
    for path, layer in module_paths(transformed).items():
        layer.training = original_modes.get(path, model.training)
    evidence = json.loads(json.dumps({"version": 1, "method": method,
        "specification": json.loads(plan.specification_json), "layers": records,
        "calibration_order": 'declared_targets_with_current_predecessors' if method == 'hassle_free' else 'original_independent_graph',
        "optimizer_rebuild_required": True}, allow_nan=False))
    transformed._fedcore_method_evidence = evidence
    return SparseTransformResult(transformed, evidence, plan)
