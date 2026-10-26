"""Owned CPU interpreter for the checked quantization graph facade."""
from dataclasses import dataclass
import io
import time
import hashlib

import torch
from torch import nn

from fedcore.algorithm.low_rank.candidate_storage import (capture_inventory, clone_preserving_storages,
    inventory_cost, inventory_fingerprint, CandidateStorageError, storage_guard)
from fedcore.algorithm.low_rank.grammar_execution import StructuralInvalid, NumericalFailure, ResourceFailure, CallbackFailure
from fedcore.algorithm.low_rank.plans import ValidationFailure, Violation
from fedcore.algorithm.low_rank.topology import module_paths, replace_modules_atomically
from .graph_specs import (CheckedCompressionPlan, QuantizationGraphRequest, Method, OperatorRecipe, ExecutionTargetSpec,
    RepresentationSpec, fingerprint, recipe_graph, validate_transform)
from .portable_layers import PortableQuantizedLinear, PortableQuantizedFactorLinear


@dataclass(frozen=True)
class ExportVerifiedArtifact:
    model: nn.Module
    plan: CheckedCompressionPlan
    checkpoint_bytes: bytes
    storage_bytes: int
    evidence: dict


def calibration_identity(calibration):
    from fedcore.algorithm.low_rank.compression_graph import CalibrationRef
    if calibration is None:
        return None
    values = sorted(calibration.items()) if type(calibration) is dict else [("model_input",calibration)]
    digest = hashlib.sha256()
    for path,value in values:
        if (type(path) is not str or type(value) is not torch.Tensor or value.device.type != "cpu"
                or value.dtype not in (torch.float32,torch.float64) or value.ndim < 2
                or not value.numel() or not bool(torch.isfinite(value).all())):
            raise ValueError("Explicit finite CPU calibration tensors with ordered rows required")
        digest.update(path.encode()); digest.update(str((tuple(value.shape),value.dtype)).encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    value = values[0][1]
    return CalibrationRef(digest.hexdigest(),tuple(value.shape),str(value.dtype)[6:])


def check_quantization_graph(model, request, *, calibration=None):
    """No forward, calibration or weight transformation occurs at this boundary."""
    errors, aliases, graphs, representations = [], [], [], []
    if type(request) is not QuantizationGraphRequest or not isinstance(model, nn.Module):
        return ValidationFailure((Violation("InvalidRequest", "request", "Typed request and torch model required"),))
    if (type(request.target) is not ExecutionTargetSpec or request.target.runtime != "cpu_reference" or request.target.allow_float_decode is not True
            or type(request.max_peak_bytes) is not int or request.max_peak_bytes < 1
            or request.max_storage_bytes is not None and (type(request.max_storage_bytes) is not int or request.max_storage_bytes < 0)
            or type(request.recipes) is not tuple or not request.recipes):
        return ValidationFailure((Violation("UnsupportedExecution", "request", "Explicit CPU reference execution and finite resource limits required"),))
    try:
        calibration_ref = calibration_identity(calibration)
        inventory = capture_inventory(model)
        source = inventory_fingerprint(inventory)
        paths = module_paths(model)
        used = set()
        for recipe in request.recipes:
            if type(recipe) is not OperatorRecipe or type(recipe.path) is not str:
                errors.append(Violation("InvalidRecipe", "recipes", "Typed recipe with string path required"))
                continue
            layer = paths.get(recipe.path)
            if type(layer) is not nn.Linear or layer.weight.dtype not in (torch.float32, torch.float64):
                errors.append(Violation("UnsupportedOperator", recipe.path, "Only original CPU float Linear is accepted; converted inference cannot re-enter QAT"))
                continue
            scope = tuple(sorted(path for path, item in paths.items() if item is layer))
            if used.intersection(scope) or any(p.startswith(q+".") or q.startswith(p+".") for p in scope for q in used):
                errors.append(Violation("OverlappingScope", recipe.path, "Closed operator scopes must be distinct"))
            used.update(scope)
            # Partial ties and shared views are not independent scopes.
            tensor_names = {p+".weight" if p else "weight" for p in scope} | {p+".bias" if p else "bias" for p in scope}
            for storage in inventory:
                names = {view.path for view in storage.views}
                if names.intersection(tensor_names) and not names.issubset(tensor_names):
                    errors.append(Violation("CoupledStorage", recipe.path, "Partial parameter ties must form one supported region"))
            failure = validate_transform(recipe.transform, tuple(layer.weight.shape))
            if failure is not None:
                errors.extend(failure.violations)
                continue
            aliases.append(scope)
            graphs.append(recipe_graph(recipe.transform))
            representations.append(RepresentationSpec(recipe.transform.method.value, tuple(layer.weight.shape), recipe.transform.rank, int(layer.bias is not None)))
        if errors:
            return ValidationFailure(tuple(errors))
        # Conservative transform workspace bound; never reported as a measured peak.
        workspace = inventory_cost(inventory, "bytes") * 4 + max(paths[r.path].weight.numel()*64 for r in request.recipes)
        if workspace > request.max_peak_bytes:
            return ValidationFailure((Violation("PeakBudget", "request", "Conservative transformation workspace exceeds the declared limit"),))
        return CheckedCompressionPlan(request, source, tuple(aliases), tuple(graphs), tuple(representations),
            fingerprint({"source": source, "request": request, "aliases": aliases, "graphs": graphs,"calibration":calibration_ref}),calibration_ref)
    except (ValueError, TypeError, KeyError, CandidateStorageError) as error:
        return ValidationFailure((Violation("UnsupportedCapability", "model", str(error)),))


def _build(layer, spec, rows):
    generator = torch.Generator().manual_seed(spec.seed)
    w, b = layer.weight.detach(), None if layer.bias is None else layer.bias.detach()
    if spec.method is Method.DENSE:
        from copy import deepcopy
        return deepcopy(layer), {"method": "dense"}
    if spec.method is Method.PTQ:
        return PortableQuantizedLinear.from_weight(w, b, spec=spec.weight, generator=generator), {"method": spec.method.value}
    if spec.method is Method.DSVD:
        u, s, vh = torch.linalg.svd(w.double(), full_matrices=False)
        left, right = u[:, :spec.rank] * s[:spec.rank].sqrt(), s[:spec.rank].sqrt()[:, None] * vh[:spec.rank]
        return PortableQuantizedFactorLinear.from_factors(left, right, b, left_spec=spec.left, right_spec=spec.right, generator=generator), {"method": spec.method.value}
    from .matrix_transforms import build_method
    return build_method(w, b, spec, rows, generator)


def execute_quantization_graph(model, plan, *, calibration=None, ledger=None):
    if type(plan) is not CheckedCompressionPlan:
        return StructuralInvalid((Violation("UncheckedPlan", "plan", "A checked plan is required"),))
    checked = check_quantization_graph(model, plan.request,calibration=calibration)
    if isinstance(checked, ValidationFailure) or checked != plan:
        return StructuralInvalid((Violation("StalePlan", "plan", "Source or checked metadata has changed"),))
    started = time.perf_counter()
    try:
        storage_guard(0, plan.request.max_peak_bytes)
        paths = module_paths(model)
        # Source is observed on an owned copy. Per-operator data is fixed M0 data.
        rows = {}
        if any(r.transform.method in (Method.CALDERA, Method.TENDER) for r in plan.request.recipes):
            if calibration is None:
                return StructuralInvalid((Violation("MissingCalibration", "calibration", "CALDERA/Tender require explicit input rows"),))
            if isinstance(calibration, dict):
                rows = {key: value.detach().clone() for key, value in calibration.items()}
            else:
                observed = clone_preserving_storages(model, max_peak_bytes=plan.request.max_peak_bytes).eval()
                hooks = []
                for recipe in plan.request.recipes:
                    def hook(_module, inputs, path=recipe.path):
                        rows[path] = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).clone()
                    hooks.append(module_paths(observed)[recipe.path].register_forward_pre_hook(hook))
                try:
                    with torch.no_grad():
                        observed(calibration)
                finally:
                    for hook in hooks:
                        hook.remove()
        replacements, evidence = {}, []
        for recipe, aliases in zip(plan.request.recipes, plan.aliases):
            if ledger is not None:
                ledger.checkpoint("build_operator")
            built, details = _build(paths[recipe.path], recipe.transform, rows.get(recipe.path))
            for path in aliases:
                replacements[path] = built
            evidence.append({"path": recipe.path, "process": plan.processes[len(evidence)], **details})
        candidate = replace_modules_atomically(clone_preserving_storages(model, max_peak_bytes=plan.request.max_peak_bytes), replacements).eval()
        cost = inventory_cost(capture_inventory(candidate), "bytes")
        if plan.request.max_storage_bytes is not None and cost > plan.request.max_storage_bytes:
            return ResourceFailure("StorageBudget", f"Loaded tensor storage {cost} exceeds {plan.request.max_storage_bytes}")
        from fedcore.tools.registry.checkpoint_manager import CheckpointManager
        manager = CheckpointManager(None)
        checkpoint = manager.serialize_to_bytes(candidate)
        restored = manager.deserialize_from_bytes(checkpoint)
        restored_cost = inventory_cost(capture_inventory(restored), "bytes")
        if restored_cost != cost:
            return NumericalFailure("StorageRoundTrip", "Physical storage changed after export")
        if calibration is not None and not isinstance(calibration, dict):
            with torch.no_grad():
                if not torch.equal(candidate(calibration), restored(calibration)):
                    return NumericalFailure("OutputRoundTrip", "Loaded whole-model output differs")
        return ExportVerifiedArtifact(restored, plan, checkpoint, cost,
            {"operators": evidence, "serialized_bytes": len(checkpoint), "loaded_tensor_bytes": cost,
             "build_seconds": time.perf_counter()-started, "observation_semantics": "frozen_snapshot_surrogate",
             "workspace_policy": "conservative_preflight_not_measured_peak"})
    except (ValueError, TypeError, KeyError, RuntimeError, OverflowError) as error:
        return NumericalFailure(type(error).__name__, str(error))


def compress_quantization_graph(model, request, *, calibration=None, ledger=None):
    plan = check_quantization_graph(model, request,calibration=calibration)
    if isinstance(plan, ValidationFailure):
        return StructuralInvalid(plan.violations)
    return execute_quantization_graph(model, plan, calibration=calibration, ledger=ledger)
