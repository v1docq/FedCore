"""Owned CPU interpreter for the checked quantization graph facade."""
from dataclasses import dataclass
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
    RepresentationSpec, fingerprint, recipe_graph, validate_transform,DEFAULT_MAX_PEAK_BYTES)


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
    if type(calibration) is dict and (not calibration or any(type(key) is not str for key in calibration)):
        raise ValueError('Nonempty calibration mapping with string paths required')
    values = sorted(calibration.items()) if type(calibration) is dict else [("model_input",calibration)]
    digest = hashlib.sha256()
    digest.update(b'operator_rows\0' if type(calibration) is dict else b'root_model_inputs\0')
    for path,value in values:
        if (type(path) is not str or type(value) is not torch.Tensor or value.device.type != "cpu"
                or value.dtype not in (torch.float32,torch.float64) or value.ndim < 2
                or not value.numel() or not bool(torch.isfinite(value).all())):
            raise ValueError("Explicit finite CPU calibration tensors with ordered rows required")
        digest.update(path.encode()); digest.update(str((tuple(value.shape),value.dtype)).encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    value = values[0][1]
    return CalibrationRef(digest.hexdigest(),tuple(value.shape),str(value.dtype)[6:])


def source_identity(model):
    """Pin tensor content, declared architecture, aliases and module modes."""
    from fedcore.tools.registry.checkpoint_manager import describe_model
    paths=module_paths(model)
    for path,layer in paths.items():
        if (any(name in layer.__dict__ for name in ('forward','state_dict','_save_to_state_dict'))
                or layer._forward_hooks or layer._forward_pre_hooks or layer._backward_hooks):
            raise ValueError('Modified source behavior/hooks need an explicit capability adapter')
    architecture=describe_model(model)
    if architecture is None:raise ValueError('A fixed checkpointable source architecture required')
    return fingerprint({'inventory':inventory_fingerprint(capture_inventory(model)),'architecture':architecture,
                        'modes':{path:layer.training for path,layer in paths.items()}})


def check_quantization_graph(model, request, *, calibration=None):
    """No forward, calibration or weight transformation occurs at this boundary."""
    errors, aliases, graphs, representations = [], [], [], []
    if type(request) is not QuantizationGraphRequest or not isinstance(model, nn.Module):
        return ValidationFailure((Violation("InvalidRequest", "request", "Typed request and torch model required"),))
    from fedcore.algorithm.low_rank.compression_graph import SourceMode
    if (type(request.source_mode) is not SourceMode or type(request.target) is not ExecutionTargetSpec or request.target.runtime != "cpu_reference" or request.target.allow_float_decode is not True
            or type(request.max_peak_bytes) is not int or not 1<=request.max_peak_bytes<=DEFAULT_MAX_PEAK_BYTES
            or request.max_storage_bytes is not None and (type(request.max_storage_bytes) is not int or request.max_storage_bytes < 0)
            or type(request.recipes) is not tuple or not request.recipes):
        return ValidationFailure((Violation("UnsupportedExecution", "request", "Explicit CPU reference execution and finite resource limits required"),))
    try:
        calibration_ref = calibration_identity(calibration)
        if request.source_mode is SourceMode.CURRENT_STUDENT and type(calibration) is dict:
            raise ValueError('Current-student observation requires root inputs, not frozen operator rows')
        inventory = capture_inventory(model)
        source = source_identity(model)
        paths = module_paths(model)
        used = set()
        for recipe in request.recipes:
            if type(recipe) is not OperatorRecipe or type(recipe.path) is not str:
                errors.append(Violation("InvalidRecipe", "recipes", "Typed recipe with string path required"))
                continue
            from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear
            from .graph_specs import TransformSpec
            layer = paths.get(recipe.path)
            if type(recipe.transform) is not TransformSpec:
                errors.append(Violation('InvalidTransform',recipe.path,'Typed transform required'));continue
            ready = type(layer) is FactorizedLinear and recipe.transform.method is Method.FACTORS
            if (type(layer) is not nn.Linear and not ready or layer is not None and
                    next(layer.parameters()).dtype not in (torch.float32,torch.float64)):
                errors.append(Violation("UnsupportedOperator", recipe.path, "Only original CPU float Linear is accepted; converted inference cannot re-enter QAT"))
                continue
            scope = tuple(sorted(path for path, item in paths.items() if item is layer))
            from fedcore.algorithm.low_rank.compression_composition import _under
            if used.intersection(scope) or any(_under(p,q) or _under(q,p) for p in scope for q in used):
                errors.append(Violation("OverlappingScope", recipe.path, "Closed operator scopes must be distinct"))
            used.update(scope)
            # Partial ties and shared views are not independent scopes.
            tensor_names = {view.path for storage in inventory for view in storage.views if any(p=='' or view.path.startswith(p+'.') for p in scope)}
            for storage in inventory:
                names = {view.path for view in storage.views}
                if names.intersection(tensor_names) and not names.issubset(tensor_names):
                    errors.append(Violation("CoupledStorage", recipe.path, "Partial parameter ties must form one supported region"))
            shape=(layer.out_features,layer.in_features)
            failure = validate_transform(recipe.transform, shape)
            if failure is not None:
                errors.extend(failure.violations)
                continue
            aliases.append(scope)
            graphs.append(recipe_graph(recipe.transform))
            bias=layer.left.bias if ready else layer.bias
            representations.append(RepresentationSpec(recipe.transform.method.value,shape,layer.rank if ready else recipe.transform.rank,int(bias is not None)))
            if type(calibration) is dict:
                if recipe.path not in calibration or calibration[recipe.path].ndim!=2 or calibration[recipe.path].shape[1]!=layer.in_features:
                    errors.append(Violation('CalibrationCoordinates',recipe.path,'Explicit operator rows must match original input coordinates'))
        if errors:
            return ValidationFailure(tuple(errors))
        # Conservative transform workspace bound; never reported as a measured peak.
        workspace = inventory_cost(inventory, "bytes") * 4 + max(paths[r.path].in_features*paths[r.path].out_features*128 +
            (paths[r.path].in_features**2*64 if r.transform.method is Method.CALDERA else 0) for r in request.recipes)
        if workspace > request.max_peak_bytes:
            return ValidationFailure((Violation("PeakBudget", "request", "Conservative transformation workspace exceeds the declared limit"),))
        return CheckedCompressionPlan(request, source, tuple(aliases), tuple(graphs), tuple(representations),
            fingerprint({"source": source, "request": request, "aliases": aliases, "graphs": graphs,"calibration":calibration_ref}),calibration_ref)
    except (ValueError, TypeError, KeyError, CandidateStorageError) as error:
        return ValidationFailure((Violation("UnsupportedCapability", "model", str(error)),))


def execute_quantization_graph(model, plan, *, calibration=None, ledger=None,cache=None):
    with torch.random.fork_rng(devices=[]):
        return _execute_quantization_graph(model,plan,calibration=calibration,ledger=ledger,cache=cache)


def _execute_quantization_graph(model, plan, *, calibration=None, ledger=None,cache=None):
    if type(plan) is not CheckedCompressionPlan:
        return StructuralInvalid((Violation("UncheckedPlan", "plan", "A checked plan is required"),))
    checked = check_quantization_graph(model, plan.request,calibration=calibration)
    if isinstance(checked, ValidationFailure) or checked != plan:
        return StructuralInvalid((Violation("StalePlan", "plan", "Source or checked metadata has changed"),))
    started = time.perf_counter()
    try:
        storage_guard(0, plan.request.max_peak_bytes)
        paths = module_paths(model)
        # Frozen snapshots and current-student observations are explicit modes.
        rows = {}
        if any(r.transform.method in (Method.CALDERA, Method.TENDER) for r in plan.request.recipes):
            if calibration is None:
                return StructuralInvalid((Violation("MissingCalibration", "calibration", "CALDERA/Tender require explicit input rows"),))
            if isinstance(calibration, dict):
                rows = {key: value.detach().clone() for key, value in calibration.items()}
            else:
                from .observation import observe_operator_rows
                rows=observe_operator_rows(model,calibration,tuple(r.path for r in plan.request.recipes),plan.request.max_peak_bytes)
        from fedcore.algorithm.low_rank.compression_graph import SourceMode
        work=clone_preserving_storages(model,max_peak_bytes=plan.request.max_peak_bytes)
        replacements, evidence = {}, []
        for recipe, aliases in zip(plan.request.recipes, plan.aliases):
            if ledger is not None:
                ledger.checkpoint("build_operator")
            from .process import execute_process
            index=len(evidence)
            predecessor=plan.source_fingerprint
            if plan.request.source_mode is SourceMode.CURRENT_STUDENT:
                predecessor=source_identity(work)
                if cache is not None:cache.observe_scope(aliases,predecessor)
                if recipe.transform.method in (Method.CALDERA,Method.TENDER):
                    from .observation import observe_operator_rows
                    from .cache import CacheIdentity
                    key=CacheIdentity('observation',plan.source_fingerprint,predecessor,plan.calibration.fingerprint,
                        ':'.join(aliases)+':original_operator_input_output','ObserveInput','current_student')
                    hit=cache.get(key) if cache is not None else None
                    rows[recipe.path]=hit[0] if hit is not None else observe_operator_rows(work,calibration,(recipe.path,),plan.request.max_peak_bytes)[recipe.path]
                    if cache is not None and hit is None:cache.put(key,(rows[recipe.path],),dependencies=(predecessor,plan.calibration.fingerprint))
            built,details=execute_process(paths[recipe.path],recipe.transform,plan.processes[index],rows.get(recipe.path),
                model_version=predecessor,scope=aliases,ledger=ledger,cache=cache,
                calibration_fingerprint=plan.calibration.fingerprint if plan.calibration else 'none',max_peak_bytes=plan.request.max_peak_bytes)
            for path in aliases:
                replacements[path] = built
            evidence.append({"path": recipe.path, "process": plan.processes[len(evidence)], **details})
            if plan.request.source_mode is SourceMode.CURRENT_STUDENT:
                work=replace_modules_atomically(work,{path:built for path in aliases})
        candidate = (work if plan.request.source_mode is SourceMode.CURRENT_STUDENT else replace_modules_atomically(work,replacements)).eval()
        cost = inventory_cost(capture_inventory(candidate), "bytes")
        if plan.request.max_storage_bytes is not None and cost > plan.request.max_storage_bytes:
            return ResourceFailure("StorageBudget", f"Loaded tensor storage {cost} exceeds {plan.request.max_storage_bytes}")
        from fedcore.tools.registry.checkpoint_manager import CheckpointManager
        from .graph_specs import payload
        candidate._fedcore_method_evidence=payload({'schema':plan.schema,'plan':plan,'operators':evidence})
        manager = CheckpointManager(None)
        checkpoint = manager.serialize_to_bytes(candidate)
        restored = manager.deserialize_from_bytes(checkpoint)
        restored_cost = inventory_cost(capture_inventory(restored), "bytes")
        if restored_cost != cost:
            return NumericalFailure("StorageRoundTrip", "Physical storage changed after export")
        if calibration is not None and not isinstance(calibration, dict):
            with torch.no_grad():
                actual,loaded=candidate(calibration),restored(calibration)
                if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(loaded).all()):
                    return NumericalFailure('OutputOverflow','Whole-model output must be finite after export')
                if not torch.equal(actual,loaded):
                    return NumericalFailure("OutputRoundTrip", "Loaded whole-model output differs")
        if ledger is not None:ledger.checkpoint('export_verified')
        return ExportVerifiedArtifact(restored, plan, checkpoint, cost,
            {"operators": evidence, "serialized_bytes": len(checkpoint), "loaded_tensor_bytes": cost,
             "build_seconds": time.perf_counter()-started, "observation_semantics": plan.request.source_mode.value,
             "workspace_policy": "conservative_preflight_not_measured_peak"})
    except (ValueError, TypeError, KeyError, RuntimeError, OverflowError) as error:
        from .resources import StageBudgetExceeded,storage_resource_failure
        if isinstance(error,StageBudgetExceeded):return error.failure
        resource_failure=storage_resource_failure(error)
        if resource_failure is not None:return resource_failure
        return NumericalFailure(type(error).__name__, str(error))


def compress_quantization_graph(model, request, *, calibration=None, ledger=None,cache=None):
    plan = check_quantization_graph(model, request,calibration=calibration)
    if isinstance(plan, ValidationFailure):
        return StructuralInvalid(plan.violations)
    return execute_quantization_graph(model, plan, calibration=calibration, ledger=ledger,cache=cache)
