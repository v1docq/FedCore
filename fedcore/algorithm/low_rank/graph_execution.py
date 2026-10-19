"""Runtime boundary for the two exact fixed compression-graph recipes.

Preparation reads capabilities and hashes source/data without copying or running
the model. Execution validates the complete plan and its source before calling
the existing whole-recipe interpreter. Graph stages document inline work in
that interpreter; no separately addressable statistics/metric cache is implied.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .compression_graph import (CalibrationRef, FixedCompressionGraph,
    ModelVersion, Recipe, RuntimeOptions, compile_fixed_graph, validate_fixed_graph)
from .execution import (WeightedProfileError, _fingerprint, _finite_observations,
                        WeightedTransformResult, plan_weighted, transform_weighted)
from .method_execution import MethodTransformResult, _rng_isolated
from .method_specs import ASVD
from .plans import (FixedRank, MetricPolicy, ParameterFraction, RankFraction,
                    ValidationFailure, Violation)
from .topology import inspect_topology


class CompressionGraphError(WeightedProfileError):
    """A structured planning, graph-contract or source-version failure."""

    def __init__(self, violations):
        self.violations = tuple(violations)
        super().__init__("; ".join(f"{v.code}: {v.path}: {v.message}" for v in self.violations))


@dataclass(frozen=True)
class FixedGraphResult:
    """Runtime result; live model/results are owned here, never in ``graph``."""
    model: nn.Module
    graph: FixedCompressionGraph
    legacy_result: WeightedTransformResult | MethodTransformResult
    evidence: dict


def _raise_failure(failure):
    if isinstance(failure, ValidationFailure):
        raise CompressionGraphError(failure.violations)
    return failure


def _fail(code, path, message):
    raise CompressionGraphError((Violation(code, path, message),))


def _calibration(calibration, options):
    if (not isinstance(calibration, torch.Tensor) or calibration.device.type != "cpu"
            or calibration.dtype not in (torch.float32, torch.float64) or calibration.ndim < 2
            or not calibration.numel() or not len(calibration)):
        _fail("InvalidCalibration", "calibration", "Separate nonempty CPU float32/float64 calibration data is required")
    if (type(options.max_workspace_bytes) is int and options.max_workspace_bytes > 0
            and calibration[0].numel() * calibration.element_size() > options.max_workspace_bytes):
        _fail("ResourceLimitExceeded", "calibration", "One calibration sample exceeds workspace")
    if not _finite_observations(calibration):
        _fail("InvalidCalibration", "calibration", "Calibration observations must be finite")


def prepare_fixed_graph(model, calibration, *, recipe=Recipe.WEIGHTED_SVD,
        method_spec=None, rank=None, rank_ratio=None, parameter_fraction=None,
        ranks=None, target_paths=None, policy=MetricPolicy(), representation="two_layers",
        batch_size=16, max_workspace_bytes=256 * 1024**2,
        max_peak_bytes=12 * 1024**3, graph_version=None,
        parameter_budget=None, tensor_byte_budget=None):
    """Read runtime metadata and compile a bounded, replayable workflow.

    Weighted supports its existing CPU float32/64 Linear/channel Conv1d/2d
    capabilities and shared rank policy. ASVD supports the current explicit
    Linear profile, including per-path ranks and its sequential current-model
    observations. Method-only labels, teachers and recovery are outside G1.

    ``graph_version`` also identifies custom forward/hidden Python state that
    cannot be inferred from a state_dict and public data-valued module settings.
    The caller must advance it when such state changes.
    """
    try:
        recipe = Recipe(recipe)
    except (TypeError, ValueError):
        _fail("UnsupportedRecipe", "recipe", "Only weighted_svd and asvd fixed recipes are executable")
    options = RuntimeOptions(batch_size, max_workspace_bytes, max_peak_bytes,
                             parameter_budget, tensor_byte_budget)
    if not isinstance(model, nn.Module):
        _fail("UnsupportedCapability", "model", "A CPU torch Module is required")
    _calibration(calibration, options)
    if graph_version is None:
        graph_version = "weighted-svd-v1" if recipe is Recipe.WEIGHTED_SVD else "p2-profile-v1"
    if type(graph_version) is not str or not graph_version:
        _fail("InvalidSource", "graph_version", "An explicit nonempty graph version is required")
    if recipe is Recipe.WEIGHTED_SVD:
        if method_spec is not None or ranks is not None:
            _fail("UnsupportedOptions", "recipe", "Weighted G1 has no ASVD method spec or per-path ranks")
        transform = plan_weighted(model, rank=rank, rank_ratio=rank_ratio,
            parameter_fraction=parameter_fraction, target_paths=target_paths,
            policy=policy, representation=representation, max_peak_bytes=max_peak_bytes)
    else:
        from .method_execution import plan_method
        method_spec = ASVD() if method_spec is None else method_spec
        if type(method_spec) is not ASVD or policy != MetricPolicy() or representation != "two_layers":
            _fail("UnsupportedOptions", "method_spec", "ASVD requires ASVD parameters and its fixed two-layer representation")
        transform = plan_method(model, method_spec, rank=rank, rank_ratio=rank_ratio,
            parameter_fraction=parameter_fraction, ranks=ranks, target_paths=target_paths,
            max_peak_bytes=max_peak_bytes)
    checkpoint, data = _fingerprint(model, calibration, graph_version)
    return _raise_failure(compile_fixed_graph(recipe, transform,
        ModelVersion(graph_version, checkpoint),
        CalibrationRef(data, tuple(calibration.shape), str(calibration.dtype).removeprefix("torch.")),
        options, method_spec=method_spec, module_aliases=inspect_topology(model).module_aliases))


def _weighted_structure(structure):
    if type(structure) is FixedRank:
        return {"rank": structure.rank}
    if type(structure) is RankFraction:
        return {"rank_ratio": structure.fraction}
    if type(structure) is ParameterFraction:
        return {"parameter_fraction": structure.fraction}
    _fail("InvalidRank", "transform.request.structure", "An executable rank rule is required")


def _allocation_failure(error):
    # torch 2.2 exposes this exception under cuda even on CPU-only paths.
    if isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)):
        return True
    # CPU allocator failures are RuntimeError in the supported torch runtime.
    message = str(error).lower()
    return "defaultcpuallocator" in message and any(text in message for text in (
        "not enough memory", "can't allocate memory", "cannot allocate memory"))


def _copy_calibration(calibration, options):
    """Preflight and own the full input buffer before arbitrary forward code.

    The floor covers this named clone plus a runtime reserve. It cannot bound
    allocations inside a user-defined forward or guarantee allocator success.
    The legacy backend performs its own resource preflight after the clone.
    """
    import os
    import psutil
    copy_bytes = calibration.numel() * calibration.element_size()
    rss_before = psutil.Process(os.getpid()).memory_info().rss
    reserve = 64 * 1024**2
    floor = rss_before + copy_bytes + reserve
    if floor > options.max_peak_bytes:
        _fail("ResourceLimitExceeded", "calibration.copy",
              f"Calibration copy requires a named-buffer floor of {floor} bytes; limit is {options.max_peak_bytes}")
    try:
        owned = calibration.detach().clone()
    except (MemoryError, RuntimeError) as error:
        if _allocation_failure(error):
            _fail("ResourceLimitExceeded", "calibration.copy", "Calibration buffer allocation failed after preflight")
        raise
    return owned, {"copy_bytes": copy_bytes, "process_rss_before_bytes": rss_before,
        "runtime_reserve_bytes": reserve, "required_floor_bytes": floor,
        "limit_bytes": options.max_peak_bytes,
        "scope": "named calibration buffer and runtime reserve; arbitrary forward/allocator RSS is not a hard bound"}


@_rng_isolated
def execute_fixed_graph(model, calibration, graph):
    """Validate exact recipe/source/capabilities, then execute its whole backend.

    Nothing supplied separately at execution can override graph parameters.
    A forged node list remains invalid even if its content digest is recomputed.
    The returned evidence distinguishes a declared stage trace from independently
    executed node outputs; numeric evidence remains the legacy result's evidence.
    A preflighted private calibration copy protects caller data from in-place
    operations in user forward code, on successful and failed execution.
    """
    _raise_failure(validate_fixed_graph(graph))
    options = graph.options
    _calibration(calibration, options)
    if not isinstance(model, nn.Module) or any(value.device.type != "cpu"
            for value in (*model.parameters(), *model.buffers())):
        _fail("UnsupportedCapability", "model", "The fixed interpreter requires a CPU torch Module")
    # Hash before re-planning so changed shapes/configurations also have a stable
    # stale-source error. No forward, hook, deepcopy or solver has run here.
    checkpoint, data = _fingerprint(model, calibration, graph.source.graph_version)
    if checkpoint != graph.source.fingerprint:
        _fail("StaleModelVersion", "source", "The original model or declared graph configuration changed")
    actual_data = CalibrationRef(data, tuple(calibration.shape), str(calibration.dtype).removeprefix("torch."))
    if actual_data != graph.calibration:
        _fail("StaleCalibration", "calibration", "Calibration content, shape, dtype or role changed")
    if inspect_topology(model).module_aliases != graph.module_aliases:
        _fail("CapabilityMismatch", "module_aliases", "Declared scopes must match the original model's exact module aliases")
    target_paths = tuple(step.operator.path for step in graph.transform.steps)
    common = {"batch_size": options.batch_size, "max_workspace_bytes": options.max_workspace_bytes,
              "max_peak_bytes": options.max_peak_bytes, "graph_version": graph.source.graph_version}
    if graph.recipe is Recipe.WEIGHTED_SVD:
        specific = {**_weighted_structure(graph.transform.request.structure), "target_paths": target_paths,
                    "policy": graph.transform.request.metric_policy,
                    "representation": graph.transform.request.representation}
        checked = plan_weighted(model, **specific, max_peak_bytes=options.max_peak_bytes)
    else:
        from .method_execution import plan_method
        structures = tuple(step.structure for step in graph.transform.steps)
        specific = {**(_weighted_structure(graph.transform.request.structure)
            if all(structure == graph.transform.request.structure for structure in structures)
            else {"ranks": {step.operator.path: step.rank for step in graph.transform.steps}}),
            "target_paths": target_paths}
        checked = plan_method(model, graph.method_spec, **specific, max_peak_bytes=options.max_peak_bytes)
        # Resolve per-path ranks at the boundary. The legacy request's global
        # rank policy is descriptive; all executed steps must match exactly.
    if checked.steps != graph.transform.steps:
        _fail("CapabilityMismatch", "transform.steps", "Current capabilities/ranks do not match the planned invocation")
    if checked.request != graph.transform.request:
        _fail("CapabilityMismatch", "transform.request", "The executed invocation differs from its plan")
    working_calibration, copy_evidence = _copy_calibration(calibration, options)
    try:
        if graph.recipe is Recipe.WEIGHTED_SVD:
            result = transform_weighted(model, working_calibration, **specific, **common)
        else:
            from .method_execution import transform_method
            result = transform_method(model, working_calibration, graph.method_spec, **specific, **common,
                parameter_budget=options.parameter_budget, tensor_byte_budget=options.tensor_byte_budget)
    except (MemoryError, RuntimeError) as error:
        if _allocation_failure(error):
            _fail("ResourceLimitExceeded", "runtime", "Whole-recipe allocation failed after preflight")
        raise
    evidence = {"graph_fingerprint": graph.fingerprint, "schema_version": graph.schema_version,
        "recipe": graph.recipe.value, "source_mode": graph.source_mode.value,
        "executor": graph.executor, "stage_trace_kind": "declared_inline_whole_recipe_stages",
        "calibration_copy": copy_evidence,
        "planned_stages": [{"node": node.key, "stage": node.stage.value,
                            "scope": list(node.scope)} for node in graph.nodes],
        "numerical_evidence": result.evidence}
    return FixedGraphResult(result.model, graph, result, evidence)
