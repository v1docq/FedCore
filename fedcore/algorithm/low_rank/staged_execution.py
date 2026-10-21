"""Real stage execution for the two canonical fixed recipes, opt-in only.

Only observationally pure eval forwards are cacheable. Model/data/mode/RNG
changes are refused, never replayed. Opaque external state remains the caller's
explicit graph_version responsibility. Statistics persist; other stage outputs
are runtime-owned and released after their last consumer.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import math
import os
import pickle
import random
import time

import numpy as np
import psutil
import torch
from torch import nn

from .approximation import prepare_weighted_metric, solve_weighted_prepared, _finite
from .artifacts import ArtifactStoreError, StatisticsKey, StatisticRef, StatisticsSnapshot
from .compression_graph import ArtifactKind, Recipe, RuntimeOptions, Stage
from .execution import (_fingerprint, _patch_rows, _storage_records, WeightedTransformResult)
from .method_execution import (_collect_rows, _explicit_encoder_graph, _install,
                               _matrix, _finish, _numerics)
from .plans import SnapshotRef
from .statistics import (StatisticsPassport, empty_second_moment, update_second_moment,
                         finish_second_moment)
from .statistical_profiles import (channel_abs_statistics, _solve_input_scales,
                                   StatisticalApproximation, ASVD_SOURCE)
from .topology import inspect_topology, module_paths, replace_modules_atomically
from .allocation import exact_unique_cost


def _fail(code, path, message):
    from .graph_execution import _fail as fail
    fail(code, path, message)


def _guard(options, extra, *, workspace=False):
    if workspace and extra > options.max_workspace_bytes:
        _fail("ResourceLimitExceeded", "stages.workspace", "Named stage buffers exceed workspace")
    floor = psutil.Process(os.getpid()).memory_info().rss + extra + 64 * 1024**2
    if floor > options.max_peak_bytes:
        _fail("ResourceLimitExceeded", "stages.peak", "Named stage buffer floor exceeds declared peak")
    return floor


@dataclass
class _Memory:
    options: RuntimeOptions
    peak_floor: int = 0

    def check(self, extra, *, workspace=False):
        floor = _guard(self.options, extra, workspace=workspace)
        self.peak_floor = max(self.peak_floor, floor)
        return floor


def _rng_fingerprint():
    digest = hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state()), protocol=4))
    digest.update(torch.get_rng_state().numpy().tobytes())
    return digest.hexdigest()


def _observation_identity(model, calibration, graph_version):
    model_id, data_id = _fingerprint(model, calibration, graph_version)
    modes = tuple((path, layer.training) for path, layer in module_paths(model).items())
    return model_id, data_id, modes, _rng_fingerprint()


def _pure_observe(model, calibration, graph_version, operation):
    from .candidate_storage import capture_inventory, inventory_fingerprint
    before = _observation_identity(model, calibration, graph_version)
    before_storage = inventory_fingerprint(capture_inventory(model))
    try:
        result = operation()
    finally:
        try:
            after = _observation_identity(model, calibration, graph_version)
            after_storage = inventory_fingerprint(capture_inventory(model))
        except Exception:
            _fail("UnsupportedObservationEffect", "observe", "Forward changed observable model/data state")
        if before != after or before_storage != after_storage:
            _fail("UnsupportedObservationEffect", "observe", "Staged observation must preserve model, calibration, modes and CPU RNG states")
    return result


def _weighted_observe(model, layer, descriptor, calibration, passports, memory):
    options = memory.options
    width = math.prod(descriptor.shape[1:])
    states = [empty_second_moment(width, passport) for passport in passports]
    forwards = 0
    def capture(_layer, args):
        nonlocal states
        if len(args) != 1 or not isinstance(args[0], torch.Tensor):
            _fail("UnsupportedObservation", descriptor.path, "One tensor input per target is required")
        value = args[0].detach()
        if value.device.type != "cpu" or value.dtype not in (torch.float32, torch.float64):
            _fail("UnsupportedObservation", descriptor.path, "Observed activations must be CPU float32/float64")
        if isinstance(layer, nn.Linear):
            patch_reserve = value.numel() * value.element_size()
        else:
            locations = math.prod((value.shape[i + 2] + 2 * layer.padding[i] - layer.dilation[i] *
                (layer.kernel_size[i] - 1) - 1) // layer.stride[i] + 1 for i in range(value.ndim - 2))
            patch_reserve = 2 * max(0, locations) * width * descriptor.groups * value.element_size()
        memory.check(2 * value.numel() * value.element_size() + patch_reserve +
               4 * descriptor.groups * width * width * 8)
        rows_per_chunk = max(1, min(1024, options.max_workspace_bytes // max(1, 4 * width * descriptor.groups * 8)))
        for grouped in _patch_rows(layer, value, options.max_workspace_bytes):
            for group, rows in enumerate(grouped):
                for chunk in rows.split(rows_per_chunk):
                    states[group] = update_second_moment(states[group], chunk)
    handle = layer.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            for batch in calibration.split(options.batch_size):
                model(batch)
                forwards += 1
    finally:
        handle.remove()
    moments = tuple(finish_second_moment(state) for state in states)
    return StatisticsSnapshot(tuple(moment.matrix for moment in moments), tuple(moment.passport for moment in moments),
                              ArtifactKind.INPUT_SECOND_MOMENT), forwards


def _weighted_matrices(layer, descriptor):
    from fedcore.models.network_impl.decomposed_layers import IDecomposed
    matrix = layer.factor_matrix().detach() if isinstance(layer, IDecomposed) else layer.weight.detach()
    if not isinstance(layer, nn.Linear) and not isinstance(layer, IDecomposed):
        matrix = matrix.reshape(descriptor.groups, descriptor.shape[0] // descriptor.groups,
                                math.prod(descriptor.shape[1:]))
        if descriptor.groups == 1:
            matrix = matrix[0]
    return matrix.unbind(0) if matrix.ndim == 3 else (matrix,)


def _weighted_build(layer, descriptor, answers, representation):
    from fedcore.models.network_impl.decomposed_layers import (
        IDecomposed, DecomposedLinear, DecomposedConv1d, DecomposedConv2d)
    from .reassembly.decomposed_recreation import to_standard_module
    base = to_standard_module(layer) if isinstance(layer, IDecomposed) else layer
    constructor = DecomposedLinear if isinstance(layer, nn.Linear) else DecomposedConv1d if isinstance(layer, nn.Conv1d) else DecomposedConv2d
    result = constructor(base, decomposing_mode=False, compose_mode=representation)
    result.decomposing_mode = True if isinstance(layer, nn.Linear) else "channel"
    grouped = descriptor.groups > 1
    factors = tuple(torch.stack([getattr(answer, name) for answer in answers]) for name in ("u", "s", "vh")) if grouped else (answers[0].u, answers[0].s, answers[0].vh)
    result.set_U_S_Vh(*factors)
    result.compose_weight_for_inference()
    return result


@dataclass(frozen=True)
class _ASVDMetric:
    scales: torch.Tensor
    statistic: torch.Tensor
    passport: StatisticsPassport


def _asvd_solve(layer, metric, spec, rank):
    policy = "reject" if spec.zero_policy == "error" else spec.zero_policy
    factors = _solve_input_scales(_matrix(layer), metric.scales, rank,
                                  objective="asvd_diagonal", zero_policy=policy)
    values = metric.statistic
    return StatisticalApproximation(factors, None, {
        "method": "asvd", "objective": "diagonal_input_column_error", "source": ASVD_SOURCE,
        "statistic": spec.abs_mode, "alpha": float(spec.alpha), "epsilon": float(spec.epsilon),
        "epsilon_placement": "before_power", "zero_policy": policy,
        "zero_channels": (values == 0).nonzero().flatten().tolist(),
        "scales": metric.scales.tolist(), "statistic_values": values.tolist(),
        "statistics_passport": asdict(metric.passport), "channel_correlations": "not_estimated",
        "abs_max_weights": "positive_support_only", "full_rank_policy": factors.nullspace_policy})


def _tensor_bytes(value):
    seen = set()
    def visit(item):
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            identity = (str(item.device), storage._cdata)
            if identity not in seen:
                seen.add(identity)
                return storage.nbytes()
        elif isinstance(item, nn.Module):
            return sum(visit(t) for t in (*item.parameters(), *item.buffers()))
        elif isinstance(item, (tuple, list)):
            return sum(visit(t) for t in item)
        elif hasattr(item, "__dataclass_fields__"):
            return sum(visit(getattr(item, name)) for name in item.__dataclass_fields__)
        return 0
    return visit(value)


@torch.no_grad()
def execute_stages(model, calibration, graph, store, *, phase_callback=None):
    """Interpret all canonical nodes, publishing statistics transactionally."""
    from .graph_execution import FixedGraphResult, CompressionGraphError, _copy_calibration, _allocation_failure
    options = graph.options
    memory = _Memory(options)
    started = time.perf_counter()
    trace, statistic_refs, records, refs = [], [], [], []
    total_forwards = 0
    try:
        # This also refuses a closed/busy store before model/calibration copies.
        with store._transaction() as transaction:
            model_bytes = inspect_topology(model).unique_storage_bytes + sum(b.numel() * b.element_size() for b in model.buffers())
            copy_bytes = model_bytes + calibration.numel() * calibration.element_size()
            stage_bounds = {}
            for step in graph.transform.steps:
                d = math.prod(step.operator.shape[1:])
                m = step.operator.shape[0] // step.operator.groups
                required = 8 * step.operator.groups * (8*d*d + 8*m*d + 4*m*m)
                if required > options.max_workspace_bytes:
                    _fail("ResourceLimitExceeded", "stages.workspace", "Named stage buffers exceed workspace")
                stage_bounds[step.operator.path] = required
            memory.check(copy_bytes + max(stage_bounds.values()))
            working, copy_evidence = _copy_calibration(calibration, options)
            work = deepcopy(model)
            if graph.recipe is Recipe.ASVD:
                work = _explicit_encoder_graph(work)
            modes = {path: layer.training for path, layer in module_paths(work).items()}
            work.eval()
            values = {graph.external_artifacts[0].key: work, graph.external_artifacts[1].key: working}
            consumers = {}
            for node in graph.nodes:
                for port in node.inputs:
                    consumers[port.artifact] = consumers.get(port.artifact, 0) + 1
            step_by_scope = {scope.aliases: (step, scope) for step, scope in zip(graph.transform.steps, graph.scopes)}
            for node in graph.nodes:
                phase_started = time.perf_counter() if phase_callback is not None else None
                if phase_callback is not None:
                    phase_callback({"phase": node.stage.value, "scope": list(node.scope), "node": node.key,
                        "graph_fingerprint": graph.fingerprint, "started_at": phase_started,
                        "completed_at": None, "state": "started"})
                supplied = {port.name: values[port.artifact] for port in node.inputs}
                hit, forwards, stat_ref = False, 0, None
                if node.stage is Stage.APPLY:
                    base = supplied["model"]
                    replacements = {}
                    for port in node.inputs:
                        if port.kind is ArtifactKind.REPLACEMENT:
                            replacement, path = supplied[port.name]
                            replacements[path] = replacement
                    output = replace_modules_atomically(base, replacements)
                    output.eval()
                    work = output
                else:
                    step, scope = step_by_scope[node.scope]
                    descriptor = step.operator
                    path = descriptor.path
                    layer = module_paths(work)[path]
                    if node.stage in (Stage.METRIC, Stage.SOLVE, Stage.BUILD):
                        memory.check(stage_bounds[path], workspace=True)
                    if node.stage is Stage.OBSERVE:
                        identity = _observation_identity(work, working, graph.source.graph_version)
                        kind = node.outputs[0].kind
                        key = StatisticsKey(identity[0], graph.source.graph_version, path, scope.aliases,
                            identity[1], kind, descriptor.dtype, options.batch_size, identity[3],
                            source_fingerprint=graph.source.fingerprint)
                        snapshot = transaction.lookup(key)
                        if snapshot is None:
                            width = math.prod(descriptor.shape[1:])
                            size = descriptor.groups * (width * width if graph.recipe is Recipe.WEIGHTED_SVD else width) * 8
                            transaction.reserve(key, size)
                            memory.check(4 * size, workspace=True)
                            passports = tuple(StatisticsPassport(graph.source.fingerprint, key.graph_version,
                                key.model_fingerprint, path + (f":input:group={index}" if graph.recipe is Recipe.WEIGHTED_SVD else ":input"),
                                key.calibration_fingerprint, method_version=key.estimator_version) for index in range(descriptor.groups))
                            def observe():
                                if graph.recipe is Recipe.WEIGHTED_SVD:
                                    return _weighted_observe(work, layer, descriptor, working, passports, memory)
                                reserve = psutil.Process(os.getpid()).memory_info().rss + 64 * 1024**2
                                rows = _collect_rows(work, layer, working, batch_size=options.batch_size,
                                    workspace=options.max_workspace_bytes, peak=options.max_peak_bytes, reserve=reserve)
                                memory.peak_floor = max(memory.peak_floor, reserve + 2 * rows.numel() * rows.element_size())
                                # _selected_rows casts to float64 and its abs /
                                # weighted reduction can coexist with raw rows.
                                reduction_bytes = (rows.numel() * rows.element_size() +
                                    4 * rows.numel() * 8 + 4 * len(rows) * 8 + rows.shape[1] * 8)
                                memory.check(reduction_bytes, workspace=True)
                                statistic = channel_abs_statistics(rows, passports[0], mode=graph.method_spec.abs_mode)
                                return StatisticsSnapshot((statistic.values,), (statistic.passport,), kind), math.ceil(len(working)/options.batch_size)
                            observed, forwards = _pure_observe(work, working, graph.source.graph_version, observe)
                            memory.check(size)
                            stat_ref = transaction.put(key, observed, max_peak_bytes=options.max_peak_bytes)
                            snapshot = transaction.lookup(key)
                            del observed
                        else:
                            hit = True
                            stat_ref = StatisticRef(key.digest, key)
                        width = math.prod(descriptor.shape[1:])
                        expected_shape = (width, width) if kind is ArtifactKind.INPUT_SECOND_MOMENT else (width,)
                        if len(snapshot.tensors) != descriptor.groups or any(tuple(t.shape) != expected_shape for t in snapshot.tensors):
                            _fail("InvalidSnapshot", path, "Cached statistic does not match operator dimensions/groups")
                        output = snapshot
                        statistic_refs.append(stat_ref)
                        refs.append(SnapshotRef(key.digest, graph.source.fingerprint, key.graph_version))
                        total_forwards += forwards
                    elif node.stage is Stage.METRIC:
                        snapshot = supplied["statistics"]
                        if graph.recipe is Recipe.WEIGHTED_SVD:
                            output = tuple(prepare_weighted_metric(tensor, graph.transform.request.metric_policy) for tensor in snapshot.tensors)
                        else:
                            spec = graph.method_spec
                            output = _ASVDMetric(_finite((snapshot.tensors[0] + spec.epsilon).pow(spec.alpha), "ASVD scales"),
                                                 snapshot.tensors[0], snapshot.passports[0])
                    elif node.stage is Stage.SOLVE:
                        metric = supplied["metric"]
                        if graph.recipe is Recipe.WEIGHTED_SVD:
                            output = tuple(solve_weighted_prepared(weight, prepared, step.rank)
                                for weight, prepared in zip(_weighted_matrices(layer, descriptor), metric))
                        else:
                            output = _asvd_solve(layer, metric, graph.method_spec, step.rank)
                        records.append({"path": path, "requested_rank": step.rank,
                                        "kind": descriptor.kind, "shape": list(descriptor.shape), "groups": descriptor.groups,
                                        "representation": graph.transform.request.representation,
                                        "statistics": [asdict(p) for p in snapshot.passports],
                                        "snapshot_id": statistic_refs[-1].key_digest,
                                        "actual_ranks": [a.actual_rank for a in output] if isinstance(output, tuple) else [output.factors.actual_rank],
                                        "numerics": [_numerics(answer) for answer in output] if isinstance(output, tuple) else _numerics(output)})
                        del metric
                    elif node.stage is Stage.BUILD:
                        answers = supplied["approximation"]
                        replacement = (_weighted_build(layer, descriptor, answers, graph.transform.request.representation)
                            if graph.recipe is Recipe.WEIGHTED_SVD else _install(layer, answers))
                        replacement.training = modes[path]
                        output = replacement, path
                        del answers
                    else:
                        _fail("UnsupportedStage", node.key, "Only canonical fixed stages are executable")
                artifact = node.outputs[0]
                values[artifact.key] = output
                trace.append({"node": node.key, "stage": node.stage.value, "executed": True,
                    "cache_hit": hit, "result_ref": artifact.key, "semantic_kind": artifact.kind.value,
                    "tensor_bytes": _tensor_bytes(output), "observe_forward_calls": forwards,
                    "statistic_ref": asdict(stat_ref) if stat_ref is not None else None})
                if phase_callback is not None:
                    phase_callback({"phase": node.stage.value, "scope": list(node.scope), "node": node.key,
                        "graph_fingerprint": graph.fingerprint, "started_at": phase_started,
                        "completed_at": time.perf_counter(), "cache_hit": hit, "state": "completed",
                        "tensor_bytes": trace[-1]["tensor_bytes"],
                        "observe_forward_calls": forwards,
                        "statistic_ref": asdict(stat_ref) if stat_ref is not None else None})
                for port in node.inputs:
                    consumers[port.artifact] -= 1
                    if consumers[port.artifact] == 0:
                        values.pop(port.artifact, None)
                # Avoid retaining previous stage values via local dictionaries.
                del supplied, output
            for path, layer in module_paths(work).items():
                if path in modes:
                    layer.training = modes[path]
            plan = replace(graph.transform, request=replace(graph.transform.request, statistics_refs=tuple(refs)))
            if graph.recipe is Recipe.ASVD:
                legacy = _finish(model, work, graph.method_spec, plan, records, graph.source.fingerprint,
                    graph.calibration.fingerprint, graph.source.graph_version, started, memory.peak_floor,
                    options.max_peak_bytes, options.max_workspace_bytes, options.parameter_budget, options.tensor_byte_budget)
            else:
                before, after = _storage_records(model), _storage_records(work)
                report = {"method": "weighted_svd", "layers": records, "checkpoint_id": graph.source.fingerprint,
                    "data_id": graph.calibration.fingerprint, "parameters_before": exact_unique_cost(before),
                    "parameters_after": exact_unique_cost(after), "calibration_forward_calls": total_forwards,
                    "method_version": 1, "objective": "input_second_moment", "policy": asdict(graph.transform.request.metric_policy),
                    "graph_version": graph.source.graph_version, "tensor_bytes_before": exact_unique_cost(before, unit="bytes"),
                    "tensor_bytes_after": exact_unique_cost(after, unit="bytes"), "optimizer_rebuild_required": True}
                work._fedcore_transform_plan = plan
                legacy = WeightedTransformResult(work, plan, report)
            result = FixedGraphResult(work, graph, legacy, {"graph_fingerprint": graph.fingerprint,
                "recipe": graph.recipe.value, "source_mode": graph.source_mode.value, "executor": graph.executor,
                "stage_trace_kind": "executed_fixed_stages", "actual_stages": trace,
                "observe_forward_calls": total_forwards, "validation_forward_calls": 0,
                "calibration_copy": copy_evidence, "numerical_evidence": legacy.evidence,
                "resources": {"required_floor_bytes": memory.peak_floor, "limit_bytes": options.max_peak_bytes,
                    "scope": "named stage buffers and owned store; arbitrary forward/allocator RSS is not a hard bound"}}, tuple(statistic_refs))
        return result
    except ArtifactStoreError as error:
        raise CompressionGraphError(error.violations) from error
    except (MemoryError, RuntimeError) as error:
        if _allocation_failure(error):
            _fail("ResourceLimitExceeded", "stages", "Stage allocation failed after preflight")
        raise
