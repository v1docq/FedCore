"""Single-owner execution of finite closed-region compression schedules.

Linear profiles execute real observe/solve/apply/advance boundaries. The bounded
ReferenceAttention adapter honestly delegates one complete checked Q/K recipe;
its internal observations are not advertised as separately cached stages.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
import time

import psutil
import torch
from torch import nn

from .artifacts import (ArtifactStoreError, StatisticsKey, StatisticRef,
                        StatisticsSnapshot, StatisticsStore)
from .candidate_storage import (CandidateStorageError, capture_inventory, clone_preserving_storages,
                                inventory_cost, inventory_fingerprint, storage_guard)
from .compression_graph import (ArtifactKind, CalibrationRef, ModelVersion,
                                RuntimeOptions, SourceMode)
from .coupled_specs import FlatQueryKey
from .execution import WeightedProfileError, _fingerprint, _finite_observations
from .grouped_workflow import (ConsumerCoordinates, CoordinateAxis, CompressionRegion,
    GroupRequest, GroupPlan, GroupedWorkflow, GroupedWorkflowError, ReferenceCoordinates,
    fail, grouped_fingerprint, region_closure, validate_grouped_workflow, validate_request)
from .method_execution import (_basis_metric, _collect_rows, _explicit_encoder_graph,
    _matrix, _moment, _numerics, _rng_isolated, _sequential_order, plan_method)
from .method_specs import BasisSharing, DRONE, SVDLLMV1, method_name
from .plans import ValidationFailure, Violation
from .statistics import StatisticsPassport
from .topology import inspect_topology, module_paths, replace_modules_atomically


@dataclass(frozen=True)
class GroupedResult:
    model: nn.Module
    workflow: GroupedWorkflow
    evidence: dict
    statistic_refs: tuple[StatisticRef, ...] = ()


def _checked(value):
    if type(value) is ValidationFailure:
        raise GroupedWorkflowError(value.violations)
    return value


def _identity(model, calibration, version):
    try:
        inventory = capture_inventory(model)
        legacy, data = _fingerprint(model, calibration, version)
    except CandidateStorageError as error:
        raise GroupedWorkflowError(error.violations) from error
    except (WeightedProfileError, ValueError, OverflowError) as error:
        fail("InvalidSource", "source", str(error))
    fingerprint = hashlib.sha256((legacy + inventory_fingerprint(inventory)).encode()).hexdigest()
    return ModelVersion(version, fingerprint), data, inventory


def _data(calibration, options):
    if (not isinstance(calibration, torch.Tensor) or calibration.device.type != "cpu"
            or calibration.dtype not in (torch.float32, torch.float64) or calibration.ndim < 2
            or not calibration.numel() or not len(calibration)):
        fail("InvalidCalibration", "calibration", "Separate nonempty CPU float32/float64 observations are required")
    if (type(options.max_workspace_bytes) is int and options.max_workspace_bytes > 0
            and calibration[0].numel() * calibration.element_size() > options.max_workspace_bytes):
        fail("ResourceLimitExceeded", "calibration", "One calibration sample exceeds workspace")
    if not _finite_observations(calibration):
        fail("InvalidCalibration", "calibration", "Calibration observations must be finite")


def _reference_coordinates(model, request):
    from .coupled_layers import ReferenceAttention
    paths = module_paths(model)
    records = []
    for path in request.target_paths:
        layer = paths[path]
        if type(layer) is not ReferenceAttention:
            fail("UnsupportedCapability", path, "FlatQueryKey requires exact bounded ReferenceAttention modules")
        records.append(ReferenceCoordinates(path, tuple(layer.head_map.tolist()),
            tuple(tuple(group) for group in layer.group_coordinates.tolist()), layer.head_dim,
            layer.rope, layer.rope_base, layer.logits_scale, layer.d_model))
    return tuple(records)


def prepare_grouped_workflow(model, calibration, requests, *,
        source_mode=SourceMode.CURRENT_STUDENT, batch_size=16,
        max_workspace_bytes=256 * 1024**2, max_peak_bytes=12 * 1024**3,
        graph_version="grouped-v1", parameter_budget=None, tensor_byte_budget=None):
    """Read source metadata only: no model copy, forward, hooks or solvers.

    ``parameter_budget`` uses the G2 physical stored-element unit, including
    buffer storage and unused backing tails. Parameter numel is reported
    separately. Generic coordinate spaces are explicit declared assumptions.
    External masks, training/recovery executors, loops and arbitrary callbacks
    are deliberately outside this finite first contract.
    """
    if not isinstance(model, nn.Module):
        fail("UnsupportedCapability", "model", "A CPU torch Module is required")
    if type(requests) is not tuple or not requests:
        fail("InvalidSchedule", "requests", "An immutable nonempty tuple of GroupRequest records is required")
    if type(source_mode) is not SourceMode:
        fail("InvalidSourceMode", "source_mode", "Use SourceMode.FROZEN_SNAPSHOT or SourceMode.CURRENT_STUDENT")
    if type(graph_version) is not str or not graph_version:
        fail("InvalidSource", "graph_version", "An explicit graph version is required")
    options = RuntimeOptions(batch_size, max_workspace_bytes, max_peak_bytes,
                             parameter_budget, tensor_byte_budget)
    _data(calibration, options)
    # Validate request metadata and the complete physical write set before
    # calling legacy planners. This gives closure/conflict errors precedence.
    from .grouped_workflow import MAX_GROUPS, _inside
    if len(requests) > MAX_GROUPS:
        fail("InvalidSchedule", "requests", f"At most {MAX_GROUPS} finite actions are supported")
    for request in requests:
        validate_request(request)
    source, data, inventory = _identity(model, calibration, graph_version)
    aliases = inspect_topology(model).module_aliases
    paths = module_paths(model)
    prior = []
    groups = []
    for request in requests:
        for path in request.target_paths:
            if path not in paths:
                fail("UnsupportedCapability", path, "Unknown region consumer")
        writes = region_closure(request.region, inventory, aliases)
        for other, other_writes in prior:
            if request.key == other.key or request.region.key == other.region.key:
                fail("RepeatedGroup", request.key, "Each region and action has one visit and one owner")
            if (set(writes) & set(other_writes)
                    or any(_inside(a, (b,)) or _inside(b, (a,))
                           for a in request.region.consumers for b in other.region.consumers)):
                fail("ConflictingWrites", request.key, "Write regions must be disjoint, including aliases and physical storage")
        prior.append((request, writes))
        reference = type(request.method_spec) is FlatQueryKey
        if not reference and any(type(paths[path]) is not nn.Linear for path in request.target_paths):
            fail("UnsupportedCapability", request.key, "The staged first contract accepts exact dense Linear targets")
        try:
            transform = plan_method(model, request.method_spec, rank=request.rank,
                target_paths=request.target_paths, max_peak_bytes=max_peak_bytes)
        except (ValueError, TypeError) as error:
            fail("UnsupportedCapability", request.key, str(error))
        atlas = _reference_coordinates(model, request) if reference else ()
        if reference:
            for binding in request.region.coordinates:
                layer = paths[binding.path]
                if (binding.input != binding.output or binding.input.indices != tuple(range(layer.d_model))):
                    fail("CoordinateMismatch", binding.path, "The bounded Q/K recipe preserves the exact external residual axis")
        groups.append(GroupPlan(request, transform,
            "reference_attention_qk_whole_recipe_v1" if reference else "declared_linear_stages_v1",
            atlas, tuple(record.storage_id for record in inventory), writes, request.region.consumers))
    workflow = GroupedWorkflow(source, CalibrationRef(data, tuple(calibration.shape),
        str(calibration.dtype).removeprefix("torch.")), source_mode, options, tuple(groups), inventory, aliases, "")
    workflow = replace(workflow, fingerprint=grouped_fingerprint(workflow))
    return _checked(validate_grouped_workflow(workflow))


def _observe_pure(model, calibration, version, operation):
    from .graph_execution import CompressionGraphError
    from .staged_execution import _pure_observe
    try:
        return _pure_observe(model, calibration, version, operation)
    except CompressionGraphError as error:
        raise GroupedWorkflowError(error.violations) from error


def _guard(options, extra, *, workspace=True):
    if workspace and extra > options.max_workspace_bytes:
        fail("ResourceLimitExceeded", "workspace", "Named live observation/solve buffers exceed workspace")
    storage_guard(extra, options.max_peak_bytes)


def _statistic(model, calibration, version, workflow, path, transaction, *, need_rows, occupied_bytes=0):
    """Observe one real snapshot; rows are always ephemeral and never cached."""
    from .staged_execution import _rng_fingerprint
    options = workflow.options
    current, data, _ = _identity(model, calibration, version)
    paths = module_paths(model)
    layer = paths[path]
    aliases = tuple(name for name, module in paths.items() if module is layer)
    key = StatisticsKey(current.fingerprint, version, path, aliases, data,
        ArtifactKind.INPUT_SECOND_MOMENT, str(layer.weight.dtype).removeprefix("torch."),
        options.batch_size, _rng_fingerprint(), source_fingerprint=workflow.source.fingerprint)
    snapshot = transaction.lookup(key)
    hit = snapshot is not None
    if not hit:
        transaction.reserve(key, layer.in_features ** 2 * 8)
    rows, forwards = None, 0
    if need_rows or not hit:
        reserve = psutil.Process(os.getpid()).memory_info().rss + 64 * 1024**2
        rows = _observe_pure(model, calibration, version, lambda: _collect_rows(
            model, layer, calibration, batch_size=options.batch_size,
            workspace=options.max_workspace_bytes - occupied_bytes,
            peak=options.max_peak_bytes, reserve=reserve + occupied_bytes,
            single_call_per_batch=need_rows))
        forwards = math.ceil(len(calibration) / options.batch_size)
        _guard(options, occupied_bytes + 2 * rows.numel() * rows.element_size() + 4 * layer.in_features ** 2 * 8)
        if not hit:
            passport = StatisticsPassport(workflow.source.fingerprint, version, current.fingerprint,
                path + ":input:group=0", data, method_version=key.estimator_version)
            moment = _moment(rows, passport)
            snapshot = StatisticsSnapshot((moment.matrix,), (moment.passport,), key.statistic)
            ref = transaction.put(key, snapshot, max_peak_bytes=options.max_peak_bytes)
            # Solves borrow the owned store snapshot, not the producer's tensor.
            snapshot = transaction.lookup(key)
        else:
            ref = StatisticRef(key.digest, key)
    else:
        ref = StatisticRef(key.digest, key)
    return rows if need_rows else None, snapshot, ref, hit, forwards


@dataclass
class _Observed:
    rows: tuple[torch.Tensor, ...]
    moments: tuple[torch.Tensor, ...]
    initialization_moments: tuple[torch.Tensor, ...]
    refs: tuple[StatisticRef, ...]
    forwards: int
    hits: tuple[bool, ...]


def _observe_group(work, frozen, calibration, workflow, group, version, transaction):
    rows, moments, initial, refs, hits = [], [], [], [], []
    forwards = 0
    spec = group.request.method_spec
    for path in group.request.target_paths:
        values, snapshot, ref, hit, count = _statistic(work, calibration, version, workflow,
            path, transaction, need_rows=type(spec) in (DRONE, SVDLLMV1))
        if values is not None:
            rows.append(values)
        moments.append(snapshot.tensors[0]); refs.append(ref); hits.append(hit); forwards += count
        if type(spec) is SVDLLMV1:
            initial_version = f"{workflow.source.graph_version}:grouped={workflow.fingerprint}:initialization=M0"
            _, initial_snapshot, initial_ref, initial_hit, initial_count = _statistic(frozen,
                calibration, initial_version, workflow, path, transaction, need_rows=False,
                occupied_bytes=sum(value.numel() * value.element_size() for value in rows))
            initial.append(initial_snapshot.tensors[0]); refs.append(initial_ref)
            hits.append(initial_hit); forwards += initial_count
    return _Observed(tuple(rows), tuple(moments), tuple(initial), tuple(refs), forwards, tuple(hits))


def _solve_group(work, group, observed, options):
    from .approximation import solve_weighted
    from .structured_profiles import (basis_sharing_factors, balanced_factors,
                                      drone_factors, svdllm_v1_fit)
    from .structured_layers import create_shared_basis_consumers, FactorizedLinear
    request, replacements = group.request, {}
    spec = request.method_spec
    layers = tuple(module_paths(work)[path] for path in request.target_paths)
    if type(spec) is BasisSharing:
        metric, scales = _basis_metric(spec, observed.moments)
        factors = basis_sharing_factors(tuple(_matrix(layer) for layer in layers), metric,
            request.rank, metric_scales=scales)
        consumers = create_shared_basis_consumers(factors, tuple(layer.bias for layer in layers))
        details = {**dict(factors.diagnostics), "metric_mode": spec.metric_mode,
                   "coordinate_evidence": "declared_assumption", "group_closed_against": "declared_consumers_and_full_physical_inventory"}
        for path, layer, consumer in zip(request.target_paths, layers, consumers):
            consumer.left.requires_grad_(layer.weight.requires_grad)
            if consumer.bias is not None:
                consumer.bias.requires_grad_(layer.bias.requires_grad)
            replacements[path] = consumer
        consumers[0].right.requires_grad_(any(layer.weight.requires_grad for layer in layers))
        if any(consumer.right is not consumers[0].right for consumer in consumers):
            fail("InvalidSharedBasis", request.key, "Every consumer must physically own the same right Parameter")
    else:
        layer, = layers
        if type(spec) is DRONE:
            pair = drone_factors(_matrix(layer), observed.rows[0].T, request.rank, rcond=spec.rcond)
            details = {**_numerics(pair), "loss_growth_tolerance": spec.loss_growth_tolerance,
                "tolerance_scope": "explicit local policy; no independence guarantee"}
        else:
            initial = balanced_factors(solve_weighted(_matrix(layer), observed.initialization_moments[0], request.rank))
            pair = svdllm_v1_fit(_matrix(layer), initial.right, observed.rows[0],
                                  rcond=spec.rcond, ridge=spec.ridge)
            details = {**_numerics(pair), "initialization_input": "original_reference_graph",
                       "fit_input": "current_graph_after_predecessor_replacements"}
        replacement = FactorizedLinear.from_factors(pair, layer.bias)
        for parameter in (replacement.left.weight, replacement.right.weight):
            parameter.requires_grad_(layer.weight.requires_grad)
        if replacement.left.bias is not None:
            replacement.left.bias.requires_grad_(layer.bias.requires_grad)
        replacements[request.target_paths[0]] = replacement
    for path, replacement in replacements.items():
        replacement.train(layers[request.target_paths.index(path)].training)
        if any(not bool(torch.isfinite(value).all()) for value in replacement.parameters()):
            fail("NumericalFailure", path, "The prepared replacement must be finite")
    return replacements, details


def _scratch(group):
    shapes = [step.operator.shape for step in group.transform.steps]
    # Basis sharing solves one vertically stacked matrix, not independent SVDs.
    if type(group.request.method_spec) is BasisSharing:
        m, n = sum(shape[0] for shape in shapes), shapes[0][1]
    else:
        m, n = max(shapes, key=lambda shape: shape[0] * shape[1])
    return 8 * (16 * n * n + 16 * m * n + 8 * m * m)


def _budget(model, options):
    inventory = capture_inventory(model)
    tensor_bytes = inventory_cost(inventory, "bytes")
    elements = inventory_cost(inventory, "parameters") if options.parameter_budget is not None else None
    if options.tensor_byte_budget is not None and tensor_bytes > options.tensor_byte_budget:
        fail("BudgetInfeasible", "tensor_byte_budget", "Full physical output storage exceeds the declared byte budget")
    if elements is not None and elements > options.parameter_budget:
        fail("BudgetInfeasible", "parameter_budget", "Full physical stored elements exceed the declared budget")
    return inventory


@_rng_isolated
@torch.no_grad()
def execute_grouped_workflow(model, calibration, workflow, *, store):
    """Run the entire finite job transactionally; publish only after success.

    The supplied StatisticsStore has one active owner for the whole job. It is
    never closed by this function. A failure discards all newly collected
    statistics and every private model change; prior store entries remain.
    """
    _checked(validate_grouped_workflow(workflow))
    if not isinstance(store, StatisticsStore):
        fail("InvalidStore", "store", "An explicit bounded StatisticsStore runtime owner is required")
    _data(calibration, workflow.options)
    current, data, inventory = _identity(model, calibration, workflow.source.graph_version)
    if current != workflow.source or data != workflow.calibration.fingerprint or inventory != workflow.source_inventory:
        fail("StaleSource", "source", "The source, calibration or physical storage provenance has changed")
    prepared = prepare_grouped_workflow(model, calibration,
        tuple(group.request for group in workflow.groups), source_mode=workflow.source_mode,
        batch_size=workflow.options.batch_size, max_workspace_bytes=workflow.options.max_workspace_bytes,
        max_peak_bytes=workflow.options.max_peak_bytes, graph_version=workflow.source.graph_version,
        parameter_budget=workflow.options.parameter_budget, tensor_byte_budget=workflow.options.tensor_byte_budget)
    if prepared != workflow:
        fail("StalePlan", "workflow", "The checked source capabilities or reference coordinate atlas no longer match")
    options, started = workflow.options, time.perf_counter()
    trace, references, records = [], [], []
    try:
        with store._transaction() as transaction:
            for group in workflow.groups:
                _guard(options, _scratch(group))
            # Backing bytes, not view numel, determine all private-copy floors.
            copies = 4 if any(type(group.request.method_spec) is SVDLLMV1 for group in workflow.groups) else 3
            _guard(options, copies * sum(record.nbytes for record in inventory)
                   + calibration.numel() * calibration.element_size(), workspace=False)
            working = calibration.detach().clone()
            work = _explicit_encoder_graph(clone_preserving_storages(model, max_peak_bytes=options.max_peak_bytes))
            modes = {path: layer.training for path, layer in module_paths(work).items()}
            work.eval()
            frozen = (clone_preserving_storages(work, max_peak_bytes=options.max_peak_bytes)
                if any(type(group.request.method_spec) is SVDLLMV1 for group in workflow.groups) else None)
            sequential_paths = tuple(group.request.target_paths[0] for group in workflow.groups
                if type(group.request.method_spec) in (DRONE, SVDLLMV1))
            if sequential_paths:
                version = workflow.source.graph_version + ":order_preflight"
                def check_order():
                    for batch in working.split(options.batch_size):
                        _sequential_order(work, sequential_paths, batch, options.batch_size)
                _observe_pure(work, working, version, check_order)
            accepted = 0
            pending = {}
            live_pending_bytes = 0
            for group in workflow.groups:
                version = f"{workflow.source.graph_version}:grouped={workflow.fingerprint}:accepted={accepted}"
                before, _, before_inventory = _identity(work, working, version)
                common = {"group": group.request.key, "graph_version": version,
                          "model_fingerprint": before.fingerprint,
                          "inventory_fingerprint": inventory_fingerprint(before_inventory)}
                if type(group.request.method_spec) is FlatQueryKey:
                    from .coupled_execution import prepare_coupled_replacements
                    replacements, details = _observe_pure(work, working, version,
                        lambda: prepare_coupled_replacements(work, working,
                            group.request.method_spec, group.transform, batch_size=options.batch_size,
                            max_workspace_bytes=options.max_workspace_bytes))
                    trace.append({**common, "stage": "checked_whole_recipe",
                        "granularity": "reference_attention_qk_whole_recipe_v1", "cacheable_stages": False})
                    details = {"reference_records": details, "coordinate_evidence": "exact_reference_adapter",
                               "external_residual_axis": "preserved", "reference_coordinates": [asdict(item) for item in group.reference_coordinates]}
                    group_refs = ()
                else:
                    observed = _observe_group(work, frozen, working, workflow, group, version, transaction)
                    group_refs = observed.refs
                    references.extend(group_refs)
                    trace.append({**common, "stage": "observe", "forwards": observed.forwards,
                        "statistic_keys": [ref.key_digest for ref in group_refs], "statistic_cache_hits": list(observed.hits),
                        "rows_cached": False})
                    observation_bytes = sum(value.numel() * value.element_size() for value in observed.rows)
                    _guard(options, live_pending_bytes + observation_bytes + _scratch(group))
                    replacements, details = _solve_group(work, group, observed, options)
                    trace.append({**common, "stage": "solve", "method": method_name(group.request.method_spec)})
                    del observed
                records.append({"group": group.request.key, "method": method_name(group.request.method_spec),
                    "source": asdict(before), "rank": group.request.rank, "targets": list(group.request.target_paths),
                    "numerics": details, "statistic_keys": [ref.key_digest for ref in group_refs]})
                if workflow.source_mode is SourceMode.FROZEN_SNAPSHOT:
                    pending.update(replacements)
                    live_pending_bytes = sum(record.nbytes for replacement in pending.values()
                        for record in capture_inventory(replacement))
                    _guard(options, live_pending_bytes)
                else:
                    work = replace_modules_atomically(work, replacements)
                    work.eval()
                    trace.append({**common, "stage": "apply", "writes": list(group.write_paths)})
                    accepted += 1
                    new_version = f"{workflow.source.graph_version}:grouped={workflow.fingerprint}:accepted={accepted}"
                    after, _, after_inventory = _identity(work, working, new_version)
                    trace.append({"group": group.request.key, "stage": "advance", "accepted": accepted,
                        "predecessor": asdict(before), "source": asdict(after),
                        "inventory_fingerprint": inventory_fingerprint(after_inventory)})
            if workflow.source_mode is SourceMode.FROZEN_SNAPSHOT:
                work = replace_modules_atomically(work, pending)
                work.eval()
                accepted = 1
                trace.append({"group": "batch", "stage": "apply", "source_mode": "frozen_snapshot",
                    "writes": [path for group in workflow.groups for path in group.write_paths]})
                new_version = f"{workflow.source.graph_version}:grouped={workflow.fingerprint}:accepted=1"
                after, _, after_inventory = _identity(work, working, new_version)
                trace.append({"group": "batch", "stage": "advance", "accepted": accepted,
                    "source": asdict(after), "inventory_fingerprint": inventory_fingerprint(after_inventory)})
            after_inventory = _budget(work, options)
            # Restore every surviving module mode, including new children below
            # a replaced root whose original mode was explicitly declared.
            for path, layer in module_paths(work).items():
                if path in modes:
                    layer.training = modes[path]
                else:
                    ancestors = [root for root in modes if root == "" or path.startswith(root + ".")]
                    layer.training = modes[max(ancestors, key=len)]
            report = {"workflow_fingerprint": workflow.fingerprint, "source_mode": workflow.source_mode.value,
                "source": asdict(workflow.source), "accepted_versions": accepted,
                "trace": trace, "groups": records, "data_role": "calibration", "recovery": "none",
                "max_visits_per_region": 1, "training_steps": 0,
                "order_preflight_forwards": math.ceil(len(working) / options.batch_size) if sequential_paths else 0,
                "parameters_before": sum(record.parameter_numel for record in workflow.source_inventory),
                "parameters_after": sum(record.parameter_numel for record in after_inventory),
                "tensor_bytes_before": sum(record.nbytes for record in workflow.source_inventory),
                "tensor_bytes_after": sum(record.nbytes for record in after_inventory),
                "storage_inventory_after": [asdict(record) for record in after_inventory],
                "parameter_budget": options.parameter_budget, "tensor_byte_budget": options.tensor_byte_budget,
                "budget_semantics": "full_backing_storage_bytes_or_stored_elements_including_buffers_and_tails",
                "resources": {"max_workspace_bytes": options.max_workspace_bytes, "max_peak_bytes": options.max_peak_bytes,
                    "scope": "named buffers and private storage copies; arbitrary forward RSS is not a hard bound"},
                "optimizer_rebuild_required": True, "total_seconds": time.perf_counter() - started}
            json.dumps(report, allow_nan=False)
            work._fedcore_requires_optimizer_rebuild = True
            work._fedcore_method_evidence = report
            # This metadata participates in the existing checkpoint mechanism.
            work._fedcore_grouped_workflow = workflow
            result = GroupedResult(work, workflow, report, tuple(references))
        return result
    except ArtifactStoreError:
        raise
    except GroupedWorkflowError:
        raise
    except CandidateStorageError as error:
        raise GroupedWorkflowError(error.violations) from error
    except ValueError as error:
        from .approximation import NumericalDomainViolation
        from .method_execution import MethodProfileError
        from .coupled_execution import CoupledProfileError
        # These two exact legacy statistics failures predate the numerical
        # domain exception. Unrelated ValueError remains a programming error.
        legacy_statistics_errors = {"Statistics overflow",
                                    "Observations require finite float32/float64 rows"}
        if isinstance(error, NumericalDomainViolation) or str(error) in legacy_statistics_errors:
            raise GroupedWorkflowError((Violation("NumericalFailure", "solve", str(error)),)) from error
        if isinstance(error, (MethodProfileError, CoupledProfileError)):
            message = str(error)
            code = ("ResourceLimitExceeded" if "ResourceLimitExceeded" in message else
                    "NumericalFailure" if "nonfinite" in message.lower() else "UnsupportedCapability")
            raise GroupedWorkflowError((Violation(code, "workflow", message),)) from error
        raise
    except (MemoryError, RuntimeError) as error:
        from .graph_execution import _allocation_failure
        if _allocation_failure(error):
            raise GroupedWorkflowError((Violation("ResourceLimitExceeded", "workflow",
                "A named CPU allocation failed after preflight"),)) from error
        raise
