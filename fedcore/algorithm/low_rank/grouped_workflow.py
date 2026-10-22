"""Closed compression regions and finite schedules, containing metadata only.

Generic Linear coordinates are explicit caller assumptions. They describe the
declared consumer boundary; they do not prove dataflow in arbitrary Python.
Reference-attention coordinates instead come from the exact bounded adapter.
This contract is independent of the frozen-candidate G3 search grammar.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math

from .candidate_storage import PhysicalStorage, validate_inventory
from .compression_graph import (CalibrationRef, ModelVersion, RuntimeOptions,
                                SourceMode, _canonical)
from .coupled_specs import FlatQueryKey
from .method_specs import BasisSharing, DRONE, SVDLLMV1
from .plans import (FixedRank, Goal, TransformPlan, ValidationFailure, Violation,
                    _metadata_only)


MAX_GROUPS = 64


@dataclass(frozen=True)
class CoordinateAxis:
    space: str
    indices: tuple[int, ...]


@dataclass(frozen=True)
class ConsumerCoordinates:
    path: str
    input: CoordinateAxis
    output: CoordinateAxis


@dataclass(frozen=True)
class CompressionRegion:
    key: str
    consumers: tuple[str, ...]
    coordinates: tuple[ConsumerCoordinates, ...]


@dataclass(frozen=True)
class GroupRequest:
    key: str
    region: CompressionRegion
    target_paths: tuple[str, ...]
    method_spec: BasisSharing | DRONE | SVDLLMV1 | FlatQueryKey
    rank: int


@dataclass(frozen=True)
class ReferenceCoordinates:
    path: str
    head_map: tuple[int, ...]
    group_coordinates: tuple[tuple[int, ...], ...]
    head_dim: int
    rope: bool
    rope_base: float
    logits_scale: float
    residual_width: int


@dataclass(frozen=True)
class GroupPlan:
    request: GroupRequest
    transform: TransformPlan
    adapter: str
    reference_coordinates: tuple[ReferenceCoordinates, ...]
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    write_paths: tuple[str, ...]


@dataclass(frozen=True)
class GroupedWorkflow:
    source: ModelVersion
    calibration: CalibrationRef
    source_mode: SourceMode
    options: RuntimeOptions
    groups: tuple[GroupPlan, ...]
    source_inventory: tuple[PhysicalStorage, ...]
    module_aliases: tuple[tuple[str, ...], ...]
    fingerprint: str
    schema_version: int = 1


class GroupedWorkflowError(ValueError):
    def __init__(self, violations):
        self.violations = tuple(violations)
        super().__init__("; ".join(f"{v.code}: {v.path}: {v.message}" for v in self.violations))


def fail(code, path, message):
    raise GroupedWorkflowError((Violation(code, path, message),))


def grouped_fingerprint(workflow):
    """Content identity, not authorization or a substitute for validation."""
    if type(workflow) is not GroupedWorkflow or not _metadata_only(workflow):
        fail("LiveResource", "workflow", "A metadata-only GroupedWorkflow is required")
    return hashlib.sha256(json.dumps(_canonical(replace(workflow, fingerprint="")),
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _paths(values, *, nonempty=True):
    return (type(values) is tuple and (bool(values) or not nonempty)
            and all(type(path) is str for path in values) and len(set(values)) == len(values))


def _inside(path, roots):
    return any(root == "" or path == root or path.startswith(root + ".") for root in roots)


def region_closure(region, inventory, module_aliases):
    """Derive physical write ownership and refuse partly covered aliases."""
    roots = region.consumers
    for aliases in module_aliases:
        selected = tuple(path for path in aliases if _inside(path, roots))
        if selected and len(selected) != len(aliases):
            fail("IncompleteGroup", region.key, "Every module alias must belong to the declared complete region")
    writes = []
    for storage in inventory:
        selected = tuple(view for view in storage.views if _inside(view.path, roots))
        if selected:
            if len(selected) != len(storage.views):
                fail("IncompleteGroup", region.key, "Every consumer of affected physical storage must belong to the region")
            # Replacements cannot silently destroy a shared view relationship.
            # Shared module aliases have the same canonical Tensor object.
            owners = {view.tensor_alias for view in selected}
            if len(owners) > 1:
                fail("UnsupportedStorage", region.key, "A replaced region cannot contain distinct views of one backing storage")
            writes.append(storage.storage_id)
    return tuple(writes)


def validate_request(request):
    if type(request) is not GroupRequest or not _metadata_only(request):
        fail("InvalidGroup", "request", "A frozen metadata-only GroupRequest is required")
    if type(request.key) is not str or not request.key:
        fail("InvalidGroup", "request.key", "A nonempty group identifier is required")
    region = request.region
    if (type(region) is not CompressionRegion or type(region.key) is not str or not region.key
            or not _paths(region.consumers) or not _paths(request.target_paths)):
        fail("InvalidRegion", request.key, "Declare a nonempty immutable complete consumer set and selected targets")
    if set(region.consumers) != set(request.target_paths):
        fail("IncompleteGroup", request.key, "Targets must cover the complete declared region")
    if region.consumers != request.target_paths:
        fail("CoordinateMismatch", request.key, "Preserve declared consumer order, including proportional metric scale associations")
    if any(a != b and _inside(b, (a,)) for a in region.consumers for b in region.consumers):
        fail("ConflictingWrites", request.key, "Region consumers cannot be ancestor/descendant modules")
    if (type(region.coordinates) is not tuple or len(region.coordinates) != len(region.consumers)
            or any(type(item) is not ConsumerCoordinates for item in region.coordinates)
            or tuple(item.path for item in region.coordinates) != region.consumers):
        fail("InvalidCoordinates", request.key, "One ordered coordinate binding is required for every declared consumer")
    for binding in region.coordinates:
        for axis in (binding.input, binding.output):
            if (type(axis) is not CoordinateAxis or type(axis.space) is not str or not axis.space
                    or type(axis.indices) is not tuple or not axis.indices
                    or any(type(i) is not int or i < 0 for i in axis.indices)
                    or len(set(axis.indices)) != len(axis.indices)):
                fail("InvalidCoordinates", binding.path, "Name each coordinate space and its ordered distinct nonnegative indices")
    if type(request.rank) is not int or request.rank < 1:
        fail("InvalidRank", request.key, "An explicit positive integer rank is required")
    if type(request.method_spec) not in (BasisSharing, DRONE, SVDLLMV1, FlatQueryKey):
        fail("UnsupportedMethod", request.key, "Only BasisSharing, DRONE, SVDLLMV1 and bounded FlatQueryKey are supported")
    # Frozen dataclasses can still be manually forged. Re-run their contracts.
    try:
        replace(request.method_spec)
    except (TypeError, ValueError) as error:
        fail("InvalidMethod", request.key, str(error))
    if type(request.method_spec) is BasisSharing:
        if len(region.consumers) < 2:
            fail("IncompleteGroup", request.key, "BasisSharing needs at least two distinct declared consumers")
        if any(binding.input != region.coordinates[0].input for binding in region.coordinates[1:]):
            fail("CoordinateMismatch", request.key, "Shared input width alone is insufficient: ordered input coordinate spaces must agree")
        scales = request.method_spec.proportional_scales
        if scales and len(scales) != len(region.consumers):
            fail("InvalidMethod", request.key, "One proportional metric scale is required per consumer")
    elif type(request.method_spec) in (DRONE, SVDLLMV1) and len(region.consumers) != 1:
        fail("InvalidGroup", request.key, "Each sequential Linear action replaces one complete module")
    return request


def _validate(workflow):
    if type(workflow) is not GroupedWorkflow or not _metadata_only(workflow):
        fail("LiveResource", "workflow", "Only frozen metadata may enter a grouped workflow")
    if type(workflow.schema_version) is not int or workflow.schema_version != 1:
        fail("InvalidWorkflow", "schema_version", "Unsupported grouped workflow version")
    if (type(workflow.source) is not ModelVersion or type(workflow.source.graph_version) is not str
            or not workflow.source.graph_version or not _digest(workflow.source.fingerprint)):
        fail("InvalidSource", "source", "An explicit source version and SHA-256 fingerprint are required")
    c = workflow.calibration
    if (type(c) is not CalibrationRef or not _digest(c.fingerprint) or c.role != "calibration"
            or type(c.shape) is not tuple or len(c.shape) < 2
            or any(type(d) is not int or d < 1 for d in c.shape) or c.dtype not in ("float32", "float64")):
        fail("InvalidCalibration", "calibration", "Separate finite CPU calibration metadata is required")
    if type(workflow.source_mode) is not SourceMode:
        fail("InvalidSourceMode", "source_mode", "Declare frozen_snapshot or current_student")
    options = workflow.options
    if type(options) is not RuntimeOptions:
        fail("InvalidOptions", "options", "RuntimeOptions required")
    for name in ("batch_size", "max_workspace_bytes", "max_peak_bytes"):
        if type(getattr(options, name)) is not int or getattr(options, name) < 1:
            fail("InvalidOptions", name, "A positive integer bound is required")
    if options.max_peak_bytes > 12 * 1024**3 or options.max_workspace_bytes > options.max_peak_bytes:
        fail("InvalidOptions", "options", "Workspace must fit a CPU peak of at most 12 GiB")
    for name in ("parameter_budget", "tensor_byte_budget"):
        value = getattr(options, name)
        if value is not None and (type(value) is not int or value < 1):
            fail("InvalidOptions", name, "An optional budget must be a positive integer")
    try:
        validate_inventory(workflow.source_inventory)
    except ValueError as error:
        fail("InvalidStorage", "source_inventory", str(error))
    if (type(workflow.module_aliases) is not tuple
            or any(not _paths(aliases) or len(aliases) < 2 for aliases in workflow.module_aliases)):
        fail("InvalidAliases", "module_aliases", "Immutable complete module alias groups are required")
    if type(workflow.groups) is not tuple or not 1 <= len(workflow.groups) <= MAX_GROUPS:
        fail("InvalidSchedule", "groups", f"Declare between one and {MAX_GROUPS} finite actions without loops or recovery")
    keys, regions, previous = set(), set(), []
    reads = tuple(storage.storage_id for storage in workflow.source_inventory)
    for group in workflow.groups:
        if type(group) is not GroupPlan:
            fail("InvalidGroup", "groups", "Typed GroupPlan records are required")
        request = validate_request(group.request)
        if request.key in keys or request.region.key in regions:
            fail("RepeatedGroup", request.key, "A region and action have one owner and one visit per workflow")
        keys.add(request.key); regions.add(request.region.key)
        if type(request.method_spec) in (DRONE, SVDLLMV1) and workflow.source_mode is not SourceMode.CURRENT_STUDENT:
            fail("InvalidSourceMode", request.key, "Sequential reconstruction requires actual current-student observations")
        expected_adapter = "reference_attention_qk_whole_recipe_v1" if type(request.method_spec) is FlatQueryKey else "declared_linear_stages_v1"
        if (group.adapter != expected_adapter or type(group.transform) is not TransformPlan
                or type(group.transform.steps) is not tuple
                or tuple(step.operator.path for step in group.transform.steps) != request.target_paths
                or any(step.rank != request.rank for step in group.transform.steps)):
            fail("InvalidTransform", request.key, "The checked transform must match the declared method, targets and rank")
        transform = group.transform
        objective = {BasisSharing: "shared_input_basis", DRONE: "drone_current_input_thin_support",
                     SVDLLMV1: "svdllm_v1_current_input_ls", FlatQueryKey: getattr(request.method_spec, "variant", "")}[type(request.method_spec)]
        goal = Goal.COUPLED_GRAPH_TRANSFORM if type(request.method_spec) is FlatQueryKey else Goal.REPLACE_OPERATOR
        if (transform.schema_version != 1 or transform.request.goal is not goal
                or transform.request.objective != objective or transform.request.statistics_refs != ()
                or transform.resource_limits.max_peak_bytes != options.max_peak_bytes
                or any(type(step.structure) is not FixedRank or step.structure.rank != request.rank
                       or step.requires_statistics is not True for step in transform.steps)):
            fail("InvalidTransform", request.key, "The transform objective, resources and stage requirements must match the group contract")
        bindings = {binding.path: binding for binding in request.region.coordinates}
        for step in group.transform.steps:
            binding = bindings[step.operator.path]
            outputs, inputs = step.operator.shape
            if type(request.method_spec) is FlatQueryKey:
                inputs = outputs  # external residual input/output width
            elif (step.operator.kind != "Linear" or step.operator.dtype not in ("float32", "float64")
                    or step.operator.groups != 1):
                fail("InvalidTransform", step.operator.path, "The staged first contract supports dense floating Linear operators")
            if len(binding.input.indices) != inputs or len(binding.output.indices) != outputs:
                fail("CoordinateMismatch", step.operator.path, "Coordinate cardinalities must match the operator's actual external axes")
        if type(request.method_spec) is BasisSharing:
            if request.rank > min(group.transform.steps[0].operator.shape[1],
                                  sum(step.operator.shape[0] for step in group.transform.steps)):
                fail("InvalidRank", request.key, "Shared rank exceeds the stacked matrix dimensions")
        elif type(request.method_spec) is FlatQueryKey:
            if any(request.rank > step.operator.shape[1] for step in group.transform.steps):
                fail("InvalidRank", request.key, "Q/K rank exceeds the current physical head width")
        elif any(request.rank > min(step.operator.shape) for step in group.transform.steps):
            fail("InvalidRank", request.key, "Rank exceeds the declared current operator dimensions")
        if expected_adapter.startswith("reference"):
            if (type(group.reference_coordinates) is not tuple
                    or tuple(item.path for item in group.reference_coordinates) != request.target_paths
                    or any(type(item) is not ReferenceCoordinates for item in group.reference_coordinates)):
                fail("InvalidCoordinates", request.key, "Actual reference attention coordinate metadata is required")
            for item in group.reference_coordinates:
                if (type(item.head_map) is not tuple or not item.head_map
                        or type(item.group_coordinates) is not tuple or not item.group_coordinates
                        or type(item.head_dim) is not int or item.head_dim < 1
                        or type(item.rope) is not bool or type(item.residual_width) is not int or item.residual_width < 1
                        or type(item.rope_base) not in (int, float) or not math.isfinite(item.rope_base) or item.rope_base <= 1
                        or type(item.logits_scale) not in (int, float) or not math.isfinite(item.logits_scale) or item.logits_scale <= 0):
                    fail("InvalidCoordinates", item.path, "Invalid bounded attention atlas")
                binding = bindings[item.path]
                step = next(step for step in transform.steps if step.operator.path == item.path)
                if (binding.input != binding.output or binding.input.indices != tuple(range(item.residual_width))
                        or item.residual_width != step.operator.shape[0] or step.operator.kind != "reference_attention"
                        or any(len(coordinates) != step.operator.shape[1] for coordinates in item.group_coordinates)
                        or item.rope and (item.head_dim % 2 or request.rank % 2)):
                    fail("CoordinateMismatch", item.path, "The Q/K adapter preserves the declared external residual coordinates and complete RoPE pairs")
                count = len(item.group_coordinates)
                if (len(item.head_map) % count or any(type(g) is not int or not 0 <= g < count for g in item.head_map)
                        or any(item.head_map.count(g) != len(item.head_map) // count for g in range(count))):
                    fail("CoordinateMismatch", item.path, "GQA head consumers must completely cover each shared KV group")
                for coordinates in item.group_coordinates:
                    if (type(coordinates) is not tuple or not coordinates
                            or any(type(i) is not int or not 0 <= i < item.head_dim for i in coordinates)
                            or len(set(coordinates)) != len(coordinates)
                            or item.rope and (len(coordinates) % 2 or any(coordinates[i] % 2 or coordinates[i + 1] != coordinates[i] + 1 for i in range(0, len(coordinates), 2)))):
                        fail("CoordinateMismatch", item.path, "The declared Q/K atlas must retain complete ordered RoPE pairs")
        elif group.reference_coordinates != ():
            fail("InvalidCoordinates", request.key, "Generic Linear groups have only the declared coordinate assumption")
        writes = region_closure(request.region, workflow.source_inventory, workflow.module_aliases)
        if group.writes != writes or group.reads != reads or group.write_paths != request.region.consumers:
            fail("InvalidOwnership", request.key, "Read/write ownership must match the complete source storage inventory")
        for earlier in previous:
            overlap = set(group.writes) & set(earlier.writes)
            nested = any(_inside(a, (b,)) or _inside(b, (a,)) for a in group.write_paths for b in earlier.write_paths)
            if overlap or nested:
                fail("ConflictingWrites", request.key, "Each physical write region has exactly one action owner; shared reads are allowed")
        previous.append(group)
    if not _digest(workflow.fingerprint) or workflow.fingerprint != grouped_fingerprint(workflow):
        fail("InvalidFingerprint", "fingerprint", "The workflow content digest does not match")
    return workflow


def validate_grouped_workflow(workflow):
    """Pure validation. Invalid plans return stable structured violations."""
    try:
        return _validate(workflow)
    except GroupedWorkflowError as error:
        return ValidationFailure(error.violations)
    except (TypeError, AttributeError, ValueError, OverflowError) as error:
        return ValidationFailure((Violation("InvalidWorkflow", "workflow", str(error)),))
