"""Deterministic descriptions of low-rank work; no live runtime resources.

The numerical solver remains the existing tdecomp boundary. These records
describe objectives and structures, rather than registering another solver API.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import math
from typing import Mapping


class Goal(str, Enum):
    REPLACE_OPERATOR = "replace_operator"
    ADD_RESIDUAL = "add_residual"
    COUPLED_GRAPH_TRANSFORM = "coupled_graph_transform"
    DIAGNOSE_INTERVENTION = "diagnose_intervention"
    BUILD_ARCHITECTURE = "build_architecture"


@dataclass(frozen=True)
class MetricPolicy:
    ridge: float = 0.0
    rcond: float | None = None
    nullspace_policy: str = "support_only"


@dataclass(frozen=True)
class FixedRank:
    rank: int


@dataclass(frozen=True)
class RankFraction:
    fraction: float


@dataclass(frozen=True)
class ParameterFraction:
    fraction: float


@dataclass(frozen=True)
class ExplicitMask:
    indices: tuple[int, ...]


@dataclass(frozen=True)
class GraphWidth:
    width: int


@dataclass(frozen=True)
class SharedBudget:
    budget: int
    unit: str = "parameters"


@dataclass(frozen=True)
class LegacyThreshold:
    """A legacy policy on the actual operator spectrum, never trained S."""
    strategy: str
    threshold: float


StructureChoice = FixedRank | RankFraction | ParameterFraction | ExplicitMask | GraphWidth | SharedBudget | LegacyThreshold


@dataclass(frozen=True)
class SnapshotRef:
    snapshot_id: str
    checkpoint_id: str = ""
    graph_version: str = ""
    method_version: str = "1"


@dataclass(frozen=True)
class OperatorDescriptor:
    path: str
    shape: tuple[int, ...]
    kind: str
    dtype: str = "float32"
    groups: int = 1
    kernel_size: tuple[int, ...] = ()
    stride: tuple[int, ...] = ()
    padding: tuple[int, ...] | str = ()
    dilation: tuple[int, ...] = ()
    bias_elements: int = 0
    storage_id: str = ""
    checkpoint_id: str = ""
    predecessor_version: str = ""
    padding_mode: str = "zeros"


@dataclass(frozen=True)
class ResourceLimits:
    max_peak_bytes: int | None = None


@dataclass(frozen=True)
class TransformRequest:
    goal: Goal
    objective: str
    solver: str
    structure: StructureChoice
    representation: str = "three_layers"
    decomposing_mode: str | bool | None = "channel"
    metric_policy: MetricPolicy = MetricPolicy()
    statistics_refs: tuple[SnapshotRef, ...] = ()


@dataclass(frozen=True)
class Violation:
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class ValidationFailure:
    violations: tuple[Violation, ...]

    def __bool__(self):
        return False


@dataclass(frozen=True)
class StepSpec:
    operator: OperatorDescriptor
    rank: int | None
    structure: StructureChoice
    requires_statistics: bool
    requires_operator_spectrum: bool = False


@dataclass(frozen=True)
class TransformPlan:
    request: TransformRequest
    steps: tuple[StepSpec, ...]
    resource_limits: ResourceLimits
    schema_version: int = 1


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _positive_integer(value):
    return type(value) is int and value > 0


def _metadata_only(value):
    if value is None or type(value) in (str, bool, int, float) or isinstance(value, Enum):
        return True
    if isinstance(value, tuple):
        return all(_metadata_only(item) for item in value)
    if is_dataclass(value) and getattr(type(value), "__dataclass_params__").frozen:
        return all(_metadata_only(getattr(value, field.name)) for field in fields(value))
    return False


def matrix_shape(operator, decomposing_mode="channel"):
    """Shape of one group, following the existing channel/spatial convention."""
    if operator.kind in ("Linear", "Embedding"):
        return operator.shape
    out, inp, *kernel = operator.shape
    if decomposing_mode == "spatial" and len(kernel) == 2:
        return out // operator.groups * kernel[0], inp * kernel[1]
    return out // operator.groups, inp * math.prod(kernel)


def plan_transform(request, operators, limits=ResourceLimits()):
    """Accumulate independent violations before producing immutable steps.

    Future goals/masks/widths can be described, but are not executable in P0/P1.
    Parameter fractions count the requested three-factor representation and bias.
    Shared budgets intentionally remain unresolved until finite candidate tables
    (with the exact representation cost) reach the allocation module.
    """
    violations = []
    def fail(code, path, message):
        violations.append(Violation(code, path, message))
    if not isinstance(request, TransformRequest):
        return ValidationFailure((Violation("InvalidRequest", "request", "TransformRequest required"),))
    if not _metadata_only(request):
        # Do not invoke equality/arithmetic on arbitrary runtime objects (a
        # tensor can have effectful or non-scalar comparison semantics).
        return ValidationFailure((Violation("LiveResource", "request", "Plan descriptions require immutable metadata and snapshot references"),))
    if not _metadata_only(limits):
        fail("LiveResource", "limits", "Resource limits require immutable metadata")
    if not isinstance(limits, ResourceLimits):
        fail("InvalidResourceLimit", "limits", "ResourceLimits required")
        limits = ResourceLimits()
    if not isinstance(request.goal, Goal):
        fail("InvalidGoal", "goal", "A declared transformation goal is required")
    elif request.goal is not Goal.REPLACE_OPERATOR:
        fail("UnsupportedProfile", "goal", "This stage executes only operator replacement")
    if request.objective not in ("frobenius", "input_second_moment"):
        fail("UnsupportedObjective", "objective", "Only Frobenius and input-second-moment objectives are executable")
    if not isinstance(request.solver, str) or not request.solver:
        fail("InvalidSolver", "solver", "Use the existing named solver boundary")
    if request.representation not in ("one_layer", "two_layers", "three_layers"):
        fail("InvalidRepresentation", "representation", "Unknown final representation")
    if not isinstance(request.statistics_refs, tuple) or any(not isinstance(ref, SnapshotRef) for ref in request.statistics_refs):
        fail("InvalidSnapshotRef", "statistics_refs", "Use immutable snapshot references")
    elif any(not isinstance(ref.snapshot_id, str) or not ref.snapshot_id or
             any(not isinstance(getattr(ref, name), str) for name in ("checkpoint_id", "graph_version", "method_version"))
             for ref in request.statistics_refs):
        fail("InvalidSnapshotRef", "statistics_refs", "Snapshot references require stable string identifiers")
    if limits.max_peak_bytes is not None and not _positive_integer(limits.max_peak_bytes):
        fail("InvalidResourceLimit", "max_peak_bytes", "Memory limit must be a positive integer")
    policy = request.metric_policy
    if not isinstance(policy, MetricPolicy):
        fail("InvalidMetricPolicy", "metric_policy", "MetricPolicy required")
    else:
        if not _number(policy.ridge) or policy.ridge < 0:
            fail("InvalidRidge", "metric_policy.ridge", "Ridge must be finite and nonnegative")
        if policy.rcond is not None and (not _number(policy.rcond) or policy.rcond < 0):
            fail("InvalidCutoff", "metric_policy.rcond", "Cutoff must be finite and nonnegative")
        if policy.nullspace_policy not in ("support_only", "preserve_nullspace"):
            fail("InvalidNullspacePolicy", "metric_policy.nullspace_policy", "Explicit null-space policy required")
    structure = request.structure
    if isinstance(structure, FixedRank):
        if not _positive_integer(structure.rank):
            fail("InvalidRank", "structure.rank", "Rank must be a positive integer")
    elif isinstance(structure, (RankFraction, ParameterFraction)):
        if not _number(structure.fraction) or not 0 < structure.fraction <= 1:
            fail("InvalidFraction", "structure.fraction", "Fraction must lie in (0, 1]")
    elif isinstance(structure, SharedBudget):
        if type(structure.budget) is not int or structure.budget < 0:
            fail("InvalidBudget", "structure.budget", "Budget must be a nonnegative integer")
        if structure.unit not in ("parameters", "bytes"):
            fail("InvalidBudgetUnit", "structure.unit", "Name parameter or byte cost explicitly")
    elif isinstance(structure, ExplicitMask):
        fail("UnsupportedProfile", "structure", "Explicit-mask execution is a later profile")
        if not isinstance(structure.indices, tuple) or not structure.indices or any(type(i) is not int or i < 0 for i in structure.indices):
            fail("InvalidMask", "structure.indices", "Mask requires distinct nonnegative integer indices")
        elif len(set(structure.indices)) != len(structure.indices):
            fail("InvalidMask", "structure.indices", "Mask requires distinct nonnegative integer indices")
    elif isinstance(structure, GraphWidth):
        fail("UnsupportedProfile", "structure", "Graph-width execution is a later profile")
        if not _positive_integer(structure.width):
            fail("InvalidWidth", "structure.width", "Width must be a positive integer")
    elif isinstance(structure, LegacyThreshold):
        if structure.strategy not in ("energy", "explained_variance", "absolute_sum", "quantile"):
            fail("InvalidThresholdStrategy", "structure.strategy", "Unknown historical spectrum policy")
        if not _number(structure.threshold) or not 0 < structure.threshold <= 1:
            fail("InvalidThreshold", "structure.threshold", "Threshold must lie in (0, 1]")
    else:
        fail("InvalidStructure", "structure", "Rank, parameter fraction, mask, width and budget are distinct choices")
    if not isinstance(operators, (tuple, list)):
        return ValidationFailure(tuple(violations) + (Violation("InvalidOperator", "operators", "A finite sequence of operator descriptors is required"),))
    operators = tuple(operators)
    paths = [op.path for op in operators if isinstance(op, OperatorDescriptor) and isinstance(op.path, str)]
    if len(set(paths)) != len(paths):
        fail("TopologyConflict", "operators", "Operator paths must be unique")
    steps = []
    for operator in operators:
        if not isinstance(operator, OperatorDescriptor) or not _metadata_only(operator):
            fail("InvalidOperator", "operators", "Immutable OperatorDescriptor required")
            continue
        if not isinstance(operator.path, str):
            fail("InvalidOperatorPath", "operators", "Operator path must be a string (empty denotes the root)")
            continue
        if any(not isinstance(getattr(operator, name), str) for name in ("dtype", "storage_id", "checkpoint_id", "predecessor_version", "padding_mode")):
            fail("InvalidOperator", operator.path, "Operator provenance and dtype require string identifiers")
        if operator.kind not in ("Linear", "Embedding", "Conv1d", "Conv2d"):
            fail("UnsupportedProfile", operator.path, "Unknown operator kind")
            continue
        shape_valid = (isinstance(operator.shape, tuple) and len(operator.shape) >= 2 and all(_positive_integer(d) for d in operator.shape))
        groups_valid = _positive_integer(operator.groups) and shape_valid and operator.shape[0] % operator.groups == 0
        if not shape_valid:
            fail("InvalidShape", operator.path, "Operator axes must be positive integers")
        if not groups_valid:
            fail("InvalidGroups", operator.path, "Groups must divide the output channels")
        if type(operator.bias_elements) is not int or operator.bias_elements < 0:
            fail("InvalidBiasCost", operator.path, "Bias element count must be nonnegative")
        if operator.kind in ("Linear", "Embedding") and (not shape_valid or len(operator.shape) != 2 or operator.groups != 1):
            fail("InvalidShape", operator.path, "Linear and embedding require an ungrouped matrix")
            shape_valid = False
        if operator.kind in ("Conv1d", "Conv2d"):
            dimensions = 1 if operator.kind == "Conv1d" else 2
            if not shape_valid or len(operator.shape) != dimensions + 2:
                fail("InvalidShape", operator.path, "Convolution weight axes do not match its kind")
                shape_valid = False
            for name in ("kernel_size", "stride", "dilation"):
                values = getattr(operator, name)
                if not isinstance(values, tuple) or len(values) != dimensions or any(not _positive_integer(v) for v in values):
                    fail("InvalidConvolution", operator.path + "." + name, "Explicit positive convolution dimensions required")
            if shape_valid and operator.kernel_size and tuple(operator.shape[2:]) != operator.kernel_size:
                fail("InvalidConvolution", operator.path, "Kernel metadata must agree with the weight shape")
            if not isinstance(operator.padding, str) and (not isinstance(operator.padding, tuple) or len(operator.padding) != dimensions or any(type(p) is not int or p < 0 for p in operator.padding)):
                fail("InvalidConvolution", operator.path + ".padding", "Padding must be explicit and nonnegative")
        weighted = request.objective == "input_second_moment"
        if weighted:
            if operator.kind not in ("Linear", "Conv1d", "Conv2d") or operator.dtype not in ("float32", "float64"):
                fail("UnsupportedProfile", operator.path, "Weighted profile supports Linear/Conv1d/Conv2d float32/float64")
            if request.decomposing_mode not in (True, "channel") or isinstance(operator.padding, str):
                fail("UnsupportedProfile", operator.path, "Weighted convolution requires channel mode and numeric padding")
            if request.solver != "svd":
                fail("UnsupportedProfile", "solver", "Weighted profile requires the checked SVD solve")
        rank = None
        if shape_valid and groups_valid:
            rows, cols = matrix_shape(operator, request.decomposing_mode)
            maximum = min(rows, cols)
            if isinstance(structure, FixedRank) and _positive_integer(structure.rank):
                rank = structure.rank
            elif isinstance(structure, RankFraction) and _number(structure.fraction) and 0 < structure.fraction <= 1:
                rank = max(1, math.floor(maximum * structure.fraction))
            elif isinstance(structure, ParameterFraction) and _number(structure.fraction) and 0 < structure.fraction <= 1:
                dense = math.prod(operator.shape) + operator.bias_elements
                per_rank = operator.groups * (rows + cols + (request.representation == "three_layers"))
                rank = min(maximum, math.floor((math.floor(dense * structure.fraction) - operator.bias_elements) / per_rank))
                if request.representation == "one_layer":
                    fail("IncompatibleStructure", operator.path, "A dense representation cannot realize a factor parameter fraction")
            if rank is not None and not 1 <= rank <= maximum:
                fail("InvalidRank", operator.path, "Requested rank does not fit the operator or parameter budget")
            if isinstance(structure, ExplicitMask) and isinstance(structure.indices, tuple) and any(type(i) is int and i >= maximum for i in structure.indices):
                fail("InvalidMask", operator.path, "Mask exceeds the operator spectrum")
        steps.append(StepSpec(operator, rank, structure, weighted, isinstance(structure, LegacyThreshold)))
    return ValidationFailure(tuple(violations)) if violations else TransformPlan(request, tuple(steps), limits)


def legacy_request(*, rank=None, solver="svd", decomposing_mode=True,
                   representation="three_layers", strategy=None, threshold=None):
    """Decode the historical rank/threshold boundary without changing semantics."""
    if strategy is not None or threshold is not None:
        if rank is not None:
            return ValidationFailure((Violation("AmbiguousStructure", "rank", "Rank and threshold are alternative policies"),))
        structure = LegacyThreshold(strategy, threshold)
    elif rank is None:
        structure = RankFraction(1.0)
    elif type(rank) is int:
        structure = FixedRank(rank)
    elif type(rank) is float:
        structure = RankFraction(rank)
    else:
        return ValidationFailure((Violation("InvalidRank", "rank", "Use an integer rank or explicit rank fraction"),))
    return TransformRequest(Goal.REPLACE_OPERATOR, "frobenius", solver, structure,
                            representation, decomposing_mode)
