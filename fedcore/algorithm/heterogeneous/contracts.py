"""Immutable contracts for the closed CV compression procedure (not a neural DAG)."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import hashlib
import json
import re


class Representation(str, Enum):
    DENSE = "dense_conv"
    COMPACT = "channel_compact_conv"
    FACTORS = "real_conv_factors"
    QDQ = "int8_qdq_conv"


class Operation(str, Enum):
    PRUNE = "coupled_internal_channel_pruning"
    SVD = "conv2_frobenius_svd"
    PTQ = "post_recovery_current_candidate_ptq"


def payload(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        result = {}
        for descriptor in fields(value):
            item = getattr(value, descriptor.name)
            if item is not None or not descriptor.metadata.get("omit_none"):
                result[descriptor.name] = payload(item)
        return result
    if type(value) in (tuple, list):
        return [payload(item) for item in value]
    if type(value) is dict:
        return {key: payload(item) for key, item in value.items()}
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(payload(value), sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class PTQPlacement:
    """Omit output Q/DQ at explicitly named logical residual convolutions."""

    omit_output_qdq: tuple[str, ...]

    def __post_init__(self):
        paths = self.omit_output_qdq
        if (type(paths) is not tuple or not paths
                or any(type(path) is not str or not re.fullmatch(r"layer[1-4]\.[01]\.conv[12]", path)
                       for path in paths)
                or paths != tuple(sorted(set(paths)))):
            raise ValueError("Use a nonempty sorted tuple of distinct logical Conv module paths")

    @classmethod
    def from_payload(cls, value):
        if (type(value) is not dict or set(value) != {"omit_output_qdq"}
                or type(value["omit_output_qdq"]) is not list):
            raise ValueError("Exact PTQ placement payload required")
        return cls(tuple(value["omit_output_qdq"]))


class BackendContractError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class TensorRepresentation:
    """Conv2 matrix coordinates; factor storage is never a dense sparsity claim."""
    kind: Representation
    shape: tuple[int, int]
    internal_channels: int
    rank: int | None = None
    storage_dtype: str = "float32"
    compute_dtype: str = "float32"
    layout: str = "conv_out_by_input_kernel_NCHW"
    channel_coordinates: str = "original_internal_channels"
    accumulator_dtype: str = "float32"

    def __post_init__(self):
        if (type(self.kind) is not Representation or type(self.shape) is not tuple or len(self.shape) != 2
                or any(type(n) is not int or n < 1 for n in self.shape)
                or type(self.internal_channels) is not int or self.internal_channels < 1
                or self.shape[1] != 9 * self.internal_channels
                or self.layout != "conv_out_by_input_kernel_NCHW"
                or self.compute_dtype != "float32" or self.channel_coordinates not in
                ("original_internal_channels", "frozen_M0_conv1_l1_selected_original_order")):
            raise ValueError("Exact bounded CV matrix coordinates and float32 boundary required")
        if self.rank is not None and (type(self.rank) is not int or not 1 <= self.rank <= min(self.shape)):
            raise ValueError("Factor rank must fit reduced Conv coordinates")
        if self.kind is Representation.FACTORS and self.rank is None:
            raise ValueError("Real factors require an explicit rank")
        if self.kind in (Representation.DENSE, Representation.COMPACT) and self.rank is not None:
            raise ValueError("Dense Conv representations cannot claim retained factors")
        if self.kind is Representation.QDQ:
            if (self.storage_dtype, self.accumulator_dtype) != ("int8", "int32"):
                raise ValueError("QDQ requires explicit int8 storage and int32 accumulation")
        elif (self.storage_dtype, self.accumulator_dtype) != ("float32", "float32"):
            raise ValueError("Float representations require float32 storage/arithmetic")


@dataclass(frozen=True)
class Dependency:
    source_version: str
    calibration_version: str
    coordinates: str
    observation_mode: str = "frozen_snapshot"
    predecessor: str | None = None

    def __post_init__(self):
        if (any(type(v) is not str or not v for v in
                (self.source_version, self.calibration_version, self.coordinates))
                or self.observation_mode not in ("frozen_snapshot", "current_student")):
            raise ValueError("Explicit source, data, coordinates and observation mode required")
        if self.observation_mode == "current_student" and not self.predecessor:
            raise ValueError("Current-candidate observations must bind the recovered predecessor")


@dataclass(frozen=True)
class Step:
    operation: Operation
    incoming: TensorRepresentation
    outgoing: TensorRepresentation
    dependency: Dependency
    ptq_placement: PTQPlacement | None = field(default=None, metadata={"omit_none": True})

    def __post_init__(self):
        if (type(self.operation) is not Operation or type(self.incoming) is not TensorRepresentation
                or type(self.outgoing) is not TensorRepresentation or type(self.dependency) is not Dependency):
            raise ValueError("Exact typed operation, tensor boundaries and dependencies required")
        if self.ptq_placement is not None and (self.operation is not Operation.PTQ
                                               or type(self.ptq_placement) is not PTQPlacement):
            raise ValueError("An explicit placement policy applies only to PTQ")
        a, b = self.incoming, self.outgoing
        if self.operation is Operation.PRUNE:
            if (a.kind is not Representation.DENSE or b.kind is not Representation.COMPACT
                    or b.shape[0] != a.shape[0] or b.internal_channels >= a.internal_channels
                    or b.rank is not None or b.channel_coordinates != "frozen_M0_conv1_l1_selected_original_order"
                    or self.dependency.observation_mode != "frozen_snapshot"):
                raise ValueError("Pruning must reduce linked internal coordinates from frozen dense M0")
        elif self.operation is Operation.SVD:
            if (a.kind not in (Representation.DENSE, Representation.COMPACT) or b.kind is not Representation.FACTORS
                    or a.shape != b.shape or a.channel_coordinates != b.channel_coordinates
                    or b.rank * sum(b.shape) >= b.shape[0] * b.shape[1]
                    or self.dependency.observation_mode != "frozen_snapshot"):
                raise ValueError("SVD must preserve coordinates and reduce real factor storage")
        elif (a.kind is Representation.QDQ or b.kind is not Representation.QDQ or a.shape != b.shape
              or a.rank != b.rank or a.channel_coordinates != b.channel_coordinates
              or self.dependency.observation_mode != "current_student"):
            raise ValueError("PTQ binds the current recovered candidate without changing coordinates")


@dataclass(frozen=True)
class Sequence:
    steps: tuple[Step, ...]

    def __post_init__(self):
        if type(self.steps) is not tuple or any(type(s) is not Step for s in self.steps):
            raise ValueError("Immutable typed steps required")
        for previous, current in zip(self.steps, self.steps[1:]):
            if previous.outgoing != current.incoming:
                raise ValueError("Ordered steps have incompatible representation boundaries")


@dataclass(frozen=True)
class Scope:
    region: str
    recipe: Sequence
    source_version: str
    read_closure: tuple[str, ...]
    write_closure: tuple[str, ...]
    external_channels: tuple[int, int]

    def __post_init__(self):
        if (type(self.region) is not str or not self.region or type(self.recipe) is not Sequence
                or type(self.source_version) is not str or not self.source_version
                or type(self.external_channels) is not tuple or len(self.external_channels) != 2
                or any(type(n) is not int or n < 1 for n in self.external_channels)):
            raise ValueError("Closed typed scope with a fixed external channel interface required")
        for closure in (self.read_closure, self.write_closure):
            if (type(closure) is not tuple or not closure or closure != tuple(sorted(set(closure)))
                    or any(type(path) is not str or not path for path in closure)):
                raise ValueError("Canonical complete read/write closures required")
        if any(s.dependency.source_version != self.source_version for s in self.recipe.steps):
            raise ValueError("Scope steps must retain their frozen source version")


def overlaps(first, second):
    return first == second or first.startswith(second + ".") or second.startswith(first + ".")


@dataclass(frozen=True)
class ParallelDisjoint:
    branches: tuple[Scope, ...]
    source_version: str

    def __post_init__(self):
        if (type(self.branches) is not tuple or not self.branches
                or any(type(b) is not Scope or b.source_version != self.source_version for b in self.branches)
                or tuple(b.region for b in self.branches) != tuple(sorted(set(b.region for b in self.branches)))):
            raise ValueError("Canonical disjoint branches must read the same immutable source")
        # Writes are candidate-local replacements; shared reads of frozen M0 are safe.
        for i, branch in enumerate(self.branches):
            for other in self.branches[i + 1:]:
                if any(overlaps(a, b) for a in branch.write_closure for b in other.write_closure):
                    raise ValueError("Parallel replacement scopes overlap or alias")


@dataclass(frozen=True)
class JoinApplyReplacements:
    parallel: ParallelDisjoint
    output_version: str

    def __post_init__(self):
        if type(self.parallel) is not ParallelDisjoint or type(self.output_version) is not str or not self.output_version:
            raise ValueError("A checked replacement join and explicit output version are required")
