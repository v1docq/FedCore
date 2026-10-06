"""Frozen experiment decisions and independently identified data roles.

This package does not replace the public FedCore DataRoles contract. It adds
sample-level checks to the reusable research runner before effects begin.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import InitVar, dataclass, field, fields
from types import MappingProxyType
from typing import Mapping

import torch
from torch import nn


ROLES = ("train", "validation", "calibration", "test")
TASKS = ("classification", "regression", "forecasting", "language_model")


class ProtocolError(ValueError):
    """Invalid research configuration; no training should have started."""


def json_value(value):
    """Convert only data values; never execute or stringify arbitrary objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolError("Nonfinite JSON values are forbidden")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ProtocolError("Configuration keys must be strings")
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    raise ProtocolError(f"Not a JSON data value: {type(value).__name__}")


def canonical_json(value):
    return json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_hash(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def role_content_identity(roles):
    """Ignore transport location while retaining IDs, shape and tensor hashes."""
    return {role: {key: value for key, value in info.items() if key != 'storage'}
            for role, info in roles.items()}


def freeze_mapping(value):
    def freeze(item):
        if isinstance(item, dict):
            return MappingProxyType({key: freeze(val) for key, val in item.items()})
        if isinstance(item, list):
            return tuple(freeze(val) for val in item)
        return item
    return freeze(json_value(value))


def tensor_hash(value):
    tensor = value.detach()
    descriptor = canonical_json({"shape": list(tensor.shape), "dtype": str(tensor.dtype)})
    digest = hashlib.sha256(descriptor.encode("utf-8"))
    # Splitting the sample axis preserves the old contiguous byte order without
    # an entire file-backed tensor's temporary byte string or CPU copy.
    rows = max(1, (8 * 1024 * 1024) // max(1, tensor[0].numel() * tensor.element_size())) if tensor.ndim and len(tensor) else 1
    batches = tensor.split(rows) if tensor.ndim else (tensor,)
    for batch in batches:
        digest.update(batch.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _finite_tensor(value):
    rows = max(1, (8 * 1024 * 1024) // max(1, value[0].numel() * value.element_size()))
    return all(bool(torch.isfinite(batch).all()) for batch in value.split(rows))


@dataclass(frozen=True)
class TensorSplit:
    x: torch.Tensor
    y: torch.Tensor
    ids: tuple[str, ...]
    unit_ids: tuple[str, ...] = ()
    intervals: tuple[tuple[int, int], ...] = ()
    _hashes: tuple[str, str] = field(init=False, repr=False)
    _backing: Mapping = field(default_factory=dict, init=False, repr=False, compare=False)
    _files: InitVar[Mapping | None] = field(default=None, kw_only=True)

    def __post_init__(self, _files):
        if _files:
            import numpy as np
            from .measurement import file_hash
            if set(_files) != {"x", "y"}:
                raise ProtocolError("File backing requires x/y NPY descriptors")
            values = []
            for axis in ("x", "y"):
                info = _files[axis]
                if set(info) != {"path", "sha256"} or file_hash(info["path"]) != info["sha256"]:
                    raise ProtocolError("File-backed split hash mismatch")
                array = np.load(info["path"], mmap_mode="c", allow_pickle=False)
                if not isinstance(array, np.memmap):
                    raise ProtocolError("File-backed splits require NPY arrays")
                values.append(torch.from_numpy(array))
            object.__setattr__(self, "x", values[0])
            object.__setattr__(self, "y", values[1])
            object.__setattr__(self, "_backing", freeze_mapping(_files))
        if not isinstance(self.x, torch.Tensor) or not isinstance(self.y, torch.Tensor):
            raise ProtocolError("Split x/y must be torch tensors")
        if self.x.ndim == 0 or self.y.ndim == 0 or len(self.x) == 0 or len(self.x) != len(self.y):
            raise ProtocolError("Nonempty aligned sample axes are required")
        ids = tuple(self.ids)
        if len(ids) != len(self.x) or any(not isinstance(item, str) or not item for item in ids):
            raise ProtocolError("One explicit nonempty string ID per sample is required")
        if len(set(ids)) != len(ids):
            raise ProtocolError("Duplicate sample IDs within a role")
        units = tuple(self.unit_ids)
        if units and (len(units) != len(ids) or any(not isinstance(item, str) or not item for item in units)):
            raise ProtocolError("One nonempty independent unit ID per sample is required")
        intervals = tuple(tuple(pair) for pair in self.intervals)
        if intervals and (len(intervals) != len(ids) or any(len(pair) != 2 or
                any(type(end) is not int for end in pair) or pair[0] > pair[1] for pair in intervals)):
            raise ProtocolError("Intervals must be aligned inclusive integer start/end pairs")
        x, y = (self.x, self.y) if self._backing else (self.x.detach().cpu().clone(), self.y.detach().cpu().clone())
        if not _finite_tensor(x) or not _finite_tensor(y):
            raise ProtocolError("Split tensors must be finite after train-only preprocessing")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "unit_ids", units)
        object.__setattr__(self, "intervals", intervals)
        object.__setattr__(self, "_hashes", (tensor_hash(x), tensor_hash(y)))

    def verify_integrity(self):
        if self._backing:
            from .measurement import file_hash
            if any(file_hash(info["path"]) != info["sha256"] for info in self._backing.values()):
                raise ProtocolError("File-backed split hash mismatch")
        if (tensor_hash(self.x), tensor_hash(self.y)) != self._hashes:
            raise ProtocolError("A frozen split tensor was modified in place")

    def manifest(self):
        self.verify_integrity()
        result = {"ids": list(self.ids), "unit_ids": list(self.unit_ids),
                "intervals": [list(pair) for pair in self.intervals],
                "x_sha256": self._hashes[0], "y_sha256": self._hashes[1],
                "x_shape": list(self.x.shape), "y_shape": list(self.y.shape)}
        if self._backing:
            result["storage"] = {"kind": "npy_copy_on_write", "files": json_value(self._backing),
                                 "scope": "bounded loading copies; OS page cache and resident pages are not capped"}
        return result

    @classmethod
    def from_npy(cls, x_path, y_path, ids, unit_ids=(), intervals=(), *, expected_hashes=None):
        """Load owned copy-on-write mappings, with safe data-only NPY replay."""
        from pathlib import Path
        from .measurement import file_hash
        files = {axis: {"path": str(Path(path).resolve()), "sha256": file_hash(path)}
                 for axis, path in (("x", x_path), ("y", y_path))}
        if expected_hashes is not None and tuple(files[axis]["sha256"] for axis in ("x", "y")) != tuple(expected_hashes):
            raise ProtocolError("File-backed split hash mismatch")
        return cls(torch.empty(0), torch.empty(0), tuple(ids), tuple(unit_ids), tuple(intervals), _files=files)


@dataclass(frozen=True)
class ExperimentBundle:
    train: TensorSplit
    validation: TensorSplit
    calibration: TensorSplit
    test: TensorSplit
    task: str
    original_model: nn.Module
    metadata: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if self.task not in TASKS or not isinstance(self.original_model, nn.Module):
            raise ProtocolError("An explicit supported task and torch model are required")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        validate_roles(self)

    def manifest(self):
        validate_roles(self)
        return {"task": self.task, "metadata": json_value(self.metadata),
                "roles": {role: getattr(self, role).manifest() for role in ROLES}}


def validate_roles(bundle):
    """Check independent objects and inclusive raw input+target windows."""
    for role in ROLES:
        split = getattr(bundle, role)
        if not isinstance(split, TensorSplit):
            raise ProtocolError(f"{role} must be TensorSplit")
        split.verify_integrity()
    for index, left_role in enumerate(ROLES):
        left = getattr(bundle, left_role)
        for right_role in ROLES[index + 1:]:
            right = getattr(bundle, right_role)
            if set(left.ids) & set(right.ids):
                raise ProtocolError(f"Sample IDs overlap: {left_role}/{right_role}")
            if left.unit_ids and right.unit_ids and set(left.unit_ids) & set(right.unit_ids):
                raise ProtocolError(f"Independent units overlap: {left_role}/{right_role}")
            if left.intervals and right.intervals:
                # Sorted sweep, including target horizon and feature history.
                a, b = sorted(left.intervals), sorted(right.intervals)
                i = j = 0
                while i < len(a) and j < len(b):
                    if a[i][0] <= b[j][1] and b[j][0] <= a[i][1]:
                        raise ProtocolError(f"Raw time windows overlap: {left_role}/{right_role}")
                    if a[i][1] < b[j][0]:
                        i += 1
                    else:
                        j += 1
    if any(getattr(bundle, role).unit_ids for role in ROLES) and not all(getattr(bundle, role).unit_ids for role in ROLES):
        raise ProtocolError("Independent unit IDs must cover all roles")
    if any(getattr(bundle, role).intervals for role in ROLES) and not all(getattr(bundle, role).intervals for role in ROLES):
        raise ProtocolError("Raw time windows must cover all roles")
    fit_ids = bundle.metadata.get("preprocessing_fit_ids")
    if fit_ids is not None and set(fit_ids) != set(bundle.train.ids):
        raise ProtocolError("Preprocessing fit IDs must equal the train role")


def split_indices(ids, *, seed=0, fractions=(0.6, 0.15, 0.1, 0.15), unit_ids=None):
    """Seeded independent-unit partition; repeated clients stay together."""
    ids = tuple(ids)
    if len(ids) != len(set(ids)) or len(ids) < 4 or len(fractions) != 4 or any(v <= 0 for v in fractions) or not math.isclose(sum(fractions), 1):
        raise ProtocolError("Four positive fractions summing to one and unique IDs are required")
    units = tuple(unit_ids) if unit_ids is not None else ids
    if len(units) != len(ids):
        raise ProtocolError("Independent units and IDs must be aligned")
    unique = sorted(set(units))
    if len(unique) < 4:
        raise ProtocolError("At least four independent units are required")
    order = torch.randperm(len(unique), generator=torch.Generator().manual_seed(seed)).tolist()
    counts = [max(1, int(len(unique) * fraction)) for fraction in fractions[:3]]
    while sum(counts) >= len(unique):
        index = max(range(3), key=lambda i: counts[i])
        counts[index] -= 1
    counts.append(len(unique) - sum(counts))
    result, offset = {}, 0
    for role, count in zip(ROLES, counts):
        chosen = {unique[index] for index in order[offset:offset + count]}
        result[role] = tuple(index for index, unit in enumerate(units) if unit in chosen)
        offset += count
    return result


@dataclass(frozen=True)
class CandidateSpec:
    method: str
    parameters: Mapping = field(default_factory=dict)
    chain: tuple["CandidateSpec", ...] = ()

    def __post_init__(self):
        if not isinstance(self.method, str) or not self.method:
            raise ProtocolError("Candidate method must be a nonempty string")
        object.__setattr__(self, "parameters", freeze_mapping(self.parameters))
        object.__setattr__(self, "chain", tuple(self.chain))
        if any(not isinstance(step, CandidateSpec) or step.chain for step in self.chain):
            raise ProtocolError("Chain steps must be unnested CandidateSpec objects")
        if self.chain and self.method != "chain":
            raise ProtocolError("A nonempty chain requires method='chain'")
        if self.method == "chain" and not self.chain:
            raise ProtocolError("An empty compression chain is not an operation")

    def to_dict(self):
        return {"method": self.method, "parameters": json_value(self.parameters),
                "chain": [step.to_dict() for step in self.chain]}

    @property
    def candidate_id(self):
        return stable_hash(self.to_dict())[:20]

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - {"method", "parameters", "chain"} or "method" not in value:
            raise ProtocolError("Invalid candidate schema")
        return cls(value["method"], value.get("parameters", {}),
                   tuple(cls.from_dict(step) for step in value.get("chain", ())))


@dataclass(frozen=True)
class ExperimentProtocol:
    seed: int = 0
    batch_size: int = 16
    baseline_epochs: int = 2
    finetune_epochs: int = 1
    learning_rate: float = 0.01
    quality_tolerance: float = 0.05
    device: str = "cpu"
    measurement_repeats: int = 10
    warmup: int = 2
    threads: int = 1
    artifact_format: str = "torchscript"
    selection_cost: str = "file_bytes"
    quality_scale: float = 1.0
    cost_scale: float = 1048576.0
    hypervolume_reference: tuple[float, float] = (2.0, 2.0)
    repeat_stage: str = "pilot"
    repeat_seeds: tuple[int, ...] = (0,)
    repeat_rationale: str = "Local pilot; confirmatory repeat count requires pilot variance."
    primary_comparison: str = "compression versus independently trained baseline"
    minimum_baseline_quality: float | None = None

    def __post_init__(self):
        for key in ("seed", "batch_size", "baseline_epochs", "finetune_epochs", "measurement_repeats", "warmup", "threads"):
            value = getattr(self, key)
            if type(value) is not int or value < (1 if key in ("batch_size", "measurement_repeats", "threads") else 0):
                raise ProtocolError(f"Invalid nonnegative integer {key}")
        for key in ("learning_rate", "quality_tolerance", "quality_scale", "cost_scale"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (key != "quality_tolerance" and value == 0):
                raise ProtocolError(f"Invalid finite numeric {key}")
        if self.selection_cost not in ("file_bytes", "latency_p50_ms") or self.artifact_format not in ("torchscript", "onnx"):
            raise ProtocolError("Unsupported selection metric or executable artifact format")
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ProtocolError("Only explicit CPU/CUDA profiles are supported")
        reference = tuple(self.hypervolume_reference)
        if len(reference) != 2 or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in reference):
            raise ProtocolError("A finite fixed 2D hypervolume reference is required")
        if self.repeat_stage not in ("pilot", "confirmatory") or not self.repeat_rationale.strip() or not self.primary_comparison.strip():
            raise ProtocolError("Declare repeat stage, rationale and primary comparison before running")
        seeds = tuple(self.repeat_seeds)
        if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ProtocolError("Repeat seeds must be unique nonnegative integers")
        if self.minimum_baseline_quality is not None and (isinstance(self.minimum_baseline_quality, bool) or
                not isinstance(self.minimum_baseline_quality, (int, float)) or not math.isfinite(self.minimum_baseline_quality)):
            raise ProtocolError("Baseline quality threshold must be finite")
        object.__setattr__(self, "repeat_seeds", seeds)
        object.__setattr__(self, "hypervolume_reference", reference)

    def to_dict(self):
        return {item.name: json_value(getattr(self, item.name)) for item in fields(self)}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - {item.name for item in fields(cls)}:
            raise ProtocolError("Unknown experiment protocol fields")
        return cls(**value)
