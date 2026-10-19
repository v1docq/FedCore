"""Physical CPU tensor storage inventories and ownership for finite candidates.

Byte cost is backing-storage size, not tensor-view numel or pickle file size.
The existing allocator's ``parameters`` unit means all stored elements; mixed
dtype views have no unambiguous element unit and are refused for that budget.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import os

import psutil
import torch
from torch import nn

from .allocation import StorageCost
from .execution import _hash_tensor
from .plans import Violation
from .topology import module_paths


class CandidateStorageError(ValueError):
    def __init__(self, code, path, message):
        self.violations = (Violation(code, path, message),)
        super().__init__(f"{code}: {path}: {message}")


@dataclass(frozen=True)
class TensorView:
    path: str
    category: str
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    offset: int
    persistent: bool
    requires_grad: bool


@dataclass(frozen=True)
class PhysicalStorage:
    storage_id: str
    nbytes: int
    elements: int | None
    views: tuple[TensorView, ...]
    content_digest: str
    parameter_numel: int


def _slots(model):
    for module_path, module in module_paths(model).items():
        def contains_tensor(value, seen):
            if isinstance(value, torch.Tensor):
                return True
            if type(value) in (tuple, list, dict, set) and id(value) not in seen:
                seen.add(id(value))
                return any(contains_tensor(item, seen) for item in ((*value.keys(), *value.values()) if type(value) is dict else value))
            return False
        if any(contains_tensor(value, set()) for name, value in vars(module).items()
               if name not in ("_parameters", "_buffers", "_modules")):
            raise CandidateStorageError("UnsupportedTensorReference", module_path,
                "Tensor references outside registered parameters/buffers are not supported by owned model copying")
        for category, collection in (("parameter", module._parameters), ("buffer", module._buffers)):
            for name, tensor in collection.items():
                if tensor is not None:
                    path = f"{module_path}.{name}" if module_path else name
                    if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                            or tensor.layout != torch.strided or tensor.is_quantized or tensor.is_conj() or tensor.is_neg()):
                        raise CandidateStorageError("UnsupportedStorage", path,
                            "Only ordinary strided CPU tensors without lazy conjugate/negative view bits are supported")
                    yield module, name, tensor, TensorView(path, category, str(tensor.dtype),
                        tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset(),
                        category == "parameter" or name not in module._non_persistent_buffers_set,
                        tensor.requires_grad)


def _raw(storage):
    return torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))


def capture_inventory(model, *, prefix="model"):
    """Immutable full-storage metadata, including nonpersistent buffer data."""
    if not isinstance(model, nn.Module):
        raise CandidateStorageError("UnsupportedStorage", "model", "A torch Module with registered tensor state is required")
    groups = {}
    for _module, _name, tensor, view in _slots(model):
        if tensor.device.type != "cpu" or tensor.layout != torch.strided or tensor.is_conj() or tensor.is_neg():
            raise CandidateStorageError("UnsupportedStorage", view.path, "Only dense strided CPU tensor storage is supported")
        storage = tensor.untyped_storage()
        identity = storage._cdata
        if identity not in groups:
            groups[identity] = (storage, [], {})
        groups[identity][1].append(view)
        if view.category == "parameter":
            groups[identity][2][id(tensor)] = tensor.numel()
    records = []
    for storage, views, parameters in sorted(groups.values(), key=lambda group: min(view.path for view in group[1])):
        views = tuple(sorted(views, key=lambda view: (view.path, view.category)))
        dtype_names = {view.dtype for view in views}
        # dtype width can be recovered without allocating the tensor data.
        widths = {torch.empty((), dtype=getattr(torch, name.removeprefix("torch."))).element_size() for name in dtype_names}
        width = next(iter(widths))
        elements = storage.nbytes() // width if len(dtype_names) == 1 and storage.nbytes() % width == 0 else None
        digest = hashlib.sha256()
        _hash_tensor(digest, _raw(storage))
        label = hashlib.sha256(json.dumps([asdict(view) for view in views], sort_keys=True).encode()).hexdigest()
        records.append(PhysicalStorage(prefix + ":" + label, storage.nbytes(), elements,
                                       views, digest.hexdigest(), sum(parameters.values())))
    return tuple(records)


def inventory_fingerprint(inventory):
    return hashlib.sha256(json.dumps([asdict(record) for record in inventory], sort_keys=True).encode()).hexdigest()


def validate_inventory(inventory):
    """Validate transportable inventory metadata without accessing live tensors."""
    if type(inventory) is not tuple:
        raise CandidateStorageError("InvalidStorage", "inventory", "Immutable storage tuple required")
    identities, paths = set(), set()
    for record in inventory:
        if (type(record) is not PhysicalStorage or type(record.storage_id) is not str or not record.storage_id
                or record.storage_id in identities or type(record.nbytes) is not int or record.nbytes < 0
                or type(record.parameter_numel) is not int or record.parameter_numel < 0
                or record.elements is not None and (type(record.elements) is not int or record.elements < 0)
                or type(record.views) is not tuple or not record.views
                or type(record.content_digest) is not str or len(record.content_digest) != 64
                or any(character not in "0123456789abcdef" for character in record.content_digest)):
            raise CandidateStorageError("InvalidStorage", "inventory", "Typed consistent physical storage metadata required")
        identities.add(record.storage_id)
        dtypes, widths = set(), set()
        for view in record.views:
            if (type(view) is not TensorView or type(view.path) is not str or view.path in paths
                    or view.category not in ("parameter", "buffer") or type(view.dtype) is not str
                    or type(view.shape) is not tuple or type(view.stride) is not tuple
                    or len(view.shape) != len(view.stride) or any(type(x) is not int or x < 0 for x in (*view.shape, *view.stride))
                    or type(view.offset) is not int or view.offset < 0
                    or type(view.persistent) is not bool or type(view.requires_grad) is not bool):
                raise CandidateStorageError("InvalidStorage", "inventory.views", "Typed tensor-view metadata required")
            paths.add(view.path)
            try:
                width = torch.empty(0, dtype=getattr(torch, view.dtype.removeprefix("torch."))).element_size()
            except (AttributeError, TypeError, RuntimeError) as error:
                raise CandidateStorageError("InvalidStorage", view.path, "Unsupported inventory dtype") from error
            if all(view.shape) and (view.offset + sum((size - 1) * stride for size, stride in zip(view.shape, view.stride)) + 1) * width > record.nbytes:
                raise CandidateStorageError("InvalidStorage", view.path, "View exceeds its backing storage")
            dtypes.add(view.dtype)
            widths.add(width)
        width = next(iter(widths))
        expected = record.nbytes // width if len(dtypes) == 1 and record.nbytes % width == 0 else None
        if record.elements != expected:
            raise CandidateStorageError("InvalidStorage", record.storage_id, "Stored-element count must match full backing bytes and dtype")


def inventory_cost(inventory, unit):
    return sum(record.numel * (record.bytes_per_element if unit == "bytes" else 1)
               for record in project_storages(inventory, unit))


def project_storages(inventory, unit):
    validate_inventory(inventory)
    if unit not in ("bytes", "parameters"):
        raise CandidateStorageError("InvalidBudgetUnit", "unit", "Use bytes or parameters (all stored elements)")
    projected = []
    for record in inventory:
        if unit == "parameters" and record.elements is None:
            raise CandidateStorageError("AmbiguousElementCost", record.storage_id, "Mixed dtype backing storage has no single stored-element unit")
        projected.append(StorageCost(record.storage_id, record.nbytes if unit == "bytes" else record.elements,
                                     1, "backing_storage"))
    return tuple(projected)


def storage_guard(extra_bytes, max_peak_bytes):
    if (type(extra_bytes) is not int or extra_bytes < 0 or type(max_peak_bytes) is not int
            or not 0 < max_peak_bytes <= 12 * 1024**3):
        raise CandidateStorageError("InvalidResourceLimit", "candidate.storage", "A nonnegative allocation and positive peak of at most 12 GiB are required")
    floor = psutil.Process(os.getpid()).memory_info().rss + extra_bytes + 64 * 1024**2
    if floor > max_peak_bytes:
        raise CandidateStorageError("ResourceLimitExceeded", "candidate.storage", "Named owned buffers exceed the declared peak floor")
    return floor


@torch.no_grad()
def clone_preserving_storages(model, *, max_peak_bytes):
    """Copy Python shells and restore exact tensor/view/storage relationships."""
    inventory = capture_inventory(model)
    size = sum(record.nbytes for record in inventory)
    # deepcopy temporaries and the final raw backing-storage copies coexist.
    storage_guard(3 * size, max_peak_bytes)
    work = deepcopy(model)
    originals, copies = module_paths(model), module_paths(work)
    storages, tensors = {}, {}
    for path, original in originals.items():
        target = copies[path]
        for category in ("_parameters", "_buffers"):
            for name, tensor in getattr(original, category).items():
                if tensor is None:
                    continue
                if id(tensor) not in tensors:
                    identity = tensor.untyped_storage()._cdata
                    if identity not in storages:
                        storages[identity] = _raw(tensor.untyped_storage()).clone().untyped_storage()
                    view = torch.empty(0, dtype=tensor.dtype).set_(storages[identity], tensor.storage_offset(),
                                                                 tensor.size(), tensor.stride())
                    tensors[id(tensor)] = nn.Parameter(view, requires_grad=tensor.requires_grad) if isinstance(tensor, nn.Parameter) else view.requires_grad_(tensor.requires_grad)
                getattr(target, category)[name] = tensors[id(tensor)]
    return work
