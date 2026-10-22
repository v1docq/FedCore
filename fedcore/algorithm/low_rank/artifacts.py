"""Bounded, process-local owned statistics; references contain metadata only.

The dictionary has O(1) expected key lookup and no eviction. Capacity refusal
is explicit. Every admitted tensor receives its own dense detached storage;
the byte count therefore equals unique owned tensor bytes, including pending
transaction entries. Public reads clone again. No metric/model is cached.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import os

import psutil
import torch

from .compression_graph import ArtifactKind
from .plans import Violation, _metadata_only
from .statistics import StatisticsPassport


class ArtifactStoreError(ValueError):
    def __init__(self, code, path, message):
        self.violations = (Violation(code, path, message),)
        super().__init__(f"{code}: {path}: {message}")


@dataclass(frozen=True)
class StatisticsKey:
    model_fingerprint: str
    graph_version: str
    path: str
    aliases: tuple[str, ...]
    calibration_fingerprint: str
    statistic: ArtifactKind
    input_dtype: str
    batch_size: int
    rng_fingerprint: str
    source_fingerprint: str = ""
    mask: str = "none"
    weights: str = "none"
    normalization: str = "weight_sum"
    coordinates: str = "original_operator_input_output"
    accumulator_dtype: str = "float64"
    estimator_version: str = "fixed_cpu_v1"
    data_role: str = "calibration"
    model_mode: str = "eval"

    @property
    def digest(self):
        if (type(self) is not StatisticsKey or not _metadata_only(self)
                or any(type(getattr(self, field.name)) is not str for field in fields(self)
                       if field.name not in ("batch_size", "aliases", "statistic"))
                or type(self.batch_size) is not int or self.batch_size <= 0
                or type(self.aliases) is not tuple or any(type(path) is not str for path in self.aliases)
                or self.path not in self.aliases or len(set(self.aliases)) != len(self.aliases)
                or type(self.statistic) is not ArtifactKind
                or self.statistic not in (ArtifactKind.INPUT_SECOND_MOMENT, ArtifactKind.INPUT_ABS_MEAN, ArtifactKind.INPUT_ABS_MAX)
                or self.mask != "none" or self.weights != "none" or self.data_role != "calibration"
                or self.model_mode != "eval" or self.normalization != "weight_sum"
                or self.accumulator_dtype != "float64" or self.input_dtype not in ("float32", "float64")
                or self.coordinates != "original_operator_input_output"):
            raise ArtifactStoreError("InvalidProvenance", "key", "Immutable StatisticsKey required")
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class StatisticRef:
    key_digest: str
    provenance: StatisticsKey


@dataclass(frozen=True)
class StatisticsSnapshot:
    tensors: tuple[torch.Tensor, ...]
    passports: tuple[StatisticsPassport, ...]
    kind: ArtifactKind


def _bytes(snapshot):
    if (type(snapshot) is not StatisticsSnapshot or type(snapshot.tensors) is not tuple
            or not snapshot.tensors or type(snapshot.passports) is not tuple
            or len(snapshot.tensors) != len(snapshot.passports)
            or type(snapshot.kind) is not ArtifactKind
            or any(type(p) is not StatisticsPassport or not _metadata_only(p) for p in snapshot.passports)
            or any(not isinstance(t, torch.Tensor) or t.device.type != "cpu" or t.dtype != torch.float64
                   or t.layout != torch.strided or not t.numel() for t in snapshot.tensors)):
        raise ArtifactStoreError("InvalidSnapshot", "snapshot", "Finite CPU float64 statistics and immutable passports required")
    return sum(t.numel() * t.element_size() for t in snapshot.tensors)


def _copy(snapshot, max_peak_bytes):
    size = _bytes(snapshot)
    if type(max_peak_bytes) is not int or max_peak_bytes <= 0:
        raise ArtifactStoreError("InvalidResourceLimit", "snapshot.copy", "Positive byte limit required")
    floor = psutil.Process(os.getpid()).memory_info().rss + size + 64 * 1024**2
    if floor > max_peak_bytes:
        raise ArtifactStoreError("ResourceLimitExceeded", "snapshot.copy", "Snapshot copy exceeds the declared named-buffer floor")
    try:
        # The finite-check bool temporary uses <= one eighth of the reserved
        # copy bytes and is released before cloning, after admission/preflight.
        if any(not bool(torch.isfinite(t).all()) for t in snapshot.tensors):
            raise ArtifactStoreError("InvalidSnapshot", "snapshot", "Statistics must contain finite values")
        if snapshot.kind in (ArtifactKind.INPUT_ABS_MEAN, ArtifactKind.INPUT_ABS_MAX) and any(bool((t < 0).any()) for t in snapshot.tensors):
            raise ArtifactStoreError("InvalidSnapshot", "snapshot", "Absolute statistics must be nonnegative")
        return StatisticsSnapshot(tuple(t.detach().clone(memory_format=torch.contiguous_format) for t in snapshot.tensors),
                                  snapshot.passports, snapshot.kind)
    except (MemoryError, RuntimeError) as error:
        message = str(error).lower()
        allocation = isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)) or (
            "defaultcpuallocator" in message and any(text in message for text in (
                "not enough memory", "can't allocate memory", "cannot allocate memory")))
        if allocation:
            raise ArtifactStoreError("ResourceLimitExceeded", "snapshot.copy", "Snapshot allocation failed after preflight") from error
        raise


def _validate_snapshot(key, snapshot):
    key.digest
    size = _bytes(snapshot)
    if snapshot.kind is not key.statistic:
        raise ArtifactStoreError("InvalidSnapshot", "snapshot.kind", "Statistic meaning must match its provenance")
    if snapshot.kind is not ArtifactKind.INPUT_SECOND_MOMENT and len(snapshot.tensors) != 1:
        raise ArtifactStoreError("InvalidSnapshot", "snapshot", "Absolute statistics have one feature vector")
    for index, (tensor, passport) in enumerate(zip(snapshot.tensors, snapshot.passports)):
        point = key.path + (f":input:group={index}" if snapshot.kind is ArtifactKind.INPUT_SECOND_MOMENT else ":input")
        expected = (key.source_fingerprint, key.model_fingerprint, key.graph_version,
                    point, key.calibration_fingerprint, key.data_role, "", "", key.normalization,
                    key.accumulator_dtype, key.estimator_version)
        actual = (passport.checkpoint_id, passport.predecessor_version, passport.graph_version,
                  passport.observation_point, passport.data_id, passport.data_role,
                  passport.mask_id, passport.weight_id, passport.normalization, passport.dtype, passport.method_version)
        if actual != expected:
            raise ArtifactStoreError("StaleArtifact", "snapshot.passport", "Snapshot passport does not match the declared key")
        if (type(passport.count) is not int or passport.count <= 0
                or type(passport.weight_sum) not in (int, float) or passport.weight_sum != passport.count):
            raise ArtifactStoreError("InvalidSnapshot", "snapshot.passport", "Unweighted observations require positive count and matching mass")
        valid_shape = (tensor.ndim == 2 and tensor.shape[0] == tensor.shape[1]) if snapshot.kind is ArtifactKind.INPUT_SECOND_MOMENT else tensor.ndim == 1
        if not valid_shape:
            raise ArtifactStoreError("InvalidSnapshot", "snapshot.shape", "Second moment must be square; absolute statistic must be a vector")
    return size


class StatisticsStore:
    """Single-owner bounded store. Concurrent/reentrant transactions are refused."""
    def __init__(self, max_bytes):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ArtifactStoreError("InvalidResourceLimit", "max_bytes", "A nonnegative integer capacity is required")
        self._max_bytes = max_bytes
        self._entries = {}
        self._used_bytes = 0
        self._active = False
        self._closed = False

    @property
    def max_bytes(self):
        return self._max_bytes

    @property
    def used_bytes(self):
        return self._used_bytes

    @property
    def refs(self):
        return tuple(StatisticRef(key.digest, key) for key in self._entries)

    def _open(self):
        if self._closed:
            raise ArtifactStoreError("StoreClosed", "store", "Statistics store has been closed")

    def _borrow(self, ref, expected):
        self._open()
        if (type(ref) is not StatisticRef or type(expected) is not StatisticsKey
                or ref.provenance != expected or ref.key_digest != expected.digest):
            raise ArtifactStoreError("StaleArtifact", "reference", "Saved reference does not match expected statistics provenance")
        if expected not in self._entries:
            raise ArtifactStoreError("MissingArtifact", "reference", "Statistics snapshot is no longer present")
        return self._entries[expected]

    def read(self, ref, *, expected, max_peak_bytes=12 * 1024**3):
        """Export an independent copy; a saved ref cannot acquire new meaning."""
        return _copy(self._borrow(ref, expected), max_peak_bytes)

    def put(self, key, snapshot, *, max_peak_bytes=12 * 1024**3):
        """Own an independent copy atomically; caller tensors remain external."""
        with self._transaction() as transaction:
            transaction.reserve(key, _validate_snapshot(key, snapshot))
            return transaction.put(key, snapshot, max_peak_bytes=max_peak_bytes)

    def clear(self):
        self._open()
        if self._active:
            raise ArtifactStoreError("StoreBusy", "store", "Cannot clear during a transaction")
        self._entries.clear()
        self._used_bytes = 0

    def close(self):
        if not self._closed:
            self.clear()
            self._closed = True

    @contextmanager
    def _transaction(self):
        self._open()
        if self._active:
            raise ArtifactStoreError("StoreBusy", "store", "Only one active transaction is supported")
        self._active = True
        transaction = _Transaction(self)
        try:
            yield transaction
            if transaction.pending.keys() != transaction.reservations.keys():
                raise ArtifactStoreError("IncompleteTransaction", "store", "Every reservation must have a completed snapshot")
            self._entries.update(transaction.pending)
            self._used_bytes += transaction.reserved
        finally:
            transaction.pending.clear()
            transaction.reservations.clear()
            transaction.reserved = 0
            transaction.finished = True
            self._active = False


class _Transaction:
    def __init__(self, store):
        self.store, self.pending, self.reservations = store, {}, {}
        self.reserved = 0
        self.finished = False

    def _open(self):
        if self.finished:
            raise ArtifactStoreError("TransactionClosed", "store", "Transaction has completed")

    def lookup(self, key):
        self._open()
        key.digest
        return self.pending.get(key, self.store._entries.get(key))

    def reserve(self, key, expected_bytes):
        """Admission occurs before observing/allocating the prospective result."""
        self._open()
        key.digest
        if key in self.reservations or self.lookup(key) is not None:
            raise ArtifactStoreError("DuplicateArtifact", "key", "Key is already stored or reserved")
        if type(expected_bytes) is not int or expected_bytes <= 0:
            raise ArtifactStoreError("InvalidResourceLimit", "snapshot", "Positive exact snapshot size required")
        if self.store.used_bytes + self.reserved + expected_bytes > self.store.max_bytes:
            raise ArtifactStoreError("StoreCapacityExceeded", "store", "Statistics admission exceeds unique owned tensor capacity")
        self.reservations[key] = expected_bytes
        self.reserved += expected_bytes

    def put(self, key, snapshot, *, max_peak_bytes=12 * 1024**3):
        self._open()
        size = _validate_snapshot(key, snapshot)
        if key not in self.reservations or self.reservations[key] != size or snapshot.kind is not key.statistic:
            raise ArtifactStoreError("InvalidSnapshot", "snapshot", "Snapshot must match an exact prior reservation and semantic type")
        owned = _copy(snapshot, max_peak_bytes)
        self.pending[key] = owned
        return StatisticRef(key.digest, key)
