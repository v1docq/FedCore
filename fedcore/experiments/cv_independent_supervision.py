"""Fail-closed CPU supervision primitives for an explicitly owned process tree.

NVML/nvidia-smi PIDs are not local PIDs unless an external observer has supplied
an explicit, verified namespace binding. This module never infers such a binding
from equal integers, /proc inodes, or a process that happens to be visible.

The classifier and state transition are pure apart from an injected local PID
resolver. Only the resolver and the explicitly called cleanup signal boundary
use psutil; importing this module has no GPU, process, or filesystem effects.
No experiment launch, retry, namespace discovery, or permission grant lives here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
import signal
from typing import Callable


def _positive_pid(value):
    return type(value) is int and value > 0


def _reported_pid(row):
    token = row.split(',', 1)[0].strip()
    return (int(token) if token.isascii() and token.isdecimal()
            and len(token) <= 20 and int(token) > 0 else None)


@dataclass(frozen=True)
class ProcessIdentity:
    """A local PID paired with the creation time recorded when ownership began."""
    pid: int
    birth: float

    def __post_init__(self):
        if (not _positive_pid(self.pid) or type(self.birth) not in (int, float)
                or not math.isfinite(self.birth) or self.birth <= 0):
            raise ValueError('A positive local PID and finite positive creation time are required')


@dataclass(frozen=True)
class PidBinding:
    """One external GPU identity mapped to a particular local process lifetime."""
    gpu_pid: int
    local: ProcessIdentity

    def __post_init__(self):
        if not _positive_pid(self.gpu_pid) or type(self.local) is not ProcessIdentity:
            raise ValueError('An explicit GPU PID to local process identity mapping is required')


@dataclass(frozen=True)
class NamespaceBinding:
    verified: bool = False
    entries: tuple[PidBinding, ...] = ()
    evidence: str | None = None

    def __post_init__(self):
        if (type(self.verified) is not bool or type(self.entries) is not tuple
                or any(type(entry) is not PidBinding for entry in self.entries)
                or len({entry.gpu_pid for entry in self.entries}) != len(self.entries)
                or self.evidence is not None and (type(self.evidence) is not str or not self.evidence.strip())
                or self.verified and self.evidence is None):
            raise ValueError('A verified namespace binding requires explicit external evidence and unique mappings')


class LocalState(str, Enum):
    LIVE = 'live'
    ZOMBIE = 'zombie'
    EXITED = 'exited'
    UNKNOWN = 'unknown'


class LocalFailure(str, Enum):
    NO_SUCH_PROCESS = 'no_such_process'
    ACCESS_DENIED = 'access_denied'
    ZOMBIE = 'zombie'
    BIRTH_CHANGED = 'birth_changed_during_probe'
    RESOLVER_ERROR = 'resolver_error'


@dataclass(frozen=True)
class LocalObservation:
    local_pid: int
    state: LocalState
    identity: ProcessIdentity | None = None
    failure: LocalFailure | None = None
    detail: str | None = None

    def __post_init__(self):
        if (not _positive_pid(self.local_pid) or type(self.state) is not LocalState
                or self.identity is not None and (type(self.identity) is not ProcessIdentity
                                                  or self.identity.pid != self.local_pid)
                or self.failure is not None and type(self.failure) is not LocalFailure
                or self.state is LocalState.LIVE and (self.identity is None or self.failure is not None)):
            raise ValueError('A local observation must identify its PID and its explicit lifecycle state')


Resolver = Callable[[int], LocalObservation]


def _local_exception(local_pid, error):
    import psutil
    if isinstance(error, psutil.ZombieProcess):
        return LocalObservation(local_pid, LocalState.ZOMBIE, failure=LocalFailure.ZOMBIE, detail=str(error))
    if isinstance(error, psutil.NoSuchProcess):
        return LocalObservation(local_pid, LocalState.EXITED, failure=LocalFailure.NO_SUCH_PROCESS, detail=str(error))
    if isinstance(error, psutil.AccessDenied):
        return LocalObservation(local_pid, LocalState.UNKNOWN, failure=LocalFailure.ACCESS_DENIED, detail=str(error))
    return LocalObservation(local_pid, LocalState.UNKNOWN, failure=LocalFailure.RESOLVER_ERROR,
                            detail=f'{type(error).__name__}: {error}')


def _probe_process(process, local_pid):
    import psutil
    first_birth = process.create_time()
    status = process.status()
    alive = process.is_running()
    last_birth = process.create_time()
    if first_birth != last_birth:
        return LocalObservation(local_pid, LocalState.UNKNOWN, failure=LocalFailure.BIRTH_CHANGED)
    identity = ProcessIdentity(local_pid, first_birth)
    if status == psutil.STATUS_ZOMBIE:
        return LocalObservation(local_pid, LocalState.ZOMBIE, identity, LocalFailure.ZOMBIE)
    if status == psutil.STATUS_DEAD or not alive:
        return LocalObservation(local_pid, LocalState.EXITED, identity, LocalFailure.NO_SUCH_PROCESS)
    return LocalObservation(local_pid, LocalState.LIVE, identity)


def resolve_local_process(local_pid, *, process_factory=None):
    """Read one *local* PID; vanished/denied/zombie observations remain failures."""
    if not _positive_pid(local_pid):
        raise ValueError('A positive local PID is required')
    import psutil
    factory = psutil.Process if process_factory is None else process_factory
    try:
        return _probe_process(factory(local_pid), local_pid)
    except Exception as error:
        return _local_exception(local_pid, error)


class GpuReason(str, Enum):
    EMPTY = 'no_gpu_processes'
    OWNED = 'verified_owned_gpu_process'
    MULTIPLE_ROWS = 'multiple_gpu_rows'
    IDENTITY_UNVERIFIED = 'gpu_identity_unverified'
    FOREIGN = 'foreign_gpu_process'
    TELEMETRY_UNAVAILABLE = 'gpu_telemetry_unavailable'


_CONTINUE_REASONS = (GpuReason.EMPTY, GpuReason.OWNED)


@dataclass(frozen=True)
class GpuObservation:
    reason: GpuReason
    raw_rows: tuple[str, ...]
    namespace_binding_verified: bool
    reported_gpu_pid: int | None = None
    binding: PidBinding | None = None
    local: LocalObservation | None = None
    detail: str | None = None

    def __post_init__(self):
        if (type(self.reason) is not GpuReason or type(self.namespace_binding_verified) is not bool
                or type(self.raw_rows) is not tuple or any(type(row) is not str or not row.strip() for row in self.raw_rows)
                or self.reported_gpu_pid is not None and not _positive_pid(self.reported_gpu_pid)
                or self.binding is not None and type(self.binding) is not PidBinding
                or self.local is not None and type(self.local) is not LocalObservation
                or self.reason is GpuReason.EMPTY and bool(self.raw_rows)
                or self.reason is GpuReason.MULTIPLE_ROWS and len(self.raw_rows) <= 1
                or self.reason in (GpuReason.OWNED, GpuReason.FOREIGN)
                and (not self.namespace_binding_verified or self.binding is None or self.local is None
                     or len(self.raw_rows) != 1 or _reported_pid(self.raw_rows[0]) != self.reported_gpu_pid
                     or self.local.state is not LocalState.LIVE or self.local.identity != self.binding.local
                     or self.reported_gpu_pid != self.binding.gpu_pid)):
            raise ValueError('A GPU observation must preserve raw rows and verified identity evidence')

    @property
    def terminal(self):
        return self.reason not in _CONTINUE_REASONS


def _validate_owned(owned):
    if (type(owned) is not tuple or any(type(identity) is not ProcessIdentity for identity in owned)
            or len({identity.pid for identity in owned}) != len(owned)):
        raise ValueError('An immutable registry of distinct owned local PID lifetimes is required')


def classify_gpu_rows(rows, *, owned=(), binding=NamespaceBinding(), resolver=resolve_local_process):
    """Classify raw nonempty GPU rows before attempting any local PID lookup.

    ``rows=None`` means the GPU query itself failed. More than one nonempty row
    always terminates before parsing, ownership checks, or resolver calls. A
    single row may be resolved only via an explicit verified binding entry.
    """
    _validate_owned(owned)
    if type(binding) is not NamespaceBinding:
        raise ValueError('An explicit NamespaceBinding value is required')
    if rows is None:
        return GpuObservation(GpuReason.TELEMETRY_UNAVAILABLE, (), binding.verified,
                              detail='The GPU process query did not produce a trustworthy sample')
    if type(rows) not in (tuple, list) or any(type(row) is not str for row in rows):
        raise ValueError('Raw GPU rows must be strings, or None for a failed query')
    raw_rows = tuple(row for row in rows if row.strip())
    if len(raw_rows) > 1:
        return GpuObservation(GpuReason.MULTIPLE_ROWS, raw_rows, binding.verified,
                              detail='Multiple raw GPU rows; local PID resolution is intentionally skipped')
    if not raw_rows:
        return GpuObservation(GpuReason.EMPTY, raw_rows, binding.verified)
    if not binding.verified:
        return GpuObservation(GpuReason.IDENTITY_UNVERIFIED, raw_rows, False,
                              detail='No externally verified namespace binding')
    gpu_pid = _reported_pid(raw_rows[0])
    if gpu_pid is None:
        return GpuObservation(GpuReason.IDENTITY_UNVERIFIED, raw_rows, True,
                              detail='The nonempty GPU row does not contain a valid reported PID')
    mapping = next((entry for entry in binding.entries if entry.gpu_pid == gpu_pid), None)
    if mapping is None:
        return GpuObservation(GpuReason.IDENTITY_UNVERIFIED, raw_rows, True, gpu_pid,
                              detail='No explicit verified mapping for this reported GPU PID')
    try:
        local = resolver(mapping.local.pid)
    except Exception as error:
        local = _local_exception(mapping.local.pid, error)
    if (type(local) is not LocalObservation or local.local_pid != mapping.local.pid
            or local.state is not LocalState.LIVE or local.identity != mapping.local):
        return GpuObservation(GpuReason.IDENTITY_UNVERIFIED, raw_rows, True, gpu_pid, mapping,
                              local if type(local) is LocalObservation else None,
                              detail='Mapped local process is unavailable, non-live, or has a different birth')
    if local.identity not in owned:
        return GpuObservation(GpuReason.FOREIGN, raw_rows, True, gpu_pid, mapping, local,
                              detail='The verified live process lifetime is absent from the owned registry')
    return GpuObservation(GpuReason.OWNED, raw_rows, True, gpu_pid, mapping, local)


@dataclass(frozen=True)
class SupervisionState:
    terminal_reason: GpuReason | None = None
    telemetry_failures: int = 0
    samples: int = 0

    def __post_init__(self):
        if (self.terminal_reason is not None and (type(self.terminal_reason) is not GpuReason
                                                  or self.terminal_reason in _CONTINUE_REASONS)
                or any(type(value) is not int or value < 0 for value in (self.telemetry_failures, self.samples))):
            raise ValueError('An irreversible terminal reason and nonnegative telemetry counters are required')


def advance_supervision(state, observation):
    """Only a verified active sample resets telemetry failures; terminal is sticky."""
    if type(state) is not SupervisionState or type(observation) is not GpuObservation:
        raise ValueError('Typed supervision state and GPU observation required')
    if state.terminal_reason is not None:
        return replace(state, samples=state.samples+1)
    failures = state.telemetry_failures
    if observation.reason in (GpuReason.IDENTITY_UNVERIFIED, GpuReason.TELEMETRY_UNAVAILABLE):
        failures += 1
    elif observation.reason in _CONTINUE_REASONS:
        failures = 0
    return SupervisionState(observation.reason if observation.terminal else None,
                            failures, state.samples+1)


@dataclass(frozen=True)
class SupervisionDecision:
    state: SupervisionState
    observation: GpuObservation

    @property
    def may_continue(self):
        return self.state.terminal_reason is None


def observe_gpu_rows(rows, *, state=SupervisionState(), owned=(), binding=NamespaceBinding(),
                     resolver=resolve_local_process):
    observation = classify_gpu_rows(rows, owned=owned, binding=binding, resolver=resolver)
    return SupervisionDecision(advance_supervision(state, observation), observation)


class CleanupState(str, Enum):
    LIVE = 'live_owned'
    ZOMBIE = 'zombie_owned'
    EXITED = 'exited_owned'
    PID_REUSED = 'pid_birth_changed'
    UNKNOWN = 'unknown_owned_state'


@dataclass(frozen=True)
class CleanupObservation:
    owned: ProcessIdentity
    state: CleanupState
    local: LocalObservation

    @property
    def may_signal(self):
        return self.state is CleanupState.LIVE and self.local.identity == self.owned


def _cleanup_state(identity, observation):
    if type(observation) is not LocalObservation or observation.local_pid != identity.pid:
        return CleanupObservation(identity, CleanupState.UNKNOWN,
            LocalObservation(identity.pid, LocalState.UNKNOWN, failure=LocalFailure.RESOLVER_ERROR,
                             detail='Resolver returned a different PID or an untyped result'))
    if observation.state is LocalState.UNKNOWN:
        state = CleanupState.UNKNOWN
    elif observation.identity is not None and observation.identity != identity:
        state = CleanupState.PID_REUSED
    elif observation.state is LocalState.ZOMBIE:
        state = CleanupState.ZOMBIE
    elif observation.state is LocalState.EXITED:
        state = CleanupState.EXITED
    elif observation.identity == identity:
        state = CleanupState.LIVE
    else:
        state = CleanupState.UNKNOWN
    return CleanupObservation(identity, state, observation)


def observe_owned_cleanup(owned, *, resolver=resolve_local_process):
    """Observe only previously captured local owned lifetimes, never GPU raw PIDs."""
    _validate_owned(owned)
    observations = []
    for identity in owned:
        try:
            local = resolver(identity.pid)
        except Exception as error:
            local = _local_exception(identity.pid, error)
        observations.append(_cleanup_state(identity, local))
    return tuple(observations)


class SignalDisposition(str, Enum):
    SENT = 'sent_to_live_owned_lifetime'
    NOT_OWNED = 'not_in_local_owned_registry'
    SKIPPED = 'skipped_non_live_or_unverified_lifetime'
    FAILED = 'signal_failed_or_process_changed'


@dataclass(frozen=True)
class SignalObservation:
    target: ProcessIdentity
    disposition: SignalDisposition
    cleanup: CleanupObservation | None = None


def signal_local_owned(target, *, owned, signal_number=signal.SIGTERM, process_factory=None):
    """Explicit signal boundary: fresh birth checks, psutil's reuse-safe sender.

    Both ``target`` and the registry are local ownership records. A reported GPU
    PID cannot be passed as a bare integer. No process group or foreign PID is
    signalled, and unknown/zombie/changed lifetimes receive no signal.
    """
    _validate_owned(owned)
    if type(target) is not ProcessIdentity:
        raise ValueError('Signals require a captured local process lifetime, not a raw GPU PID')
    if target not in owned:
        return SignalObservation(target, SignalDisposition.NOT_OWNED)
    if not isinstance(signal_number, (int, signal.Signals)) or isinstance(signal_number, bool) or signal_number <= 0:
        raise ValueError('A positive signal number is required')
    import psutil
    factory = psutil.Process if process_factory is None else process_factory
    try:
        process = factory(target.pid)
        observed = _probe_process(process, target.pid)
        cleanup = _cleanup_state(target, observed)
        if not cleanup.may_signal:
            return SignalObservation(target, SignalDisposition.SKIPPED, cleanup)
        # send_signal on this freshly created psutil.Process also guards against
        # PID reuse. Never replace it with os.kill on a reported GPU PID.
        process.send_signal(signal_number)
        return SignalObservation(target, SignalDisposition.SENT, cleanup)
    except Exception as error:
        local = _local_exception(target.pid, error)
        return SignalObservation(target, SignalDisposition.FAILED, _cleanup_state(target, local))
