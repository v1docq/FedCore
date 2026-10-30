"""Guard v2 bridge; a trusted external namespace observer is mandatory.

No host observer is implemented or discovered here. Without a caller-supplied,
verified provider this API writes a blocked receipt before spawning anything.
Calling this API is not authorization to repeat an experiment. There is no CLI
override, namespace guess, group kill, ML import, or automatic numerical retry.

Integration still requires a time-bounded host adapter and an external OS-level
deadline for the whole guard. An arbitrary injected callback can block forever;
the Python timer can stop known owned children but does not prove that adapter
boundedness or the guard's own eight-hour termination. Neither is verified here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
from typing import Callable, Protocol

import psutil

from .cv_independent_supervision import (CleanupState, GpuReason, LocalState,
    NamespaceBinding, ProcessIdentity, SupervisionState, advance_supervision,
    classify_gpu_rows, observe_owned_cleanup, resolve_local_process, signal_local_owned)

MAX_SECONDS, WORKER_SECONDS = 28800, 28790
MAX_RSS_BYTES, MAX_VRAM_BYTES, MINIMUM_DISK_BYTES = 32*1024**3, 16*1024**3, 100*1024**3
STOP_SIGNAL = getattr(signal, 'SIGKILL', signal.SIGTERM)


class BindingProvider(Protocol):
    def __call__(self, raw_rows: tuple[str, ...], owned: tuple[ProcessIdentity, ...]) -> NamespaceBinding:
        """Return fresh mappings/evidence within the host adapter's externally enforced time bound."""


def query_gpu_rows():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=5)
    return tuple(row for row in result.stdout.splitlines() if row.strip())


@dataclass(frozen=True)
class GuardEffects:
    popen: Callable = subprocess.Popen
    process: Callable = psutil.Process
    gpu_rows: Callable = query_gpu_rows
    clock: Callable = time.perf_counter
    sleep: Callable = time.sleep
    timer: Callable = threading.Timer
    disk_free: Callable = lambda path: shutil.disk_usage(path).free
    guard_rss: Callable = lambda: psutil.Process().memory_info().rss


def _atomic(path, value):
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


class OwnedTree:
    """Keep local births captured from the private launched tree, never GPU rows."""
    def __init__(self, effects):
        self.effects, self.identities, self.lock = effects, {}, threading.RLock()

    @property
    def owned(self):
        with self.lock:
            return tuple(self.identities.values())

    def attach(self, pid):
        process = self.effects.process(pid)
        identity = ProcessIdentity(pid, process.create_time())
        with self.lock:
            if pid in self.identities and self.identities[pid] != identity:
                raise RuntimeError('Owned local PID birth changed')
            self.identities[pid] = identity

    def capture_rss(self):
        with self.lock:
            pending, seen, rss = list(self.identities.values()), set(), self.effects.guard_rss()
            while pending:
                identity = pending.pop()
                if identity.pid in seen:
                    continue
                seen.add(identity.pid)
                try:
                    process = self.effects.process(identity.pid)
                    local = resolve_local_process(identity.pid, process_factory=lambda pid: process)
                    if local.identity is not None and local.identity != identity or local.state is LocalState.UNKNOWN:
                        raise RuntimeError('Owned process identity is unverified')
                    if local.state is not LocalState.LIVE:
                        continue
                    rss += process.memory_info().rss
                    for child in process.children(recursive=True):
                        try:
                            child_identity = ProcessIdentity(child.pid, child.create_time())
                        except psutil.NoSuchProcess:
                            continue
                        if child.pid in self.identities and self.identities[child.pid] != child_identity:
                            raise RuntimeError('Descendant local PID birth changed')
                        self.identities[child.pid] = child_identity
                        pending.append(child_identity)
                except psutil.NoSuchProcess:
                    # Completed sequential children may disappear from local RSS
                    # scans. This exception is never used for raw GPU identities.
                    continue
            return rss

    def stop(self):
        with self.lock:
            capture_error = None
            try:
                self.capture_rss()
            except Exception as error:
                capture_error = f'{type(error).__name__}: {error}'
            owned = self.owned
            signals = [asdict(signal_local_owned(identity, owned=owned, signal_number=STOP_SIGNAL,
                process_factory=self.effects.process)) for identity in reversed(owned)]
            observed = observe_owned_cleanup(owned,
                resolver=lambda pid: resolve_local_process(pid, process_factory=self.effects.process))
            return {'signals': signals, 'capture_error': capture_error,
                'live': [asdict(row) for row in observed if row.state is CleanupState.LIVE],
                'zombie': [asdict(row) for row in observed if row.state is CleanupState.ZOMBIE],
                'unknown': [asdict(row) for row in observed if row.state in (CleanupState.UNKNOWN, CleanupState.PID_REUSED)],
                'exited': [asdict(row) for row in observed if row.state is CleanupState.EXITED]}


def _bound_observation(rows, owned, provider, effects):
    preliminary = classify_gpu_rows(rows, owned=owned)
    if preliminary.terminal and preliminary.reason is not GpuReason.IDENTITY_UNVERIFIED:
        return preliminary, None
    try:
        binding = provider(tuple(rows), owned)
    except Exception as error:
        raise RuntimeError('namespace_binding_unverified') from error
    if type(binding) is not NamespaceBinding or not binding.verified:
        raise RuntimeError('namespace_binding_unverified')
    return classify_gpu_rows(rows, owned=owned, binding=binding,
        resolver=lambda pid: resolve_local_process(pid, process_factory=effects.process)), binding


def _gpu_bytes(rows):
    used = 0
    for row in rows:
        fields = row.split(',')
        if len(fields) != 2 or not fields[1].strip().isascii() or not fields[1].strip().isdecimal():
            raise RuntimeError('Malformed GPU memory telemetry')
        used += int(fields[1])*1024**2
    return used


def _absolute_deadline(began, environment):
    limits = [began+WORKER_SECONDS]
    for raw in (os.environ.get('FEDCORE_PILOT_ABSOLUTE_DEADLINE'),
                (environment or {}).get('FEDCORE_PILOT_ABSOLUTE_DEADLINE')):
        if raw is not None:
            value = float(raw)
            if not math.isfinite(value) or value <= began:
                raise ValueError('External absolute deadline must be finite and in the future')
            limits.append(value)
    return min(limits)


def run_guard(command, *, receipt_path, binding_provider=None, cwd=None, environment=None, effects=None):
    """Run at most one private child; persist every stop and owned cleanup result."""
    effects = GuardEffects() if effects is None else effects
    receipt = Path(receipt_path).resolve()
    ledger, console = receipt.with_name(receipt.name+'.jsonl'), receipt.with_name(receipt.name+'.console.log')
    if any(path.exists() for path in (receipt, ledger, console)):
        raise FileExistsError('Preserve earlier guard evidence; choose a new receipt')
    receipt.parent.mkdir(parents=True, exist_ok=True)
    began, tree = effects.clock(), OwnedTree(effects)
    state = {'schema': 'fedcore-independent-cv-guard-v2', 'status': 'blocked', 'automatic_retry': False,
        'spawned_child_count': 0, 'max_seconds': MAX_SECONDS, 'worker_seconds': WORKER_SECONDS,
        'max_rss_bytes': MAX_RSS_BYTES, 'max_vram_bytes': MAX_VRAM_BYTES,
        'started_at': began, 'telemetry_failures': 0, 'peak_tree_rss_bytes': 0, 'peak_gpu_used_bytes': 0}
    child, timer, deadline_fired, monitor = None, None, threading.Event(), SupervisionState()

    def record(row):
        with ledger.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
        _atomic(receipt, state)

    def expire():
        deadline_fired.set()
        tree.stop()

    try:
        if binding_provider is None or not callable(binding_provider):
            state['reason'] = 'namespace_binding_provider_missing'
            return state
        rows = effects.gpu_rows()
        observation, binding = _bound_observation(rows, (), binding_provider, effects)
        state['preflight_gpu'] = asdict(observation)
        state['namespace_binding'] = asdict(binding) if binding is not None else None
        if observation.terminal or observation.reason is not GpuReason.EMPTY:
            state['reason'] = observation.reason.value
            return state
        if effects.disk_free(cwd or receipt.parent) < MINIMUM_DISK_BYTES:
            state['reason'] = 'less_than_100_GiB_free_disk'
            return state
        absolute = _absolute_deadline(began, environment)
        state['absolute_deadline'] = absolute
        if effects.clock() >= absolute:
            state['reason'] = 'absolute_deadline_before_spawn'
            return state
        env = dict(os.environ)
        env.update(environment or {})
        env['FEDCORE_PILOT_ABSOLUTE_DEADLINE'] = str(absolute)
        with console.open('x', encoding='utf-8') as output:
            child = effects.popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status='running', spawned_child_count=1, controller_pid=child.pid)
            tree.attach(child.pid)
            timer = effects.timer(max(0., absolute-effects.clock()), expire)
            timer.daemon = True; timer.start()
            while True:
                reason, observation = None, None
                try:
                    rows = effects.gpu_rows()
                    # Raw row multiplicity wins before tree/provider/local probes.
                    preliminary = classify_gpu_rows(rows, owned=tree.owned)
                    if preliminary.terminal and preliminary.reason is not GpuReason.IDENTITY_UNVERIFIED:
                        observation = preliminary
                    else:
                        rss = tree.capture_rss()
                        observation, binding = _bound_observation(rows, tree.owned, binding_provider, effects)
                        state['namespace_binding'] = asdict(binding) if binding is not None else None
                        state['peak_tree_rss_bytes'] = max(state['peak_tree_rss_bytes'], rss)
                        if rss > MAX_RSS_BYTES: reason = 'tree_rss_limit'
                        if not observation.terminal:
                            used = _gpu_bytes(rows)
                            state['peak_gpu_used_bytes'] = max(state['peak_gpu_used_bytes'], used)
                            if used > MAX_VRAM_BYTES: reason = reason or 'vram_limit'
                    monitor = advance_supervision(monitor, observation)
                    if observation.terminal: reason = monitor.terminal_reason.value
                except Exception as error:
                    monitor = replace(monitor, telemetry_failures=monitor.telemetry_failures+1)
                    state['telemetry_error'] = f'{type(error).__name__}: {error}'
                    reason = ('gpu_identity_unverified' if str(error) == 'namespace_binding_unverified'
                              else 'telemetry_unavailable')
                if deadline_fired.is_set() or effects.clock() >= absolute: reason = reason or 'absolute_deadline'
                state.update(telemetry_failures=monitor.telemetry_failures, elapsed_seconds=effects.clock()-began,
                    last_gpu=asdict(observation) if observation else None)
                record({'elapsed_seconds': state['elapsed_seconds'], 'reason': reason, 'gpu': state['last_gpu'],
                        'telemetry_failures': monitor.telemetry_failures})
                if reason:
                    state.update(status='stopped', reason=reason)
                    break
                code = child.poll()
                if code is not None:
                    state.update(status='finished', reason='controller_exit', returncode=code)
                    break
                effects.sleep(.5)
    except Exception as error:
        state.update(reason=('gpu_identity_unverified' if str(error) == 'namespace_binding_unverified'
                             else 'guard_preflight_or_runtime_failure'), failure=f'{type(error).__name__}: {error}',
                     status='stopped' if child is not None else 'blocked')
    finally:
        if timer is not None: timer.cancel()
        state['cleanup'] = tree.stop()
        if child is not None and not tree.owned and child.poll() is None:
            state['cleanup']['unknown'].append({'local_pid': child.pid, 'reason': 'spawned_child_birth_not_captured'})
        if child is not None:
            try: state['returncode'] = child.wait(timeout=max(0., min(1., began+MAX_SECONDS-effects.clock())))
            except subprocess.TimeoutExpired: state['returncode'] = child.poll()
        state.update(elapsed_seconds=effects.clock()-began, owned=[asdict(x) for x in tree.owned])
        state['within_authorized_window'] = state['elapsed_seconds'] <= MAX_SECONDS
        if not state['within_authorized_window']:
            state.update(status='stopped', reason='authorized_window_exceeded')
        if state['cleanup']['live'] or state['cleanup']['unknown']:
            state.update(status='stopped', reason=state.get('reason') or 'owned_cleanup_unverified')
        record({'event': 'guard_stopped', 'status': state['status'], 'reason': state.get('reason'),
                'cleanup': state['cleanup']})
    return state
