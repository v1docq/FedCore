"""External deadline and telemetry for one owned pilot process, never other jobs."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from .cv_sparse_protocol import PilotProtocol


class MonitorAction(str, Enum):
    FINISHED = 'finished'
    WAIT = 'wait'
    DEADLINE = 'deadline'
    MEMORY = 'memory'
    CONTENTION = 'external_gpu_contention'
    TELEMETRY = 'telemetry_unavailable'


@dataclass(frozen=True)
class ProcessObservation:
    returncode: int | None
    elapsed: float
    rss_bytes: int | None
    gpu_process_count: int | None
    telemetry_failures: int = 0


def monitor_action(observation, *, max_seconds, max_rss_bytes):
    # A child exit is authoritative. Missing RSS after exit is normal and must
    # not turn a successful terminal state into the old UnknownRSS failure.
    if observation.returncode is not None:
        return MonitorAction.FINISHED
    if observation.elapsed >= max_seconds:
        return MonitorAction.DEADLINE
    if observation.telemetry_failures >= 3:
        return MonitorAction.TELEMETRY
    if observation.rss_bytes is not None and observation.rss_bytes > max_rss_bytes:
        return MonitorAction.MEMORY
    if observation.gpu_process_count is not None and observation.gpu_process_count > 1:
        return MonitorAction.CONTENTION
    return MonitorAction.WAIT


def gpu_processes():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=8)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _atomic(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def terminate_owned_group(pgid):
    """The private group may outlive its leader; do not use leader.poll here."""
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def expire_owned_worker(child, pgid, deadline_fired):
    """A timer after a completed child must not rewrite its terminal outcome."""
    if child is not None and child.poll() is None:
        deadline_fired.set()
    terminate_owned_group(pgid)


def supervise(data_root, output_root, manifest, *, continuation_parent=None, parent_inventory=None,
              diagnostic_completion=None, completion_inventory=None, recovery_completion=None,
              recovery_diagnostic=None, diagnostic_inventory=None, evaluation_parent=None,
              evaluation_inventory=None):
    if os.name != 'posix':
        raise RuntimeError('The A100 supervisor requires Linux process-group isolation')
    import psutil
    protocol = PilotProtocol()
    evaluation = evaluation_parent is not None or evaluation_inventory is not None
    if evaluation and (
            evaluation_parent is None or evaluation_inventory is None
            or any(value is not None for value in (continuation_parent, parent_inventory,
                   diagnostic_completion, completion_inventory, recovery_completion,
                   recovery_diagnostic, diagnostic_inventory))):
        raise ValueError('Saved evaluation requires its parent and inventory as the only worker profile')
    if (continuation_parent is None) != (parent_inventory is None):
        raise ValueError('Continuation requires both the parent and its pinned inventory')
    recovery = any(value is not None for value in
                   (recovery_completion, recovery_diagnostic, diagnostic_inventory))
    if recovery and (any(value is None for value in
                        (recovery_completion, completion_inventory, recovery_diagnostic, diagnostic_inventory))
                     or continuation_parent is not None or diagnostic_completion is not None):
        raise ValueError('Recovery requires both immutable parents and inventories, as the only worker profile')
    if not recovery and (diagnostic_completion is None) != (completion_inventory is None):
        raise ValueError('Diagnostics require both the completed run and its pinned inventory')
    continuation = continuation_parent is not None
    diagnostic = diagnostic_completion is not None
    if continuation and diagnostic:
        raise ValueError('Choose one named worker profile')
    max_seconds = 1800 if continuation or diagnostic or recovery or evaluation else protocol.max_slot_seconds
    output = Path(output_root).resolve()
    source = Path(manifest).resolve()
    if output.exists() or not source.is_file():
        raise ValueError('A new output directory and a source manifest are required')
    output.parent.mkdir(parents=True, exist_ok=True)
    state_path = output.parent / (output.name + '-supervision.json')
    tape_path = output.parent / (output.name + '-supervision.jsonl')
    console_path = output.parent / (output.name + '-console.log')
    if any(path.exists() for path in (state_path, tape_path, console_path)):
        raise ValueError('Preserve previous supervision evidence; choose a new output name')
    lock = output.parent / '.fedcore-sparse-cv.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    child = None
    owned_pgid = None
    timer = None
    began, deadline_fired = time.perf_counter(), threading.Event()
    # Leave a small termination reserve inside the approved four-hour slot.
    limit = max_seconds - 10
    state = {'schema': 'cv-pilot-supervision-v1', 'status': 'preflight',
             'max_slot_seconds': max_seconds, 'worker_limit_seconds': limit,
             'termination_reserve_seconds': 10, 'protocol_sha256': protocol.sha256}
    if diagnostic:
        from .cv_sparse_diagnostic_protocol import sha256
        state['diagnostic_protocol_sha256'] = sha256()
    if recovery:
        from .cv_sparse_recovery_protocol import sha256
        state['recovery_protocol_sha256'] = sha256()
    if evaluation:
        from .cv_sparse_evaluation_protocol import sha256
        state['evaluation_protocol_sha256'] = sha256()

    def stop_owned():
        terminate_owned_group(owned_pgid)

    def timer_stop():
        expire_owned_worker(child, owned_pgid, deadline_fired)

    try:
        preflight = gpu_processes()
        state['preflight_gpu_processes'] = preflight
        if preflight:
            state.update(status='not_started', reason='gpu_not_free')
            _atomic(state_path, state)
            return 75
        environment = dict(os.environ)
        environment.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1',
                           OMP_NUM_THREADS=str(protocol.cpu_threads), MKL_NUM_THREADS=str(protocol.cpu_threads),
                           OPENBLAS_NUM_THREADS=str(protocol.cpu_threads), NUMEXPR_NUM_THREADS=str(protocol.cpu_threads),
                           FEDCORE_PILOT_ABSOLUTE_DEADLINE=str(began + limit))
        command = [sys.executable, '-m', 'fedcore.experiments.cv_sparse_pilot', '--execute',
                   '--data-root', str(Path(data_root).resolve()), '--output-root', str(output),
                   '--source-manifest', str(source)]
        if continuation:
            command[2] = 'fedcore.experiments.cv_sparse_continuation'
            command.extend(['--parent-root', str(Path(continuation_parent).resolve()),
                            '--parent-inventory', str(Path(parent_inventory).resolve())])
        if diagnostic:
            command[2] = 'fedcore.experiments.cv_sparse_diagnostics'
            command.extend(['--completion-root', str(Path(diagnostic_completion).resolve()),
                            '--completion-inventory', str(Path(completion_inventory).resolve())])
        if recovery:
            command[2] = 'fedcore.experiments.cv_sparse_recovery'
            command.extend(['--completion-root', str(Path(recovery_completion).resolve()),
                            '--completion-inventory', str(Path(completion_inventory).resolve()),
                            '--diagnostic-root', str(Path(recovery_diagnostic).resolve()),
                            '--diagnostic-inventory', str(Path(diagnostic_inventory).resolve())])
        if evaluation:
            command[2] = 'fedcore.experiments.cv_sparse_evaluation'
            command.extend(['--recovery-root', str(Path(evaluation_parent).resolve()),
                            '--recovery-inventory', str(Path(evaluation_inventory).resolve())])
        state['worker_command'] = command
        with console_path.open('x', encoding='utf-8') as console:
            child = subprocess.Popen(command, env=environment, stdout=console, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            owned_pgid = child.pid
            try:
                process = psutil.Process(child.pid)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                process = None
            timer = threading.Timer(max(0., began + limit - time.perf_counter()), timer_stop)
            timer.daemon = True
            timer.start()
            failures, peak_rss = 0, 0
            state.update(status='running', owned_pid=child.pid)
            with tape_path.open('x', encoding='utf-8') as tape:
                while True:
                    code = child.poll()
                    rss, count = None, None
                    if code is None:
                        try:
                            if process is None:
                                process = psutil.Process(child.pid)
                            rss = process.memory_info().rss
                            count = len(gpu_processes())
                            failures = 0
                        except (psutil.NoSuchProcess, psutil.ZombieProcess):
                            code = child.poll()
                            # A disappearing process is a terminal race; poll
                            # again instead of inventing a missing-memory result.
                            if code is None:
                                time.sleep(.05)
                                continue
                        except (psutil.AccessDenied, RuntimeError, OSError, subprocess.TimeoutExpired):
                            failures += 1
                    elapsed = time.perf_counter() - began
                    observation = ProcessObservation(code, elapsed, rss, count, failures)
                    action = monitor_action(observation, max_seconds=limit, max_rss_bytes=16 * 1024**3)
                    peak_rss = max(peak_rss, rss or 0)
                    row = {'elapsed_seconds': elapsed, 'returncode': code, 'rss_bytes': rss,
                           'gpu_process_count': count, 'action': action.value, 'telemetry_failures': failures}
                    tape.write(json.dumps(row, allow_nan=False) + '\n')
                    tape.flush()
                    state.update(row, peak_rss_bytes=peak_rss)
                    _atomic(state_path, state)
                    if action is MonitorAction.FINISHED:
                        state.update(status='stopped' if deadline_fired.is_set() else 'finished',
                                     reason='deadline' if deadline_fired.is_set() else 'child_exit')
                        return 124 if deadline_fired.is_set() else code
                    if action is not MonitorAction.WAIT:
                        stop_owned()
                        child.wait(timeout=8)
                        state.update(status='stopped', reason=action.value, returncode=child.returncode)
                        return 124 if action is MonitorAction.DEADLINE else 70
                    time.sleep(1)
    finally:
        if timer is not None:
            timer.cancel()
        stop_owned()
        state['elapsed_seconds'] = time.perf_counter() - began
        _atomic(state_path, state)
        # Remove only the lock created by this invocation.
        if lock.is_file() and lock.read_text(encoding='utf-8') == str(os.getpid()):
            lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--continuation-parent')
    parser.add_argument('--parent-inventory')
    parser.add_argument('--diagnostic-completion')
    parser.add_argument('--completion-inventory')
    parser.add_argument('--recovery-completion')
    parser.add_argument('--recovery-diagnostic')
    parser.add_argument('--diagnostic-inventory')
    parser.add_argument('--evaluation-parent')
    parser.add_argument('--evaluation-inventory')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({'protocol': PilotProtocol().payload(), 'protocol_sha256': PilotProtocol().sha256,
                          'status': 'dry_run', 'cuda_context_opened': False}))
        return 0
    extra = {}
    if any(value is not None for value in
           (args.recovery_completion, args.recovery_diagnostic, args.diagnostic_inventory)):
        extra = {'recovery_completion': args.recovery_completion,
                 'recovery_diagnostic': args.recovery_diagnostic,
                 'diagnostic_inventory': args.diagnostic_inventory}
    if args.evaluation_parent is not None or args.evaluation_inventory is not None:
        extra.update(evaluation_parent=args.evaluation_parent, evaluation_inventory=args.evaluation_inventory)
    return supervise(args.data_root, args.output_root, args.source_manifest,
                     continuation_parent=args.continuation_parent, parent_inventory=args.parent_inventory,
                     diagnostic_completion=args.diagnostic_completion, completion_inventory=args.completion_inventory,
                     **extra)


if __name__ == '__main__':
    raise SystemExit(main())
