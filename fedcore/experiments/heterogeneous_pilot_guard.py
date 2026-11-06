"""Direct-host pilot guard inside the launcher's private systemd service.

Importing never observes a host, releases a gate or launches a worker.
The trusted host PID namespace is provisioned explicitly by the operator.
"""
from dataclasses import replace
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def run_pilot_guard(*, config_path, data_root, run_directory, boundary_receipt,
                    namespace_id, started_at, python_executable=None):
    from .heterogeneous_pilot import (pilot_config_from_payload,
        verify_worker_boundary, wait_for_start_gate)
    from .cv_independent_guard import GuardEffects, GuardLimits, run_guard
    from .heterogeneous_direct_host import DirectHostBindingProvider
    from .heterogeneous_host import query_device_gpu_rows
    from .heterogeneous_deadline import _read_bounded

    root = Path(run_directory).resolve()
    if (sys.platform != 'linux' or type(started_at) not in (float, int)
            or not math.isfinite(started_at) or not 0 <= started_at <= time.monotonic()
            or not root.is_dir()):
        raise ValueError('An existing private Linux run directory and original start are required')
    config = pilot_config_from_payload(json.loads(_read_bounded(config_path)))
    # The launcher publishes this file only after live OS-boundary verification.
    wait_for_start_gate(boundary_receipt, timeout=10.)
    provenance = verify_worker_boundary(boundary_receipt, config=config, started_at=started_at,
        deadline=started_at + config.gpu_seconds - config.cleanup_seconds)
    if os.readlink('/proc/self/ns/pid') != namespace_id:
        raise RuntimeError('Guard is outside the explicitly provisioned host PID namespace')
    with (root / 'guard-boundary.json').open('x', encoding='utf-8') as stream:
        json.dump(provenance, stream, indent=2, allow_nan=False)
    python_executable = python_executable or sys.executable
    provider = DirectHostBindingProvider((python_executable, '-m',
        'fedcore.experiments.heterogeneous_direct_host'), namespace_id)
    gate = root / 'worker-start-gate.json'
    def spawn(command, **kwargs):
        # run_guard calls this only after an empty-GPU and host-binding preflight.
        # The worker checks the live unit itself again after reading this gate.
        with gate.open('x', encoding='utf-8') as stream:
            json.dump({'boundary_receipt': str(boundary_receipt),
                       'invocation_id': provenance['invocation_id']}, stream)
            stream.flush(); os.fsync(stream.fileno())
        return subprocess.Popen(command, **kwargs)
    command = (python_executable, str(Path(__file__).resolve().parents[2] /
        'scripts' / 'run_heterogeneous_a100_pilot.py'), '--config', str(config_path),
        '--data-root', str(data_root), '--output', str(root / 'pilot'),
        '--boundary-receipt', str(boundary_receipt), '--start-gate', str(gate),
        '--started-at', str(started_at))
    effects = replace(GuardEffects(), popen=spawn,
        gpu_rows=lambda: query_device_gpu_rows(config.gpu_uuid), clock=time.monotonic)
    return run_guard(command, receipt_path=root / 'guard.json', binding_provider=provider,
        cwd=str(Path(__file__).resolve().parents[2]),
        environment={'CUDA_VISIBLE_DEVICES': config.gpu_uuid,
                     'FEDCORE_PILOT_ABSOLUTE_DEADLINE': str(started_at + config.gpu_seconds - config.cleanup_seconds)},
        effects=effects, limits=GuardLimits(config.gpu_seconds, config.cleanup_seconds), started_at=started_at)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'data-root', 'run-directory', 'boundary-receipt', 'namespace-id'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--started-at', required=True, type=float)
    args = parser.parse_args(argv)
    result = run_pilot_guard(config_path=args.config, data_root=args.data_root,
        run_directory=args.run_directory, boundary_receipt=args.boundary_receipt,
        namespace_id=args.namespace_id, started_at=args.started_at)
    print(json.dumps(result, allow_nan=False))
    return 0 if result['status'] == 'finished' and result.get('returncode') == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
