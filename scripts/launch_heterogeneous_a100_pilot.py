"""Explicit one-shot direct-host pilot launch with a private OS deadline.

Run only on the authorized host after hardware preflight. No SSH, container
broker, guessed GPU index, automatic retry or time-budget fallback is provided.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import secrets
import subprocess
import sys
import time

# Permit the explicit script entry point in an otherwise clean source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedcore.experiments.heterogeneous_deadline import (
    DeadlineBudget, DeadlineReceipt, DeadlineRequest,
    SystemdDeadlineLauncher, _read_bounded)
from fedcore.experiments.heterogeneous_pilot import pilot_config_from_payload


def launch(*, config_path, data_root, run_directory, namespace_id):
    started_at = time.monotonic()
    raw = json.loads(_read_bounded(config_path))
    config = pilot_config_from_payload(raw)
    root = Path(run_directory).resolve()
    cwd = Path(__file__).resolve().parents[1]
    data = Path(data_root).resolve(strict=True)
    if sys.platform != 'linux' or not data.is_dir():
        raise ValueError('Explicit direct Linux host and existing CIFAR data directory required')
    head = subprocess.run(('git', 'rev-parse', 'HEAD'), cwd=cwd, capture_output=True,
                          text=True, timeout=3, check=True).stdout.strip()
    dirty = subprocess.run(('git', 'status', '--porcelain'), cwd=cwd, capture_output=True,
                           text=True, timeout=3, check=True).stdout
    if head != config.source_revision or dirty.strip():
        raise ValueError('The source checkout must be clean and match the frozen revision')
    root.mkdir(parents=True, exist_ok=False)
    config_copy = root / 'config.json'
    with config_copy.open('x', encoding='utf-8') as stream:
        json.dump(raw, stream, indent=2, allow_nan=False)
    boundary = root / 'os-boundary.json'
    request = DeadlineRequest((sys.executable, '-m', 'fedcore.experiments.heterogeneous_pilot_guard',
        '--config', str(config_copy), '--data-root', str(data), '--run-directory', str(root),
        '--boundary-receipt', str(boundary), '--namespace-id', namespace_id,
        '--started-at', str(started_at)), str(cwd), secrets.token_hex(16), gpu_uuid=config.gpu_uuid)
    budget = DeadlineBudget(started_at, config.gpu_seconds, cleanup_seconds=config.cleanup_seconds)
    launcher = SystemdDeadlineLauncher()
    outcome = launcher.launch(request, budget)
    with (root / 'launch.json').open('x', encoding='utf-8') as stream:
        json.dump({'started_at': started_at, 'outcome_type': type(outcome).__name__,
                   'outcome': asdict(outcome), 'automatic_retry': False}, stream,
                  indent=2, allow_nan=False)
    if type(outcome) is DeadlineReceipt:
        # Atomic publication is the first gate: the service cannot spawn a
        # numerical worker until the launcher has verified its own invocation.
        pending = boundary.with_suffix('.tmp')
        with pending.open('x', encoding='utf-8') as stream:
            json.dump(asdict(outcome), stream, indent=2, allow_nan=False)
            stream.flush()
            import os
            os.fsync(stream.fileno())
        pending.replace(boundary)
    return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'data-root', 'run-directory', 'namespace-id'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    outcome = launch(config_path=args.config, data_root=args.data_root,
                     run_directory=args.run_directory, namespace_id=args.namespace_id)
    print(json.dumps({'outcome_type': type(outcome).__name__, 'outcome': asdict(outcome)}, allow_nan=False))
    return 0 if type(outcome) is DeadlineReceipt else 1


if __name__ == '__main__':
    raise SystemExit(main())
