"""Owned pilot worker, launched only behind an external guard's start gate.

This entry point does not connect to a host, launch a service or retry a run.
It imports no numerical package until the gate and live OS boundary pass.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", "--data", dest="data_root", required=True)
    parser.add_argument("--boundary-receipt", required=True)
    parser.add_argument("--start-gate", required=True)
    parser.add_argument("--started-at", required=True, type=float)
    args = parser.parse_args(argv)
    # Support an explicit script path without changing the installed package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fedcore.experiments.heterogeneous_pilot import (
        pilot_config_from_payload, run_pilot, verify_worker_boundary, wait_for_start_gate)
    with Path(args.config).open("r", encoding="utf-8") as stream:
        raw = stream.read(65537)
    if len(raw.encode()) > 65536:
        raise ValueError("Pilot config exceeds bounded capacity")
    config = pilot_config_from_payload(json.loads(raw))
    if Path(args.output).exists():
        raise FileExistsError("Pilot output must be fresh; no resume or overwrite")
    if args.started_at < 0 or args.started_at > time.monotonic():
        raise ValueError("Original controller monotonic start is required")
    wait_for_start_gate(args.start_gate, timeout=min(10., max(.001,
        args.started_at + config.gpu_seconds - config.cleanup_seconds - time.monotonic())))
    def boundary_check(**kwargs):
        return verify_worker_boundary(args.boundary_receipt, **kwargs)
    boundary_check(config=config, started_at=args.started_at,
                   deadline=args.started_at + config.gpu_seconds - config.cleanup_seconds)
    from fedcore.experiments.heterogeneous_data import CifarTrainingData
    data = CifarTrainingData(args.data_root, split_seed=config.split_seed)
    receipt = run_pilot(data, args.output, config=config, started_at=args.started_at, boundary_check=boundary_check)
    print(json.dumps(dict(status=receipt["status"], experiment_started=receipt["experiment_started"],
                          receipt_path=str(Path(args.output).resolve() / "receipt.json")), allow_nan=False))
    return 0 if receipt["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
