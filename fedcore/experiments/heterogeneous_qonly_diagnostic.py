"""One bounded read-only comparison of the rejected 2026-11-04 Q-only files.

The paired-logit profile reuses saved files only. A separate fixed layerwise
profile permits one diagnostic instrumented-engine build. Both reuse the
container supervisor/guard, perform no training or quality evaluation, and
never publish a deployment manifest. Importing this module imports no runtime.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import time
from typing import Literal

from . import heterogeneous_container as container
from .heterogeneous_pilot import PilotConfig, PilotError, wait_for_start_gate


WORKFLOW = "4fb0722e39b1fd3a8a0ad1cfefe3b6195feb25cd54694fe61f7cc7c90bb987d9"
ORIGINAL_REVISION = "6b7a92a3779a4a4ac9bd9921c8d6e74ea1bd7e38"
GPU_UUID = "GPU-c31a29dc-f57d-9ce4-0303-849ada1f7cb8"
DATA_SHA256 = "66fad8df85ae5f50d331d559168e38039b84242caaeec89f256961d6ec598e4b"
ROLES_SHA256 = "74727fe62ccab45226a0c18c860049883ac84faf34e3cfe83461403bbe051e14"
CALIBRATION_SHA256 = "f15d9a418dbbb1210915d304d215e5cfabc780264e8b64e77050a0dda5ad8d61"
PRIOR_PAIRS_SHA256 = "f2ce33689ff0647130bb293636b792615ea9e00a6b637a4c6e8946f2acd3337f"
SAVED_SHA256 = {
    "model.onnx": "fe2a9052f9ea093339bfbb7a24f796961440dc655591bc50ea5455ca74b19f2a",
    "model.engine": "858d52d319f012483f25d83205fb02affe041cca55dfc8050e0af330f108feba",
    "engine-inspector.json": "a64af8ae462e6357a5a30643313cc14b9d6048eabc7c4640ba1442615a1e6214",
}
ATTEMPT_SHA256 = "41281ad759454fb1e0a05ec2f42309292fd0381fe0fe1c463b3fc300a6e8eb65"
PILOT_SHA256 = "b80fb15636cd1160f1ea2e5890cc2e595c58074c1660912d5b1a53fadd6620b0"
SCRIPT = "run_heterogeneous_qonly_diagnostic.py"
RTOL, ATOL = .02, .02


@dataclass(frozen=True)
class DiagnosticConfig:
    source_revision: str
    artifact_directory: str
    rows: int = 2000
    batch_size: int = 128
    gpu_uuid: str = GPU_UUID
    profile: Literal["paired_logits_v1", "layerwise_v1", "placement_v1"] = "paired_logits_v1"
    candidate_onnx: str | None = None

    def __post_init__(self):
        if (type(self.source_revision) is not str
                or re.fullmatch(r"[a-f0-9]{40}", self.source_revision) is None
                or type(self.artifact_directory) is not str
                or not (Path(self.artifact_directory).is_absolute() or self.artifact_directory.startswith("/"))
                or any(c in self.artifact_directory for c in ("\0", "\n", "\r"))
                or self.gpu_uuid != GPU_UUID
                or self.profile not in ("paired_logits_v1", "layerwise_v1", "placement_v1")
                or type(self.rows) is not int or not 2 <= self.rows <= 2000
                or type(self.batch_size) is not int or not 1 <= self.batch_size <= 128
                or self.profile == "layerwise_v1" and (self.rows != 8 or self.batch_size != 8)):
            raise ValueError("Exact saved Q-only device/revision, absolute artifact path and 2..2000 rows required")
        if self.profile == "placement_v1":
            if (self.rows != 2000 or self.batch_size != 128 or type(self.candidate_onnx) is not str
                    or not (Path(self.candidate_onnx).is_absolute() or self.candidate_onnx.startswith("/"))
                    or any(c in self.candidate_onnx for c in ("\0", "\n", "\r"))):
                raise ValueError("Placement requires the fixed 2000 rows, batch 128 and an absolute candidate path")
        elif self.candidate_onnx is not None:
            raise ValueError("A candidate graph is only valid for the placement diagnostic")

    def control_config(self):
        # PilotConfig is only the existing boundary's transport contract. Its
        # training budget is unused; diagnostic.json records the actual zero
        # training/export/search budget of this fixed driver.
        return PilotConfig(self.gpu_uuid, self.source_revision,
                           gpu_seconds=300. if self.profile == "paired_logits_v1" else 600.,
                           cleanup_seconds=60., recipes=("Q_only",))

    def actual_budget(self):
        if self.profile == "placement_v1":
            return dict(max_seconds=600, cleanup_seconds=60, unique_calibration_rows=2000,
                        inference_row_visits_both_runtimes=8000, inference_calls=68,
                        engine_builds=1, sgd=0, exports=0, searches=0, automatic_retries=0)
        if self.profile == "layerwise_v1":
            return dict(max_seconds=600, cleanup_seconds=60, unique_calibration_rows=8,
                        inference_row_visits_both_runtimes=72, inference_calls=28,
                        onnx_instrumentations=1, engine_builds=1, sgd=0, exports=0,
                        searches=0, automatic_retries=0)
        return dict(max_seconds=300, cleanup_seconds=60, calibration_rows=self.rows,
                    sgd=0, exports=0, engine_builds=0, searches=0, automatic_retries=0)


def config_from_payload(value):
    if (type(value) is not dict or set(value) - set(DiagnosticConfig.__dataclass_fields__)
            or not {"source_revision", "artifact_directory"} <= set(value)):
        raise ValueError("Only the fixed diagnostic configuration fields are allowed")
    return DiagnosticConfig(**value)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(value):
    """SHA-256 of NumPy shape/dtype prefix plus contiguous bytes (not _calibration)."""
    import numpy as np
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode())
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _runtime_versions(runtime):
    """Use imported runtime versions; TensorRT may be packaged as tensorrt-cu12."""
    import torch
    import onnxruntime as ort
    versions = dict(torch=str(torch.__version__), onnxruntime=ort.__version__, tensorrt=runtime.trt.__version__)
    if versions != dict(torch="2.2.0+cu121", onnxruntime="1.18.1", tensorrt="10.0.1"):
        raise PilotError("Diagnostic runtime versions differ from the original saved Q-only environment")
    return versions


def artifact_snapshot(config):
    """Check the known rejected raw files without constructing an accepted manifest."""
    root = Path(config.artifact_directory).resolve(strict=True)
    if (root.name != "artifact" or root.parent.name != WORKFLOW or root.parent.parent.name != "Q_only"
            or (root / "manifest.json").exists() or (root / "manifest.json").is_symlink()):
        raise PilotError("Only the saved rejected Q-only directory without a deployment manifest is allowed")
    files = {name: dict(path=str(root / name), sha256=_sha(root / name),
                        bytes=(root / name).stat().st_size) for name in SAVED_SHA256}
    if any(files[name]["sha256"] != expected for name, expected in SAVED_SHA256.items()):
        raise PilotError("Saved Q-only ONNX, engine or inspector hash differs")
    attempt_path, pilot_path = root.parent / "receipt.json", root.parents[3] / "receipt.json"
    if _sha(attempt_path) != ATTEMPT_SHA256 or _sha(pilot_path) != PILOT_SHA256:
        raise PilotError("Original rejected attempt or pilot receipt hash differs")
    attempt = json.loads(container._read(attempt_path))
    pilot = json.loads(container._read(pilot_path))
    candidate = pilot.get("candidates", {}).get("Q_only", {})
    if (attempt.get("status") != "failed" or attempt.get("workflow_fingerprint") != WORKFLOW
            or attempt.get("sgd_attempted") != 200 or attempt.get("sgd_completed") != 200
            or not any(v.get("code") == "TensorRTOutputMismatch"
                       for v in attempt.get("failure", {}).get("violations", []))
            or pilot.get("status") != "incomplete" or candidate.get("workflow") != WORKFLOW
            or candidate.get("outcome") != "StructuralInvalid" or "artifact" in candidate
            or pilot.get("config", {}).get("source_revision") != ORIGINAL_REVISION
            or pilot.get("config", {}).get("split_seed") != 20261102
            or pilot.get("config", {}).get("gpu_uuid") != config.gpu_uuid):
        raise PilotError("Saved receipts do not prove the original rejected Q-only attempt")
    return dict(files=files, attempt_receipt=dict(path=str(attempt_path), sha256=_sha(attempt_path)),
                pilot_receipt=dict(path=str(pilot_path), sha256=_sha(pilot_path)),
                original_status="rejected_no_manifest", workflow_fingerprint=WORKFLOW)


def batch_ranges(rows, batch_size):
    """First two reproduce the original shape; the remainder visits each row once."""
    if (type(rows) is not int or not 2 <= rows <= 2000
            or type(batch_size) is not int or not 1 <= batch_size <= 128):
        raise ValueError("Bounded calibration rows and batch size required")
    return ((0, 2),) + tuple((start, min(start + batch_size, rows)) for start in range(2, rows, batch_size))


def comparison(reference, actual):
    """Retain every logit's errors under the unchanged original tolerance."""
    import numpy as np
    if (not isinstance(reference, np.ndarray) or not isinstance(actual, np.ndarray)
            or reference.dtype != np.float32 or actual.dtype != np.float32
            or reference.ndim != 2 or reference.shape[1] != 10 or reference.shape != actual.shape
            or not reference.size or not np.isfinite(reference).all() or not np.isfinite(actual).all()):
        raise PilotError("Matching finite float32 N-by-10 classifier logits required")
    signed = actual.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(signed)
    bounds = ATOL + RTOL * np.abs(reference.astype(np.float64))
    close = np.isclose(actual, reference, rtol=RTOL, atol=ATOL)
    return dict(cpu_qdq_logits=reference.tolist(), tensorrt_logits=actual.tolist(),
                signed_error=signed.tolist(), abs_error=absolute.tolist(),
                tolerance_bound=bounds.tolist(), normalized_error=(absolute / bounds).tolist(),
                violating=(~close).tolist(), violating_count=int((~close).sum()),
                max_abs_error=float(absolute.max()), mean_abs_error=float(absolute.mean()),
                max_normalized_error=float((absolute / bounds).max()),
                argmax_agree_count=int((actual.argmax(1) == reference.argmax(1)).sum()),
                within_original_tolerance=bool(close.all()))


def run_pairs(calibration, indices, output, *, config, ort_runner, trt_runner, deadline,
              clock=time.monotonic):
    """Small CPU-testable loop; native worker alone supplies the raw runtimes.

    Each durable line contains the original train indices, precise batch shape,
    input hash and both outputs. No labels or held-out role are consumed.
    """
    import numpy as np
    if (not isinstance(calibration, np.ndarray) or calibration.dtype != np.float32
            or calibration.shape != (2000, 3, 32, 32) or not np.isfinite(calibration).all()
            or type(indices) is not tuple or len(indices) != 2000 or indices != tuple(sorted(set(indices)))):
        raise PilotError("The exact ordered 2000-row calibration tensor and train indices are required")
    output = Path(output)
    total_rows, violating, agree, abs_sum, max_abs, max_norm = 0, 0, 0, 0., 0., 0.
    original_probe = None
    with output.open("x", encoding="utf-8") as stream:
        for batch_number, (start, stop) in enumerate(batch_ranges(config.rows, config.batch_size)):
            if clock() >= deadline:
                raise TimeoutError("Original diagnostic worker deadline exhausted; no retry")
            batch = np.ascontiguousarray(calibration[start:stop])
            digest = _array_sha(batch)
            began = clock()
            reference = ort_runner(batch)
            if clock() >= deadline:
                raise TimeoutError("Original diagnostic deadline exhausted after ORT")
            actual = trt_runner(batch)
            if _array_sha(batch) != digest:
                raise PilotError("A diagnostic runtime changed calibration inputs")
            if reference.shape[0] != stop - start:
                raise PilotError("A diagnostic runtime changed batch cardinality")
            evidence = comparison(reference, actual)
            row = dict(schema="heterogeneous-qonly-logit-pairs-v1", batch_number=batch_number,
                       original_first_two_probe=batch_number == 0, calibration_positions=list(range(start, stop)),
                       train_indices=list(indices[start:stop]), input_shape=list(batch.shape),
                       input_dtype=str(batch.dtype), input_sha256=digest, input_bytes=batch.nbytes,
                       output_shape=list(reference.shape), reference_sha256=_array_sha(reference),
                       actual_sha256=_array_sha(actual), began_at=began, ended_at=clock(),
                       backend_rtol=RTOL, backend_atol=ATOL, **evidence)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
            count = stop - start
            total_rows += count; violating += evidence["violating_count"]; agree += evidence["argmax_agree_count"]
            abs_sum += evidence["mean_abs_error"] * count * 10
            max_abs = max(max_abs, evidence["max_abs_error"])
            max_norm = max(max_norm, evidence["max_normalized_error"])
            if batch_number == 0:
                original_probe = dict(batch_number=0, rows=2, input_sha256=digest,
                                      within_original_tolerance=evidence["within_original_tolerance"],
                                      max_abs_error=evidence["max_abs_error"],
                                      violating_count=evidence["violating_count"])
            if clock() >= deadline:
                raise TimeoutError("Original diagnostic deadline exhausted after durable batch publication")
    return dict(unique_calibration_rows=total_rows, inference_row_visits_per_runtime=total_rows,
                inference_row_visits_both_runtimes=2 * total_rows, batches=len(batch_ranges(config.rows, config.batch_size)),
                logits_per_runtime=total_rows * 10, violating_count=violating,
                max_abs_error=max_abs, mean_abs_error=abs_sum / (total_rows * 10),
                max_normalized_error=max_norm, argmax_agreement=agree / total_rows,
                within_original_tolerance=violating == 0, original_first_two_probe=original_probe,
                pairs_sha256=_sha(output))


def _diagnostic_config(root, plan):
    raw = container._read(Path(root) / "diagnostic.json", binary=True)
    seal = json.loads(container._read(Path(root) / "diagnostic-seal.json"))
    if seal != dict(token=plan.token, sha256=hashlib.sha256(raw).hexdigest()):
        raise PilotError("Frozen diagnostic config differs from its launch seal")
    cfg = config_from_payload(json.loads(raw))
    if asdict(cfg.control_config()) != asdict(container.pilot_config_from_payload(plan.config)):
        raise PilotError("Diagnostic config differs from the exact container boundary plan")
    return cfg, seal["sha256"]


def _admit_worker(plan, cfg, *, preflight=None):
    root = Path(plan.run_directory)
    control = cfg.control_config()
    container._protect_parent(container.local_process(os.getppid()))
    wait_for_start_gate(root / "worker-start.json", timeout=10.)
    boundary = container._boundary(root, control, plan.started_at, "worker")
    identity = container.local_process(os.getpid())
    if json.loads(container._read(root / "worker-start.json")) != dict(
            token=plan.token, pid=identity.pid, birth_ticks=identity.birth_ticks):
        raise PilotError("Worker start gate names a different lifetime")
    deadline = plan.started_at + control.gpu_seconds - control.cleanup_seconds
    if preflight is not None:
        preflight()
    if time.monotonic() >= deadline:
        raise TimeoutError("Original diagnostic deadline exhausted before CUDA binding")
    from .heterogeneous_device import require_cuda_binding
    binding = require_cuda_binding(cfg.gpu_uuid)
    container._publish(root / "worker-ready.json", dict(token=plan.token, pid=identity.pid,
        birth_ticks=identity.birth_ticks, gpu_uuid=cfg.gpu_uuid, actual_gpu_uuid=binding["actual_uuid"].lower()))
    handshake_end = min(deadline, time.monotonic() + container.HANDSHAKE_SECONDS)
    while not (root / "worker-go.json").is_file():
        if time.monotonic() >= handshake_end:
            raise TimeoutError("Guard did not admit the sole raw GPU row")
        time.sleep(.05)
    gate = json.loads(container._read(root / "worker-go.json"))
    if (gate.get("token") != plan.token or gate.get("host_pid_identity_proven") is not False
            or gate.get("gpu_provenance") != container.PROVENANCE):
        raise PilotError("Missing explicit correlation-only guard admission")
    boundary = container._boundary(root, control, plan.started_at, "worker")
    return binding, boundary, deadline


def run_worker(plan):
    root = Path(plan.run_directory)
    cfg, config_sha = _diagnostic_config(root, plan)
    if cfg.profile == "placement_v1":
        from .heterogeneous_placement_admission import placement_snapshot, run_placement_worker
        frozen = {}
        binding, boundary, deadline = _admit_worker(plan, cfg, preflight=lambda: frozen.update(placement_snapshot(cfg)))
        return run_placement_worker(plan, cfg, binding, boundary, deadline, config_sha, frozen)
    binding, boundary, deadline = _admit_worker(plan, cfg)
    if cfg.profile == "layerwise_v1":
        from .heterogeneous_layerwise_runtime import run_layerwise_worker
        return run_layerwise_worker(plan, cfg, binding, boundary, config_sha)
    status, failure, summary, before, after, versions = "failed", None, None, None, None, {}
    original_probe_fingerprint = None
    try:
        before = artifact_snapshot(cfg)
        from .heterogeneous_data import CifarTrainingData
        from fedcore.algorithm.low_rank.cv_execution import _calibration
        from fedcore.algorithm.heterogeneous.backend import TensorRTDeployment, _ort_session
        data = CifarTrainingData(plan.data_root, split_seed=20261102)
        calibration = data.calibration_tensor()
        if (data.fingerprint != DATA_SHA256 or data.roles.sha256 != ROLES_SHA256
                or _calibration(calibration) != CALIBRATION_SHA256):
            raise PilotError("Saved training corpus, split or normalized calibration identity differs")
        original_probe_fingerprint = _calibration(calibration[:2].contiguous())
        if time.monotonic() >= deadline:
            raise TimeoutError("Original diagnostic deadline exhausted before runtime loading")
        # This is an in-memory runtime boundary, not a deployment manifest.
        # The raw constructor itself re-verifies the actual CUDA UUID.
        runtime_boundary = dict(diagnostic_only=True, backend_spec=dict(device_uuid=binding["declared_uuid"], max_batch_size=128))
        session = _ort_session(Path(cfg.artifact_directory) / "model.onnx")
        runtime = TensorRTDeployment(Path(cfg.artifact_directory) / "model.engine", runtime_boundary)
        versions = _runtime_versions(runtime)
        summary = run_pairs(calibration.numpy(), data.roles.calibration, root / "logit-pairs.jsonl", config=cfg,
            ort_runner=lambda value: session.run(None, {"images": value})[0], trt_runner=runtime,
            deadline=deadline)
        if _calibration(calibration) != CALIBRATION_SHA256:
            raise PilotError("Calibration identity changed during inference")
        after = artifact_snapshot(cfg)
        if before != after:
            raise PilotError("Saved rejected artifacts or receipts changed during inference")
        container._boundary(root, control, plan.started_at, "worker")
        status = "complete"
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error))
    receipt = dict(schema="heterogeneous-qonly-diagnostic-v1", status=status, failure=failure,
        config=asdict(cfg), diagnostic_config_sha256=config_sha, started_at=plan.started_at,
        worker_deadline=deadline, hard_deadline=plan.started_at + 300., ended_at=time.monotonic(),
        backend_rtol=RTOL, backend_atol=ATOL, cuda_binding=binding, boundary=boundary,
        artifacts_before=before, artifacts_after=after, data_sha256=DATA_SHA256,
        roles_sha256=ROLES_SHA256, calibration_fingerprint=CALIBRATION_SHA256, split_seed=20261102,
        original_first_two_probe_fingerprint=original_probe_fingerprint,
        array_sha256_encoding="SHA256(str((tuple(shape), str(numpy_dtype))).encode() + contiguous_array_bytes)",
        calibration_fingerprint_encoding="existing cv_execution._calibration with torch dtype prefix",
        summary=summary, versions=versions, module_sha256=_sha(__file__),
        container_module_sha256=_sha(container.__file__), script_sha256=_sha(plan.script), python=sys.version,
        original_artifact_status="rejected_no_manifest", deployment_accepted=False,
        sgd_attempted=0, sgd_completed=0, exports=0, engine_builds=0, searches=0,
        Vsearch_opened=False, Vreport_opened=False, official_test_opened=False, labels_evaluated=False,
        automatic_retries=0, host_pid_identity_proven=False,
        guard_budget_semantics="PilotConfig budget is boundary transport only; actual diagnostic SGD/export/search budget is zero",
        interpretation="read-only calibration numerical diagnosis; original rejection unchanged; no quality or latency claim")
    container._publish(root / "diagnostic-receipt.json", receipt)
    return 0 if status == "complete" else 1


def launch(*, config_path, data_root, run_directory):
    """One fresh fixed-profile window under the unchanged GNU-timeout controls."""
    started_at = time.monotonic()
    cfg = config_from_payload(json.loads(container._read(config_path)))
    control = cfg.control_config()
    hard_deadline = started_at + control.gpu_seconds
    worker_deadline = hard_deadline - control.cleanup_seconds
    if sys.platform != "linux":
        raise PilotError("Diagnostic launch requires Linux and GNU timeout")
    version = subprocess.run((container.TIMEOUT, "--version"), capture_output=True, text=True, check=True, timeout=2).stdout
    if not version.startswith("timeout (GNU coreutils)"):
        raise PilotError("Only the explicitly provisioned GNU timeout executable is supported")
    cwd = Path(__file__).resolve().parents[2]
    head = subprocess.run(("git", "rev-parse", "HEAD"), cwd=cwd, capture_output=True, text=True, check=True, timeout=3).stdout.strip()
    dirty = subprocess.run(("git", "status", "--porcelain"), cwd=cwd, capture_output=True, text=True, check=True, timeout=3).stdout
    if head != cfg.source_revision or dirty.strip():
        raise PilotError("Clean source checkout must match the frozen diagnostic revision")
    before = artifact_snapshot(cfg)
    placement = None
    if cfg.profile == "placement_v1":
        from .heterogeneous_placement_admission import placement_snapshot
        placement = placement_snapshot(cfg)
    data, root = Path(data_root).resolve(strict=True), Path(run_directory).resolve()
    artifact = Path(cfg.artifact_directory).resolve(strict=True)
    if not data.is_dir() or root == artifact or root in artifact.parents or artifact in root.parents:
        raise PilotError("Existing local CIFAR directory and separate fresh output directory required")
    if placement is not None and root in Path(cfg.candidate_onnx).resolve().parents:
        raise PilotError("The prepared candidate must be separate from the fresh launch directory")
    root.mkdir(parents=True, exist_ok=False)
    container._publish(root / "config.json", asdict(control))
    payload = asdict(cfg)
    if cfg.candidate_onnx is None:
        payload.pop("candidate_onnx")
    container._publish(root / "diagnostic.json", payload)
    compiled_at = time.monotonic()
    duration = math.floor((hard_deadline - compiled_at - container.HARD_RESERVE_SECONDS
                           - container.LAUNCH_RESERVE_SECONDS) * 1000) / 1000
    plan = container.ContainerPlan(started_at, compiled_at, duration, secrets.token_hex(16),
        str(Path(sys.executable).absolute()), str(cwd / "scripts" / SCRIPT), str(root / "config.json"),
        str(data), str(root), json.loads(json.dumps(asdict(control))))
    container._publish(root / "diagnostic-seal.json", dict(token=plan.token, sha256=_sha(root / "diagnostic.json")))
    receipt = dict(artifacts=before, actual_budget=cfg.actual_budget(),
                   diagnostic_config_sha256=_sha(root / "diagnostic.json"))
    if placement is not None:
        receipt["placement"] = placement
    container._publish(root / "diagnostic-launch.json", receipt)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=cfg.gpu_uuid,
        FEDCORE_PILOT_ABSOLUTE_DEADLINE=str(worker_deadline))
    with (root / "console.log").open("x", encoding="utf-8") as console:
        child = subprocess.Popen(plan.timeout_argv, stdin=subprocess.DEVNULL, shell=False,
            stdout=console, stderr=subprocess.STDOUT, start_new_session=True, env=environment, cwd=cwd)
    try:
        captured = container.local_process(child.pid)
        end = min(hard_deadline, time.monotonic() + 2.)
        while (captured.argv != plan.timeout_argv or captured.pgid != captured.pid
                or not container._timer_is_present(captured.pid)):
            if time.monotonic() >= end or child.poll() is not None:
                raise PilotError("GNU timeout failed to establish its private group")
            time.sleep(.01); captured = container.local_process(child.pid)
        if time.monotonic() + duration > hard_deadline - container.HARD_RESERVE_SECONDS:
            raise PilotError("Timeout submission consumed its original-window launch reserve")
        container._publish(root / "os-boundary.json", dict(schema="fedcore-container-timeout-v1",
            plan=asdict(plan), timeout=asdict(captured)))
        container._event(root, "diagnostic_launched", timeout_pid=child.pid, max_seconds=control.gpu_seconds,
            actual_sgd_budget=0, gpu_provenance=container.PROVENANCE, host_pid_identity_proven=False)
        return dict(run_directory=str(root), timeout_pid=child.pid, started_at=started_at,
                    deadline_at=hard_deadline, automatic_retries=0)
    except Exception:
        child.kill(); child.wait(timeout=2)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("launch", "supervisor", "guard", "worker"), default="launch")
    for name in ("config", "data-root", "run-directory"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--started-at", type=float)
    parser.add_argument("--token")
    args = parser.parse_args(argv)
    if args.mode == "launch":
        if args.started_at is not None or args.token is not None:
            parser.error("Launch creates its original clock and token")
        print(json.dumps(launch(config_path=args.config, data_root=args.data_root,
                                run_directory=args.run_directory), allow_nan=False))
        return 0
    wait_for_start_gate(Path(args.run_directory) / "os-boundary.json", timeout=10.)
    plan, _ = container._load_plan(args.run_directory)
    if (args.started_at != plan.started_at or args.token != plan.token or args.config != plan.config_path
            or args.data_root != plan.data_root or args.run_directory != plan.run_directory
            or plan.script != str(Path(__file__).resolve().parents[2] / "scripts" / SCRIPT)):
        raise PilotError("Diagnostic child command differs from the fixed launch record")
    cfg, _ = _diagnostic_config(args.run_directory, plan)
    if args.mode == "guard" and cfg.profile == "placement_v1":
        return container.run_guard(plan, require_terminal_handoff=True)
    if args.mode == "supervisor" and cfg.profile == "placement_v1":
        return container.run_supervisor(plan, require_clean_exit=True)
    return {"supervisor": container.run_supervisor, "guard": container.run_guard, "worker": run_worker}[args.mode](plan)
