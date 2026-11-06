"""One-build calibration-only comparison of the two frozen Q-only graphs."""
from pathlib import Path
import hashlib
import json
import os
import time

from . import heterogeneous_container as container
from . import heterogeneous_qonly_diagnostic as qonly
from .heterogeneous_pilot import PilotError


FIRST_BATCH = (
    "bc7ff2f72fc7ded4d71f16c9715946eef63a73f72621b6159e8a3eb2f91fa69d",
    "cab6cf4ff824173877e9fdd768e1b0698bb89af8d61764dae923f9617953e723",
    "341bf0e652b29f26069eaf688d5f1fdfa9671250b9d99dc234186af25b403b68",
)


def placement_snapshot(config):
    from .heterogeneous_qonly_placement import verify_diagnostic_graph

    artifact = Path(config.artifact_directory)
    return verify_diagnostic_graph(artifact / "model.onnx", artifact / "model.engine", config.candidate_onnx)


def _stats():
    return dict(violations=0, max_abs_error=0., sum_abs_error=0., max_normalized_error=0., argmax_agree=0)


def _add_stats(stats, evidence, size):
    stats["violations"] += evidence["violating_count"]
    stats["max_abs_error"] = max(stats["max_abs_error"], evidence["max_abs_error"])
    stats["sum_abs_error"] += evidence["mean_abs_error"] * size * 10
    stats["max_normalized_error"] = max(stats["max_normalized_error"], evidence["max_normalized_error"])
    stats["argmax_agree"] += evidence["argmax_agree_count"]


def run_admission(calibration, indices, *, original_ort, original_trt, candidate_ort, build_candidate,
                  record, accounting, deadline, clock=time.monotonic):
    """Count the four runtimes and gate the sole build on a frozen baseline replay."""
    import numpy as np

    if (not isinstance(calibration, np.ndarray) or calibration.dtype != np.float32
            or calibration.shape != (2000, 3, 32, 32) or not np.isfinite(calibration).all()
            or type(indices) is not tuple or len(indices) != 2000 or indices != tuple(sorted(set(indices)))):
        raise PilotError("The frozen ordered 2000-row calibration input is required")

    def check_deadline():
        if clock() >= deadline:
            raise TimeoutError("Placement diagnostic reached its original worker deadline")

    def infer(name, runtime):
        check_deadline()
        accounting["calls_started"] += 1
        accounting["rows_started"] += len(batch)
        record("inference_started", runtime=name, batch=batch_number, rows=len(batch), input_sha256=input_sha)
        logits = runtime(batch)
        if qonly._array_sha(batch) != input_sha:
            raise PilotError("Inference changed a calibration input")
        if (not isinstance(logits, np.ndarray) or logits.dtype != np.float32
                or logits.shape != (len(batch), 10) or not np.isfinite(logits).all()):
            raise PilotError("Inference must return finite float32 classifier logits")
        accounting["calls_completed"] += 1
        accounting["rows_completed"] += len(batch)
        record("inference_completed", runtime=name, batch=batch_number, logits_sha256=qonly._array_sha(logits),
               logits=logits.tolist())
        check_deadline()
        return logits

    stats = {name: _stats() for name in ("original_trt_vs_ort", "candidate_trt_vs_ort", "ort_candidate_vs_original")}
    candidate_trt = None
    rows = 0
    for batch_number, (start, stop) in enumerate(qonly.batch_ranges(2000, 128)):
        batch = np.ascontiguousarray(calibration[start:stop]).copy()
        input_sha = qonly._array_sha(batch)
        record("batch_started", batch=batch_number, positions=[start, stop], train_indices=list(indices[start:stop]))
        baseline_ort = infer("original_ort", original_ort)
        baseline_trt = infer("original_trt", original_trt)
        standard_pair = qonly.comparison(baseline_ort, baseline_trt)
        if batch_number == 0 and ((input_sha, qonly._array_sha(baseline_ort),
                                  qonly._array_sha(baseline_trt)) != FIRST_BATCH
                                  or standard_pair["violating_count"] != 6):
            raise PilotError("The original two-row Q-only failure did not reproduce bitwise")
        alternative_ort = infer("candidate_ort", candidate_ort)
        if batch_number == 0:
            check_deadline()
            accounting["engine_builds_started"] += 1
            record("engine_build_started")
            candidate_trt = build_candidate()
            accounting["engine_loads_verified"] += 1
            record("engine_load_verified")
        alternative_trt = infer("candidate_trt", candidate_trt)
        comparisons = dict(original_trt_vs_ort=standard_pair,
            candidate_trt_vs_ort=qonly.comparison(alternative_ort, alternative_trt),
            ort_candidate_vs_original=qonly.comparison(baseline_ort, alternative_ort))
        record("batch_compared", batch=batch_number, comparisons={
            name: {key: value for key, value in evidence.items() if key not in ("cpu_qdq_logits", "tensorrt_logits")}
            for name, evidence in comparisons.items()})
        rows += stop - start
        for name, evidence in comparisons.items():
            _add_stats(stats[name], evidence, stop - start)
        if batch_number == 0 and comparisons["candidate_trt_vs_ort"]["violating_count"]:
            break
    for entry in stats.values():
        entry["mean_abs_error"] = entry.pop("sum_abs_error") / (rows * 10)
        entry["argmax_agreement"] = entry.pop("argmax_agree") / rows
    return dict(unique_rows=rows, comparisons=stats,
                numeric_gate_passed=rows == 2000 and stats["candidate_trt_vs_ort"]["violations"] == 0)


def run_placement_worker(plan, config, binding, boundary, deadline, config_sha, frozen):
    from .heterogeneous_data import CifarTrainingData
    from fedcore.algorithm.heterogeneous.backend import (
        BackendSpec, TensorRTDeployment, _compile_tensorrt, _ort_session,
    )
    from fedcore.algorithm.low_rank.cv_execution import _calibration

    root = Path(plan.run_directory)
    output = root / "candidate-engine"
    accounting = dict(calls_started=0, calls_completed=0, rows_started=0, rows_completed=0,
                      engine_builds_started=0, engine_loads_verified=0)
    failure, summary, built, before, after, versions = None, None, {}, None, None, {}
    identity = container.local_process(os.getpid())

    def record(event, **fields):
        with (root / "placement-progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(at=time.monotonic(), event=event, **fields), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def ort_runner(path):
        session = _ort_session(path)
        return lambda batch: session.run(None, {"images": batch})[0]

    try:
        before = qonly.artifact_snapshot(config)
        data = CifarTrainingData(plan.data_root, split_seed=20261102)
        calibration = data.calibration_tensor()
        if (data.fingerprint != qonly.DATA_SHA256 or data.roles.sha256 != qonly.ROLES_SHA256
                or _calibration(calibration) != qonly.CALIBRATION_SHA256):
            raise PilotError("Saved calibration input or split identity differs")
        source_candidate = Path(config.candidate_onnx)
        raw = source_candidate.read_bytes()
        if hashlib.sha256(raw).hexdigest() != frozen["candidate_onnx_sha256"]:
            raise PilotError("Prepared ONNX changed before use")
        candidate = root / source_candidate.name
        with candidate.open("xb") as stream:
            stream.write(raw)
        if time.monotonic() >= deadline:
            raise TimeoutError("Placement deadline reached before loading the runtimes")
        runtime_boundary = dict(diagnostic_only=True, backend_spec=dict(
            device_uuid=binding["declared_uuid"], max_batch_size=128))
        original_trt = TensorRTDeployment(Path(config.artifact_directory) / "model.engine", runtime_boundary)
        versions = qonly._runtime_versions(original_trt)
        def build_candidate():
            if time.monotonic() >= deadline:
                raise TimeoutError("Placement deadline reached before the sole build")
            container._boundary(root, config.control_config(), plan.started_at, "worker")
            if qonly._sha(candidate) != frozen["candidate_onnx_sha256"]:
                raise PilotError("Prepared ONNX changed before build")
            output.mkdir()
            path, inspection = _compile_tensorrt(candidate, output, BackendSpec(
                runtime="tensorrt", device_uuid=binding["declared_uuid"]),
                tuple(frozen["selected_conv_nodes"].values()))
            built.update(engine_sha256=qonly._sha(path), engine_bytes=path.stat().st_size, **inspection)
            return TensorRTDeployment(path, runtime_boundary)

        summary = run_admission(calibration.numpy(), data.roles.calibration,
            original_ort=ort_runner(Path(config.artifact_directory) / "model.onnx"), original_trt=original_trt,
            candidate_ort=ort_runner(candidate), build_candidate=build_candidate, record=record,
            accounting=accounting, deadline=deadline)
        after = qonly.artifact_snapshot(config)
        if before != after or any(qonly._sha(path) != frozen["candidate_onnx_sha256"]
                                  for path in (candidate, source_candidate)):
            raise PilotError("Frozen source or candidate bytes changed during evaluation")
        if _calibration(calibration) != qonly.CALIBRATION_SHA256:
            raise PilotError("Frozen calibration input changed during evaluation")
        container._boundary(root, config.control_config(), plan.started_at, "worker")
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error))
    engine_files = {}
    for name in ("model.engine", "engine-inspector.json"):
        path = output / name
        if path.is_file():
            engine_files[name] = dict(sha256=qonly._sha(path), bytes=path.stat().st_size)
    container._publish(root / "diagnostic-receipt.json", dict(
        schema="heterogeneous-qonly-placement-admission-v1", status="complete" if failure is None else "failed",
        failure=failure, summary=summary, accounting=accounting, candidate_engine=built, engine_files=engine_files,
        cuda_binding=binding, boundary=boundary, config_sha256=config_sha,
        started_at=plan.started_at, worker_deadline=deadline, hard_deadline=plan.started_at + 600.,
        ended_at=time.monotonic(), actual_budget=config.actual_budget(), graph=frozen,
        artifacts_before=before, artifacts_after=after, versions=versions,
        backend_atol=qonly.ATOL, backend_rtol=qonly.RTOL, deployment_manifest_written=False))
    container._publish(root / "worker-terminal.json", dict(
        token=plan.token, pid=identity.pid, birth_ticks=identity.birth_ticks))
    return 0 if failure is None else 1
