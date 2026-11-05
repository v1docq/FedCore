"""CPU contracts only; no CUDA initialization or saved-runtime execution."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fedcore.experiments import heterogeneous_qonly_diagnostic as diagnostic
from fedcore.experiments.heterogeneous_pilot import PilotError


def config(**kwargs):
    return diagnostic.DiagnosticConfig("a" * 40, "/saved/Q_only/" + diagnostic.WORKFLOW + "/artifact", **kwargs)


def test_config_round_trip_and_frozen_zero_work_control_window():
    cfg = config()
    assert diagnostic.config_from_payload(json.loads(json.dumps(asdict(cfg)))) == cfg
    control = cfg.control_config()
    assert control.gpu_seconds == 300. and control.cleanup_seconds == 60.
    assert control.split_seed == 20261102 and control.recipes == ("Q_only",)


def test_diagnostic_plan_keeps_exact_roles_and_original_300s_timeout_ceiling():
    control = config().control_config()
    plan = diagnostic.container.ContainerPlan(100., 101., 286., "b" * 32,
        "/env/bin/python", "/repo/scripts/" + diagnostic.SCRIPT, "/run/config.json", "/data", "/run",
        json.loads(json.dumps(asdict(control))))
    assert plan.timeout_argv[:3] == ("/usr/bin/timeout", "--signal=KILL", "286.000s")
    assert plan.command("worker")[:5] == ("/env/bin/python", "-u", "/repo/scripts/" + diagnostic.SCRIPT, "--mode", "worker")
    with pytest.raises(ValueError):
        replace(plan, timeout_seconds=286.001)


@pytest.mark.parametrize("field,value", [
    ("rows", 2001), ("rows", 1), ("rows", True), ("rows", 2.),
    ("batch_size", 129), ("batch_size", 0), ("batch_size", True),
    ("gpu_uuid", "GPU-12345678-1234-1234-1234-123456789abc"),
    ("artifact_directory", "relative"), ("artifact_directory", "/saved\nworker"),
    ("source_revision", "a" * 39),
])
def test_unapproved_controls_rejected(field, value):
    with pytest.raises(ValueError):
        diagnostic.config_from_payload({**asdict(config()), field: value})


@pytest.mark.parametrize("extra", ["gpu_seconds", "cleanup_seconds", "backend_rtol", "backend_atol", "split_seed", "export", "worker"])
def test_no_time_tolerance_data_or_worker_override(extra):
    with pytest.raises(ValueError):
        diagnostic.config_from_payload({**asdict(config()), extra: 123})


def test_missing_fields_fail_as_config_validation():
    with pytest.raises(ValueError):
        diagnostic.config_from_payload({"rows": 2000})


def test_versions_use_loaded_tensorrt_even_without_distribution_metadata(monkeypatch):
    import sys
    import importlib.metadata
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="2.2.0+cu121"))
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(__version__="1.18.1"))
    monkeypatch.setattr(importlib.metadata, "version", lambda *a: pytest.fail("TensorRT distribution name is not portable"))
    runtime = SimpleNamespace(trt=SimpleNamespace(__version__="10.0.1"))
    assert diagnostic._runtime_versions(runtime)["tensorrt"] == "10.0.1"
    runtime.trt.__version__ = "10.1.0"
    with pytest.raises(PilotError, match="runtime versions differ"):
        diagnostic._runtime_versions(runtime)


@pytest.mark.parametrize("rows,size", [(2, 1), (3, 128), (130, 128), (1999, 7), (2000, 128)])
def test_batching_preserves_original_probe_and_visits_every_requested_row_once(rows, size):
    spans = diagnostic.batch_ranges(rows, size)
    assert spans[0] == (0, 2)
    assert [i for start, stop in spans for i in range(start, stop)] == list(range(rows))
    assert all(1 <= stop - start <= size for start, stop in spans[1:])
    if rows == 2000 and size == 128:
        assert spans[-1] == (1922, 2000) and len(spans) == 17


def test_per_logit_evidence_uses_asymmetric_original_reference_tolerance():
    reference = np.zeros((2, 10), dtype=np.float32)
    reference[0, 1] = 10
    actual = reference.copy()
    actual[0, 0] = .021  # fails the near-zero bound
    actual[0, 1] += .21  # passes the .22 bound
    value = diagnostic.comparison(reference, actual)
    assert value["violating_count"] == 1 and value["violating"][0][0]
    assert not value["violating"][0][1]
    assert value["cpu_qdq_logits"] == reference.tolist()
    assert value["tensorrt_logits"] == actual.tolist()
    assert value["normalized_error"][0][0] == pytest.approx(1.05)
    assert value["abs_error"][0][1] == pytest.approx(.21, abs=1e-6)
    assert value["tolerance_bound"][0][1] == pytest.approx(.22)
    assert not value["within_original_tolerance"]


@pytest.mark.parametrize("bad", [np.zeros((1, 10), np.float32), np.zeros((2, 9), np.float32),
                                 np.full((2, 10), np.nan, np.float32), np.zeros((2, 10), np.float64)])
def test_invalid_runtime_output_never_becomes_comparison_evidence(bad):
    with pytest.raises(PilotError):
        diagnostic.comparison(np.zeros((2, 10), np.float32), bad)


def calibration():
    value = np.zeros((2000, 3, 32, 32), np.float32)
    value[:, 0, 0, 0] = np.arange(2000)
    return value


def test_public_pair_loop_saves_first2_and_all2000_exactly_once_with_correct_global_aggregates(tmp_path):
    tensor = calibration()
    calls = {"ort": [], "trt": []}
    def run(name):
        def call(batch):
            calls[name].extend(batch[:, 0, 0, 0].astype(int).tolist())
            output = np.zeros((len(batch), 10), np.float32)
            if name == "trt":
                output[:, 1] = .03
            return output
        return call
    pairs = tmp_path / "pairs.jsonl"
    summary = diagnostic.run_pairs(tensor, tuple(range(2000)), pairs, config=config(),
        ort_runner=run("ort"), trt_runner=run("trt"), deadline=100., clock=lambda: 1.)
    rows = [json.loads(line) for line in pairs.read_text().splitlines()]
    assert calls == dict(ort=list(range(2000)), trt=list(range(2000)))
    assert rows[0]["input_shape"] == [2, 3, 32, 32]
    assert rows[0]["original_first_two_probe"] is True
    assert rows[0]["calibration_positions"] == [0, 1]
    assert rows[1]["calibration_positions"] == list(range(2, 130))
    assert rows[-1]["train_indices"] == list(range(1922, 2000))
    assert summary["unique_calibration_rows"] == summary["inference_row_visits_per_runtime"] == 2000
    assert summary["inference_row_visits_both_runtimes"] == 4000
    assert summary["violating_count"] == 2000 and summary["argmax_agreement"] == 0.
    assert summary["mean_abs_error"] == pytest.approx(.003)
    assert summary["pairs_sha256"] == diagnostic._sha(pairs)
    assert np.array_equal(tensor[:, 0, 0, 0], np.arange(2000))
    with pytest.raises(FileExistsError):
        diagnostic.run_pairs(tensor, tuple(range(2000)), pairs, config=config(),
            ort_runner=run("ort"), trt_runner=run("trt"), deadline=100., clock=lambda: 1.)


def test_deadline_preserves_completed_batch_and_never_runs_second_batch(tmp_path):
    now = [1.]
    calls = []
    def runtime(batch):
        calls.append(len(batch))
        return np.zeros((len(batch), 10), np.float32)
    def clock():
        return now[0]
    def trt(batch):
        result = runtime(batch)
        now[0] = 100.
        return result
    pairs = tmp_path / "pairs.jsonl"
    with pytest.raises(TimeoutError):
        diagnostic.run_pairs(calibration(), tuple(range(2000)), pairs, config=config(),
            ort_runner=runtime, trt_runner=trt, deadline=100., clock=clock)
    assert calls == [2, 2]
    assert len(pairs.read_text().splitlines()) == 1
    assert json.loads(pairs.read_text())["ended_at"] == 100.


def test_input_mutation_stops_before_publication(tmp_path):
    def ort(batch):
        batch[0, 0, 0, 0] += 1
        return np.zeros((len(batch), 10), np.float32)
    with pytest.raises(PilotError, match="changed calibration"):
        diagnostic.run_pairs(calibration(), tuple(range(2000)), tmp_path / "pairs", config=config(rows=2),
            ort_runner=ort, trt_runner=lambda x: np.zeros((len(x), 10), np.float32),
            deadline=100., clock=lambda: 1.)


def saved_tree(tmp_path, monkeypatch):
    pilot = tmp_path / "pilot"
    artifact = pilot / "recipes" / "Q_only" / diagnostic.WORKFLOW / "artifact"
    artifact.mkdir(parents=True)
    for name in diagnostic.SAVED_SHA256:
        (artifact / name).write_bytes(name.encode())
    attempt = dict(status="failed", workflow_fingerprint=diagnostic.WORKFLOW, sgd_attempted=200, sgd_completed=200,
                   failure=dict(violations=[dict(code="TensorRTOutputMismatch")]))
    (artifact.parent / "receipt.json").write_text(json.dumps(attempt))
    original = dict(status="incomplete", config=dict(source_revision=diagnostic.ORIGINAL_REVISION,
        split_seed=20261102, gpu_uuid=diagnostic.GPU_UUID), candidates=dict(Q_only=dict(
            workflow=diagnostic.WORKFLOW, outcome="StructuralInvalid")))
    (pilot / "receipt.json").write_text(json.dumps(original))
    monkeypatch.setattr(diagnostic, "SAVED_SHA256", {name: diagnostic._sha(artifact / name) for name in diagnostic.SAVED_SHA256})
    monkeypatch.setattr(diagnostic, "ATTEMPT_SHA256", diagnostic._sha(artifact.parent / "receipt.json"))
    monkeypatch.setattr(diagnostic, "PILOT_SHA256", diagnostic._sha(pilot / "receipt.json"))
    return replace(config(), artifact_directory=str(artifact))


def test_saved_artifact_snapshot_requires_original_rejection_and_never_writes_manifest(tmp_path, monkeypatch):
    cfg = saved_tree(tmp_path, monkeypatch)
    before = diagnostic.artifact_snapshot(cfg)
    assert before["original_status"] == "rejected_no_manifest"
    assert not (Path(cfg.artifact_directory) / "manifest.json").exists()
    assert diagnostic.artifact_snapshot(cfg) == before
    (Path(cfg.artifact_directory) / "model.engine").write_bytes(b"changed")
    with pytest.raises(PilotError, match="hash differs"):
        diagnostic.artifact_snapshot(cfg)


def test_existing_deployment_manifest_blocks_raw_diagnostic(tmp_path, monkeypatch):
    cfg = saved_tree(tmp_path, monkeypatch)
    (Path(cfg.artifact_directory) / "manifest.json").write_text("{}")
    with pytest.raises(PilotError, match="without a deployment manifest"):
        diagnostic.artifact_snapshot(cfg)


def test_launch_rejects_non_gnu_timeout_before_artifact_read_or_spawn(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic.sys, "platform", "linux")
    monkeypatch.setattr(diagnostic.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="timeout (uutils coreutils)\n"))
    monkeypatch.setattr(diagnostic.subprocess, "Popen", lambda *a, **kw: pytest.fail("Unbounded launch"))
    config_path = tmp_path / "diagnostic.json"
    config_path.write_text(json.dumps(asdict(config())))
    with pytest.raises(PilotError, match="GNU timeout"):
        diagnostic.launch(config_path=config_path, data_root="/unused", run_directory=tmp_path / "never")
    assert not (tmp_path / "never").exists()
