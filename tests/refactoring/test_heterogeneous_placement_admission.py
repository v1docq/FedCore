from types import SimpleNamespace
from dataclasses import asdict
from pathlib import Path
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from fedcore.experiments import heterogeneous_container as container
from fedcore.experiments import heterogeneous_placement_admission as placement
from fedcore.experiments import heterogeneous_qonly_diagnostic as qonly
from fedcore.experiments.heterogeneous_pilot import PilotConfig, PilotError


@pytest.mark.parametrize("case,expected", [("pass", (68, 8000, 1, True)),
    ("early_difference", (4, 8, 1, False)), ("late_difference", (68, 8000, 1, False))])
def test_fixed_admission_accounting_and_pairing(monkeypatch, case, expected):
    calibration = np.zeros((2000, 3, 32, 32), dtype=np.float32)
    accounting = dict(calls_started=0, calls_completed=0, rows_started=0, rows_completed=0,
                      engine_builds_started=0, engine_loads_verified=0)
    events = []

    def original_ort(value):
        return np.zeros((len(value), 10), dtype=np.float32)

    def original_trt(value):
        logits = original_ort(value)
        logits[0, :6] += .1
        return logits

    def candidate_ort(value):
        return original_ort(value) + .5

    def candidate_trt(value):
        logits = candidate_ort(value)
        if (case == "early_difference" and len(value) == 2) or (case == "late_difference" and len(value) == 78):
            logits[0, 0] += .1
        return logits

    first = calibration[:2]
    monkeypatch.setattr(placement, "FIRST_BATCH", tuple(map(qonly._array_sha,
        (first, original_ort(first), original_trt(first)))))
    summary = placement.run_admission(calibration, tuple(range(2000)),
        original_ort=original_ort, original_trt=original_trt, candidate_ort=candidate_ort,
        build_candidate=lambda: candidate_trt,
        record=lambda name, **data: events.append((name, data.get("runtime"))), accounting=accounting,
        deadline=1., clock=lambda: 0.)
    calls, rows, builds, accepted = expected
    assert accounting == dict(calls_started=calls, calls_completed=calls, rows_started=rows, rows_completed=rows,
                              engine_builds_started=builds, engine_loads_verified=builds)
    assert summary["numeric_gate_passed"] is accepted
    assert summary["comparisons"]["ort_candidate_vs_original"]["max_abs_error"] == .5
    assert events[:9] == [
        ("batch_started", None), ("inference_started", "original_ort"), ("inference_completed", "original_ort"),
        ("inference_started", "original_trt"), ("inference_completed", "original_trt"),
        ("inference_started", "candidate_ort"), ("inference_completed", "candidate_ort"),
        ("engine_build_started", None), ("engine_load_verified", None),
    ]


def test_baseline_mismatch_cannot_compile():
    calibration = np.zeros((2000, 3, 32, 32), dtype=np.float32)
    accounting = dict(calls_started=0, calls_completed=0, rows_started=0, rows_completed=0,
                      engine_builds_started=0, engine_loads_verified=0)

    def unexpected(*args):
        pytest.fail("Runtime reached an effect past the baseline gate")

    def baseline(x):
        return np.zeros((len(x), 10), dtype=np.float32)

    with pytest.raises(PilotError, match="did not reproduce"):
        placement.run_admission(calibration, tuple(range(2000)),
            original_ort=baseline, original_trt=baseline, candidate_ort=unexpected, build_candidate=unexpected,
            record=lambda *args, **kwargs: None, accounting=accounting, deadline=1., clock=lambda: 0.)
    assert accounting["calls_completed"] == 2
    assert accounting["engine_builds_started"] == 0


def test_new_profile_has_exact_budget_and_does_not_change_old_limits(tmp_path):
    fields = dict(source_revision="a" * 40, artifact_directory=str(tmp_path))
    config = qonly.DiagnosticConfig(**fields, profile="placement_v1", candidate_onnx=str(tmp_path / "candidate.onnx"))
    assert config.actual_budget() == dict(max_seconds=600, cleanup_seconds=60, unique_calibration_rows=2000,
        inference_row_visits_both_runtimes=8000, inference_calls=68, engine_builds=1,
        sgd=0, exports=0, searches=0, automatic_retries=0)
    assert config.control_config().gpu_seconds == 600
    assert qonly.DiagnosticConfig(**fields).control_config().gpu_seconds == 300
    with pytest.raises(ValueError, match="fixed 2000"):
        qonly.DiagnosticConfig(**fields, profile="placement_v1", rows=8, candidate_onnx=str(tmp_path / "candidate.onnx"))
    with pytest.raises(ValueError, match="only valid"):
        qonly.DiagnosticConfig(**fields, candidate_onnx=str(tmp_path / "candidate.onnx"))


@pytest.mark.parametrize("graph_ok", [False, True])
def test_graph_preflight_runs_in_protected_worker_before_cuda(tmp_path, monkeypatch, graph_ok):
    from fedcore.experiments import heterogeneous_device

    plan = SimpleNamespace(token="test", started_at=time.monotonic(), run_directory=str(tmp_path))
    config = qonly.DiagnosticConfig("a" * 40, str(tmp_path), profile="placement_v1",
                                   candidate_onnx=str(tmp_path / "candidate.onnx"))
    identity = SimpleNamespace(pid=123, birth_ticks=456)
    order = []
    container._publish(tmp_path / "worker-start.json", {"token": "test", "pid": 123, "birth_ticks": 456})
    container._publish(tmp_path / "worker-go.json", {
        "token": "test", "host_pid_identity_proven": False, "gpu_provenance": container.PROVENANCE})
    monkeypatch.setattr(container, "local_process", lambda pid: identity)
    monkeypatch.setattr(container, "_protect_parent", lambda parent: order.append("parent"))
    monkeypatch.setattr(container, "_boundary", lambda *args: order.append("boundary") or {})
    monkeypatch.setattr(heterogeneous_device, "require_cuda_binding",
                        lambda uuid: order.append("cuda") or {"actual_uuid": uuid})

    def verify():
        order.append("verify")
        if not graph_ok:
            raise PilotError("graph mismatch")

    if graph_ok:
        qonly._admit_worker(plan, config, preflight=verify)
        assert order == ["parent", "boundary", "verify", "cuda", "boundary"]
    else:
        with pytest.raises(PilotError, match="graph mismatch"):
            qonly._admit_worker(plan, config, preflight=verify)
        assert order == ["parent", "boundary", "verify"]


@pytest.mark.parametrize("rows,error", [
    ([("42, 100",), (), ()], None),
    ([("42, 100",), (), ("42, 100",)], "reappeared"),
    ([("43, 100",)], "PID changed"),
    ([("42, 100", "43, 100")], "at most one"),
])
def test_terminal_wait_reaps_only_captured_child_and_preserves_gpu_continuity(tmp_path, monkeypatch, rows, error):
    elapsed = 0.
    identity = SimpleNamespace(pid=123, birth_ticks=456)
    plan = SimpleNamespace(token="test")
    config = SimpleNamespace(gpu_uuid="saved-device")
    container._publish(tmp_path / "diagnostic-receipt.json", {"status": "complete"})
    container._publish(tmp_path / "worker-terminal.json", {"token": "test", "pid": 123, "birth_ticks": 456})
    samples = iter(rows)
    tree = SimpleNamespace(capture=lambda: (0, ()))
    worker = SimpleNamespace(poll=lambda: None if elapsed < .5 else 0)

    def wait(seconds):
        nonlocal elapsed
        elapsed += seconds

    monkeypatch.setattr(container.time, "monotonic", lambda: elapsed)
    monkeypatch.setattr(container.time, "sleep", wait)
    args = (tmp_path, plan, config, worker, identity, container.GpuCorrelation(42, True), tree, 3.,
            lambda uuid: next(samples))
    if error:
        with pytest.raises(PilotError, match=error):
            container._wait_for_terminal_worker(*args)
    else:
        assert container._wait_for_terminal_worker(*args) == 0
        assert elapsed == .5


def test_terminal_grace_cannot_extend_original_deadline(tmp_path, monkeypatch):
    elapsed = 0.

    def wait(seconds):
        nonlocal elapsed
        elapsed += seconds

    container._publish(tmp_path / "diagnostic-receipt.json", {"status": "complete"})
    container._publish(tmp_path / "worker-terminal.json", {"token": "test", "pid": 123, "birth_ticks": 456})
    monkeypatch.setattr(container.time, "monotonic", lambda: elapsed)
    monkeypatch.setattr(container.time, "sleep", wait)
    with pytest.raises(TimeoutError, match="not confirmed"):
        container._wait_for_terminal_worker(tmp_path, SimpleNamespace(token="test"), SimpleNamespace(gpu_uuid="gpu"),
            SimpleNamespace(poll=lambda: None), SimpleNamespace(pid=123, birth_ticks=456),
            container.GpuCorrelation(42, True), SimpleNamespace(capture=lambda: (0, ())), .1, lambda uuid: ())
    assert elapsed == .1


@pytest.mark.skipif(sys.platform != "linux", reason="Native Linux GNU timeout, procfs and process reaping")
def test_placement_supervision_linux_cpu_terminal_handoff(tmp_path):
    """Exercise the actual role chain and OS deadline with synthetic telemetry."""
    pytest.importorskip("psutil")
    version = subprocess.run((container.TIMEOUT, "--version"), check=True, capture_output=True, text=True, timeout=2)
    if not version.stdout.startswith("timeout (GNU coreutils)"):
        pytest.skip("GNU timeout is required")
    repository = Path(container.__file__).resolve().parents[2]
    script = tmp_path / "terminal_smoke.py"
    script.write_text("import sys\nsys.path.insert(0, " + repr(str(repository)) + ")\n" + """
import argparse
import os
from pathlib import Path
import time

from fedcore.experiments import heterogeneous_container as c, heterogeneous_host as host

parser = argparse.ArgumentParser()
for key in ("mode", "config", "data-root", "run-directory", "started-at", "token"):
    parser.add_argument("--" + key, required=True)
args = parser.parse_args()
root = Path(args.run_directory)
c.MIN_DISK_BYTES = 0
c.wait_for_start_gate(root / "os-boundary.json", timeout=10.)
plan, _ = c._load_plan(root)
config = c.pilot_config_from_payload(plan.config)
host.query_device_gpu_rows = lambda _: (
    ("999999, 10",) if (root / "worker-ready.json").exists() and not (root / "worker-terminal.json").exists() else ()
)
if args.mode == "supervisor":
    raise SystemExit(c.run_supervisor(plan, require_clean_exit=True))
if args.mode == "guard":
    raise SystemExit(c.run_guard(plan, require_terminal_handoff=True))
c._protect_parent(c.local_process(os.getppid()))
c.wait_for_start_gate(root / "worker-start.json", timeout=10.)
c._boundary(root, config, plan.started_at, "worker")
identity = c.local_process(os.getpid())
terminal = dict(token=plan.token, pid=identity.pid, birth_ticks=identity.birth_ticks)
c._publish(root / "worker-ready.json", dict(terminal, gpu_uuid=config.gpu_uuid, actual_gpu_uuid=config.gpu_uuid.lower()))
c.wait_for_start_gate(root / "worker-go.json", timeout=10.)
c._publish(root / "diagnostic-receipt.json", dict(status="complete"))
c._publish(root / "worker-terminal.json", terminal)
time.sleep(.8)
""", encoding="utf-8")
    config = PilotConfig(qonly.GPU_UUID, "a" * 40, gpu_seconds=40., cleanup_seconds=12.)
    config_path = tmp_path / "config.json"
    container._publish(config_path, asdict(config))
    started_at = time.monotonic()
    plan = container.ContainerPlan(started_at, time.monotonic(), 14., "b" * 32,
        str(Path(sys.executable).absolute()), str(script), str(config_path), str(tmp_path),
        str(tmp_path), json.loads(json.dumps(asdict(config))))
    output_path = tmp_path / "cpu-terminal.log"
    with output_path.open("x", encoding="utf-8") as output:
        child = subprocess.Popen(plan.timeout_argv, stdin=subprocess.DEVNULL, stdout=output,
            stderr=subprocess.STDOUT, start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=config.gpu_uuid))
    try:
        until = time.monotonic() + 2.
        while True:
            captured = container.local_process(child.pid)
            if captured.argv == plan.timeout_argv and container._timer_is_present(child.pid):
                break
            assert time.monotonic() < until, "Native watchdog did not arm"
            time.sleep(.01)
        container._publish(tmp_path / "os-boundary.json",
            dict(schema="fedcore-container-timeout-v1", plan=asdict(plan), timeout=asdict(captured)))
        assert child.wait(timeout=17) == 0, output_path.read_text()
        guard = json.loads((tmp_path / "guard.json").read_text())
        supervisor = json.loads((tmp_path / "supervisor.json").read_text())
        events = [json.loads(line) for line in (tmp_path / "container-events.jsonl").read_text().splitlines()]
        assert guard["returncode"] == supervisor["returncode"] == 0
        assert guard["reason"] == "worker_exited" and guard["correlation"]["admitted"]
        samples = [event for event in events if event["event"] == "terminal_gpu_sample"]
        assert len(samples) >= 2 and not any(event["raw_rows"] for event in samples)
        for name in ("guard_gpu_after_cleanup", "supervisor_gpu_after_cleanup"):
            event, = [record for record in events if record["event"] == name]
            assert event["raw_rows"] == [] and event["cleanup_intervention"] is False
    finally:
        # The armed GNU timeout owns the full process group.
        child.wait(timeout=17)
