"""Independent fail-closed pilot checks; no process, model, CUDA or ONNX run."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fedcore.experiments import heterogeneous_pilot as pilot
from fedcore.experiments import cv_independent_guard as guard
from fedcore.experiments.cv_independent_supervision import NamespaceBinding


GPU = "GPU-12345678-1234-1234-1234-123456789abc"


def config(**kwargs):
    return pilot.PilotConfig(GPU, "a" * 40, **kwargs)


def forbidden(*args, **kwargs):
    pytest.fail("A forbidden numerical or process boundary was reached")


def test_elapsed_boundary_verification_cannot_start_cuda_admission(tmp_path):
    """A successful but late boundary cannot authorize the next effect."""
    now = [100.]
    cfg = config()
    deadline = 100. + cfg.gpu_seconds - cfg.cleanup_seconds

    def boundary(**kwargs):
        assert kwargs["started_at"] == 100. and kwargs["deadline"] == deadline
        now[0] = deadline
        return {"verified_before_response_arrived": True}

    receipt = pilot.run_pilot(None, tmp_path / "pilot", config=cfg, started_at=100.,
        boundary_check=boundary, clock=lambda: now[0],
        effects=replace(pilot.PilotEffects(), admission=forbidden, train_baseline=forbidden))
    assert receipt["status"] == "admission_failed" and not receipt["experiment_started"]
    assert receipt["failure"]["type"] == "TimeoutError"
    assert set(receipt["counters"].values()) == {0}
    assert receipt["ended_at"] == receipt["deadline"] == deadline


def test_expired_original_worker_window_prevents_guard_spawn(tmp_path, monkeypatch):
    """The guard starts during cleanup; it cannot invent another 7140 seconds."""
    monkeypatch.delenv("FEDCORE_PILOT_ABSOLUTE_DEADLINE", raising=False)
    effects = guard.GuardEffects(popen=forbidden, process=forbidden, gpu_rows=lambda: (),
        clock=lambda: 7250., sleep=forbidden, timer=forbidden,
        disk_free=lambda path: guard.MINIMUM_DISK_BYTES + 1, guard_rss=lambda: 0)
    result = guard.run_guard(("never-spawn",), receipt_path=tmp_path / "guard.json",
        started_at=100., limits=guard.GuardLimits(7200., 60.), effects=effects,
        binding_provider=lambda rows, owned: NamespaceBinding(True, (), "injected host binding"))
    assert result["started_at"] == 100. and result["guard_started_at"] == 7250.
    assert result["absolute_deadline"] == 7240.
    assert result["spawned_child_count"] == 0 and result["reason"] == "absolute_deadline_before_spawn"
    assert result["max_seconds"] == 7200. and result["worker_seconds"] == 7140.
    assert result["within_authorized_window"]  # The remaining 50 seconds belong to cleanup.


@pytest.mark.parametrize("release_worker", [False, True])
def test_worker_gate_is_created_only_at_guard_spawn_boundary(tmp_path, monkeypatch, release_worker):
    from fedcore.experiments import heterogeneous_pilot_guard as bridge
    events = []
    cfg = config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(cfg)), encoding="utf-8")
    boundary_path = tmp_path / "os-boundary.json"
    gate = tmp_path / "worker-start-gate.json"
    monkeypatch.setattr(bridge.sys, "platform", "linux")
    monkeypatch.setattr(bridge.time, "monotonic", lambda: 200.)
    monkeypatch.setattr(bridge.os, "readlink", lambda path: "pid:[12345]")
    monkeypatch.setattr(pilot, "wait_for_start_gate", lambda path, **kw: events.append("launcher_gate"))

    def boundary(path, **kwargs):
        assert path == boundary_path and not gate.exists()
        events.append("live_boundary")
        return {"invocation_id": "b" * 32, "kind": "injected_boundary"}

    monkeypatch.setattr(pilot, "verify_worker_boundary", boundary)

    def popen(command, **kwargs):
        assert gate.is_file()
        payload = json.loads(gate.read_text(encoding="utf-8"))
        assert payload == {"boundary_receipt": str(boundary_path), "invocation_id": "b" * 32}
        events.append("popen")
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(bridge.subprocess, "Popen", popen)

    def run_guard(command, **kwargs):
        events.append("guard_preflight")
        assert not gate.exists()
        assert kwargs["started_at"] == 100.
        assert kwargs["limits"] == guard.GuardLimits(7200., 60.)
        assert kwargs["environment"]["CUDA_VISIBLE_DEVICES"] == GPU
        assert float(kwargs["environment"]["FEDCORE_PILOT_ABSOLUTE_DEADLINE"]) == 7240.
        if release_worker:
            kwargs["effects"].popen(command)
        return {"status": "finished" if release_worker else "blocked", "returncode": 0}

    monkeypatch.setattr(guard, "run_guard", run_guard)
    bridge.run_pilot_guard(config_path=config_path, data_root=tmp_path,
        run_directory=tmp_path, boundary_receipt=boundary_path, namespace_id="pid:[12345]",
        started_at=100., python_executable="/usr/bin/python3")
    assert events == ["launcher_gate", "live_boundary", "guard_preflight"] + (["popen"] if release_worker else [])
    assert gate.exists() is release_worker


def test_missing_start_gate_has_bounded_wait_without_wall_clock_sleep(tmp_path):
    now, intervals = [100.], []

    def advance(seconds):
        intervals.append(seconds)
        now[0] += seconds

    with pytest.raises(TimeoutError, match="did not release"):
        pilot.wait_for_start_gate(tmp_path / "absent", timeout=.12,
            clock=lambda: now[0], sleep=advance)
    assert now[0] == pytest.approx(100.12)
    assert len(intervals) <= 3 and max(intervals) <= .05


def test_gate_arriving_exactly_at_deadline_is_not_ready(tmp_path):
    now = [100.]
    gate = tmp_path / "late-gate"

    def advance(seconds):
        now[0] = 100.05
        gate.write_text("late", encoding="ascii")

    with pytest.raises(TimeoutError, match="arrived after"):
        pilot.wait_for_start_gate(gate, timeout=.05, clock=lambda: now[0], sleep=advance)


def test_export_validator_receives_private_calibration_snapshot(tmp_path, monkeypatch):
    import torch
    from fedcore.algorithm.heterogeneous import backend
    from fedcore.algorithm.low_rank.grammar_execution import StructuralInvalid
    calibration = torch.zeros(1, 3, 32, 32)
    expected = calibration.clone()
    workflow = SimpleNamespace(space=SimpleNamespace(structural_space=SimpleNamespace(
        calibration=SimpleNamespace(fingerprint=backend._calibration(calibration)))))
    observed = []

    def stop_after_validation(model, snapshot, value):
        observed.append(snapshot)
        assert snapshot.data_ptr() != calibration.data_ptr()
        calibration.add_(1)  # A caller changes its tensor after the snapshot was taken.
        assert torch.equal(snapshot, expected)
        raise backend.BackendContractError("IndependentProbeStop", "No ONNX or model execution")

    monkeypatch.setattr(backend, "_check_recovered", stop_after_validation)
    result = backend.export_onnx_qdq(object(), calibration, workflow, tmp_path / "never-export")
    assert type(result) is StructuralInvalid
    assert result.violations[0].code == "IndependentProbeStop" and len(observed) == 1
    assert not (tmp_path / "never-export").exists()


def test_mutation_between_initial_identity_read_and_clone_cannot_rebind_workflow(tmp_path, monkeypatch):
    import torch
    from fedcore.algorithm.heterogeneous import backend
    from fedcore.algorithm.low_rank.grammar_execution import StructuralInvalid
    calibration = torch.zeros(1, 3, 32, 32)
    real_identity = backend._calibration
    expected = real_identity(calibration)
    workflow = SimpleNamespace(space=SimpleNamespace(structural_space=SimpleNamespace(
        calibration=SimpleNamespace(fingerprint=expected))))
    reads = []

    def racing_identity(value):
        identity = real_identity(value)
        reads.append(identity)
        if len(reads) == 1:
            calibration.add_(1)
        return identity

    def validate_snapshot(model, value, bound_workflow):
        if backend._calibration(value) != bound_workflow.space.structural_space.calibration.fingerprint:
            raise backend.BackendContractError("StaleCalibration", "Snapshot differs from the sealed workflow")

    monkeypatch.setattr(backend, "_calibration", racing_identity)
    monkeypatch.setattr(backend, "_check_recovered", validate_snapshot)
    result = backend.export_onnx_qdq(object(), calibration, workflow, tmp_path / "never-export")
    assert type(result) is StructuralInvalid and result.violations[0].code == "StaleCalibration"
    assert reads[0] == expected and reads[-1] != expected
    assert not (tmp_path / "never-export").exists()


def worker_boundary_fixture(tmp_path, monkeypatch):
    """A real serialized pure launch plan; every live host read is substituted."""
    from fedcore.experiments import heterogeneous_deadline as deadline_module
    cfg = config()
    request = deadline_module.DeadlineRequest(
        ("/usr/bin/python3", "-m", "fedcore.experiments.heterogeneous_pilot_guard"),
        "/srv/fedcore", "1" * 32, gpu_uuid=GPU)
    budget = deadline_module.DeadlineBudget(100., 7200., cleanup_seconds=60.)
    plan = deadline_module.compile_launch(request, budget, now=105.)
    group = "/user.slice/user-1000.slice/user@1000.service/app.slice/" + request.unit
    receipt = deadline_module.DeadlineReceipt(plan, "b" * 32, group, "active")
    payload = json.loads(json.dumps(asdict(receipt)))
    receipt_path = tmp_path / "os-boundary.json"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    values = dict(InvocationID=receipt.invocation_id, Description=request.description,
        ControlGroup=group, ActiveState="active", KillMode="control-group", SendSIGKILL="yes",
        Delegate="no", ProtectControlGroups="yes", Restart="no", TimeoutStopFailureMode="kill",
        NoNewPrivileges="yes", FinalKillSignal="SIGKILL", Type="exec", WorkingDirectory="/srv/fedcore",
        RuntimeMaxUSec=str(plan.runtime_microseconds) + "us",
        TimeoutStartUSec=str(plan.start_microseconds) + "us",
        TimeoutStopUSec=str(plan.stop_microseconds) + "us")
    calls = []

    def transport(argv, *, timeout):
        calls.append((argv, timeout))
        assert argv[:3] == ("systemctl", "--user", "show")
        assert argv[-2:] == ("--", request.unit) and timeout == 3.
        fields = [arg.split("=", 1)[1] for arg in argv if arg.startswith("--property=")]
        return deadline_module.CommandResult(0, "".join(key + "=" + values[key] + "\n" for key in fields), "")

    actual_read_text = Path.read_text

    def read_text(path, *args, **kwargs):
        if path == Path("/proc/self/cgroup"):
            return "0::" + group + "\n"
        return actual_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(pilot.sys, "platform", "linux")
    monkeypatch.setenv("INVOCATION_ID", receipt.invocation_id)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU)
    monkeypatch.setattr(deadline_module, "BoundedLocalTransport", lambda: transport)
    return SimpleNamespace(config=cfg, path=receipt_path, payload=payload, values=values,
                           calls=calls, plan=plan, request=request)


def test_valid_original_os_plan_and_live_worker_boundary_agree(tmp_path, monkeypatch):
    fixture = worker_boundary_fixture(tmp_path, monkeypatch)
    result = pilot.verify_worker_boundary(fixture.path, config=fixture.config,
        started_at=100., deadline=7240.)
    assert result["kind"] == "live_systemd_worker_boundary"
    assert result["deadline_at"] == 7300. and result["unit"] == fixture.request.unit
    assert result["invocation_id"] == fixture.payload["invocation_id"]
    assert result["properties"] == fixture.values
    assert len(fixture.calls) == 1 and len(result["receipt_sha256"]) == 64


def test_forged_runtime_matching_live_unit_cannot_extend_original_window(tmp_path, monkeypatch):
    fixture = worker_boundary_fixture(tmp_path, monkeypatch)
    enlarged = fixture.plan.runtime_microseconds + 3_600_000_000
    fixture.payload["plan"]["runtime_microseconds"] = enlarged
    fixture.payload["plan"]["argv"] = [
        "--property=RuntimeMaxSec=" + str(enlarged) + "us"
        if arg.startswith("--property=RuntimeMaxSec=") else arg
        for arg in fixture.payload["plan"]["argv"]]
    fixture.values["RuntimeMaxUSec"] = str(enlarged) + "us"
    fixture.path.write_text(json.dumps(fixture.payload), encoding="utf-8")
    with pytest.raises(pilot.PilotError, match="original bounded controller window"):
        pilot.verify_worker_boundary(fixture.path, config=fixture.config, started_at=100., deadline=7240.)
    assert fixture.calls == []  # Matching live values cannot rescue a forged duration.


@pytest.mark.parametrize("changed_field", ["compiled_at", "argv"])
def test_altered_serialized_launch_plan_is_rejected_before_live_probe(tmp_path, monkeypatch, changed_field):
    fixture = worker_boundary_fixture(tmp_path, monkeypatch)
    if changed_field == "compiled_at":
        fixture.payload["plan"]["compiled_at"] += 1.
    else:
        fixture.payload["plan"]["argv"].append("unapproved-literal-argument")
    fixture.path.write_text(json.dumps(fixture.payload), encoding="utf-8")
    with pytest.raises(pilot.PilotError, match="original bounded controller window"):
        pilot.verify_worker_boundary(fixture.path, config=fixture.config, started_at=100., deadline=7240.)
    assert fixture.calls == []


def test_live_start_timeout_cannot_exceed_original_launch_plan(tmp_path, monkeypatch):
    fixture = worker_boundary_fixture(tmp_path, monkeypatch)
    fixture.values["TimeoutStartUSec"] = str(fixture.plan.start_microseconds + 1) + "us"
    with pytest.raises(pilot.PilotError, match="deadline exceeds"):
        pilot.verify_worker_boundary(fixture.path, config=fixture.config, started_at=100., deadline=7240.)
    assert len(fixture.calls) == 1
