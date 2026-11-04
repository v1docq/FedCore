"""CPU-only containment contracts; these tests are not A100 validation."""
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest

from fedcore.experiments import heterogeneous_container as container
from fedcore.experiments.heterogeneous_pilot import PilotConfig, PilotError


GPU = "GPU-12345678-1234-1234-1234-123456789abc"


def plan():
    cfg = PilotConfig(GPU, "a" * 40)
    return container.ContainerPlan(100., 101., 7186., "b" * 32,
        "/env/bin/python", "/repo/scripts/run_heterogeneous_container_pilot.py",
        "/run/config.json", "/data", "/run", json.loads(json.dumps(asdict(cfg))))


def boundary(tmp_path, monkeypatch, *, mutate=None, now=102., timer=True, role="worker"):
    p = plan()
    timeout = container.LocalProcess(1, 10000, 99, 1, 1, p.timeout_argv, str(container.Path(container.TIMEOUT).resolve()))
    processes = {
        1: timeout,
        2: container.LocalProcess(2, 10001, 1, 1, 1, p.command("supervisor"), "/real/python"),
        3: container.LocalProcess(3, 10002, 2, 1, 1, p.command("guard"), "/real/python"),
        4: container.LocalProcess(4, 10003, 3, 1, 1, p.command("worker"), "/real/python"),
    }
    receipt = tmp_path / "os-boundary.json"
    receipt.write_text(json.dumps(dict(schema="fedcore-container-timeout-v1", plan=asdict(p), timeout=asdict(timeout))), encoding="utf-8")
    if mutate:
        mutate(processes)
    monkeypatch.setattr(container.sys, "platform", "linux")
    monkeypatch.setattr(container.os, "getpid", lambda: {"supervisor": 2, "guard": 3, "worker": 4}[role])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU)
    return container.verify_container_boundary(receipt, config=PilotConfig(GPU, "a" * 40),
        started_at=100., deadline=7240., role=role, process=processes.__getitem__,
        now=now, boot_now=now, hz=100, timer_present=lambda pid: timer)


@pytest.mark.parametrize("role", ["supervisor", "guard", "worker"])
def test_live_timeout_and_exact_role_ancestry(tmp_path, monkeypatch, role):
    result = boundary(tmp_path, monkeypatch, role=role)
    assert result["kind"] == "live_gnu_timeout_container_boundary"
    assert result["host_pid_identity_proven"] is False
    assert result["worker_deadline"] == 7240. and result["hard_deadline_at"] == 7300.


def test_detached_timeout_reparenting_preserves_boundary(tmp_path, monkeypatch):
    result = boundary(tmp_path, monkeypatch, mutate=lambda rows: rows.update({1: replace(rows[1], ppid=7)}))
    assert result["timeout_birth_ticks"] == 10000


@pytest.mark.parametrize("pid,field,value", [
    (1, "birth_ticks", 10001), (1, "pgid", 7), (1, "session", 7),
    (1, "executable", "/tmp/timeout"), (1, "argv", ("/usr/bin/timeout", "--foreground")),
    (2, "argv", ("/bin/sh",)), (3, "ppid", 1), (4, "pgid", 4),
    (4, "session", 4), (4, "argv", ("/bin/other-worker",)),
])
def test_changed_boundary_cannot_admit_worker(tmp_path, monkeypatch, pid, field, value):
    def mutate(rows):
        rows[pid] = replace(rows[pid], **{field: value})
    with pytest.raises(PilotError):
        boundary(tmp_path, monkeypatch, mutate=mutate)


def test_missing_live_timer_and_expired_original_window_fail(tmp_path, monkeypatch):
    with pytest.raises(PilotError, match="timer"):
        boundary(tmp_path, monkeypatch, timer=False)
    with pytest.raises(PilotError, match="expired"):
        boundary(tmp_path, monkeypatch, now=7240.)


def test_timeout_duration_cannot_grow_or_restart_original_clock():
    p = plan()
    with pytest.raises(ValueError):
        replace(p, timeout_seconds=p.timeout_seconds + .001)
    assert p.timeout_argv[:3] == ("/usr/bin/timeout", "--signal=KILL", "7186.000s")
    assert "--foreground" not in p.timeout_argv
    assert p.config["protocol"]["epochs"] == 80


def test_all_sgd_and_fixed_recipe_controls_reused():
    cfg = container.pilot_config_from_payload(plan().config)
    assert cfg.budget() == dict(models=1, recipes=12, baseline_sgd=25040,
        recovery_sgd=2400, total_sgd=27440, max_seconds=7200., searches=0, automatic_retries=0)


def test_single_row_correlates_only_after_cuda_ready_without_host_identity_claim():
    state = container.observe_gpu((), container.GpuCorrelation(), ready=False)
    # A host PID may be absent from container procfs. It is never resolved here.
    state = container.observe_gpu(("987654321, 800",), state, ready=False)
    assert state == container.GpuCorrelation(987654321, False)
    state = container.observe_gpu(("987654321, 900",), state, ready=True)
    assert state == container.GpuCorrelation(987654321, True)
    assert "no authenticated host PID identity proof" in container.PROVENANCE


@pytest.mark.parametrize("rows", [
    ("999999, 10", "888888, 20"), ("999999, 10", "999999, 10"),
    ("999999",), ("999999, N/A",), ("N/A, 10",), ("0, 10",),
    ("999999, -1",), ("999999, 1.5",), ("999999, 99999999999",), ("999999, 10, x",),
])
def test_unknown_extra_or_malformed_raw_rows_never_disappear(rows):
    with pytest.raises(PilotError):
        container.observe_gpu(rows, container.GpuCorrelation(), ready=True)


@pytest.mark.parametrize("rows", [(), ("222222, 10",)])
def test_row_loss_or_change_is_terminal(rows):
    with pytest.raises(PilotError):
        container.observe_gpu(rows, container.GpuCorrelation(111111, True), ready=True)


def test_fixed_child_launch_never_creates_new_session(monkeypatch):
    calls = []
    monkeypatch.setattr(container.subprocess, "Popen", lambda *args, **kwargs: calls.append((args, kwargs)))
    container._spawn(plan(), "worker")
    assert calls[0][0] == (plan().command("worker"),)
    assert calls[0][1]["start_new_session"] is False
    assert calls[0][1]["shell"] is False


def test_parent_death_protection_precedes_birth_check(monkeypatch):
    parent = container.LocalProcess(17, 400, 1, 7, 7, ("python",), "/bin/python")
    events = []
    monkeypatch.setattr(container, "_prctl", lambda option, value: events.append((option, value)))
    monkeypatch.setattr(container.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(container.os, "getppid", lambda: 17)
    monkeypatch.setattr(container, "local_process", lambda pid: replace(parent, birth_ticks=401))
    with pytest.raises(PilotError):
        container._protect_parent(parent)
    assert events == [(1, container.signal.SIGKILL)]


def test_cleanup_signals_captured_local_birth_even_after_capture_rejects_escape(monkeypatch):
    import psutil
    tree = object.__new__(container.OwnedDescendants)
    tree.psutil, tree.owned = psutil, {42: 123.}
    def capture():
        raise PilotError("owned process escaped")
    tree.capture = capture
    calls = []
    monkeypatch.setattr(psutil, "Process", lambda pid: SimpleNamespace(
        create_time=lambda: 123., kill=lambda: calls.append(pid)))
    result = tree.kill()
    assert calls == [42]
    assert result[0]["result"] == "capture_failed"
    assert result[1]["result"] == "owned_kill_sent"


def test_cleanup_refuses_reused_pid(monkeypatch):
    import psutil
    tree = object.__new__(container.OwnedDescendants)
    tree.psutil, tree.owned = psutil, {42: 123.}
    tree.capture = lambda: (0, ())
    monkeypatch.setattr(psutil, "Process", lambda pid: SimpleNamespace(
        create_time=lambda: 124., kill=lambda: pytest.fail("Reused PID must not receive a signal")))
    assert tree.kill() == [dict(pid=42, result="birth_changed_not_signalled")]


def test_control_publication_is_fresh_and_no_retry(tmp_path):
    target = tmp_path / "gate.json"
    container._publish(target, dict(token="owned"))
    assert json.loads(target.read_text()) == dict(token="owned")
    with pytest.raises(FileExistsError):
        container._publish(target, dict(token="overwrite"))


@pytest.mark.parametrize("exit_code", [1, -9, None])
def test_guard_handles_exit_between_poll_and_ready_process_read(tmp_path, monkeypatch, exit_code):
    from fedcore.experiments import heterogeneous_host
    cfg = container.pilot_config_from_payload(plan().config)
    p = SimpleNamespace(**{**vars(plan()), "run_directory": str(tmp_path)})
    identity = container.LocalProcess(77, 10003, 3, 1, 1, ("worker",), "/python")

    class Worker:
        pid, returncode, polls = 77, None, 0
        def poll(self):
            self.polls += 1
            if self.polls >= 2:
                self.returncode = exit_code
            return self.returncode

    worker = Worker()
    reads = []
    def process(pid):
        if pid != worker.pid:
            return identity
        reads.append(pid)
        if len(reads) > 1:
            raise FileNotFoundError("worker exited during procfs observation")
        return identity

    (tmp_path / "worker-ready.json").write_text(json.dumps(dict(token=p.token,
        pid=worker.pid, birth_ticks=identity.birth_ticks, gpu_uuid=GPU,
        actual_gpu_uuid=GPU.lower())), encoding="utf-8")
    monkeypatch.setattr(container, "_config", lambda path: cfg)
    monkeypatch.setattr(container, "_protect_parent", lambda parent: None)
    monkeypatch.setattr(container, "_boundary", lambda *args: {})
    monkeypatch.setattr(container, "_prctl", lambda *args: None)
    monkeypatch.setattr(container.os, "getppid", lambda: 3)
    monkeypatch.setattr(container, "local_process", process)
    monkeypatch.setattr(container, "OwnedDescendants", lambda: SimpleNamespace(capture=lambda: (0, ())))
    monkeypatch.setattr(container, "_spawn", lambda *args: worker)
    cleanup = []
    monkeypatch.setattr(container, "_finish_owned", lambda *args, **kwargs: cleanup.append(True))
    monkeypatch.setattr(container.time, "monotonic", lambda: 102.)
    monkeypatch.setattr(container.shutil, "disk_usage", lambda path: SimpleNamespace(free=container.MIN_DISK_BYTES + 1))
    monkeypatch.setattr(heterogeneous_host, "query_device_gpu_rows", lambda gpu: ())

    assert container.run_guard(p) == 1
    receipt = json.loads((tmp_path / "guard.json").read_text(encoding="utf-8"))
    assert receipt["returncode"] == exit_code
    if exit_code is not None:
        assert receipt["reason"] == "worker_exited"
    else:
        assert "FileNotFoundError" in receipt["reason"]
    assert cleanup == [True] and reads == [77, 77]
