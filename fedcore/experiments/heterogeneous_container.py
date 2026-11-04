"""One-shot container pilot with a GNU timeout ceiling and one CUDA worker.

This is an explicitly weaker profile than authenticated host PID binding. An
empty device followed by one stable raw NVML PID and the worker's CUDA-ready
handshake is correlation, NOT proof of that host PID's identity. Raw GPU PIDs
are never resolved in container procfs or used as signal targets. Unexpected
rows stop only captured local descendants. No discovery, CUDA or launch runs
on import. This profile covers the fixed, trusted, non-escaping worker only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time

from .heterogeneous_pilot import PilotConfig, PilotError, pilot_config_from_payload, wait_for_start_gate


TIMEOUT = "/usr/bin/timeout"
HARD_RESERVE_SECONDS = 10.
LAUNCH_RESERVE_SECONDS = 3.
HANDSHAKE_SECONDS = 60.
POLL_SECONDS = .25
MAX_RSS_BYTES = 32 * 1024**3
MAX_VRAM_MIB = 16 * 1024
MIN_DISK_BYTES = 100 * 1024**3
PROVENANCE = ("empty-device preflight + one CUDA-ready worker + one unchanged raw NVML PID; "
              "temporal correlation only, no authenticated host PID identity proof")


def _read(path, *, binary=False):
    with Path(path).open("rb" if binary else "r", **({} if binary else {"encoding": "utf-8"})) as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise PilotError("Container control record exceeds bounded capacity")
    return raw


def _publish(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or path.is_symlink():
        raise FileExistsError("Container control files must be fresh")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


@dataclass(frozen=True)
class LocalProcess:
    pid: int
    birth_ticks: int
    ppid: int
    pgid: int
    session: int
    argv: tuple[str, ...]
    executable: str


def local_process(pid):
    """Read one stable procfs lifetime, never an NVML-reported host PID."""
    if type(pid) is not int or pid <= 0:
        raise ValueError("A positive captured local PID is required")
    root = Path("/proc") / str(pid)
    first = _read(root / "stat")
    fields = first[first.rfind(")") + 2:].split()
    argv = tuple(part.decode("utf-8") for part in _read(root / "cmdline", binary=True).split(b"\0") if part)
    executable = os.readlink(root / "exe")
    last = _read(root / "stat")
    if first.split(")", 1)[0] != last.split(")", 1)[0] or fields[19] != last[last.rfind(")") + 2:].split()[19]:
        raise PilotError("Local process lifetime changed during observation")
    return LocalProcess(pid, int(fields[19]), int(fields[1]), int(fields[2]), int(fields[3]), argv, executable)


@dataclass(frozen=True)
class ContainerPlan:
    started_at: float
    compiled_at: float
    timeout_seconds: float
    token: str
    python: str
    script: str
    config_path: str
    data_root: str
    run_directory: str
    config: dict

    def __post_init__(self):
        cfg = pilot_config_from_payload(self.config)
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in
                (self.started_at, self.compiled_at, self.timeout_seconds))
                or not 0 <= self.started_at <= self.compiled_at
                or self.timeout_seconds <= 0 or cfg.cleanup_seconds <= HARD_RESERVE_SECONDS
                or self.timeout_seconds > math.floor((self.started_at + cfg.gpu_seconds - self.compiled_at
                    - HARD_RESERVE_SECONDS - LAUNCH_RESERVE_SECONDS) * 1000) / 1000
                or type(self.token) is not str or re.fullmatch(r"[a-f0-9]{32}", self.token) is None
                or any(type(p) is not str or not p.startswith("/") or any(c in p for c in ("\0", "\n", "\r"))
                       for p in (self.python, self.script, self.config_path, self.data_root, self.run_directory))):
            raise ValueError("An original <=7200s window, explicit paths and reserved OS ceiling are required")

    def command(self, role):
        if role not in ("supervisor", "guard", "worker"):
            raise ValueError("Unknown fixed container role")
        return (self.python, "-u", self.script, "--mode", role, "--config", self.config_path,
                "--data-root", self.data_root, "--run-directory", self.run_directory,
                "--started-at", str(self.started_at), "--token", self.token)

    @property
    def timeout_argv(self):
        # --foreground would exempt children from the group deadline; forbidden.
        return (TIMEOUT, "--signal=KILL", format(self.timeout_seconds, ".3f") + "s") + self.command("supervisor")


def _load_plan(root):
    value = json.loads(_read(Path(root) / "os-boundary.json"))
    if type(value) is not dict or set(value) != {"schema", "plan", "timeout"} or value["schema"] != "fedcore-container-timeout-v1":
        raise PilotError("Exact container OS-boundary record required")
    plan = ContainerPlan(**value["plan"])
    raw = value["timeout"]
    captured = LocalProcess(**{**raw, "argv": tuple(raw["argv"])})
    return plan, captured


def _timer_is_present(pid):
    text = _read(Path("/proc") / str(pid) / "timers")
    # GNU timeout creates a CLOCK_REALTIME POSIX timer delivering SIGALRM.
    return (re.search(r"(?m)^signal:\s*14/", text) is not None
            and re.search(r"(?m)^notify:\s*signal/pid\." + str(pid) + r"\s*$", text) is not None
            and re.search(r"(?m)^ClockID:\s*0\s*$", text) is not None)


def _same_timeout_lifetime(first, second):
    # The detached launch client exits normally. Timeout's outer parent may
    # then change, while its owned lifetime and group remain the same.
    return (first.pid, first.birth_ticks, first.pgid, first.session, first.argv, first.executable) == (
        second.pid, second.birth_ticks, second.pgid, second.session, second.argv, second.executable)


def verify_container_boundary(receipt_path, *, config, started_at, deadline, role="worker",
                              process=None, now=None, boot_now=None, hz=None, timer_present=None):
    """Native check: live timeout ancestor, births, exact argv and common PGID.

    The JSON record is an input to live verification, never its proof. The
    trusted GNU executable and its live POSIX timer remain outside Python's
    cooperative deadline. A hard kill may leave only the durable partial log.
    """
    if sys.platform != "linux":
        raise PilotError("Container profile requires Linux procfs and GNU timeout")
    plan, captured = _load_plan(Path(receipt_path).parent)
    if (type(config) is not PilotConfig or asdict(config) != asdict(pilot_config_from_payload(plan.config))
            or started_at != plan.started_at or deadline != started_at + config.gpu_seconds - config.cleanup_seconds
            or os.environ.get("CUDA_VISIBLE_DEVICES") != config.gpu_uuid):
        raise PilotError("Container plan, original window and GPU visibility differ")
    process = process or local_process
    now = time.monotonic() if now is None else now
    boot_now = time.clock_gettime(time.CLOCK_BOOTTIME) if boot_now is None else boot_now
    hz = os.sysconf("SC_CLK_TCK") if hz is None else hz
    if not plan.compiled_at <= now < deadline:
        raise PilotError("Original worker window expired before live boundary verification")
    live = process(captured.pid)
    if (not _same_timeout_lifetime(live, captured) or live.argv != plan.timeout_argv
            or live.executable != str(Path(TIMEOUT).resolve()) or live.pgid != live.pid or live.session != live.pid
            or not (timer_present or _timer_is_present)(live.pid)):
        raise PilotError("Live GNU timeout lifetime, timer, executable or private group differs")
    # A process-start tick is rounded down. Add one tick to bound its latest
    # birth; the launch reserve covers normal timeout fork/timer setup latency.
    age = boot_now - live.birth_ticks / hz
    if (not math.isfinite(age) or age < -1 / hz
            or now + plan.timeout_seconds - age + 1 / hz > started_at + config.gpu_seconds - HARD_RESERVE_SECONDS):
        raise PilotError("Live timeout would exceed the original reserved hard ceiling")
    current = process(os.getpid())
    if current.argv != plan.command(role):
        raise PilotError("Current process is not the exact fixed container role")
    lineage, cursor = [], current
    while cursor.pid != live.pid:
        if len(lineage) >= 4 or cursor.pgid != live.pid or cursor.session != live.pid:
            raise PilotError("Role escaped the timeout process group or ancestry")
        lineage.append(cursor)
        cursor = process(cursor.ppid)
    expected = {"supervisor": ("supervisor",), "guard": ("guard", "supervisor"),
                "worker": ("worker", "guard", "supervisor")}[role]
    if tuple(p.argv for p in lineage) != tuple(plan.command(item) for item in expected):
        raise PilotError("Exact direct supervisor/guard/worker ancestry is required")
    if not _same_timeout_lifetime(process(live.pid), live):
        raise PilotError("Timeout lifetime changed during verification")
    for prior in lineage:
        if process(prior.pid) != prior:
            raise PilotError("A boundary ancestor changed during verification")
    return dict(kind="live_gnu_timeout_container_boundary", timeout_pid=live.pid,
                timeout_birth_ticks=live.birth_ticks, pgid=live.pgid, worker_pid=current.pid,
                original_started_at=started_at, hard_deadline_at=started_at + config.gpu_seconds,
                worker_deadline=deadline, timeout_seconds=plan.timeout_seconds,
                live_posix_timer_observed=True, host_pid_identity_proven=False,
                gpu_provenance=PROVENANCE, receipt_sha256=hashlib.sha256(_read(receipt_path, binary=True)).hexdigest())


def _prctl(option, value):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(option, value, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Required Linux process-lifetime control failed")


def _protect_parent(expected):
    """PDEATHSIG plus a fresh parent birth check closes the pre-prctl race."""
    _prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    if os.getppid() != expected.pid or not _same_timeout_lifetime(local_process(expected.pid), expected):
        raise PilotError("Direct parent lifetime changed before child protection")


class OwnedDescendants:
    """Capture only descendants of this subreaper; never signal raw GPU PIDs."""
    def __init__(self):
        import psutil
        self.psutil = psutil
        self.parent = psutil.Process()
        self.group = os.getpgrp()
        self.owned = {}

    def capture(self):
        rss, live = self.parent.memory_info().rss, []
        for child in self.parent.children(recursive=True):
            try:
                birth = child.create_time()
                if child.pid in self.owned and self.owned[child.pid] != birth:
                    raise PilotError("An owned descendant PID was reused")
                self.owned[child.pid] = birth
                if child.status() != self.psutil.STATUS_ZOMBIE:
                    rss += child.memory_info().rss
                    live.append((child.pid, birth))
                if os.getpgid(child.pid) != self.group:
                    raise PilotError("A fixed worker descendant escaped the timeout group")
            except (self.psutil.NoSuchProcess, ProcessLookupError):
                continue
        return rss, tuple(live)

    def kill(self):
        results = []
        try:
            self.capture()
        except Exception as error:
            # In particular, capture records an escaping descendant before it
            # rejects its group. That owned birth must still receive cleanup.
            results.append(dict(result="capture_failed", error=repr(error)))
        for pid, birth in reversed(tuple(self.owned.items())):
            try:
                child = self.psutil.Process(pid)
                if child.create_time() != birth:
                    results.append(dict(pid=pid, result="birth_changed_not_signalled")); continue
                child.kill()  # psutil performs a further process-reuse check.
                results.append(dict(pid=pid, birth=birth, result="owned_kill_sent"))
            except self.psutil.NoSuchProcess:
                results.append(dict(pid=pid, result="already_exited"))
        return results


@dataclass(frozen=True)
class GpuCorrelation:
    host_pid: int | None = None
    admitted: bool = False


def observe_gpu(rows, state, *, ready):
    """Conserve every raw row. Equal container/host PID integers prove nothing."""
    if type(rows) is not tuple or type(state) is not GpuCorrelation or len(rows) > 1:
        raise PilotError("Expected at most one raw NVML row; every row remains visible")
    if not rows:
        if state.host_pid is not None:
            raise PilotError("The correlated raw GPU process disappeared")
        return state
    row = rows[0]
    if type(row) is not str or len(row) > 4096:
        raise PilotError("Malformed raw NVML row")
    fields = tuple(part.strip() for part in row.split(","))
    if (len(fields) != 2 or any(not v.isascii() or not v.isdecimal() or len(v) > 20 for v in fields)
            or int(fields[0]) < 1 or int(fields[1]) > MAX_VRAM_MIB):
        raise PilotError("Malformed or excessive raw GPU telemetry")
    pid = int(fields[0])
    if state.host_pid is not None and state.host_pid != pid:
        raise PilotError("Raw GPU PID changed during the single-worker window")
    return GpuCorrelation(pid, state.admitted or ready)


def _event(root, name, **value):
    with (Path(root) / "container-events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(at=time.monotonic(), event=name, **value), allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def _config(path):
    return pilot_config_from_payload(json.loads(_read(path)))


def _boundary(root, config, started_at, role):
    return verify_container_boundary(Path(root) / "os-boundary.json", config=config,
        started_at=started_at, deadline=started_at + config.gpu_seconds - config.cleanup_seconds, role=role)


def _spawn(plan, role):
    # Only the outer timeout receives start_new_session=True. All fixed roles
    # and their ordinary descendants inherit that same group and session.
    return subprocess.Popen(plan.command(role), stdin=subprocess.DEVNULL, shell=False,
                            start_new_session=False, cwd=str(Path(plan.script).parent.parent))


def _finish_owned(root, tree, child, *, name):
    """Never let timeout's direct child exit while a known/adopted child lives."""
    while True:
        try:
            signals = tree.kill()
            child.poll()
            # Reap children adopted after an intermediate guard dies.
            while True:
                try:
                    pid, _ = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if not pid:
                    break
            _, live = tree.capture()
            if not live:
                _event(root, name + "_cleanup_complete", signals=signals)
                return
        except Exception as error:
            # Keep timeout alive on uncertain cleanup rather than releasing
            # its kernel timer while descendants may remain.
            _event(root, name + "_cleanup_uncertain", error=repr(error))
        time.sleep(POLL_SECONDS)


def run_supervisor(plan):
    root, cfg = Path(plan.run_directory), _config(plan.config_path)
    _protect_parent(local_process(os.getppid()))
    wait_for_start_gate(root / "os-boundary.json", timeout=10.)
    _boundary(root, cfg, plan.started_at, "supervisor")
    _prctl(36, 1)  # PR_SET_CHILD_SUBREAPER
    tree = OwnedDescendants()
    guard = _spawn(plan, "guard")
    try:
        while guard.poll() is None:
            _, live = tree.capture()
            if time.monotonic() >= plan.started_at + cfg.gpu_seconds - cfg.cleanup_seconds:
                break
            time.sleep(POLL_SECONDS)
        result = guard.poll()
    finally:
        _finish_owned(root, tree, guard, name="supervisor")
    _publish(root / "supervisor.json", dict(returncode=result, ended_at=time.monotonic(),
        host_pid_identity_proven=False, gpu_provenance=PROVENANCE, automatic_retries=0))
    return 0 if result == 0 else 1


def run_guard(plan):
    from .heterogeneous_host import query_device_gpu_rows
    root, cfg = Path(plan.run_directory), _config(plan.config_path)
    _protect_parent(local_process(os.getppid()))
    _boundary(root, cfg, plan.started_at, "guard")
    _prctl(36, 1)
    tree = OwnedDescendants()
    worker, result, reason = None, None, "preflight_failed"
    state = GpuCorrelation()
    deadline = plan.started_at + cfg.gpu_seconds - cfg.cleanup_seconds
    try:
        rows = query_device_gpu_rows(cfg.gpu_uuid)
        _event(root, "empty_gpu_preflight", raw_rows=rows)
        observe_gpu(rows, state, ready=False)
        if rows:
            raise PilotError("Device must be empty before spawning the sole CUDA worker")
        if shutil.disk_usage(root).free < MIN_DISK_BYTES or time.monotonic() >= deadline:
            raise PilotError("Insufficient disk or no original worker window remains")
        worker = _spawn(plan, "worker")
        worker_identity = local_process(worker.pid)
        tree.capture()
        _publish(root / "worker-start.json", dict(token=plan.token, pid=worker.pid,
            birth_ticks=worker_identity.birth_ticks))
        ready_deadline = min(deadline, time.monotonic() + HANDSHAKE_SECONDS)
        reason = "worker_running"
        while True:
            rss, _ = tree.capture()
            if rss > MAX_RSS_BYTES or shutil.disk_usage(root).free < MIN_DISK_BYTES:
                raise PilotError("Local RAM or free-disk guard limit reached")
            if worker.poll() is not None:
                result, reason = worker.returncode, "worker_exited"
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Original cooperative worker deadline reached")
            ready_path = root / "worker-ready.json"
            ready = ready_path.is_file()
            if ready:
                value = json.loads(_read(ready_path))
                if (value != dict(token=plan.token, pid=worker.pid, birth_ticks=worker_identity.birth_ticks,
                        gpu_uuid=cfg.gpu_uuid, actual_gpu_uuid=cfg.gpu_uuid.lower())
                        or local_process(worker.pid) != worker_identity):
                    raise PilotError("CUDA-ready handshake does not match the sole worker lifetime")
            rows = query_device_gpu_rows(cfg.gpu_uuid)
            _event(root, "gpu_sample", raw_rows=rows, worker_ready=ready)
            state = observe_gpu(rows, state, ready=ready)
            if not state.admitted and time.monotonic() >= ready_deadline:
                raise TimeoutError("CUDA-ready/raw-row handshake did not complete")
            if state.admitted and not (root / "worker-go.json").exists():
                _boundary(root, cfg, plan.started_at, "guard")
                _publish(root / "worker-go.json", dict(token=plan.token, assumed_host_pid=state.host_pid,
                    host_pid_identity_proven=False, gpu_provenance=PROVENANCE))
            time.sleep(POLL_SECONDS)
    except Exception as error:
        reason = repr(error)
        _event(root, "guard_stopped", reason=reason)
    finally:
        if worker is not None:
            _finish_owned(root, tree, worker, name="guard")
        _publish(root / "guard.json", dict(returncode=result, reason=reason, ended_at=time.monotonic(),
            correlation=asdict(state), host_pid_identity_proven=False, gpu_provenance=PROVENANCE,
            budget=cfg.budget(), automatic_retries=0, hard_kill_may_leave_partial_receipt=True))
    return 0 if result == 0 and reason == "worker_exited" else 1


def run_worker(plan):
    root, cfg = Path(plan.run_directory), _config(plan.config_path)
    parent = local_process(os.getppid())
    _protect_parent(parent)
    wait_for_start_gate(root / "worker-start.json", timeout=10.)
    _boundary(root, cfg, plan.started_at, "worker")
    identity = local_process(os.getpid())
    start = json.loads(_read(root / "worker-start.json"))
    if start != dict(token=plan.token, pid=identity.pid, birth_ticks=identity.birth_ticks):
        raise PilotError("Worker start gate names a different lifetime")
    # Only this role imports numerical packages or creates a CUDA context.
    from .heterogeneous_device import require_cuda_binding
    binding = require_cuda_binding(cfg.gpu_uuid)
    _publish(root / "worker-ready.json", dict(token=plan.token, pid=identity.pid,
        birth_ticks=identity.birth_ticks, gpu_uuid=cfg.gpu_uuid, actual_gpu_uuid=binding["actual_uuid"].lower()))
    end = min(plan.started_at + cfg.gpu_seconds - cfg.cleanup_seconds, time.monotonic() + HANDSHAKE_SECONDS)
    while not (root / "worker-go.json").is_file():
        if time.monotonic() >= end:
            raise TimeoutError("Guard did not admit the single raw GPU row")
        time.sleep(.05)
    def boundary_check(**kwargs):
        gate = json.loads(_read(root / "worker-go.json"))
        if (gate.get("token") != plan.token or gate.get("host_pid_identity_proven") is not False
                or gate.get("gpu_provenance") != PROVENANCE):
            raise PilotError("Missing explicit correlation-only guard admission")
        return verify_container_boundary(root / "os-boundary.json", **kwargs)
    boundary_check(config=cfg, started_at=plan.started_at,
        deadline=plan.started_at + cfg.gpu_seconds - cfg.cleanup_seconds)
    from .heterogeneous_data import CifarTrainingData
    from .heterogeneous_pilot import run_pilot
    data = CifarTrainingData(plan.data_root, split_seed=cfg.split_seed)
    receipt = run_pilot(data, root / "pilot", config=cfg, started_at=plan.started_at,
                        boundary_check=boundary_check)
    return 0 if receipt["status"] == "complete" else 1


def launch(*, config_path, data_root, run_directory):
    """One explicit launch; detached timeout owns the ordinary local tree."""
    started_at = time.monotonic()
    cfg = _config(config_path)
    if sys.platform != "linux":
        raise PilotError("Container launch requires Linux and /usr/bin/timeout")
    version = subprocess.run((TIMEOUT, "--version"), capture_output=True, text=True, check=True, timeout=2).stdout
    if not version.startswith("timeout (GNU coreutils)"):
        raise PilotError("Only the explicitly provisioned GNU timeout executable is supported")
    cwd = Path(__file__).resolve().parents[2]
    head = subprocess.run(("git", "rev-parse", "HEAD"), cwd=cwd, capture_output=True, text=True, check=True, timeout=3).stdout.strip()
    dirty = subprocess.run(("git", "status", "--porcelain"), cwd=cwd, capture_output=True, text=True, check=True, timeout=3).stdout
    if head != cfg.source_revision or dirty.strip():
        raise PilotError("Clean source checkout must match the frozen revision")
    data, root = Path(data_root).resolve(strict=True), Path(run_directory).resolve()
    if not data.is_dir():
        raise ValueError("An existing local CIFAR data directory is required")
    root.mkdir(parents=True, exist_ok=False)
    config_copy = root / "config.json"
    _publish(config_copy, asdict(cfg))
    compiled_at = time.monotonic()
    duration = math.floor((started_at + cfg.gpu_seconds - compiled_at - HARD_RESERVE_SECONDS
                            - LAUNCH_RESERVE_SECONDS) * 1000) / 1000
    plan = ContainerPlan(started_at, compiled_at, duration, secrets.token_hex(16),
        str(Path(sys.executable).absolute()), str(cwd / "scripts" / "run_heterogeneous_container_pilot.py"),
        str(config_copy), str(data), str(root), json.loads(json.dumps(asdict(cfg))))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=cfg.gpu_uuid,
        FEDCORE_PILOT_ABSOLUTE_DEADLINE=str(started_at + cfg.gpu_seconds - cfg.cleanup_seconds))
    with (root / "console.log").open("x", encoding="utf-8") as console:
        child = subprocess.Popen(plan.timeout_argv, stdin=subprocess.DEVNULL, shell=False,
            stdout=console, stderr=subprocess.STDOUT, start_new_session=True, env=environment, cwd=cwd)
    try:
        captured = local_process(child.pid)
        end = min(started_at + cfg.gpu_seconds, time.monotonic() + 2.)
        while (captured.argv != plan.timeout_argv or captured.pgid != captured.pid
                or not _timer_is_present(captured.pid)):
            if time.monotonic() >= end or child.poll() is not None:
                raise PilotError("GNU timeout failed to establish its private group")
            time.sleep(.01); captured = local_process(child.pid)
        if time.monotonic() + duration > started_at + cfg.gpu_seconds - HARD_RESERVE_SECONDS:
            raise PilotError("Timeout submission consumed its original-window launch reserve")
        _publish(root / "os-boundary.json", dict(schema="fedcore-container-timeout-v1",
                 plan=asdict(plan), timeout=asdict(captured)))
        _event(root, "launched", timeout_pid=child.pid, budget=cfg.budget(),
            gpu_provenance=PROVENANCE, host_pid_identity_proven=False,
            hard_kill_may_leave_partial_receipt=True)
        return dict(run_directory=str(root), timeout_pid=child.pid, started_at=started_at,
                    deadline_at=started_at + cfg.gpu_seconds, automatic_retries=0)
    except Exception:
        # Before publication the gate forbids CUDA. After publication,
        # supervisor -> guard -> worker PDEATHSIG protection cascades the
        # timeout's death; their live ancestry checks also close setup races.
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
            parser.error("Launch creates its own original clock and private token")
        print(json.dumps(launch(config_path=args.config, data_root=args.data_root,
                               run_directory=args.run_directory), allow_nan=False))
        return 0
    # All child arguments must agree with the exact launcher record.
    wait_for_start_gate(Path(args.run_directory) / "os-boundary.json", timeout=10.)
    plan, _ = _load_plan(args.run_directory)
    if (args.started_at != plan.started_at or args.token != plan.token or args.config != plan.config_path
            or args.data_root != plan.data_root or args.run_directory != plan.run_directory):
        raise PilotError("Child command differs from the one-shot launch record")
    return {"supervisor": run_supervisor, "guard": run_guard, "worker": run_worker}[args.mode](plan)
