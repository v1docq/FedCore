"""Linux OS deadline boundary for a direct, trusted guard + worker tree.

This module does NOT configure or probe the A100 host on import. The supported
path starts the guard directly in a private transient user systemd service in
the SAME PID namespace as its user manager. All worker descendants must stay in
that service cgroup. SSH, docker/podman exec/run and shell/broker wrappers are
explicitly outside this profile: killing their clients cannot kill broker-owned
workers. Trusted guard code must not delegate work through such APIs either.

RuntimeMaxSec + TimeoutStartSec + TimeoutStopSec fit the original absolute
budget, including submission reserve and cleanup. KillMode=control-group and
SendSIGKILL=yes give the service manager ownership of termination even if the
Python launcher disappears. This is an OS signal boundary, not a guarantee that
a broken kernel/driver can reap an uninterruptible task. No PGID, guessed PID,
ordinary Python timer fallback, automatic retry or privileged reconfiguration is
used. Real host compatibility remains unverified until separately authorized.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import selectors
import subprocess
import sys
import time
from typing import Callable


class DeadlineError(RuntimeError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


@dataclass(frozen=True)
class DeadlineBudget:
    started_at: float
    total_seconds: float
    cleanup_seconds: float = 10.
    start_timeout_seconds: float = 5.

    def __post_init__(self):
        if (type(self.started_at) not in (int, float) or not math.isfinite(self.started_at)
                or self.started_at < 0 or any(not _positive(v) for v in
                (self.total_seconds, self.cleanup_seconds, self.start_timeout_seconds))
                or not math.isfinite(self.started_at + self.total_seconds)
                or any(v < .001 or v > (2**63 - 1) / 1_000_000 for v in
                       (self.total_seconds, self.cleanup_seconds, self.start_timeout_seconds))
                or self.total_seconds <= self.cleanup_seconds + self.start_timeout_seconds):
            raise ValueError("Original monotonic start and finite whole-window reserves required")

    @property
    def deadline_at(self):
        return self.started_at + self.total_seconds


@dataclass(frozen=True)
class DeadlineRequest:
    argv: tuple[str, ...]
    working_directory: str
    token: str
    transport_timeout_seconds: float = 3.
    launch_kind: str = "direct_guard_worker"
    gpu_uuid: str | None = None

    def __post_init__(self):
        if self.gpu_uuid is not None:
            from .heterogeneous_device import validate_gpu_uuid
            validate_gpu_uuid(self.gpu_uuid)
        if (type(self.argv) is not tuple or not self.argv or len(self.argv) > 256
                or any(type(v) is not str or not v or any(c in v for c in ("\x00", "\n", "\r", "$", "%"))
                       or len(v) > 8192 for v in self.argv)
                or not self.argv[0].startswith("/")
                or type(self.working_directory) is not str or not self.working_directory.startswith("/")
                or any(c in self.working_directory for c in ("\x00", "\n", "\r", "%"))
                or type(self.token) is not str or re.fullmatch(r"[a-f0-9]{32}", self.token) is None
                or not _positive(self.transport_timeout_seconds) or not .5 <= self.transport_timeout_seconds <= 5
                or self.launch_kind != "direct_guard_worker"):
            raise ValueError("Explicit absolute direct argv, private token and bounded local transport required")
        brokers = {"ssh", "sshd", "docker", "podman", "nerdctl", "kubectl", "srun", "sbatch",
                   "systemd-run", "systemctl", "sudo", "su", "nsenter", "unshare", "sh", "bash", "dash", "zsh", "fish"}
        if any(Path(v).name.lower() in brokers for v in self.argv):
            raise ValueError("Brokered or shell launches do not place the real worker in the owned cgroup")

    @property
    def unit(self):
        return "fedcore-heterogeneous-" + self.token + ".service"

    @property
    def description(self):
        return "fedcore-owned-deadline:" + self.token


@dataclass(frozen=True)
class DeadlinePlan:
    request: DeadlineRequest
    deadline_at: float
    compiled_at: float
    runtime_microseconds: int
    start_microseconds: int
    stop_microseconds: int
    argv: tuple[str, ...]


def compile_launch(request, budget, *, now):
    """Pure plan: launch transport, activation and stop grace consume this window."""
    if type(request) is not DeadlineRequest or type(budget) is not DeadlineBudget:
        raise ValueError("Typed deadline request and original budget required")
    if type(now) not in (int, float) or not math.isfinite(now) or now < budget.started_at:
        raise ValueError("Current monotonic time cannot precede the original budget")
    remaining = budget.deadline_at - now
    runtime = math.floor((remaining - request.transport_timeout_seconds - budget.start_timeout_seconds
                          - budget.cleanup_seconds) * 1_000_000)
    if runtime < 1000:
        raise DeadlineError("DeadlineExhausted", "No runtime remains after submission, start and cleanup reserves")
    start, stop = (math.floor(v * 1_000_000) for v in (budget.start_timeout_seconds, budget.cleanup_seconds))
    properties = ("Type=exec", "Restart=no", "RemainAfterExit=no", "KillMode=control-group",
                  "SendSIGKILL=yes", "FinalKillSignal=SIGKILL", "TimeoutStopFailureMode=kill",
                  "Delegate=no", "ProtectControlGroups=yes", "NoNewPrivileges=yes",
                  "RuntimeMaxSec=" + str(runtime) + "us", "TimeoutStartSec=" + str(start) + "us",
                  "TimeoutStopSec=" + str(stop) + "us", "WorkingDirectory=" + request.working_directory,
                  "Description=" + request.description)
    if request.gpu_uuid is not None:
        properties += ('Environment=CUDA_VISIBLE_DEVICES=' + request.gpu_uuid,)
    argv = ("systemd-run", "--user", "--quiet", "--unit=" + request.unit) + tuple(
        "--property=" + value for value in properties) + ("--",) + request.argv
    return DeadlinePlan(request, budget.deadline_at, now, runtime, start, stop, argv)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class BoundedLocalTransport:
    """Bound BOTH pipe memory and wall time; terminate only our own local client."""
    def __init__(self, max_output_bytes=65536):
        if type(max_output_bytes) is not int or not 1024 <= max_output_bytes <= 1024**2:
            raise ValueError("Finite small response limit required")
        self.max_output_bytes = max_output_bytes

    def __call__(self, argv, *, timeout):
        if not _positive(timeout) or timeout > 5 or timeout <= .25:
            raise ValueError("Bounded local transport timeout required")
        end = time.monotonic() + timeout - .25  # Client kill/reap reserve stays inside transport time.
        process = subprocess.Popen(tuple(argv), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, shell=False, close_fds=True,
                                   env={**os.environ, "LC_ALL": "C"})
        streams, count = {"stdout": bytearray(), "stderr": bytearray()}, 0
        try:
            with selectors.DefaultSelector() as selector:
                for name in streams:
                    pipe = getattr(process, name)
                    os.set_blocking(pipe.fileno(), False)
                    selector.register(pipe, selectors.EVENT_READ, name)
                while selector.get_map():
                    left = end - time.monotonic()
                    if left <= 0:
                        raise DeadlineError("TransportTimeout", "Local systemd client exceeded its deadline")
                    for key, _ in selector.select(left):
                        chunk = os.read(key.fd, min(8192, self.max_output_bytes - count + 1))
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        count += len(chunk)
                        if count > self.max_output_bytes:
                            raise DeadlineError("TransportOutputLimit", "Local response exceeds bounded stdout/stderr capacity")
                        streams[key.data].extend(chunk)
                left = end - time.monotonic()
                if left <= 0:
                    raise DeadlineError("TransportTimeout", "Local client did not exit inside its deadline")
                returncode = process.wait(timeout=left)
            return CommandResult(returncode, streams["stdout"].decode("utf-8", errors="strict"),
                                 streams["stderr"].decode("utf-8", errors="strict"))
        except subprocess.TimeoutExpired as error:
            raise DeadlineError("TransportTimeout", "Local client did not exit inside its deadline") from error
        finally:
            if process.poll() is None:
                process.kill()  # Only this Popen client, never a worker PID/PGID.
                try:
                    process.wait(timeout=.25)
                except subprocess.TimeoutExpired:
                    pass
            process.stdout.close()
            process.stderr.close()


@dataclass(frozen=True)
class Preconditions:
    manager_control_group: str
    manager_pid: int
    pid_namespace: str
    systemd_version: int


@dataclass(frozen=True)
class DeadlineReceipt:
    plan: DeadlinePlan
    invocation_id: str
    control_group: str
    active_state: str


@dataclass(frozen=True)
class DeadlineFailure:
    code: str
    message: str
    plan: DeadlinePlan | None = None
    may_be_running: bool = False
    stop_requested: bool = False


def _properties(result):
    if type(result) is not CommandResult or len(result.stdout.encode()) + len(result.stderr.encode()) > 65536:
        raise DeadlineError("InvalidTransport", "Typed bounded command response required")
    if result.returncode != 0:
        raise DeadlineError("SystemdCommandFailed", result.stderr[:2048] or result.stdout[:2048])
    values = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            raise DeadlineError("InvalidSystemdResponse", "Expected systemctl key=value properties")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise DeadlineError("InvalidSystemdResponse", "Duplicate or empty property name")
        values[key] = value
    return values


class SystemdDeadlineLauncher:
    """Local-only launcher. Inject transport/filesystem for CPU precondition fakes."""
    def __init__(self, *, transport=None, clock=time.monotonic, platform=None,
                 cgroup_root="/sys/fs/cgroup", proc_root="/proc", uid=None,
                 read_text=None, readlink=None, exists=None, writable=None):
        self.transport = transport or BoundedLocalTransport()
        self.clock = clock
        self.platform = sys.platform if platform is None else platform
        self.cgroup_root, self.proc_root = Path(cgroup_root), Path(proc_root)
        self.uid = os.getuid() if uid is None and hasattr(os, "getuid") else uid
        self.read_text = read_text or _read_bounded
        self.readlink = readlink or os.readlink
        self.exists = exists or (lambda path: Path(path).exists())
        self.writable = writable or (lambda path: os.access(path, os.W_OK))

    def _command(self, argv, request, budget):
        left = budget.deadline_at - self.clock()
        if left <= .25:
            raise DeadlineError("DeadlineExhausted", "Original whole-window deadline elapsed")
        timeout = min(request.transport_timeout_seconds, left)
        result = self.transport(tuple(argv), timeout=timeout)
        if self.clock() > budget.deadline_at:
            raise DeadlineError("TransportTimeout", "Response arrived after the original deadline")
        if type(result) is not CommandResult or len(result.stdout.encode()) + len(result.stderr.encode()) > 65536:
            raise DeadlineError("InvalidTransport", "Bounded typed transport response required")
        return result

    def check_preconditions(self, request, budget):
        if self.platform != "linux" or type(self.uid) is not int or self.uid < 0:
            raise DeadlineError("UnsupportedPlatform", "A direct Linux user systemd namespace is required")
        if not self.exists(self.cgroup_root / "cgroup.controllers"):
            raise DeadlineError("CgroupV2Required", "Unified cgroup v2 is mandatory; no PGID fallback")
        own_cgroup = self.read_text(self.proc_root / "self/cgroup")
        if len(own_cgroup) > 65536 or not re.fullmatch(r"0::/[^\n]*\n?", own_cgroup):
            raise DeadlineError("CgroupV2Required", "Require one unified self cgroup entry")
        # System unit properties prove user-manager delegation and identify its
        # real PID namespace; manager existence is not inferred from a CLI file.
        service = "user@" + str(self.uid) + ".service"
        values = _properties(self._command(("systemctl", "show", "--property=ActiveState", "--property=Delegate",
            "--property=ControlGroup", "--property=MainPID", "--", service), request, budget))
        group, pid = values.get("ControlGroup", ""), values.get("MainPID", "")
        if (values.get("ActiveState") != "active" or values.get("Delegate") != "yes"
                or not re.fullmatch(r"/[A-Za-z0-9_@.\-/]+", group) or ".." in group.split("/")
                or not pid.isascii() or not pid.isdecimal() or int(pid) < 1):
            raise DeadlineError("UndelegatedManager", "An active delegated user manager and exact control group are required")
        current_ns = self.readlink(self.proc_root / "self/ns/pid")
        manager_ns = self.readlink(self.proc_root / pid / "ns/pid")
        if current_ns != manager_ns:
            raise DeadlineError("NamespaceMismatch", "Guard and systemd manager must share the PID namespace")
        directory = self.cgroup_root / group.lstrip("/")
        if (not self.exists(directory / "cgroup.controllers") or not self.exists(directory / "cgroup.kill")
                or not self.writable(directory / "cgroup.procs") or not self.writable(directory / "cgroup.subtree_control")):
            raise DeadlineError("UndelegatedCgroup", "Writable delegated cgroup v2 hierarchy with cgroup.kill is required")
        manager = _properties(self._command(("systemctl", "--user", "show", "--property=Version"), request, budget))
        version = re.match(r"(\d+)", manager.get("Version", ""))
        if version is None or int(version.group(1)) < 250:
            raise DeadlineError("SystemdVersion", "This transient-service profile requires systemd >=250")
        return Preconditions(group, int(pid), current_ns, int(version.group(1)))

    def launch(self, request, budget):
        plan, submitted, owned_receipt = None, False, None
        try:
            if type(request) is not DeadlineRequest or type(budget) is not DeadlineBudget:
                raise DeadlineError("InvalidDeadlineRequest", "Typed direct request and original budget required")
            preconditions = self.check_preconditions(request, budget)
            plan = compile_launch(request, budget, now=self.clock())
            submitted = True
            result = self._command(plan.argv, request, budget)
            if result.returncode != 0:
                raise DeadlineError("SystemdLaunchFailed", result.stderr[:2048])
            fields = ("InvocationID", "Description", "ControlGroup", "ActiveState", "KillMode", "SendSIGKILL",
                      "RuntimeMaxUSec", "TimeoutStartUSec", "TimeoutStopUSec", "Type", "Delegate", "ProtectControlGroups",
                      "Restart", "FinalKillSignal", "TimeoutStopFailureMode", "NoNewPrivileges", "WorkingDirectory")
            values = _properties(self._command(("systemctl", "--user", "show") + tuple(
                "--property=" + field for field in fields) + ("--", request.unit), request, budget))
            group = values.get("ControlGroup", "")
            ownership = (values.get("Description") == request.description
                         and re.fullmatch(r"[a-f0-9]{32}", values.get("InvocationID", "")) is not None
                         and group.startswith(preconditions.manager_control_group.rstrip("/") + "/")
                         and group.endswith("/" + request.unit))
            if ownership:
                owned_receipt = DeadlineReceipt(plan, values["InvocationID"], group, values.get("ActiveState", "unknown"))
            if (not ownership
                    or values.get("KillMode") != "control-group" or values.get("SendSIGKILL") != "yes"
                    or values.get("Type") != "exec" or values.get("Delegate") != "no"
                    or values.get("ProtectControlGroups") != "yes"
                    or values.get("Restart") != "no" or values.get("FinalKillSignal") not in ("9", "SIGKILL")
                    or values.get("TimeoutStopFailureMode") != "kill" or values.get("NoNewPrivileges") != "yes"
                    or values.get("WorkingDirectory") != request.working_directory
                    or values.get("ActiveState") not in ("active", "activating", "inactive", "failed")):
                raise DeadlineError("UnverifiedBoundary", "Transient unit ownership/namespace/kill contract is not established")
            # systemctl renders these durations as human-readable time spans.
            for field, limit in (("RuntimeMaxUSec", plan.runtime_microseconds),
                                 ("TimeoutStartUSec", plan.start_microseconds), ("TimeoutStopUSec", plan.stop_microseconds)):
                actual = parse_systemd_duration(values.get(field, ""))
                if actual <= 0 or actual > limit:
                    raise DeadlineError("UnverifiedDeadline", "Manager did not apply the bounded " + field)
            return DeadlineReceipt(plan, values["InvocationID"], group, values["ActiveState"])
        except DeadlineError as error:
            stopped = self._cleanup_verified(owned_receipt, budget)
            return DeadlineFailure(error.code, str(error), plan, submitted, stopped)
        except (ValueError, OSError, UnicodeError) as error:
            stopped = self._cleanup_verified(owned_receipt, budget)
            return DeadlineFailure("DeadlineTransportFailed", str(error), plan, submitted, stopped)

    def _cleanup_verified(self, receipt, budget):
        if receipt is None:
            return False
        try:
            self.stop_owned(receipt, budget)
            return True  # Submission only; never claim completed physical termination.
        except (DeadlineError, ValueError, OSError, UnicodeError):
            return False

    def stop_owned(self, receipt, budget):
        """Only a verified exact unit invocation can be stopped; never a PID/PGID."""
        if type(receipt) is not DeadlineReceipt:
            raise ValueError("A verified private unit receipt is required")
        request = receipt.plan.request
        values = _properties(self._command(("systemctl", "--user", "show", "--property=Description",
            "--property=InvocationID", "--", request.unit), request, budget))
        if values.get("Description") != request.description or values.get("InvocationID") != receipt.invocation_id:
            raise DeadlineError("OwnershipChanged", "The current service invocation is not the launched owned tree")
        result = self._command(("systemctl", "--user", "stop", "--no-block", "--", request.unit), request, budget)
        if result.returncode != 0:
            raise DeadlineError("StopSubmissionFailed", result.stderr[:2048])
        return result


def _read_bounded(path):
    with Path(path).open("r", encoding="ascii") as stream:
        return stream.read(65537)


def parse_systemd_duration(value):
    """Parse only bounded systemctl time spans; infinity/unknown units fail closed."""
    if type(value) is not str or not value or value == "infinity":
        raise DeadlineError("InvalidDuration", "Finite manager duration required")
    factors = {"us": 1, "ms": 1000, "s": 1_000_000, "min": 60_000_000, "h": 3_600_000_000,
               "d": 86_400_000_000, "w": 604_800_000_000}
    tokens = re.findall(r"(\d+(?:\.\d+)?)(us|ms|min|s|h|d|w)", value)
    if not tokens or re.sub(r"\s+", "", value) != "".join(number + unit for number, unit in tokens):
        raise DeadlineError("InvalidDuration", "Unknown systemctl duration format")
    result = sum(float(number) * factors[unit] for number, unit in tokens)
    if not math.isfinite(result) or result > 2**63 - 1:
        raise DeadlineError("InvalidDuration", "Duration exceeds finite integer range")
    return round(result)
