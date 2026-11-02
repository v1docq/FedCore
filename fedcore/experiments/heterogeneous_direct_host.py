"""Authenticated direct-host PID binding for the systemd deadline profile.

The guard, its worker tree and this explicitly provisioned HOST observer must
share the configured host PID namespace. ``namespace_id`` is trusted provisioned
configuration, never discovered as an approval or inferred from equal integers.
The observer confirms namespace, singleton host NSpid and exact process births.

This provider is compatible with a direct host systemd service; the existing
Docker provider describes a different launch profile. SSH/docker/podman brokers
and namespace wrappers are rejected, not treated as covered cgroup descendants.
Importing performs no observation, GPU call, transport or launch. A host must be
configured and authorized separately; neither CPU fakes nor these contracts
prove current A100 compatibility. Run the CLI explicitly on the trusted host.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import secrets
import sys
import time

from .cv_independent_supervision import NamespaceBinding, PidBinding, ProcessIdentity
from .heterogeneous_host import _run_bounded


def _namespace(value):
    return (type(value) is str and re.fullmatch(r"pid:\[[0-9]{1,20}\]", value) is not None
            and int(value[5:-1]) > 0)


def _reported(rows):
    if (type(rows) is not tuple or len(rows) > 1 or any(type(row) is not str or not row.strip()
            or len(row) > 4096 for row in rows)):
        raise ValueError("At most one bounded, explicit raw GPU process row is supported")
    result = []
    for row in rows:
        parts = row.split(",")
        token = parts[0].strip()
        if (len(parts) != 2 or not token.isascii() or not token.isdecimal()
                or len(token) > 20 or int(token) < 1):
            raise ValueError("Malformed raw GPU process row")
        result.append(int(token))
    return tuple(result)


@dataclass(frozen=True)
class DirectHostRequest:
    namespace_id: str
    nonce: str
    rows: tuple[str, ...]
    owned: tuple[ProcessIdentity, ...]
    guard: ProcessIdentity

    def __post_init__(self):
        if (not _namespace(self.namespace_id) or type(self.nonce) is not str
                or re.fullmatch(r"[a-f0-9]{32}", self.nonce) is None
                or type(self.owned) is not tuple or any(type(p) is not ProcessIdentity for p in self.owned)
                or len(self.owned) > 512 or len({p.pid for p in self.owned}) != len(self.owned)
                or type(self.guard) is not ProcessIdentity):
            raise ValueError("Exact namespace, nonce, guard lifetime and distinct owned lifetimes required")
        _reported(self.rows)


def _local_identity():
    import psutil
    return ProcessIdentity(os.getpid(), psutil.Process(os.getpid()).create_time())


@dataclass(frozen=True)
class DirectHostBindingProvider:
    command: tuple[str, ...]
    namespace_id: str
    timeout: float = 3.

    def __post_init__(self):
        if (type(self.command) is not tuple or not self.command
                or any(type(c) is not str or not c or "\x00" in c for c in self.command)
                or not self.command[0].startswith("/")
                or not _namespace(self.namespace_id) or type(self.timeout) not in (int, float)
                or not math.isfinite(self.timeout) or not 0 < self.timeout <= 5):
            raise ValueError("Explicit trusted direct host command, provisioned namespace and bounded timeout required")
        brokers = {"ssh", "sshd", "docker", "podman", "nerdctl", "kubectl", "srun", "sbatch", "sudo", "su",
                   "nsenter", "unshare", "sh", "bash", "dash", "zsh", "fish", "systemd-run", "systemctl"}
        if any(Path(c).name.lower() in brokers for c in self.command):
            raise ValueError("Direct-host observer must remain a local cgroup descendant, not a broker client")

    def __call__(self, rows, owned):
        guard = _local_identity()
        if os.readlink("/proc/self/ns/pid") != self.namespace_id:
            raise RuntimeError("Guard is not in the provisioned direct-host PID namespace")
        request = DirectHostRequest(self.namespace_id, secrets.token_hex(16), tuple(rows), tuple(owned), guard)
        began = time.monotonic()
        raw = _run_bounded(self.command, json.dumps(asdict(request), allow_nan=False), self.timeout)
        if time.monotonic() - began > self.timeout or type(raw) is not str or len(raw.encode()) > 65536:
            raise RuntimeError("Direct-host evidence is late or exceeds bounded response capacity")
        if _local_identity() != guard or os.readlink("/proc/self/ns/pid") != self.namespace_id:
            raise RuntimeError("Guard lifetime or PID namespace changed during observation")
        value = json.loads(raw)
        fields = {"schema", "namespace_id", "nonce", "rows", "guard", "observer", "entries", "verified", "evidence"}
        if (type(value) is not dict or set(value) != fields or value["schema"] != "fedcore-direct-host-binding-v1"
                or value["namespace_id"] != request.namespace_id or value["nonce"] != request.nonce
                or value["rows"] != list(request.rows) or value["guard"] != asdict(guard)
                or value["verified"] is not True or type(value["entries"]) is not list
                or len(value["entries"]) != len(request.rows)
                or type(value["observer"]) is not dict or set(value["observer"]) != {"pid", "birth"}
                or type(value["evidence"]) is not str or not value["evidence"].strip()):
            raise RuntimeError("Untrusted, mismatched or replayed direct-host response")
        ProcessIdentity(**value["observer"])
        entries = []
        for item in value["entries"]:
            if (type(item) is not dict or set(item) != {"gpu_pid", "local"} or type(item["local"]) is not dict
                    or set(item["local"]) != {"pid", "birth"}):
                raise RuntimeError("Malformed direct-host owned lifetime")
            binding = PidBinding(item["gpu_pid"], ProcessIdentity(**item["local"]))
            if binding.gpu_pid != binding.local.pid or binding.local not in request.owned:
                raise RuntimeError("Direct GPU PID is not an exact owned host process lifetime")
            entries.append(binding)
        if tuple(entry.gpu_pid for entry in entries) != _reported(request.rows):
            raise RuntimeError("Every raw GPU PID needs fresh exact direct-host evidence")
        return NamespaceBinding(True, tuple(entries), value["evidence"])


def _read_bounded(path):
    with Path(path).open("r", encoding="ascii") as stream:
        text = stream.read(65537)
    if len(text) > 65536:
        raise RuntimeError("Host procfs record exceeds the bounded profile")
    return text


def collect_direct_host_binding(request, *, proc_root="/proc", process_birth=None,
                                observer_pid=None, readlink=None, read_text=None):
    """Explicit host observer. Verify guard, observer and GPU PID before mapping.

    The host command is authenticated/provisioned by the caller. This function
    does not turn an arbitrary local/container procfs into trusted host evidence.
    Every namespace and birth is checked again to detect reuse during the probe.
    """
    if type(request) is not DirectHostRequest:
        raise ValueError("Validated direct-host request required")
    if process_birth is None:
        import psutil
        process_birth = lambda pid: psutil.Process(pid).create_time()
    readlink = readlink or os.readlink
    read_text = read_text or _read_bounded
    observer_pid = os.getpid() if observer_pid is None else observer_pid
    observer = ProcessIdentity(observer_pid, process_birth(observer_pid))
    root = Path(proc_root)
    if request.guard.pid == observer.pid and request.guard != observer:
        raise RuntimeError("Guard and observer report conflicting lifetimes for one PID")
    expected = {request.guard.pid: request.guard, observer.pid: observer}
    for pid in _reported(request.rows):
        owned = next((identity for identity in request.owned if identity.pid == pid), None)
        if owned is None:
            raise RuntimeError("Raw GPU PID is foreign to the launched owned tree")
        if pid in expected and expected[pid] != owned:
            raise RuntimeError("GPU ownership conflicts with a guard/observer process lifetime")
        expected[pid] = owned
    for pid, identity in expected.items():
        if process_birth(pid) != identity.birth:
            raise RuntimeError("Host process birth differs from recorded ownership")
        if readlink(root / str(pid) / "ns/pid") != request.namespace_id:
            raise RuntimeError("Guard, observer and GPU process must share the provisioned host PID namespace")
        status = read_text(root / str(pid) / "status")
        if type(status) is not str or len(status) > 65536:
            raise RuntimeError("Host procfs record exceeds the bounded profile")
        lines = [line for line in status.splitlines() if line.startswith("NSpid:")]
        ids = lines[0].split()[1:] if len(lines) == 1 else []
        if len(ids) != 1 or not ids[0].isascii() or not ids[0].isdecimal() or int(ids[0]) != pid:
            raise RuntimeError("Direct host evidence requires singleton NSpid equal to the actual host PID")
    for pid, identity in expected.items():
        if process_birth(pid) != identity.birth or readlink(root / str(pid) / "ns/pid") != request.namespace_id:
            raise RuntimeError("Host process lifetime or namespace changed during observation")
    entries = [{"gpu_pid": pid, "local": asdict(expected[pid])} for pid in _reported(request.rows)]
    return dict(schema="fedcore-direct-host-binding-v1", namespace_id=request.namespace_id,
                nonce=request.nonce, rows=list(request.rows), guard=asdict(request.guard), observer=asdict(observer),
                entries=entries, verified=True,
                evidence="authenticated direct host observer + exact shared PID namespace + singleton host NSpid + stable owned process births")


def main():
    raw = sys.stdin.read(65537)
    if len(raw.encode()) > 65536:
        raise ValueError("Direct-host request exceeds bounded transport limit")
    value = json.loads(raw)
    if (type(value) is not dict or set(value) != {"namespace_id", "nonce", "rows", "owned", "guard"}
            or type(value["rows"]) is not list or type(value["owned"]) is not list
            or type(value["guard"]) is not dict or set(value["guard"]) != {"pid", "birth"}
            or any(type(p) is not dict or set(p) != {"pid", "birth"} for p in value["owned"])):
        raise ValueError("Exact direct-host request fields required")
    request = DirectHostRequest(value["namespace_id"], value["nonce"], tuple(value["rows"]),
                                tuple(ProcessIdentity(**p) for p in value["owned"]), ProcessIdentity(**value["guard"]))
    print(json.dumps(collect_direct_host_binding(request), allow_nan=False))


if __name__ == "__main__":
    main()
