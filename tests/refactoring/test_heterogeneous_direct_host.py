"""CPU-fake direct host binding; no process, GPU, SSH or host command is run."""
from dataclasses import asdict, replace
import json
import pytest

from fedcore.experiments import heterogeneous_direct_host as host
from fedcore.experiments.cv_independent_supervision import ProcessIdentity


NAMESPACE = "pid:[4026531836]"
GUARD = ProcessIdentity(101, 10.)
WORKER = ProcessIdentity(202, 20.)
OBSERVER = ProcessIdentity(303, 30.)


def request(rows=("202, 125",), owned=(WORKER,)):
    return host.DirectHostRequest(NAMESPACE, "a" * 32, rows, owned, GUARD)


def collector(req, *, namespaces=None, statuses=None, births=None):
    namespaces, statuses = namespaces or {}, statuses or {}
    default_births = {GUARD.pid: GUARD.birth, WORKER.pid: WORKER.birth, OBSERVER.pid: OBSERVER.birth}
    return host.collect_direct_host_binding(req, observer_pid=OBSERVER.pid,
        process_birth=births or (lambda pid: default_births[pid]),
        readlink=lambda path: namespaces.get(int(path.parent.parent.name), NAMESPACE),
        read_text=lambda path: statuses.get(int(path.parent.name), "NSpid:\t" + path.parent.name + "\n"))


def test_exact_guard_observer_and_gpu_host_lifetimes_are_confirmed():
    result = collector(request())
    assert result["verified"] is True and result["namespace_id"] == NAMESPACE
    assert result["guard"] == asdict(GUARD) and result["observer"] == asdict(OBSERVER)
    assert result["entries"] == [{"gpu_pid": 202, "local": asdict(WORKER)}]
    assert result["nonce"] == request().nonce and result["rows"] == list(request().rows)


@pytest.mark.parametrize("pid", [GUARD.pid, WORKER.pid, OBSERVER.pid])
def test_namespace_mismatch_for_any_participant_fails_closed(pid):
    with pytest.raises(RuntimeError, match="namespace"):
        collector(request(), namespaces={pid: "pid:[999]"})


@pytest.mark.parametrize("pid", [GUARD.pid, WORKER.pid, OBSERVER.pid])
def test_nested_or_mismatched_nspid_never_becomes_direct_host_evidence(pid):
    for status in ("NSpid:\t99\t" + str(pid) + "\n", "NSpid:\t999\n", "Name:\tworker\n"):
        with pytest.raises(RuntimeError, match="singleton NSpid"):
            collector(request(), statuses={pid: status})


def test_foreign_pid_and_birth_reuse_are_rejected_without_cleanup_of_foreign_processes():
    with pytest.raises(RuntimeError, match="foreign"):
        collector(request(rows=("404, 100",)))
    with pytest.raises(RuntimeError, match="birth"):
        collector(request(), births=lambda pid: {101: 10., 202: 21., 303: 30.}[pid])
    counts = {}
    def changing(pid):
        counts[pid] = counts.get(pid, 0) + 1
        original = {101: 10., 202: 20., 303: 30.}[pid]
        return original + (1. if pid == 202 and counts[pid] > 1 else 0.)
    with pytest.raises(RuntimeError, match="changed"):
        collector(request(), births=changing)


def test_empty_gpu_rows_still_require_direct_guard_and_observer_namespace_evidence():
    result = collector(request(rows=(), owned=()))
    assert result["entries"] == [] and result["verified"] is True
    with pytest.raises(RuntimeError, match="namespace"):
        collector(request(rows=(), owned=()), namespaces={GUARD.pid: "pid:[123]"})


def install_provider_fakes(monkeypatch, transform=lambda response: response):
    monkeypatch.setattr(host, "_local_identity", lambda: GUARD)
    monkeypatch.setattr(host.os, "readlink", lambda path: NAMESPACE)
    calls = []
    def transport(command, raw, timeout):
        value = json.loads(raw)
        req = host.DirectHostRequest(value["namespace_id"], value["nonce"], tuple(value["rows"]),
            tuple(ProcessIdentity(**identity) for identity in value["owned"]), ProcessIdentity(**value["guard"]))
        calls.append((command, raw, timeout))
        return json.dumps(transform(collector(req)))
    monkeypatch.setattr(host, "_run_bounded", transport)
    return calls


def test_provider_binds_nonce_rows_namespace_and_exact_owned_identity(monkeypatch):
    calls = install_provider_fakes(monkeypatch)
    provider = host.DirectHostBindingProvider(("/usr/bin/python3", "-m", host.__name__), NAMESPACE)
    binding = provider(("202, 125",), (WORKER,))
    assert binding.verified and binding.entries[0].local == WORKER
    assert len(calls) == 1 and calls[0][2] == 3.


@pytest.mark.parametrize("changes", [
    {"nonce": "0" * 32}, {"namespace_id": "pid:[999]"}, {"rows": ["202, 999"]},
    {"guard": {"pid": 101, "birth": 999.}}, {"verified": False},
    {"entries": [{"gpu_pid": 202, "local": {"pid": 202, "birth": 21.}}]},
    {"entries": [{"gpu_pid": 202, "local": {"pid": 404, "birth": 20.}}]},
])
def test_provider_rejects_replay_mismatch_foreign_and_partial_responses(monkeypatch, changes):
    install_provider_fakes(monkeypatch, lambda result: {**result, **changes})
    provider = host.DirectHostBindingProvider(("/usr/bin/python3", "observer.py"), NAMESPACE)
    with pytest.raises(RuntimeError):
        provider(("202, 125",), (WORKER,))


def test_guard_namespace_must_match_configuration_before_any_transport(monkeypatch):
    calls = install_provider_fakes(monkeypatch)
    monkeypatch.setattr(host.os, "readlink", lambda path: "pid:[999]")
    provider = host.DirectHostBindingProvider(("/usr/bin/python3", "observer.py"), NAMESPACE)
    with pytest.raises(RuntimeError, match="provisioned"):
        provider(("202, 125",), (WORKER,))
    assert not calls


@pytest.mark.parametrize("command", [("/usr/bin/ssh", "host", "observer.py"),
    ("/usr/bin/docker", "exec", "container", "observer.py"), ("/bin/bash", "-c", "observer.py")])
def test_brokered_observer_cannot_claim_direct_cgroup_coverage(command):
    with pytest.raises(ValueError, match="broker"):
        host.DirectHostBindingProvider(command, NAMESPACE)
