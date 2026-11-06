"""CPU fakes only: never execute systemd, a worker, SSH or container commands."""
from dataclasses import replace
import pytest

from fedcore.experiments.heterogeneous_deadline import (
    CommandResult, DeadlineBudget, DeadlineError, DeadlineFailure, DeadlineReceipt,
    DeadlineRequest, SystemdDeadlineLauncher, compile_launch, parse_systemd_duration)


class Clock:
    value = 100.
    def __call__(self):
        return self.value


class FakeTransport:
    group = "/user.slice/user-1000.slice/user@1000.service"
    def __init__(self, clock):
        self.clock = clock
        self.calls, self.properties = [], {}
        self.delegate = "yes"
        self.version = "252.38-1"
        self.invocation = "a" * 32
        self.override = {}
        self.advance = 1.
        self.flood = False
        self.launch_failure = False

    def __call__(self, argv, *, timeout):
        self.calls.append((argv, timeout))
        self.clock.value += self.advance
        if self.flood:
            return CommandResult(0, "x" * 65537, "")
        if argv[0] == "systemd-run":
            self.properties = dict(value[len("--property="):].split("=", 1)
                                   for value in argv if value.startswith("--property="))
            if self.launch_failure:
                raise DeadlineError("TransportTimeout", "submission status is ambiguous")
            return CommandResult(0, "", "")
        if "--property=Version" in argv:
            return CommandResult(0, "Version=" + self.version + "\n", "")
        if "user@1000.service" in argv:
            return CommandResult(0, "ActiveState=active\nDelegate=" + self.delegate
                                 + "\nControlGroup=" + self.group + "\nMainPID=123\n", "")
        if "stop" in argv:
            return CommandResult(0, "", "")
        values = dict(self.properties)
        values.update(InvocationID=self.invocation, ControlGroup=self.group + "/app.slice/" + argv[-1],
                      ActiveState="active", RuntimeMaxUSec=values.get("RuntimeMaxSec", "infinity"),
                      TimeoutStartUSec=values.get("TimeoutStartSec", "infinity"),
                      TimeoutStopUSec=values.get("TimeoutStopSec", "infinity"))
        values.update(self.override)
        fields = [part[len("--property="):] for part in argv if part.startswith("--property=")]
        return CommandResult(0, "".join(field + "=" + values.get(field, "") + "\n" for field in fields), "")


def request():
    return DeadlineRequest(("/usr/bin/python3", "-m", "fedcore.experiments.cv_independent_guard", "--worker", "worker.json"),
                           "/srv/fedcore", "1" * 32)


def setup(**overrides):
    clock = Clock()
    transport = FakeTransport(clock)
    options = dict(transport=transport, clock=clock, platform="linux", uid=1000,
                   exists=lambda path: True, writable=lambda path: True,
                   read_text=lambda path: "0::/user.slice/user-1000.slice/session-1.scope\n",
                   readlink=lambda path: "pid:[4026531836]")
    options.update(overrides)
    return clock, transport, SystemdDeadlineLauncher(**options), DeadlineBudget(100., 60., 10., 5.)


def test_pure_command_preserves_literal_argv_and_keeps_all_reserves_in_original_window():
    r, b = request(), DeadlineBudget(100., 60., 10., 5.)
    r = replace(r, argv=r.argv + ("literal space;rm -rf",))
    plan = compile_launch(r, b, now=104.)
    assert plan.argv[plan.argv.index("--") + 1:] == r.argv
    assert plan.runtime_microseconds == 38_000_000
    assert "--property=KillMode=control-group" in plan.argv
    assert "--property=SendSIGKILL=yes" in plan.argv
    assert "--property=TimeoutStopFailureMode=kill" in plan.argv
    assert "--property=ProtectControlGroups=yes" in plan.argv
    assert "--wait" not in plan.argv and "--scope" not in plan.argv
    assert plan.compiled_at + r.transport_timeout_seconds + (
        plan.runtime_microseconds + plan.start_microseconds + plan.stop_microseconds) / 1e6 <= b.deadline_at


@pytest.mark.parametrize("argv", [
    ("/usr/bin/docker", "exec", "container", "python", "worker.py"),
    ("/usr/bin/ssh", "host", "python worker.py"),
    ("/usr/bin/python3", "guard.py", "--worker", "/usr/bin/podman", "exec", "container"),
    ("/bin/bash", "-c", "python worker.py"),
    ("/usr/bin/python3", "$WORKER"), ("/usr/bin/python3", "%n"),
])
def test_broker_shell_and_manager_expansion_boundaries_are_rejected(argv):
    with pytest.raises(ValueError):
        replace(request(), argv=argv)
    with pytest.raises(ValueError):
        replace(request(), launch_kind="docker_exec")


def test_direct_service_is_verified_before_receipt_and_stop_binds_exact_invocation():
    clock, transport, launcher, budget = setup()
    receipt = launcher.launch(request(), budget)
    assert type(receipt) is DeadlineReceipt
    assert receipt.plan.compiled_at == 102.  # Both precondition calls consumed original time.
    assert receipt.plan.runtime_microseconds == 40_000_000
    assert receipt.control_group.endswith("/" + request().unit)
    assert all(0 < timeout <= 3 for _, timeout in transport.calls)
    launcher.stop_owned(receipt, budget)
    assert transport.calls[-1][0] == ("systemctl", "--user", "stop", "--no-block", "--", request().unit)
    transport.invocation = "b" * 32
    before = len(transport.calls)
    with pytest.raises(DeadlineError, match="not the launched"):
        launcher.stop_owned(receipt, budget)
    assert len(transport.calls) == before + 1  # Read only; no stop of a reused unit.


@pytest.mark.parametrize("options,code", [
    ({"platform": "win32"}, "UnsupportedPlatform"),
    ({"exists": lambda path: False}, "CgroupV2Required"),
    ({"read_text": lambda path: "1:name=systemd:/\n"}, "CgroupV2Required"),
    ({"writable": lambda path: False}, "UndelegatedCgroup"),
    ({"readlink": lambda path: "pid:[1]" if str(path).replace('\\', '/').endswith('/self/ns/pid') else "pid:[2]"}, "NamespaceMismatch"),
])
def test_preconditions_fail_closed_without_any_launch(options, code):
    _, transport, launcher, budget = setup(**options)
    result = launcher.launch(request(), budget)
    assert type(result) is DeadlineFailure and result.code == code and not result.may_be_running
    assert not any(argv[0] == "systemd-run" for argv, _ in transport.calls)


def test_non_delegated_or_old_manager_and_overflow_do_not_trigger_fallback():
    for change, code in (({"delegate": "no"}, "UndelegatedManager"),
                         ({"version": "249"}, "SystemdVersion"),
                         ({"flood": True}, "InvalidTransport")):
        _, transport, launcher, budget = setup()
        for name, value in change.items():
            setattr(transport, name, value)
        result = launcher.launch(request(), budget)
        assert type(result) is DeadlineFailure and result.code == code
        assert not any(argv[0] == "systemd-run" for argv, _ in transport.calls)


@pytest.mark.parametrize("override,code", [
    ({"KillMode": "process"}, "UnverifiedBoundary"),
    ({"Description": "someone else's service"}, "UnverifiedBoundary"),
    ({"ControlGroup": "/other/tree"}, "UnverifiedBoundary"),
    ({"RuntimeMaxUSec": "infinity"}, "InvalidDuration"),
    ({"TimeoutStopUSec": "1h"}, "UnverifiedDeadline"),
    ({"SendSIGKILL": "no"}, "UnverifiedBoundary"),
])
def test_ignored_properties_or_unbounded_deadline_cannot_grant_verified_receipt(override, code):
    _, transport, launcher, budget = setup()
    transport.override = override
    result = launcher.launch(request(), budget)
    assert type(result) is DeadlineFailure and result.code == code and result.may_be_running
    assert sum(argv[0] == "systemd-run" for argv, _ in transport.calls) == 1
    assert result.stop_requested == (override.keys() not in ({"Description"}, {"ControlGroup"}))


def test_ambiguous_submit_retains_os_deadline_plan_and_never_retries_or_kills_pgid():
    _, transport, launcher, budget = setup()
    transport.launch_failure = True
    result = launcher.launch(request(), budget)
    assert type(result) is DeadlineFailure and result.code == "TransportTimeout" and result.may_be_running
    assert not result.stop_requested
    assert result.plan is not None and result.plan.stop_microseconds == 10_000_000
    assert sum(argv[0] == "systemd-run" for argv, _ in transport.calls) == 1
    assert not any("kill" in argv or "stop" in argv for argv, _ in transport.calls)


def test_exhausted_window_cannot_restart_clock_or_leave_out_cleanup():
    with pytest.raises(DeadlineError, match="reserves"):
        compile_launch(request(), DeadlineBudget(100., 60., 10., 5.), now=150.)
    clock, transport, launcher, budget = setup()
    transport.advance = 35.
    result = launcher.launch(request(), budget)
    assert type(result) is DeadlineFailure and result.code == "TransportTimeout"
    assert not any(argv[0] == "systemd-run" for argv, _ in transport.calls)


def test_systemctl_duration_parser_rejects_infinity_and_unknown_units():
    assert parse_systemd_duration("1min 2s 3ms 4us") == 62_003_004
    assert parse_systemd_duration("1.25s") == 1_250_000
    for malformed in ("infinity", "∞", "-1s", "1", "1fortnight", "1e10s", "NaNs", "1s trailing"):
        with pytest.raises(DeadlineError):
            parse_systemd_duration(malformed)
    with pytest.raises(ValueError):
        DeadlineBudget(100., 60., cleanup_seconds=1e-9)
