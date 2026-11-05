"""Small CPU process-ownership tests; no model/data replay."""
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from scripts import run_heterogeneous_block_oracle as supervisor


class MissingProcess(Exception):
    pass


class FakeProcesses:
    NoSuchProcess = MissingProcess
    def __init__(self):
        self.rows = {11: dict(birth=1., parent=1, rss=2_000_000, alive=True),
                     22: dict(birth=2., parent=11, rss=30_000_000, alive=True)}
        self.signals = []
    def Process(self, pid):
        owner = self
        def row():
            value = owner.rows.get(pid)
            if value is None or not value["alive"]:
                raise MissingProcess(pid)
            return value
        row()
        class Process:
            def __init__(self):
                self.pid = pid
            def create_time(self):
                return row()["birth"]
            def ppid(self):
                return row()["parent"]
            def memory_info(self):
                return SimpleNamespace(rss=row()["rss"])
            def children(self, recursive):
                assert recursive is False
                return [owner.Process(child) for child, value in owner.rows.items()
                        if value["alive"] and value["parent"] == pid]
            def kill(self):
                row()["alive"] = False
                owner.signals.append(pid)
        return Process()


class FakeChild:
    pid = 11
    def __init__(self, processes):
        self.processes = processes
    def poll(self):
        return None if self.processes.rows[self.pid]["alive"] else -9
    def wait(self, timeout):
        result = self.poll()
        if result is None:
            raise subprocess.TimeoutExpired("owned CPU launcher", timeout)
        return result


def test_redirector_sample_sums_launcher_and_real_worker_rss_and_keeps_captured_births():
    processes = FakeProcesses()
    tree = supervisor.OwnedCpuTree(FakeChild(processes), processes)
    rss, live = tree.capture()
    assert rss == 32_000_000 and set(live) == {11, 22}
    assert tree.owned == {11: (1., 0), 22: (2., 1)}
    processes.rows[11]["alive"] = False
    processes.rows[22]["parent"] = 1  # already captured child remains owned after reparenting
    assert tree.capture() == (30_000_000, (22,))


def test_normal_exit_between_sample_and_poll_uses_fresh_descendant_lifetimes():
    processes = FakeProcesses()
    class ExitingChild(FakeChild):
        def poll(self):
            for row in self.processes.rows.values(): row["alive"] = False
            return 0
    child = ExitingChild(processes)
    tree = supervisor.OwnedCpuTree(child, processes)
    result = supervisor.monitor_owned_child(child, tree, work_deadline=10., hard_deadline=14.,
        clock=lambda: 1., sleep=lambda seconds: None)
    assert result["reason"] == "worker_exited" and result["returncode"] == 0
    assert result["known_owned_tree_gone"] and processes.signals == []


@pytest.mark.parametrize("limit,deadline,reason", [(100_000_000, 1., "deadline"), (20_000_000, 10., "RSS_limit")])
def test_limit_or_timeout_cleans_captured_child_before_launcher(limit, deadline, reason):
    processes = FakeProcesses(); child = FakeChild(processes)
    tree = supervisor.OwnedCpuTree(child, processes)
    now = [1.]
    result = supervisor.monitor_owned_child(child, tree, work_deadline=deadline, hard_deadline=5.,
        rss_limit=limit, clock=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
    assert result["reason"] == reason and result["returncode"] == -9
    assert result["peak_RSS_bytes"] == 32_000_000
    assert processes.signals == [22, 11]
    assert result["known_owned_tree_gone"] and not result["remaining_owned_pids"]


def test_captured_pid_reuse_fails_closed_and_never_signals_the_replacement():
    processes = FakeProcesses(); child = FakeChild(processes)
    tree = supervisor.OwnedCpuTree(child, processes)
    tree.capture()
    processes.rows[22]["birth"] = 99.
    now = [1.]
    result = supervisor.monitor_owned_child(child, tree, work_deadline=10., hard_deadline=5.,
        clock=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
    assert result["reason"] == "supervision_failed" and "birth changed" in result["failure"]
    assert result["reused_pids_not_signalled"] == [22]
    assert processes.signals == [11] and processes.rows[22]["alive"]
    assert result["known_owned_tree_gone"]  # captured old lifetime gone, unrelated replacement untouched


@pytest.mark.parametrize("previous,expected", [(0., (180., 176.)), (3.14, (176.86, 172.86)), (174.9, (5.1, 1.1))])
def test_original_budget_subtracts_previous_cost_and_keeps_four_second_reserve(previous, expected):
    remaining, work = supervisor.remaining_budget(previous)
    assert (remaining, work) == pytest.approx(expected)
    assert remaining - work == pytest.approx(4.) and previous + remaining == pytest.approx(180.)


@pytest.mark.parametrize("previous", [-1., 175., float("nan"), float("inf"), True])
def test_invalid_previous_cost_cannot_expand_or_reset_budget(previous):
    with pytest.raises(ValueError):
        supervisor.remaining_budget(previous)


def test_native_short_cpu_redirector_child_is_measured_and_reaped():
    psutil = pytest.importorskip("psutil")
    # A synthetic redirector waits for its child; the child holds 12 MiB briefly.
    code = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-u','-c','import time; x=bytearray(12*1024*1024); x[::4096]=bytes([1])*(len(x)//4096); print(\"ready\",flush=True); time.sleep(.75)'],creationflags=0x08000000 if sys.platform=='win32' else 0); p.wait()"
    started = time.monotonic()
    child = subprocess.Popen([sys.executable, "-u", "-c", code], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    assert child.stdout.readline().strip() == "ready"
    tree = supervisor.OwnedCpuTree(child, psutil)
    result = supervisor.monitor_owned_child(child, tree, work_deadline=started + 4.,
        hard_deadline=started + 5., rss_limit=64 * 1024**2)
    assert result["returncode"] == 0 and result["reason"] == "worker_exited"
    assert len(result["owned_processes"]) >= 2
    assert 12 * 1024**2 <= result["peak_RSS_bytes"] <= 64 * 1024**2
    assert result["known_owned_tree_gone"]
