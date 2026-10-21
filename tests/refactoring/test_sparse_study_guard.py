"""Independent process identity/race cases; no actual processes are killed."""
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from scripts.run_bounded_sparse_study import (
    WORKER_SCRIPT, OwnershipError, snapshot_family, kill_same_process, worker_stage,
)


class Process:
    def __init__(self, pid, created=100., rss=10, children=(), stage="worker", vanished=False):
        self.pid, self.created, self.rss = pid, created, rss
        self.descendants, self.stage, self.vanished = children, stage, vanished
        self.killed = False

    def create_time(self):
        if self.vanished:
            raise psutil.NoSuchProcess(self.pid)
        return self.created

    def memory_info(self):
        return SimpleNamespace(rss=self.rss)

    def cmdline(self):
        return ["python", str(WORKER_SCRIPT), "--stage", self.stage]

    def children(self, recursive=False):
        return self.descendants

    def kill(self):
        self.killed = True

    def is_running(self):
        return not self.vanished


def test_a_vanished_child_does_not_drop_later_samples_or_other_children():
    gone, live = Process(2, vanished=True), Process(3, rss=90)
    root = Process(1, rss=20, children=(gone, live), stage="run")
    rss, members, handles, incomplete = snapshot_family(root, (1, 100.), 200.)
    assert rss == 110 and [item["pid"] for item in members] == [1, 3]
    assert [process.pid for process, identity in handles] == [1, 3]
    assert not incomplete
    live.rss = 190
    assert snapshot_family(root, (1, 100.), 201.)[0] == 210


def test_pid_reuse_cannot_give_kill_authority_or_root_ownership():
    reused = Process(42, created=101.)
    assert not kill_same_process(reused, (42, 100.)) and not reused.killed
    with pytest.raises(OwnershipError):
        snapshot_family(reused, (42, 100.), 200.)
    assert kill_same_process(reused, (42, 101.)) and reused.killed


def test_child_enumeration_race_recovers_on_next_iteration():
    root = Process(1, children=(Process(3, rss=90),), stage="run")
    original = root.children
    def disappeared(recursive=False):
        raise psutil.NoSuchProcess(2)
    root.children = disappeared
    rss, members, _, incomplete = snapshot_family(root, (1, 100.), 200.)
    assert rss == 10 and [item["pid"] for item in members] == [1] and incomplete
    root.children = original
    assert snapshot_family(root, (1, 100.), 201.)[0] == 100


@pytest.mark.parametrize("stage", ["run", "reload", "other"])
def test_only_exact_owned_worker_or_timing_entry_point_has_candidate_deadline(stage):
    assert worker_stage(["python", str(WORKER_SCRIPT), "--stage", stage]) is None
    assert worker_stage(["python", "unrelated.py", "--stage", "worker"]) is None
    assert worker_stage(["python", "-c", str(WORKER_SCRIPT), "--stage", "worker"]) is None
    assert worker_stage(["python", str(WORKER_SCRIPT), "--stage", "worker"]) == "worker"
    assert worker_stage(["python", str(WORKER_SCRIPT), "--stage", "time"]) == "time"
