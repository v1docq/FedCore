"""Run a NEW v2 series with an independent process-family resource guard.

Added after the measured v2 series. Numerical runtime and old evidence are
unchanged. The raw guard log lives beside the new series directory because
the frozen runner requires the series itself not to exist before preparation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

REPO = Path(__file__).resolve().parents[1]
WORKER_SCRIPT = REPO/"scripts/run_sparse_pretrained_v2.py"
RSS_LIMIT = 15*1024**3


class OwnershipError(RuntimeError):
    pass


def worker_stage(command, script=WORKER_SCRIPT):
    """Only the exact owned Python entry point may receive a candidate limit."""
    if len(command) < 4:
        return None
    try:
        if Path(command[1]).resolve() != script.resolve():
            return None
        index = command.index("--stage")
        value = command[index+1]
    except (ValueError, IndexError, OSError):
        return None
    return value if value in ("worker", "time") else None


def snapshot_family(root, root_identity, now, *, script=WORKER_SCRIPT):
    """Skip an individually vanished child without ending the whole monitor.

    Return identity-bearing handles for only the root and its current children;
    caller must recheck each identity immediately before any termination.
    """
    if (root.pid, root.create_time()) != root_identity:
        raise OwnershipError("Owned root PID was reused; refusing unrelated processes")
    try:
        family = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        # Enumeration can race with a transient descendant. The next sample
        # enumerates again; it does not fall into an unguarded blocking wait.
        if not root.is_running():
            return 0, [], [], True
        family = [root]
        incomplete = True
    else:
        incomplete = False
    rss, members, handles = 0, [], []
    for process in family:
        try:
            identity = (process.pid, process.create_time())
            memory = process.memory_info().rss
            stage = worker_stage(process.cmdline(), script)
            rss += memory
            members.append({"pid": process.pid, "created_unix": identity[1], "rss_bytes": memory,
                            "stage": stage, "age_seconds": max(0., now-identity[1])})
            handles.append((process, identity))
        except psutil.NoSuchProcess:
            continue
    return rss, members, handles, incomplete


def kill_same_process(process, identity):
    """A PID alone is insufficient authority to terminate a process."""
    try:
        if (process.pid, process.create_time()) != identity:
            return False
        process.kill()
        return True
    except psutil.NoSuchProcess:
        return False


def stop_owned_tree(process, identity):
    try:
        if (process.pid, process.create_time()) != identity:
            return
        children = process.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for child in reversed(children):
        try:
            child_identity = (child.pid, child.create_time())
        except psutil.NoSuchProcess:
            continue
        kill_same_process(child, child_identity)
    kill_same_process(process, identity)


def run_guarded(source, output):
    if output.name != "local-bert-mlm-v2" or output.exists():
        raise ValueError("Require a fresh parent/local-bert-mlm-v2 destination")
    output.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = output.parent/f"resource-guard-{stamp}-{time.time_ns()}.jsonl"
    command = [sys.executable, str(WORKER_SCRIPT), "--stage", "run", "--source", str(source), "--output", str(output)]
    started = time.time()
    process = subprocess.Popen(command, cwd=REPO)
    root = psutil.Process(process.pid)
    identity = (root.pid, root.create_time())
    maximum, reason = 0, None
    with log_path.open("x", encoding="utf-8", buffering=1) as log:
        def emit(value):
            log.write(json.dumps({"unix": time.time(), **value}, allow_nan=False)+"\n")
        emit({"event": "started", "root_pid": root.pid, "root_created_unix": identity[1],
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "command": command, "output": str(output), "rss_limit_bytes": RSS_LIMIT,
              "total_seconds_limit": 7200, "candidate_seconds_limit": 600, "sampling_seconds": .5})
        try:
            while process.poll() is None:
                now = time.time()
                budget = output/"series_budget.json"
                deadline = started+7200
                if budget.exists():
                    # Keep the numerical runner's original preparation clock.
                    deadline = json.loads(budget.read_text())["started_unix"]+7200
                try:
                    rss, members, handles, incomplete = snapshot_family(root, identity, now)
                except psutil.NoSuchProcess:
                    if process.poll() is not None:
                        break
                    continue
                maximum = max(maximum, rss)
                emit({"event": "sample", "rss_bytes": rss, "members": members,
                      "enumeration_incomplete": incomplete, "deadline_unix": deadline})
                reason = "family_rss_limit" if rss > RSS_LIMIT else ("two_hour_limit" if now > deadline else None)
                if reason:
                    emit({"event": "stop", "reason": reason})
                    stop_owned_tree(root, identity)
                    break
                expired = {item["pid"] for item in members if item["stage"] in ("worker", "time") and item["age_seconds"] > 600}
                for member, member_identity in handles:
                    if member.pid in expired:
                        emit({"event": "candidate_stop", "pid": member.pid, "created_unix": member_identity[1],
                              "reason": "ten_minute_limit"})
                        stop_owned_tree(member, member_identity)
                time.sleep(.5)
        except BaseException as error:
            emit({"event": "guard_error", "error_type": type(error).__name__, "reason": str(error)})
            stop_owned_tree(root, identity)
            raise
        code = process.wait()
        emit({"event": "finished", "exit_code": code, "stop_reason": reason,
              "observed_family_rss_max_bytes": maximum, "rss_kind": "maximum sampled family RSS; not allocator/model peak"})
    print(f"Resource log: {log_path}", flush=True)
    return 124 if reason else code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    return run_guarded(args.source, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
