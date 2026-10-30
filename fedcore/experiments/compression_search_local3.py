"""One serial CPU packet with frozen provenance, clocks and nonrefundable charges.

The parent imports no numerical runtime. Workers are admitted exactly once;
there is deliberately no resume, hidden pilot or scientific retry command.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

from .compression_search_local2 import (
    _telemetry, append_event, count_lines, read_json, source_hashes, write_json)
from .compression_search_local3_protocol import (
    MATRIX_LIMIT, MLP_LIMIT, PLANNED_MATRIX, PLANNED_MLP, RESOURCE_MAX_GAP,
    SPLITS, STAGE_SECONDS, metadata_tapes, resource_coverage, schedule)

REPO = Path(__file__).resolve().parents[2]
MODULE = "fedcore.experiments.compression_search_local3"
PLAN = REPO/"docs/svd_compression/PLAN_ISSUE_99_LOCAL_3.md"
PUBLISHED_INVENTORY = REPO/"docs/svd_compression/results/issue-99-local-2-2026-10-21/artifact-manifest.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_witnesses(source_root, matrix_root):
    """Read only permitted historical files, never train/test tensor files."""
    source_root, matrix_root = Path(source_root), Path(matrix_root)
    files = {}
    for seed in SPLITS:
        base = source_root/str(seed)
        roles = read_json(base/"roles.json")
        selected = [base/"manifest.json", base/"roles.json", base/"baseline.pt"]
        selected += [base/roles["roles"][role]["file"] for role in ("calibration", "validation")]
        for path in selected:
            if path.resolve().parent != base.resolve():
                raise RuntimeError("Historical source escaped its declared role directory")
            files[path.relative_to(source_root).as_posix()] = sha(path)
    matrix = {"manifest.json": sha(matrix_root/"manifest.json")}
    records = read_json(matrix_root/"manifest.json")
    for n in (32, 512, 1024):
        record = next(row for row in records if row["n"] == n)
        path = matrix_root/record["source_file"]
        if path.resolve().parent != matrix_root.resolve() or sha(path) != record["source_sha256"]:
            raise RuntimeError("Historical matrix manifest/source changed")
        matrix[path.name] = sha(path)
    published = {row["path"]: row["sha256"] for row in read_json(PUBLISHED_INVENTORY)}
    for prefix, current in (("sources/", files), ("matrix-sources/", matrix)):
        for relative, digest in current.items():
            if published.get(prefix+relative) != digest:
                raise RuntimeError("Historical source differs from published local-2 evidence: "+prefix+relative)
    return files, matrix


def verify_frozen(protocol):
    if source_hashes() != protocol["source_files"] or sha(PLAN) != protocol["approved_plan_sha256"]:
        raise RuntimeError("Frozen executed source or approved plan changed")
    if sha(PUBLISHED_INVENTORY) != protocol["published_inventory_sha256"]:
        raise RuntimeError("Published historical source inventory changed")
    for root_key, files_key in (("source_root", "expected_source_files"),
                               ("matrix_source_root", "expected_matrix_files")):
        for relative, digest in protocol[files_key].items():
            if sha(Path(protocol[root_key])/relative) != digest:
                raise RuntimeError("Frozen historical source changed: "+relative)


def _kill_owned(process, identities):
    import psutil
    errors = []
    try:
        root = psutil.Process(process.pid)
        for item in [root, *root.children(recursive=True)]:
            identities[item.pid] = item.create_time()
    except psutil.NoSuchProcess:
        pass
    except psutil.AccessDenied as error:
        errors.append(str(error))
    for pid, created in reversed(tuple(identities.items())):
        try:
            item = psutil.Process(pid)
            if item.create_time() == created:
                item.kill()
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied as error:
            errors.append(str(error))
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        errors.append("Root worker did not terminate in 10 seconds")
    return errors


def _survivors(identities):
    import psutil
    live, errors = [], []
    for pid, created in identities.items():
        try:
            item = psutil.Process(pid)
            if item.create_time() == created and item.is_running() and item.status() != psutil.STATUS_ZOMBIE:
                live.append(pid)
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied as error:
            errors.append(str(error))
    return live, errors


@contextmanager
def _worker_lifetime(directory, identities):
    """Every exceptional watchdog exit closes its live worker before escape."""
    state = {"process": None}
    failed = False
    try:
        yield state
    except BaseException:
        failed = True
        raise
    finally:
        process = state["process"]
        if process is not None and (failed or process.poll() is None):
            errors = _kill_owned(process, identities)
            live, inspection_errors = _survivors(identities)
            write_json(Path(directory)/"emergency-cleanup.json", dict(
                failed_watchdog=failed, root_pid=process.pid, live_observed_pids=live,
                errors=errors+inspection_errors, closed=not live and not errors and not inspection_errors,
                completed_at=time.perf_counter()))


def run_job(output, job, protocol, package_start, stage_deadline):
    import psutil
    verify_frozen(protocol)
    now = time.perf_counter()
    if now >= min(stage_deadline, package_start+6900):
        return dict(job=job["job"], kind=job["kind"], status="not_started", stop="admission_deadline")
    matrix = job["kind"].startswith("matrix")
    family, limit = ("matrix", MATRIX_LIMIT) if matrix else ("mlp", MLP_LIMIT)
    if count_lines(Path(output)/(family+"-attempts.jsonl")) >= limit:
        return dict(job=job["job"], kind=job["kind"], status="not_started", stop="attempt_limit")
    directory = Path(output)/"jobs"/job["job"]
    directory.mkdir(exist_ok=False)
    started = time.perf_counter()
    hard = min(started+job["hard_seconds"], stage_deadline, package_start+7200)
    admission = min(started+job.get("budget_seconds", job["hard_seconds"]), hard, package_start+6900)
    if job["kind"] in ("matrix", "statistics_streams"):
        admission = min(admission, hard-20)
    job = dict(job, output=str(Path(output).resolve()), directory=str(directory.resolve()), repo=str(REPO),
        source_root=protocol["source_root"], matrix_source_root=protocol["matrix_source_root"],
        expected_source_files=protocol["expected_source_files"],
        expected_tapes=protocol["metadata_tapes"],
        worker_started_at=started, admission_deadline=admission, hard_deadline=hard,
        package_deadline=package_start+7200, compute_admission_deadline=min(stage_deadline, package_start+6900),
        protocol_sha256=protocol["protocol_sha256"], source_sha256=protocol["source_sha256"])
    if job["kind"] == "matrix":
        job["request_admission_deadline"] = admission
    write_json(directory/"job.json", job)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2", HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", PYTHONUTF8="1", PYTHONHASHSEED="0",
        TEMP=str(Path(output).resolve()/"tmp"), TMP=str(Path(output).resolve()/"tmp"))
    samples, identities, stop, monitor_error, kill_errors = [], {}, None, None, []
    with (directory/"stdout.log").open("w", encoding="utf-8") as stdout, (directory/"stderr.log").open("w", encoding="utf-8") as stderr, _worker_lifetime(directory, identities) as lifetime:
        process = subprocess.Popen([sys.executable, "-m", MODULE, "--worker", str(directory/"job.json")],
            cwd=REPO, env=env, stdout=stdout, stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        lifetime["process"] = process
        root = psutil.Process(process.pid)
        identities[root.pid] = root.create_time()
        while process.poll() is None:
            now, rss, sample_error = time.perf_counter(), None, None
            try:
                owned = [root, *root.children(recursive=True)]
                memory = []
                for item in owned:
                    try:
                        identities[item.pid] = item.create_time()
                        memory.append(item.memory_info().rss)
                    except psutil.NoSuchProcess:
                        if item.pid == root.pid and process.poll() is None:
                            raise
                rss = sum(memory) if memory else None
            except psutil.NoSuchProcess as error:
                if process.poll() is not None:
                    break
                sample_error = "Unobserved live worker: "+str(error)
            except psutil.AccessDenied as error:
                sample_error = str(error)
            available = psutil.virtual_memory().available
            sample = dict(timestamp=now, rss_bytes=rss, available_ram_bytes=available,
                          observed_pids=sorted(identities), monitor_error=sample_error)
            samples.append(sample)
            append_event(directory/"resources.jsonl", sample)
            if sample_error or rss is None:
                stop, monitor_error = "resource_monitor_failed", sample_error or "Unknown RSS"
            elif rss > 8*1024**3:
                stop = "rss_limit"
            elif available < 4*1024**3:
                stop = "ram_reserve"
            elif now >= hard:
                stop = "hard_deadline"
            active = directory/"active-request.json"
            try:
                state = read_json(active) if active.exists() else None
                if state and state.get("active") and time.perf_counter() >= state["hard_deadline"]:
                    stop = "request_hard_deadline"
            except (OSError, json.JSONDecodeError, KeyError) as error:
                stop, monitor_error = "status_monitor_io_failed", f"{type(error).__name__}: {error}"
            if stop:
                kill_errors = _kill_owned(process, identities)
                break
            time.sleep(.1)
        process.wait(timeout=10)
    completed = time.perf_counter()
    live, errors = _survivors(identities)
    if live:
        kill_errors += _kill_owned(process, identities)
        live, errors = _survivors(identities)
        stop = "descendant_survived_worker"
    kill_errors += errors
    coverage = resource_coverage(samples, started, completed)
    if not coverage["complete"] and monitor_error is None:
        stop, monitor_error = "resource_coverage_failed", "Resource sampling did not cover the worker interval"
    result_path = directory/"result.json"
    try:
        result = read_json(result_path) if result_path.exists() else None
    except (OSError, json.JSONDecodeError) as error:
        result, monitor_error = None, "Result publication/read failed: "+str(error)
    if monitor_error or kill_errors or live or stop == "descendant_survived_worker":
        status = "failed"
    elif process.returncode == 0 and result and result.get("status") == "succeeded":
        status = "partial" if result.get("partial") else "succeeded"
    elif stop in ("hard_deadline", "request_hard_deadline") or (result and result.get("status") == "partial"):
        status = "partial"
    else:
        status = "failed"
    record = dict(job=job["job"], kind=job["kind"], session_id=job.get("session_id"), status=status,
        returncode=process.returncode, stop=stop, result_path=str(result_path),
        svd_telemetry=_telemetry(directory), coverage=coverage,
        descendants_closed=not live and not kill_errors, descendant_errors=kill_errors,
        resources=dict(started_at=started, completed_at=completed, elapsed_seconds=completed-started,
            peak_tree_rss_bytes=coverage["peak_rss_bytes"], samples=len(samples),
            minimum_available_ram_bytes=min((row["available_ram_bytes"] for row in samples), default=None)),
        error=monitor_error or (None if status == "succeeded" else (result or {}).get("error", "Worker incomplete")))
    write_json(directory/"resources.json", record)
    return record


def run_package(output, source_root, matrix_source_root):
    import psutil
    from .compression_search_local3_matrix import session_schedule
    started = time.perf_counter()
    output, source_root, matrix_source_root = (Path(path).resolve() for path in (output, source_root, matrix_source_root))
    output.mkdir(exist_ok=False)
    (output/"jobs").mkdir()
    (output/"tmp").mkdir()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).strip():
        raise RuntimeError("Commit a clean, reviewed implementation before any experiment")
    if psutil.virtual_memory().available < 4*1024**3:
        raise RuntimeError("Insufficient available RAM")
    tapes = metadata_tapes()
    write_json(output/"metadata-tapes.json", tapes)
    packages = sorted((dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions())
    prior_environment = read_json(source_root.parent/"protocol.json")["environment"]
    if sys.version_info[:3] != (3, 10, 0) or importlib.metadata.version("torch") != "2.2.0+cpu":
        raise RuntimeError("The approved Python 3.10.0 / torch 2.2.0+cpu environment is required")
    if packages != sorted(tuple(row) for row in prior_environment["packages"]):
        raise RuntimeError("Installed dependency inventory differs from the sealed local-2 environment")
    expected, matrices = source_witnesses(source_root, matrix_source_root)
    files = source_hashes()
    protocol = dict(schema="issue99-local3-v1", package_started_at=started,
        utc=datetime.now(timezone.utc).isoformat(),
        git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        approved_plan_sha256=sha(PLAN), source_files=files,
        source_normalization="LF-normalized Python source bytes",
        source_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        source_root=str(source_root), matrix_source_root=str(matrix_source_root),
        expected_source_files=expected, expected_matrix_files=matrices,
        metadata_tapes=tapes,
        published_inventory_sha256=sha(PUBLISHED_INVENTORY),
        schedule=list(schedule()), stage_seconds=STAGE_SECONDS,
        matrix_schedule=[row for job in schedule() if job["kind"] == "matrix"
                         for row in session_schedule(job["n"], job["session_id"])],
        limits=dict(seconds=7200, no_new_after=6900, mlp_actions=MLP_LIMIT,
                    matrix_requests=MATRIX_LIMIT, planned_mlp=PLANNED_MLP, planned_matrix=PLANNED_MATRIX,
                    training_updates=0, test_evaluations=0, matrix_request_peak_bytes=1024**3,
                    tree_rss_bytes=8*1024**3, available_ram_bytes=4*1024**3,
                    resource_max_gap_seconds=RESOURCE_MAX_GAP, store_bytes=8*1024**2),
        environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
            processor=platform.processor(), logical_cpus=psutil.cpu_count(), physical_cpus=psutil.cpu_count(logical=False),
            packages=packages))
    protocol["protocol_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    write_json(output/"protocol.json", protocol)
    write_json(output/"schedule.json", protocol["schedule"])
    ledger = dict(status="running", jobs=[], stages=[], started_at=started, stop_reason=None)

    def save():
        ledger.update(elapsed_seconds=time.perf_counter()-started,
            mlp_attempts=count_lines(output/"mlp-attempts.jsonl"),
            matrix_attempts=count_lines(output/"matrix-attempts.jsonl"))
        write_json(output/"resource-ledger.json", ledger)

    try:
        for name in ("prepare", "matrix", "search", "statistics"):
            stage_start = started if name == "prepare" else time.perf_counter()
            deadline = min(stage_start+STAGE_SECONDS[name], started+7200)
            stage = dict(stage=name, started_at=stage_start, deadline=deadline, jobs=[])
            ledger["stages"].append(stage)
            for job in (row for row in protocol["schedule"] if row["stage"] == name):
                end = min(deadline-120, time.perf_counter()+job["hard_seconds"]) if job["kind"] == "matrix" else deadline
                row = run_job(output, job, protocol, started, end)
                ledger["jobs"].append(row)
                stage["jobs"].append(row["job"])
                save()
                print(json.dumps(dict(job=row["job"], status=row["status"], elapsed=ledger["elapsed_seconds"],
                    mlp=ledger["mlp_attempts"], matrix=ledger["matrix_attempts"])), flush=True)
                if row["status"] == "failed":
                    raise RuntimeError("Unknown/resource/numerical failure: "+row["job"]+": "+str(row["error"]))
                if name == "prepare" or job["kind"] == "matrix_admission":
                    if row["status"] != "succeeded":
                        raise TimeoutError("Correctness admission incomplete; dependent experiments not started")
            stage["completed_at"] = time.perf_counter()
            save()
        finalize_start = time.perf_counter()
        ledger["stages"].append(dict(stage="finalize", started_at=finalize_start,
            deadline=min(finalize_start+STAGE_SECONDS["finalize"], started+7200)))
        verify_frozen(protocol)
        save()
        if ledger["mlp_attempts"] > MLP_LIMIT or ledger["matrix_attempts"] > MATRIX_LIMIT:
            raise RuntimeError("Global attempt counter exceeded")
        ledger["computation_status"] = ("succeeded" if len(ledger["jobs"]) == 30
            and all(row["status"] == "succeeded" for row in ledger["jobs"])
            and ledger["mlp_attempts"] == PLANNED_MLP and ledger["matrix_attempts"] == PLANNED_MATRIX else "partial")
        ledger["status"] = "awaiting_file_audit"
        ledger["compute_completed_at"] = time.perf_counter()
    except BaseException as error:
        ledger.update(status="partial" if isinstance(error, TimeoutError) else "stopped",
            stop_reason=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    save()
    return ledger


def finalize_package(output, audit_path):
    """Publish the terminal status only after a file audit in the same clock."""
    output, audit_path = Path(output).resolve(), Path(audit_path).resolve()
    protocol, ledger = read_json(output/"protocol.json"), read_json(output/"resource-ledger.json")
    if ledger["status"] != "awaiting_file_audit":
        raise RuntimeError("Only an awaiting_file_audit packet can be finalized once")
    verify_frozen(protocol)
    audit = read_json(audit_path)
    current = dict(mlp=count_lines(output/"mlp-attempts.jsonl"),
                   matrix=count_lines(output/"matrix-attempts.jsonl"))
    if (audit.get("status") != "passed" or audit.get("protocol_sha256") != protocol["protocol_sha256"]
            or audit.get("counts", {}).get("mlp") != current["mlp"]
            or audit.get("counts", {}).get("matrix") != current["matrix"]):
        raise RuntimeError("Independent file audit failed or refers to different packet counters")
    now = time.perf_counter()
    final = ledger["stages"][-1]
    in_time = now <= final["deadline"] and now <= protocol["package_started_at"]+7200
    validation = output/"validation"
    validation.mkdir(exist_ok=True)
    saved = validation/"independent-audit.json"
    shutil.copyfile(audit_path, saved)
    final["completed_at"] = now
    ledger.update(status=ledger["computation_status"] if in_time else "partial",
        elapsed_seconds=now-protocol["package_started_at"], finalized_at=now,
        independent_audit_sha256=sha(saved), finalization_within_deadline=in_time,
        mlp_attempts=current["mlp"], matrix_attempts=current["matrix"])
    if not in_time:
        ledger["stop_reason"] = "File audit/finalization completed after its frozen deadline"
    write_json(output/"resource-ledger.json", ledger)
    return ledger


def worker(path):
    job = read_json(path)
    ordinary_stops = (TimeoutError,)
    try:
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        if job["kind"].startswith("matrix"):
            from .compression_search_matrix import MatrixDeadline
            ordinary_stops += (MatrixDeadline,)
            from .compression_search_local3_matrix import run_admission, run_session
            result = run_admission(job) if job["kind"] == "matrix_admission" else run_session(job)
        else:
            from .compression_search_digits_evaluation import DigitsDeadline
            ordinary_stops += (DigitsDeadline,)
            from .compression_search_local3_worker import run_job as digits_job
            result = digits_job(job)
        write_json(Path(job["directory"])/"result.json", dict(status="succeeded", job=job["job"], kind=job["kind"], **result))
        return 0
    except BaseException as error:
        status = "partial" if isinstance(error, ordinary_stops) else "failed"
        write_json(Path(job["directory"])/"result.json", dict(status=status,
            error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()))
        return 0 if status == "partial" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--matrix-source-root", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--finalize", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    if args.finalize:
        if not args.audit:
            parser.error("--finalize requires --audit")
        value = finalize_package(args.finalize, args.audit)
        print(json.dumps({key: value.get(key) for key in ("status", "elapsed_seconds", "mlp_attempts", "matrix_attempts")}), flush=True)
        return 0
    if not all((args.output, args.source_root, args.matrix_source_root)):
        parser.error("--output, --source-root and --matrix-source-root are required")
    value = run_package(args.output, args.source_root, args.matrix_source_root)
    print(json.dumps({key: value.get(key) for key in ("status", "elapsed_seconds", "mlp_attempts", "matrix_attempts", "stop_reason")}), flush=True)
    return 0 if value["status"] in ("succeeded", "partial", "awaiting_file_audit") else 1


if __name__ == "__main__":
    raise SystemExit(main())
