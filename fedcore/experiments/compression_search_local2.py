"""Serial parent watchdog for the approved issue-99 second local package.

The parent imports no ML runtime. No resume or automatic scientific retry is
provided. Interrupted actions and their original clocks remain in the ledger.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
MODULE = "fedcore.experiments.compression_search_local2"
SPLITS = (19901, 19911, 19921)
PLAN = REPO/"docs/svd_compression/PLAN_ISSUE_99_LOCAL_2.md"
STAGE_SECONDS = dict(prepare=300, train=900, admission=300, search=2400,
                     oracle=600, matrix=1800, finalize=300)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   indent=2, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_event(path, value):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False)+"\n")
        stream.flush()


def count_lines(path):
    return sum(bool(line.strip()) for line in Path(path).read_text(encoding="utf-8").splitlines()) if Path(path).exists() else 0


def source_hashes():
    return {path.relative_to(REPO).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for path in sorted((REPO/"fedcore").rglob("*.py"))}


def search_jobs():
    jobs = []
    for seed in SPLITS:
        for budget in ("attempts", "time"):
            for search_seed, order in ((11, "RTE"), (22, "TER"), (33, "ERT")):
                for mode in order:
                    jobs.append(dict(job=f"search-{seed}-{budget}-{search_seed}-{mode}",
                        kind="search", split_seed=seed, model_seed=seed, budget=budget,
                        search_seed=search_seed, mode=mode,
                        budget_seconds=45 if budget == "attempts" else 30,
                        hard_seconds=60 if budget == "attempts" else 45))
    return jobs


def _telemetry(directory):
    path = Path(directory)/"svd-telemetry.bin"
    if not path.exists():
        return None
    data = path.read_bytes()
    if len(data) != 64:
        return dict(error="invalid telemetry length")
    started, completed, start_time, finish_time = struct.unpack_from("<QQdd", data)
    return dict(started=started, completed=completed, last_started_at=start_time,
                last_completed_at=finish_time, format="<QQdd; 64-byte file; read after worker termination")


def run_job(output, job, protocol, package_start, stage_deadline):
    import psutil
    if source_hashes() != protocol["source_files"]:
        raise RuntimeError("Executed source changed after protocol freeze")
    now = time.perf_counter()
    if now >= min(stage_deadline, package_start+6900):
        return dict(job=job["job"], kind=job["kind"], status="not_started", stop="admission_deadline")
    family = "matrix" if job["kind"].startswith("matrix") else "mlp"
    limit = 4350 if family == "matrix" else 10300
    if count_lines(Path(output)/(family+"-attempts.jsonl")) >= limit:
        return dict(job=job["job"], kind=job["kind"], status="not_started", stop="attempt_limit")
    directory = Path(output)/"jobs"/job["job"]
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    hard_deadline = min(started+job["hard_seconds"], stage_deadline, package_start+7200)
    admission_deadline = min(started+job.get("budget_seconds", job["hard_seconds"]),
                             hard_deadline, package_start+6900)
    job = dict(job, output=str(Path(output).resolve()), directory=str(directory.resolve()), repo=str(REPO),
        worker_started_at=started, package_deadline=package_start+7200,
        compute_admission_deadline=min(stage_deadline, package_start+6900),
        admission_deadline=admission_deadline, hard_deadline=hard_deadline,
        protocol_sha256=protocol["protocol_sha256"], source_sha256=protocol["source_sha256"])
    if job["kind"] == "matrix":
        job["request_admission_deadline"] = min(admission_deadline-20, package_start+6900)
    write_json(directory/"job.json", job)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2", HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", PYTHONUTF8="1", PYTHONHASHSEED="0",
        TEMP=str(Path(output).resolve()/"tmp"), TMP=str(Path(output).resolve()/"tmp"))
    peak, minimum_ram, samples, stop = 0, None, 0, None
    with (directory/"stdout.log").open("w", encoding="utf-8") as stdout, (directory/"stderr.log").open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen([sys.executable, "-m", MODULE, "--worker", str(directory/"job.json")],
            cwd=REPO, env=env, stdout=stdout, stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        root = psutil.Process(process.pid)
        while process.poll() is None:
            owned = [root]
            try:
                owned += root.children(recursive=True)
                rss = sum(item.memory_info().rss for item in owned if item.is_running())
            except psutil.NoSuchProcess:
                rss = 0
            except psutil.AccessDenied:
                rss, stop = 0, "resource_monitor_access_denied"
            now = time.perf_counter()
            ram = psutil.virtual_memory().available
            peak, samples = max(peak, rss), samples+1
            minimum_ram = ram if minimum_ram is None else min(minimum_ram, ram)
            if rss > 8*1024**3:
                stop = "rss_limit"
            elif ram < 4*1024**3:
                stop = "ram_reserve"
            elif now >= hard_deadline:
                stop = "hard_deadline"
            active_path = directory/"active-request.json"
            if job["kind"].startswith("matrix") and active_path.exists():
                active = read_json(active_path)
                if active.get("active") and now >= active["hard_deadline"]:
                    stop = "request_hard_deadline"
            if stop:
                for child in reversed(owned):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                process.wait(timeout=10)
                break
            time.sleep(.1)
    result_path = directory/"result.json"
    result = read_json(result_path) if result_path.exists() else None
    if process.returncode == 0 and result and result.get("status") == "succeeded":
        status = "partial" if result.get("partial") else "succeeded"
    elif stop in ("hard_deadline", "request_hard_deadline") or (result and result.get("status") == "partial"):
        status = "partial"
    else:
        status = "failed"
    completed = time.perf_counter()
    record = dict(job=job["job"], kind=job["kind"], status=status, returncode=process.returncode,
        stop=stop, result_path=str(result_path), svd_telemetry=_telemetry(directory),
        resources=dict(started_at=started, completed_at=completed, elapsed_seconds=completed-started,
                       peak_tree_rss_bytes=peak, minimum_available_ram_bytes=minimum_ram, samples=samples),
        error=None if status == "succeeded" else (result or {}).get("error", "Worker stopped or incomplete"))
    write_json(directory/"resources.json", record)
    return record


def freeze_selections(output):
    """Pure file selection; never import tensors or access held-out labels."""
    selected, eligible_jobs = [], []
    for path in sorted((Path(output)/"jobs").glob("*/result.json")):
        job = read_json(path.parent/"job.json")
        if job["kind"] not in ("search", "oracle"):
            continue
        result = read_json(path)
        if result.get("status") != "succeeded":
            continue
        rows = result.get("evaluations", [])
        if job["kind"] == "oracle" and len(rows) != 440:
            continue
        eligible_jobs.append(job["job"])
        deadline = result.get("deadline", job["admission_deadline"])
        archive = [row for row in rows if row.get("status") == "finite" and row.get("quality_accepted")
                   and row.get("completed_at", float("inf")) <= deadline]
        for cap in (20671, 41342, 58568):
            available = [row for row in archive if row["tensor_bytes"] <= cap]
            chosen = min(available, key=lambda row: (row["quality"], row["tensor_bytes"], row["solution_id"])) if available else None
            selected.append(dict(model_seed=job["split_seed"], split_seed=job["split_seed"],
                job=job["job"], byte_cap=cap, solution_id=chosen["solution_id"] if chosen else None,
                validation_quality=chosen["quality"] if chosen else None,
                validation_cross_entropy=chosen["metrics"]["cross_entropy"] if chosen else None,
                tensor_bytes=chosen["tensor_bytes"] if chosen else None))
    value = dict(selected=selected, eligible_jobs=eligible_jobs, frozen_at=time.perf_counter(),
                 rule="validation CE; then actual tensor bytes; then canonical ID; before opening test")
    path = Path(output)/"selected.json"
    if path.exists():
        raise RuntimeError("Refuse to overwrite a frozen test selection")
    write_json(path, value)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_package(output):
    import psutil
    from .compression_search_matrix import SIZES, SIZE_SECONDS, stream_schedule
    started = time.perf_counter()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/"jobs").mkdir()
    (output/"tmp").mkdir()
    if psutil.virtual_memory().available < 4*1024**3:
        raise RuntimeError("Insufficient free RAM before package admission")
    git_revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    if dirty:
        raise RuntimeError("Freeze a committed, clean source before experiments")
    sources = source_hashes()
    source_digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    protocol = dict(schema="issue99-local2-v1", approved_plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
        git_revision=git_revision, source_files=sources, source_sha256=source_digest,
        utc=datetime.now(timezone.utc).isoformat(), package_started_at=started,
        limits=dict(seconds=7200, no_new_after=6900, mlp_actions=10300, matrix_requests=4350,
                    training_updates=2040, rss_bytes=8*1024**3, available_ram_bytes=4*1024**3,
                    matrix_request_peak_bytes=1024**3),
        environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
            processor=platform.processor(), logical_cpus=psutil.cpu_count(), physical_cpus=psutil.cpu_count(logical=False),
            packages=sorted((dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions())),
        stage_seconds=STAGE_SECONDS, search_jobs=search_jobs(),
        matrix_schedule=[row for n in SIZES for row in stream_schedule(n)])
    protocol["protocol_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    write_json(output/"protocol.json", protocol)
    ledger = dict(status="running", jobs=[], stages=[], started_at=started, stop_reason=None)

    def save():
        ledger.update(elapsed_seconds=time.perf_counter()-started,
            mlp_attempts=count_lines(output/"mlp-attempts.jsonl"),
            matrix_attempts=count_lines(output/"matrix-attempts.jsonl"))
        write_json(output/"resource-ledger.json", ledger)

    def stage(name, jobs):
        start = time.perf_counter()
        deadline = min(start+STAGE_SECONDS[name], started+7200)
        record = dict(stage=name, started_at=start, deadline=deadline, jobs=[])
        ledger["stages"].append(record)
        results = []
        for job in jobs:
            result = run_job(output, job, protocol, started, deadline)
            ledger["jobs"].append(result)
            record["jobs"].append(result["job"])
            results.append(result)
            save()
            print(json.dumps(dict(job=result["job"], status=result["status"], elapsed=ledger["elapsed_seconds"],
                                 mlp=ledger["mlp_attempts"], matrix=ledger["matrix_attempts"])), flush=True)
            if result["status"] == "failed":
                raise RuntimeError(f"Unknown/resource/numerical worker failure: {result}")
        record["completed_at"] = time.perf_counter()
        save()
        return results

    try:
        preparation = stage("prepare", [dict(job="prepare", kind="prepare", hard_seconds=300)])
        digits_ready = preparation[0]["status"] == "succeeded"
        if digits_ready:
            trained = stage("train", [dict(job=f"train-{seed}", kind="train", split_seed=seed,
                        model_seed=seed, hard_seconds=300) for seed in SPLITS])
            digits_ready = all(row["status"] == "succeeded" for row in trained)
            if digits_ready:
                accuracies = [read_json(row["result_path"])["baseline_validation"]["accuracy"] for row in trained]
                digits_ready = all(value >= .9 for value in accuracies)
                ledger["baseline_gate"] = dict(accuracies=accuracies, threshold=.9, passed=digits_ready)
        if digits_ready:
            admission = stage("admission", [dict(job=f"admission-{seed}", kind="admission",
                split_seed=seed, model_seed=seed, hard_seconds=100) for seed in SPLITS])
            digits_ready = all(row["status"] == "succeeded" for row in admission)
        if digits_ready:
            stage("search", search_jobs())
            stage("oracle", [dict(job=f"oracle-{seed}", kind="oracle", split_seed=seed,
                                 model_seed=seed, hard_seconds=200) for seed in SPLITS])
        else:
            ledger["digits_skipped"] = "Preparation/training/admission incomplete or baseline quality gate failed"
        # One frozen 30-minute envelope, with size windows that cannot borrow.
        matrix_start = time.perf_counter()
        matrix_deadline = min(matrix_start+1800, started+7200)
        matrix_stage = dict(stage="matrix", started_at=matrix_start, deadline=matrix_deadline, jobs=[])
        ledger["stages"].append(matrix_stage)
        gate = run_job(output, dict(job="matrix-admission", kind="matrix_admission", hard_seconds=180),
                       protocol, started, min(matrix_start+180, matrix_deadline))
        ledger["jobs"].append(gate)
        matrix_stage["jobs"].append(gate["job"])
        save()
        if gate["status"] == "failed":
            raise RuntimeError(f"Matrix correctness admission failed: {gate}")
        if gate["status"] == "succeeded":
            for n in SIZES:
                # Per-size maximum begins now; unused time never enlarges a later maximum.
                end = min(time.perf_counter()+SIZE_SECONDS[n], matrix_deadline-120)
                row = run_job(output, dict(job=f"matrix-{n}", kind="matrix", n=n,
                    hard_seconds=SIZE_SECONDS[n]), protocol, started, end)
                ledger["jobs"].append(row)
                matrix_stage["jobs"].append(row["job"])
                save()
                print(json.dumps(dict(job=row["job"], status=row["status"], elapsed=ledger["elapsed_seconds"],
                                     matrix=ledger["matrix_attempts"])), flush=True)
                if row["status"] == "failed":
                    raise RuntimeError(f"Matrix comparison failed: {row}")
        matrix_stage["completed_at"] = time.perf_counter()
        if digits_ready:
            selection_hash = freeze_selections(output)
            ledger["selections_sha256"] = selection_hash
            stage("finalize", [dict(job=f"finalize-{seed}", kind="finalize", split_seed=seed,
                model_seed=seed, selections_sha256=selection_hash, hard_seconds=100) for seed in SPLITS])
        partial = not digits_ready or any(row["status"] != "succeeded" for row in ledger["jobs"])
        ledger["status"] = "partial" if partial else "succeeded"
    except BaseException as error:
        ledger.update(status="stopped", stop_reason=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    save()
    return ledger


def worker(path):
    job = read_json(path)
    try:
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        if job["kind"].startswith("matrix"):
            from .compression_search_matrix import run_matrix_admission, run_matrix_job
            result = run_matrix_admission(job) if job["kind"] == "matrix_admission" else run_matrix_job(job)
        else:
            from .compression_search_digits_worker import run_job as digits_job
            result = digits_job(job)
        write_json(Path(job["directory"])/"result.json", dict(status="succeeded", job=job["job"], kind=job["kind"], **result))
        return 0
    except BaseException as error:
        from .compression_search_matrix import MatrixDeadline
        status = "partial" if isinstance(error, (TimeoutError, MatrixDeadline)) else "failed"
        write_json(Path(job["directory"])/"result.json", dict(status=status,
            error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()))
        return 0 if status == "partial" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    if not args.output:
        parser.error("--output is required")
    ledger = run_package(args.output)
    print(json.dumps({key: ledger.get(key) for key in ("status", "elapsed_seconds", "mlp_attempts", "matrix_attempts", "stop_reason")}), flush=True)
    return 0 if ledger["status"] in ("succeeded", "partial") else 1


if __name__ == "__main__":
    raise SystemExit(main())
