"""Durable per-job state and isolated processes with explicit cancellation."""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path

from .contracts import CompressionRequest, ContractError, plan_request
from .security import confined_path


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


def transition(state: JobState, event: JobState) -> JobState:
    allowed = {JobState.QUEUED: {JobState.RUNNING, JobState.CANCELLED, JobState.FAILED},
               JobState.RUNNING: {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}}
    if event not in allowed.get(state, set()):
        raise ContractError("invalid_transition", f"Cannot transition {state.value} to {event.value}")
    return event


class JobStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, state TEXT NOT NULL, request TEXT NOT NULL, result TEXT, created REAL NOT NULL)")

    def connection(self):
        return sqlite3.connect(self.root / "jobs.sqlite3", timeout=30)

    def directory(self, job_id):
        if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise ContractError("invalid_job_id", "Job id must be a generated UUID hex string")
        path = (self.root / job_id).resolve()
        if not path.is_relative_to(self.root):
            raise ContractError("invalid_path", "Job directory escaped storage")
        return path

    def create(self, request):
        plan_request(request)
        encoded_request = json.dumps(request.to_dict(), allow_nan=False)
        if len(encoded_request.encode("utf-8")) > 65536:
            raise ContractError("size_limit", "Request JSON exceeds 64 KiB")
        job_id = uuid.uuid4().hex
        self.directory(job_id).mkdir()
        with self.connection() as connection:
            connection.execute("INSERT INTO jobs VALUES (?, ?, ?, NULL, ?)",
                               (job_id, JobState.QUEUED.value, json.dumps(request.to_dict()), time.time()))
        return job_id

    def get(self, job_id):
        self.directory(job_id)
        with self.connection() as connection:
            row = connection.execute("SELECT state, request, result, created FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise ContractError("unknown_job", "Job does not exist")
        return {"id": job_id, "state": row[0], "request": json.loads(row[1]),
                "result": json.loads(row[2]) if row[2] else None, "created": row[3]}

    def advance(self, job_id, event, result=None):
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT state FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise ContractError("unknown_job", "Job does not exist")
            new = transition(JobState(row[0]), event)
            connection.execute("UPDATE jobs SET state=?, result=? WHERE id=?", (new.value, json.dumps(result) if result else None, job_id))

    def recover_interrupted(self):
        """Call once at service startup, before accepting requests."""
        with self.connection() as connection:
            rows = connection.execute("SELECT id FROM jobs WHERE state IN ('running','queued')").fetchall()
            result = json.dumps({"version": 1, "status": "failed", "error": {"code": "interrupted", "message": "Service restarted before completion"}})
            connection.execute("UPDATE jobs SET state='failed', result=? WHERE state IN ('running','queued')", (result,))
        for (job_id,) in rows:
            self._remove_outputs(job_id)

    def _remove_outputs(self, job_id):
        for path in self.directory(job_id).glob("compressed.*"):
            path.unlink(missing_ok=True)

    def delete(self, job_id):
        state = JobState(self.get(job_id)["state"])
        if state in (JobState.QUEUED, JobState.RUNNING):
            raise ContractError("active_job", "Cancel an active job before removing artifacts")
        path = self.directory(job_id)
        if path.exists():
            shutil.rmtree(path)
        with self.connection() as connection:
            connection.execute("DELETE FROM jobs WHERE id=?", (job_id,))


class JobRunner:
    def __init__(self, store: JobStore, *, python_executable=None, workers=2):
        self.store = store
        self.python_executable = str(python_executable or sys.executable)
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fedcore-job")
        self.processes = {}
        self.owned_jobs = set()
        self.lock = threading.RLock()

    def submit(self, request: CompressionRequest, input_dir):
        # Validate and bound all bytes before creating any visible job.
        plan_request(request)
        encoded_request = json.dumps(request.to_dict(), allow_nan=False)
        if len(encoded_request.encode("utf-8")) > 65536:
            raise ContractError("size_limit", "Request JSON exceeds 64 KiB")
        names = {request.model, request.example, request.data.validation}
        names.update(p for p in (request.data.train, request.data.calibration) if p is not None)
        paths = {name: confined_path(input_dir, name) for name in names}
        if sum(path.stat().st_size for path in paths.values()) > request.resources.max_bytes:
            raise ContractError("size_limit", "Total request input bytes exceed the job limit")
        job_id = self.store.create(request)
        root = self.store.directory(job_id)
        try:
            for name, source in paths.items():
                target = confined_path(root, name, must_exist=False)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            (root / "request.json").write_text(encoded_request, encoding="utf-8")
        except Exception as error:
            self.store.advance(job_id, JobState.FAILED, {"status": "failed", "error": {"code": "input_copy_failed", "message": str(error)}})
            raise
        with self.lock:
            self.owned_jobs.add(job_id)
            self.pool.submit(self._run, job_id, request)
        return job_id

    def _run(self, job_id, request):
        root = self.store.directory(job_id)
        with self.lock:
            try:
                current = self.store.get(job_id)["state"]
            except ContractError as error:
                if error.code == "unknown_job":
                    return  # A cancelled and stopped job may already be deleted.
                raise
            if current == "cancelled":
                return
            self.store.advance(job_id, JobState.RUNNING)
            env = os.environ.copy()
            # Source checkout support; only the child environment is changed.
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
            env["OMP_NUM_THREADS"] = str(request.resources.threads)
            with open(root / "worker.log", "wb") as output:
                try:
                    process = subprocess.Popen([self.python_executable, "-m", "fedcore.external_runtime.worker", "--job-dir", str(root)],
                                               cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT)
                except Exception as error:
                    self.store.advance(job_id, JobState.FAILED, {"status": "failed", "error": {"code": "worker_start_failed", "message": str(error)}})
                    return
            self.processes[job_id] = process
        try:
            process.wait(timeout=request.resources.timeout_seconds)
            result_path = root / "result.json"
            if result_path.is_file() and result_path.stat().st_size <= 1048576:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                result = {"version": 1, "status": "failed", "error": {"code": "worker_failed", "message": "Worker produced no bounded result"}}
            if process.returncode != 0 and result.get("status") == "succeeded":
                result = {"status": "failed", "error": {"code": "worker_failed", "message": "Worker exited unsuccessfully"}}
            if result.get("status") == "succeeded":
                artifact = confined_path(root, result.get("artifact", ""))
                if artifact.stat().st_size > request.resources.max_bytes:
                    raise ContractError("size_limit", "Output artifact exceeds job byte limit")
            final = JobState.SUCCEEDED if result.get("status") == "succeeded" else JobState.FAILED
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            final, result = JobState.FAILED, {"status": "failed", "error": {"code": "timeout", "message": "Worker exceeded time limit"}}
        except Exception as error:
            final, result = JobState.FAILED, {"status": "failed", "error": {"code": getattr(error, "code", "invalid_result"), "message": str(error)}}
        with self.lock:
            self.processes.pop(job_id, None)
            try:
                state = self.store.get(job_id)["state"]
            except ContractError as error:
                if error.code == "unknown_job":
                    return
                raise
            if state == "cancelled":
                self.store._remove_outputs(job_id)
                return
            if final == JobState.FAILED:
                self.store._remove_outputs(job_id)
            self.store.advance(job_id, final, result)

    def cancel(self, job_id):
        with self.lock:
            state = JobState(self.store.get(job_id)["state"])
            if state not in (JobState.QUEUED, JobState.RUNNING):
                raise ContractError("terminal_job", "Completed jobs cannot be cancelled")
            self.store.advance(job_id, JobState.CANCELLED, {"status": "cancelled", "version": 1})
            process = self.processes.get(job_id)
            if process is not None:
                process.kill()
                process.wait()
            self.store._remove_outputs(job_id)

    def wait(self, job_id, timeout=120):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.store.get(job_id)
            if job["state"] not in ("queued", "running"):
                return job
            time.sleep(0.02)
        raise TimeoutError("Caller wait expired; the job continues independently")

    def close(self):
        with self.lock:
            for job_id in tuple(self.owned_jobs):
                try:
                    if self.store.get(job_id)["state"] in ("queued", "running"):
                        self.cancel(job_id)
                except ContractError as error:
                    if error.code != "unknown_job":
                        raise
        self.pool.shutdown(wait=True, cancel_futures=True)
