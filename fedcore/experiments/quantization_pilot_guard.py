"""Fail-closed process-family guard. Imports only stdlib and psutil.

Invoke this file directly, before importing torch or the experiment worker.
The authorization deadline is read from LIMITS.json and is never reset.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil


@dataclass(frozen=True)
class Limits:
    deadline_unix: float
    deadline_monotonic: float
    hard_rss: int
    soft_rss: int
    disk_bytes: int

    @classmethod
    def read(cls, path):
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if "limits" in raw:
            raw = raw["limits"]
        values = (raw["deadline_unix"], raw["deadline_monotonic"])
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError("Finite absolute deadlines required")
        caps = (raw["max_family_rss_bytes"], raw["safety_soft_rss_bytes"], raw["max_artifact_bytes"])
        if any(type(v) is not int or v <= 0 for v in caps) or caps[1] >= caps[0]:
            raise ValueError("Positive caps and a strict RSS safety margin required")
        if caps[0] > 8*1024**3 or caps[1] > 7*1024**3 or caps[2] > 5*1024**3:
            raise ValueError("Authorization caps may only be narrowed")
        if type(raw.get("max_concurrent_numerical_workers")) is not int or raw["max_concurrent_numerical_workers"] != 1 or raw.get("gpu_allowed") is not False or raw.get("cluster_allowed") is not False:
            raise ValueError("This guard accepts only the authorized single local CPU worker")
        return cls(*map(float, values), *caps)

    def remaining(self, epoch=None, monotonic=None):
        return min(self.deadline_unix - (time.time() if epoch is None else epoch),
                   self.deadline_monotonic - (time.monotonic() if monotonic is None else monotonic))


def directory_bytes(root):
    total = 0
    def unobservable(error):
        raise error
    for base, dirs, files in os.walk(root, followlinks=False, onerror=unobservable):
        for name in dirs:
            if (Path(base) / name).is_symlink():
                raise RuntimeError("Artifact directories cannot contain links")
        for name in files:
            path = Path(base) / name
            try:
                if path.is_symlink():
                    raise RuntimeError("Artifact files cannot be links")
                total += path.stat().st_size
            except FileNotFoundError:
                continue  # Atomic rename/removal between enumeration and stat.
    return total


def sampled_rss(identities, process_factory=psutil.Process, samples=None):
    """Vanished members are harmless; unobservable live membership is fatal."""
    total = 0
    live = {}
    for pid, created in identities.items():
        try:
            process = process_factory(pid)
            if process.create_time() != created:
                continue  # PID reuse is not ownership.
            rss = process.memory_info().rss
            total += rss
            live[pid] = created
            if samples is not None:
                samples.append({"pid": pid, "create_time": created, "rss_bytes": rss})
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return total, live


class WindowsJob:
    """Atomically create a suspended process in a kill-on-close Windows job.

    PROC_THREAD_ATTRIBUTE_JOB_LIST prevents the orphan window between process
    creation and AssignProcessToJobObject. Unsupported Windows versions refuse
    launch. A closed/crashed guardian closes its sole job handle in the kernel.
    """
    commit_limit_bytes = 6 * 1024**3

    def __init__(self, command, cwd, stdout, stderr):
        from ctypes import wintypes as w
        import msvcrt
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        self.k, self.handle, self.process, self.pid = k, None, None, None
        class STARTUPINFO(ctypes.Structure):
            _fields_ = [("cb", w.DWORD), ("lpReserved", w.LPWSTR), ("lpDesktop", w.LPWSTR), ("lpTitle", w.LPWSTR),
                        ("dwX", w.DWORD), ("dwY", w.DWORD), ("dwXSize", w.DWORD), ("dwYSize", w.DWORD),
                        ("dwXCountChars", w.DWORD), ("dwYCountChars", w.DWORD), ("dwFillAttribute", w.DWORD),
                        ("dwFlags", w.DWORD), ("wShowWindow", w.WORD), ("cbReserved2", w.WORD),
                        ("lpReserved2", ctypes.POINTER(w.BYTE)), ("hStdInput", w.HANDLE), ("hStdOutput", w.HANDLE), ("hStdError", w.HANDLE)]
        class STARTUPINFOEX(ctypes.Structure):
            _fields_ = [("StartupInfo", STARTUPINFO), ("lpAttributeList", ctypes.c_void_p)]
        class PROCESS_INFORMATION(ctypes.Structure):
            _fields_ = [("hProcess", w.HANDLE), ("hThread", w.HANDLE), ("dwProcessId", w.DWORD), ("dwThreadId", w.DWORD)]
        class BASIC_LIMIT(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", w.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", w.DWORD), ("Affinity", ctypes.c_size_t), ("PriorityClass", w.DWORD), ("SchedulingClass", w.DWORD)]
        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]
        class EXTENDED_LIMIT(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BASIC_LIMIT), ("IoInfo", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]
        k.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]; k.CreateJobObjectW.restype = w.HANDLE
        k.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        k.QueryInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.POINTER(w.DWORD)]
        k.InitializeProcThreadAttributeList.argtypes = [ctypes.c_void_p, w.DWORD, w.DWORD, ctypes.POINTER(ctypes.c_size_t)]
        k.UpdateProcThreadAttribute.argtypes = [ctypes.c_void_p, w.DWORD, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]
        k.DeleteProcThreadAttributeList.argtypes = [ctypes.c_void_p]
        k.CreateProcessW.argtypes = [w.LPCWSTR, w.LPWSTR, ctypes.c_void_p, ctypes.c_void_p, w.BOOL, w.DWORD, ctypes.c_void_p, w.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(PROCESS_INFORMATION)]
        k.ResumeThread.argtypes = [w.HANDLE]; k.ResumeThread.restype = w.DWORD
        k.CloseHandle.argtypes = [w.HANDLE]
        k.GetExitCodeProcess.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
        k.TerminateJobObject.argtypes = [w.HANDLE, w.UINT]
        self.handle = k.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        attributes = None; attributes_initialized = False
        info = PROCESS_INFORMATION()
        try:
            limit = EXTENDED_LIMIT(); limit.BasicLimitInformation.LimitFlags = 0x2000 | 0x200  # KILL_ON_JOB_CLOSE | JOB_MEMORY
            limit.JobMemoryLimit = self.commit_limit_bytes
            if not k.SetInformationJobObject(self.handle, 9, ctypes.byref(limit), ctypes.sizeof(limit)):
                raise ctypes.WinError(ctypes.get_last_error())
            size = ctypes.c_size_t()
            k.InitializeProcThreadAttributeList(None, 2, 0, ctypes.byref(size))
            attributes = ctypes.create_string_buffer(size.value)
            if not k.InitializeProcThreadAttributeList(attributes, 2, 0, ctypes.byref(size)):
                raise ctypes.WinError(ctypes.get_last_error())
            attributes_initialized = True
            jobs = (w.HANDLE * 1)(self.handle)
            if not k.UpdateProcThreadAttribute(attributes, 0, 0x0002000D, jobs, ctypes.sizeof(jobs), None, None):
                raise ctypes.WinError(ctypes.get_last_error())
            with open(os.devnull, "rb") as stdin:
                handles = (w.HANDLE * 3)(*(msvcrt.get_osfhandle(stream.fileno()) for stream in (stdin, stdout, stderr)))
                for stream in (stdin, stdout, stderr):
                    os.set_handle_inheritable(msvcrt.get_osfhandle(stream.fileno()), True)
                if not k.UpdateProcThreadAttribute(attributes, 0, 0x00020002, handles, ctypes.sizeof(handles), None, None):
                    raise ctypes.WinError(ctypes.get_last_error())
                startup = STARTUPINFOEX(); startup.StartupInfo.cb = ctypes.sizeof(startup)
                startup.StartupInfo.dwFlags = 0x100; startup.lpAttributeList = ctypes.addressof(attributes)
                startup.StartupInfo.hStdInput, startup.StartupInfo.hStdOutput, startup.StartupInfo.hStdError = handles
                flags = 0x00080000 | 0x00000004 | 0x08000000  # EXTENDED_STARTUPINFO | SUSPENDED | NO_WINDOW
                ok = k.CreateProcessW(None, ctypes.create_unicode_buffer(subprocess.list2cmdline(command)), None, None, True,
                                      flags, None, str(cwd), ctypes.byref(startup), ctypes.byref(info))
                for stream in (stdout, stderr):
                    os.set_handle_inheritable(msvcrt.get_osfhandle(stream.fileno()), False)
                if not ok:
                    raise ctypes.WinError(ctypes.get_last_error())
            self.process, self.pid = info.hProcess, info.dwProcessId
            if k.ResumeThread(info.hThread) == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            self.close()
            if info.hProcess and not self.process:
                k.CloseHandle(info.hProcess)
            raise
        finally:
            if info.hThread:
                k.CloseHandle(info.hThread)
            if attributes_initialized:
                k.DeleteProcThreadAttributeList(attributes)

    def pids(self):
        from ctypes import wintypes as w
        # A deliberately finite membership cap; overflow refuses the job.
        data = ctypes.create_string_buffer(8 + 4096 * ctypes.sizeof(ctypes.c_size_t))
        returned = w.DWORD()
        if not self.k.QueryInformationJobObject(self.handle, 3, data, len(data), ctypes.byref(returned)):
            raise ctypes.WinError(ctypes.get_last_error())
        count = ctypes.c_uint32.from_buffer(data, 4).value
        return list((ctypes.c_size_t * count).from_buffer(data, 8))

    def poll(self):
        from ctypes import wintypes as w
        code = w.DWORD()
        if not self.k.GetExitCodeProcess(self.process, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return None if code.value == 259 else code.value

    def close(self):
        if self.handle:
            self.k.TerminateJobObject(self.handle, 125)
            self.k.CloseHandle(self.handle); self.handle = None
        if self.process:
            self.k.CloseHandle(self.process); self.process = None


class PosixFamily:
    def __init__(self, command, cwd, stdout, stderr):
        self.process = subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL, start_new_session=True)
        self.pid = self.process.pid
        self.known = {}

    def pids(self):
        try:
            parent = psutil.Process(self.pid)
            self.known.update({p.pid: p.create_time() for p in [parent] + parent.children(recursive=True)})
        except psutil.NoSuchProcess:
            pass
        _, self.known = sampled_rss(self.known)
        return list(self.known)

    def poll(self):
        return self.process.poll()

    def close(self):
        try:
            os.killpg(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.process.wait(timeout=2)


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def run_guarded(command, *, limits_path, run_root, report_path, cwd, timeout_seconds=None, interval=.02):
    limits = Limits.read(limits_path)
    root = Path(run_root).resolve(); root.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_path).resolve()
    if not report_path.is_relative_to(root):
        raise ValueError("Guard report must be inside the capped artifact directory")
    if timeout_seconds is not None and (not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("Positive finite per-job timeout required")
    started = time.monotonic(); own = psutil.Process(); owner = {own.pid: own.create_time()}
    lock_path = root / ".numerical-worker.lock"
    # A stale lock is deliberately a manual recovery event, never silent reuse.
    lock = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    os.write(lock, json.dumps({"pid": own.pid, "create_time": own.create_time()}).encode()); os.close(lock)
    report = {"schema": "pr166_guard_v1", "command": command, "started_unix": time.time(), "sample_interval_seconds": interval,
              "rss_semantics": "maximum sampled sum of simultaneous guardian and owned-family RSS; short peaks may be missed",
              "peak_family_rss_bytes": 0, "samples": 0, "status": "refused", "reason": None}
    report["kernel_job_private_commit_cap_bytes"] = WindowsJob.commit_limit_bytes if os.name == "nt" else None
    report["kernel_cap_semantics"] = "aggregate job private committed memory; additional protection, not RSS"
    child = None
    old_handlers = {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Guardian interrupted")
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, interrupted)
        if limits.remaining() <= 0:
            raise TimeoutError("Shared absolute deadline exhausted before launch")
        if directory_bytes(root) >= limits.disk_bytes:
            raise RuntimeError("Artifact byte cap reached before launch")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        telemetry_path = report_path.with_suffix(".rss.jsonl")
        report["raw_rss_samples"] = str(telemetry_path.relative_to(root))
        with report_path.with_suffix(".stdout.log").open("wb") as stdout, report_path.with_suffix(".stderr.log").open("wb") as stderr, telemetry_path.open("w", encoding="utf-8") as telemetry:
            child = (WindowsJob if os.name == "nt" else PosixFamily)(command, cwd, stdout, stderr)
            report["worker_pid"] = child.pid
            next_disk = started
            while True:
                now = time.monotonic()
                if limits.remaining(monotonic=now) <= 0:
                    raise TimeoutError("Shared absolute deadline exhausted")
                if timeout_seconds is not None and now - started >= timeout_seconds:
                    raise TimeoutError("Per-job deadline exhausted within shared deadline")
                identities = dict(owner)
                for pid in child.pids():
                    try:
                        identities[pid] = psutil.Process(pid).create_time()
                    except psutil.NoSuchProcess:
                        continue
                members = []
                rss, _ = sampled_rss(identities, samples=members)
                telemetry.write(json.dumps({"unix": time.time(), "monotonic": now, "family_rss_bytes": rss,
                                            "guardian_pid": own.pid, "members": members})+"\n")
                telemetry.flush()
                report["samples"] += 1
                report["peak_family_rss_bytes"] = max(report["peak_family_rss_bytes"], rss)
                if rss >= limits.soft_rss:
                    raise MemoryError("Family RSS safety threshold reached")
                if now >= next_disk:
                    report["artifact_bytes_sampled"] = directory_bytes(root)
                    if report["artifact_bytes_sampled"] >= limits.disk_bytes:
                        raise RuntimeError("Artifact byte cap reached")
                    next_disk = now + .25
                code = child.poll()
                if code is not None:
                    # Descendants remain owned by the job and are killed below.
                    descendants = [pid for pid in child.pids() if pid != child.pid]
                    report["cleanup_evidence"] = {"live_descendant_pids_at_root_exit": descendants,
                                                  "live_descendants_terminated_by_guard": len(descendants)}
                    report.update(status="completed" if code == 0 else "worker_failed", returncode=code)
                    break
                time.sleep(min(interval, max(0, limits.remaining())))
    except BaseException as error:
        report.update(status="aborted", reason=type(error).__name__ + ": " + str(error))
    finally:
        if child is not None:
            child.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        report["wall_seconds"] = time.monotonic() - started
        report["finished_unix"] = time.time()
        atomic_json(report_path, report)
        lock_path.unlink(missing_ok=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limits", required=True); parser.add_argument("--run-root", required=True)
    parser.add_argument("--report", required=True); parser.add_argument("--cwd", default=os.getcwd())
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A worker command is required")
    report = run_guarded(command, limits_path=args.limits, run_root=args.run_root, report_path=args.report,
                         cwd=args.cwd, timeout_seconds=args.timeout_seconds)
    print(json.dumps(report, allow_nan=False))
    return 0 if report["status"] == "completed" else 125


if __name__ == "__main__":
    raise SystemExit(main())
