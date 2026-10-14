"""Unprofiled inference and separate operator traces for the second SVD packet.

Only the owning controller observes external GPU quietness.  A worker must have
an admission token before any CUDA query, context creation, export or warmup.
CUDA events span stream elapsed time, including possible host launch gaps.
"""
from __future__ import annotations

import atexit
import gc
import os
from pathlib import Path
import statistics
import time

import torch

from fedcore.tools.atomic_json import write_atomic_json
from fedcore.tools.export import export_model
from . import measurement
from .protocol import ProtocolError, tensor_hash


def precision_profile(threads=1):
    if threads not in (1, 8):
        raise ProtocolError("Only approved one/eight thread profiles are supported")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _summary(raw):
    return {"p50_ms": statistics.median(raw),
            "p95_ms": measurement._percentile(raw, .95),
            "min_ms": min(raw), "max_ms": max(raw), "calls": len(raw)}


def require_admission(job):
    """Pure precondition; deliberately calls no torch CUDA API."""
    import json
    token = json.loads(Path(job["admission_path"]).read_text(encoding="utf-8"))
    if (token.get("status") != "admitted" or token.get("session_id") != job["session_id"]
            or token.get("protocol_sha256") != job["protocol_sha256"]
            or token.get("before_worker_spawn") is not True):
        raise ProtocolError("A matching external pre-CUDA admission token is required")
    return token


def timing_worker(job):
    """One new process and export per session; never train or retry a session."""
    admission = require_admission(job)
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "result.json").exists():
        raise ProtocolError("Never overwrite a recorded worker result")
    raw_wall, raw_path, raw_events, blocked = [], [], [], []
    phase = "setup"

    def stage(name):
        nonlocal phase
        phase = name
        write_atomic_json(directory / "stage.json", {"stage": name, "pid": os.getpid(),
                          "utc_epoch": time.time(), "monotonic_seconds": time.monotonic()})

    def retain_failure():
        write_atomic_json(directory / "failed-raw.json", {"job": job, "status": "failed_or_incomplete",
            "stage": phase, "raw_synchronized_wall_ms": raw_wall, "raw_full_path_ms": raw_path,
            "raw_cuda_event_ms": raw_events, "blocked_throughput": blocked})

    atexit.register(retain_failure)
    stage("setup")
    precision_profile(1)
    device = job["device"]
    cuda = device.startswith("cuda")
    # This is the first CUDA query in the worker, after admission above.
    if cuda and not torch.cuda.is_available():
        raise ProtocolError("Approved CUDA unavailable; substitution is forbidden")
    if cuda and job.get("expected_gpu_name") and torch.cuda.get_device_name(0) != job["expected_gpu_name"]:
        raise ProtocolError("CUDA index0 differs from the externally admitted frozen GPU")
    source, input_path = Path(job["source_artifact"]), Path(job["input_path"])
    if measurement.file_hash(source) != job["source_artifact_sha256"]:
        raise ProtocolError("Source artifact changed")
    if measurement.file_hash(input_path) != job["input_file_sha256"]:
        raise ProtocolError("Timing input changed")
    x_cpu = torch.load(input_path, map_location="cpu", weights_only=True)[:job["batch"]].clone()
    if x_cpu.dtype != torch.float32 or len(x_cpu) != job["batch"]:
        raise ProtocolError("Expected complete fixed FP32 input batch")
    # On Windows the C++ filename loader can reject a Unicode path which
    # Python opens correctly.  Keep the exact artifact/hash and pass its stream.
    with source.open("rb") as stream:
        reference = torch.jit.load(stream, map_location="cpu").eval()
    if any(p.dtype != torch.float32 for p in reference.parameters()):
        raise ProtocolError("Expected FP32 model parameters")
    source_weight_hash = tensor_hash(torch.cat([p.detach().flatten().cpu() for p in reference.parameters()]))
    reference.to(device)
    x = x_cpu.to(device)
    start = time.perf_counter()
    artifact_path = export_model(reference, "torchscript", directory / "model", x)
    measurement._synchronize(device)
    export_seconds = time.perf_counter() - start
    with torch.inference_mode():
        expected = reference(x).detach().cpu()
        alternate = x_cpu[:1] if job["batch"] == 64 else x_cpu.repeat(64, *([1] * (x_cpu.ndim - 1)))
        expected_alternate = reference(alternate.to(device)).detach().cpu()
    del reference
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    allocated_before = torch.cuda.memory_allocated(device) if cuda else None
    artifact = {"path": str(artifact_path.resolve()), "sha256": measurement.file_hash(artifact_path),
                "format": "torchscript", "device": device, "threads": 1}
    start = time.perf_counter()
    loaded = measurement.load_artifact(artifact)
    measurement._synchronize(device)
    load_seconds = time.perf_counter() - start
    with torch.inference_mode():
        start = time.perf_counter_ns()
        actual = loaded(x)
        measurement._synchronize(device)
        first_call_ms = (time.perf_counter_ns() - start) / 1e6
        torch.testing.assert_close(actual.detach().cpu(), expected, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(loaded(alternate.to(device)).detach().cpu(), expected_alternate,
                                   rtol=1e-4, atol=1e-5)
        stage("warmup")
        warmup_start = time.perf_counter()
        warmup_calls = 0
        for _ in range(job["warmup"]):
            loaded(x)
            warmup_calls += 1
        measurement._synchronize(device)
        while time.perf_counter() - warmup_start < job.get("minimum_warmup_seconds", .2):
            if warmup_calls >= job.get("maximum_warmup_calls", 100000):
                raise ProtocolError("Bounded warmup cannot reach minimum elapsed duration")
            loaded(x)
            warmup_calls += 1
            measurement._synchronize(device)
        measurement._synchronize(device)
        warmup_seconds = time.perf_counter() - warmup_start
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        stage("profiled" if job.get("profiler") else "timed")
        if job.get("profiler"):
            activities = [torch.profiler.ProfilerActivity.CPU]
            if cuda:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, record_shapes=True,
                                        profile_memory=True, with_stack=False) as profiler:
                for _ in range(job["profile_calls"]):
                    loaded(x)
                measurement._synchronize(device)
            trace_path = directory / "trace.json"
            profiler.export_chrome_trace(str(trace_path))
            if trace_path.stat().st_size > job["trace_limit_bytes"]:
                raise ProtocolError("Profiler trace exceeded frozen per-file cap; retained for audit")
            operators = [{"name": event.key, "calls": event.count,
                "self_cpu_time_us": event.self_cpu_time_total, "cpu_time_us": event.cpu_time_total,
                "self_cuda_time_us": event.self_cuda_time_total, "cuda_time_us": event.cuda_time_total,
                "cpu_memory_bytes": event.cpu_memory_usage, "cuda_memory_bytes": event.cuda_memory_usage,
                "input_shapes": event.input_shapes} for event in profiler.key_averages(group_by_input_shape=True)]
        else:
            for _ in range(job["repeats"]):
                measurement._synchronize(device)
                start = time.perf_counter_ns()
                loaded(x)
                measurement._synchronize(device)
                raw_wall.append((time.perf_counter_ns() - start) / 1e6)
            for _ in range(job["repeats"]):
                measurement._synchronize(device)
                start = time.perf_counter_ns()
                loaded(x_cpu.to(device)).detach().cpu()
                measurement._synchronize(device)
                raw_path.append((time.perf_counter_ns() - start) / 1e6)
            if cuda:
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                for _ in range(job["repeats"]):
                    begin.record()
                    loaded(x)
                    end.record()
                    end.synchronize()
                    raw_events.append(begin.elapsed_time(end))
            for _ in range(job["throughput_blocks"]):
                measurement._synchronize(device)
                start = time.perf_counter_ns()
                for _ in range(job["throughput_calls"]):
                    loaded(x)
                measurement._synchronize(device)
                seconds = (time.perf_counter_ns() - start) / 1e9
                blocked.append({"calls": job["throughput_calls"], "seconds": seconds,
                                "samples_per_second": len(x) * job["throughput_calls"] / seconds})
    stage("drain")
    memory = {"process_rss_bytes": measurement._rss(), "loaded_tensor_bytes": measurement.tensor_state_bytes(loaded),
        "cuda_allocated_before_load_bytes": allocated_before,
        "cuda_allocated_bytes": torch.cuda.memory_allocated(device) if cuda else None,
        "cuda_reserved_bytes": torch.cuda.memory_reserved(device) if cuda else None,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else None,
        "scope": "owned torch allocator; reference released before loaded graph; physical GPU separate"}
    if cuda and max(memory["cuda_peak_allocated_bytes"], memory["cuda_peak_reserved_bytes"]) > job["gpu_allocator_limit_bytes"]:
        raise ProtocolError("Owned torch allocator exceeded 14GiB")
    result = {"schema": 2, "status": "succeeded", "job": job, "os_pid": os.getpid(),
        "source_artifact_sha256": job["source_artifact_sha256"], "source_parameter_values_sha256": source_weight_hash,
        "input_sha256": tensor_hash(x_cpu), "output_sha256": tensor_hash(expected), "artifact": artifact,
        "admission": admission, "memory": memory,
        "cold": {"export_seconds": export_seconds, "load_seconds": load_seconds,
                 "first_loaded_graph_call_ms": first_call_ms, "device_prepared_by_export": True},
        "warmup": {"calls": warmup_calls, "seconds": warmup_seconds,
                   "minimum_seconds": job.get("minimum_warmup_seconds", .2)},
        "raw_synchronized_wall_ms": raw_wall, "raw_full_path_ms": raw_path,
        "raw_cuda_event_ms": raw_events, "blocked_throughput": blocked,
        "wall": _summary(raw_wall) if raw_wall else None, "full_path": _summary(raw_path) if raw_path else None,
        "cuda_event": _summary(raw_events) if raw_events else None,
        "profile": {"precision": "float32", "threads": 1,
                    "matmul_tf32": False, "cudnn_tf32": False, "cudnn_benchmark": False,
                    "cudnn_deterministic": True, "runtime": "torch.jit", "profiler": bool(job.get("profiler")),
                    "batch_shape_checked": True, "statistical_unit": "one session within a paired timing block",
                    "cuda_event_scope": "stream elapsed interval; may include host launch gaps; not pure kernel time"}}
    if job.get("profiler"):
        result.update(operator_rows=operators, trace={"path": str(trace_path.resolve()),
            "sha256": measurement.file_hash(trace_path), "bytes": trace_path.stat().st_size},
            diagnostic_only=True)
    write_atomic_json(directory / "worker-result.json", result)
    stage("done")
    atexit.unregister(retain_failure)
    return result
