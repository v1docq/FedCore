"""Fresh-process TorchScript pilot measurements for the local SVD package.

Each result is one session, not an independent training replicate.  Individual
wall-clock, transfer-path, CUDA-event and blocked-throughput passes are separate.
"""
from __future__ import annotations

import gc
import atexit
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch

from fedcore.tools.export import export_model
from fedcore.tools.atomic_json import write_atomic_json
from . import measurement
from .protocol import tensor_hash


def precision_profile():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def background_snapshot():
    """Allowlisted host observations; GPU memory is physical global usage."""
    import psutil
    result = {"utc_epoch": time.time(), "available_ram_bytes": psutil.virtual_memory().available,
              "cpu_percent": psutil.cpu_percent(interval=.2), "pid": os.getpid()}
    try:
        raw = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             check=True, timeout=10).stdout.strip()
        fields = raw.splitlines()[0].split(",")
        result["gpu"] = {"name": fields[0].strip(), "utilization_percent": int(fields[1]),
                         "physical_used_mib": int(fields[2]), "physical_free_mib": int(fields[3]),
                         "scope": "global physical GPU usage, including unrelated processes and desktop renderer; not attributable to this model"}
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result["gpu_observation_error"] = type(error).__name__
    return result


def noise_reasons(before, after, device):
    reasons = []
    for name, sample in (("before", before), ("after", after)):
        if sample["cpu_percent"] > 25:
            reasons.append(f"background CPU >25% {name}")
        if str(device).startswith("cuda"):
            gpu = sample.get("gpu")
            if gpu is None:
                reasons.append(f"GPU background observation unavailable {name}")
            elif gpu["utilization_percent"] > 10:
                reasons.append(f"background GPU >10% {name}")
    return reasons


def _summary(raw):
    return {"p50_ms": statistics.median(raw), "p95_ms": measurement._percentile(raw, .95),
            "min_ms": min(raw), "max_ms": max(raw), "calls": len(raw)}


def timing_worker(job):
    """Export a new profile artifact, free the reference, reload and measure."""
    precision_profile()
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    raw_wall, raw_path, raw_events, blocked = [], [], [], []
    def retain_failed_raw():
        # No disk I/O during timed passes.  Normal Python failures retain even
        # a partially completed pass; forced process termination is recorded
        # by the owning resource monitor and cannot promise an atexit flush.
        write_atomic_json(directory/"failed-raw.json",{"job":job,"status":"failed_or_incomplete",
            "raw_synchronized_wall_ms":raw_wall,"raw_full_path_ms":raw_path,
            "raw_cuda_event_ms":raw_events,"blocked_throughput":blocked})
    atexit.register(retain_failed_raw)
    device = job["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Approved CUDA profile is unavailable; no CPU substitution")
    source = Path(job["source_artifact"])
    if measurement.file_hash(source) != job["source_artifact_sha256"]:
        raise ValueError("Frozen source artifact changed")
    input_path = Path(job["input_path"])
    if measurement.file_hash(input_path) != job["input_file_sha256"]:
        raise ValueError("Frozen timing input changed")
    x_cpu = torch.load(input_path, map_location="cpu", weights_only=True)[:job["batch"]].clone()
    with source.open("rb") as stream:
        original = torch.jit.load(stream, map_location="cpu").eval()
    source_weight_hash = tensor_hash(torch.cat([p.detach().flatten().cpu()
                                                for p in original.parameters()]))
    before = background_snapshot()
    original.to(device)
    x = x_cpu.to(device)
    start = time.perf_counter()
    path = export_model(original, "torchscript", directory / "model", x)
    measurement._synchronize(device)
    export_seconds = time.perf_counter() - start
    with torch.inference_mode():
        expected = original(x).detach().cpu()
        other = x_cpu[:1] if job["batch"] == 64 else x_cpu.repeat(64, *([1]*(x_cpu.ndim-1)))
        expected_other = original(other.to(device)).detach().cpu()
    # The original reference must not inflate loaded-model allocator snapshots.
    del original
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    before_load_allocated = torch.cuda.memory_allocated(device) if device.startswith("cuda") else None
    artifact = {"path": str(path.resolve()), "sha256": measurement.file_hash(path),
                "format": "torchscript", "device": device, "threads": 1}
    start = time.perf_counter()
    loaded = measurement.load_artifact(artifact)
    measurement._synchronize(device)
    load_seconds = time.perf_counter()-start
    with torch.inference_mode():
        measurement._synchronize(device)
        start = time.perf_counter_ns()
        actual = loaded(x)
        measurement._synchronize(device)
        first_call_ms = (time.perf_counter_ns()-start)/1e6
        torch.testing.assert_close(actual.detach().cpu(), expected, rtol=1e-4, atol=1e-5)
        # Check the other approved shape on this same graph, with an explicit
        # new fixed-shape descriptor for the timed profile.
        other_output = loaded(other.to(device)).detach().cpu()
        torch.testing.assert_close(other_output, expected_other, rtol=1e-4, atol=1e-5)
        for _ in range(job["warmup"]):
            loaded(x)
        measurement._synchronize(device)
        cuda = device.startswith("cuda")
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        raw_wall, raw_path, raw_events = [], [], []
        for _ in range(job["repeats"]):
            measurement._synchronize(device)
            start = time.perf_counter_ns()
            loaded(x)
            measurement._synchronize(device)
            raw_wall.append((time.perf_counter_ns()-start)/1e6)
        for _ in range(job["repeats"]):
            measurement._synchronize(device)
            start = time.perf_counter_ns()
            loaded(x_cpu.to(device)).detach().cpu()
            measurement._synchronize(device)
            raw_path.append((time.perf_counter_ns()-start)/1e6)
        if cuda:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for _ in range(job["repeats"]):
                begin.record()
                loaded(x)
                end.record()
                end.synchronize()
                raw_events.append(begin.elapsed_time(end))
        blocked = []
        for _ in range(5):
            measurement._synchronize(device)
            start = time.perf_counter_ns()
            for _ in range(100):
                loaded(x)
            measurement._synchronize(device)
            elapsed = (time.perf_counter_ns()-start)/1e9
            blocked.append({"calls": 100, "seconds": elapsed, "samples_per_second": len(x)*100/elapsed})
    memory = {"process_rss_bytes": measurement._rss(),
              "loaded_tensor_bytes": measurement.tensor_state_bytes(loaded),
              "cuda_allocated_before_load_bytes": before_load_allocated,
              "cuda_allocated_bytes": torch.cuda.memory_allocated(device) if cuda else None,
              "cuda_reserved_bytes": torch.cuda.memory_reserved(device) if cuda else None,
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
              "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else None,
              "scope": "PyTorch process allocator; reference released before loaded graph; physical global GPU usage separately"}
    if cuda and max(memory["cuda_peak_allocated_bytes"],memory["cuda_peak_reserved_bytes"]) > job.get("gpu_allocator_limit_bytes",14*1024**3):
        raise ValueError("Approved14GiB GPU process allocator budget exceeded")
    # NVIDIA utilization averages include our own preceding calls.  Let that
    # sampling window drain before observing independent background utilization.
    time.sleep(1.2 if cuda else .2)
    after = background_snapshot()
    reasons = noise_reasons(before, after, device)
    result = {"schema": 1, "status": "invalid_noise" if reasons else "succeeded", "noise_reasons": reasons,
              "job": job, "os_pid": os.getpid(), "parent_pid": os.getppid(),
              "source_artifact_sha256": job["source_artifact_sha256"],
              "source_weight_sha256": source_weight_hash, "input_sha256": tensor_hash(x_cpu),
              "output_sha256": tensor_hash(expected), "artifact": artifact,
              "background_before": before, "background_after": after, "memory": memory,
              "cold": {"export_seconds": export_seconds, "load_seconds": load_seconds,
                       "first_loaded_graph_call_ms": first_call_ms,
                       "device_prepared_by_export": True},
              "raw_synchronized_wall_ms": raw_wall, "raw_full_path_ms": raw_path,
              "raw_cuda_event_ms": raw_events, "blocked_throughput": blocked,
              "wall": _summary(raw_wall), "full_path": _summary(raw_path),
              "cuda_event": _summary(raw_events) if raw_events else None,
              "profile": {"precision": "float32", "matmul_tf32": False, "cudnn_tf32": False,
                          "cudnn_benchmark": False, "cudnn_deterministic": True,
                          "runtime": "torch.jit", "statistical_unit": "one session within paired block",
                          "profiler": False, "batch_shape_checked": True}}
    write_atomic_json(directory / "result.json", result)
    atexit.unregister(retain_failed_raw)
    return result
