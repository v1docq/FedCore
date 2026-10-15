"""Same-process counterbalanced TorchScript inference; no profiler or training."""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import statistics
import time

import torch

from fedcore.tools.atomic_json import write_atomic_json
from . import measurement
from .protocol import ProtocolError, tensor_hash
from .svd_second_timing import precision_profile, require_admission


def _load(record, device):
    source = Path(record["path"])
    if measurement.file_hash(source) != record["sha256"]:
        raise ProtocolError("Frozen graph changed")
    with source.open("rb") as stream:
        return torch.jit.load(stream, map_location=device).eval()


def measure(model, x, x_cpu, device, calls, throughput_calls, raw=None):
    """Three separate latency series and one equal-work throughput duration."""
    cuda = device.startswith("cuda")
    if raw is None:
        raw = {"wall_ms": [], "full_path_ms": [], "cuda_event_ms": [], "throughput_seconds": []}
    for _ in range(calls):
        measurement._synchronize(device)
        started = time.perf_counter_ns()
        model(x)
        measurement._synchronize(device)
        raw["wall_ms"].append((time.perf_counter_ns() - started) / 1e6)
    for _ in range(calls):
        measurement._synchronize(device)
        started = time.perf_counter_ns()
        model(x_cpu.to(device)).detach().cpu()
        measurement._synchronize(device)
        raw["full_path_ms"].append((time.perf_counter_ns() - started) / 1e6)
    if cuda:
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(calls):
            begin.record()
            model(x)
            end.record()
            end.synchronize()
            raw["cuda_event_ms"].append(begin.elapsed_time(end))
    measurement._synchronize(device)
    started = time.perf_counter_ns()
    for _ in range(throughput_calls):
        model(x)
    measurement._synchronize(device)
    raw["throughput_seconds"].append((time.perf_counter_ns() - started) / 1e9)
    raw["samples_per_second"] = len(x) * throughput_calls / raw["throughput_seconds"][0]
    return raw


def timing_worker(job):
    admission = require_admission(job)
    directory = Path(job["directory"])
    protocol = json.loads(Path(job["protocol_path"]).read_text(encoding="utf-8"))
    if protocol["protocol_sha256"] != job["protocol_sha256"]:
        raise ProtocolError("Worker protocol differs")
    records, phase = [], "setup"

    def stage(value):
        nonlocal phase
        phase = value
        write_atomic_json(directory / "stage.json", {"stage": value, "pid": os.getpid(), "utc_epoch": time.time()})

    def retain():
        write_atomic_json(directory / "failed-raw.json", {"job": job, "stage": phase, "records": records,
                                                       "status": "failed_or_incomplete"})

    atexit.register(retain)
    stage("setup")
    precision_profile(1)
    device, cuda = job["device"], job["device"].startswith("cuda")
    if cuda and (not torch.cuda.is_available() or torch.cuda.get_device_name(0) != job["expected_gpu_name"]):
        raise ProtocolError("Frozen GPU unavailable or changed")
    prepared = protocol["prepared"]
    source = Path(prepared["input"]["path"])
    if measurement.file_hash(source) != prepared["input"]["sha256"]:
        raise ProtocolError("Frozen input file differs")
    x_cpu = torch.load(source, map_location="cpu", weights_only=True)[:job["batch"]].clone()
    input_hash = tensor_hash(x_cpu)
    if input_hash != prepared["input"]["tensor_sha256_by_batch"][str(job["batch"])]:
        raise ProtocolError("Frozen input tensor differs")
    x = x_cpu.to(device)
    copies = {name: tuple(_load(record, device) for _ in range(2)) for name, record in prepared["graphs"].items()}
    warmup, parity = {}, []
    with torch.inference_mode():
        stage("warmup")
        for name, models in copies.items():
            torch.testing.assert_close(models[0](x), models[1](x), rtol=0, atol=0)
            for index, model in enumerate(models):
                started, count = time.perf_counter(), 0
                while count < 50 or time.perf_counter() - started < .25:
                    if count >= 10000:
                        raise ProtocolError("Fixed bounded warmup exhausted")
                    model(x)
                    measurement._synchronize(device)
                    count += 1
                warmup[f"{name}-{index}"] = {"calls": count, "seconds": time.perf_counter() - started}
        for name in ("svd", "weighted"):
            expected = copies[f"{name}_factor"][0](x)
            actual = copies[f"{name}_materialized"][0](x)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            parity.append({"subject": name, "max_abs": (actual - expected).abs().max().item(),
                           "argmax_agreement": bool(torch.equal(actual.argmax(1), expected.argmax(1)))})
        if cuda and max(torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)) > 14 * 1024**3:
            raise ProtocolError("Owned CUDA allocator above14GiB during load/warmup")
        for unit in protocol["schedules"][job["stage"]]:
            stage(unit["unit_id"])
            models = {side: copies[name][int(side == "b" and unit["kind"] == "control")]
                      for side, name in unit["models"].items()}
            for side in unit["order"]:
                for _ in range(2):
                    models[side](x)
                measurement._synchronize(device)
            record = {"unit": unit, "status": "incomplete", "sides": {}}
            records.append(record)
            for side in unit["order"]:
                raw = {"wall_ms": [], "full_path_ms": [], "cuda_event_ms": [], "throughput_seconds": [],
                       "source_sha256": prepared["graphs"][unit["models"][side]]["sha256"], "input_sha256": input_hash}
                record["sides"][side] = raw
                measure(models[side], x, x_cpu, device, 10 if job["stage"] == "pilot" else 20, 50, raw)
                # Persist every completed side, including a partial failed pair.
                with (directory / "calls.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"unit": unit, "side": side, "raw": raw}, allow_nan=False) + "\n")
            record["status"] = "succeeded"
            write_atomic_json(directory / "progress.json", {"completed_units": len(records), "last": record})
            if cuda and max(torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)) > 14 * 1024**3:
                raise ProtocolError("Owned CUDA allocator above14GiB")
    memory = {"rss_bytes": measurement._rss(), "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
              "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else None}
    result = {"schema": 3, "status": "succeeded", "job": job, "records": records, "warmup": warmup,
              "parity": parity, "memory": memory, "admission": admission, "pid": os.getpid(),
              "control_runtime": "two independently loaded TorchScript objects from identical bytes, both warmed",
              "metrics": {"wall_ms": "pre-resident input, synchronized host wall per forward",
                          "full_path_ms": "host input transfer + forward + output to CPU, synchronized host wall",
                          "cuda_event_ms": "stream elapsed interval, includes potential host launch gaps; not pure kernels",
                          "throughput_seconds": "50 forwards with pre-resident inputs and one final synchronization"},
              "profile": {"dtype": "float32", "threads": 1, "runtime": "torch.jit", "profiler": False,
                          "tf32": False, "cudnn_benchmark": False, "cudnn_deterministic": True}}
    write_atomic_json(directory / "worker-result.json", result)
    atexit.unregister(retain)
    stage("done")
    return result
