"""Measure the executable artifact, with explicit units and raw repetitions."""
from __future__ import annotations

import copy
import hashlib
import math
import time
from pathlib import Path

import torch
from torch import nn

from fedcore.tools.export import ExportError, export_model


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_bytes(model):
    """Tensor contents in state_dict, including buffers and quantized packs.

    This is distinct from the serialized file size and process memory. Aliases
    of the same tensor storage are counted once, including quantized tensors.
    """
    seen = set()
    def count(value):
        if isinstance(value, torch.Tensor):
            key = (str(value.device), value.untyped_storage().data_ptr())
            if key in seen:
                return 0
            seen.add(key)
            return value.untyped_storage().nbytes()
        if isinstance(value, dict):
            return sum(count(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(count(item) for item in value)
        return 0
    return count(model.state_dict())


def _percentile(values, quantile):
    values = sorted(values)
    position = (len(values) - 1) * quantile
    low = int(math.floor(position))
    high = int(math.ceil(position))
    return values[low] + (values[high] - values[low]) * (position - low)


def _rss():
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        return None


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def load_artifact(artifact):
    path = Path(artifact["path"])
    if not path.is_file() or file_hash(path) != artifact["sha256"]:
        raise ValueError("Artifact hash or existence check failed")
    if artifact["format"] == "torchscript":
        with path.open("rb") as stream:
            return torch.jit.load(stream, map_location=artifact["device"]).eval()
    if artifact["format"] == "onnx":
        if artifact["device"] != "cpu":
            raise ValueError("This ONNX measurement profile supports CPUExecutionProvider only")
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = artifact["threads"]
        options.inter_op_num_threads = 1
        return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    raise ValueError("Unsupported artifact loader")


def _call_loaded(loaded, artifact, x):
    if artifact["format"] == "torchscript":
        return loaded(x)
    outputs = loaded.run(None, {loaded.get_inputs()[0].name: x.detach().cpu().numpy()})
    if len(outputs) != 1:
        raise ValueError("The experiment profile requires one tensor output")
    return torch.from_numpy(outputs[0])


def predict_loaded_artifact(artifact, x):
    loaded = load_artifact(artifact)
    with torch.inference_mode():
        return _call_loaded(loaded, artifact, x.to(artifact["device"])).detach().cpu()


def measure_artifact(model: nn.Module, example_input: torch.Tensor, output_path,
                     *, format="torchscript", device="cpu", repeats=10,
                     warmup=2, threads=1):
    """Export, reload, verify and time sequential calls on a declared device.

    The caller must reserve an idle device. No search work is interleaved here.
    End-to-end calls include tensor transfer and output transfer, not data disk
    I/O. Loader cost is measured separately. RSS values are snapshots, not peak.
    """
    if any(type(value) is not int or value < minimum for value, minimum in ((repeats, 1), (warmup, 0), (threads, 1))):
        raise ValueError("Positive repeats/threads and nonnegative warmup are required")
    profile = {"device": str(device), "format": format, "threads": threads,
               "warmup": warmup, "repeats": repeats,
               "batch_size": len(example_input), "input_shape": list(example_input.shape),
               "dtype": str(example_input.dtype), "synchronization": "torch.cuda.synchronize" if str(device).startswith("cuda") else "synchronous CPU",
               "concurrency": 1, "idle_device_required": True,
               "units": {"latency_p50_ms": "ms/batch", "latency_p95_ms": "ms/batch",
                         "throughput": "samples/s", "file_bytes": "bytes",
                         "tensor_state_bytes": "bytes", "cpu_rss_before_bytes": "bytes",
                         "cpu_rss_after_bytes": "bytes", "cuda_peak_allocated_bytes": "bytes",
                         "cuda_peak_reserved_bytes": "bytes", "load_seconds": "s"}}
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return {"status": "unsupported", "reason": "CUDA is unavailable", "profile": profile}
    if format == "onnx" and device != "cpu":
        return {"status": "unsupported", "reason": "ONNX profile declares CPUExecutionProvider only", "profile": profile}
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        local = copy.deepcopy(model).to(device).eval()
        x_cpu = example_input.detach().cpu().clone()
        x = x_cpu.to(device)
        before = _rss()
        options = {"dynamic_axes": {"input": {0: "batch"}, "output": {0: "batch"}}} if format == "onnx" else None
        path = export_model(local, format, output_path, x, framework_config=options)
        artifact = {"path": str(path.resolve()), "format": format, "sha256": file_hash(path),
                    "device": str(device), "threads": threads}
        start = time.perf_counter()
        loaded = load_artifact(artifact)
        load_seconds = time.perf_counter() - start
        with torch.inference_mode():
            expected = local(x).detach().cpu()
            actual = _call_loaded(loaded, artifact, x).detach().cpu()
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            for _ in range(warmup):
                _call_loaded(loaded, artifact, x)
            _synchronize(device)
            if torch.device(device).type == "cuda":
                cuda_allocated_before = torch.cuda.memory_allocated(device)
                cuda_reserved_before = torch.cuda.memory_reserved(device)
                torch.cuda.reset_peak_memory_stats(device)
            raw, end_to_end = [], []
            for _ in range(repeats):
                _synchronize(device)
                start = time.perf_counter_ns()
                _call_loaded(loaded, artifact, x)
                _synchronize(device)
                raw.append((time.perf_counter_ns() - start) / 1e6)
            for _ in range(repeats):
                _synchronize(device)
                start = time.perf_counter_ns()
                result = _call_loaded(loaded, artifact, x_cpu.to(device))
                result.detach().cpu()
                _synchronize(device)
                end_to_end.append((time.perf_counter_ns() - start) / 1e6)
        cuda = torch.device(device).type == "cuda"
        values = {"latency_p50_ms": _percentile(raw, 0.5), "latency_p95_ms": _percentile(raw, 0.95),
                  "throughput": len(x) * len(raw) / (sum(raw) / 1000),
                  "end_to_end_p50_ms": _percentile(end_to_end, 0.5),
                  "end_to_end_p95_ms": _percentile(end_to_end, 0.95),
                  "file_bytes": path.stat().st_size, "tensor_state_bytes": tensor_state_bytes(local),
                  "cpu_rss_before_bytes": before, "cpu_rss_after_bytes": _rss(),
                  "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
                  "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else None,
                  "cuda_allocated_before_bytes": cuda_allocated_before if cuda else None,
                  "cuda_reserved_before_bytes": cuda_reserved_before if cuda else None,
                  "cuda_allocated_increment_bytes": max(0, torch.cuda.max_memory_allocated(device) - cuda_allocated_before) if cuda else None,
                  "cuda_reserved_increment_bytes": max(0, torch.cuda.max_memory_reserved(device) - cuda_reserved_before) if cuda else None,
                  "load_seconds": load_seconds}
        profile["runtime"] = "torch.jit" if format == "torchscript" else "onnxruntime CPUExecutionProvider"
        profile["hardware"] = torch.cuda.get_device_name(device) if cuda else "CPU"
        return {"status": "succeeded", "artifact": artifact, "profile": profile,
                "metrics": values, "raw_inference_ms": raw, "raw_end_to_end_ms": end_to_end,
                "rss_kind": "process snapshots; not peak and not attributed exclusively to model",
                "cuda_memory_kind": "process-wide peaks during measurement and increment over premeasurement snapshots; not model-exclusive",
                "quality_parity_checked": True}
    except (ImportError, ExportError) as error:
        return {"status": "unsupported" if isinstance(error, ImportError) else "failed",
                "reason": str(error), "error_type": type(error).__name__, "profile": profile,
                "error": error.to_dict() if isinstance(error, ExportError) else {"message": str(error)}}
    except Exception as error:
        return {"status": "failed", "reason": str(error), "error_type": type(error).__name__, "profile": profile}
    finally:
        torch.set_num_threads(original_threads)
