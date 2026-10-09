"""Effect interpreter for validated plans; all model execution occurs in the worker."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from .contracts import ContractError, ExecutionPlan
from .models import load_model_bundle
from .security import confined_path, safe_load


def _factorize(model, request, decisions):
    from fedcore.models.network_impl.decomposed_layers import DecomposedLinear, DecomposedConv1d, DecomposedConv2d
    from fedcore.algorithm.low_rank.rank_pruning import rank_threshold_pruning_in_place
    if type(model) is nn.Sequential:
        return nn.Sequential(*(_factorize(layer, request, decisions) for layer in model.children()))
    constructors = {nn.Linear: DecomposedLinear, nn.Conv1d: DecomposedConv1d, nn.Conv2d: DecomposedConv2d}
    if type(model) not in constructors:
        return model
    # Corrected FedCore components use tdecomp; this dependency never points back.
    decomposed = constructors[type(model)](model, decomposer="svd")
    if request.rank is None:
        rank_threshold_pruning_in_place(decomposed, threshold=request.retained_energy,
                                        strategy="explained_variance", round_to_times=1)
    else:
        u, s, vh = decomposed.canonicalize()
        if request.rank > s.shape[-1]:
            raise ContractError("invalid_rank", "Requested rank exceeds the operator dimensions", "rank")
        decomposed.set_U_S_Vh(u[..., :request.rank], s[..., :request.rank], vh[..., :request.rank, :])
    u, s, vh = decomposed.get_U_S_Vh()
    rank = s.shape[-1]
    decisions.append({"type": type(model).__name__, "rank": rank, "groups": getattr(model, "groups", 1),
                      "criterion": "rank" if request.rank is not None else "squared_frobenius_energy"})
    dtype = model.weight.dtype
    if type(model) is nn.Linear:
        first = nn.Linear(model.in_features, rank, bias=False, dtype=dtype)
        last = nn.Linear(rank, model.out_features, bias=model.bias is not None, dtype=dtype)
        weights = (vh, u * s.unsqueeze(-2))
    else:
        first_weight, last_weight, args1, args2 = decomposed.factor_weights()
        cls = type(model)
        first = cls(model.in_channels, rank * model.groups, model.kernel_size,
                    groups=model.groups, bias=False, padding_mode=model.padding_mode, dtype=dtype, **args1)
        last = cls(rank * model.groups, model.out_channels, 1,
                   groups=model.groups, bias=model.bias is not None, dtype=dtype, **args2)
        weights = (first_weight, last_weight)
    with torch.no_grad():
        first.weight.copy_(weights[0])
        last.weight.copy_(weights[1])
        if model.bias is not None:
            last.bias.copy_(model.bias)
    candidate = nn.Sequential(first, last).eval()
    # A factorization that increases storage remains a valid approximation, but
    # materialize it to avoid claiming a parameter reduction that did not occur.
    before = sum(p.numel() for p in model.parameters())
    after = sum(p.numel() for p in candidate.parameters())
    if after >= before:
        from fedcore.algorithm.low_rank.reassembly.decomposed_recreation import to_standard_module
        decisions[-1]["representation"] = "materialized"
        return to_standard_module(decomposed).eval()
    decisions[-1]["representation"] = "two_layers"
    return candidate


def _dataset(path, spec, max_bytes):
    obj = safe_load(path, max_bytes)
    if not isinstance(obj, dict) or set(obj) != {"kind", "version", "features", "targets"} or obj["kind"] != "fedcore_tensor_dataset" or type(obj["version"]) is not int or obj["version"] != 1:
        raise ContractError("invalid_dataset", "Expected versioned features/targets tensor dataset")
    features, targets = obj["features"], obj["targets"]
    spec.validate_tensor(features, batch_dynamic=True)
    if type(targets) is not torch.Tensor or targets.ndim == 0 or targets.shape[0] != features.shape[0]:
        raise ContractError("invalid_dataset", "Targets and features must have aligned sample axes")
    return features, targets


def _quality(output, target, task):
    if not isinstance(output, torch.Tensor) or not torch.isfinite(output).all():
        raise ContractError("invalid_output", "Model must return one finite tensor")
    if task == "regression":
        if output.shape != target.shape:
            raise ContractError("target_mismatch", "Regression output and targets require identical shapes")
        return {"name": "mean_squared_error", "value": float((output.to(torch.float64) - target).square().mean())}
    if output.ndim != 2 or target.ndim != 1 or target.dtype != torch.int64 or target.min() < 0 or target.max() >= output.shape[1]:
        raise ContractError("target_mismatch", "Classification requires [N, classes] scores and int64 [N] labels")
    return {"name": "accuracy", "value": float((output.argmax(1) == target).to(torch.float64).mean())}


def _latency(model, example, repetitions):
    with torch.inference_mode():
        model(example)  # explicit one warmup, same input for both models
        start = time.perf_counter()
        for _ in range(repetitions):
            model(example)
    return (time.perf_counter() - start) * 1000 / repetitions


def execute_plan(plan: ExecutionPlan, job_dir) -> dict:
    worker_started = time.perf_counter()
    timings = {}
    request = plan.request
    root = Path(job_dir)
    torch.set_num_threads(request.resources.threads)
    model_path = confined_path(root, request.model)
    stage_started = time.perf_counter()
    model = load_model_bundle(model_path, request.resources.max_bytes)
    timings["model_load_seconds"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    example = safe_load(confined_path(root, request.example), request.resources.max_bytes)
    request.input_spec.validate_tensor(example)
    validation, targets = _dataset(confined_path(root, request.data.validation), request.input_spec, request.resources.max_bytes)
    for name in (request.data.train, request.data.calibration):
        if name is not None:
            _dataset(confined_path(root, name), request.input_spec, request.resources.max_bytes)
    timings["data_decode_seconds"] = time.perf_counter() - stage_started
    decisions = []
    stage_started = time.perf_counter()
    compressed = _factorize(model, request, decisions).cpu().eval() if request.method == "svd" else model
    timings["transformation_seconds"] = time.perf_counter() - stage_started
    if not decisions and request.method == "svd":
        raise ContractError("unsupported_architecture", "No supported Linear/Conv layer to compress")
    stage_started = time.perf_counter()
    with torch.inference_mode():
        before = model(validation)
        after = compressed(validation)
        quality_before = _quality(before, targets, request.task)
        quality_after = _quality(after, targets, request.task)
        difference = torch.linalg.vector_norm((after - before).to(torch.float64))
        norm = torch.linalg.vector_norm(before.to(torch.float64))
        relative = float(difference / norm) if norm > 0 else (0.0 if difference == 0 else float("inf"))
    timings["validation_seconds"] = time.perf_counter() - stage_started
    if relative > request.max_relative_error:
        raise ContractError("error_budget_exceeded", f"Validation output relative error {relative:g} exceeds {request.max_relative_error:g}")
    from fedcore.tools.export import export_model
    stage_started = time.perf_counter()
    artifact = export_model(compressed, request.artifact_format, confined_path(root, plan.artifact_name, must_exist=False), example)
    timings["export_seconds"] = time.perf_counter() - stage_started
    parameters_before = sum(p.numel() for p in model.parameters())
    parameters_after = sum(p.numel() for p in compressed.parameters())
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "uninstalled-source"
    from fedcore import __version__ as code_version
    stage_started = time.perf_counter()
    baseline_latency = _latency(model, example, request.resources.repetitions)
    compressed_latency = _latency(compressed, example, request.resources.repetitions)
    timings["inference_measurement_seconds"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    result = {"version": 1, "status": "succeeded", "artifact": artifact.name,
            "timings": timings,
            "artifact_format": request.artifact_format, "input_spec": asdict(request.input_spec),
            "profile": asdict(request.profile), "task": request.task, "method": request.method,
            "parameters": {"rank": request.rank, "retained_energy": request.retained_energy,
                           "max_relative_error": request.max_relative_error, "layers": decisions},
            "versions": {"fedcore": code_version, "fedcore_distribution": version("fedcore"), "torch": torch.__version__, "tdecomp": version("tdecomp"), "python": platform.python_version()},
            "provenance": {"model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                           "validation_sha256": hashlib.sha256(confined_path(root, request.data.validation).read_bytes()).hexdigest(),
                           "request": request.to_dict()},
            "metrics": {"relative_output_error": relative,
                        "baseline": {"quality": quality_before, "parameters": parameters_before,
                                     "tensor_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                                     "latency_ms": baseline_latency},
                        "compressed": {"quality": quality_after, "parameters": parameters_after,
                                       "tensor_bytes": sum(p.numel() * p.element_size() for p in compressed.parameters()),
                                       "latency_ms": compressed_latency},
                        "measurement": {"device": "cpu", "batch_size": example.shape[0], "validation_samples": validation.shape[0],
                                        "repetitions": request.resources.repetitions, "warmup": 1, "threads": request.resources.threads,
                                        "latency_method": "perf_counter_wall_clock", "artifact_bytes": artifact.stat().st_size,
                                        "energy": {"status": "unsupported", "reason": "No calibrated energy meter is configured"}}}}
    timings["provenance_and_result_seconds"] = time.perf_counter() - stage_started
    timings["execute_plan_seconds"] = time.perf_counter() - worker_started
    return result
