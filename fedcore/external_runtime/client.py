"""Caller-side convenience adapter; transports tensors without importing FEDOT."""
from __future__ import annotations
import tempfile
from pathlib import Path
import torch
from .contracts import (CompressionRequest, DataRoles, DeviceProfile, InputSpec,
                        Resources, WeightedOptions, MethodOptions, P2_EXTERNAL_METHODS)
from .jobs import JobRunner, JobStore
from .models import save_model_bundle
from .security import safe_save


def save_dataset(path, features, targets):
    safe_save({"kind": "fedcore_tensor_dataset", "version": 1,
                "features": features.detach().cpu().clone(), "targets": targets.detach().cpu().clone()}, path)


def compress(model, example, validation, *, jobs_root, task="regression", rank=None,
             retained_energy=1.0, max_relative_error=1e-4, artifact_format="torchscript",
             profile=None, resources=None, train=None, calibration=None, python_executable=None,
             method="svd", weighted_options=None, method_options=None):
    if not isinstance(example, torch.Tensor):
        from .contracts import ContractError
        raise ContractError("unsupported_input", "External v1 requires one tensor input; multi-input is unsupported")
    spec = InputSpec(tuple(example.shape), str(example.dtype).removeprefix("torch."))
    roles = DataRoles("validation.fcb", "train.fcb" if train is not None else None,
                      "calibration.fcb" if calibration is not None else None)
    weighted = WeightedOptions() if method == "weighted_svd" and weighted_options is None else weighted_options
    if method in P2_EXTERNAL_METHODS and method_options is None:
        from fedcore.algorithm.low_rank.method_specs import parse_method
        method_options = MethodOptions(parse_method(method))
    request = CompressionRequest("model.fcb", "example.fcb", spec, roles, task=task, rank=rank, method=method,
                                 retained_energy=retained_energy, max_relative_error=max_relative_error,
                                 artifact_format=artifact_format, profile=profile or DeviceProfile(), resources=resources or Resources(),
                                 version=3 if method in P2_EXTERNAL_METHODS else (2 if method == "weighted_svd" else 1),
                                 weighted=weighted, method_profile=method_options)
    if method == "weighted_svd":
        from fedcore.algorithm.low_rank.execution import plan_weighted
        from fedcore.algorithm.low_rank.plans import MetricPolicy
        plan_weighted(model, rank=rank, policy=MetricPolicy(weighted.ridge, weighted.rcond,
                      weighted.nullspace_policy), max_peak_bytes=weighted.max_peak_bytes)
    elif method in P2_EXTERNAL_METHODS:
        from fedcore.algorithm.low_rank.method_execution import plan_method
        plan_method(model, method_options.spec, rank=rank, target_paths=method_options.target_paths,
                    max_peak_bytes=method_options.max_peak_bytes)
        from .contracts import ContractError
        if (not isinstance(calibration, (tuple, list)) or len(calibration) != 2
                or not all(isinstance(value, torch.Tensor) for value in calibration)):
            raise ContractError('invalid_calibration', 'Separate tensor calibration features and targets required')
        features, targets = calibration
        if features.ndim < 2 or targets.ndim == 0 or len(features) != len(targets):
            raise ContractError('invalid_calibration', 'Calibration feature/target lengths must agree')
        if method in ('fwsvd','bolaco'):
            if targets.is_complex() or (targets.is_floating_point() and not bool(torch.isfinite(targets).all())):
                raise ContractError('invalid_calibration', 'Calibration targets must be finite')
            integral = method=='bolaco' or method_options.spec.loss=='cross_entropy'
            if integral and targets.dtype not in (torch.int32, torch.int64):
                raise ContractError('invalid_calibration', 'This method requires integral calibration targets')
            if method=='bolaco' and targets.ndim!=1:
                raise ContractError('invalid_calibration', 'One explicit group ID per calibration observation required')
    store = JobStore(jobs_root)
    runner = JobRunner(store, python_executable=python_executable)
    try:
        with tempfile.TemporaryDirectory(prefix="fedcore-input-") as source:
            root = Path(source)
            save_model_bundle(model, root / "model.fcb")
            safe_save(example.detach().cpu().clone(), root / "example.fcb")
            save_dataset(root / roles.validation, *validation)
            if train is not None:
                save_dataset(root / roles.train, *train)
            if calibration is not None:
                save_dataset(root / roles.calibration, *calibration)
            job_id = runner.submit(request, root)
        job = runner.wait(job_id, request.resources.timeout_seconds + 30)
        result = dict(job["result"] or {})
        result.setdefault("version", 1)
        result.setdefault("status", job["state"])
        if job["state"] in ("failed", "cancelled") and "error" not in result:
            result["error"] = {"code": job["state"], "message": "Compression job did not complete successfully"}
        result["job_id"] = job_id
        result["job_directory"] = str(store.directory(job_id))
        return result
    finally:
        runner.close()
