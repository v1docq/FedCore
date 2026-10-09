"""Allowlisted, data-only architecture descriptors with independent model factories."""
from __future__ import annotations

import math
from pathlib import Path
import torch
from torch import nn
from .contracts import ContractError
from .security import safe_load, safe_save

MODEL_KIND = "fedcore_safe_model"


def model_descriptor(model):
    from fedcore.algorithm.low_rank.topology import inspect_topology
    topology = inspect_topology(model)
    if topology.module_aliases or topology.parameter_aliases or topology.storage_aliases:
        raise ContractError('unsupported_topology', 'Portable model bundles do not support shared modules, Parameters or storage views', 'model')
    if type(model) is nn.Linear:
        config = {"in_features": model.in_features, "out_features": model.out_features, "bias": model.bias is not None}
    elif type(model) in (nn.Conv1d, nn.Conv2d):
        config = {key: getattr(model, key) for key in ("in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "padding_mode")}
        config["bias"] = model.bias is not None
    elif type(model) is nn.Sequential:
        return {"type": "Sequential", "layers": [model_descriptor(m) for m in model.children()]}
    elif type(model) is nn.ReLU:
        config = {"inplace": False}
    elif type(model) is nn.Flatten:
        config = {"start_dim": model.start_dim, "end_dim": model.end_dim}
    elif type(model) is nn.Identity:
        config = {}
    else:
        raise ContractError("unsupported_architecture", f"Factory is not allowlisted: {type(model).__name__}", "model")
    return {"type": type(model).__name__, "config": config,
            "dtype": str(next(model.parameters(), torch.empty(0)).dtype).removeprefix("torch.")}


def _allocation_bytes(descriptor, depth=0):
    if depth > 8 or not isinstance(descriptor, dict):
        raise ContractError("invalid_architecture", "Invalid nested architecture")
    if descriptor.get("type") == "Sequential":
        layers = descriptor.get("layers")
        if not isinstance(layers, (list, tuple)) or not 1 <= len(layers) <= 128:
            raise ContractError("invalid_architecture", "Invalid layer list")
        return sum(_allocation_bytes(layer, depth+1) for layer in layers)
    config = descriptor.get("config")
    if not isinstance(config, dict):
        raise ContractError("invalid_architecture", "Expected factory configuration")
    def dimension(key):
        value = config.get(key)
        if type(value) is not int or not 1 <= value <= 100000:
            raise ContractError("invalid_architecture", "Invalid positive model dimension")
        return value
    if descriptor.get("type") == "Linear":
        elements = dimension("out_features") * (dimension("in_features") + 1)
    elif descriptor.get("type") in ("Conv1d", "Conv2d"):
        kernel = config.get("kernel_size")
        kernel = kernel if isinstance(kernel, (tuple, list)) else (kernel,) * (1 if descriptor["type"] == "Conv1d" else 2)
        if not kernel or len(kernel) > 2 or any(type(k) is not int or not 1 <= k <= 100000 for k in kernel):
            raise ContractError("invalid_architecture", "Invalid positive kernel dimensions")
        elements = dimension("out_channels") * (dimension("in_channels") * math.prod(kernel) + 1)
    else:
        elements = 0
    return elements * (8 if descriptor.get("dtype") == "float64" else 4)


def build_model(descriptor, *, max_bytes=64 * 1024 * 1024, _depth=0):
    if _depth == 0 and _allocation_bytes(descriptor) > max_bytes:
        raise ContractError("size_limit", "Combined model allocation exceeds the job byte limit")
    if _depth > 8 or not isinstance(descriptor, dict):
        raise ContractError("invalid_architecture", "Invalid architecture descriptor")
    kind = descriptor.get("type")
    if kind == "Sequential":
        if set(descriptor) != {"type", "layers"} or not isinstance(descriptor["layers"], (list, tuple)) or not 1 <= len(descriptor["layers"]) <= 128:
            raise ContractError("invalid_architecture", "Sequential requires a bounded nonempty layer list")
        model = nn.Sequential(*(build_model(d, max_bytes=max_bytes, _depth=_depth+1) for d in descriptor["layers"]))
    else:
        keys = {"Linear": {"in_features", "out_features", "bias"},
                "Conv1d": {"in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "bias", "padding_mode"},
                "Conv2d": {"in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "bias", "padding_mode"},
                "ReLU": {"inplace"}, "Flatten": {"start_dim", "end_dim"}, "Identity": set()}
        if kind not in keys or set(descriptor) != {"type", "config", "dtype"}:
            raise ContractError("unsupported_architecture", "Unknown or malformed allowlisted factory")
        config = descriptor["config"]
        if not isinstance(config, dict) or set(config) != keys[kind] or descriptor["dtype"] not in ("float32", "float64"):
            raise ContractError("invalid_architecture", "Factory configuration fields or dtype are invalid")
        for key, value in config.items():
            if key in ("bias", "inplace"):
                if type(value) is not bool:
                    raise ContractError("invalid_architecture", f"{key} must be boolean")
            elif key == "padding_mode":
                if value not in ("zeros", "reflect", "replicate", "circular"):
                    raise ContractError("invalid_architecture", "Unknown padding mode")
            else:
                values = value if isinstance(value, (tuple, list)) else (value,)
                if len(values) > 2 or any(type(v) is not int or not -5 <= v <= 100000 for v in values):
                    raise ContractError("invalid_architecture", f"Invalid bounded dimensions: {key}")
        if kind == "Linear":
            estimate = config["in_features"] * config["out_features"] + config["out_features"]
        elif kind.startswith("Conv"):
            kernel = config["kernel_size"]
            kernel = kernel if isinstance(kernel, (tuple, list)) else (kernel,) * (1 if kind == "Conv1d" else 2)
            estimate = config["in_channels"] * config["out_channels"] * math.prod(kernel) + config["out_channels"]
        else:
            estimate = 0
        if estimate < 0 or estimate * (8 if descriptor["dtype"] == "float64" else 4) > max_bytes:
            raise ContractError("size_limit", "Architecture exceeds model allocation limit")
        factory = getattr(nn, kind)
        try:
            model = factory(**config)
            if kind in ("Linear", "Conv1d", "Conv2d"):
                model = model.to(dtype=getattr(torch, descriptor["dtype"]))
        except (ValueError, TypeError, ZeroDivisionError) as error:
            raise ContractError("invalid_architecture", "Invalid factory parameters") from error
    if sum(p.numel() * p.element_size() for p in model.parameters()) > max_bytes:
        raise ContractError("size_limit", "Combined architecture exceeds allocation limit")
    return model.eval()


def save_model_bundle(model, path):
    descriptor = model_descriptor(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert OrderedDict to a plain dict; only tensors and primitives enter the archive.
    safe_save({"kind": MODEL_KIND, "version": 1, "architecture": descriptor,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}, path)
    return path


def load_model_bundle(path, max_bytes=64 * 1024 * 1024):
    value = safe_load(path, max_bytes)
    if not isinstance(value, dict) or set(value) != {"kind", "version", "architecture", "state_dict"} or value["kind"] != MODEL_KIND or type(value["version"]) is not int or value["version"] != 1:
        raise ContractError("invalid_model_bundle", "Expected a versioned safe model bundle")
    state = value["state_dict"]
    if not isinstance(state, dict) or any(type(v) is not torch.Tensor for v in state.values()):
        raise ContractError("invalid_weights", "state_dict must contain tensors only")
    # Factory initialization is overwritten by validated weights; do not consume
    # the caller's random stream when an adapter reads its fitted model.
    with torch.random.fork_rng(devices=[]):
        model = build_model(value["architecture"], max_bytes=max_bytes)
    expected = model.state_dict()
    if state.keys() != expected.keys() or any(v.shape != expected[k].shape or v.dtype != expected[k].dtype for k, v in state.items()):
        raise ContractError("invalid_weights", "Weights do not match the allowlisted architecture")
    model.load_state_dict(state, strict=True)
    return model
