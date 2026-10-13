"""Fixed sparse constructor allowlist for trusted local checkpoints.

Descriptions contain finite JSON values. Allocation is bounded before any
constructor runs; state coordinates are validated before assignment.
"""
from __future__ import annotations

import json
import math
import torch

KINDS = ("PackedNMLinear", "IndexedResidualLinear", "SparseMLP", "PartitionedMLP",
         "FlexiGPTTransformer", "SparseReferenceTransformer")
MAX_ALLOCATION_BYTES = 512 * 1024**2


def _class_for(name):
    if name in ("PackedNMLinear", "IndexedResidualLinear"):
        from . import layers
        return getattr(layers, name)
    if name in ("SparseMLP", "PartitionedMLP"):
        from . import channels
        return getattr(channels, name)
    if name in ("FlexiGPTTransformer", "SparseReferenceTransformer"):
        from . import architectures
        return getattr(architectures, name)
    raise ValueError("Unknown sparse checkpoint constructor")


def _known(model):
    name = type(model).__name__
    return name in KINDS and type(model) is _class_for(name)


def _guard(model):
    from fedcore.algorithm.low_rank.architecture_checkpoint import _runtime_guard
    _runtime_guard(model)


def _bound(config, max_bytes=MAX_ALLOCATION_BYTES):
    """Conservative bound on dimensions, nesting and total constructor storage."""
    serialized = json.dumps(config, allow_nan=False)
    if len(serialized) > 1024 * 1024:
        raise ValueError("Sparse configuration exceeds metadata limit")
    dimensions = []
    def visit(value, depth=0):
        if depth > 20:
            raise ValueError("Sparse configuration exceeds nesting limit")
        if isinstance(value, dict):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValueError("Configuration keys must be strings")
                if key in ("in_features", "out_features", "hidden_size", "intermediate_size",
                           "d_model", "d_ff", "width", "hidden_dim", "input_dim", "output_dim"):
                    if type(child) is not int or not 0 <= child <= 100000:
                        raise ValueError("Sparse dimensions must be bounded integers")
                    dimensions.append(child)
                visit(child, depth+1)
        elif isinstance(value, (list, tuple)):
            if len(value) > 10000:
                raise ValueError("Sparse configuration sequence exceeds limit")
            for child in value:
                visit(child, depth+1)
        elif value is not None and type(value) not in (str, bool, int, float):
            raise ValueError("Sparse configuration must contain JSON values")
    visit(config)
    # Each declared operator can need multiple factor/projection matrices.
    # Deliberately conservative; it is a safety limit, not an artifact estimate.
    if dimensions and sum(dimensions) * max(dimensions) * 32 > max_bytes:
        raise ValueError("Sparse constructor exceeds allocation bound")


def describe_sparse(model):
    if not _known(model):
        return None
    _guard(model)
    validate_sparse_state(model, model.state_dict())
    config = json.loads(json.dumps(type(model).representation_config(model), allow_nan=False))
    _bound(config)
    values = [v for v in model.state_dict().values() if v.is_floating_point()]
    dtype = str(values[0].dtype) if values else "torch.float32"
    if dtype not in ("torch.float32", "torch.float64"):
        raise ValueError("Sparse checkpoint dtype is unsupported")
    return {"type": "SparseRepresentation", "version": 1, "representation": type(model).__name__,
            "config": config, "dtype": dtype}


def build_sparse(description):
    if (not isinstance(description, dict) or set(description) !=
            {"type", "version", "representation", "config", "dtype"}
            or description["type"] != "SparseRepresentation"
            or type(description["version"]) is not int or description["version"] != 1
            or description["representation"] not in KINDS
            or not isinstance(description["config"], dict)
            or description["dtype"] not in ("torch.float32", "torch.float64")):
        raise ValueError("Invalid sparse representation descriptor")
    _bound(description["config"])
    cls = _class_for(description["representation"])
    with torch.random.fork_rng(devices=[]):
        result = cls.from_config(description["config"], device="cpu",
                                 dtype=getattr(torch, description["dtype"][6:]))
    if type(result) is not cls:
        raise ValueError("Sparse factory returned an unexpected type")
    return result


def validate_sparse_state(model, state, prefix=""):
    if not _known(model):
        return
    _guard(model)
    expected = model.state_dict()
    local = {name: state[prefix+name] for name in expected if prefix+name in state}
    if local.keys() != expected.keys() or any(
            not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
            or value.dtype != expected[name].dtype
            or (value.is_floating_point() and not bool(torch.isfinite(value).all()))
            for name, value in local.items()):
        raise ValueError("Sparse state has invalid shapes, dtypes or nonfinite values")
    cls = type(model)
    if cls.__name__ in ("PackedNMLinear", "IndexedResidualLinear"):
        cls.validate_checkpoint_state(cls.representation_config(model), local)
    else:
        cls.validate_state(model, state, prefix=prefix)
