"""Allowlisted P5 constructor and semantic-state checkpoint boundary."""
from __future__ import annotations

import json
import torch

_KINDS = ("SliceGPTModel", "LightFormerTransformer", "LRTTransformer")


def _class_for(name):
    if name == "SliceGPTModel":
        from .slicegpt_layers import SliceGPTModel
        return SliceGPTModel
    if name == "LightFormerTransformer":
        from .lightformer_layers import LightFormerTransformer
        return LightFormerTransformer
    if name == "LRTTransformer":
        from .lrt_layers import LRTTransformer
        return LRTTransformer
    raise ValueError("Unsupported P5 architecture")


def _known(model):
    name = type(model).__name__
    return name in _KINDS and type(model) is _class_for(name)


def _runtime_guard(model):
    # Do not invoke instance-overridable traversal before checking overrides.
    # Whole LightFormer blocks intentionally occur under multiple paths.
    pending = [(model, frozenset())]
    seen = set()
    while pending:
        child, ancestors = pending.pop()
        if not isinstance(child, torch.nn.Module) or id(child) in ancestors:
            raise ValueError("P5 checkpoints require an acyclic declared module graph")
        if id(child) in seen:
            continue
        seen.add(id(child))
        if any(callable(value) for value in vars(child).values()) or any(
                bool(getattr(child, name, {})) for name in (
                    "_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks",
                    "_state_dict_hooks", "_state_dict_pre_hooks", "_load_state_dict_pre_hooks",
                    "_load_state_dict_post_hooks")):
            raise ValueError("P5 checkpoints require the declared graph without hooks or instance overrides")
        pending.extend((value, ancestors | {id(child)})
                       for value in vars(child).get('_modules', {}).values() if value is not None)


def describe_p5(model):
    if not _known(model):
        return None
    _runtime_guard(model)
    validate_p5_state(model, model.state_dict())
    config = json.loads(json.dumps(model.representation_config(), allow_nan=False))
    dtype = str(next(model.parameters()).dtype)
    if dtype not in ("torch.float32", "torch.float64"):
        raise ValueError("P5 checkpoints require float32 or float64")
    return {"type": "P5Architecture", "architecture": type(model).__name__,
            "version": 1, "config": config, "dtype": dtype}


def build_p5(description):
    if (not isinstance(description, dict)
            or set(description) != {"type", "architecture", "version", "config", "dtype"}
            or description["type"] != "P5Architecture" or type(description["version"]) is not int
            or description["version"] != 1 or description["architecture"] not in _KINDS
            or not isinstance(description["config"], dict)
            or description["dtype"] not in ("torch.float32", "torch.float64")):
        raise ValueError("Invalid P5 architecture descriptor")
    cls = _class_for(description["architecture"])
    dtype = getattr(torch, description["dtype"][6:])
    with torch.random.fork_rng(devices=[]):
        model = cls.from_config(description["config"], device="cpu", dtype=dtype)
    if type(model) is not cls:
        raise ValueError("P5 constructor returned a different architecture")
    return model


def validate_p5_state(model, state, prefix=""):
    if not _known(model):
        return
    _runtime_guard(model)
    expected = model.state_dict()
    for name, value in expected.items():
        actual = state.get(prefix + name)
        if (not isinstance(actual, torch.Tensor) or actual.shape != value.shape
                or actual.dtype != value.dtype
                or (actual.is_floating_point() and not bool(torch.isfinite(actual).all()))):
            raise ValueError("P5 state requires finite tensors with the declared shapes and dtypes")
    model.validate_checkpoint_state(state, prefix=prefix)
