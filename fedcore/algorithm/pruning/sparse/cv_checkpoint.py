"""Exact allowlisted constructor for the bounded CIFAR CV profile."""
from __future__ import annotations

import json
import torch

from .cv_channels import CifarResNet18, validate_cv_model, validate_cv_state


def describe_cv(model):
    if type(model) is not CifarResNet18:
        return None
    validate_cv_model(model)
    validate_cv_state(model, model.state_dict())
    config = json.loads(json.dumps(model.representation_config(), allow_nan=False))
    return {"type": "CVArchitecture", "architecture": "CifarResNet18", "version": 1,
            "config": config, "dtype": str(model.conv1.weight.dtype)}


def build_cv(description):
    if (type(description) is not dict
            or set(description) != {"type", "architecture", "version", "config", "dtype"}
            or description["type"] != "CVArchitecture" or description["architecture"] != "CifarResNet18"
            or type(description["version"]) is not int or description["version"] != 1
            or description["dtype"] != "torch.float32"):
        raise ValueError("Exact version-1 CIFAR CV checkpoint descriptor required")
    with torch.random.fork_rng(devices=[]):
        model = CifarResNet18.from_config(description["config"], dtype=getattr(torch, description["dtype"][6:]))
    validate_cv_model(model, cpu=True)
    return model
