"""Export PyTorch models to deployment formats.

Currently ONNX is the only implemented target. Any other requested framework
is silently mapped to ONNX.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import torch
from torch import nn

PathLike = Union[str, Path]


def normalize_framework(framework: str) -> str:
    """Map a requested framework name to an implemented export backend.

    Only ``onnx`` is implemented; every other value falls back to ``onnx``
    without raising or emitting a warning.
    """
    name = (framework or "onnx").strip().lower()
    if name == "onnx":
        return "onnx"
    return "onnx"


def _ensure_onnx_path(output_path: PathLike) -> Path:
    path = Path(output_path)
    if path.suffix.lower() != ".onnx":
        path = path.with_suffix(".onnx")
    return path


def export_to_onnx(
    model: nn.Module,
    output_path: PathLike,
    example_input: torch.Tensor,
    *,
    opset_version: int = 17,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    do_constant_folding: bool = True,
) -> Path:
    """Export a PyTorch module to an ONNX file via ``torch.onnx.export``."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be torch.nn.Module, got {type(model)!r}")
    if example_input is None:
        raise ValueError("example_input is required for ONNX export")

    path = _ensure_onnx_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    input_names = list(input_names) if input_names is not None else ["input"]
    output_names = list(output_names) if output_names is not None else ["output"]
    if dynamic_axes is None:
        dynamic_axes = {
            input_names[0]: {0: "batch_size"},
            output_names[0]: {0: "batch_size"},
        }

    model = model.eval()
    export_kwargs: Dict[str, Any] = dict(
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    # dynamo=False is needed on newer PyTorch; older versions reject it.
    try:
        torch.onnx.export(
            model, example_input, str(path), dynamo=False, **export_kwargs
        )
    except TypeError:
        torch.onnx.export(model, example_input, str(path), **export_kwargs)

    return path


def export_model(
    model: nn.Module,
    framework: str,
    output_path: PathLike,
    example_input: torch.Tensor,
    framework_config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export ``model`` for ``framework``, falling back to ONNX when needed."""
    framework_config = dict(framework_config or {})
    normalize_framework(framework)

    opset_version = int(framework_config.get("opset_version", 17))
    input_names = framework_config.get("input_names")
    output_names = framework_config.get("output_names")
    dynamic_axes = framework_config.get("dynamic_axes")
    do_constant_folding = bool(
        framework_config.get("do_constant_folding", True)
    )

    return export_to_onnx(
        model,
        output_path,
        example_input,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=do_constant_folding,
    )
