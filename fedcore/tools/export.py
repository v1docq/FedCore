"""Export PyTorch models to deployment formats.

Supported backends
------------------
* ``torchscript`` (aliases: ``pt``, ``pytorch``) → ``.pt``
* ``onnx`` → ``.onnx``
* ``tensorrt`` (aliases: ``engine``, ``trt``) → ``.engine``

Any other framework name is silently mapped to ONNX.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import torch
from torch import nn

PathLike = Union[str, Path]

_FRAMEWORK_SUFFIX = {
    "torchscript": ".pt",
    "onnx": ".onnx",
    "tensorrt": ".engine",
}

_DEFAULT_OUTPUT_STEM = "converted-model"


def normalize_framework(framework: str) -> str:
    """Map a requested framework name to an implemented export backend.

    Unknown names fall back to ``onnx`` without raising or emitting a warning.
    """
    name = (framework or "onnx").strip().lower()
    if name in {"torchscript", "pt", "pytorch"}:
        return "torchscript"
    if name in {"tensorrt", "engine", "trt"}:
        return "tensorrt"
    if name == "onnx":
        return "onnx"
    return "onnx"


def default_output_path(framework: str) -> Path:
    """Default artifact path for a normalized (or raw) framework name."""
    backend = normalize_framework(framework)
    return Path(f"{_DEFAULT_OUTPUT_STEM}{_FRAMEWORK_SUFFIX[backend]}")


def _default_example_input() -> torch.Tensor:
    return torch.randn(1, 3, 224, 224)


def _ensure_suffix(output_path: PathLike, suffix: str) -> Path:
    path = Path(output_path)
    if path.suffix.lower() != suffix.lower():
        path = path.with_suffix(suffix)
    return path


def export_to_torchscript(
    model: nn.Module,
    output_path: PathLike,
    example_input: torch.Tensor,
) -> Path:
    """Export via TorchScript: script → trace → ``torch.save(model)``."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be torch.nn.Module, got {type(model)!r}")

    path = _ensure_suffix(output_path, ".pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval()

    try:
        scripted = torch.jit.script(model)
        scripted.save(str(path))
        return path
    except Exception:
        pass

    try:
        traced = torch.jit.trace(model, example_input)
        traced.save(str(path))
        return path
    except Exception:
        pass

    torch.save(model, str(path))
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

    path = _ensure_suffix(output_path, ".onnx")
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


def export_to_tensorrt(
    model: nn.Module,
    output_path: PathLike,
    example_input: torch.Tensor,
    framework_config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export via ONNX intermediate to a TensorRT ``.engine`` file.

    Raises
    ------
    ImportError
        If the ``tensorrt`` package is not installed.
    RuntimeError
        If engine build fails.
    """
    try:
        import tensorrt as trt
    except ImportError as error:
        raise ImportError(
            "TensorRT export requires the 'tensorrt' package "
            "(and typically a CUDA GPU). Install NVIDIA TensorRT / "
            "pip install nvidia-tensorrt."
        ) from error

    framework_config = dict(framework_config or {})
    path = _ensure_suffix(output_path, ".engine")
    path.parent.mkdir(parents=True, exist_ok=True)

    onnx_path = path.with_suffix(".onnx")
    export_to_onnx(
        model,
        onnx_path,
        example_input,
        opset_version=int(framework_config.get("opset_version", 17)),
        input_names=framework_config.get("input_names"),
        output_names=framework_config.get("output_names"),
        dynamic_axes=framework_config.get("dynamic_axes"),
        do_constant_folding=bool(
            framework_config.get("do_constant_folding", True)
        ),
    )

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            errors = [
                parser.get_error(i).desc() for i in range(parser.num_errors)
            ]
            raise RuntimeError(
                "TensorRT ONNX parse failed: " + "; ".join(errors)
            )

    config = builder.create_builder_config()
    workspace_bytes = int(framework_config.get("workspace_size", 1 << 30))
    # Newer TensorRT uses set_memory_pool_limit; older uses max_workspace_size.
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    except AttributeError:
        config.max_workspace_size = workspace_bytes

    try:
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT build_serialized_network returned None")
        engine_bytes = bytes(serialized)
    except AttributeError:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("TensorRT build_engine returned None")
        engine_bytes = bytes(engine.serialize())

    with open(path, "wb") as engine_file:
        engine_file.write(engine_bytes)

    return path


def export_model(
    model: nn.Module,
    framework: str,
    output_path: PathLike,
    example_input: Optional[torch.Tensor] = None,
    framework_config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export ``model`` for ``framework`` (unknown names → ONNX)."""
    framework_config = dict(framework_config or {})
    backend = normalize_framework(framework)

    if example_input is None:
        example_input = _default_example_input()

    if backend == "torchscript":
        return export_to_torchscript(model, output_path, example_input)

    if backend == "tensorrt":
        return export_to_tensorrt(
            model, output_path, example_input, framework_config
        )

    return export_to_onnx(
        model,
        output_path,
        example_input,
        opset_version=int(framework_config.get("opset_version", 17)),
        input_names=framework_config.get("input_names"),
        output_names=framework_config.get("output_names"),
        dynamic_axes=framework_config.get("dynamic_axes"),
        do_constant_folding=bool(
            framework_config.get("do_constant_folding", True)
        ),
    )
