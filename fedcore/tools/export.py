"""Strict deployment exports: success means the declared loader accepted the artifact."""
from __future__ import annotations

import copy
import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import Union

import torch
from torch import nn

PathLike = Union[str, Path]
_FRAMEWORK_SUFFIX = {"torchscript": ".pt", "onnx": ".onnx", "tensorrt": ".engine"}


class ExportError(RuntimeError):
    """Expected export failure with a stable machine-readable code."""
    def __init__(self, code, backend, message, causes=()):
        super().__init__(message)
        self.code, self.backend, self.causes = code, backend, tuple(causes)

    def to_dict(self):
        return {"code": self.code, "backend": self.backend,
                "message": str(self), "causes": list(self.causes)}


def normalize_framework(framework: str) -> str:
    if not isinstance(framework, str):
        raise ExportError("unsupported_backend", str(framework), "An explicit backend is required")
    aliases = {"pt": "torchscript", "pytorch": "torchscript", "trt": "tensorrt", "engine": "tensorrt"}
    name = aliases.get(framework.strip().lower(), framework.strip().lower())
    if name not in _FRAMEWORK_SUFFIX:
        raise ExportError("unsupported_backend", name, f"Unsupported export backend: {framework!r}")
    return name


def default_output_path(framework: str) -> Path:
    return Path("converted-model" + _FRAMEWORK_SUFFIX[normalize_framework(framework)])


def input_spec(example_input: torch.Tensor) -> dict:
    if not isinstance(example_input, torch.Tensor):
        raise ExportError("unsupported_input", "", "Export requires one explicit tensor input; containers are unsupported")
    if not example_input.numel() or example_input.ndim == 0:
        raise ExportError("invalid_input", "", "Example input must be a nonempty tensor with a batch axis")
    return {"version": 1, "inputs": [{"name": "input", "shape": list(example_input.shape),
                                    "dtype": str(example_input.dtype).removeprefix("torch.")}],
            "layout": "single_tensor", "dynamic_batch": False}


def _same_output(actual, expected):
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape:
            raise ValueError("Exported output shape or type differs")
        torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), rtol=1e-4, atol=1e-5)
    elif isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)) or len(actual) != len(expected):
            raise ValueError("Exported output structure differs")
        for left, right in zip(actual, expected):
            _same_output(left, right)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError("Exported output keys differ")
        for key in expected:
            _same_output(actual[key], expected[key])
    else:
        raise ValueError("Only tensor outputs and tensor containers are supported")


def _prepare(model, example_input, backend):
    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module")
    spec = input_spec(example_input)
    local = copy.deepcopy(model).eval()
    with torch.inference_mode():
        try:
            expected = local(example_input)
        except Exception as error:
            raise ExportError("invalid_input", backend, "Model rejected the declared InputSpec", (str(error),)) from error
    return local, expected, spec


def _temporary_path(output_path, backend):
    path = Path(output_path).with_suffix(_FRAMEWORK_SUFFIX[backend])
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    return path, Path(temporary)


def _publish(temporary, path, spec, backend):
    if not temporary.is_file() or not temporary.stat().st_size:
        raise ExportError("invalid_artifact", backend, "Exporter produced an empty artifact")
    meta = path.with_suffix(path.suffix + ".json")
    meta_temporary = temporary.with_suffix(temporary.suffix + ".json")
    meta_temporary.write_text(json.dumps({"format": backend, "input_spec": spec, "torch_version": torch.__version__}, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    os.replace(meta_temporary, meta)
    return path


def export_to_torchscript(model: nn.Module, output_path: PathLike, example_input: torch.Tensor) -> Path:
    input_spec(example_input)
    example_input = example_input.detach().clone()
    model, expected, spec = _prepare(model, example_input, "torchscript")
    path, temporary = _temporary_path(output_path, "torchscript")
    failures = []
    try:
        compilers = (lambda: torch.jit.script(model), lambda: torch.jit.trace(model, example_input, check_trace=True))
        for compile_model in compilers:
            try:
                compiled = compile_model()
                compiled.save(str(temporary))
                loaded = torch.jit.load(str(temporary), map_location=example_input.device).eval()
                with torch.inference_mode():
                    _same_output(loaded(example_input), expected)
                return _publish(temporary, path, spec, "torchscript")
            except Exception as error:
                failures.append(f"{type(error).__name__}: {error}")
        raise ExportError("compilation_failed", "torchscript", "Both TorchScript scripting and tracing failed", failures)
    finally:
        temporary.unlink(missing_ok=True)


def export_to_onnx(model, output_path, example_input, *, opset_version=17,
                   input_names=None, output_names=None, dynamic_axes=None, do_constant_folding=True):
    input_spec(example_input)
    example_input = example_input.detach().clone()
    model, expected, spec = _prepare(model, example_input, "onnx")
    path, temporary = _temporary_path(output_path, "onnx")
    try:
        import onnx
        import onnxruntime as ort
        names = list(input_names or ["input"])
        if len(names) != 1:
            raise ValueError("Single tensor input requires exactly one input name")
        spec["inputs"][0]["name"] = names[0]
        spec["dynamic_batch"] = bool(dynamic_axes and 0 in dynamic_axes.get(names[0], {}))
        kwargs = dict(export_params=True, opset_version=opset_version,
                      do_constant_folding=do_constant_folding, input_names=names,
                      output_names=list(output_names or ["output"]), dynamic_axes=dynamic_axes)
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            kwargs["dynamo"] = False
        torch.onnx.export(model, example_input, str(temporary), **kwargs)
        onnx.checker.check_model(onnx.load(str(temporary)))
        session = ort.InferenceSession(str(temporary), providers=["CPUExecutionProvider"])
        outputs = session.run(None, {names[0]: example_input.detach().cpu().numpy()})
        if not isinstance(expected, torch.Tensor) or len(outputs) != 1:
            raise ValueError("ONNX v1 supports a single tensor output")
        _same_output(torch.from_numpy(outputs[0]), expected)
        return _publish(temporary, path, spec, "onnx")
    except ExportError:
        raise
    except Exception as error:
        raise ExportError("export_failed", "onnx", "ONNX export or loader validation failed", (str(error),)) from error
    finally:
        temporary.unlink(missing_ok=True)


def export_to_tensorrt(model, output_path, example_input, framework_config=None):
    input_spec(example_input)
    example_input = example_input.detach().clone()
    _, _, spec = _prepare(model, example_input, "tensorrt")
    path, temporary = _temporary_path(output_path, "tensorrt")
    intermediate = temporary.with_suffix(".onnx")
    try:
        import tensorrt as trt
        options = dict(framework_config or {})
        export_to_onnx(model, intermediate, example_input, opset_version=int(options.get("opset_version", 17)))
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(intermediate.read_bytes()):
            raise RuntimeError("; ".join(parser.get_error(i).desc() for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        workspace = int(options.get("workspace_size", 1 << 30))
        if workspace <= 0:
            raise ValueError("workspace_size must be positive")
        if hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
        else:
            config.max_workspace_size = workspace
        if hasattr(builder, "build_serialized_network"):
            serialized = builder.build_serialized_network(network, config)
        else:
            engine = builder.build_engine(network, config)
            serialized = engine.serialize() if engine is not None else None
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed")
        temporary.write_bytes(bytes(serialized))
        runtime = trt.Runtime(logger)
        if runtime.deserialize_cuda_engine(temporary.read_bytes()) is None:
            raise RuntimeError("TensorRT loader rejected the engine")
        return _publish(temporary, path, spec, "tensorrt")
    except Exception as error:
        raise ExportError("export_failed", "tensorrt", "TensorRT export or loader validation failed", (str(error),)) from error
    finally:
        temporary.unlink(missing_ok=True)
        intermediate.unlink(missing_ok=True)
        intermediate.with_suffix(".onnx.json").unlink(missing_ok=True)


def export_model(model, framework, output_path, example_input=None, framework_config=None):
    backend = normalize_framework(framework)
    if example_input is None:
        raise ExportError("missing_input_spec", backend, "Provide an example input matching the actual model")
    options = dict(framework_config or {})
    if backend == "torchscript":
        return export_to_torchscript(model, output_path, example_input)
    if backend == "tensorrt":
        return export_to_tensorrt(model, output_path, example_input, options)
    return export_to_onnx(model, output_path, example_input,
                          opset_version=int(options.get("opset_version", 17)),
                          input_names=options.get("input_names"), output_names=options.get("output_names"),
                          dynamic_axes=options.get("dynamic_axes"),
                          do_constant_folding=bool(options.get("do_constant_folding", True)))
