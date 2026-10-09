"""Validated operations over safe model bundles. No shared FedCore API instance."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
import torch
from torch import nn
from fedcore.external_runtime.client import compress
from fedcore.external_runtime.contracts import ContractError
from fedcore.external_runtime.models import load_model_bundle, model_descriptor
from fedcore.tools.export import export_model as export_artifact, normalize_framework

EXPORT_OPS = ["export_onnx", "export_tensorrt", "export_torchscript"]
VALID_KINDS = frozenset({"auto", "convolutional", "attention_embedding", "other"})
KIND_OPERATIONS = {kind: list(EXPORT_OPS) + ["low_rank"] for kind in VALID_KINDS}


@dataclass
class ModelCapabilities:
    kind: str
    suggested_kind: str
    has_conv: bool
    has_emb: bool
    has_attn: bool
    findings: list[str] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def detect_capabilities(model, kind="auto"):
    if kind not in VALID_KINDS:
        raise ValueError("Unknown model kind")
    conv = any(isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) for m in model.modules())
    emb = any(isinstance(m, nn.Embedding) for m in model.modules())
    attn = any(isinstance(m, nn.MultiheadAttention) for m in model.modules())
    suggested = "convolutional" if conv else "attention_embedding" if emb and attn else "other"
    selected = suggested if kind == "auto" else kind
    supported = any(type(m) in (nn.Linear, nn.Conv1d, nn.Conv2d) for m in model.modules())
    try:
        model_descriptor(model)
    except ContractError:
        supported = False
    ops = list(EXPORT_OPS) + (["low_rank"] if supported else [])
    return ModelCapabilities(selected, suggested, conv, emb, attn,
                             ["SVD is supported only for allowlisted tensor architectures; pruning and quantization are not enabled by this service"], ops)


def load_torch_module(model_path):
    return load_model_bundle(model_path)


def load_dataloader_from_bundle(loader_path):
    try:
        from .loader_bundle import LoaderBundle
    except ImportError:
        from loader_bundle import LoaderBundle
    return LoaderBundle.to_dataloader(LoaderBundle.load(loader_path))


def example_input_from_loader(loader, fallback_shape=None):
    if loader is None:
        if fallback_shape is None:
            raise ContractError("missing_input_spec", "Provide a loader or explicit input shape")
        from fedcore.external_runtime.contracts import InputSpec
        spec = InputSpec(tuple(fallback_shape))
        return torch.zeros(spec.shape)
    batch = next(iter(loader))
    value = batch[0] if isinstance(batch, (tuple, list)) else batch
    if not isinstance(value, torch.Tensor):
        raise ContractError("unsupported_input", "Only a single tensor feature input is supported")
    return value[:1].detach().cpu()


def export_via_fedcore(model, *, framework, export_dir, model_name="model", example_input=None, opset_version=17):
    backend = normalize_framework(framework)
    if not isinstance(model_name, str) or Path(model_name).name != model_name or model_name in ("", ".", "..") or "\\" in model_name:
        raise ContractError("invalid_path", "model_name must be a filename stem")
    path = export_artifact(model, backend, Path(export_dir) / model_name, example_input, {"opset_version": opset_version})
    return {"message": "Artifact exported and loader validated", "via": "fedcore.tools.export.export_model",
            "framework": backend, "file": str(path), "size_bytes": path.stat().st_size}


def _loader_tensors(loader):
    values = list(loader)
    if not values or any(not isinstance(b, (tuple, list)) or len(b) != 2 for b in values):
        raise ContractError("invalid_dataset", "Dataset requires features and targets")
    return torch.cat([b[0] for b in values]), torch.cat([b[1] for b in values])


def run_operation(operation, model_path, *, loader_path=None, validation_loader_path=None,
                  export_dir="results/exports", model_name="model", pruning_ratio=0.3,
                  kind="auto", task=None, example_input=None, rank=None,
                  retained_energy=1.0, max_relative_error=1e-4, profile=None):
    # Closed allowlist is enforced before every export alias and before loading.
    if operation not in EXPORT_OPS + ["low_rank"]:
        raise PermissionError(f"Operation {operation!r} is not enabled")
    model = load_torch_module(model_path)
    capabilities = detect_capabilities(model, kind)
    if operation not in capabilities.operations:
        raise PermissionError(f"Operation {operation!r} is unsupported for this architecture")
    loader = load_dataloader_from_bundle(loader_path) if loader_path else None
    example = example_input if example_input is not None else example_input_from_loader(loader)
    if operation in EXPORT_OPS:
        return export_via_fedcore(model, framework=operation.removeprefix("export_"), export_dir=export_dir,
                                 model_name=model_name, example_input=example)
    if task not in ("classification", "regression") or validation_loader_path is None:
        raise ContractError("missing_data_roles", "SVD requires an explicit task and separate validation data")
    if loader_path and Path(loader_path).resolve() == Path(validation_loader_path).resolve():
        raise ContractError("overlapping_data_roles", "Train and validation loaders must be separate")
    validation = _loader_tensors(load_dataloader_from_bundle(validation_loader_path))
    train = _loader_tensors(loader) if loader else None
    return compress(model, example, validation, jobs_root=export_dir, task=task, train=train,
                    rank=rank, retained_energy=retained_energy, max_relative_error=max_relative_error, profile=profile)


def output_path_for_operation(operation, *, export_dir, model_name):
    if operation not in EXPORT_OPS:
        return None
    return str(Path(export_dir) / (model_name + {"export_onnx": ".onnx", "export_tensorrt": ".engine", "export_torchscript": ".pt"}[operation]))
