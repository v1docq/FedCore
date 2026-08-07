"""FedCore-backed operations for the GUI server.

Export / compression go through FedCore packages
(``FedCore.export``, ``fedcore.algorithm.*``).
"""

from __future__ import annotations

import copy
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fedcore.tools.export import (
    export_model as export_artifact,
    normalize_framework,
)


logger = logging.getLogger(__name__)

EXPORT_OPS = ["export_onnx", "export_tensorrt", "export_torchscript"]

KIND_OPERATIONS: Dict[str, List[str]] = {
    "convolutional": EXPORT_OPS + ["quantize", "prune"],
    "attention_embedding": EXPORT_OPS + ["quantize", "prune", "low_rank"],
    "other": EXPORT_OPS + ["quantize", "prune"],
}

VALID_KINDS = frozenset({"auto", "convolutional", "attention_embedding", "other"})

_fedcore_instance = None


@dataclass
class ModelCapabilities:
    kind: str
    suggested_kind: str
    has_conv: bool
    has_emb: bool
    has_attn: bool
    findings: List[str] = field(default_factory=list)
    operations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _scan_modules(model: nn.Module) -> tuple[bool, bool, bool]:
    has_conv = any(
        isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) for m in model.modules()
    )
    has_emb = any(
        isinstance(m, (nn.Embedding, nn.EmbeddingBag)) for m in model.modules()
    )
    has_attn = any(
        isinstance(m, nn.MultiheadAttention)
        or "Attention" in m.__class__.__name__
        for m in model.modules()
    )
    return has_conv, has_emb, has_attn


def suggest_kind(has_conv: bool, has_emb: bool, has_attn: bool) -> str:
    if has_emb and has_attn:
        return "attention_embedding"
    if has_conv:
        return "convolutional"
    return "other"


def build_findings(has_conv: bool, has_emb: bool, has_attn: bool) -> List[str]:
    findings: List[str] = []
    if has_conv:
        findings.append("Найдены свёртки (Conv1d/2d/3d)")
    if has_emb:
        findings.append("Найдены embeddings (Embedding / EmbeddingBag)")
    if has_attn:
        findings.append("Найден механизм внимания (Attention)")
    if not findings:
        findings.append("Типовые блоки (conv / emb / attention) не обнаружены")
    return findings


def detect_capabilities(
    model: nn.Module, kind: Optional[str] = "auto"
) -> ModelCapabilities:
    """Detect module types and resolve ops for the selected kind.

    Detection is informational; the active kind (auto or manual) selects ops:
    * convolutional → quantize, prune (+ export)
    * attention_embedding → quantize, prune, low_rank (+ export)
    """
    has_conv, has_emb, has_attn = _scan_modules(model)
    suggested = suggest_kind(has_conv, has_emb, has_attn)
    findings = build_findings(has_conv, has_emb, has_attn)

    raw = (kind or "auto").strip().lower()
    if raw not in VALID_KINDS:
        raise ValueError(f"Unknown kind '{kind}'. Expected one of {sorted(VALID_KINDS)}")

    selected = suggested if raw == "auto" else raw
    ops = list(KIND_OPERATIONS.get(selected, KIND_OPERATIONS["other"]))

    return ModelCapabilities(
        kind=selected,
        suggested_kind=suggested,
        has_conv=has_conv,
        has_emb=has_emb,
        has_attn=has_attn,
        findings=findings,
        operations=ops,
    )


def load_torch_module(model_path: str | Path) -> nn.Module:
    path = Path(model_path)
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, nn.Module):
        return obj.eval()
    if isinstance(obj, dict) and isinstance(obj.get("model"), nn.Module):
        return obj["model"].eval()
    raise TypeError(
        f"Expected full nn.Module checkpoint at {path}, got {type(obj).__name__}"
    )


def load_dataloader_from_bundle(loader_path: str | Path) -> DataLoader:
    from loader_bundle import LoaderBundle

    bundle = LoaderBundle.load(loader_path)
    return LoaderBundle.to_dataloader(bundle)


def example_input_from_loader(
    loader: Optional[DataLoader], fallback_shape=(1, 3, 224, 224)
) -> torch.Tensor:
    if loader is not None:
        batch = next(iter(loader))
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        if isinstance(x, torch.Tensor):
            return x[:1].detach().cpu()
    return torch.randn(*fallback_shape)


def _infer_num_classes(loader: DataLoader) -> int:
    try:
        batch = next(iter(loader))
        y = batch[1] if isinstance(batch, (list, tuple)) and len(batch) >= 2 else None
        if not isinstance(y, torch.Tensor) or y.numel() == 0:
            return 10
        if y.ndim > 1 and y.shape[-1] > 1:
            return int(y.shape[-1])
        return int(y.max().item()) + 1
    except Exception:
        return 10


def _compression_data(
    model: nn.Module,
    loader: DataLoader,
    *,
    num_classes: Optional[int] = None,
):
    from fedcore.data.data import CompressionInputData
    from fedot.core.repository.tasks import Task, TaskTypesEnum

    return CompressionInputData(
        features=np.zeros((1, 1), dtype=np.float32),
        target=model,
        model=model,
        train_dataloader=loader,
        val_dataloader=loader,
        task=Task(TaskTypesEnum.classification),
        input_dim=3,
        num_classes=num_classes if num_classes is not None else _infer_num_classes(loader),
    )


def _get_fedcore():
    """Build / cache a FedCore instance for ``FedCore.export``."""
    global _fedcore_instance
    if _fedcore_instance is not None:
        return _fedcore_instance

    try:
        from fedcore.api.main import FedCore
        from fedcore.api.api_configs import (
            APIConfigTemplate,
            AutoMLConfigTemplate,
            FedotConfigTemplate,
            LearningConfigTemplate,
        )
        from fedcore.api.config_factory import ConfigFactory
    except Exception as exc:
        # Clear a half-imported module so the next attempt can succeed after deps install
        for key in list(sys.modules):
            if key == "fedcore.api.main" or key.startswith("fedcore.api.main."):
                del sys.modules[key]
        raise RuntimeError(
            "Cannot import FedCore API for FedCore.export. "
            "Install: evaluate, dask, distributed, bokeh, pynvml. "
            f"Original error: {exc}"
        ) from exc

    template = APIConfigTemplate(
        automl_config=AutoMLConfigTemplate(
            fedot_config=FedotConfigTemplate(
                problem="classification",
                metric=["accuracy"],
                pop_size=1,
                timeout=1,
                initial_assumption="ResNet18",
            )
        ),
        learning_config=LearningConfigTemplate(
            criterion="cross_entropy",
            learning_strategy="from_scratch",
        ),
    )
    _fedcore_instance = FedCore(ConfigFactory.from_template(template)())
    return _fedcore_instance


def export_via_fedcore(
    model: nn.Module,
    *,
    framework: str,
    export_dir: str | Path,
    model_name: str = "model",
    example_input: Optional[torch.Tensor] = None,
    opset_version: int = 17,
) -> Dict[str, Any]:
    """Export via public ``FedCore.export`` API."""
    backend = normalize_framework(framework)
    suffix = {"torchscript": ".pt", "onnx": ".onnx", "tensorrt": ".engine"}[backend]
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    output_path = export_dir / f"{model_name}{suffix}"

    if example_input is None:
        example_input = example_input_from_loader(None)

    path = export_artifact(
        model,
        backend,
        output_path,
        example_input,
        {
            "opset_version": opset_version,
        },
    )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FedCore.export reported path but file missing: {path}")

    return {
        "message": f"Exported via FedCore.export to {backend}",
        "via": "fedcore.tools.export.export_model",
        "framework": backend,
        "file": str(path).replace("\\", "/"),
        "size_bytes": path.stat().st_size,
    }


class _RunnableQuantizer:
    def __init__(self, params: dict):
        from fedcore.algorithm.quantization.quantizers import BaseQuantizer
        from fedcore.algorithm.quantization.utils import QDQWrapper, QDQWrapping

        self._QDQWrapper = QDQWrapper
        self._QDQWrapping = QDQWrapping

        class _Q(BaseQuantizer):
            @property
            def model_before(self):
                return getattr(self, "_mb", None)

            @model_before.setter
            def model_before(self, model):
                self._mb = model

            @property
            def model_after(self):
                return getattr(self, "_ma", None)

            @model_after.setter
            def model_after(self, model):
                self._ma = model

            def _init_hooks(self, input_data):
                self._on_epoch_start, self._on_epoch_end = [], []

            def _init_trainer_model_before_model_after_and_incapsulate_hooks(
                self, input_data
            ):
                self._init_model(input_data)

        self.quantizer = _Q(params)
        self.quantizer.logger.setLevel(logging.WARNING)

    def fit(
        self,
        model: nn.Module,
        loader: DataLoader,
        allow_conv: bool,
        allow_emb: bool,
    ):
        qconfig = self.quantizer.qconfig[""]
        model = copy.deepcopy(model).cpu().eval()
        types: List[type] = []
        if allow_conv:
            types.extend([nn.Conv2d, nn.BatchNorm2d, nn.Linear])
        if allow_emb:
            types.extend([nn.Embedding, nn.EmbeddingBag, nn.Linear])
        if not types:
            types = [nn.Linear]

        for name, layer in list(model.named_modules()):
            if not name or not isinstance(layer, tuple(types)):
                continue
            layer.qconfig = qconfig
            self._QDQWrapper.set_module(
                model, name, self._QDQWrapping(layer, mode="both", qconfig=qconfig)
            )

        data = _compression_data(model, loader)
        return self.quantizer.fit(data).cpu().eval()


def quantize_via_fedcore(
    model: nn.Module,
    loader: DataLoader,
    *,
    allow_conv: bool,
    allow_emb: bool,
    output_path: str | Path,
) -> Dict[str, Any]:
    engine = (
        "fbgemm"
        if "fbgemm" in torch.backends.quantized.supported_engines
        else "x86"
    )
    runner = _RunnableQuantizer(
        {
            "quant_type": "static",
            "backend": engine,
            "device": torch.device("cpu"),
            "dtype": torch.qint8,
            "allow_conv": allow_conv,
            "allow_emb": allow_emb,
        }
    )
    qmodel = runner.fit(model, loader, allow_conv=allow_conv, allow_emb=allow_emb)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(qmodel, output_path)
    return {
        "message": "Static quantization via FedCore BaseQuantizer",
        "operation": "quantize",
        "file": str(output_path).replace("\\", "/"),
        "size_bytes": output_path.stat().st_size,
    }


def prune_via_fedcore(
    model: nn.Module,
    loader: DataLoader,
    *,
    pruning_ratio: float = 0.3,
    output_path: str | Path,
) -> Dict[str, Any]:
    from fedcore.algorithm.pruning.pruners import BasePruner
    from fedcore.algorithm.pruning.pruning_validation import PruningValidator

    # Magnitude / zero-shot pruning: criterion is still required by the trainer
    # loop; prune_each must be present in params for the pruning hook to attach.
    agent = BasePruner(
        {
            "importance": "magnitude",
            "pruning_ratio": pruning_ratio,
            "pruning_iterations": 1,
            "prune_each": 1,
            "epochs": 1,
            "criterion": "cross_entropy",
            "optimizer": "adam",
            "device": torch.device("cpu"),
        }
    )
    data = _compression_data(copy.deepcopy(model).cpu(), loader)
    try:
        # Prefer direct structured prune (avoids fragile trainer-hook path).
        agent._init_model_before_model_after(data)
        agent.pruner = agent._init_pruner_with_model_after(data)
        if agent.pruner is None:
            raise RuntimeError("FedCore pruning agent was not initialized")
        for _ in range(agent.pruning_iterations):
            groups = agent.pruner.step(interactive=True)
            for group in groups:
                group.prune()
        PruningValidator.validate_pruned_layers(agent.model_after)
        result = agent.model_after
    except Exception as exc:
        logger.warning("Direct prune failed (%s); falling back to BasePruner.fit", exc)
        try:
            result = agent.fit(data)
        except Exception as fit_exc:
            result = getattr(agent, "model_after", None)
            if result is None:
                raise RuntimeError(
                    f"FedCore pruning failed: {fit_exc}"
                ) from fit_exc

    if not isinstance(result, nn.Module):
        result = getattr(agent, "model_after", result)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.cpu().eval(), output_path)
    return {
        "message": "Pruning via FedCore BasePruner",
        "operation": "prune",
        "file": str(output_path).replace("\\", "/"),
        "size_bytes": output_path.stat().st_size,
    }


def low_rank_via_fedcore(
    model: nn.Module,
    loader: DataLoader,
    *,
    output_path: str | Path,
    decomposing_mode: str = "channel",
) -> Dict[str, Any]:
    from fedcore.algorithm.low_rank.low_rank_opt import LowRankModel

    lr = LowRankModel(
        {
            "decomposer": "svd",
            "decomposing_mode": decomposing_mode,
            "device": torch.device("cpu"),
            "criterion": "cross_entropy",
            "optimizer": "adam",
            "epochs": 1,
        }
    )
    data = _compression_data(copy.deepcopy(model).cpu(), loader)
    try:
        result = lr.fit(data)
    except Exception as exc:
        result = getattr(lr, "model_after", None)
        if result is None:
            raise RuntimeError(f"FedCore low_rank failed: {exc}") from exc
    if not isinstance(result, nn.Module):
        result = getattr(lr, "model_after", result)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.cpu().eval(), output_path)
    return {
        "message": "Low-rank via FedCore LowRankModel",
        "operation": "low_rank",
        "file": str(output_path).replace("\\", "/"),
        "size_bytes": output_path.stat().st_size,
    }


def output_path_for_operation(
    operation: str,
    *,
    export_dir: str | Path,
    model_name: str,
) -> Optional[str]:
    out_dir = Path(export_dir)
    mapping = {
        "quantize": out_dir / f"{model_name}_int8.pt",
        "prune": out_dir / f"{model_name}_pruned.pt",
        "low_rank": out_dir / f"{model_name}_lowrank.pt",
        "export_onnx": out_dir / f"{model_name}.onnx",
        "export_tensorrt": out_dir / f"{model_name}.engine",
        "export_torchscript": out_dir / f"{model_name}.pt",
    }
    path = mapping.get(operation)
    return str(path).replace("\\", "/") if path else None


def run_operation(
    operation: str,
    model_path: str | Path,
    *,
    loader_path: Optional[str | Path] = None,
    export_dir: str | Path = "results/exports",
    model_name: str = "model",
    pruning_ratio: float = 0.3,
    kind: str = "auto",
) -> Dict[str, Any]:
    model = load_torch_module(model_path)
    caps = detect_capabilities(model, kind=kind)

    if operation.startswith("export_"):
        fw = operation.replace("export_", "", 1)
        loader = load_dataloader_from_bundle(loader_path) if loader_path else None
        dummy = example_input_from_loader(loader)
        return export_via_fedcore(
            model,
            framework=fw,
            export_dir=export_dir,
            model_name=model_name,
            example_input=dummy,
        )

    if operation not in caps.operations:
        raise PermissionError(
            f"Operation '{operation}' is not available for kind '{caps.kind}'. "
            f"Allowed: {caps.operations}"
        )
    if not loader_path:
        raise ValueError(f"Operation '{operation}' requires a loader .pt bundle")

    loader = load_dataloader_from_bundle(loader_path)
    out_dir = Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if operation == "quantize":
        allow_conv = caps.kind == "convolutional" or caps.has_conv
        allow_emb = caps.kind == "attention_embedding" or caps.has_emb
        if caps.kind == "other":
            allow_conv = caps.has_conv
            allow_emb = caps.has_emb or not caps.has_conv
        return quantize_via_fedcore(
            model,
            loader,
            allow_conv=allow_conv,
            allow_emb=allow_emb,
            output_path=out_dir / f"{model_name}_int8.pt",
        )
    if operation == "prune":
        return prune_via_fedcore(
            model,
            loader,
            pruning_ratio=pruning_ratio,
            output_path=out_dir / f"{model_name}_pruned.pt",
        )
    if operation == "low_rank":
        return low_rank_via_fedcore(
            model,
            loader,
            output_path=out_dir / f"{model_name}_lowrank.pt",
        )
    raise ValueError(f"Unknown operation: {operation}")
