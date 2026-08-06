"""Smoke-тест экспорта модели через FedCore.export().

Поддерживаемые форматы (как в model_exporter WebUI):
* torchscript → .pt
* onnx → .onnx
* tensorrt → .engine

Сценарий (onnx по умолчанию):
1. Создать FedCore с минимальным api_config (без fit / без датасета).
2. Взять ResNet18 из реестра FedCore со случайными весами.
3. Экспортировать через FedCore.export().
3.1. Для onnx: checker, размер, сравнение с PyTorch, dynamic batch.
3.2. Для torchscript: загрузка и forward на dummy.
3.3. Для tensorrt: экспорт (требует NVIDIA TensorRT); при отсутствии SDK — ошибка.

Запуск::

    python examples/export_onnx/export_onnx.py
    python examples/export_onnx/export_onnx.py --format torchscript
    python examples/export_onnx/export_onnx.py --format tensorrt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn

from fedcore.api.api_configs import (
    APIConfigTemplate,
    AutoMLConfigTemplate,
    FedotConfigTemplate,
    LearningConfigTemplate,
)
from fedcore.api.config_factory import ConfigFactory
from fedcore.api.main import FedCore
from fedcore.inference.onnx import ONNXInferenceModel
from fedcore.models.backbone.convolutional.resnet import CLF_MODELS
from fedcore.tools.export import normalize_framework

OUTPUT_DIR = REPO_ROOT / "results" / "export_onnx"
SEED = 42
MAX_ABS_DIFF = 1e-4

_FORMAT_SUFFIX = {
    "torchscript": ".pt",
    "onnx": ".onnx",
    "tensorrt": ".engine",
}


class FedCoreOnnxRunner(nn.Module):
    """Обёртка над ONNXInferenceModel с нормализацией формы выхода.

    ``ONNXInferenceModel.forward`` оборачивает список выходов ORT в
    ``torch.Tensor``, из-за чего появляется лишнее измерение. Берём первый
    выход и при необходимости сжимаем ведущую ось размера 1.
    """

    def __init__(self, onnx_path: Path):
        super().__init__()
        self.backend = ONNXInferenceModel(str(onnx_path))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raw = self.backend(inputs)
        if isinstance(raw, (list, tuple)):
            raw = raw[0]
        if not isinstance(raw, torch.Tensor):
            raw = torch.as_tensor(raw)
        if raw.ndim == 3 and raw.shape[0] == 1:
            raw = raw.squeeze(0)
        return raw.float()


def build_fedcore() -> FedCore:
    """Минимальный FedCore без обучения и без данных."""
    fedot_config = FedotConfigTemplate(
        problem="classification",
        metric=["accuracy"],
        pop_size=1,
        timeout=1,
        initial_assumption="ResNet18",
    )
    automl_config = AutoMLConfigTemplate(fedot_config=fedot_config)
    learning_config = LearningConfigTemplate(
        criterion="cross_entropy",
        learning_strategy="from_scratch",
    )
    api_template = APIConfigTemplate(
        automl_config=automl_config,
        learning_config=learning_config,
    )
    APIConfig = ConfigFactory.from_template(api_template)
    return FedCore(APIConfig())


def create_model() -> nn.Module:
    """ResNet18 из FedCore со случайными весами (вход 224×224)."""
    model = CLF_MODELS["ResNet18"](weights=None, num_classes=10)
    return model.cpu().eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FedCore multi-format export smoke")
    parser.add_argument(
        "--format",
        default="onnx",
        choices=["torchscript", "onnx", "tensorrt", "pt", "engine", "trt"],
        help="Export format (default: onnx)",
    )
    return parser.parse_args()


def verify_onnx(model: nn.Module, exported: Path, dummy: torch.Tensor) -> None:
    import onnx

    print("2. onnx.checker")
    onnx.checker.check_model(onnx.load(str(exported)))
    print("   OK")

    size_mib = exported.stat().st_size / 1024**2
    print(f"   ONNX size: {size_mib:.3f} MiB")

    print("3. PyTorch vs ONNX (dummy batch)")
    onnx_runner = FedCoreOnnxRunner(exported)
    with torch.inference_mode():
        pt_out = model(dummy)
        onnx_out = onnx_runner(dummy)
        if pt_out.shape != onnx_out.shape:
            raise RuntimeError(
                f"Shape mismatch: PyTorch {tuple(pt_out.shape)} "
                f"vs ONNX {tuple(onnx_out.shape)}"
            )
        max_diff = float(torch.max(torch.abs(pt_out - onnx_out)).item())
    print(f"   max |PyTorch − ONNX| = {max_diff:.6e}")
    if max_diff > MAX_ABS_DIFF:
        raise RuntimeError(
            f"Too large discrepancy: {max_diff:.6e} > {MAX_ABS_DIFF}"
        )

    print("4. Dynamic batch (batch=1 and batch=4)")
    with torch.inference_mode():
        for batch_size in (1, 4):
            batch = dummy[:1].repeat(batch_size, 1, 1, 1)
            output = onnx_runner(batch)
            expected = (batch_size, 10)
            if tuple(output.shape) != expected:
                raise RuntimeError(
                    f"Expected shape={expected}, got {tuple(output.shape)}"
                )
            print(f"   dynamic batch={batch_size}: OK, shape={tuple(output.shape)}")


def verify_torchscript(model: nn.Module, exported: Path, dummy: torch.Tensor) -> None:
    print("2. Load TorchScript / PyTorch artifact and forward")
    size_mib = exported.stat().st_size / 1024**2
    print(f"   size: {size_mib:.3f} MiB")
    try:
        loaded = torch.jit.load(str(exported), map_location="cpu")
    except Exception:
        loaded = torch.load(str(exported), map_location="cpu")
    loaded.eval()
    with torch.inference_mode():
        pt_out = model(dummy)
        ts_out = loaded(dummy)
        if pt_out.shape != ts_out.shape:
            raise RuntimeError(
                f"Shape mismatch: PyTorch {tuple(pt_out.shape)} "
                f"vs loaded {tuple(ts_out.shape)}"
            )
        max_diff = float(torch.max(torch.abs(pt_out - ts_out)).item())
    print(f"   max |PyTorch − loaded| = {max_diff:.6e}")
    if max_diff > MAX_ABS_DIFF:
        raise RuntimeError(
            f"Too large discrepancy: {max_diff:.6e} > {MAX_ABS_DIFF}"
        )


def verify_tensorrt(exported: Path) -> None:
    size_mib = exported.stat().st_size / 1024**2
    print("2. TensorRT engine written")
    print(f"   size: {size_mib:.3f} MiB")
    if exported.suffix.lower() != ".engine" or size_mib <= 0:
        raise RuntimeError(f"Unexpected TensorRT artifact: {exported}")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    backend = normalize_framework(args.format)
    suffix = _FORMAT_SUFFIX[backend]
    output_path = OUTPUT_DIR / f"resnet18_smoke{suffix}"

    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("_____FedCore + ResNet18 (random weights)_____")
    fedcore = build_fedcore()
    model = create_model()
    dummy = torch.randn(1, 3, 224, 224)

    print(f"1. FedCore.export → {backend}")
    try:
        exported = fedcore.export(
            framework=backend,
            framework_config={
                "output_path": str(output_path),
                "opset_version": 17,
                "input_names": ["input"],
                "output_names": ["logits"],
                "example_inputs": dummy,
            },
            supplementary_data={"model_to_export": model},
        )
    except ImportError as error:
        raise SystemExit(str(error)) from error

    print(f"    saved: {exported}")

    if backend == "onnx":
        verify_onnx(model, exported, dummy)
    elif backend == "torchscript":
        verify_torchscript(model, exported, dummy)
    else:
        verify_tensorrt(exported)

    print(f"Done. Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
