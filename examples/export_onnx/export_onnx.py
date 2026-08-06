"""Smoke-тест экспорта модели в ONNX через FedCore.export().

Сценарий:
1. Создать FedCore с минимальным api_config (без fit / без датасета).
2. Взять ResNet18 из реестра FedCore со случайными весами.
3. Экспортировать через FedCore.export() → файл .onnx.
4. Проверить структуру через onnx.checker; вывести размер файла.
5. Сравнить ответы PyTorch и ONNX Runtime на одном dummy batch.
6. Проверить dynamic axes для batch=1 и batch=4.

Запуск::

    python examples/export_onnx/export_onnx_new.py
"""

from __future__ import annotations

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

try:
    import onnx
except ImportError as error:
    raise SystemExit(
        "Нужен пакет onnx для onnx.checker. Установка: pip install onnx"
    ) from error


OUTPUT_DIR = REPO_ROOT / "results" / "export_onnx"
ONNX_PATH = OUTPUT_DIR / "resnet18_smoke.onnx"
SEED = 42
MAX_ABS_DIFF = 1e-4


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


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("_____FedCore + ResNet18 (random weights)_____")
    fedcore = build_fedcore()
    model = create_model()
    dummy = torch.randn(1, 3, 224, 224)

    print("1. FedCore.export → ONNX")
    exported = fedcore.export(
        framework="ONNX",
        framework_config={
            "output_path": str(ONNX_PATH),
            "opset_version": 17,
            "input_names": ["input"],
            "output_names": ["logits"],
            "example_inputs": dummy,
        },
        supplementary_data={"model_to_export": model},
    )
    print(f"    saved: {exported}")

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

    print(f"Done. Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
