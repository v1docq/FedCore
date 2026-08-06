"""Структура эксперимента:
1. Частичное обучение FP32-модели.
2. Перевод модели в режим model.eval().
3. Экспорт через torch.onnx.export → файл .onnx.
4. Проверка структуры через onnx.checker.
5. Запуск через FedCore ONNXInferenceModel (обёртка FedCoreOnnxRunner).
6. Сравнение ответов PyTorch и ONNX.

Частичное обучение означает: выполняется несколько эпох на уменьшенной
выборке, чтобы получить осмысленный FP32-baseline для экспорта и сравнения.
Этого достаточно для демонстрации переноса модели в ONNX; полный цикл
обучения здесь не требуется, потому что цель примера — корректный экспорт
и совпадение ответов, а не максимальная accuracy.

Используемые элементы FedCore:
* CLF_MODELS — реестр архитектур; отсюда берётся ResNet18;
* ONNXInferenceModel — загрузка .onnx и инференс через ONNX Runtime.
FedCore.export() не используется: метод сейчас пустой (pass).

Классы примера:
* ResNet18ExperimentModel — создание и частичное обучение FP32 ResNet18;
* OnnxExporter — torch.onnx.export + onnx.checker;
* FedCoreOnnxRunner — обёртка над ONNXInferenceModel с нормализацией выхода;
* MetricEvaluator — accuracy / размер / latency для PyTorch и ONNX;
* Infographic — PNG: accuracy, размер, latency.

Вывод в консоль:
Данные
Исходная FP32-модель
Экспорт в ONNX
FedCore-инференс и сравнение
CSV + инфографика

Проводимые сравнения:
* accuracy PyTorch и ONNX на одном test-наборе;
* размер .pt / .onnx;
* latency batch=1;
* max |PyTorch − ONNX| на одном batch;
* проверка onnx.checker;
* forward с batch=1 и batch=4 (dynamic axes).

Критерий успешного завершения:
checker без ошибок, формы совпадают, расхождение малое (порядка 1e-4…1e-5),
CSV и PNG созданы.

Запуск::

    cd examples/export_onnx
    python export_to_onnx.py
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from fedcore.inference.onnx import ONNXInferenceModel
from fedcore.models.backbone.convolutional.resnet import CLF_MODELS

try:
    import onnx
except ImportError as error:
    raise SystemExit(
        "Нужен пакет onnx для onnx.checker. Установка: pip install onnx"
    ) from error


@dataclass
class Config:
    """Параметры воспроизводимого эксперимента экспорта в ONNX."""

    seed: int = 42
    epochs: int = 1
    train_samples: int = 2_000
    test_samples: int = 500
    batch_size: int = 64
    opset_version: int = 17
    output_dir: Path = REPO_ROOT / "results" / "export_onnx"
    data_dir: Path = REPO_ROOT / "datasets"


@dataclass
class Metrics:
    """Метрики одной версии модели для CSV и инфографики."""

    model: str
    accuracy: float
    size_mib: float
    latency_ms: float


class Cifar10Data:
    """Загрузка CIFAR-10 и подготовка train/test DataLoader.

    CIFAR-10 содержит 60 000 цветных изображений 32×32 десяти классов.
    Подмножества используются для ускорения демонстрационного прогона на CPU.
    """

    def __init__(self, config: Config):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616),
                ),
            ]
        )
        train = datasets.CIFAR10(
            config.data_dir, train=True, download=True, transform=transform
        )
        test = datasets.CIFAR10(
            config.data_dir, train=False, download=True, transform=transform
        )
        generator = torch.Generator().manual_seed(config.seed)
        train_ids = torch.randperm(len(train), generator=generator)
        test_ids = torch.randperm(len(test), generator=generator)

        self.train = DataLoader(
            Subset(train, train_ids[: config.train_samples]),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )
        self.test = DataLoader(
            Subset(test, test_ids[: config.test_samples]),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )


class ResNet18ExperimentModel:
    """Создание и частичное обучение исходной ResNet18 из реестра FedCore.

    ``CLF_MODELS`` — реестр поддерживаемых архитектур FedCore.
    Из него берётся torchvision ResNet18; входной stem адаптируется под
    CIFAR-10 (изображения 32×32 вместо ImageNet 224×224).
    """

    @staticmethod
    def create() -> nn.Module:
        """Создание ResNet18 из FedCore с входом под 32×32."""

        model = CLF_MODELS["ResNet18"](weights=None, num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model.cpu()

    @staticmethod
    def train(model: nn.Module, loader: DataLoader, epochs: int) -> None:
        """Частичный FP32-цикл: forward → loss → backward → optimizer.

        Цикл частичный намеренно: цель примера — экспорт и совпадение ответов
        PyTorch/ONNX, а не достижение максимальной accuracy на CIFAR-10.
        """

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_function = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            model.train()
            loss_sum, correct, total = 0.0, 0, 0
            for images, labels in loader:
                optimizer.zero_grad()
                logits = model(images)
                loss = loss_function(logits, labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(labels)
                correct += (logits.argmax(1) == labels).sum().item()
                total += len(labels)
            print(
                f"Epoch {epoch + 1}/{epochs}: "
                f"loss={loss_sum / total:.4f}, accuracy={correct / total:.2%}"
            )
        model.eval()


class OnnxExporter:
    """Экспорт PyTorch-модели в ONNX и проверка структуры файла.

    Экспорт выполняется через ``torch.onnx.export``: API ``FedCore.export()``
    сейчас пустой (``pass``), поэтому используется прямой экспорт. Имя входа
    обязательно ``input`` — так жёстко ожидает ``ONNXInferenceModel`` FedCore.
    """

    @staticmethod
    def export(
        model: nn.Module, example_input: torch.Tensor, path: Path, opset: int
    ) -> Path:
        """Сохранение .onnx с динамическим размером batch."""

        model.eval()
        path.parent.mkdir(parents=True, exist_ok=True)
        # dynamo=False нужен в новых PyTorch; в 2.3.1 такого аргумента ещё нет.
        export_kwargs = dict(
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )
        try:
            torch.onnx.export(
                model, example_input, str(path), dynamo=False, **export_kwargs
            )
        except TypeError:
            torch.onnx.export(model, example_input, str(path), **export_kwargs)
        print(f"ONNX-файл сохранён: {path}")
        return path

    @staticmethod
    def check(path: Path) -> None:
        """Проверка структуры файла через onnx.checker."""

        onnx_model = onnx.load(str(path))
        onnx.checker.check_model(onnx_model)
        print("onnx.checker: структура модели корректна")


class FedCoreOnnxRunner(nn.Module):
    """Запуск .onnx через ``ONNXInferenceModel`` FedCore.

    ``ONNXInferenceModel`` открывает файл через ONNX Runtime.
    В текущей версии ``forward`` оборачивает весь список выходов ORT в
    ``torch.Tensor``, из-за чего появляется лишнее измерение. Здесь берётся
    первый выход и возвращается тензор формы ``(batch, classes)``.
    """

    def __init__(self, onnx_path: Path):
        super().__init__()
        # FedCore: загрузка сессии ONNX Runtime с именем входа "input".
        self.backend = ONNXInferenceModel(str(onnx_path))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Инференс с нормализацией формы ответа."""

        raw = self.backend(inputs)
        if isinstance(raw, (list, tuple)):
            raw = raw[0]
        if not isinstance(raw, torch.Tensor):
            raw = torch.as_tensor(raw)
        if raw.ndim == 3 and raw.shape[0] == 1:
            raw = raw.squeeze(0)
        return raw.float()


class MetricEvaluator:
    """Расчёт accuracy, размера артефакта и latency на CPU."""

    @staticmethod
    @torch.inference_mode()
    def evaluate(
        name: str,
        model: nn.Module,
        loader: DataLoader,
        artifact_path: Path,
    ) -> Metrics:
        model.eval()
        correct = total = 0
        for images, labels in loader:
            correct += (model(images).argmax(1) == labels).sum().item()
            total += len(labels)

        size = artifact_path.stat().st_size / 1024**2
        sample = next(iter(loader))[0][:1]
        for _ in range(3):
            model(sample)
        times = []
        for _ in range(20):
            started = time.perf_counter()
            model(sample)
            times.append((time.perf_counter() - started) * 1000)

        return Metrics(name, correct / total, size, statistics.fmean(times))

    @staticmethod
    @torch.inference_mode()
    def max_abs_difference(
        pytorch_model: nn.Module, onnx_model: nn.Module, batch: torch.Tensor
    ) -> float:
        """Расчёт max |PyTorch − ONNX| на одном batch."""

        left = pytorch_model(batch)
        right = onnx_model(batch)
        if left.shape != right.shape:
            raise RuntimeError(
                f"Формы не совпадают: PyTorch {tuple(left.shape)} "
                f"vs ONNX {tuple(right.shape)}"
            )
        return float(torch.max(torch.abs(left - right)).item())

    @staticmethod
    @torch.inference_mode()
    def check_dynamic_batch(onnx_model: nn.Module, sample: torch.Tensor) -> None:
        """Проверка поддержки dynamic axes для batch=1 и batch=4."""

        for batch_size in (1, 4):
            batch = sample[:1].repeat(batch_size, 1, 1, 1)
            output = onnx_model(batch)
            if tuple(output.shape) != (batch_size, 10):
                raise RuntimeError(
                    f"Ожидалась shape={(batch_size, 10)}, получена {tuple(output.shape)}"
                )
            print(f"dynamic batch={batch_size}: OK, shape={tuple(output.shape)}")


class Infographic:
    """Сохранение PNG-сравнения PyTorch и ONNX."""

    @staticmethod
    def save(
        pytorch: Metrics, onnx_metrics: Metrics, max_diff: float, path: Path
    ) -> None:
        colors = ["#3977d6", "#ef7d32"]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        fig.suptitle("FedCore: ResNet18 PyTorch vs ONNX", fontsize=16)
        for axis, values, title, unit in (
            (
                axes[0],
                [pytorch.accuracy, onnx_metrics.accuracy],
                "Accuracy",
                "доля",
            ),
            (
                axes[1],
                [pytorch.size_mib, onnx_metrics.size_mib],
                "Размер файла",
                "MiB",
            ),
            (
                axes[2],
                [pytorch.latency_ms, onnx_metrics.latency_ms],
                "Latency batch=1",
                "ms",
            ),
        ):
            bars = axis.bar(["PyTorch", "ONNX"], values, color=colors)
            axis.set_title(title)
            axis.set_ylabel(unit)
            axis.grid(axis="y", alpha=0.25)
            axis.bar_label(bars, fmt="%.3f", padding=3)
        fig.text(
            0.5,
            0.01,
            f"max |PyTorch − ONNX| = {max_diff:.6f}",
            ha="center",
        )
        fig.tight_layout(rect=(0, 0.06, 1, 0.92))
        fig.savefig(path, dpi=160)
        plt.close(fig)


def section(number: int, title: str, text: str) -> None:
    """Печать этапа эксперимента в консоль."""

    print(f"\n{'=' * 72}\nЭТАП {number}. {title}\n{'=' * 72}\n{text}")


def save_report(metrics: list[Metrics], max_diff: float, path: Path) -> None:
    """Сохранение CSV-отчёта сравнения."""

    rows = []
    for item in metrics:
        row = asdict(item)
        row["max_abs_difference"] = max_diff
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> Config:
    """Разбор параметров эксперимента из командной строки."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-samples", type=int, default=2_000)
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--opset-version", type=int, default=17)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "export_onnx",
    )
    args = parser.parse_args()
    return Config(
        epochs=args.epochs,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        batch_size=args.batch_size,
        opset_version=args.opset_version,
        output_dir=args.output_dir.resolve(),
    )


def main() -> None:
    """Последовательность: частичное обучение → экспорт → сравнение."""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    section(
        1,
        "Данные",
        "Загрузка CIFAR-10 и подготовка train/test DataLoader.",
    )
    data = Cifar10Data(config)
    print(f"train={len(data.train.dataset)}, test={len(data.test.dataset)}")

    section(
        2,
        "Исходная FP32-модель",
        "Создание ResNet18 из CLF_MODELS FedCore и частичное обучение.",
    )
    pytorch_model = ResNet18ExperimentModel.create()
    ResNet18ExperimentModel.train(pytorch_model, data.train, config.epochs)
    state_dict_path = config.output_dir / "resnet18_fp32_state_dict.pt"
    torch.save(pytorch_model.state_dict(), state_dict_path)
    pytorch_metrics = MetricEvaluator.evaluate(
        "pytorch", pytorch_model, data.test, state_dict_path
    )
    print(pytorch_metrics)

    section(
        3,
        "Экспорт в ONNX",
        "Сохранение графа и весов через torch.onnx.export; "
        "проверка структуры через onnx.checker.",
    )
    example_input = next(iter(data.test))[0][:1]
    onnx_path = config.output_dir / "resnet18.onnx"
    OnnxExporter.export(
        pytorch_model, example_input, onnx_path, config.opset_version
    )
    OnnxExporter.check(onnx_path)

    section(
        4,
        "FedCore-инференс и сравнение",
        "Запуск .onnx через ONNXInferenceModel; сравнение ответов с PyTorch.",
    )
    onnx_model = FedCoreOnnxRunner(onnx_path)
    MetricEvaluator.check_dynamic_batch(onnx_model, example_input)
    max_diff = MetricEvaluator.max_abs_difference(
        pytorch_model, onnx_model, example_input
    )
    print(f"max |PyTorch − ONNX| = {max_diff:.6f}")
    if max_diff > 1e-4:
        raise RuntimeError(
            f"Слишком большое расхождение ответов: {max_diff:.6f} > 1e-4"
        )

    onnx_metrics = MetricEvaluator.evaluate(
        "onnx", onnx_model, data.test, onnx_path
    )
    print(onnx_metrics)

    section(5, "CSV + инфографика", "Сохранение отчёта и PNG-сравнения.")
    save_report(
        [pytorch_metrics, onnx_metrics],
        max_diff,
        config.output_dir / "export_report.csv",
    )
    Infographic.save(
        pytorch_metrics,
        onnx_metrics,
        max_diff,
        config.output_dir / "onnx_comparison.png",
    )
    print(f"Результаты: {config.output_dir}")


if __name__ == "__main__":
    main()
