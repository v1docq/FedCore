"""Обучение исходной ResNet18 и статическая INT8-квантизация через FedCore.

Структура эксперимента:
1. Загрузка CIFAR-10.
2. Создание ResNet18 из реестра CLF_MODELS FedCore.
3. Полноценное обучение FP32-модели на train-выборке.
4. Передача обученной модели в BaseQuantizer и получение INT8-модели.
5. Сравнение accuracy, размера файла и latency; сохранение CSV и PNG.

Выбрана static post-training quantization (статическая квантизация после
обучения), потому что ResNet18 почти целиком состоит из свёрточных слоёв:

* dynamic quantization хорошо ускоряет Linear-слои, но в ResNet18 они занимают
  малую часть вычислений;
* QAT имитирует INT8 прямо во время обучения и может лучше сохранить accuracy,
  но заметно усложняет учебный пример;
* static PTQ даёт понятный цикл: сначала обучение FP32, затем калибровка без
  обучения и однократное преобразование в INT8.

Модули FedCore в примере:
* CLF_MODELS — реестр архитектур; отсюда берётся ResNet18;
* CompressionInputData — контейнер модели и DataLoader для BaseQuantizer;
* BaseQuantizer — prepare → calibration → convert;
* QDQWrapper / QDQWrapping — границы Quantize/DeQuantize вокруг leaf-слоёв.

Запуск::

    cd examples/quantization_resnet18
    python run_quantization_resnet18.py

Быстрый прогон (как ранее)::

    python run_quantization_resnet18.py --epochs 3 --train-samples 5000 --test-samples 1000

После прогона в ``results/quantization_resnet18/`` лежат:
``*_state_dict.pt`` (веса), ``*_model.pt`` (полный модуль для анализатора),
``*_graph.json`` (слои/рёбра), ``loader_*.pt`` (данные без role-меток).
Для Web UI: модель ``*_model.pt``, любые ``loader_*.pt``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_EXPORTER_DIR = REPO_ROOT / "model_exporter"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MODEL_EXPORTER_DIR))

import matplotlib.pyplot as plt
import numpy as np
import torch
from fedot.core.repository.tasks import Task, TaskTypesEnum
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from fedcore.algorithm.quantization.quantizers import BaseQuantizer
from fedcore.algorithm.quantization.utils import QDQWrapper, QDQWrapping
from fedcore.data.data import CompressionInputData
from fedcore.models.backbone.convolutional.resnet import CLF_MODELS
from loader_bundle import LoaderBundle
from model_graph_view import ModelGraphBuilder


@dataclass
class Config:
    """Параметры воспроизводимого эксперимента квантизации."""

    seed: int = 42
    epochs: int = 10
    train_samples: int = 50_000
    test_samples: int = 10_000
    calibration_samples: int = 1_024
    batch_size: int = 64
    learning_rate: float = 1e-3
    output_dir: Path = REPO_ROOT / "results" / "quantization_resnet18"
    data_dir: Path = REPO_ROOT / "datasets"


@dataclass
class Metrics:
    """Метрики одной версии модели для CSV и инфографики."""

    model: str
    accuracy: float
    size_mib: float
    latency_ms: float
    quantized_modules: int


class RunnableBaseQuantizer(BaseQuantizer):
    """Адаптер к текущей версии FedCore для исполняемого static PTQ.

    ``BaseQuantizer.fit`` уже содержит рабочий static PTQ, но класс формально
    не реализует один abstract-метод родителя. Training hooks имеют устаревшую
    сигнатуру и для PTQ не нужны: обучение уже завершено. Адаптер не заменяет
    алгоритм — prepare/calibration/convert выполняются исходным ``fit``.

    Свойства model_before/model_after хранят модели в памяти и не запускают
    ModelRegistry: для одного примера промежуточный registry избыточен.
    """

    @property
    def model_before(self):
        return self._model_before_cached

    @model_before.setter
    def model_before(self, model):
        self._model_before_cached = model

    @property
    def model_after(self):
        return self._model_after_cached

    @model_after.setter
    def model_after(self, model):
        self._model_after_cached = model

    def _init_hooks(self, input_data):
        """Отключение training hooks: static PTQ выполняется после обучения."""

        self._on_epoch_start, self._on_epoch_end = [], []

    def _init_trainer_model_before_model_after_and_incapsulate_hooks(
        self, input_data
    ):
        """Реализация abstract-контракта родительского класса."""

        self._init_model(input_data)


class Cifar10Data:
    """Загрузка CIFAR-10 и подготовка train/test/calibration DataLoader.

    CIFAR-10 содержит 60 000 цветных изображений 32×32 десяти классов:
    самолёт, автомобиль, птица, кошка, олень, собака, лягушка, лошадь,
    корабль и грузовик. По умолчанию используется полный train/test набор;
    размер можно уменьшить аргументами CLI для ускорения прогона.
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

        n_train = min(config.train_samples, len(train))
        n_test = min(config.test_samples, len(test))
        n_calib = min(config.calibration_samples, n_train)

        self.train = DataLoader(
            Subset(train, train_ids[:n_train]),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )
        self.calibration = DataLoader(
            Subset(train, train_ids[:n_calib]),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )
        self.test = DataLoader(
            Subset(test, test_ids[:n_test]),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )


class ResNet18ExperimentModel:
    """Создание и обучение исходной FP32 ResNet18 для CIFAR-10."""

    @staticmethod
    def create() -> nn.Module:
        """Создание ResNet18 из FedCore с входом под изображения 32×32."""

        # FedCore: CLF_MODELS — реестр classification-backbone.
        model = CLF_MODELS["ResNet18"](weights=None, num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model.cpu()

    @staticmethod
    @torch.inference_mode()
    def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
        """Расчёт loss и accuracy на переданном DataLoader."""

        model.eval()
        loss_function = nn.CrossEntropyLoss()
        loss_sum, correct, total = 0.0, 0, 0
        for images, labels in loader:
            logits = model(images)
            loss_sum += loss_function(logits, labels).item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
        return loss_sum / total, correct / total

    @staticmethod
    def train(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        learning_rate: float,
    ) -> None:
        """Полный FP32-цикл обучения с валидацией после каждой эпохи.

        На каждой эпохе:
        1. model.train() — включение Dropout/BatchNorm в режиме обучения;
        2. forward → CrossEntropyLoss → backward → Adam.step;
        3. model.eval() и оценка на test-выборке без градиентов;
        4. StepLR уменьшает learning rate каждые 4 эпохи.
        """

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=4, gamma=0.5
        )
        loss_function = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            loss_sum, correct, total = 0.0, 0, 0
            for images, labels in train_loader:
                optimizer.zero_grad()
                logits = model(images)
                loss = loss_function(logits, labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(labels)
                correct += (logits.argmax(1) == labels).sum().item()
                total += len(labels)

            train_loss = loss_sum / total
            train_acc = correct / total
            val_loss, val_acc = ResNet18ExperimentModel.evaluate(
                model, test_loader
            )
            scheduler.step()
            print(
                f"Epoch {epoch + 1}/{epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.2%}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.2%}, "
                f"lr={scheduler.get_last_lr()[0]:.5f}"
            )
        model.eval()


class FedCoreStaticQuantizer:
    """Преобразование обученной FP32 ResNet18 в INT8 через BaseQuantizer.

    Q/DQ означает Quantize/DeQuantize — границы между FP32-тензорами и
    INT8-слоями. В текущем FedCore автоматические границы не покрывают
    torchvision ResNet18, поэтому QDQWrapping явно ставится вокруг Conv2d,
    BatchNorm2d и Linear. Residual-сложения остаются в FP32.
    """

    @staticmethod
    def quantize(model: nn.Module, calibration: DataLoader) -> nn.Module:
        engine = (
            "fbgemm"
            if "fbgemm" in torch.backends.quantized.supported_engines
            else "x86"
        )
        # FedCore: BaseQuantizer выполняет static PTQ.
        quantizer = RunnableBaseQuantizer(
            {
                "quant_type": "static",
                "backend": engine,
                "device": torch.device("cpu"),
                "dtype": torch.qint8,
                "allow_conv": True,
            }
        )
        quantizer.logger.setLevel(logging.WARNING)

        qconfig = quantizer.qconfig[""]
        model = copy.deepcopy(model).eval()
        layers = [
            (name, layer)
            for name, layer in model.named_modules()
            if name and isinstance(layer, (nn.Conv2d, nn.BatchNorm2d, nn.Linear))
        ]
        for name, layer in layers:
            layer.qconfig = qconfig
            # FedCore: QDQWrapper / QDQWrapping — явные Q/DQ-границы.
            QDQWrapper.set_module(
                model, name, QDQWrapping(layer, mode="both", qconfig=qconfig)
            )

        # FedCore: CompressionInputData — контракт данных для quantizer.fit.
        data = CompressionInputData(
            features=np.zeros((1, 1), dtype=np.float32),
            target=model,
            train_dataloader=calibration,
            val_dataloader=calibration,
            task=Task(TaskTypesEnum.classification),
            input_dim=3,
            num_classes=10,
        )
        print(
            f"FedCore: prepare → calibration ({len(calibration.dataset)} images) "
            "→ convert to INT8"
        )
        return quantizer.fit(data).cpu().eval()


class ModelArtifacts:
    """Сохранение весов, полного модуля и графа слоёв для анализатора."""

    @staticmethod
    def save(name: str, model: nn.Module, output_dir: Path) -> dict[str, Path]:
        """Пишет state_dict, полный ``nn.Module`` и JSON-граф.

        Полный ``*_model.pt`` нужен Web-анализатору (``/analyze_model``):
        там ожидается ``torch.nn.Module``, не голый ``state_dict``.
        JSON-граф — иерархическое представление ``ModelGraphBuilder``.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        model = model.cpu().eval()

        state_path = output_dir / f"resnet18_{name}_state_dict.pt"
        model_path = output_dir / f"resnet18_{name}_model.pt"
        graph_path = output_dir / f"resnet18_{name}_graph.json"

        torch.save(model.state_dict(), state_path)
        torch.save(model, model_path)

        graph = ModelGraphBuilder().build(model).to_dict()
        graph_path.write_text(
            json.dumps(graph, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(
            f"Артефакты [{name}]:\n"
            f"  state_dict: {state_path}\n"
            f"  model:      {model_path}\n"
            f"  graph:      {graph_path} "
            f"({graph.get('total_modules', graph.get('total_layers', 0))} modules)"
        )
        return {
            "state_dict": state_path,
            "model": model_path,
            "graph": graph_path,
        }


class LoaderArtifacts:
    """Сохранение DataLoader как нейтральные FedCore bundles (без role)."""

    @staticmethod
    def save(data: Cifar10Data, output_dir: Path) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        specs = (
            (data.train, "loader_01.pt"),
            (data.calibration, "loader_02.pt"),
            (data.test, "loader_03.pt"),
        )
        saved: dict[str, Path] = {}
        for index, (loader, filename) in enumerate(specs, start=1):
            path = LoaderBundle.save(
                output_dir / filename,
                loader,
                name=Path(filename).stem,
                num_classes=10,
            )
            saved[f"loader_{index:02d}"] = path
        return saved


class MetricEvaluator:
    """Расчёт accuracy, размера state_dict и latency на CPU."""

    @staticmethod
    @torch.inference_mode()
    def evaluate(
        name: str, model: nn.Module, loader: DataLoader, output_dir: Path
    ) -> Metrics:
        model.eval()
        correct = total = 0
        for images, labels in loader:
            correct += (model(images).argmax(1) == labels).sum().item()
            total += len(labels)

        artifacts = ModelArtifacts.save(name, model, output_dir)
        size = artifacts["state_dict"].stat().st_size / 1024**2

        sample = next(iter(loader))[0][:1]
        for _ in range(3):
            model(sample)
        times = []
        for _ in range(20):
            started = time.perf_counter()
            model(sample)
            times.append((time.perf_counter() - started) * 1000)

        quantized = sum(
            ".quantized" in layer.__class__.__module__.lower()
            for layer in model.modules()
        )
        return Metrics(
            name, correct / total, size, statistics.fmean(times), quantized
        )


class Infographic:
    """Сохранение PNG-сравнения исходной FP32 и INT8-модели."""

    @staticmethod
    def save(fp32: Metrics, int8: Metrics, path: Path) -> None:
        colors = ["#3977d6", "#ef7d32"]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        fig.suptitle("FedCore: ResNet18 до и после static INT8 PTQ", fontsize=16)
        for axis, values, title, unit in (
            (axes[0], [fp32.accuracy, int8.accuracy], "Accuracy", "доля"),
            (axes[1], [fp32.size_mib, int8.size_mib], "Размер state_dict", "MiB"),
            (axes[2], [fp32.latency_ms, int8.latency_ms], "Latency batch=1", "ms"),
        ):
            bars = axis.bar(["FP32", "INT8"], values, color=colors)
            axis.set_title(title)
            axis.set_ylabel(unit)
            axis.grid(axis="y", alpha=0.25)
            axis.bar_label(bars, fmt="%.3f", padding=3)
        fig.text(
            0.5,
            0.01,
            f"Уменьшение размера: {fp32.size_mib / int8.size_mib:.2f}×; "
            f"ускорение: {fp32.latency_ms / int8.latency_ms:.2f}×; "
            f"INT8-модулей: {int8.quantized_modules}",
            ha="center",
        )
        fig.tight_layout(rect=(0, 0.06, 1, 0.92))
        fig.savefig(path, dpi=160)
        plt.close(fig)


def section(number: int, title: str, text: str) -> None:
    """Печать этапа эксперимента в консоль."""

    print(f"\n{'=' * 72}\nЭТАП {number}. {title}\n{'=' * 72}\n{text}")


def save_report(metrics: list[Metrics], path: Path) -> None:
    """Сохранение таблицы сравнения в CSV."""

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(metrics[0])))
        writer.writeheader()
        writer.writerows(asdict(metric) for metric in metrics)


def parse_args() -> Config:
    """Разбор параметров эксперимента из командной строки."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=50_000)
    parser.add_argument("--test-samples", type=int, default=10_000)
    parser.add_argument("--calibration-samples", type=int, default=1_024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "quantization_resnet18",
    )
    args = parser.parse_args()
    return Config(
        epochs=args.epochs,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        calibration_samples=args.calibration_samples,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir.resolve(),
    )


def main() -> None:
    """Последовательность: обучение → static PTQ → сравнение метрик."""

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
        "Загрузка CIFAR-10 и подготовка train/test/calibration DataLoader.",
    )
    data = Cifar10Data(config)
    print(
        f"train={len(data.train.dataset)}, test={len(data.test.dataset)}, "
        f"calibration={len(data.calibration.dataset)}"
    )
    loader_paths = LoaderArtifacts.save(data, config.output_dir)

    section(
        2,
        "Исходная модель",
        "Создание ResNet18 из CLF_MODELS FedCore и полноценное FP32-обучение.",
    )
    fp32_model = ResNet18ExperimentModel.create()
    ResNet18ExperimentModel.train(
        fp32_model,
        data.train,
        data.test,
        config.epochs,
        config.learning_rate,
    )
    fp32 = MetricEvaluator.evaluate("fp32", fp32_model, data.test, config.output_dir)
    print(fp32)

    section(
        3,
        "Static INT8 PTQ",
        "Обучение завершено. Калибровка только наблюдает диапазоны значений; "
        "BaseQuantizer выполняет prepare → convert.",
    )
    int8_model = FedCoreStaticQuantizer.quantize(fp32_model, data.calibration)
    int8 = MetricEvaluator.evaluate("int8", int8_model, data.test, config.output_dir)
    if int8.quantized_modules == 0 or int8.size_mib >= fp32.size_mib:
        raise RuntimeError("Проверка не подтвердила настоящую INT8-квантизацию.")
    print(int8)

    section(4, "Результаты", "Сохранение CSV-таблицы и PNG-инфографики.")
    save_report([fp32, int8], config.output_dir / "metrics.csv")
    Infographic.save(
        fp32, int8, config.output_dir / "quantization_comparison.png"
    )
    print(f"Уменьшение размера: {fp32.size_mib / int8.size_mib:.2f}×")
    print(f"Ускорение: {fp32.latency_ms / int8.latency_ms:.2f}×")
    print(f"Результаты: {config.output_dir}")
    _ = loader_paths


if __name__ == "__main__":
    main()
