"""Small real-data scenarios with explicit isolation and data provenance.

No builder downloads data, loads undocumented weights, or measures the test set.
The shared runner trains and records each common baseline before compression.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .protocol import ExperimentBundle, TensorSplit

ROLES = ("train", "validation", "calibration", "test")
DIGITS_SOURCE = "https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits"


class ScenarioUnavailable(ValueError):
    """A required dataset, license, or capability has not been supplied."""


def _digest(data: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


def _model(factory, seed: int) -> nn.Module:
    # The builder does not alter its caller's RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return factory()


def split_indices(y: np.ndarray, seed: int, *, stratify: bool = True) -> dict[str, np.ndarray]:
    """60/15/10/15 split of independent samples; deterministic under seed."""
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(y))
    if len(indices) < 12:
        raise ValueError("At least 12 independent samples are required for four roles")
    train, rest = train_test_split(indices, test_size=.4, random_state=seed,
                                   stratify=y if stratify else None)
    validation, tail = train_test_split(rest, test_size=.625, random_state=seed + 1,
                                        stratify=y[rest] if stratify else None)
    calibration, test = train_test_split(tail, test_size=.6, random_state=seed + 2,
                                         stratify=y[tail] if stratify else None)
    return dict(zip(ROLES, (train, validation, calibration, test)))


@dataclass(frozen=True)
class TrainPreprocessor:
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, train: np.ndarray) -> "TrainPreprocessor":
        train = np.asarray(train, dtype=np.float64)
        if train.ndim != 2 or not len(train) or np.isinf(train).any():
            raise ValueError("Training features must be a finite 2D matrix (NaN is allowed)")
        if np.isnan(train).all(axis=0).any():
            raise ValueError("Cannot fit a median for a completely missing training feature")
        median = np.nanmedian(train, axis=0)
        filled = np.where(np.isnan(train), median, train)
        mean, scale = filled.mean(axis=0), filled.std(axis=0)
        scale = np.where(scale == 0, 1., scale)
        return cls(median, mean, scale)

    def transform(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != len(self.mean) or np.isinf(features).any():
            raise ValueError("Features do not match the fitted preprocessor")
        return ((np.where(np.isnan(features), self.median, features) - self.mean) / self.scale).astype(np.float32)

    def metadata(self) -> dict:
        return {"fitted_on": "train", "imputation": "median", "scaling": "population_std",
                "median": self.median.tolist(), "mean": self.mean.tolist(), "scale": self.scale.tolist()}


class BigMLP(nn.Sequential):
    def __init__(self, inputs: int, outputs: int):
        super().__init__(nn.Linear(inputs, 128), nn.ReLU(), nn.Linear(128, 64),
                         nn.ReLU(), nn.Linear(64, outputs))


class DigitsCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.Flatten())
        self.classifier = nn.Sequential(nn.Linear(16 * 8 * 8, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


class SeriesCNN(nn.Module):
    def __init__(self, channels: int, outputs: int):
        super().__init__()
        self.features = nn.Sequential(nn.Conv1d(channels, 32, 3, padding=1), nn.ReLU(),
                                      nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Linear(256, 64), nn.ReLU())
        self.classifier = nn.Linear(64, outputs)

    def forward(self, x):
        return self.classifier(self.features(x))


class ScaledRegressor(nn.Module):
    """Return predictions in original target units, including after export."""
    def __init__(self, inner: nn.Module, mean: float, scale: float):
        super().__init__()
        self.inner = inner
        self.register_buffer("target_mean", torch.tensor(float(mean)))
        self.register_buffer("target_scale", torch.tensor(float(scale)))

    def forward(self, x):
        return self.inner(x) * self.target_scale + self.target_mean


def _splits(x: np.ndarray, y: np.ndarray, indices: Mapping[str, np.ndarray], prefix: str,
            *, units: Sequence[str] | None = None) -> dict[str, TensorSplit]:
    return {role: TensorSplit(torch.as_tensor(x[index], dtype=torch.float32),
                              torch.as_tensor(y[index]),
                              tuple(f"{prefix}:{int(i)}" for i in index),
                              unit_ids=tuple(str(units[i]) for i in index) if units is not None else ())
            for role, index in indices.items()}


def build_cv_digits(seed: int = 42) -> ExperimentBundle:
    from sklearn.datasets import load_digits
    data = load_digits()
    x, y = data.images[:, None].astype(np.float32) / 16., data.target.astype(np.int64)
    splits = _splits(x, y, split_indices(y, seed), "sklearn-digits")
    return ExperimentBundle(**splits, task="classification", original_model=_model(DigitsCNN, seed),
                            metadata={"scenario": "cv_digits", "dataset": "sklearn.load_digits",
                                      "source": DIGITS_SOURCE, "license": "CC BY 4.0",
                                      "data_sha256": _digest(data.data), "target_sha256": _digest(y),
                                      "resolution": [8, 8], "classes": list(range(10)),
                                      "preprocessing": "fixed division by 16; no fitted statistics",
                                      "evidence_scope": "small real-data experiment; does not reproduce CIFAR/ImageNette",
                                      "initial_state": "untrained; shared runner must train baseline"})


def build_cv_dataset(dataset: str, data_dir: Path, *, architecture: str = "resnet18",
                     seed: int = 42, resolution: int | None = None,
                     max_samples: int | None = None, dataset_revision: str | None = None,
                     storage_dir: Path | None = None, max_materialized_bytes: int = 256 * 1024 * 1024) -> ExperimentBundle:
    """Local CIFAR10/ImageNette route; all changed heads really train from scratch.

    A subset is an explicitly labeled pilot. No historical checkpoint is loaded.
    ImageNette requires the caller to identify the downloaded dataset version.
    """
    from torchvision import datasets, models, transforms
    if architecture not in ("resnet18", "resnet50") or dataset not in ("cifar10", "imagenette"):
        raise ScenarioUnavailable("CV supports CIFAR10/ImageNette with ResNet18/ResNet50")
    if max_samples is not None and max_samples < 80:
        raise ValueError("A CV subset needs at least 80 total examples")
    resolution = resolution or (32 if dataset == "cifar10" else 128)
    if resolution < 16:
        raise ValueError("ResNet resolution must be at least 16")
    if type(max_materialized_bytes) is not int or max_materialized_bytes <= 0:
        raise ValueError("CV materialization limit must be a positive integer")
    transform = transforms.Compose([transforms.Resize((resolution, resolution)), transforms.ToTensor()])
    directory = Path(data_dir)
    try:
        if dataset == "cifar10":
            train_data = datasets.CIFAR10(directory, train=True, download=False, transform=transform)
            test_data = datasets.CIFAR10(directory, train=False, download=False, transform=transform)
            source, license_name = "https://www.cs.toronto.edu/~kriz/cifar.html", "CIFAR10 dataset terms; cite Krizhevsky 2009"
            revision = dataset_revision or "cifar-10-python official archive"
        else:
            if not dataset_revision:
                raise ScenarioUnavailable("ImageNette requires --dataset-revision (for example imagenette2-160)")
            train_data = datasets.ImageFolder(directory / "train", transform=transform)
            test_data = datasets.ImageFolder(directory / "val", transform=transform)
            source, license_name, revision = "https://github.com/fastai/imagenette", "ImageNet source image terms; verify intended reuse", dataset_revision
    except (FileNotFoundError, RuntimeError) as error:
        raise ScenarioUnavailable(f"Supply local {dataset} files; this runner never downloads data") from error
    if train_data.class_to_idx != test_data.class_to_idx or len(train_data.classes) != 10:
        raise ValueError("CV archives must have the same ten classes and label mapping")
    from sklearn.model_selection import train_test_split
    train_ids, test_ids = np.arange(len(train_data)), np.arange(len(test_data))
    if max_samples is not None:
        requested_train = min(len(train_ids), int(max_samples * .8))
        requested_test = min(len(test_ids), max_samples - requested_train)
        if requested_train < len(train_ids):
            train_ids, _ = train_test_split(train_ids, train_size=requested_train, random_state=seed,
                                            stratify=np.asarray(train_data.targets))
        if requested_test < len(test_ids):
            test_ids, _ = train_test_split(test_ids, train_size=requested_test, random_state=seed + 1,
                                           stratify=np.asarray(test_data.targets))
    labels = np.asarray(train_data.targets)
    train, rest = train_test_split(train_ids, test_size=.25, random_state=seed + 2, stratify=labels[train_ids])
    validation, calibration = train_test_split(rest, test_size=.4, random_state=seed + 3, stratify=labels[rest])
    tensor_bytes = (len(train_ids) + len(test_ids)) * (3 * resolution * resolution * 4 + 8)
    if storage_dir is None and 2 * tensor_bytes > max_materialized_bytes:
        raise ScenarioUnavailable("CV tensors and owned copies exceed the materialization limit; supply storage_dir for file-backed NPY splits")
    storage = None
    if storage_dir is not None:
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        storage = Path(tempfile.mkdtemp(prefix=f"{dataset}-s{seed}-", dir=storage_dir))
    def materialize(data, index, prefix, role):
        shape = (len(index), 3, resolution, resolution)
        x = np.lib.format.open_memmap(storage / f"{role}-x.npy", mode="w+", dtype=np.float32, shape=shape) if storage else np.empty(shape, dtype=np.float32)
        y = np.lib.format.open_memmap(storage / f"{role}-y.npy", mode="w+", dtype=np.int64, shape=(len(index),)) if storage else np.empty(len(index), dtype=np.int64)
        for row, sample_index in enumerate(index):
            image, label = data[int(sample_index)]
            if tuple(image.shape) != shape[1:] or image.dtype != torch.float32:
                raise ScenarioUnavailable("CV preprocessing must yield finite float32 RGB images with the declared shape")
            x[row], y[row] = image.numpy(), label
        ids = tuple(f"{prefix}:{int(i)}" for i in index)
        if storage:
            x.flush(); y.flush()
            del x, y
            return TensorSplit.from_npy(storage / f"{role}-x.npy", storage / f"{role}-y.npy", ids)
        return TensorSplit(torch.from_numpy(x), torch.from_numpy(y), ids)
    splits = {role: materialize(train_data, index, f"{dataset}:official_train", role) for role, index in
              dict(train=train, validation=validation, calibration=calibration).items()}
    splits["test"] = materialize(test_data, test_ids, f"{dataset}:official_test", "test")
    constructor = models.resnet18 if architecture == "resnet18" else models.resnet50
    def factory():
        model = constructor(weights=None, num_classes=10)
        if dataset == "cifar10":
            model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        return model
    return ExperimentBundle(**splits, task="classification", original_model=_model(factory, seed),
                            metadata={"scenario": f"{dataset}_{architecture}", "dataset": dataset, "dataset_revision": revision,
                                      "source": source, "license": license_name, "architecture": architecture,
                                      "resolution": [resolution, resolution], "classes": train_data.class_to_idx,
                                      "max_samples": max_samples, "subset_status": "pilot_subset" if max_samples else "full_dataset",
                                      "raw_source_path": str(directory),
                                      "tensor_storage": "npy_copy_on_write" if storage else "owned_in_memory",
                                      "tensor_bytes": tensor_bytes, "materialization_limit_bytes": max_materialized_bytes,
                                      "memory_scope": "one image loading workspace with NPY; OS page cache/resident pages and model execution not capped" if storage else "preallocated arrays with TensorSplit owned copy; no image list/stack",
                                      "preprocessing": "deterministic resize and [0,1] conversion; no fitted statistics",
                                      "initial_state": "from scratch including fc/conv1; runner must train every changed layer"})


def build_tabular(seed: int = 42, *, features: np.ndarray | None = None,
                  labels: np.ndarray | None = None) -> ExperimentBundle:
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    x = data.data if features is None else np.asarray(features)
    y = data.target if labels is None else np.asarray(labels)
    if len(x) != len(y) or y.ndim != 1 or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Tabular features and integer labels must describe the same samples")
    indices = split_indices(y, seed)
    preprocessor = TrainPreprocessor.fit(x[indices["train"]])
    splits = _splits(preprocessor.transform(x), y.astype(np.int64), indices, "tabular")
    return ExperimentBundle(**splits, task="classification",
                            original_model=_model(lambda: BigMLP(x.shape[1], int(y.max()) + 1), seed),
                            metadata={"scenario": "tabular_breast_cancer" if features is None else "tabular_supplied",
                                      "dataset": "sklearn.load_breast_cancer" if features is None else "caller_supplied",
                                      "source": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic" if features is None else "caller_supplied",
                                      "license": "CC BY 4.0" if features is None else "caller_must_supply",
                                      "classes": ["malignant", "benign"] if features is None else list(range(int(y.max()) + 1)),
                                      "data_sha256": _digest(x), "target_sha256": _digest(y),
                                      "preprocessing": preprocessor.metadata(),
                                      "preprocessing_fit_ids": list(splits["train"].ids),
                                      "initial_state": "untrained; shared runner must train baseline"})


def read_ts_regression(path: Path, *, return_intervals=False):
    """Read numeric, equal-length regression .ts and preserve real timestamps."""
    rows, targets, intervals = [], [], []
    in_data = False
    timestamped = False
    for number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not in_data:
            if line.lower().startswith("@timestamps"):
                timestamped = line.lower() == "@timestamps true"
            in_data = line.lower() == "@data"
            continue
        # A timestamp itself contains colons, so split only outside parentheses.
        fields, depth, begin = [], 0, 0
        for position, char in enumerate(line):
            depth += (char == "(") - (char == ")")
            if depth < 0:
                raise ValueError(f"Unbalanced .ts timestamp tuple at row {number}")
            if char == ":" and depth == 0:
                fields.append(line[begin:position])
                begin = position + 1
        fields.append(line[begin:])
        if depth:
            raise ValueError(f"Unbalanced .ts timestamp tuple at row {number}")
        try:
            if timestamped:
                channels, channel_times = [], []
                for channel in fields[:-1]:
                    pairs = re.findall(r"\(([^,()]+),([^()]+)\)", channel)
                    if not pairs or re.sub(r"\([^()]+\)", "", channel).strip(","):
                        raise ValueError("Malformed timestamp/value tuples")
                    times = [int(datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc).timestamp())
                             if not stamp.lstrip("-+").isdigit() else int(stamp) for stamp, _ in pairs]
                    if any(b <= a for a, b in zip(times, times[1:])):
                        raise ValueError("Timestamps must increase inside a series")
                    channel_times.append(times)
                    channels.append([float(v) if v != "?" else float("nan") for _, v in pairs])
                if any(times != channel_times[0] for times in channel_times[1:]):
                    raise ValueError("Series channels have different timestamps")
                intervals.append((channel_times[0][0], channel_times[0][-1]))
            else:
                channels = [[float(v) if v != "?" else float("nan") for v in c.split(",")] for c in fields[:-1]]
            target = float(fields[-1])
        except ValueError as error:
            raise ValueError(f"Invalid numeric .ts data at row {number}") from error
        if not channels or len({len(c) for c in channels}) != 1:
            raise ValueError(f"Unequal channel lengths at row {number}")
        rows.append(channels)
        targets.append(target)
    if not rows or len({(len(r), len(r[0])) for r in rows}) != 1:
        raise ValueError("The .ts dataset is empty or has unequal sample shapes")
    result = (np.asarray(rows, dtype=np.float32), np.asarray(targets, dtype=np.float32)[:, None])
    return (*result, tuple(intervals)) if return_intervals else result


def build_ts_regression(dataset_dir: Path, seed: int = 42) -> ExperimentBundle:
    """Use supplied archive train/test roles; never claim chronological isolation."""
    directory = Path(dataset_dir)
    train_path, test_path = directory / "AppliancesEnergy_TRAIN.ts", directory / "AppliancesEnergy_TEST.ts"
    if not train_path.is_file() or not test_path.is_file():
        raise ScenarioUnavailable("Supply the AppliancesEnergy archive TRAIN.ts and TEST.ts files")
    x_train, y_train, train_intervals = read_ts_regression(train_path, return_intervals=True)
    x_test, y_test, test_intervals = read_ts_regression(test_path, return_intervals=True)
    if x_train.shape[1:] != x_test.shape[1:]:
        raise ValueError("Training and test series shapes differ")
    # Holdouts are taken ONLY from the archive TRAIN file, with original TEST retained.
    from sklearn.model_selection import train_test_split
    train, remainder = train_test_split(np.arange(len(y_train)), test_size=.25, random_state=seed)
    validation, calibration = train_test_split(remainder, test_size=.4, random_state=seed + 1)
    channels = x_train.shape[1]
    preprocessor = TrainPreprocessor.fit(x_train[train].transpose(0, 2, 1).reshape(-1, channels))
    def transform(x):
        return preprocessor.transform(x.transpose(0, 2, 1).reshape(-1, channels)).reshape(len(x), -1, channels).transpose(0, 2, 1).copy()
    splits = _splits(transform(x_train), y_train, dict(train=train, validation=validation, calibration=calibration), "AppliancesEnergy:archive_train")
    if train_intervals:
        for role, index in dict(train=train, validation=validation, calibration=calibration).items():
            old = splits[role]
            splits[role] = TensorSplit(old.x, old.y, old.ids, intervals=tuple(train_intervals[i] for i in index))
    splits["test"] = TensorSplit(torch.from_numpy(transform(x_test)), torch.from_numpy(y_test),
                                  tuple(f"AppliancesEnergy:archive_test:{i}" for i in range(len(y_test))), intervals=test_intervals)
    target_mean, target_scale = float(y_train[train].mean()), max(float(y_train[train].std()), 1e-8)
    return ExperimentBundle(**splits, task="regression", original_model=_model(lambda: ScaledRegressor(SeriesCNN(channels, 1), target_mean, target_scale), seed),
                            metadata={"scenario": "ts_appliances_regression", "dataset": "AppliancesEnergy .ts archive",
                                      "source": "https://www.timeseriesregression.org/", "license": "verify archive terms before publication",
                                      "file_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (train_path, test_path)},
                                      "raw_source_path": str(directory), "preprocessing_fit_ids": list(splits["train"].ids),
                                      "preprocessing": preprocessor.metadata(), "target_mean": target_mean,
                                      "target_scale": target_scale, "target_units": "kWh (derived archive header; raw UCI Appliances is Wh)",
                                      "split_assumption": "archive TRAIN/TEST roles; timestamp spans checked for raw overlap; chronological role ordering not claimed",
                                      "forecast_horizon": None, "evidence_scope": "regression of aggregate energy; not future forecasting",
                                      "initial_state": "untrained; shared runner must train baseline"})


def build_ts_classification(dataset_dir: Path, seed: int = 42) -> ExperimentBundle:
    directory = Path(dataset_dir)
    train_path, test_path = directory / "CinCECGTorso_TRAIN.tsv", directory / "CinCECGTorso_TEST.tsv"
    if not train_path.is_file() or not test_path.is_file():
        raise ScenarioUnavailable("Supply CinCECGTorso TRAIN.tsv and TEST.tsv files")
    raw_train, raw_test = np.loadtxt(train_path), np.loadtxt(test_path)
    classes = sorted(np.unique(raw_train[:, 0]).tolist())
    mapping = {label: i for i, label in enumerate(classes)}
    if not set(raw_test[:, 0]).issubset(mapping):
        raise ValueError("Test contains classes absent from the training archive")
    y_train = np.asarray([mapping[v] for v in raw_train[:, 0]], dtype=np.int64)
    y_test = np.asarray([mapping[v] for v in raw_test[:, 0]], dtype=np.int64)
    from sklearn.model_selection import train_test_split
    train, remainder = train_test_split(np.arange(len(y_train)), test_size=.4, random_state=seed, stratify=y_train)
    validation, calibration = train_test_split(remainder, test_size=.5, random_state=seed + 1, stratify=y_train[remainder])
    preprocessor = TrainPreprocessor.fit(raw_train[train, 1:])
    splits = _splits(preprocessor.transform(raw_train[:, 1:])[:, None], y_train,
                     dict(train=train, validation=validation, calibration=calibration), "CinCECGTorso:archive_train")
    splits["test"] = TensorSplit(torch.from_numpy(preprocessor.transform(raw_test[:, 1:])[:, None]), torch.from_numpy(y_test),
                                  tuple(f"CinCECGTorso:archive_test:{i}" for i in range(len(y_test))))
    return ExperimentBundle(**splits, task="classification", original_model=_model(lambda: SeriesCNN(1, len(classes)), seed),
                            metadata={"scenario": "ts_cincecgtorso_classification", "classes": classes,
                                      "file_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (train_path, test_path)},
                                      "raw_source_path": str(directory), "preprocessing_fit_ids": list(splits["train"].ids),
                                      "preprocessing": preprocessor.metadata(), "dataset": "CinCECGTorso UCR archive",
                                      "source": "https://www.timeseriesclassification.com/description.php?Dataset=CinCECGTorso",
                                      "split_assumption": "archive sample isolation; patient isolation cannot be verified from TSV",
                                      "initial_state": "untrained; shared runner must train baseline"})


def chronological_windows(values: np.ndarray, *, context: int, horizon: int,
                           boundaries: Sequence[int] | None = None) -> dict[str, tuple[np.ndarray, np.ndarray, tuple[tuple[int, int], ...]]]:
    """Construct windows inside each role; intervals are half-open raw row spans."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or not np.isfinite(values).all() or context < 1 or horizon < 1:
        raise ValueError("Forecasting needs finite rows and positive context/horizon")
    n = len(values)
    cuts = tuple(boundaries or (0, int(n * .6), int(n * .75), int(n * .85), n))
    if len(cuts) != 5 or cuts[0] != 0 or cuts[-1] != n or any(b <= a for a, b in zip(cuts, cuts[1:])):
        raise ValueError("Forecasting requires four ordered, nonempty role intervals")
    result = {}
    for role, left, right in zip(ROLES, cuts, cuts[1:]):
        spans = tuple((start, start + context + horizon) for start in range(left, right - context - horizon + 1))
        if not spans:
            raise ValueError(f"The {role} interval is too short for context + horizon")
        x = np.stack([values[a:a + context].T for a, _ in spans])
        y = np.stack([values[a + context:b, 0] for a, b in spans])
        result[role] = (x, y, spans)
    return result


def build_forecasting(csv_path: Path, *, context: int = 24, horizon: int = 6, seed: int = 42) -> ExperimentBundle:
    import pandas as pd
    path = Path(csv_path)
    if not path.is_file():
        raise ScenarioUnavailable("Supply the original UCI energydata_complete.csv; .ts regression files are not forecasting data")
    data = pd.read_csv(path)
    if not {"date", "Appliances"}.issubset(data.columns):
        raise ValueError("Forecasting CSV must contain date and Appliances")
    dates = pd.to_datetime(data["date"], errors="raise")
    if not dates.is_monotonic_increasing or dates.duplicated().any() or not (dates.diff().dropna() == pd.Timedelta(minutes=10)).all():
        raise ValueError("Forecasting requires unique, ordered, consecutive 10-minute timestamps")
    values = data[["Appliances"]].to_numpy(dtype=np.float32)
    windows = chronological_windows(values, context=context, horizon=horizon)
    train_rows = int(len(values) * .6)
    preprocessor = TrainPreprocessor.fit(values[:train_rows])
    splits = {}
    for role, (x, y, spans) in windows.items():
        normalized = preprocessor.transform(x.transpose(0, 2, 1).reshape(-1, 1)).reshape(len(x), context, 1).transpose(0, 2, 1).copy()
        splits[role] = TensorSplit(torch.from_numpy(normalized), torch.from_numpy(y),
                                  tuple(f"UCI374:{a}:{b}" for a, b in spans), intervals=tuple((a, b - 1) for a, b in spans))
    return ExperimentBundle(**splits, task="forecasting", original_model=_model(lambda: ScaledRegressor(SeriesCNN(1, horizon), float(preprocessor.mean[0]), float(preprocessor.scale[0])), seed),
                            metadata={"scenario": "uci_appliances_forecasting", "source": "https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction",
                                      "license": "CC BY 4.0", "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                      "context": context, "horizon": horizon, "time_step_minutes": 10, "target_units": "Wh",
                                      "first_timestamp": dates.iloc[0].isoformat(), "last_timestamp": dates.iloc[-1].isoformat(),
                                      "raw_source_path": str(path), "preprocessing_fit_ids": list(splits["train"].ids),
                                      "preprocessing_raw_rows": [0, train_rows], "preprocessing": preprocessor.metadata(),
                                      "initial_state": "untrained; shared runner must train baseline"})


def client_split_indices(client_ids: Sequence[str], seed: int = 42) -> dict[str, np.ndarray]:
    """Assign complete clients to roles before any fitted preprocessing."""
    units = np.asarray(client_ids, dtype=str)
    unique = np.unique(units)
    groups = split_indices(np.arange(len(unique)), seed, stratify=False)
    return {role: np.flatnonzero(np.isin(units, unique[index])) for role, index in groups.items()}


def build_financial(npz_path: Path, access_manifest: Path, seed: int = 42) -> ExperimentBundle:
    """Accessible-data route; it does not fabricate Alpha/Age or CoLES results."""
    manifest = json.loads(Path(access_manifest).read_text(encoding="utf-8"))
    required = ("source", "revision", "license", "access_authorized", "dataset_sha256", "task", "target_definition")
    if any(key not in manifest for key in required) or manifest["access_authorized"] is not True:
        raise ScenarioUnavailable("A source/revision/license/authorized-access manifest is required for financial data")
    path = Path(npz_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("Financial dataset does not match its access manifest checksum")
    if manifest["task"] != "classification":
        raise ScenarioUnavailable("The financial route currently supports client classification only")
    with np.load(path, allow_pickle=False) as data:
        x, y, clients = data["x"], data["y"], data["client_ids"].astype(str)
    if x.ndim != 3 or len(x) != len(y) or len(clients) != len(y) or y.ndim != 1:
        raise ValueError("Financial NPZ requires x[N,T,F], integer y[N], and client_ids[N]")
    if not np.issubdtype(y.dtype, np.integer) or y.min() < 0:
        raise ValueError("Financial labels must be nonnegative integers")
    indices = client_split_indices(clients, seed)
    preprocessor = TrainPreprocessor.fit(x[indices["train"]].reshape(-1, x.shape[-1]))
    x = preprocessor.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).transpose(0, 2, 1).copy()
    splits = _splits(x, y.astype(np.int64), indices, "financial", units=clients)
    return ExperimentBundle(**splits, task="classification", original_model=_model(lambda: SeriesCNN(x.shape[1], int(y.max()) + 1), seed),
                            metadata={**manifest, "scenario": "authorized_financial_sequence", "preprocessing": preprocessor.metadata(),
                                      "raw_source_path": str(path), "preprocessing_fit_ids": list(splits["train"].ids),
                                      "encoder": "supervised SeriesCNN; not CoLES", "historical_tables_restored": False,
                                      "initial_state": "untrained; shared runner must train baseline"})


def build_open_sequence(seed: int = 42) -> ExperimentBundle:
    """Open independent sequence task: 8 image rows as 8 ordered timesteps."""
    from sklearn.datasets import load_digits
    data = load_digits()
    x, y = data.images.transpose(0, 2, 1).astype(np.float32) / 16., data.target.astype(np.int64)
    indices = split_indices(y, seed)
    # The open task has independent images, not bank customers or handwriting-author IDs.
    splits = _splits(x, y, indices, "open-digits-sequence", units=[f"image:{i}" for i in range(len(y))])
    return ExperimentBundle(**splits, task="classification", original_model=_model(lambda: SeriesCNN(8, 10), seed),
                            metadata={"scenario": "open_digits_row_sequence", "dataset": "sklearn.load_digits",
                                      "source": DIGITS_SOURCE, "license": "CC BY 4.0", "data_sha256": _digest(data.data),
                                      "sequence_definition": "8 ordered rows of an 8x8 digit image; 8 features per step",
                                      "unit_definition": "independent image; author IDs are unavailable",
                                      "encoder": "supervised SeriesCNN; not CoLES", "historical_tables_restored": False,
                                      "evidence_scope": "separate open sequence classification, no financial conclusion",
                                      "initial_state": "untrained; shared runner must train baseline"})


class ByteLanguageModel(nn.Module):
    """Causal GRU with a fixed UTF-8 vocabulary; never attends to future tokens."""
    def __init__(self, vocabulary: int = 258):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, 32, padding_idx=0)
        self.recurrent = nn.GRU(32, 64, batch_first=True)
        self.output = nn.Linear(64, vocabulary)

    def forward(self, x):
        hidden, _ = self.recurrent(self.embedding(x))
        return self.output(hidden)


def causal_token_nll(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> dict[str, float | int]:
    """Token-weighted next-token NLL/PPL, excluding shifted padding labels."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or logits.shape[1] < 2:
        raise ValueError("Causal logits must be [batch,time,vocabulary] with matching labels")
    target = labels[:, 1:].reshape(-1).long()
    count = int((target != ignore_index).sum())
    if count == 0:
        raise ValueError("No valid next-token targets remain after shifting and padding masking")
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), target, ignore_index=ignore_index, reduction="sum")
    nll = float(loss.detach().double() / count)
    if not math.isfinite(nll) or nll >= 709:
        raise ValueError("Causal NLL or perplexity is nonfinite; the metric cannot be published")
    return {"nll": nll, "perplexity": math.exp(nll), "token_count": count}


def build_language_model(documents_path: Path, seed: int = 42, *, sequence_length: int = 64) -> ExperimentBundle:
    manifest = json.loads(Path(documents_path).read_text(encoding="utf-8"))
    if any(not manifest.get(key) for key in ("source", "revision", "license", "documents")):
        raise ScenarioUnavailable("Language corpus requires source, revision, license, and documents")
    documents = manifest["documents"]
    ids = [str(doc["id"]) for doc in documents]
    hashes = [hashlib.sha256(doc["text"].encode("utf-8")).hexdigest() for doc in documents]
    if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate documents or IDs would leak across language-model roles")
    if sequence_length < 2:
        raise ValueError("sequence_length must allow at least one next-token target")
    indices = split_indices(np.arange(len(documents)), seed, stratify=False)
    splits = {}
    for role, index in indices.items():
        rows, labels, chunk_ids, unit_ids = [], [], [], []
        for i in index:
            tokens = [1] + [int(byte) + 2 for byte in documents[i]["text"].encode("utf-8")]
            for chunk, start in enumerate(range(0, len(tokens) - 1, sequence_length)):
                part = tokens[start:start + sequence_length]
                if len(part) < 2:
                    continue
                rows.append(part + [0] * (sequence_length - len(part)))
                labels.append(part + [-100] * (sequence_length - len(part)))
                chunk_ids.append(f"{ids[i]}:chunk:{chunk}")
                unit_ids.append(ids[i])
        if not rows:
            raise ValueError(f"The {role} role has no document with two tokens")
        splits[role] = TensorSplit(torch.tensor(rows, dtype=torch.long), torch.tensor(labels, dtype=torch.long), tuple(chunk_ids), unit_ids=tuple(unit_ids))
    return ExperimentBundle(**splits, task="language_model", original_model=_model(ByteLanguageModel, seed),
                            metadata={"scenario": "byte_causal_language_model", "source": manifest["source"],
                                      "revision": manifest["revision"], "license": manifest["license"],
                                      "corpus_sha256": hashlib.sha256(Path(documents_path).read_bytes()).hexdigest(),
                                      "model_revision": "ByteLanguageModel-v1", "tokenizer_revision": "utf8-byte-v1",
                                      "vocabulary_size": 258, "padding_token": 0, "padding_label": -100,
                                      "sequence_length": sequence_length, "metric": "next-token token-weighted NLL/PPL",
                                      "documents_split_before_tokenization": True, "generation_status": "separate diagnostic; no quality evidence",
                                      "initial_state": "untrained; shared runner must train baseline"})
