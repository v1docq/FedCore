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
from torchvision.models import resnet18

from fedcore.algorithm.quantization.quantizers import BaseQuantizer
from fedcore.algorithm.quantization.utils import QDQWrapper, QDQWrapping
from fedcore.data.data import CompressionInputData
from fedcore.models.backbone.convolutional.resnet import CLF_MODELS

from fedcore.api.config_factory import ConfigFactory
from fedcore.api.api_configs import (APIConfigTemplate, AutoMLConfigTemplate, FedotConfigTemplate,
                                     LearningConfigTemplate,
                                       ModelArchitectureConfigTemplate,
                                     TrainingTemplate, 
                                       DeviceConfigTemplate, ComputeConfigTemplate,
                                     QuantizationTemplate)

from fedcore.api.main import FedCore


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



class Cifar10Data:
    """Загрузка CIFAR-10 и подготовка train/test/calibration DataLoader."""

    def __init__(self, config: Config, one_hot: bool = True):
        """
        Args:
            config: Configuration parameters
            one_hot: If True, returns one-hot encoded labels instead of class indices
        """
        self.one_hot = one_hot
        self.num_classes = 10
        
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
            collate_fn=self._collate_one_hot if one_hot else None,
        )
        self.calibration = DataLoader(
            Subset(train, train_ids[:n_calib]),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self._collate_one_hot if one_hot else None,
        )
        self.test = DataLoader(
            Subset(test, test_ids[:n_test]),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self._collate_one_hot if one_hot else None,
        )
    
    def _collate_one_hot(self, batch):
        """Custom collate function to one-hot encode labels."""
        data = torch.stack([item[0] for item in batch])
        labels = torch.stack([
            torch.nn.functional.one_hot(torch.tensor(item[1]), num_classes=10).float()
            for item in batch
        ])
        return data, labels


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


    data = Cifar10Data(config)
    print(
        f"train={len(data.train.dataset)}, test={len(data.test.dataset)}, "
        f"calibration={len(data.calibration.dataset)}"
    )
    # Load pretrained ResNet18
    model = resnet18(pretrained=True)

    # Modify the final fully connected layer for CIFAR-10 (10 classes)
    model.fc = nn.Linear(model.fc.in_features, 10)

    # Optional: Modify the first conv layer for CIFAR-10 (32x32 images)
    # ResNet expects 224x224 by default, but CIFAR is 32x32
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool to keep dimensions

    example_batch = next(iter(data.train))
    example_input = example_batch[0]  
    compression_data = CompressionInputData(
        features=example_input,  
        target=model, 
        train_dataloader=data.train,
        val_dataloader=data.calibration,
        test_dataloader=data.test,
        task=Task(TaskTypesEnum.classification),  
        input_dim=example_input.size(-1),
    )

    fedot_config = FedotConfigTemplate(
        problem='classification',
        metric= [
            'BinaryAccuracy',
                'Latency', 
                'ModelSize'
                ],
        pop_size=1,
        timeout=0.1,
        initial_assumption=model
    )

    peft_config = QuantizationTemplate(quant_type='static', # 'static', 'dynamic', 'qat'
                                allow_emb=False,
                                allow_conv=True,
                                    )

    automl_config = AutoMLConfigTemplate(fedot_config=fedot_config)

    learning_config = LearningConfigTemplate(criterion='cross_entropy',
                                            learning_strategy='from_checkpoint',                                          
                                            peft_strategy_params=[peft_config])

    api_template = APIConfigTemplate(automl_config=automl_config,
                                    learning_config=learning_config)
    
    APIConfig = ConfigFactory.from_template(api_template)
    api_config = APIConfig()
    fedcore_compressor = FedCore(api_config)
    fedcore_compressor.fit(compression_data)
    model_comparison = fedcore_compressor.get_report(compression_data)
    print(model_comparison)
    save_path = (REPO_ROOT / 'results' / 'quantization_resnet18/')
    save_path.mkdir(parents=True, exist_ok=True)
    model_comparison.to_csv(save_path / 'metrics.csv')




if __name__ == "__main__":
    main()
