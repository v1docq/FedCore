import logging
from typing import Optional

import torch
import torch.nn
import torchvision.datasets
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.tasks import Task, TaskTypesEnum
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from fedcore.api.main import FedCore
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.architecture.utils.paths import data_path
from fedcore.data.data import CompressionInputData
from fedcore.metrics.multi_objective import MultiobjectiveCompression
from fedcore.repository.initializer_industrial_models import FedcoreModels
from fedcore.tools.ruler import PerformanceEvaluator


class CompressionBenchmark:
    """Abstract class for benchmarks.

    This class defines the interface that all benchmarks must implement.
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        """Initialize the benchmark.

        Args:
            name: The name of the benchmark.
            description: A short description of the benchmark.
            **kwargs: Additional arguments that may be required by the
                benchmark.
        """
        self.repo = FedcoreModels().setup_repository()
        self.criterion = params.get("loss", nn.CrossEntropyLoss())
        self.optimizer = params.get("optimizer", optim.Adam)
        self.epochs = params.get("epochs", 2)
        self.learning_rate = params.get("lr", 0.001)
        self.benchmark_type = {
            "quantisation": self.quant_model,
            "pruning": self.quant_model,
        }
        self.optimisation_type = {"multi_objective": MultiobjectiveCompression()}
        self.fedcore_setup = params.get("fedcore_setup", None)

        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def _config(self):
        raise NotImplementedError()

    def _init_dataset(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        train_dataset = torchvision.datasets.CIFAR10(
            data_path("CIFAR10"), train=True, download=True, transform=transform
        )
        val_dataset = torchvision.datasets.CIFAR10(
            data_path("CIFAR10"), train=False, download=True, transform=transform
        )
        val_dataset, test_dataset = torch.utils.data.random_split(
            val_dataset, [0.1, 0.9]
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=64, shuffle=True, num_workers=1
        )

        val_dataloader = DataLoader(
            val_dataset, batch_size=100, shuffle=False, num_workers=1
        )
        return train_dataset, test_dataset, train_dataloader, val_dataloader

    def _init_model(self):
        model = resnet18(pretrained=True).to(default_device())
        model.fc = nn.Linear(512, 10).to(default_device())
        model.train()
        return model

    def run(self, algo_type: str = "quantisation", framework: str = "ONNX"):
        """Run the benchmark and return the results.

        Returns:
            A dictionary containing the results of the benchmark.
        """
        model = self._init_model()
        train_dataset, test_dataset, train_dataloader, val_dataloader = (
            self._init_dataset()
        )
        model, performance = self.evaluate_loop(model, train_dataloader, test_dataset)

        compressed_model, compression_performance = self.benchmark_type[algo_type](
            model,
            val_dataloader,
            train_dataloader,
            train_dataset,
            test_dataset,
            framework,
        )
        print("Before quantization")
        print(performance)
        print("After quantization")
        print(compression_performance)
        return compressed_model, compression_performance

    def run_optimisation(
        self, algo_type: str = "multi_objective", framework: str = "ONNX"
    ):
        """Run the benchmark and return the results.

        Returns:
            A dictionary containing the results of the benchmark.
        """
        model = self._init_model()
        train_dataset, test_dataset, train_dataloader, val_dataloader = (
            self._init_dataset()
        )
        model, performance = self.evaluate_loop(model, train_dataloader, val_dataloader)
        input_data = CompressionInputData(
            num_classes=10,
            calib_dataloader=val_dataloader,
            train_dataloader=train_dataloader,
            task=Task(TaskTypesEnum.classification),
            target=model,
        )
        compressed_model, compression_performance = self.optimisation_type[
            algo_type
        ].evaluate(input_data)
        print("Before optimisation")
        print(performance)
        print("After optimisation")
        print(compression_performance)
        return compressed_model, compression_performance

    def evaluate_loop(
        self, model, train_dataloader, test_dataset, train_mode: bool = True
    ):
        if not train_mode:
            performance = self.evaluate_perfomance(model, test_dataset)
            return model, performance
        else:
            self.optimizer = self.optimizer(model.parameters(), lr=self.learning_rate)
            for epoch in range(self.epochs):  # loop over the dataset multiple times
                running_loss = 0.0
                for i, data in enumerate(train_dataloader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = model(inputs.to(default_device()))
                    loss = self.criterion(outputs, labels.to(default_device()))
                    loss.backward()
                    self.optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    if i % 200 == 0:  # print every 2000 mini-batches
                        print(
                            "[%d, %5d] loss: %.3f"
                            % (epoch + 1, i + 1, running_loss / 200)
                        )
                        running_loss = 0.0
            performance = self.evaluate_perfomance(model, test_dataset)
            return model, performance

    def quant_model(
        self,
        model,
        val_dataloader,
        train_dataloader,
        train_dataset,
        test_dataset,
        framework,
    ):
        supplementary_data = {
            "torch_model": model.cpu(),
            "test_dataset": val_dataloader,
            "train_dataset": train_dataloader,
        }
        fedcore_compressor = FedCore(**self.fedcore_setup)
        input_data = fedcore_compressor.load_data(
            path=None, supplementary_data=supplementary_data
        )
        fedcore_compressor.fit(input_data)
        quant_model = fedcore_compressor.predict(input_data).predict
        convertation_supplementary_data = {
            "train_dataset": train_dataset,
            "model_to_export": quant_model,
        }
        onnx_model = fedcore_compressor.convert_model(
            framework=framework, supplementary_data=convertation_supplementary_data
        )

        performance = self.evaluate_perfomance(onnx_model, test_dataset)
        return onnx_model, performance

    def prune_model(self):
        pass

    def evaluate_perfomance(self, model, test_dataset):
        evaluator = PerformanceEvaluator(model, test_dataset, batch_size=64)
        performance = evaluator.eval()
        return performance
