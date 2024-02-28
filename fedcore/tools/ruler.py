import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fedcore.architecture.comptutaional.devices import default_device
from fedcore.architecture.dataset.dummy_clf import DummyDatasetCLF
from fedcore.inference.onnx import ONNXInferenceModel
from fedcore.metrics.cv_metrics import MetricCounter, ClassificationMetricCounter
from fedcore.models.backbone.resnet import ResNet


class PerformanceEvaluator:
    def __init__(self, model, dataset, device=None, batch_size=32):
        self.model = model.model if hasattr(model, 'model') else model
        self.dataset = dataset
        self.batch_size = batch_size
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.device = default_device() if not device else device
        self.model.to(self.device)

        # Measured performance metrics
        self.latency = None
        self.throughput = None
        self.model_size = None
        self.target_metrics = None

    def eval(self):

        result = dict(latency=self.measure_latency(),
                      throughput=self.measure_throughput(),
                      model_size=self.measure_model_size(),
                      target_metrics=self.measure_target_metric())
        self.report()
        return result

    def measure_latency(self, reps: int = 50):
        timings = np.zeros((reps, 1))
        if torch.cuda.is_available():
            self.warm_up_cuda()
        with torch.no_grad():
            with tqdm(total=reps, desc='Measuring latency', unit='rep') as pbar:
                for rep in range(reps):
                    for inputs, _ in self.data_loader:
                        start_time = time.time()
                        _ = self.model(inputs.to(self.device))
                        end_time = time.time()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        curr_time = (end_time - start_time) * 1000
                        timings[rep] = curr_time / inputs.size(0)
                        break
                    pbar.update(1)
        self.latency = round(np.mean(timings) / reps, 5)
        return self.latency

    def measure_throughput(self, batches: int = 5):
        total_data_size = 0
        start_time = time.time()
        # measure for n batches
        with torch.no_grad():
            with tqdm(total=batches, desc='Measuring throughput', unit='batch') as pbar:
                for inputs, _ in self.data_loader:
                    inputs = inputs.to(self.device)
                    if batches == 0:
                        break
                    total_data_size += inputs.size(0)
                    _ = self.model(inputs)
                    batches -= 1
                    pbar.update(1)
        if self.device == 'cuda':
            torch.cuda.synchronize()
        total_time = (time.time() - start_time) / 1000
        self.throughput = round(total_data_size / total_time, 0)
        return self.throughput

    def measure_target_metric(self, metric_counter: MetricCounter = None):
        if not metric_counter:
            metric_counter = ClassificationMetricCounter()
        with torch.no_grad():
            with tqdm(desc='Measuring throughput', unit='batch') as pbar:
                for inputs, labels in self.data_loader:
                    inputs = inputs.to(self.device)
                    prediction = self.model(inputs)
                    if len(prediction.size()) > 2:
                        prediction = prediction[0]
                    metric_counter.update(prediction.cpu(), labels.cpu())
                    pbar.update(1)
        if self.device == 'cuda':
            torch.cuda.synchronize()
        self.target_metrics = metric_counter.compute()
        return self.target_metrics

    def measure_model_size(self):
        if isinstance(self.model, ONNXInferenceModel):
            size_all_mb = round(self.model.size(), 3) / 1024 ** 2
        else:
            param_size = 0
            for param in self.model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in self.model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()

            size_all_mb = (param_size + buffer_size) / 1024 ** 2
        self.model_size = round(size_all_mb, 3)
        return self.model_size

    def warm_up_cuda(self, num_iterations=10):
        """Warm up CUDA by performing some dummy computations"""
        if torch.cuda.is_available():
            for _ in range(num_iterations):
                inputs, _ = next(iter(self.data_loader))
                inputs = inputs.to(self.device)
                _ = self.model(inputs)

    def report(self):
        print(f"Latency: {self.latency} ms/sample with batch_size {self.batch_size}")
        print(f"Throughput: {self.throughput} samples/s with batch_size {self.batch_size}")
        print(f"Model size: {self.model_size} MB")


if __name__ == "__main__":
    # Example usage:
    # load MNIST dataset

    # data_set = MNIST(
    #     root='data',
    #     train=True,
    #     download=True,
    #     transform=ToTensor()
    # )
    data_set = DummyDatasetCLF(num_samples=1000)
    # define model from dataset configuration
    resnet = ResNet(input_dim=1,
                    output_dim=len(data_set.classes),
                    model_name='ResNet18one')
    evaluator = PerformanceEvaluator(resnet, data_set)
    evaluator.measure_model_size()
    evaluator.measure_latency()
    evaluator.measure_throughput()

    # or
    # performance = evaluator.eval()
    # print(performance)
