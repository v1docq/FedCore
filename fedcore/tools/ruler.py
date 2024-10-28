import time

import numpy as np
import torch
import torch.utils
from fedot.core.pipelines.pipeline import Pipeline
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

# from fedcore.api.utils.data import DataLoaderHandler
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.inference.onnx import ONNXInferenceModel
from fedcore.metrics.metric_impl import (
    ClassificationMetricCounter,
    MetricCounter,
    ObjectDetectionMetricCounter,
)


class PerformanceEvaluator:
    def __init__(
        self,
        model,
        data,
        device=default_device(),
        batch_size=32,
        n_batches=8,
        collate_fn=None,
    ):
        is_class_container = hasattr(model, "model")
        is_pipeline_class = isinstance(model, Pipeline)
        dataset_from_directory = isinstance(
            data, str
        )  ### where's func for string dataset loading
        self.model = model.model if is_class_container else model
        self.model = (
            model.operator.root_node.fitted_operation.model
            if is_pipeline_class
            else model
        )
        self.n_batches = n_batches
        if isinstance(data, DataLoader):
            collate_fn = data.collate_fn
            dataset = data.dataset
        elif dataset_from_directory:
            pass  # TODO some logic for string dataset downloading
        else:
            dataset = data
        self.data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
        self.batch_size = batch_size or self.data_loader.batch_size
        self.device = device
        self.model.to(device)
        # Measured performance metrics
        self.latency = None
        self.throughput = None
        self.model_size = None
        self.target_metrics = None

    def _preloaded_batches_gen(self, max_num=float("inf"), data_loader=None):
        num = 0  # for cases when dataset has no __len__
        for i in data_loader or self.data_loader:
            yield i
            num += 1
            if num >= max_num:
                break

    def eval(self):
        self.warm_up_cuda()
        lat, thr = self.measure_latency_throughput()
        result = dict(latency=lat, throughput=thr, model_size=self.measure_model_size())
        self.report()
        return result

    @torch.no_grad()
    def throughput_eval(self, num_iterations=30):
        self.model.eval()
        thr_list = []
        for batch, _ in tqdm(
            self._preloaded_batches_gen(self.n_batches), desc="batches", unit="batch"
        ):
            X = (
                batch.cuda(non_blocking=True)
                if hasattr(batch, "cuda")
                else batch.to(self.device)
            )
            batch_size = len(X)
            torch.cuda.synchronize(self.device)
            tic1 = time.time()
            for i in range(num_iterations):
                self.model(X)
            torch.cuda.synchronize(self.device)
            tic2 = time.time()
            thr_list.append(num_iterations * batch_size / (tic2 - tic1))
        return thr_list

    @torch.no_grad()
    def latency_eval(self, max_samples=None):
        self.model.eval()
        lat_list = []
        for batch, _ in tqdm(
            self._preloaded_batches_gen(max_samples or self.batch_size)
        ):
            batch = (
                batch if hasattr(batch, "__iter__") else [batch]
            )  ### case batch is not iterable
            for sample in batch:
                sample = (
                    sample.cuda(non_blocking=True)
                    if hasattr(sample, "cuda")
                    else sample.to(self.device)
                )
                tic1 = time.time()
                self.model(sample)
                torch.cuda.synchronize()
                tic2 = time.time()
                lat_list.append((tic2 - tic1))
        return lat_list

    def measure_latency_throughput(self, reps: int = 3, batches: int = 10):
        timings_lat = []
        timings_thr = []
        with tqdm(
            total=reps, desc="Measuring latency and throughput", unit="rep"
        ) as pbar:
            for rep in range(reps):
                timings_thr.append(self.throughput_eval())
                timings_lat.append(self.latency_eval())
                pbar.update(1)

        latency = np.array([[np.mean(x), np.std(x)] for x in timings_lat])
        throughput = np.array([[np.mean(x), np.std(x)] for x in timings_thr])
        self.latency, self.throughput = np.mean(latency, axis=0), np.mean(
            throughput, axis=0
        )
        return self.latency, self.throughput

    def measure_model_size(self):
        size_constant = 1024**2
        if isinstance(self.model, ONNXInferenceModel):
            size_all_mb = round(self.model.size(), 3) / size_constant
        else:
            param_size = 0
            for param in self.model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in self.model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()

            size_all_mb = (param_size + buffer_size) / size_constant
        self.model_size = round(size_all_mb, 3)
        return self.model_size

    @torch.no_grad()
    def warm_up_cuda(self, n_batches=3):
        """Warm up CUDA by performing some dummy computations"""
        batch_sample = self._preloaded_batches_gen(n_batches)
        if torch.cuda.is_available():
            for inputs, _ in tqdm(batch_sample, desc="warming"):
                _ = self.model(inputs.to(self.device))

    def report(self):
        print(f"Latency: {self.latency} ms/sample with batch_size {self.batch_size}")
        print(
            f"Throughput: {self.throughput} samples/s with batch_size {self.batch_size}"
        )
        print(f"Model size: {self.model_size} MB")


class PerformanceEvaluatorOD:
    def __init__(self, model, data_loader, device=None, batch_size=32):
        self.model = model.model if hasattr(model, "model") else model
        self.data_loader = data_loader
        # self.dataset = dataset
        self.batch_size = batch_size
        # self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.device = default_device() if not device else device
        self.model.to(self.device)

        # Measured performance metrics
        self.latency = None
        self.throughput = None
        self.model_size = None
        self.target_metrics = None

    def eval(self):

        result = dict(
            latency=self.measure_latency(),
            throughput=self.measure_throughput(),
            model_size=self.measure_model_size(),
            target_metrics=self.measure_target_metric(),
        )
        self.report()
        return result

    def measure_latency(self, reps: int = 50):
        timings = np.zeros((reps, 1))
        if torch.cuda.is_available():
            self.warm_up_cuda()
        with torch.no_grad():
            with tqdm(total=reps, desc="Measuring latency", unit="rep") as pbar:
                for rep in range(reps):
                    for inputs, _ in self.data_loader:
                        start_time = time.time()
                        _ = self.model(list(input.to(self.device) for input in inputs))
                        end_time = time.time()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        curr_time = (end_time - start_time) * 1000
                        timings[rep] = curr_time / inputs[0].size(0)
                        break
                    pbar.update(1)
        self.latency = round(np.mean(timings) / reps, 5)
        return self.latency

    def measure_throughput(self, batches: int = 5):
        total_data_size = 0
        start_time = time.time()
        # measure for n batches
        with torch.no_grad():
            with tqdm(total=batches, desc="Measuring throughput", unit="batch") as pbar:
                for inputs, _ in self.data_loader:
                    inputs = list(input.to(self.device) for input in inputs)
                    if batches == 0:
                        break
                    total_data_size += inputs[0].size(0)
                    _ = self.model(inputs)
                    batches -= 1
                    pbar.update(1)
        if self.device == "cuda":
            torch.cuda.synchronize()
        total_time = (time.time() - start_time) / 1000
        self.throughput = round(total_data_size / total_time, 0)
        return self.throughput

    def measure_target_metric(self, metric_counter: MetricCounter = None):
        if not metric_counter:
            metric_counter = ObjectDetectionMetricCounter()
        with torch.no_grad():
            with tqdm(desc="Measuring target metric", unit="batch") as pbar:
                for batch in self.data_loader:
                    inputs, targets = batch
                    inputs = list(input.to(self.device) for input in inputs)
                    prediction = self.model(inputs)
                    metric_counter.update(prediction, targets)
                    pbar.update(1)
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.target_metrics = metric_counter.compute()
        return self.target_metrics

    def measure_model_size(self):
        if isinstance(self.model, ONNXInferenceModel):
            size_all_mb = round(self.model.size(), 3) / 1024**2
        else:
            param_size = 0
            for param in self.model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in self.model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()

            size_all_mb = (param_size + buffer_size) / 1024**2
        self.model_size = round(size_all_mb, 3)
        return self.model_size

    def warm_up_cuda(self, num_iterations=10):
        """Warm up CUDA by performing some dummy computations"""
        if torch.cuda.is_available():
            for _ in range(num_iterations):
                inputs, _ = next(iter(self.data_loader))
                inputs = list(input.to(self.device) for input in inputs)
                _ = self.model(inputs)

    def report(self):
        print(f"Latency: {self.latency} ms/sample with batch_size {self.batch_size}")
        print(
            f"Throughput: {self.throughput} samples/s with batch_size {self.batch_size}"
        )
        print(f"Model size: {self.model_size} MB")
