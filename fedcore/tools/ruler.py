import time
from typing import Union

import numpy as np
import torch
import torch.utils
from torchinfo import summary
from fedot.core.pipelines.pipeline import Pipeline
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from fedcore.architecture.comptutaional.devices import default_device
from fedcore.inference.onnx import ONNXInferenceModel
from fedcore.metrics.metric_impl import (
    ClassificationMetricCounter,
    MetricCounter,
    ObjectDetectionMetricCounter,
)
from functools import partial
from fedcore.api.utils.data import DataLoaderHandler
from fedcore.tools.edge_device import PowerEstimator
from time import time
from fedcore.tools.registry.model_registry import ModelRegistry


def format_num(num: int, bytes=False):
    """Scale bytes to its proper format, e.g. 1253656 => '1.20MB'"""
    factor = 1024 if bytes else 1000
    suffix = "B" if bytes else ""
    for unit in ["", " K", " M", " G", " T", " P"]:
        if num < factor:
            return f"{num:.2f}{unit}{suffix}"
        num /= factor


def format_time(seconds, return_time=False):
    """Defines how to format time in FunctionEvent"""
    US_IN_SECOND = 1000.0 * 1000.0
    US_IN_MS = 1000.0
    prefix = 'us'
    time_us = seconds * 1e6
    if time_us >= US_IN_SECOND:
        time_us = time_us / US_IN_SECOND
        prefix = 's'
    if time_us >= US_IN_MS:
        time_us = time_us / US_IN_MS
        prefix = 'ms'
    if return_time:
        return round(time_us,3)
    else:
        return f"{round(time_us,3)} {prefix}"


class PerformanceEvaluator:
    def __init__(
            self,
            model: callable,
            model_regime: str = 'model_after',
            data: Union[DataLoader, str] = None,
            device=None,
            batch_size=32,
            n_batches=8,
            collate_fn=None,
    ):
        self.model_regime = model_regime
        self.n_batches = n_batches
        self.batch_size = batch_size  # or self.data_loader.batch_size
        self._init_null_object()
        self._registry = ModelRegistry()

        self.device = device or default_device()
        self.cuda_allowed = True if self.device.type == 'cuda' else False
        self.transfer_to_device_fn = torch.Tensor.to
        self._init_model(model)

        dataset_from_directory = isinstance(data, str)  ### where's func for string dataset loading
        if isinstance(data, DataLoader):
            collate_fn = data.collate_fn
            dataset = data.dataset
        elif dataset_from_directory:
            pass  # TODO some logic for string dataset downloading
        else:
            dataset = data
        self.data_loader = partial(DataLoaderHandler.check_convert, dataloader=DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
                                   )

    def _init_null_object(self):
        # Measured performance metrics
        self.latency = None
        self.throughput = None
        self.model_size = None
        self.target_metrics = None

    def _init_model(self, model):
        is_class_container = hasattr(model, "model")
        is_pipeline_class = isinstance(model, Pipeline)

        if is_pipeline_class:
            fitted = model.operator.root_node.fitted_operation
            model_from_attr = getattr(fitted, self.model_regime, None)
            
            if model_from_attr is None:
                raise ValueError(f"Model regime '{self.model_regime}' not found in fitted operation")
            
            fedcore_id = getattr(fitted, '_fedcore_id', None)
            model_id = getattr(fitted, '_model_id', None)
            
            if fedcore_id and model_id:
                self.model = self._registry.get_model_with_fallback(
                    fedcore_id=fedcore_id,
                    model_id=model_id,
                    fallback_model=model_from_attr,
                    device=self.device
                )
            else:
                print("No fedcore_id or model_id found, using model from operation attributes")
                self.model = model_from_attr
            
            operation_device = getattr(fitted, 'device', None)
            if operation_device:
                self.device = operation_device
                self.cuda_allowed = (self.device.type == 'cuda')
                
        elif is_class_container:
            self.model = model.model
        else:
            self.model = model
            
        self.model.to(self.device)

    def eval(self):
        self.warm_up_cuda()
        # lat, thr = self.measure_latency_throughput(10, self.n_batches)
        # lat = self.eval_detailed_latency()
        lats = self.latency_eval()
        thrs = self.throughput_eval()
        result = dict(
            latency=[np.mean(lats), np.std(lats)],
            throughput=[np.mean(thrs), np.std(thrs)],
            model_size=[self.measure_model_size(), 0.],
        )
        # self.report()
        return result

    # def eval_detailed_latency(self, sample):
    #     with torch.autograd.profiler.profile(use_cuda=(self.device.type == "cuda"), profile_memory=True) as prof:
    #         self.transfer_to_device_fn(self.model(self.transfer_to_device_fn(sample, self.device)), "cpu")
    #
    #     detailed_timing = prof.key_averages().table(sort_by="self_cpu_time_total")
    #     return detailed_timing

    def eval_detailed_latency(self, num_runs=100):
        mean_latency = np.inf
        std_latency = np.inf
        t_cpu_2_gpu, t_device, t_gpu_2_cpu, t_total = [], [], [], []
        for batch in tqdm(self.data_loader(max_batches=self.n_batches), desc="batches", unit="batch"):
            sample = batch[0] if isinstance(batch, (tuple, list)) else batch
            break

        for _ in tqdm(range(num_runs), desc=f"Measuring inference for batch_size={self.batch_size}"):
            start_on_cpu = time()
            device_sample = self.transfer_to_device_fn(sample, self.device)

            if self.cuda_allowed:
                start_event = torch.cuda.Event(enable_timing=True)
                stop_event = torch.cuda.Event(enable_timing=True)
                start_event.record()  # For GPU timing
            start_on_device = time()  # For CPU timing

            device_result = self.model(device_sample)

            if self.cuda_allowed:
                stop_event.record()
                torch.cuda.synchronize()
                elapsed_on_device = stop_event.elapsed_time(start_event)
                stop_on_device = time()
            else:
                stop_on_device = time()
                elapsed_on_device = stop_on_device - start_on_device

            self.transfer_to_device_fn(device_result, "cpu")
            stop_on_cpu = time()

            t_cpu_2_gpu.append(start_on_device - start_on_cpu)
            t_device.append(elapsed_on_device)
            t_gpu_2_cpu.append(stop_on_cpu - stop_on_device)
            t_total.append(stop_on_cpu - start_on_cpu)

        results_dict = {}
        for _ in [t_gpu_2_cpu, t_cpu_2_gpu, t_device, t_total]:
            _.pop(0)  # delete first result in cycle, because of unstable
        times_and_titles = [(t_device, "on_host_inference")]
        if self.cuda_allowed:
            times_and_titles.extend([(t_cpu_2_gpu, "cpu_to_gpu"), (t_gpu_2_cpu, "gpu_to_cpu"), (t_total, "total")])

        for s_per_batch, title in times_and_titles:
            s_per_batch = np.array(s_per_batch)
            batches_per_s = 1 / s_per_batch

            metrics = {
                "batches_per_second_mean": float(batches_per_s.mean()),
                "batches_per_second_std": float(batches_per_s.std()),
                "batches_per_second_min": float(batches_per_s.min()),
                "batches_per_second_max": float(batches_per_s.max()),
                "seconds_per_batch_mean": float(s_per_batch.mean()),
                "seconds_per_batch_std": float(s_per_batch.std()),
                "seconds_per_batch_min": float(s_per_batch.min()),
                "seconds_per_batch_max": float(s_per_batch.max()),
            }

            convert_to_report = {
                "batches_per_second": f"{format_num(batches_per_s.mean())} "
                                      f"+/- {format_num(batches_per_s.std())} [{format_num(batches_per_s.min())}, "
                                      f"{format_num(batches_per_s.max())}]",

                "batch_latency": f"{format_time(s_per_batch.mean())}"
                                 f" +/- {format_time(s_per_batch.std())} [{format_time(s_per_batch.min())},"
                                 f" {format_time(s_per_batch.max())}]",
            }
            mean_latency = format_time(s_per_batch.mean(), return_time=True)
            std_latency = format_time(s_per_batch.std(), return_time=True)
            results_dict[title] = {"metrics": metrics, "convert_to_report": convert_to_report}

        return np.array([mean_latency, std_latency])

    def measure_energy(
            model,
            sample,
            model_device,
            transfer_to_device_fn=torch.Tensor.to,
            num_runs=100,
            batch_size: int = None,
            include_transfer_costs=True,
            print_fn=None
    ):
        def test_with_transfer():
            nonlocal model, sample
            transfer_to_device_fn(
                model(transfer_to_device_fn(sample, model_device)),
                "cpu",
            )

        def test_without_transfer():
            nonlocal model, sample
            model(sample)

        if include_transfer_costs:
            test_fn = test_with_transfer
        else:
            test_fn = test_without_transfer
            sample = sample.to(model_device)

        # Try jetson power
        try:
            p_est = PowerEstimator(print_fn=print_fn)
            # index 0 is total energy, index 1 is energy over idle consumption:
            meas = []
            for _ in tqdm(range(num_runs), desc=f"Measuring energy for batch_size={batch_size}"):
                meas.append(p_est.estimate_fn_power(test_fn)[0] / 1000)
            inference_joules = float(np.array(meas).mean())
        except Exception:
            pass

        return inference_joules

    @torch.no_grad()
    def throughput_eval(self, num_iterations=30):
        self.model.eval()
        thr_list = []
        steps_iter = range(num_iterations)
        for batch in tqdm(self.data_loader(max_batches=self.n_batches), desc="batches", unit="batch"):
            batch = batch[0] if isinstance(batch, (tuple, list)) else batch
            is_already_cuda = all([hasattr(batch, "cuda"), self.cuda_allowed])
            X = batch.cuda(non_blocking=True) if is_already_cuda else batch.to(self.device)
            if is_already_cuda:
                start_events = [torch.cuda.Event(enable_timing=True) for _ in steps_iter]
                end_events = [torch.cuda.Event(enable_timing=True) for _ in steps_iter]
                for i in steps_iter:
                    start_events[i].record()
                    self.model(X)
                    end_events[i].record()
                torch.cuda.synchronize(self.device)
                times = ([s.elapsed_time(e) for s, e in zip(start_events, end_events)])
            else:
                times = list()
                for i in steps_iter:
                    start_events = time()
                    self.model(X)
                    end_events = time()
                    times.append(end_events - start_events)
                times = (times)
            thr_list.extend(times)
        return len(batch) / np.array(thr_list)

    @torch.no_grad()
    def latency_eval(self, max_samples=None):
        def cuda_latency_eval(sample_batch):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            self.model(sample_batch)
            end_event.record()
            torch.cuda.synchronize(self.device)
            time = start_event.elapsed_time(end_event)
            return time
        
        def cpu_latency_eval(sample_batch):
            start_on_device = time()
            self.model(sample_batch)
            stop_on_device = time()
            elapsed_on_device = stop_on_device - start_on_device
            return elapsed_on_device

        self.model.eval()
        lat_list = []
        for batch in tqdm(self.data_loader(max_batches=max_samples or self.batch_size)):
            if isinstance(batch, tuple) or isinstance(batch, list):
                features = batch[0]
                if isinstance(features, torch.Tensor):
                    for sample in features:
                        is_already_cuda = all([hasattr(sample, "cuda"), self.cuda_allowed])
                        sample = sample.cuda(non_blocking=True) if is_already_cuda else sample.to(self.device)
                        sample.to(self.device)
                        sample_batch = sample[None, ...]
                        lat_list.append(cuda_latency_eval(sample_batch)) if is_already_cuda else \
                        lat_list.append(cpu_latency_eval(sample_batch))
            else:
                lat_list.append(cuda_latency_eval(batch))
        return np.array(lat_list)

    def measure_latency_throughput(self, reps: int = 3, batches: int = 10):
        timings_lat = []
        timings_thr = []
        with tqdm(
                total=reps, desc="Measuring latency and throughput", unit="rep"
        ) as pbar:
            for rep in range(reps):
                timings_thr.append(self.throughput_eval(reps))
                timings_lat.append(self.latency_eval())
                pbar.update(1)
        latency = np.array([[np.mean(x), np.std(x)] for x in timings_lat])
        throughput = np.array([[np.mean(x), np.std(x)] for x in timings_thr])
        self.latency, self.throughput = np.mean(np.array(latency), axis=0), np.mean(
            np.array(throughput), axis=0
        )
        return self.latency, self.throughput

    def measure_model_size(self):
        size = summary(self.model).total_param_bytes
        size_constant = 1 << 20
        size /= size_constant
        self.model_size = round(size, 3)
        return self.model_size
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
        if self.cuda_allowed:
            for batch in tqdm(self.data_loader(max_batches=n_batches), desc="warming"):
                if isinstance(batch, tuple) or isinstance(batch, list):
                    inputs = batch[0]
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
                inputs = list(input.to(self.device) for input in inputs)
                _ = self.model(inputs)

    def report(self):
        print(f"Latency: {self.latency} ms/sample with batch_size {self.batch_size}")
        print(
            f"Throughput: {self.throughput} samples/s with batch_size {self.batch_size}"
        )
        print(f"Model size: {self.model_size} MB")