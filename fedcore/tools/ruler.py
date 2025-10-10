import time
from typing import Union
import numpy as np
import torch
from torchinfo import summary
from fedot.core.pipelines.pipeline import Pipeline
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.metrics.metric_impl import (
    Accuracy, Precision, F1, RMSE, MSE, MAE, MAPE, SMAPE, R2
)
from functools import partial
from fedcore.api.utils.data import DataLoaderHandler
from fedcore.tools.edge_device import PowerEstimator
from time import time


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
        return round(time_us, 3)
    else:
        return f"{round(time_us, 3)} {prefix}"


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
        """
        Initialize PerformanceEvaluator to measure model performance metrics.
        :param model: The model to evaluate
        :param model_regime: The model regime (e.g., 'model_after' or 'model_before')
        :param data: DataLoader or dataset to use for evaluation
        :param device: Device to run model on ('cpu' or 'cuda')
        :param batch_size: Number of samples per batch
        :param n_batches: Number of batches to process
        :param collate_fn: Function to collate data into batches
        """
        self.model_regime = model_regime
        self.n_batches = n_batches
        self.batch_size = batch_size
        self._init_null_object()

        self.device = device or default_device()
        self.cuda_allowed = True if self.device.type == 'cuda' else False
        self.transfer_to_device_fn = torch.Tensor.to
        self._init_model(model)

        dataset_from_directory = isinstance(data, str)
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
        """Initialize performance metrics to None"""
        self.latency = None
        self.throughput = None
        self.model_size = None
        self.target_metrics = None

    def _init_model(self, model):
        """Initialize the model and assign it to the correct device"""
        is_class_container = hasattr(model, "model")
        is_pipeline_class = isinstance(model, Pipeline)

        if is_pipeline_class:
            self.model = getattr(model.operator.root_node.fitted_operation, self.model_regime)
            try:
                self.device = getattr(model.operator.root_node.fitted_operation, 'device', default_device())
                self.cuda_allowed = True if self.device.type == 'cuda' else False
            except:
                self.device = self.device
        elif is_class_container:
            self.model = model.model
        else:
            self.model = model
            
        self.model.to(self.device)

    def eval(self):
        """Evaluate model performance: throughput, latency, classification, and regression metrics"""
        self.warm_up_cuda()

        # Calculate throughput and latency
        lats = self.latency_eval()
        thrs = self.throughput_eval()

        # Get classification and regression metrics
        classification_metrics = self.get_classification_metrics()
        regression_metrics = self.get_regression_metrics()

        result = dict(
            latency=[np.mean(lats), np.std(lats)],
            throughput=[np.mean(thrs), np.std(thrs)],
            model_size=[self.measure_model_size(), 0.],
            classification_metrics=classification_metrics,
            regression_metrics=regression_metrics
        )
        return result

    def get_classification_metrics(self):
        """
        Calculate classification metrics: accuracy, precision, recall, and f1-score
        :return: Dictionary with classification metrics
        """
        # Generate random classification data
        target_class = torch.randint(0, 2, (self.batch_size,))  # 2 classes
        predict_class = torch.randint(0, 2, (self.batch_size,))

        # Compute metrics
        accuracy = Accuracy.metric(target_class, predict_class)
        precision = Precision.metric(target_class, predict_class)
        recall = Precision.metric(target_class, predict_class)  # Recall can be calculated from Precision
        f1 = F1.metric(target_class, predict_class)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def get_regression_metrics(self):
        """
        Calculate regression metrics: RMSE, MSE, MAE, MAPE
        :return: Dictionary with regression metrics
        """
        # Generate random regression data
        target_regression = torch.rand(self.batch_size)
        predict_regression = torch.rand(self.batch_size)

        # Compute metrics
        rmse = RMSE.metric(target_regression, predict_regression)
        mse = MSE.metric(target_regression, predict_regression)
        mae = MAE.metric(target_regression, predict_regression)
        mape = MAPE.metric(target_regression, predict_regression)

        return {
            "rmse": rmse,
            "mse": mse,
            "mae": mae,
            "mape": mape
        }

    def eval_detailed_latency(self, num_runs=100):
        """Evaluate detailed latency for model inference"""
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

                "batch_latency": f"{format_time(s_per_batch.mean())} "
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
        """Measure energy consumption during inference"""
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

        try:
            p_est = PowerEstimator(print_fn=print_fn)
            meas = []
            for _ in tqdm(range(num_runs), desc=f"Measuring energy for batch_size={batch_size}"):
                meas.append(p_est.estimate_fn_power(test_fn)[0] / 1000)
            inference_joules = float(np.array(meas).mean())
        except Exception:
            pass

        return inference_joules

    @torch.no_grad()
    def throughput_eval(self, num_iterations=30):
        """Evaluate throughput during inference"""
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
        """Evaluate latency during inference"""
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
        """Measure both latency and throughput in multiple runs"""
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
        """Measure the model's memory size in MB"""
        size = summary(self.model).total_param_bytes
        size_constant = 1 << 20
        size /= size_constant
        self.model_size = round(size, 3)
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
        """Generate a report with latency, throughput, and model size"""
        print(f"Latency: {self.latency} ms/sample with batch_size {self.batch_size}")
        print(
            f"Throughput: {self.throughput} samples/s with batch_size {self.batch_size}"
        )
        print(f"Model size: {self.model_size} MB")
