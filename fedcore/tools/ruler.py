import time
from typing import Union
import numpy as np
import torch
from torchinfo import summary
import pynvml
from fedot.core.pipelines.pipeline import Pipeline
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from fedcore.architecture.comptutaional.devices import default_device, extract_device
from fedcore.metrics.metric_impl import (
    Accuracy, Precision, F1, RMSE, MSE, MAE, MAPE, SMAPE, R2
)
from functools import partial
from fedcore.api.utils.data import DataLoaderHandler
from fedcore.tools.edge_device import PowerEstimator
from time import time


import time
from typing import Union, Dict, List, Tuple, Optional, Callable
import numpy as np
import torch
from torchinfo import summary
from fedot.core.pipelines.pipeline import Pipeline
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from dataclasses import dataclass
from contextlib import contextmanager

# Local imports
from fedcore.architecture.comptutaional.devices import default_device
from functools import partial
from fedcore.api.utils.data import DataLoaderHandler
from fedcore.tools.edge_device import PowerEstimator

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Data class to store performance metrics"""
    model_size: Tuple[float, float]  # (mean, std)
    
    cpu_latency: Tuple[float, float]  # (mean, std)
    cpu_throughput: Tuple[float, float]  # (mean, std)
    cpu_energy_consumption: Optional[float] = None

    gpu_latency: Tuple[float, float]  # (mean, std)
    gpu_throughput: Tuple[float, float]  # (mean, std)
    gpu_energy_consumption: Optional[float] = None

@dataclass
class TimingResult:
    """Data class to store timing results"""
    mean: float
    std: float
    min: float
    max: float
    unit: str = "ms"

class PerformanceEvaluator:
    """
    Comprehensive model performance evaluator for measuring inference metrics
    including latency, throughput, model size, and quality metrics.
    """
    
    # Constants
    BYTES_TO_MB = 1 << 20
    WARMUP_BATCHES = 3
    DEFAULT_NUM_RUNS = 100
    DEFAULT_BATCH_SIZE = 32

    def __init__(
        self,
        model: Callable,
        model_regime: str = 'model_after',
        data: Union[DataLoader, str] = None,
        device: Optional[torch.device] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        n_batches: int = 8,
        collate_fn: Optional[Callable] = None,
    ):
        """
        Initialize PerformanceEvaluator.
        
        Args:
            model: The model to evaluate (callable, Pipeline, or model container)
            model_regime: Model regime for Pipeline objects
            data: DataLoader or dataset path for evaluation
            device: Device to run evaluation on
            batch_size: Number of samples per batch
            n_batches: Number of batches to process
            collate_fn: Function to collate data into batches
        """
        self.model_regime = model_regime
        self.n_batches = n_batches
        self.batch_size = batch_size
        
        self._init_metrics()
        self.device = device or default_device()
        self._cuda_available = torch.cuda.is_available()
        
        self._initialize_model(model)
        self._initialize_data_loader(data, collate_fn)

    def _init_metrics(self) -> None:
        """Initialize all performance metrics to None"""
        self._metrics = {
            'latency': None,
            'throughput': None, 
            'model_size': None,
            'energy_consumption': None
        }

    def _initialize_model(self, model: Callable) -> None:
        """Initialize model and handle different model types"""
        try:
            if isinstance(model, Pipeline):
                self.model = self._extract_model_from_pipeline(model)
            elif hasattr(model, "model"):
                self.model = model.model  # Model container
            else:
                self.model = model
                
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    def _extract_model_from_pipeline(self, pipeline: Pipeline) -> Callable:
        """Extract model from FEDOT pipeline"""
        try:
            model = getattr(pipeline.operator.root_node.fitted_operation, self.model_regime)
            # Try to get device from pipeline if available
            self.device = extract_device(model)
            return model
        except AttributeError as e:
            logger.error(f"Failed to extract model from pipeline: {e}")
            raise

    def _initialize_data_loader(self, data: Union[DataLoader, str], collate_fn: Optional[Callable]) -> None:
        """Initialize data loader for evaluation"""
        if isinstance(data, DataLoader):
            collate_fn = data.collate_fn
            dataset = data.dataset
        elif isinstance(data, str):
            # TODO: Implement dataset loading from directory
            raise NotImplementedError("Dataset loading from directory not implemented")
        else:
                dataset = data

        self.data_loader = partial(
            DataLoaderHandler.check_convert,
            dataloader=DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                collate_fn=collate_fn
            )
        )

    @torch.no_grad()
    def evaluate(self) -> PerformanceMetrics:
        """
        Comprehensive model evaluation.
        
        Returns:
            PerformanceMetrics object containing all evaluation results
        """
        logger.info("Starting model evaluation...")
        devices = [torch.device('cpu')]
        if self._cuda_available():
            devices.append(self.device)

        metrics = {
            'latency': self.measure_latency,
            'throughput': self.measure_throughput,
            'power_consumption': self.measure_power
        }

        result = {'model_size': self.measure_model_size()}
        for device in devices:
            # Warm up if using CUDA
            if self._cuda_available:
                self._warmup_cuda()

            # Measure performance metrics
            result.update({
                device.type + metric: method(device) for metric, method in metrics.items()
            })
        
        return result
    
    def _generate_example_batch(self, num_samples, return_sample=False, device='cpu', metric=''):
        for batch in tqdm(
            self.data_loader(max_batches=num_samples or self.batch_size),
            desc=f"Measuring {metric}",
            unit="batch"
        ):
            features = batch[0] if isinstance(batch, (tuple, list)) else batch
            if return_sample:
                for sample in features:
                    sample = sample.to(device).unsqueeze(0)  # Add batch dimension
                    yield sample
            else:
                batch = batch.to(device)
                yield batch
    
    @torch.no_grad()
    def measure_power(self, device=torch.device('cpu')):

         # Clear any pending operations
        
        
        
        return 
    
    def _eval_single_power(self, batch):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Measurement
        start_power = pynvml.nvmlDeviceGetPowerUsage(handle)
        
        output = self.model(batch)
        
        # CRITICAL: Wait for GPU to finish
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        end_power = pynvml.nvmlDeviceGetPowerUsage(handle)
        return end_power - start_power
        

    @torch.no_grad()
    def measure_latency(self, device, num_samples: Optional[int] = None) -> Tuple[float, float]:
        """Measure inference latency"""
        method = self._cuda_latency_eval if device.type != 'cpu' else self._cpu_latency_eval
        latencies = []
        for sample in self._generate_example_batch(num_samples, device=device, return_sample=False, metric='latency'):
            latencies.append(method(sample, device))        
        latencies = np.array(latencies)
        return float(np.mean(latencies)), float(np.std(latencies))

    def _cuda_latency_eval(self, sample: torch.Tensor) -> float:
        """Measure latency on CUDA device"""
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(self.device)

        start_event.record()
        self.model(sample)
        torch.cuda.synchronize(self.device)
        end_event.record()
        
        return start_event.elapsed_time(end_event)

    def _cpu_latency_eval(self, sample: torch.Tensor) -> float:
        """Measure latency on CPU"""
        start_time = time.perf_counter()
        self.model(sample)
        end_time = time.perf_counter()
        return (end_time - start_time) * 1000  # Convert to milliseconds

    @torch.no_grad()
    def measure_throughput(self, device, num_iterations: int = 30) -> Tuple[float, float]:
        """Measure inference throughput"""
        throughputs = []
        method = self._cuda_throughput_eval if device.type != 'cpu' else self._cpu_throughput_eval
        for batch in self._generate_example_batch(self.n_batches, device=device, metric='throughput', return_sample=False):
            batch_throughputs = method(batch, num_iterations)
            throughputs.extend(batch_throughputs)
        
        throughputs = np.array(throughputs)
        return float(np.mean(throughputs)), float(np.std(throughputs))

    def _cuda_throughput_eval(self, batch: torch.Tensor, num_iterations: int) -> List[float]:
        """Measure throughput on CUDA"""
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
        torch.cuda.synchronize()
        for i in range(num_iterations):
            start_events[i].record()
            self.model(batch)
            torch.cuda.synchronize(self.device)
            end_events[i].record()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        return [len(batch) / (t / 1000) for t in times]  # samples per second

    def _cpu_throughput_eval(self, batch: torch.Tensor, num_iterations: int) -> List[float]:
        """Measure throughput on CPU"""
        times = []
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            self.model(batch)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        return [len(batch) / t for t in times]  # samples per second

    def measure_model_size(self) -> Tuple[float, float]:
        """Measure model size in MB"""
        try:
            model_summary = summary(self.model, verbose=0)
            size_mb = model_summary.total_param_bytes / self.BYTES_TO_MB
            return round(size_mb, 3), 0.0  # std is 0 for deterministic measurement
        except Exception as e:
            logger.warning(f"Model size measurement failed: {e}")
            # Fallback: calculate size manually
            total_params = sum(p.numel() for p in self.model.parameters())
            size_mb = (total_params * 4) / self.BYTES_TO_MB  # Assume float32
            return round(size_mb, 3), 0.0

    @torch.no_grad()
    def _warmup_cuda(self, n_batches: int = WARMUP_BATCHES) -> None:
        """Warm up CUDA by performing dummy computations"""
        if not self._cuda_available:
            return
            
        logger.info("Warming up CUDA...")
        for i, batch in enumerate(self.data_loader(max_batches=n_batches)):
            if isinstance(batch, (tuple, list)):
                inputs = batch[0]
            else:
                inputs = batch
            _ = self.model(inputs.to(self.device))
            
        if self._cuda_available:
            torch.cuda.synchronize()

    def generate_report(self) -> str:
        """Generate a comprehensive performance report"""
        metrics = self.evaluate()
        
        report = [
            "=" * 50,
            "MODEL PERFORMANCE REPORT",
            "=" * 50,
            f"Device: {self.device}",
            f"Batch size: {self.batch_size}",
            "",
            "LATENCY:",
            f"  Mean: {metrics.latency[0]:.2f} ± {metrics.latency[1]:.2f} ms",
            "",
            "THROUGHPUT:",
            f"  Mean: {metrics.throughput[0]:.2f} ± {metrics.throughput[1]:.2f} samples/s",
            "",
            "MODEL SIZE:",
            f"  {metrics.model_size[0]:.2f} MB",
            "",
            "CLASSIFICATION METRICS:",
        ]
        
        for name, value in metrics.classification_metrics.items():
            report.append(f"  {name}: {value:.4f}")
            
        report.extend([
            "",
            "REGRESSION METRICS:",
        ])
        
        for name, value in metrics.regression_metrics.items():
            report.append(f"  {name}: {value:.4f}")
            
        if metrics.energy_consumption:
            report.extend([
                "",
                "ENERGY CONSUMPTION:",
                f"  {metrics.energy_consumption:.2f} Joules",
            ])
            
        report.append("=" * 50)
        
        return "\n".join(report)

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context exit"""
        if self._cuda_available:
            torch.cuda.empty_cache()

# Utility functions (keep these at module level)
def format_size(num: int, bytes: bool = False) -> str:
    """Format number to human-readable size"""
    factor = 1024 if bytes else 1000
    suffix = "B" if bytes else ""
    for unit in ["", "K", "M", "G", "T", "P"]:
        if num < factor:
            return f"{num:.2f}{unit}{suffix}"
        num /= factor
    return f"{num:.2f}P{suffix}"

def format_duration(seconds: float, return_raw: bool = False) -> Union[str, float]:
    """Format duration to human-readable time"""
    if seconds >= 1:
        value, unit = seconds, "s"
    elif seconds >= 1e-3:
        value, unit = seconds * 1e3, "ms"
    else:
        value, unit = seconds * 1e6, "μs"
    
    return value if return_raw else f"{value:.2f} {unit}"


# def format_num(num: int, bytes=False):
#     """Scale bytes to its proper format, e.g. 1253656 => '1.20MB'"""
#     factor = 1024 if bytes else 1000
#     suffix = "B" if bytes else ""
#     for unit in ["", " K", " M", " G", " T", " P"]:
#         if num < factor:
#             return f"{num:.2f}{unit}{suffix}"
#         num /= factor


# def format_time(seconds, return_time=False):
#     """Defines how to format time in FunctionEvent"""
#     US_IN_SECOND = 1000.0 * 1000.0
#     US_IN_MS = 1000.0
#     prefix = 'us'
#     time_us = seconds * 1e6
#     if time_us >= US_IN_SECOND:
#         time_us = time_us / US_IN_SECOND
#         prefix = 's'
#     if time_us >= US_IN_MS:
#         time_us = time_us / US_IN_MS
#         prefix = 'ms'
#     if return_time:
#         return round(time_us, 3)
#     else:
#         return f"{round(time_us, 3)} {prefix}"


# class PerformanceEvaluator:
#     def __init__(
#             self,
#             model: callable,
#             model_regime: str = 'model_after',
#             data: Union[DataLoader, str] = None,
#             device=None,
#             batch_size=32,
#             n_batches=8,
#             collate_fn=None,
#     ):
#         """
#         Initialize PerformanceEvaluator to measure model performance metrics.
#         :param model: The model to evaluate
#         :param model_regime: The model regime (e.g., 'model_after' or 'model_before')
#         :param data: DataLoader or dataset to use for evaluation
#         :param device: Device to run model on ('cpu' or 'cuda')
#         :param batch_size: Number of samples per batch
#         :param n_batches: Number of batches to process
#         :param collate_fn: Function to collate data into batches
#         """
#         self.model_regime = model_regime
#         self.n_batches = n_batches
#         self.batch_size = batch_size
#         self._init_null_object()

#         self.device = device or default_device()
#         self.cuda_allowed = True if self.device.type == 'cuda' else False
#         self.transfer_to_device_fn = torch.Tensor.to
#         self._init_model(model)

#         dataset_from_directory = isinstance(data, str)
#         if isinstance(data, DataLoader):
#             collate_fn = data.collate_fn
#             dataset = data.dataset
#         elif dataset_from_directory:
#             pass  # TODO some logic for string dataset downloading
#         else:
#             dataset = data
#         self.data_loader = partial(DataLoaderHandler.check_convert, dataloader=DataLoader(
#             dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
#                                    )

#     def _init_null_object(self):
#         """Initialize performance metrics to None"""
#         self.latency = None
#         self.throughput = None
#         self.model_size = None
#         self.target_metrics = None

#     def _init_model(self, model):
#         """Initialize the model and assign it to the correct device"""
#         is_class_container = hasattr(model, "model")
#         is_pipeline_class = isinstance(model, Pipeline)

#         if is_pipeline_class:
#             self.model = getattr(model.operator.root_node.fitted_operation, self.model_regime)
#             try:
#                 self.device = getattr(model.operator.root_node.fitted_operation, 'device', default_device())
#                 self.cuda_allowed = True if self.device.type == 'cuda' else False
#             except:
#                 self.device = self.device
#         elif is_class_container:
#             self.model = model.model
#         else:
#             self.model = model
            
#         self.model.to(self.device)

#     def eval(self):
#         """Evaluate model performance: throughput, latency, classification, and regression metrics"""
#         self.warm_up_cuda()

#         # Calculate throughput and latency
#         lats = self.latency_eval()
#         thrs = self.throughput_eval()

#         # Get classification and regression metrics
#         classification_metrics = self.get_classification_metrics()
#         regression_metrics = self.get_regression_metrics()

#         result = dict(
#             latency=[np.mean(lats), np.std(lats)],
#             throughput=[np.mean(thrs), np.std(thrs)],
#             model_size=[self.measure_model_size(), 0.],
#             classification_metrics=classification_metrics,
#             regression_metrics=regression_metrics
#         )
#         return result

#     def get_classification_metrics(self):
#         """
#         Calculate classification metrics: accuracy, precision, recall, and f1-score
#         :return: Dictionary with classification metrics
#         """
#         # Generate random classification data
#         target_class = torch.randint(0, 2, (self.batch_size,))  # 2 classes
#         predict_class = torch.randint(0, 2, (self.batch_size,))

#         # Compute metrics
#         accuracy = Accuracy.metric(target_class, predict_class)
#         precision = Precision.metric(target_class, predict_class)
#         recall = Precision.metric(target_class, predict_class)  # Recall can be calculated from Precision
#         f1 = F1.metric(target_class, predict_class)

#         return {
#             "accuracy": accuracy,
#             "precision": precision,
#             "recall": recall,
#             "f1": f1
#         }

#     def get_regression_metrics(self):
#         """
#         Calculate regression metrics: RMSE, MSE, MAE, MAPE
#         :return: Dictionary with regression metrics
#         """
#         # Generate random regression data
#         target_regression = torch.rand(self.batch_size)
#         predict_regression = torch.rand(self.batch_size)

#         # Compute metrics
#         rmse = RMSE.metric(target_regression, predict_regression)
#         mse = MSE.metric(target_regression, predict_regression)
#         mae = MAE.metric(target_regression, predict_regression)
#         mape = MAPE.metric(target_regression, predict_regression)

#         return {
#             "rmse": rmse,
#             "mse": mse,
#             "mae": mae,
#             "mape": mape
#         }

#     def eval_detailed_latency(self, num_runs=100):
#         """Evaluate detailed latency for model inference"""
#         mean_latency = np.inf
#         std_latency = np.inf
#         t_cpu_2_gpu, t_device, t_gpu_2_cpu, t_total = [], [], [], []
#         for batch in tqdm(self.data_loader(max_batches=self.n_batches), desc="batches", unit="batch"):
#             sample = batch[0] if isinstance(batch, (tuple, list)) else batch
#             breakS

#         for _ in tqdm(range(num_runs), desc=f"Measuring inference for batch_size={self.batch_size}"):
#             start_on_cpu = time()
#             device_sample = self.transfer_to_device_fn(sample, self.device)

#             if self.cuda_allowed:
#                 start_event = torch.cuda.Event(enable_timing=True)
#                 stop_event = torch.cuda.Event(enable_timing=True)
#                 start_event.record()  # For GPU timing
#             start_on_device = time()  # For CPU timing

#             device_result = self.model(device_sample)

#             if self.cuda_allowed:
#                 stop_event.record()
#                 torch.cuda.synchronize()
#                 elapsed_on_device = stop_event.elapsed_time(start_event)
#                 stop_on_device = time()
#             else:
#                 stop_on_device = time()
#                 elapsed_on_device = stop_on_device - start_on_device

#             self.transfer_to_device_fn(device_result, "cpu")
#             stop_on_cpu = time()

#             t_cpu_2_gpu.append(start_on_device - start_on_cpu)
#             t_device.append(elapsed_on_device)
#             t_gpu_2_cpu.append(stop_on_cpu - stop_on_device)
#             t_total.append(stop_on_cpu - start_on_cpu)

#         results_dict = {}
#         for _ in [t_gpu_2_cpu, t_cpu_2_gpu, t_device, t_total]:
#             _.pop(0)  # delete first result in cycle, because of unstable
#         times_and_titles = [(t_device, "on_host_inference")]
#         if self.cuda_allowed:
#             times_and_titles.extend([(t_cpu_2_gpu, "cpu_to_gpu"), (t_gpu_2_cpu, "gpu_to_cpu"), (t_total, "total")])

#         for s_per_batch, title in times_and_titles:
#             s_per_batch = np.array(s_per_batch)
#             batches_per_s = 1 / s_per_batch

#             metrics = {
#                 "batches_per_second_mean": float(batches_per_s.mean()),
#                 "batches_per_second_std": float(batches_per_s.std()),
#                 "batches_per_second_min": float(batches_per_s.min()),
#                 "batches_per_second_max": float(batches_per_s.max()),
#                 "seconds_per_batch_mean": float(s_per_batch.mean()),
#                 "seconds_per_batch_std": float(s_per_batch.std()),
#                 "seconds_per_batch_min": float(s_per_batch.min()),
#                 "seconds_per_batch_max": float(s_per_batch.max()),
#             }

#             convert_to_report = {
#                 "batches_per_second": f"{format_num(batches_per_s.mean())} "
#                                       f"+/- {format_num(batches_per_s.std())} [{format_num(batches_per_s.min())}, "
#                                       f"{format_num(batches_per_s.max())}]",

#                 "batch_latency": f"{format_time(s_per_batch.mean())} "
#                                  f" +/- {format_time(s_per_batch.std())} [{format_time(s_per_batch.min())},"
#                                  f" {format_time(s_per_batch.max())}]",
#             }
#             mean_latency = format_time(s_per_batch.mean(), return_time=True)
#             std_latency = format_time(s_per_batch.std(), return_time=True)
#             results_dict[title] = {"metrics": metrics, "convert_to_report": convert_to_report}

#         return np.array([mean_latency, std_latency])

#     def measure_energy(
#             model,
#             sample,
#             model_device,
#             transfer_to_device_fn=torch.Tensor.to,
#             num_runs=100,
#             batch_size: int = None,
#             include_transfer_costs=True,
#             print_fn=None
#     ):
#         """Measure energy consumption during inference"""
#         def test_with_transfer():
#             nonlocal model, sample
#             transfer_to_device_fn(
#                 model(transfer_to_device_fn(sample, model_device)),
#                 "cpu",
#             )

#         def test_without_transfer():
#             nonlocal model, sample
#             model(sample)

#         if include_transfer_costs:
#             test_fn = test_with_transfer
#         else:
#             test_fn = test_without_transfer
#             sample = sample.to(model_device)

#         try:
#             p_est = PowerEstimator(print_fn=print_fn)
#             meas = []
#             for _ in tqdm(range(num_runs), desc=f"Measuring energy for batch_size={batch_size}"):
#                 meas.append(p_est.estimate_fn_power(test_fn)[0] / 1000)
#             inference_joules = float(np.array(meas).mean())
#         except Exception:
#             pass

#         return inference_joules

#     @torch.no_grad()
#     def throughput_eval(self, num_iterations=30):
#         """Evaluate throughput during inference"""
#         self.model.eval()
#         thr_list = []
#         steps_iter = range(num_iterations)
#         for batch in tqdm(self.data_loader(max_batches=self.n_batches), desc="batches", unit="batch"):
#             batch = batch[0] if isinstance(batch, (tuple, list)) else batch
#             is_already_cuda = all([hasattr(batch, "cuda"), self.cuda_allowed])
#             X = batch.cuda(non_blocking=True) if is_already_cuda else batch.to(self.device)
#             if is_already_cuda:
#                 start_events = [torch.cuda.Event(enable_timing=True) for _ in steps_iter]
#                 end_events = [torch.cuda.Event(enable_timing=True) for _ in steps_iter]
#                 for i in steps_iter:
#                     start_events[i].record()
#                     self.model(X)
#                     end_events[i].record()
#                 torch.cuda.synchronize(self.device)
#                 times = ([s.elapsed_time(e) for s, e in zip(start_events, end_events)])
#             else:
#                 times = list()
#                 for i in steps_iter:
#                     start_events = time()
#                     self.model(X)
#                     end_events = time()
#                     times.append(end_events - start_events)
#                 times = (times)
#             thr_list.extend(times)
#         return len(batch) / np.array(thr_list)

#     @torch.no_grad()
#     def latency_eval(self, max_samples=None):
#         """Evaluate latency during inference"""
#         def cuda_latency_eval(sample_batch):
#             start_event = torch.cuda.Event(enable_timing=True)
#             end_event = torch.cuda.Event(enable_timing=True)
#             start_event.record()
#             self.model(sample_batch)
#             end_event.record()
#             torch.cuda.synchronize(self.device)
#             time = start_event.elapsed_time(end_event)
#             return time
        
#         def cpu_latency_eval(sample_batch):
#             start_on_device = time()
#             self.model(sample_batch)
#             stop_on_device = time()
#             elapsed_on_device = stop_on_device - start_on_device
#             return elapsed_on_device

#         self.model.eval()
#         lat_list = []
#         for batch in tqdm(self.data_loader(max_batches=max_samples or self.batch_size)):
#             if isinstance(batch, tuple) or isinstance(batch, list):
#                 features = batch[0]
#                 if isinstance(features, torch.Tensor):
#                     for sample in features:
#                         is_already_cuda = all([hasattr(sample, "cuda"), self.cuda_allowed])
#                         sample = sample.cuda(non_blocking=True) if is_already_cuda else sample.to(self.device)
#                         sample.to(self.device)
#                         sample_batch = sample[None, ...]
#                         lat_list.append(cuda_latency_eval(sample_batch)) if is_already_cuda else \
#                         lat_list.append(cpu_latency_eval(sample_batch))
#             else:
#                 lat_list.append(cuda_latency_eval(batch))
#         return np.array(lat_list)

#     def measure_latency_throughput(self, reps: int = 3, batches: int = 10):
#         """Measure both latency and throughput in multiple runs"""
#         timings_lat = []
#         timings_thr = []
#         with tqdm(
#                 total=reps, desc="Measuring latency and throughput", unit="rep"
#         ) as pbar:
#             for rep in range(reps):
#                 timings_thr.append(self.throughput_eval(reps))
#                 timings_lat.append(self.latency_eval())
#                 pbar.update(1)
#         latency = np.array([[np.mean(x), np.std(x)] for x in timings_lat])
#         throughput = np.array([[np.mean(x), np.std(x)] for x in timings_thr])
#         self.latency, self.throughput = np.mean(np.array(latency), axis=0), np.mean(
#             np.array(throughput), axis=0
#         )
#         return self.latency, self.throughput

#     def measure_model_size(self):
#         """Measure the model's memory size in MB"""
#         size = summary(self.model).total_param_bytes
#         size_constant = 1 << 20
#         size /= size_constant
#         self.model_size = round(size, 3)
#         return self.model_size

#     @torch.no_grad()
#     def warm_up_cuda(self, n_batches=3):
#         """Warm up CUDA by performing some dummy computations"""
#         if self.cuda_allowed:
#             for batch in tqdm(self.data_loader(max_batches=n_batches), desc="warming"):
#                 if isinstance(batch, tuple) or isinstance(batch, list):
#                     inputs = batch[0]
#                 _ = self.model(inputs.to(self.device))

#     def report(self):
#         """Generate a report with latency, throughput, and model size"""
#         print(f"Latency: {self.latency} ms/sample with batch_size {self.batch_size}")
#         print(
#             f"Throughput: {self.throughput} samples/s with batch_size {self.batch_size}"
#         )
#         print(f"Model size: {self.model_size} MB")
