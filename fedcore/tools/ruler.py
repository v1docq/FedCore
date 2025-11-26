import time
from typing import Union
import numpy as np
import torch
from torchinfo import summary
import pynvml
from fedot.core.pipelines.pipeline import Pipeline
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from fedcore.architecture.computational.devices import default_device, extract_device
from fedcore.metrics.metric_impl import (
    Accuracy, Precision, F1, RMSE, MSE, MAE, MAPE, SMAPE, R2
)
from functools import partial
from fedcore.tools.registry.model_registry import ModelRegistry
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
from fedcore.architecture.computational.devices import default_device
from functools import partial
from fedcore.api.utils.data import DataLoaderHandler
from fedcore.tools.edge_device import PowerEstimator

# Configure logging
logger = logging.getLogger(__name__)

# @dataclass
# class PerformanceMetrics:
#     """Data class to store performance metrics"""
#     model_size: Tuple[float, float]  # (mean, std)
    
#     cpu_latency: Tuple[float, float]  # (mean, std)
#     cpu_throughput: Tuple[float, float]  # (mean, std)
#     cpu_energy_consumption: Optional[float] = None

#     gpu_latency: Tuple[float, float]  # (mean, std)
#     gpu_throughput: Tuple[float, float]  # (mean, std)
#     gpu_energy_consumption: Optional[float] = None

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
        need_wrap: bool = False
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
        self._need_wrap = need_wrap
        self.device = device
        self._registry = ModelRegistry()

        
        self._init_metrics()
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
            if self.device is None:
                self.device = extract_device(self.model)
            else:
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
        if not self._need_wrap:
            self.data_loader = data
            return

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
    def evaluate(self):
        """
        Comprehensive model evaluation.
        
        Returns:
            PerformanceMetrics object containing all evaluation results
        """
        logger.info("Starting model evaluation...")
        devices = [torch.device('cpu')]

        if self._cuda_available:
            devices.append(self.device)

        metrics = {
            'latency': self.measure_latency,
            'throughput': self.measure_throughput,
            'power_consumption': self.measure_power_consumption
        }

        result = {'model_size': self.measure_model_size()}
        for device in devices:
            # Warm up if using CUDA
            self.model.to(device)
            if str(device.type) != 'cpu':
                self._warmup_cuda()

            # Measure performance metrics
            result.update({
                device.type + '_' + metric: method(device) for metric, method in metrics.items()
            })
        
        return result
    
    def _generate_example_batch(self, num_samples, return_sample=False, device='cpu', metric=''):
        num_samples = num_samples or float('inf')
        dataloader = self.data_loader(max_batches=num_samples) if self._need_wrap else self.data_loader
        count = 0
        for batch in tqdm(
            dataloader,
            desc=f"Measuring {metric}",
            unit="batch"
        ):
            
            features = batch[0] if isinstance(batch, (tuple, list)) else batch
            if return_sample:
                for sample in features:
                    sample = sample.to(device).unsqueeze(0)  # Add batch dimension
                    yield sample
                    count += return_sample
                    if num_samples <= count:
                        return
                    
            else:
                features = features.to(device)
                yield features
                count += return_sample
                if num_samples <= count:
                    return
    
    @torch.no_grad()
    def measure_power_consumption(self, device=torch.device('cpu'), num_samples=None):
        """Measure inference power consumption"""
        if device.type == 'cpu':
            return float('inf'), float('inf')
        self.model.to(device)
        powers = []
        for sample in self._generate_example_batch(num_samples, device=device, return_sample=False, metric='latency'):
            powers.append(self._eval_single_power(sample))        
        powers = np.array(powers)
        return float(np.mean(powers)), float(np.std(powers))
    
    def _eval_single_power(self, batch):
        index = self.device.index or 0
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
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
        self.model.to(device)
        latencies = []
        for sample in self._generate_example_batch(num_samples, device=device, return_sample=False, metric='latency'):
            latencies.append(method(sample))        
        latencies = np.array(latencies)
        return float(np.mean(latencies)), float(np.std(latencies))

    def _cuda_latency_eval(self, sample: torch.Tensor) -> float:
        """Measure latency on CUDA device"""
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()

        start_event.record()
        self.model(sample)
        end_event.record()
        torch.cuda.synchronize()
        
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
        self.model.to(device)
        method = self._cuda_throughput_eval if str(device.type) != 'cpu' else self._cpu_throughput_eval
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
            end_events[i].record()
            torch.cuda.synchronize()
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

    def measure_model_size(self, device=None) -> Tuple[float, float]:
        """Measure model size in MB
        device is for compatibility, not used"""
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
        for batch in self._generate_example_batch(n_batches, device=self.device):
            _ = self.model(batch)
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