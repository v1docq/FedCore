import time
from typing import Union

import numpy as np
import torch
import torch.utils
from torchinfo import summary
from fedot.core.pipelines.pipeline import Pipeline
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from fedcore.architecture.computational.devices import default_device, extract_device
from fedcore.inference.onnx import ONNXInferenceModel
from functools import partial
from fedcore.tools.registry.model_registry import ModelRegistry
from fedcore.api.utils.data import DataLoaderHandler
from fedcore.tools.edge_device import PowerEstimator
from time import time

try:
    from fedcore.metrics.nlp_metrics import NLPAccuracy, NLPF1
    NLP_METRICS_AVAILABLE = True
except ImportError:
    NLP_METRICS_AVAILABLE = False


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


class BasePerformanceEvaluator:
    """Base class with common functionality for performance evaluators."""
    
    SIZE_CONSTANT_MB = 1024 ** 2
    
    def measure_model_size(self):
        """Measure model size in MB."""
        if isinstance(self.model, ONNXInferenceModel):
            size_all_mb = round(self.model.size(), 3) / self.SIZE_CONSTANT_MB
        else:
            param_size = sum(param.nelement() * param.element_size() for param in self.model.parameters())
            buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in self.model.buffers())
            size_all_mb = (param_size + buffer_size) / self.SIZE_CONSTANT_MB
        
        self.model_size = round(size_all_mb, 3)
        return self.model_size
    
    def report(self):
        """Print performance metrics."""
        print(f"Latency: {self.latency} ms/sample with batch_size {self.batch_size}")
        print(f"Throughput: {self.throughput} samples/s with batch_size {self.batch_size}")
        print(f"Model size: {self.model_size} MB")
    
    @staticmethod
    def _extract_batch(batch):
        """Extract first element from batch if it's a tuple/list."""
        return batch[0] if isinstance(batch, (tuple, list)) else batch
    
    @staticmethod
    def _transfer_to_cpu(result):
        """Transfer model output to CPU (handles Tensors, HuggingFace outputs, etc.)."""
        if hasattr(result, 'to'):
            result.to("cpu")
        elif hasattr(result, 'logits'):
            result.logits.to("cpu")
    
    @staticmethod
    def _compute_timing_stats(times_array):
        """Compute statistics for timing measurements."""
        return {
            "batches_per_second_mean": float((1 / times_array).mean()),
            "batches_per_second_std": float((1 / times_array).std()),
            "batches_per_second_min": float((1 / times_array).min()),
            "batches_per_second_max": float((1 / times_array).max()),
            "seconds_per_batch_mean": float(times_array.mean()),
            "seconds_per_batch_std": float(times_array.std()),
            "seconds_per_batch_min": float(times_array.min()),
            "seconds_per_batch_max": float(times_array.max()),
        }


class PerformanceEvaluator(BasePerformanceEvaluator):
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
            
            operation_device = getattr(fitted, 'device', None)
            if operation_device:
                self.device = operation_device
                self.cuda_allowed = (self.device.type == 'cuda')
            
            model_from_attr = getattr(fitted, self.model_regime, None)
            
            if model_from_attr is None:
                if self.model_regime == 'model_after':
                    model_from_attr = getattr(fitted, 'model_before', None)
                    if model_from_attr is None:
                        model_from_attr = getattr(fitted, 'model', None)
                elif self.model_regime == 'model_before':
                    model_from_attr = getattr(fitted, 'model', None)
            
            if model_from_attr is None:
                raise ValueError(f"Model regime '{self.model_regime}' not found in fitted operation")
            
            operation_device = getattr(fitted, 'device', None)
            if operation_device:
                self.device = operation_device
                self.cuda_allowed = (self.device.type == 'cuda')
            
            fedcore_id = getattr(fitted, '_fedcore_id', None)
            model_id = getattr(fitted, '_model_id', None)
            
            if fedcore_id and model_id:
                self.model = self._registry.get_model_with_fallback(
                    fedcore_id=fedcore_id,
                    model_id=model_id,
                    fallback_model=model_from_attr,
                    device=self.device
                )
                self._loaded_from_registry = True
                self._fedcore_id = fedcore_id
                self._model_id = model_id
            else:
                print("No fedcore_id or model_id found, using model from operation attributes")
                self.model = model_from_attr
                self._loaded_from_registry = False
                
        elif is_class_container:
            self.model = model.model
            self._loaded_from_registry = False
        else:
            self.model = model
            self._loaded_from_registry = False
        
        if not hasattr(self, 'device') or self.device is None:
            self.device = default_device()
            self.cuda_allowed = (self.device.type == 'cuda')
        
        model_device = extract_device(self.model) if hasattr(self.model, 'parameters') else None
        if model_device is not None and model_device.type != self.device.type:
            if model_device.type == 'cuda' and self.device.type == 'cpu':
                self.device = model_device
                self.cuda_allowed = True
            
        self.model.to(self.device)
    
    def cleanup_model(self):
        """Clean up loaded model from memory.
        
        DEPRECATED: Use ModelRegistry.cleanup_fedcore_instance() instead.
        This method is kept for backward compatibility only.
        """
        if hasattr(self, 'model') and self.model is not None:
            self._registry._delete_model_from_memory(self.model)
            self.model = None

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
            sample = self._extract_batch(batch)
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

            self._transfer_to_cpu(device_result)
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
            metrics = self._compute_timing_stats(s_per_batch)
            batches_per_s = 1 / s_per_batch

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
            batch = self._extract_batch(batch)
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
        latency_fn = cuda_latency_eval if self.cuda_allowed else cpu_latency_eval
        
        for batch in tqdm(self.data_loader(max_batches=max_samples or self.batch_size)):
            features = self._extract_batch(batch)
            if isinstance(features, torch.Tensor):
                for sample in features:
                    is_already_cuda = all([hasattr(sample, "cuda"), self.cuda_allowed])
                    sample = sample.cuda(non_blocking=True) if is_already_cuda else sample.to(self.device)
                    sample_batch = sample[None, ...]
                    lat_list.append(latency_fn(sample_batch))
            else:
                features = features.to(self.device) if isinstance(features, torch.Tensor) else features
                lat_list.append(latency_fn(features))
        return lat_list

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

    @torch.no_grad()
    def warm_up_cuda(self, n_batches=3):
        """Warm up CUDA by performing some dummy computations"""
        if self.cuda_allowed:
            for batch in tqdm(self.data_loader(max_batches=n_batches), desc="warming"):
                inputs = self._extract_batch(batch)
                _ = self.model(inputs.to(self.device))


class PerformanceEvaluatorOD(BasePerformanceEvaluator):
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

    def eval(self, metric_class=None, metric_names=None, is_nlp=False, tokenizer=None):
        """
        Evaluate model performance.
        
        Args:
            metric_class: Metric class to use for evaluation.
            metric_names: List of metric names (for NLP tasks).
            is_nlp: Whether this is an NLP task.
            tokenizer: Tokenizer for NLP tasks.
        
        Returns:
            dict: Performance metrics
        """
        result = dict(
            latency=self.measure_latency(),
            throughput=self.measure_throughput(),
            model_size=self.measure_model_size(),
            target_metrics=self.measure_target_metric(
                metric_class=metric_class,
                metric_names=metric_names,
                is_nlp=is_nlp,
                tokenizer=tokenizer
            ),
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

    def _extract_batch_data(self, batch):
        """Extract inputs and targets from batch."""
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0], batch[1]
        return (batch[0] if isinstance(batch, (list, tuple)) else batch), None
    
    def _move_to_device(self, inputs):
        """Move inputs to device and get predictions."""
        if isinstance(inputs, list):
            inputs = [inp.to(self.device) for inp in inputs]
        elif isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
        return self.model(inputs)
    
    def _process_nlp_batch(self, prediction, targets, tokenizer):
        """Process NLP predictions and targets."""
        logits = prediction.logits if hasattr(prediction, 'logits') else prediction
        pred_ids = torch.argmax(logits, dim=-1)
        
        pred_texts, target_texts = [], []
        for i in range(pred_ids.shape[0]):
            pred_texts.append(tokenizer.decode(pred_ids[i], skip_special_tokens=True))
            if targets is not None and isinstance(targets, torch.Tensor):
                target_ids = targets[i][targets[i] != -100]
                target_texts.append(tokenizer.decode(target_ids, skip_special_tokens=True))
        
        return pred_ids.cpu(), targets.cpu() if isinstance(targets, torch.Tensor) else None, pred_texts, target_texts
    
    def _compute_nlp_metrics(self, predictions, targets, pred_texts, target_texts, metric_names):
        """Compute NLP metrics from predictions."""
        metrics = {}
        metric_names = metric_names or ['accuracy', 'f1']
        
        if predictions and targets:
            flat_preds = torch.cat([p.flatten() for p in predictions]).numpy()
            flat_targets = torch.cat([t.flatten() for t in targets]).numpy()
            valid_mask = flat_targets != -100
            flat_preds, flat_targets = flat_preds[valid_mask], flat_targets[valid_mask]
            
            metric_map = {'accuracy': NLPAccuracy, 'f1': NLPF1}
            for name in metric_names:
                if name in metric_map:
                    try:
                        result = metric_map[name]().compute(y_pred=flat_preds.tolist(), y_true=flat_targets.tolist())
                        metrics[name] = result.get(name, 0.0)
                    except Exception as e:
                        print(f"Warning: Could not compute {name}: {e}")
        
        if pred_texts and target_texts:
            if any(m in metric_names for m in ['sacrebleu', 'bleu']):
                try:
                    from fedcore.metrics.nlp_metrics import SacreBLEU
                    result = SacreBLEU().compute(y_pred=pred_texts, y_true=[[t] for t in target_texts])
                    metrics['bleu'] = result.get('score', 0.0)
                except Exception as e:
                    print(f"Warning: Could not compute BLEU: {e}")
            
            if 'rouge' in metric_names:
                try:
                    from fedcore.metrics.nlp_metrics import ROUGE
                    result = ROUGE().compute(y_pred=pred_texts, y_true=target_texts)
                    metrics.update({k: result.get(k, 0.0) for k in ['rouge1', 'rougeL']})
                except Exception as e:
                    print(f"Warning: Could not compute ROUGE: {e}")
        
        return metrics
    
    def _compute_standard_metrics(self, predictions, targets, metric_class):
        """Compute standard metrics from predictions."""
        if not (predictions and targets):
            return {}
        
        preds = torch.cat(predictions).numpy() if isinstance(predictions[0], torch.Tensor) else np.array(predictions)
        targs = torch.cat(targets).numpy() if isinstance(targets[0], torch.Tensor) else np.array(targets)
        
        if metric_class and hasattr(metric_class, 'metric'):
            return {metric_class.__name__.lower(): metric_class.metric(targs, preds)}
        
        return {'accuracy': accuracy_score(targs, preds)}
    
    def measure_target_metric(self, metric_class=None, metric_names=None, is_nlp=False, tokenizer=None):
        """
        Measure target metrics for the model.
        
        Args:
            metric_class: Metric class (QualityMetric subclass). Default: Accuracy
            metric_names: List of metric names for NLP tasks ['accuracy', 'f1', 'sacrebleu', 'rouge']
            is_nlp: Whether this is an NLP task
            tokenizer: Tokenizer for NLP text-based metrics
        
        Returns:
            dict: Computed metrics
        """
        all_preds, all_targets, pred_texts, target_texts = [], [], [], []
        
        with torch.no_grad():
            for batch in tqdm(self.data_loader, desc="Measuring target metric", unit="batch"):
                inputs, targets = self._extract_batch_data(batch)
                prediction = self._move_to_device(inputs)
                
                if is_nlp and tokenizer:
                    pred_ids, targ_ids, p_texts, t_texts = self._process_nlp_batch(prediction, targets, tokenizer)
                    all_preds.append(pred_ids)
                    if targ_ids is not None:
                        all_targets.append(targ_ids)
                    pred_texts.extend(p_texts)
                    target_texts.extend(t_texts)
                else:
                    pred = prediction.cpu() if isinstance(prediction, torch.Tensor) else prediction
                    all_preds.extend(pred) if isinstance(pred, list) else all_preds.append(pred)
                    if targets is not None:
                        targ = targets.cpu() if isinstance(targets, torch.Tensor) else targets
                        all_targets.extend(targ) if isinstance(targ, list) else all_targets.append(targ)
        
        if self.device == "cuda":
            torch.cuda.synchronize()

        if is_nlp and NLP_METRICS_AVAILABLE:
            metrics = self._compute_nlp_metrics(all_preds, all_targets, pred_texts, target_texts, metric_names)
        else:
            metrics = self._compute_standard_metrics(all_preds, all_targets, metric_class)
        
        self.target_metrics = metrics
        return metrics

    def warm_up_cuda(self, num_iterations=10):
        """Warm up CUDA by performing some dummy computations"""
        if torch.cuda.is_available():
            for _ in range(num_iterations):
                inputs, _ = next(iter(self.data_loader))
                inputs = list(input.to(self.device) for input in inputs)
                _ = self.model(inputs)