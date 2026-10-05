"""Measured inference costs with explicit units and device availability."""
import io
import time
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader


class MeasurementUnavailable(RuntimeError):
    code='measurement_unavailable'


@dataclass
class TimingResult:
    mean:float
    std:float
    min:float
    max:float
    unit:str='ms'


class PerformanceEvaluator:
    BYTES_TO_MB=1<<20
    WARMUP_BATCHES=3
    DEFAULT_NUM_RUNS=100
    DEFAULT_BATCH_SIZE=32
    def __init__(self,model,model_regime='model_after',data=None,device=None,batch_size=32,
                 n_batches=8,collate_fn=None,need_wrap=False,warmup_batches=3,
                 clock=None,power_reader=None,synchronize=None):
        if not isinstance(n_batches,int) or n_batches<=0:raise ValueError('n_batches must be positive')
        if not isinstance(model,torch.nn.Module):
            if hasattr(model,'model'):model=model.model
            elif hasattr(model,'operator'):model=getattr(model.operator.root_node.fitted_operation,model_regime)
            else:raise TypeError('Performance evaluation requires nn.Module')
        self.model=deepcopy(model).eval()
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.n_batches=n_batches;self.batch_size=batch_size;self.warmup_batches=warmup_batches
        self._clock=clock or time.perf_counter;self._power_reader=power_reader;self._synchronize=synchronize
        self._cuda_available=torch.cuda.is_available()
        self._need_wrap=need_wrap;self.measurement_info={}
        if hasattr(data,'test_dataloader'):data=data.test_dataloader
        if data is None:raise ValueError('Measurement data is required')
        self.data_loader=data if isinstance(data,DataLoader) or need_wrap else DataLoader(data,batch_size=batch_size,shuffle=False,collate_fn=collate_fn)

    def _sync(self,device):
        if self._synchronize is not None:self._synchronize()
        elif device.type=='cuda':torch.cuda.synchronize(device)

    def _generate_example_batch(self,num_samples=None,return_sample=False,device='cpu',metric=''):
        limit=self.n_batches if num_samples is None else num_samples
        if not isinstance(limit,int) or limit<=0:raise ValueError('Measurement limit must be positive')
        loader=self.data_loader(max_batches=limit) if self._need_wrap else self.data_loader
        count=0
        for batch in loader:
            features=batch[0] if isinstance(batch,(tuple,list)) else batch
            if return_sample:
                for sample in features:
                    yield sample.to(device).unsqueeze(0)
                    count+=1
                    if count>=limit:return
            else:
                yield features.to(device)
                count+=1
                if count>=limit:return

    def _warmup(self,device):
        for batch in self._generate_example_batch(self.warmup_batches,device=device) if self.warmup_batches else ():
            self.model(batch)
        self._sync(device)

    def _record(self,name,device,count,unit):
        if not count:raise MeasurementUnavailable('No measurement batches available')
        self.measurement_info[name]={'device':str(device),'measured_batches':count,
            'warmup_batches':self.warmup_batches,'unit':unit,'scope':'whole model inference'}

    @torch.no_grad()
    def measure_latency(self,device=torch.device('cpu'),num_samples=None):
        device=torch.device(device);self.model.to(device);self._warmup(device)
        values=[]
        for batch in self._generate_example_batch(num_samples,device=device):
            self._sync(device);start=self._clock();self.model(batch);self._sync(device)
            values.append((self._clock()-start)*1000)
        self._record('latency',device,len(values),'ms/batch')
        return float(np.mean(values)),float(np.std(values))

    @torch.no_grad()
    def measure_throughput(self,device=torch.device('cpu'),num_iterations=30):
        if num_iterations<=0:raise ValueError('num_iterations must be positive')
        device=torch.device(device);self.model.to(device);self._warmup(device)
        values=[];count=0
        for batch in self._generate_example_batch(self.n_batches,device=device):
            count+=1
            for _ in range(num_iterations):
                self._sync(device);start=self._clock();self.model(batch);self._sync(device)
                elapsed=self._clock()-start
                if elapsed<=0:raise MeasurementUnavailable('Nonpositive measured elapsed time')
                values.append(len(batch)/elapsed)
        self._record('throughput',device,count,'samples/s')
        return float(np.mean(values)),float(np.std(values))

    def _read_power_watts(self,device):
        if self._power_reader is not None:return float(self._power_reader())
        if device.type!='cuda' or not torch.cuda.is_available():
            raise MeasurementUnavailable('CUDA device and NVML power readings are required')
        try:
            import pynvml
            pynvml.nvmlInit()
            handle=pynvml.nvmlDeviceGetHandleByIndex(device.index or 0)
            return pynvml.nvmlDeviceGetPowerUsage(handle)/1000.0
        except Exception as exc:raise MeasurementUnavailable(f'NVML power reading unavailable: {exc}') from exc

    def _eval_single_power(self,batch,device=None):
        """Trapezoidal device energy estimate in joules, elapsed seconds, mean watts."""
        device=torch.device(device or self.device)
        self._sync(device)
        power_start=self._read_power_watts(device);start=self._clock()
        self.model(batch);self._sync(device)
        end=self._clock();power_end=self._read_power_watts(device)
        elapsed=end-start
        if elapsed<=0 or not np.isfinite([power_start,power_end,elapsed]).all() or min(power_start,power_end)<0:
            raise MeasurementUnavailable('Invalid power/timing samples')
        watts=(power_start+power_end)/2
        return watts*elapsed,elapsed,watts

    @torch.no_grad()
    def _measure_power_energy(self,device,num_samples):
        device=torch.device(device)
        self._read_power_watts(device) # availability before inference effects
        self.model.to(device);self._warmup(device)
        measurements=[self._eval_single_power(batch,device) for batch in self._generate_example_batch(num_samples,device=device)]
        self._record('energy',device,len(measurements),'J/batch')
        self.measurement_info['energy'].update(scope='whole device over inference interval',estimator='endpoint trapezoidal NVML power')
        return measurements

    def measure_energy(self,device=torch.device('cpu'),num_samples=None):
        values=[m[0] for m in self._measure_power_energy(device,num_samples)]
        return float(np.mean(values)),float(np.std(values))

    def measure_power(self,device=torch.device('cpu'),num_samples=None):
        measured=self._measure_power_energy(device,num_samples)
        mean=sum(m[0] for m in measured)/sum(m[1] for m in measured)
        self.measurement_info['power']={**self.measurement_info['energy'],'unit':'W'}
        return float(mean),float(np.std([m[2] for m in measured]))

    measure_power_consumption=measure_energy
    measure_energy_consumption=measure_energy

    def measure_model_size(self,device=None):
        """Serialized state size in MiB, including quantized packed weights/buffers."""
        buffer=io.BytesIO();torch.save(self.model.state_dict(),buffer)
        return len(buffer.getvalue())/self.BYTES_TO_MB,0.

    @torch.no_grad()
    def evaluate(self):
        result={'model_size':self.measure_model_size()}
        for device in [torch.device('cpu')]+([self.device] if self.device.type=='cuda' else []):
            for name,method in [('latency',self.measure_latency),('throughput',self.measure_throughput),('power',self.measure_power),('energy',self.measure_energy)]:
                try:result[f'{device.type}_{name}']=method(device)
                except MeasurementUnavailable as exc:result[f'{device.type}_{name}']={'status':'unavailable','reason':str(exc)}
        result['measurement_info']=self.measurement_info
        return result

    def generate_report(self):return str(self.evaluate())
    def __enter__(self):return self
    def __exit__(self,*args):return False


def format_size(num,bytes=False):
    factor=1024 if bytes else 1000;suffix='B' if bytes else ''
    for unit in ['', 'K','M','G','T','P']:
        if num<factor:return f'{num:.2f}{unit}{suffix}'
        num/=factor
    return f'{num:.2f}P{suffix}'


def format_duration(seconds,return_raw=False):
    value,unit=(seconds,'s') if seconds>=1 else ((seconds*1e3,'ms') if seconds>=1e-3 else (seconds*1e6,'μs'))
    return value if return_raw else f'{value:.2f} {unit}'
