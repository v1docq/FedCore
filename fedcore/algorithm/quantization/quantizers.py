"""CPU quantization with independent baselines and observable QAT stages."""
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
from torch.ao.quantization import get_default_qconfig,get_default_qat_qconfig,quantize_dynamic,QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx,prepare_qat_fx,convert_fx
from fedcore.algorithm.base_compression_model import BaseCompressionModel


@dataclass(frozen=True)
class QuantizationResult:
    status:str
    mode:str
    model:Optional[nn.Module]=None
    reason:Optional[str]=None
    training_steps:int=0


class QuantizationError(RuntimeError):
    def __init__(self,result):
        self.result=result
        super().__init__(f'{result.status}: {result.reason}')


def validate_quantization_request(model,example,mode,backend,dtype):
    if mode not in ('dynamic','static','qat'):
        raise ValueError(f'Unsupported quantization mode: {mode}')
    if backend not in torch.backends.quantized.supported_engines or backend=='none':
        raise ValueError(f'Unsupported quantization backend: {backend}')
    if mode!='dynamic' and dtype!=torch.qint8:
        raise ValueError('Static quantization and QAT require qint8')
    if dtype not in (torch.qint8,torch.float16):
        raise ValueError('Quantization dtype must be qint8 or float16')
    if not isinstance(model,nn.Module):raise TypeError('Quantization requires nn.Module')
    if any(type(m).__module__.startswith(('torch.ao.nn.quantized','torch.nn.quantized')) for m in model.modules()):
        raise ValueError('Already quantized graphs cannot be composed with a float quantization node')
    if not isinstance(example,torch.Tensor) or example.is_quantized or not example.is_floating_point():
        raise ValueError('Quantization requires a floating tensor input')
    if mode=='dynamic' and not any(isinstance(m,(nn.Linear,nn.LSTM,nn.GRU,nn.RNN)) for m in model.modules()):
        raise ValueError('No supported dynamic quantization operation')


class BaseQuantizer(BaseCompressionModel):
    def __init__(self,params=None):
        params=params or {}
        super().__init__(params)
        self.quant_type=params.get('quant_type','dynamic')
        self.backend=params.get('backend','fbgemm')
        self.dtype=params.get('dtype',torch.qint8)
        self.device=torch.device('cpu')
        self.qat_params=dict(params.get('qat_params') or {})
        self.history={'train_loss':[],'val_loss':[]}
        self.quantization_result=QuantizationResult('not_started',self.quant_type)

    def _get_example_input(self,input_data):
        loader=getattr(input_data,'calibration_dataloader',None)
        if loader is None:loader=input_data.train_dataloader
        batch=next(iter(loader))
        return (batch[0] if isinstance(batch,(tuple,list)) else batch).detach().cpu()

    def _init_model(self,input_data):
        source=input_data.model
        self.data_batch_for_calib=self._get_example_input(input_data)
        try:validate_quantization_request(source,self.data_batch_for_calib,self.quant_type,self.backend,self.dtype)
        except (ValueError,TypeError) as exc:
            self.quantization_result=QuantizationResult('not_applicable',self.quant_type,reason=str(exc))
            raise QuantizationError(self.quantization_result) from exc
        self.model_before=deepcopy(source)
        self.quant_model=deepcopy(source).cpu().eval()

    def _prepare_model(self,input_data):
        if self.quant_type=='dynamic':return self.quant_model
        qconfig=(get_default_qat_qconfig(self.backend) if self.quant_type=='qat' else get_default_qconfig(self.backend))
        mapping=QConfigMapping().set_global(qconfig)
        prepare=prepare_qat_fx if self.quant_type=='qat' else prepare_fx
        self.quant_model.train(self.quant_type=='qat')
        self.quant_model=prepare(self.quant_model,mapping,(self.data_batch_for_calib,))
        if self.quant_type=='static':
            with torch.no_grad():
                loader=getattr(input_data,'calibration_dataloader',None)
                if loader is None:loader=input_data.train_dataloader
                for batch in loader:self.quant_model(batch[0].cpu() if isinstance(batch,(tuple,list)) else batch.cpu())
        return self.quant_model

    def _train_qat(self,input_data):
        epochs=self.qat_params.get('epochs',2)
        if isinstance(epochs,bool) or not isinstance(epochs,int) or epochs<=0:raise ValueError('QAT requires at least one training epoch')
        optimizer_name=self.qat_params.get('optimizer','adam')
        optimizer_cls={'adam':torch.optim.Adam,'sgd':torch.optim.SGD}.get(optimizer_name)
        if optimizer_cls is None:raise ValueError(f'Unsupported QAT optimizer: {optimizer_name}')
        optimizer=optimizer_cls(self.quant_model.parameters(),lr=self.qat_params.get('lr',.001))
        criterion=self.qat_params.get('criterion','cross_entropy')
        if isinstance(criterion,str):
            criterion={'cross_entropy':nn.CrossEntropyLoss,'mse':nn.MSELoss}.get(criterion)
            if criterion is None:raise ValueError('Unsupported QAT criterion')
            criterion=criterion()
        steps=0
        self.quant_model.train()
        for _ in range(epochs):
            for features,target in input_data.train_dataloader:
                optimizer.zero_grad()
                loss=criterion(self.quant_model(features.cpu()),target.cpu())
                if loss.ndim!=0 or not torch.isfinite(loss):raise ValueError('QAT requires a finite scalar training loss')
                loss.backward();optimizer.step();steps+=1
                self.history['train_loss'].append(float(loss.detach()))
        if steps==0:raise ValueError('QAT training loader is empty')
        return steps

    def fit(self,input_data):
        self._init_model(input_data)
        engine=torch.backends.quantized.engine
        steps=0
        try:
            torch.backends.quantized.engine=self.backend
            self._prepare_model(input_data)
            if self.quant_type=='qat':steps=self._train_qat(input_data)
            if self.quant_type=='dynamic':
                candidate=quantize_dynamic(nn.Sequential(self.quant_model),{nn.Linear,nn.LSTM,nn.GRU,nn.RNN},dtype=self.dtype,inplace=False)
            else:
                candidate=convert_fx(self.quant_model.eval())
            if not any(type(m).__module__.startswith(('torch.ao.nn.quantized','torch.nn.quantized')) for m in candidate.modules()):
                raise ValueError('Conversion produced no quantized operations')
            with torch.no_grad():candidate(self.data_batch_for_calib)
            self.quant_model=candidate
            self.model_after=candidate
            self.quantization_result=QuantizationResult('completed',self.quant_type,candidate,training_steps=steps)
            return candidate
        except Exception as exc:
            self.quantization_result=QuantizationResult('error',self.quant_type,reason=str(exc),training_steps=steps)
            raise QuantizationError(self.quantization_result) from exc
        finally:torch.backends.quantized.engine=engine

    def predict_for_fit(self,input_data,output_mode='fedcore'):
        if self.quantization_result.status!='completed':raise QuantizationError(self.quantization_result)
        return self.model_after if output_mode=='fedcore' else self.model_before

    predict=predict_for_fit
