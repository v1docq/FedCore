"""Knowledge distillation with an immutable teacher snapshot and student gradients."""
from copy import deepcopy
from dataclasses import dataclass,field
from collections.abc import Mapping
import math
import torch
from torch import nn,optim
from fedcore.algorithm.base_compression_model import BaseCompressionModel
from fedcore.losses.distilation_loss import KLLossSoft


@dataclass(frozen=True)
class DistillationLossConfig:
    loss_weight:float=.5
    last_layer_loss_weight:float=.5
    intermediate_attn_layers_weights:tuple=()
    intermediate_feat_layers_weights:tuple=()
    student_teacher_attention_mapping:dict=field(default_factory=dict)
    student_teacher_feature_mapping:dict=field(default_factory=dict)


def _loss_config(value=None,defaults=None):
    """Normalize the actual loss weights before training or model registration."""
    names=tuple(DistillationLossConfig.__dataclass_fields__)
    data=dict(defaults or {})
    if value is not None:
        if hasattr(value,"to_dict"):value=value.to_dict()
        if isinstance(value,Mapping):
            unknown=set(value)-set(names)
            if unknown:raise ValueError(f"Unknown distillation loss fields: {sorted(unknown)}")
            data.update(value)
        else:
            data.update({name:getattr(value,name) for name in names if hasattr(value,name)})
    for name in ("loss_weight","last_layer_loss_weight"):
        weight=data.get(name,.5)
        if isinstance(weight,bool) or not isinstance(weight,(int,float)) or not math.isfinite(weight) or weight<0:
            raise ValueError(f"{name} must be finite and nonnegative")
        data[name]=float(weight)
    for name in ("intermediate_attn_layers_weights","intermediate_feat_layers_weights"):
        weights=tuple(data.get(name,()) or ())
        if any(isinstance(w,bool) or not isinstance(w,(int,float)) or not math.isfinite(w) or w<0 for w in weights):
            raise ValueError(f"{name} must contain finite nonnegative weights")
        data[name]=weights
    for name in ("student_teacher_attention_mapping","student_teacher_feature_mapping"):
        mapping=dict(data.get(name,{}) or {})
        if any(isinstance(i,bool) or not isinstance(i,int) or i<0 for pair in mapping.items() for i in pair):
            raise ValueError(f"{name} requires nonnegative integer indices")
        data[name]=mapping
    return DistillationLossConfig(**data)


class BaseDistilator(BaseCompressionModel):
    def __init__(self,params=None):
        params=params.to_dict() if hasattr(params,"to_dict") else dict(params or {})
        weights={name:params[name] for name in DistillationLossConfig.__dataclass_fields__ if name in params}
        self.distilation_params=_loss_config(params.get("distilation_params"),weights)
        super().__init__(params)
        self.epochs=params.get('epochs',15)
        self.criterion=params.get('loss',nn.CrossEntropyLoss())
        if isinstance(self.criterion,str):
            self.criterion={"cross_entropy":nn.CrossEntropyLoss,"mse":nn.MSELoss}.get(self.criterion)
            if self.criterion is None:raise ValueError("Unsupported distillation supervised loss")
            self.criterion=self.criterion()
        self.optimizer=params.get('optimizer',optim.Adam)
        if isinstance(self.optimizer,str):
            self.optimizer={"adam":optim.Adam,"sgd":optim.SGD}.get(self.optimizer.lower())
            if self.optimizer is None:raise ValueError("Unsupported distillation optimizer")
        self.learning_rate=params.get('lr',.001)
        self.temperature=params.get('temperature',1.)
        self._student_template=params.get('student_model')
        self.history=[]

    def _init_distil_model(self,teacher_model):
        return deepcopy(self._student_template if self._student_template is not None else teacher_model)

    def _calc_losses(self,loss,train_params,output_dict):
        result=loss*train_params.loss_weight
        if train_params.last_layer_loss_weight:
            result=result+train_params.last_layer_loss_weight*KLLossSoft()(
                output_dict['student_logits'],output_dict['teacher_logits'],temperature=self.temperature)
        for key,weights in (('attentions',getattr(train_params,'intermediate_attn_layers_weights',())),
                            ('hidden_states',getattr(train_params,'intermediate_feat_layers_weights',()))):
            if not weights or not any(weights):
                continue
            student=output_dict.get('student_'+key);teacher=output_dict.get('teacher_'+key)
            if student is None or teacher is None:
                raise ValueError(f'Intermediate {key} requested but model does not return it')
            mapping=train_params.student_teacher_attention_mapping if key=='attentions' else train_params.student_teacher_feature_mapping
            for index,weight in enumerate(weights):
                if weight:
                    target_index=mapping.get(index,index)
                    if index>=len(student) or target_index>=len(teacher) or student[index].shape!=teacher[target_index].shape:
                        raise ValueError(f'Incompatible intermediate {key} mapping')
                    result=result+weight*nn.functional.mse_loss(student[index],teacher[target_index].detach())
        return result

    def finetune(self,input_data,train_params):
        device=next(self.student_model.parameters()).device
        teacher_flags=[p.requires_grad for p in self.base_model.parameters()]
        self.base_model.eval()
        self.base_model.requires_grad_(False)
        optimizer=self.optimizer((p for p in self.student_model.parameters() if p.requires_grad),lr=self.learning_rate)
        try:
            for epoch in range(self.epochs):
                self.student_model.train()
                for batch in input_data.train_dataloader:
                    optimizer.zero_grad()
                    if isinstance(batch,dict):
                        features=batch['pixel_values'].to(device);labels=batch['labels'].to(device)
                        kwargs={'pixel_values':features,'labels':labels,'output_attentions':True,'output_hidden_states':True}
                        student=self.student_model(**kwargs)
                        with torch.no_grad():teacher=self.base_model(**kwargs)
                        logits=student.logits;teacher_logits=teacher.logits
                        supervised=getattr(student,'loss',None)
                        if supervised is None:supervised=self.criterion(logits,labels)
                    else:
                        features,labels=batch[0].to(device),batch[1].to(device)
                        logits=self.student_model(features)
                        with torch.no_grad():teacher_logits=self.base_model(features)
                        student=teacher=None
                        supervised=self.criterion(logits,labels)
                    outputs={'student_logits':logits,'teacher_logits':teacher_logits}
                    for key in ('attentions','hidden_states'):
                        outputs['student_'+key]=getattr(student,key,None)
                        outputs['teacher_'+key]=getattr(teacher,key,None)
                    loss=self._calc_losses(supervised,train_params,outputs)
                    if loss.ndim!=0 or not torch.isfinite(loss):
                        raise ValueError('Distillation must produce a finite scalar loss')
                    loss.backward();optimizer.step()
                    self.history.append(float(loss.detach()))
        finally:
            for p,flag in zip(self.base_model.parameters(),teacher_flags):p.requires_grad_(flag)

    def _fit_distil_model(self,input_data,distilation_params=None):
        config=self.distilation_params if distilation_params is None else _loss_config(distilation_params)
        self.finetune(input_data,config)

    def fit(self,input_data):
        if isinstance(self.epochs,bool) or not isinstance(self.epochs,int) or self.epochs<1:
            raise ValueError('Distillation epochs must be a positive integer')
        source=getattr(input_data,'model',None)
        if source is None:source=getattr(input_data,'target',None)
        if not isinstance(source,nn.Module):raise TypeError('Distillation requires a teacher nn.Module')
        self.model_before=deepcopy(source)
        self.base_model=deepcopy(source).to(self.device)
        self.student_model=self._init_distil_model(source).to(self.device)
        self._fit_distil_model(input_data)
        self.model_after=self.student_model.eval()
        return self.model_after

    def predict_for_fit(self,input_data,output_mode='fedcore'):
        return self.model_after if output_mode=='fedcore' else self.model_before

    predict=predict_for_fit
