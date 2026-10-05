"""Native LoRA operation extracted from PR #46 onto the current model contract."""
from copy import deepcopy
import torch
from torch import nn
from fedcore.algorithm.base_compression_model import BaseCompressionModel
from fedcore.models.network_modules.layers.lora import apply_lora,LoRALayer


class BaseLoRA(BaseCompressionModel):
    def __init__(self,params=None):
        params=params.to_dict() if hasattr(params, "to_dict") else dict(params or {})
        if params.get('use_peft',False):raise ValueError('Native LoRA operation requires use_peft=False; PEFT remains a separate integration')
        if params.get('lora_bias','none')!='none':raise ValueError('Native LoRA supports frozen base bias (lora_bias=none)')
        super().__init__(params)
        self.lora_r=params.get('lora_r',8)
        self.lora_alpha=params.get('lora_alpha',16)
        self.lora_dropout=params.get('lora_dropout',0.)
        self.lora_target_modules=params.get('lora_target_modules')
        self.epochs=params.get('epochs',3)
        self.learning_rate=params.get('lr',1e-4)
        self.history=[]

    def _apply_lora_custom(self,model,params=None):
        targets=self.lora_target_modules
        # PR46 selected target name fragments; resolve them to concrete paths first.
        if targets:
            targets=[name for name,layer in model.named_modules() if type(layer) in (nn.Linear,nn.Conv2d,nn.Embedding)
                     and any(fragment in name for fragment in targets)]
            if not targets:raise ValueError('No supported layers match LoRA targets')
        else:targets=None
        adapted=apply_lora(model,rank=self.lora_r,lora_alpha=self.lora_alpha,target_layers=targets,lora_dropout=self.lora_dropout)
        for name,p in adapted.named_parameters():
            p.requires_grad_(any(component in name.split('.') for component in LoRALayer.adapter_layer_names))
        if not any(p.requires_grad for p in adapted.parameters()):raise ValueError('No trainable LoRA parameters')
        return adapted

    _apply_lora_to_model=_apply_lora_custom

    def fit(self,input_data):
        source=getattr(input_data,'model',None)
        if source is None:source=getattr(input_data,'target',None)
        if not isinstance(source,nn.Module):raise TypeError('LoRA requires a local nn.Module')
        if isinstance(self.epochs,bool) or not isinstance(self.epochs,int) or self.epochs<1:
            raise ValueError('LoRA epochs must be a positive integer')
        candidate=self._apply_lora_custom(source).to(self.device)
        self.model_before=deepcopy(source)
        device=next(candidate.parameters()).device
        optimizer=torch.optim.Adam((p for p in candidate.parameters() if p.requires_grad),lr=self.learning_rate)
        candidate.train()
        for _ in range(self.epochs):
            for batch in input_data.train_dataloader:
                optimizer.zero_grad()
                if isinstance(batch,dict):
                    values={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
                    result=candidate(**values)
                    loss=getattr(result,'loss',None)
                    if loss is None:raise ValueError('Dictionary LoRA batches require a model-provided scalar loss')
                else:
                    features,target=batch[0].to(device),batch[1].to(device)
                    output=candidate(features)
                    loss=nn.functional.cross_entropy(getattr(output,'logits',output),target)
                if loss.ndim!=0 or not torch.isfinite(loss):raise ValueError('LoRA training requires a finite scalar loss')
                loss.backward();optimizer.step();self.history.append(float(loss.detach()))
        if not self.history:raise ValueError('LoRA training loader is empty')
        self.model_after=candidate.eval()
        return self.model_after

    def predict_for_fit(self,input_data,output_mode='fedcore'):
        return self.model_after if output_mode=='fedcore' else self.model_before

    predict=predict_for_fit
