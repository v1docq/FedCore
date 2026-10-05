"""LoRA adapters with independent base weights and reversible inference merges."""
import math
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F


class LoRALayer(nn.Module):
    adapter_layer_names=('lora_A','lora_B','lora_embedding_A','lora_embedding_B')
    def __init__(self,base_layer,adapter_name='default',r=0,lora_alpha=1,lora_dropout=0.,init_lora_weights=True,**kwargs):
        super().__init__()
        if kwargs.get('use_dora') or kwargs.get('fan_in_fan_out') or kwargs.get('is_target_conv_1d_layer'):
            raise ValueError('DoRA and transposed Conv1D weights require a separate adapter contract')
        if not isinstance(base_layer,(nn.Linear,nn.Conv2d,nn.Embedding)):
            raise TypeError('LoRA supports Linear, Conv2d and Embedding')
        self.base_layer=deepcopy(base_layer)
        self.base_layer.requires_grad_(False)
        self.r={};self.lora_alpha={};self.scaling={}
        self.lora_A=nn.ModuleDict();self.lora_B=nn.ModuleDict();self.lora_dropout=nn.ModuleDict()
        self.lora_embedding_A=nn.ParameterDict();self.lora_embedding_B=nn.ParameterDict()
        self._active_adapter=[adapter_name];self._disable_adapters=False;self.merged_adapters=[]
        self._merge_snapshot=None
        self.update_layer(adapter_name,r,lora_alpha,lora_dropout,init_lora_weights)
        self.train(base_layer.training)

    @property
    def active_adapters(self):return list(self._active_adapter)
    @property
    def merged(self):return bool(self.merged_adapters)
    @property
    def disable_adapters(self):return self._disable_adapters
    def get_base_layer(self):return self.base_layer
    def set_adapter(self,adapter_names):
        if self.merged:self.unmerge()
        names=[adapter_names] if isinstance(adapter_names,str) else list(adapter_names)
        if any(n not in self.r for n in names):raise KeyError('Unknown adapter')
        self._active_adapter=names
    def enable_adapters(self,enabled=True):
        if self.merged:self.unmerge()
        self._disable_adapters=not enabled

    def update_layer(self,adapter_name,r,lora_alpha=1,lora_dropout=0.,init_lora_weights=True):
        if isinstance(r,bool) or not isinstance(r,int) or r<=0:raise ValueError('r must be a positive integer')
        if not math.isfinite(lora_alpha):raise ValueError('lora_alpha must be finite')
        if not 0<=lora_dropout<1:raise ValueError('lora_dropout must be in [0,1)')
        if init_lora_weights not in (True,False,'gaussian'):raise ValueError('Unsupported LoRA initialization')
        if self.merged:raise ValueError('Unmerge before updating adapters')
        base=self.base_layer;opts={'device':base.weight.device,'dtype':base.weight.dtype}
        self.r[adapter_name]=r;self.lora_alpha[adapter_name]=lora_alpha;self.scaling[adapter_name]=lora_alpha/r
        self.lora_dropout[adapter_name]=nn.Dropout(lora_dropout) if lora_dropout else nn.Identity()
        if isinstance(base,nn.Embedding):
            self.lora_embedding_A[adapter_name]=nn.Parameter(torch.empty(base.num_embeddings,r,**opts))
            self.lora_embedding_B[adapter_name]=nn.Parameter(torch.empty(r,base.embedding_dim,**opts))
        elif isinstance(base,nn.Conv2d):
            self.lora_A[adapter_name]=nn.Conv2d(base.in_channels,r*base.groups,base.kernel_size,base.stride,
                base.padding,base.dilation,base.groups,False,base.padding_mode,**opts)
            self.lora_B[adapter_name]=nn.Conv2d(r*base.groups,base.out_channels,1,groups=base.groups,bias=False,**opts)
        else:
            self.lora_A[adapter_name]=nn.Linear(base.in_features,r,bias=False,**opts)
            self.lora_B[adapter_name]=nn.Linear(r,base.out_features,bias=False,**opts)
        self.reset_lora_parameters(adapter_name,init_lora_weights)

    def reset_lora_parameters(self,adapter,init_lora_weights=True):
        if adapter in self.lora_embedding_A:
            nn.init.normal_(self.lora_embedding_A[adapter]);nn.init.zeros_(self.lora_embedding_B[adapter])
            if not init_lora_weights:nn.init.normal_(self.lora_embedding_B[adapter])
        else:
            a=self.lora_A[adapter].weight;b=self.lora_B[adapter].weight
            if init_lora_weights=='gaussian':nn.init.normal_(a,std=1/self.r[adapter])
            else:nn.init.kaiming_uniform_(a,a=math.sqrt(5))
            nn.init.zeros_(b) if init_lora_weights else nn.init.kaiming_uniform_(b,a=math.sqrt(5))

    def get_delta_weight(self,adapter):
        base=self.base_layer
        if isinstance(base,nn.Embedding):delta=self.lora_embedding_A[adapter]@self.lora_embedding_B[adapter]
        else:
            a=self.lora_A[adapter].weight;b=self.lora_B[adapter].weight
            if isinstance(base,nn.Conv2d):
                g=base.groups;r=self.r[adapter]
                delta=torch.bmm(b.reshape(g,base.out_channels//g,r),a.reshape(g,r,-1)).reshape_as(base.weight)
            else:delta=b@a
        return delta*self.scaling[adapter]

    def merge(self,safe_merge=False,adapter_names=None):
        if self.training:raise ValueError('Merge is an inference operation; call eval() first')
        if isinstance(self.base_layer,nn.Embedding) and self.base_layer.max_norm is not None:
            raise ValueError('Embedding max_norm modifies weights; merge is unsupported for this combination')
        names=self.active_adapters if adapter_names is None else list(adapter_names)
        if any(n not in self.r for n in names):raise KeyError('Unknown adapter')
        new=[n for n in names if n not in self.merged_adapters]
        if not new:return
        with torch.no_grad():
            candidate=self.base_layer.weight.detach().clone()
            for n in new:candidate+=self.get_delta_weight(n)
            if safe_merge and not torch.isfinite(candidate).all():raise ValueError('Nonfinite merged weights')
            if self._merge_snapshot is None:self._merge_snapshot=self.base_layer.weight.detach().clone()
            self.base_layer.weight.copy_(candidate)
        self.merged_adapters.extend(new)

    def unmerge(self):
        if not self.merged:return
        with torch.no_grad():self.base_layer.weight.copy_(self._merge_snapshot)
        self._merge_snapshot=None;self.merged_adapters=[]

    def forward(self,x,*args,**kwargs):
        if kwargs.get('adapter_names') is not None:raise ValueError('Mixed adapter batches are unsupported')
        if self.merged and self.training:raise ValueError('Unmerge before training')
        result=self.base_layer(x,*args,**kwargs)
        if self._disable_adapters:return result
        for adapter in self.active_adapters:
            if adapter in self.merged_adapters:continue
            if isinstance(self.base_layer,nn.Embedding):
                update=F.embedding(x,self.lora_embedding_A[adapter],padding_idx=self.base_layer.padding_idx,
                                   scale_grad_by_freq=self.base_layer.scale_grad_by_freq)@self.lora_embedding_B[adapter]
                update=self.lora_dropout[adapter](update)
            else:update=self.lora_B[adapter](self.lora_A[adapter](self.lora_dropout[adapter](x)))
            result=result+update*self.scaling[adapter]
        return result


class Linear(LoRALayer):
    def __init__(self,base_layer,adapter_name='default',r=0,lora_alpha=1,lora_dropout=0.,fan_in_fan_out=False,
                 is_target_conv_1d_layer=False,init_lora_weights=True,**kwargs):
        if not isinstance(base_layer,nn.Linear):raise TypeError('Linear adapter requires nn.Linear')
        super().__init__(base_layer,adapter_name,r,lora_alpha,lora_dropout,init_lora_weights,
                         fan_in_fan_out=fan_in_fan_out,is_target_conv_1d_layer=is_target_conv_1d_layer,**kwargs)


class Embedding(LoRALayer):
    def __init__(self,base_layer,*args,**kwargs):
        if not isinstance(base_layer,nn.Embedding):raise TypeError('Embedding adapter requires nn.Embedding')
        super().__init__(base_layer,*args,**kwargs)


class Conv2d(LoRALayer):
    def __init__(self,base_layer,*args,**kwargs):
        if not isinstance(base_layer,nn.Conv2d):raise TypeError('Conv2d adapter requires nn.Conv2d')
        super().__init__(base_layer,*args,**kwargs)


def apply_lora(model,rank=4,lora_alpha=1,target_layers=None,adapter_name='default',lora_dropout=0.):
    """Return an independent model with adapters at explicitly selected layer paths."""
    result=deepcopy(model)
    mapping={nn.Linear:Linear,nn.Embedding:Embedding,nn.Conv2d:Conv2d}
    if type(result) in mapping and (target_layers is None or list(target_layers)==['']):
        return mapping[type(result)](result,adapter_name,r=rank,lora_alpha=lora_alpha,lora_dropout=lora_dropout)
    prepared=[]
    for name,layer in result.named_modules():
        if name and type(layer) in mapping and (target_layers is None or name in target_layers):
            prepared.append((name,mapping[type(layer)](layer,adapter_name,r=rank,lora_alpha=lora_alpha,lora_dropout=lora_dropout)))
    if target_layers is not None and set(target_layers)!={n for n,_ in prepared}:
        raise ValueError('Every target layer must be a supported module path')
    for name,layer in prepared:
        parent_path,_,leaf=name.rpartition('.')
        setattr(result.get_submodule(parent_path) if parent_path else result,leaf,layer)
    return result


def transpose(weight,fan_in_fan_out):return weight.T if fan_in_fan_out else weight


class LoRAParametrization(nn.Module):
    def __init__(self,features_in,features_out,rank=1,alpha=1,device='cpu'):
        super().__init__()
        if rank<=0:raise ValueError('rank must be positive')
        self.lora_A=nn.Parameter(torch.randn(rank,features_out,device=device)/math.sqrt(rank))
        self.lora_B=nn.Parameter(torch.zeros(features_in,rank,device=device))
        self.scale=alpha/rank;self.enabled=True
    def forward(self,original_weights):
        return original_weights+(self.lora_B@self.lora_A).to(original_weights)*self.scale if self.enabled else original_weights


def linear_layer_parameterization(layer,device,rank=1,lora_alpha=1):
    return LoRAParametrization(*layer.weight.shape,rank=rank,alpha=lora_alpha,device=device)
