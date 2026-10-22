"""Portable factor modules with explicit residual/shared/vocabulary topology.

Every constructor has a JSON-safe ``representation_config`` counterpart. Loading
weights needs the constructor topology too: a state_dict alone cannot declare
that two different operators share one Parameter or one embedding table.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .approximation import NumericalDomainViolation
from .structured_profiles import FactorPair, SharedBasisFactors, GroupReduceFactors


def _dimensions(*values):
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("module dimensions must be positive integers")


def _pair_shape(pair):
    if (not isinstance(pair,FactorPair) or pair.left.ndim!=2 or pair.right.ndim!=2
            or not pair.left.numel() or not pair.right.numel()
            or pair.left.shape[1]!=pair.right.shape[0]
            or pair.left.dtype not in (torch.float32,torch.float64)
            or pair.left.dtype!=pair.right.dtype or pair.left.device!=pair.right.device
            or not bool(torch.isfinite(pair.left).all()) or not bool(torch.isfinite(pair.right).all())):
        raise ValueError("finite compatible nonempty FactorPair required")
    return pair.right.shape[1],pair.left.shape[0],pair.rank


def _bias(bias, out_features, reference):
    if bias is not None and (not isinstance(bias,torch.Tensor) or bias.shape!=(out_features,)
                             or not bool(torch.isfinite(bias).all())):
        raise ValueError("bias must be a finite output vector")
    return None if bias is None else nn.Parameter(bias.detach().clone().to(reference))


class FactorizedLinear(nn.Module):
    """Two real Linear operators, suitable for the existing per-factor LoRA."""

    def __init__(self,in_features,out_features,rank,bias=True,*,device=None,dtype=None):
        super().__init__()
        _dimensions(in_features,out_features,rank)
        self.in_features=in_features; self.out_features=out_features; self.rank=rank
        self.right=nn.Linear(in_features,rank,bias=False,device=device,dtype=dtype)
        self.left=nn.Linear(rank,out_features,bias=bias,device=device,dtype=dtype)

    @classmethod
    def from_factors(cls,pair,bias=None):
        in_features,out_features,rank=_pair_shape(pair)
        result=cls(in_features,out_features,rank,bias is not None,
                   device=pair.left.device,dtype=pair.left.dtype)
        with torch.no_grad():
            result.left.weight.copy_(pair.left); result.right.weight.copy_(pair.right)
            if bias is not None:
                result.left.bias.copy_(_bias(bias,out_features,pair.left))
        return result

    def forward(self,x):
        return self.left(self.right(x))

    def representation_config(self):
        left=self.left if isinstance(self.left,nn.Linear) else self.left.get_base_layer()
        return {"layer_type":"FactorizedLinear","in_features":self.in_features,
                "out_features":self.out_features,"rank":self.rank,
                "bias":left.bias is not None,"version":1}


class ResidualLinear(nn.Module):
    """Keep the original base object; a correction has no independent bias."""

    def __init__(self,base,residual: FactorPair,*,freeze_base=True):
        super().__init__()
        in_features,out_features,rank=_pair_shape(residual)
        if (not isinstance(base,nn.Module) or getattr(base,"in_features",None)!=in_features
                or getattr(base,"out_features",None)!=out_features):
            raise ValueError("base must explicitly expose compatible Linear input/output dimensions")
        self.in_features=in_features; self.out_features=out_features; self.residual_rank=rank
        self.base=base
        if freeze_base:
            self.base.requires_grad_(False)
        self.freeze_base=bool(freeze_base)
        self.residual=FactorizedLinear.from_factors(residual)

    @classmethod
    def for_training(cls,base,rank,*,seed=0):
        """Zero output with one nonzero factor, enabling the first gradient."""
        _dimensions(rank)
        parameter=next(base.parameters())
        generator=torch.Generator(device=parameter.device).manual_seed(seed)
        right=torch.randn((rank,base.in_features),generator=generator,
                          device=parameter.device,dtype=parameter.dtype)/math.sqrt(base.in_features)
        left=parameter.new_zeros((base.out_features,rank))
        return cls(base,FactorPair(left,right),freeze_base=True)

    def validate_trainable_initialization(self):
        left=self.residual.left.weight; right=self.residual.right.weight
        if (left.requires_grad and right.requires_grad
                and not bool(torch.count_nonzero(left)) and not bool(torch.count_nonzero(right))):
            raise ValueError("two zero trainable residual factors have zero gradients")

    def forward(self,x):
        return self.base(x)+self.residual(x)

    def representation_config(self):
        # The base has a separate recursive constructor record; packed formats
        # require their own allowlisted runtime rather than a dense replacement.
        return {"layer_type":"ResidualLinear","in_features":self.in_features,
                "out_features":self.out_features,"residual_rank":self.residual_rank,
                "freeze_base":self.freeze_base,"base_type":type(self.base).__name__,"version":1}


class SharedBasisLinear(nn.Module):
    """Distinct Linear consumers can physically share this right Parameter."""

    def __init__(self,in_features,out_features,rank,bias=True,*,basis=None,device=None,dtype=None):
        super().__init__()
        _dimensions(in_features,out_features,rank)
        self.in_features=in_features; self.out_features=out_features; self.rank=rank
        if basis is not None and (not isinstance(basis,nn.Parameter) or basis.shape!=(rank,in_features)):
            raise ValueError("shared basis must be one Parameter of shape [rank, in]")
        self.right=basis if basis is not None else nn.Parameter(torch.empty((rank,in_features),device=device,dtype=dtype))
        self.left=nn.Parameter(torch.empty((out_features,rank),device=self.right.device,dtype=self.right.dtype))
        self.bias=nn.Parameter(torch.zeros(out_features,device=self.right.device,dtype=self.right.dtype)) if bias else None
        nn.init.kaiming_uniform_(self.left,a=math.sqrt(5))
        if basis is None:
            nn.init.kaiming_uniform_(self.right,a=math.sqrt(5))

    def forward(self,x):
        return F.linear(F.linear(x,self.right),self.left,self.bias)

    def representation_config(self):
        return {"layer_type":"SharedBasisLinear","in_features":self.in_features,
                "out_features":self.out_features,"rank":self.rank,"bias":self.bias is not None,"version":1}


def create_shared_basis_consumers(factors: SharedBasisFactors,biases=None):
    """Build independent consumers for atomic replacement of original paths."""
    if not isinstance(factors,SharedBasisFactors) or not factors.lefts:
        raise ValueError("nonempty SharedBasisFactors required")
    biases=(None,)*len(factors.lefts) if biases is None else tuple(biases)
    if len(biases)!=len(factors.lefts):
        raise ValueError("one optional bias per shared consumer required")
    basis=nn.Parameter(factors.right.detach().clone())
    consumers=[]
    for left,bias in zip(factors.lefts,biases):
        _pair_shape(FactorPair(left,factors.right))
        consumer=SharedBasisLinear(basis.shape[1],left.shape[0],basis.shape[0],
                                   bias is not None,basis=basis)
        with torch.no_grad():
            consumer.left.copy_(left)
            if bias is not None:
                consumer.bias.copy_(_bias(bias,left.shape[0],basis))
        consumers.append(consumer)
    return tuple(consumers)


def restore_shared_basis(consumers):
    """Restore constructor topology after loading equal saved basis tensors.

    A mismatch is an invalid checkpoint, never a request to average weights.
    Rebinding must run before constructing any optimizer.
    """
    consumers=tuple(consumers)
    if not consumers or any(not isinstance(c,SharedBasisLinear) for c in consumers):
        raise ValueError("nonempty SharedBasisLinear consumers required")
    common=consumers[0].right
    if any(c.right.shape!=common.shape or c.right.dtype!=common.dtype or c.right.device!=common.device
           or not torch.equal(c.right,common) for c in consumers):
        raise ValueError("saved shared basis values disagree")
    for consumer in consumers[1:]:
        consumer.right=common
    return common


class GroupedEmbedding(nn.Module):
    """Lookup executes the selected group's short row; no full table is built."""

    def __init__(self,num_embeddings,embedding_dim,group_sizes,ranks,*,padding_idx=None,device=None,dtype=None):
        super().__init__()
        _dimensions(num_embeddings,embedding_dim)
        sizes=tuple(group_sizes); ranks=tuple(ranks)
        if (not sizes or len(sizes)!=len(ranks) or sum(sizes)!=num_embeddings
                or any(type(n) is not int or n<0 for n in sizes)
                or any(type(r) is not int or (r!=0 if n==0 else not 1<=r<=min(n,embedding_dim)) for n,r in zip(sizes,ranks))):
            raise ValueError("valid group sizes/ranks must partition the vocabulary")
        if padding_idx is not None and (type(padding_idx) is not int or not 0<=padding_idx<num_embeddings):
            raise ValueError("padding_idx must be a vocabulary index")
        self.num_embeddings=num_embeddings; self.embedding_dim=embedding_dim
        self.group_sizes=sizes; self.ranks=ranks; self.padding_idx=padding_idx
        self.coefficients=nn.ParameterList([nn.Parameter(torch.zeros((n,r),device=device,dtype=dtype)) for n,r in zip(sizes,ranks)])
        self.bases=nn.ParameterList([nn.Parameter(torch.zeros((r,embedding_dim),device=device,dtype=dtype)) for r in ranks])
        groups=torch.repeat_interleave(torch.arange(len(sizes),device=device),torch.tensor(sizes,device=device))
        local=torch.cat([torch.arange(n,device=device) for n in sizes])
        self.register_buffer("token_to_group",groups.to(torch.int64))
        self.register_buffer("token_to_local",local.to(torch.int64))

    @classmethod
    def from_factors(cls,factors: GroupReduceFactors,*,padding_idx=None):
        if not isinstance(factors,GroupReduceFactors) or not factors.factors:
            raise ValueError("GroupReduceFactors required")
        reference=factors.factors[0].right
        result=cls(len(factors.token_to_group),reference.shape[1],
                   [len(tokens) for tokens in factors.group_tokens],
                   [pair.rank for pair in factors.factors],padding_idx=padding_idx,
                   device=reference.device,dtype=reference.dtype)
        with torch.no_grad():
            result.token_to_group.copy_(factors.token_to_group)
            result.token_to_local.copy_(factors.token_to_local)
            for coefficient,basis,pair in zip(result.coefficients,result.bases,factors.factors):
                coefficient.copy_(pair.left); basis.copy_(pair.right)
        result.validate_vocabulary_maps()
        return result

    def validate_vocabulary_maps(self):
        groups=self.token_to_group; local=self.token_to_local
        if (groups.shape!=(self.num_embeddings,) or local.shape!=(self.num_embeddings,)
                or groups.dtype!=torch.int64 or local.dtype!=torch.int64
                or bool(((groups<0)|(groups>=len(self.group_sizes))).any())):
            raise ValueError("invalid portable token maps")
        for group,size in enumerate(self.group_sizes):
            selected=local[groups==group]
            if len(selected)!=size or not torch.equal(selected.sort().values,torch.arange(size,device=selected.device)):
                raise ValueError("token maps do not bijectively enumerate group rows")

    def _load_from_state_dict(self,*args,**kwargs):
        super()._load_from_state_dict(*args,**kwargs)
        self.validate_vocabulary_maps()

    def forward(self,tokens):
        if not torch.jit.is_tracing():
            if tokens.dtype not in (torch.int32,torch.int64):
                raise ValueError("embedding lookup requires integer token ids")
            if bool(((tokens<0)|(tokens>=self.num_embeddings)).any()):
                raise IndexError("embedding token outside vocabulary")
        flat=tokens.reshape(-1).long()
        group_ids=self.token_to_group[flat]; local_ids=self.token_to_local[flat]
        reference=self.coefficients[0]
        result=reference.new_zeros((flat.numel(),self.embedding_dim))
        for group,(coefficients,basis) in enumerate(zip(self.coefficients,self.bases)):
            # Nonzero-index selection is traceable and does not materialize VxD.
            locations=torch.nonzero(group_ids==group,as_tuple=False).flatten()
            selected=local_ids[locations]
            values=F.embedding(selected,coefficients)@basis
            if self.padding_idx is not None:
                # Preserve the stored padding value while disabling its gradient,
                # matching ordinary Embedding's padding-row behavior.
                padding=(flat[locations]==self.padding_idx)[:,None]
                values=torch.where(padding,values.detach(),values)
            result=result.index_copy(0,locations,values)
        return result.reshape(*tokens.shape,self.embedding_dim)

    def logits(self,hidden):
        if hidden.shape[-1]!=self.embedding_dim:
            raise ValueError("LM head hidden dimension must match embedding dimension")
        group_outputs=[F.linear(F.linear(hidden,basis),coefficients)
                       for coefficients,basis in zip(self.coefficients,self.bases)]
        grouped=torch.cat(group_outputs,dim=-1)
        offsets=torch.tensor((0,)+tuple(self.group_sizes[:-1]),device=hidden.device,dtype=torch.int64).cumsum(0)
        # Each original vocabulary id maps to its position in grouped logits.
        order=offsets[self.token_to_group]+self.token_to_local
        return grouped.index_select(-1,order)

    def representation_config(self):
        return {"layer_type":"GroupedEmbedding","num_embeddings":self.num_embeddings,
                "embedding_dim":self.embedding_dim,"group_sizes":list(self.group_sizes),
                "ranks":list(self.ranks),"padding_idx":self.padding_idx,"version":1}

    @property
    def parameter_elements(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def map_bytes(self):
        return sum(t.numel()*t.element_size() for t in (self.token_to_group,self.token_to_local))


class GroupedLMHead(nn.Module):
    """The same table module supports a tied head in original vocabulary order."""

    def __init__(self,table: GroupedEmbedding,bias=None):
        super().__init__()
        if not isinstance(table,GroupedEmbedding):
            raise TypeError("GroupedLMHead requires GroupedEmbedding table")
        self.table=table
        self.in_features=table.embedding_dim; self.out_features=table.num_embeddings
        self.bias=_bias(bias,self.out_features,table.coefficients[0])

    def forward(self,hidden):
        result=self.table.logits(hidden)
        return result if self.bias is None else result+self.bias

    def representation_config(self):
        return {"layer_type":"GroupedLMHead","in_features":self.in_features,
                "out_features":self.out_features,"bias":self.bias is not None,"version":1}
