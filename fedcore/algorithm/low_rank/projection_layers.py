"""Executable P3 affine projections and a deliberately explicit CTR graph."""
import torch
from torch import nn
from torch.nn import functional as F

from .approximation import NumericalDomainViolation
from .structured_layers import FactorizedLinear
from .structured_profiles import FactorPair


class FixedProjectorLinear(nn.Module):
    """Full W is trainable; P is a fixed orthonormal buffer until compilation."""
    def __init__(self, in_features, out_features, rank, bias=True, *, device=None, dtype=None):
        super().__init__()
        if any(type(v) is not int or v <= 0 for v in (in_features,out_features,rank)) or rank > in_features:
            raise NumericalDomainViolation("Invalid fixed-projector dimensions")
        self.in_features, self.out_features, self.rank = in_features,out_features,rank
        self.weight = nn.Parameter(torch.zeros((out_features,in_features),device=device,dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(out_features,device=device,dtype=dtype)) if bias else None
        self.register_buffer("projection",torch.eye(in_features,rank,device=device,dtype=dtype))

    @classmethod
    def from_linear(cls, layer, basis):
        if type(layer) is not nn.Linear or basis.ndim != 2 or basis.shape[0] != layer.in_features:
            raise NumericalDomainViolation("Fixed projector requires a standard Linear and compatible basis")
        p = basis.detach().to(layer.weight)
        tolerance = 1e-5 if p.dtype == torch.float32 else 1e-10
        if not bool(torch.isfinite(p).all()) or not torch.allclose(p.T@p,torch.eye(p.shape[1],dtype=p.dtype,device=p.device),rtol=tolerance,atol=tolerance):
            raise NumericalDomainViolation("Projector columns must be orthonormal")
        result = cls(layer.in_features,layer.out_features,p.shape[1],layer.bias is not None,
                     device=layer.weight.device,dtype=layer.weight.dtype)
        with torch.no_grad():
            result.weight.copy_(layer.weight)
            result.projection.copy_(p)
            if layer.bias is not None:
                result.bias.copy_(layer.bias)
        result.weight.requires_grad_(layer.weight.requires_grad)
        if result.bias is not None:
            result.bias.requires_grad_(layer.bias.requires_grad)
        result.train(layer.training)
        return result

    def forward(self, x):
        return F.linear(F.linear(F.linear(x,self.projection.T),self.projection),self.weight,self.bias)

    def compact(self):
        result = FactorizedLinear.from_factors(FactorPair(
            (self.weight.detach()@self.projection).clone(),self.projection.T.detach().clone()),
            None if self.bias is None else self.bias.detach())
        result.train(self.training)
        result.left.weight.requires_grad_(self.weight.requires_grad)
        result.right.weight.requires_grad_(self.weight.requires_grad)
        if result.left.bias is not None:
            result.left.bias.requires_grad_(self.bias.requires_grad)
        return result

    def representation_config(self):
        return {"layer_type":"FixedProjectorLinear","in_features":self.in_features,
                "out_features":self.out_features,"rank":self.rank,"bias":self.bias is not None,"version":1}


class CenteredPCAEmbedding(nn.Module):
    """Return compact rows; decode selected rows only for interaction consumers."""
    def __init__(self,num_embeddings,embedding_dim,rank,*,device=None,dtype=None):
        super().__init__()
        if any(type(v) is not int or v <= 0 for v in (num_embeddings,embedding_dim,rank)) or rank > min(num_embeddings,embedding_dim):
            raise NumericalDomainViolation("Invalid CTR table dimensions")
        self.num_embeddings,self.embedding_dim,self.rank = num_embeddings,embedding_dim,rank
        self.coefficients = nn.Parameter(torch.zeros((num_embeddings,rank),device=device,dtype=dtype))
        self.register_buffer("basis",torch.eye(embedding_dim,rank,device=device,dtype=dtype))
        self.register_buffer("mean",torch.zeros(embedding_dim,device=device,dtype=dtype))
        self.register_buffer("affine_correction",torch.zeros(embedding_dim,device=device,dtype=dtype))

    @classmethod
    def from_factors(cls, factors):
        table,basis = factors.coefficients,factors.basis
        result = cls(len(table),len(basis),table.shape[1],device=table.device,dtype=table.dtype)
        with torch.no_grad():
            result.coefficients.copy_(table)
            result.basis.copy_(basis)
            result.mean.copy_(factors.mean)
            result.affine_correction.copy_(factors.affine_correction)
        return result

    def forward(self,tokens):
        return F.embedding(tokens,self.coefficients)

    def decode(self,compact):
        return F.linear(compact,self.basis,self.mean)

    def representation_config(self):
        return {"layer_type":"CenteredPCAEmbedding","num_embeddings":self.num_embeddings,
                "embedding_dim":self.embedding_dim,"rank":self.rank,"version":1,
                "coefficient_convention":"centered_(E-mean)U","decoder_offset":"mean"}


class CTRLinearSum(nn.Module):
    """Sum all independently projected field contributions and a single bias."""
    def __init__(self, field_widths, out_features, bias=True, *, device=None,dtype=None):
        super().__init__()
        widths = dict(field_widths)
        if not widths or any(not isinstance(k,str) or not k or type(v) is not int or v < 1 for k,v in widths.items()):
            raise NumericalDomainViolation("CTR linear sum needs named positive field widths")
        self.blocks = nn.ModuleDict({name:nn.Linear(width,out_features,bias=False,device=device,dtype=dtype) for name,width in widths.items()})
        self.bias = nn.Parameter(torch.zeros(out_features,device=device,dtype=dtype)) if bias else None
        self.out_features = out_features

    def forward(self,fields):
        if set(fields) != set(self.blocks):
            raise NumericalDomainViolation("Every linear field consumer must be supplied exactly once")
        outputs = [block(fields[name]) for name,block in self.blocks.items()]
        result = torch.stack(outputs).sum(0)
        return result if self.bias is None else result+self.bias

    def representation_config(self):
        return {"layer_type":"CTRLinearSum","field_widths":{k:v.in_features for k,v in self.blocks.items()},
                "out_features":self.out_features,"bias":self.bias is not None,"version":1}


class CTRFieldGraph(nn.Module):
    """Allowlisted field graph: one additive Linear and declared dot products.

    This class is the architecture boundary, not a claim to handle arbitrary
    CTR forward methods. Each field must have exactly one linear consumer;
    interaction pairs are the only additional consumers in the executable
    graph. Arbitrary subclasses/extra children are rejected before transform.
    """
    def __init__(self, fields, linear, interaction_pairs=()):
        super().__init__()
        self.fields = nn.ModuleDict(dict(fields))
        self.linear = linear
        self.interaction_pairs = tuple(tuple(pair) for pair in interaction_pairs)
        self.validate_consumers()

    def validate_consumers(self):
        if type(self) is not CTRFieldGraph or set(self._modules) != {"fields","linear"} or type(self.linear) is not CTRLinearSum:
            raise NumericalDomainViolation("CTRAFM requires the complete explicit allowlisted field graph")
        if not self.fields or set(self.fields) != set(self.linear.blocks) or set(self.linear._modules) != {"blocks"}:
            raise NumericalDomainViolation("All fields require their declared linear consumer")
        for name, table in self.fields.items():
            if type(table) not in (nn.Embedding,CenteredPCAEmbedding):
                raise NumericalDomainViolation("Unknown CTR field table consumer semantics")
            if type(table) is nn.Embedding and (table.padding_idx is not None or table.max_norm is not None or table.scale_grad_by_freq or table.sparse):
                raise NumericalDomainViolation("Initial CTR profile requires ordinary dense embedding semantics")
            width = table.rank if isinstance(table,CenteredPCAEmbedding) else table.embedding_dim
            if type(self.linear.blocks[name]) is not nn.Linear or self.linear.blocks[name].in_features != width or self.linear.blocks[name].out_features != self.linear.out_features:
                raise NumericalDomainViolation("CTR field and additive linear consumer widths disagree")
        if any(len(pair) != 2 or any(name not in self.fields for name in pair) or self.fields[pair[0]].embedding_dim != self.fields[pair[1]].embedding_dim for pair in self.interaction_pairs):
            raise NumericalDomainViolation("Dot-product pairs require declared fields of equal decoded width")
        return tuple((name,("linear",)+tuple(f"dot:{i}" for i,pair in enumerate(self.interaction_pairs) if name in pair)) for name in self.fields)

    def forward(self,tokens):
        if isinstance(tokens,torch.Tensor):
            if tokens.ndim != 2 or tokens.shape[1] != len(self.fields) or tokens.dtype not in (torch.int32,torch.int64):
                raise NumericalDomainViolation("CTR tensor inputs require [batch, field] integer ids in declared order")
            tokens = {name:tokens[:,index] for index,name in enumerate(self.fields)}
        if not isinstance(tokens,dict) or set(tokens) != set(self.fields):
            raise NumericalDomainViolation("Supply one token tensor per declared CTR field")
        values = {name:table(tokens[name]) for name,table in self.fields.items()}
        result = {"linear":self.linear(values)}
        if self.interaction_pairs:
            decoded = {name:(table.decode(values[name]) if isinstance(table,CenteredPCAEmbedding) else values[name]) for name,table in self.fields.items()}
            result["interactions"] = torch.stack([(decoded[a]*decoded[b]).sum(-1) for a,b in self.interaction_pairs],dim=-1)
        return result

    def representation_config(self):
        return {"layer_type":"CTRFieldGraph","interaction_pairs":[list(p) for p in self.interaction_pairs],"version":1}


class ReLUFactorizedLinear(FactorizedLinear):
    """CTR author architecture variant: never linearly equivalent to LR factors."""
    def forward(self,x):
        return self.left(F.relu(self.right(x)))

    def representation_config(self):
        return {**super().representation_config(),"layer_type":"ReLUFactorizedLinear","version":1,
                "nonlinear_variant":"relu_between_factors_v1"}
