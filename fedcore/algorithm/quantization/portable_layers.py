"""Portable reference execution. Packed storage does not promise integer GEMM."""
import torch
from torch import nn
from torch.nn import functional as F
from .tensor_formats import PackedTensor, QuantizedTensorSpec


def _input(x, width):
    if (x.device.type != "cpu" or x.dtype not in (torch.float32, torch.float64)
            or x.ndim < 1 or x.shape[-1] != width or not bool(torch.isfinite(x).all())):
        raise ValueError("Compatible finite CPU floating input required")


class PortableQuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, *, spec=QuantizedTensorSpec()):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = PackedTensor((out_features, in_features), spec)
        self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float64) if bias else None)

    @classmethod
    def from_weight(cls, weight, bias=None, *, spec=QuantizedTensorSpec(), generator=None):
        result = cls(weight.shape[1], weight.shape[0], bias is not None, spec=spec)
        result.weight = PackedTensor.from_tensor(weight, spec, generator=generator)
        if bias is not None:
            if bias.shape != (weight.shape[0],) or not bool(torch.isfinite(bias).all()):
                raise ValueError("Invalid bias")
            result.bias = bias.detach().double().clone()
        return result

    def forward(self, x):
        _input(x, self.in_features)
        return F.linear(x.double(), self.weight.decode(), self.bias).to(x.dtype)


class PortableQuantizedFactorLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True, *, left_spec=QuantizedTensorSpec(), right_spec=QuantizedTensorSpec()):
        super().__init__()
        self.in_features, self.out_features, self.rank = in_features, out_features, rank
        self.left = PackedTensor((out_features, rank), left_spec)
        self.right = PackedTensor((rank, in_features), right_spec)
        self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float64) if bias else None)

    @classmethod
    def from_factors(cls, left, right, bias=None, *, left_spec=QuantizedTensorSpec(), right_spec=QuantizedTensorSpec(), generator=None):
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
            raise ValueError("Factor coordinates disagree")
        result = cls(right.shape[1], left.shape[0], left.shape[1], bias is not None, left_spec=left_spec, right_spec=right_spec)
        result.left = PackedTensor.from_tensor(left, left_spec, generator=generator)
        result.right = PackedTensor.from_tensor(right, right_spec, generator=generator)
        if bias is not None:
            if bias.shape != (left.shape[0],) or not bool(torch.isfinite(bias).all()):
                raise ValueError("Invalid bias")
            result.bias = bias.detach().double().clone()
        return result

    def forward(self, x):
        _input(x, self.in_features)
        return F.linear(F.linear(x.double(), self.right.decode()), self.left.decode(), self.bias).to(x.dtype)


class QuantizedSumLinear(nn.Module):
    """Q + LR with a single explicit bias on the dense branch."""
    def __init__(self, base, correction):
        super().__init__()
        if (type(base) is not PortableQuantizedLinear or type(correction) is not PortableQuantizedFactorLinear
                or correction.bias is not None or (base.in_features, base.out_features) != (correction.in_features, correction.out_features)):
            raise ValueError("Compatible Q+LR branches and exactly one bias required")
        self.base, self.correction = base, correction
        self.in_features, self.out_features = base.in_features, base.out_features

    def forward(self, x):
        return self.base(x) + self.correction(x)
