"""Portable zero-rank endpoint; search-only modules never enter checkpoints."""
import torch
from torch import nn


class UnsupportedLearnedAccessError(ValueError):
    """A parent bypasses a transient module through a dense weight read."""


class ZeroLinear(nn.Module):
    """A zero linear operator plus an optional bias, without a dense weight."""

    def __init__(self, in_features, out_features, bias=True, *, device=None, dtype=None):
        super().__init__()
        if any(type(value) is not int or value < 1 for value in (in_features, out_features)) or type(bias) is not bool:
            raise ValueError("ZeroLinear requires positive dimensions and an explicit bias flag")
        self.in_features, self.out_features, self.rank = in_features, out_features, 0
        self.register_buffer("_anchor", torch.empty(0, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype)) if bias else None

    @classmethod
    def from_linear(cls, layer):
        result = cls(layer.in_features, layer.out_features, layer.bias is not None,
                     device=layer.weight.device, dtype=layer.weight.dtype)
        if layer.bias is not None:
            with torch.no_grad():
                result.bias.copy_(layer.bias)
        return result

    def forward(self, inputs):
        if inputs.shape[-1] != self.in_features:
            raise ValueError("ZeroLinear input width mismatch")
        result = inputs.new_zeros((*inputs.shape[:-1], self.out_features))
        return result if self.bias is None else result + self.bias

    def representation_config(self):
        return {"layer_type": "ZeroLinear", "version": 1, "in_features": self.in_features,
                "out_features": self.out_features, "bias": self.bias is not None}
