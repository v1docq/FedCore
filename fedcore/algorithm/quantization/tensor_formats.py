"""Versioned symmetric uniform CPU formats with actual little-endian packing.

The unused most-negative signed code is forbidden. Scale is float64; codes
are offset by qmax before packing. Padding bits must be zero, not arbitrary.
"""
from dataclasses import asdict, dataclass
from enum import Enum
import math

import torch
from torch import nn


class Rounding(str, Enum):
    NEAREST = "nearest_even"
    STOCHASTIC = "stochastic"


class Grid(str, Enum):
    SYMMETRIC = "symmetric_zero"
    ENDPOINT = "uniform_endpoints"


@dataclass(frozen=True)
class QuantizedTensorSpec:
    bits: int = 8
    axis: int | None = None
    rounding: Rounding = Rounding.NEAREST
    bound: float | None = None
    version: int = 1
    grid: Grid = Grid.SYMMETRIC

    def __post_init__(self):
        if (type(self.bits) is not int or self.bits not in (2, 4, 8)
                or self.axis is not None and (type(self.axis) is not int or self.axis not in (0, 1))
                or type(self.rounding) is not Rounding or type(self.grid) is not Grid or type(self.version) is not int or self.version != 1
                or self.bound is not None and (type(self.bound) not in (int, float)
                    or not math.isfinite(self.bound) or self.bound <= 0)):
            raise ValueError("Invalid uniform quantization specification")

    def payload(self):
        return {**asdict(self), "rounding": self.rounding.value, "grid": self.grid.value}

    @classmethod
    def parse(cls, value):
        if type(value) is not dict or set(value) != {"bits", "axis", "rounding", "bound", "version", "grid"}:
            raise ValueError("Unknown quantized tensor schema")
        return cls(**{**value, "rounding": Rounding(value["rounding"]), "grid": Grid(value["grid"])})


def pack_codes(codes, bits):
    """Pack unsigned row-major codes, first code in the least significant bits."""
    if type(bits) is not int or bits not in (2, 4, 8) or codes.device.type != "cpu" or codes.dtype != torch.int64:
        raise ValueError("CPU int64 codes and a supported bit width required")
    if bool(((codes < 0) | (codes >= 2**bits)).any()):
        raise ValueError("Code outside declared bit width")
    flat = codes.reshape(-1)
    count = 8 // bits
    padded = torch.zeros(((flat.numel() + count - 1) // count) * count, dtype=torch.int64)
    padded[:flat.numel()] = flat
    packed = torch.zeros(padded.numel() // count, dtype=torch.int64)
    for i in range(count):
        packed |= padded[i::count] << (i * bits)
    return packed.to(torch.uint8)


def unpack_codes(packed, bits, elements):
    if (type(bits) is not int or bits not in (2, 4, 8) or type(elements) is not int or elements < 1
            or packed.dtype != torch.uint8 or packed.device.type != "cpu" or packed.ndim != 1
            or packed.numel() != (elements * bits + 7) // 8):
        raise ValueError("Corrupt packed shape or container")
    count = 8 // bits
    flat = torch.stack([(packed.long() >> (i * bits)) & (2**bits - 1)
                        for i in range(count)], 1).flatten()
    if bool((flat[elements:] != 0).any()):
        raise ValueError("Nonzero padding bits")
    return flat[:elements]


def quantize_codes(value, spec, *, generator=None, scale=None):
    if (not isinstance(value, torch.Tensor) or value.ndim != 2 or value.device.type != "cpu"
            or value.dtype not in (torch.float32, torch.float64) or not value.numel()
            or not bool(torch.isfinite(value).all()) or type(spec) is not QuantizedTensorSpec):
        raise ValueError("Finite nonempty CPU float matrix required")
    qmax = 2**(spec.bits - 1) - 1
    x = value.detach().double()
    if scale is None:
        maximum = x.abs().max().reshape(1) if spec.axis is None else x.abs().amax(1-spec.axis)
        if spec.bound is not None:
            maximum = torch.full_like(maximum, spec.bound)
        denominator = qmax if spec.grid is Grid.SYMMETRIC else (2**spec.bits-1)/2
        scale = (torch.where(maximum == 0, torch.ones_like(maximum), maximum / denominator)
                 if spec.grid is Grid.SYMMETRIC else maximum / denominator)
    expected = (1 if spec.axis is None else value.shape[spec.axis],)
    if (type(scale) is not torch.Tensor or scale.device.type != "cpu" or scale.dtype != torch.float64
            or tuple(scale.shape) != expected or not bool(torch.isfinite(scale).all())
            or bool((scale < 0 if spec.grid is Grid.ENDPOINT else scale <= 0).any())):
        raise ValueError("Scale must match the declared axis and finite positive/zero-range policy")
    if spec.grid is Grid.ENDPOINT:
        zero = scale == 0
        maximum = x.abs().max().reshape(1) if spec.axis is None else x.abs().amax(1-spec.axis)
        if bool((zero & (maximum != 0)).any()):
            raise ValueError("Zero scale only represents an identically zero range")
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    divisor = safe_scale if spec.axis is None else safe_scale.unsqueeze(1-spec.axis)
    midpoint = (2**spec.bits-1)/2
    y = ((x / divisor).clamp(-qmax, qmax) if spec.grid is Grid.SYMMETRIC
         else (x / divisor + midpoint).clamp(0, 2**spec.bits-1))
    if spec.rounding is Rounding.STOCHASTIC:
        if not isinstance(generator, torch.Generator):
            raise ValueError("Stochastic rounding requires an owned generator")
        lower = y.floor()
        y = lower + (torch.rand(y.shape, generator=generator, dtype=y.dtype) < y-lower)
    else:
        y = y.round()
    return y.long(), scale.detach().double().clone().contiguous()


class PackedTensor(nn.Module):
    """Only compact codes and scale are retained; decode is an explicit CPU view."""
    def __init__(self, shape, spec=QuantizedTensorSpec()):
        super().__init__()
        if (type(shape) not in (tuple, list) or len(shape) != 2
                or any(type(n) is not int or n < 1 for n in shape) or math.prod(shape) > 2**26):
            raise ValueError("Bounded positive matrix shape required")
        self.shape, self.spec = tuple(shape), spec
        if type(spec) is not QuantizedTensorSpec:
            raise ValueError("Typed quantization specification required")
        size=(math.prod(shape)*spec.bits+7)//8+(1 if spec.axis is None else shape[spec.axis])*8
        if size>64*1024**2:raise ValueError('Packed tensor constructor exceeds portable 64 MiB component limit')
        self.register_buffer("packed", torch.zeros((math.prod(shape)*spec.bits+7)//8, dtype=torch.uint8))
        self.register_buffer("scale", torch.ones(1 if spec.axis is None else shape[spec.axis], dtype=torch.float64))
        # Offset codes representing zero, including the final zero padding bits.
        self.packed = pack_codes(torch.full(self.shape, 2**(spec.bits-1)-1, dtype=torch.int64), spec.bits)

    @classmethod
    def from_tensor(cls, value, spec=QuantizedTensorSpec(), *, generator=None, scale=None):
        codes, scale = quantize_codes(value, spec, generator=generator, scale=scale)
        result = cls(tuple(value.shape), spec)
        result.packed = pack_codes(codes + ((2**(spec.bits-1)-1) if spec.grid is Grid.SYMMETRIC else 0), spec.bits)
        result.scale = scale
        result.validate_state(result.state_dict())
        return result

    def codes(self):
        raw = unpack_codes(self.packed, self.spec.bits, math.prod(self.shape)).reshape(self.shape)
        return raw - (2**(self.spec.bits-1)-1) if self.spec.grid is Grid.SYMMETRIC else raw

    def decode(self):
        scale = self.scale if self.spec.axis is None else self.scale.unsqueeze(1-self.spec.axis)
        codes = self.codes().double()
        if self.spec.grid is Grid.ENDPOINT:
            codes = codes - (2**self.spec.bits-1)/2
        return codes * scale

    def validate_state(self, state, prefix=""):
        packed, scale = state[prefix+"packed"], state[prefix+"scale"]
        codes = unpack_codes(packed, self.spec.bits, math.prod(self.shape))
        if self.spec.grid is Grid.SYMMETRIC and bool((codes == 2**self.spec.bits-1).any()):
            raise ValueError("Reserved signed code")
        if (scale.dtype != torch.float64 or tuple(scale.shape) != tuple(self.scale.shape)
                or scale.device.type != "cpu" or not bool(torch.isfinite(scale).all())
                or bool((scale < 0 if self.spec.grid is Grid.ENDPOINT else scale <= 0).any())):
            raise ValueError("Invalid quantization scales")
        for value in (packed, scale):
            if (not value.is_contiguous() or value.storage_offset() != 0
                    or value.untyped_storage().nbytes() != value.numel()*value.element_size()):
                raise ValueError("Packed tensors require minimal contiguous storage")

    def config(self):
        return {"shape": list(self.shape), "spec": self.spec.payload()}

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        try:
            self.validate_state(state_dict, prefix)
        except (ValueError, KeyError, TypeError) as error:
            error_msgs.append(str(error))
            return
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
