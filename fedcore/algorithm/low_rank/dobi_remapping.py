"""Portable mixed precision Dobi storage, explicitly distinct from author bnb.

Paper https://arxiv.org/html/2502.02723v1#S3.SS3 and pinned author code
https://github.com/wangqinsi1/Dobi-SVD/blob/220ab55026dccfa520b68130d8ce88736a24e055/modules/remapping.py
(blob 796b6dfb2513b2ae58064d13093909856244f73a): quantize the two short
factor blocks and retain the long-side tail. Author uses bnb dynamic codes,
half scales and CUDA. Here signed symmetric per-component int8 codes and
float32/64 scales replace that backend for portable, inspectable decoding.
Payload is 2*k*min(m,n) code bytes plus k*abs(m-n) tail elements, 2*k
scales and unchanged bias. It is not exactly k*max(m,n) 16-bit elements.

Baseline ordinary same-dtype factors stays the default. Casting both factors
to half alone does not implement the paper's block storage. Literal bnb was
rejected because it would introduce a CUDA-only dependency and opaque state.
Forward temporarily decodes both factors, without caching dense W; storage
savings imply neither a latency improvement nor a bound on allocator peaks.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .approximation import NumericalDomainViolation


_TAIL_DTYPES = ("float16", "bfloat16", "float32", "float64")
_CORE_DTYPES = ("float32", "float64")
_DTYPE_BYTES = {"float16":2, "bfloat16":2, "float32":4, "float64":8}


def remapped_tensor_bytes(in_features, out_features, rank, *, tail_dtype="float16",
                          scale_dtype="float32", bias_dtype="float32", bias=True):
    """Exact named tensor bytes, including scales and bias; no file overhead."""
    if (any(type(value) is not int or value<1 for value in (in_features,out_features))
            or type(rank) is not int or not 0<=rank<=min(in_features,out_features)
            or type(bias) is not bool or tail_dtype not in _TAIL_DTYPES
            or scale_dtype not in _CORE_DTYPES or bias_dtype not in _CORE_DTYPES):
        raise ValueError("Invalid remapping byte-cost dimensions or precision")
    short = min(in_features, out_features)
    return (2 * short * rank + abs(in_features-out_features) * rank * _DTYPE_BYTES[tail_dtype] +
            2 * rank * _DTYPE_BYTES[scale_dtype] +
            (out_features * _DTYPE_BYTES[bias_dtype] if bias else 0))


def _cast_checked(value, dtype, name):
    result = value.to(dtype=dtype)
    if not bool(torch.isfinite(result).all()):
        raise NumericalDomainViolation(f"Remapping {name} storage overflow or nonfinite values")
    return result


def _encode(block, axis, scale_dtype, compute_dtype):
    maximum = block.abs().amax(dim=axis)
    # Unit scale for exact-zero components keeps zero decoding well-defined.
    scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 127)
    scale = _cast_checked(scale, scale_dtype, "scale")
    if bool((scale <= 0).any()) or bool((scale.to(compute_dtype) <= 0).any()):
        raise NumericalDomainViolation("Remapping scale underflow")
    if not bool(torch.isfinite(scale.to(compute_dtype)).all()):
        raise NumericalDomainViolation("Remapping scale compute overflow")
    divisor = scale.double().unsqueeze(axis)
    codes = torch.round(block / divisor).clamp(-127, 127).to(torch.int8)
    return codes.contiguous(), scale.contiguous()


class RemappedLinear(nn.Module):
    """Frozen signed codes + scales + tail; two decoded Linear operations.

    Orientation: tall/square uses US and Vt, wide uses U and SVt. Codes
    left[m_short,k], right[k,n_short] have independent scales per component.
    Full rank is supported by this representation; zero rank is ZeroLinear.
    Bias retains source dtype. Inputs/outputs preserve their floating dtype.
    """
    __constants__ = ["in_features", "out_features", "rank", "tail_dtype",
                     "scale_dtype", "compute_dtype", "bias_dtype", "tall"]

    def __init__(self, in_features, out_features, rank, bias=True, *,
                 tail_dtype="float16", scale_dtype="float32", compute_dtype="float32",
                 bias_dtype="float32"):
        super().__init__()
        if (any(type(v) is not int or v < 1 for v in (in_features, out_features, rank))
                or rank > min(in_features, out_features) or type(bias) is not bool
                or tail_dtype not in _TAIL_DTYPES or scale_dtype not in _CORE_DTYPES
                or compute_dtype not in _CORE_DTYPES or bias_dtype not in _CORE_DTYPES):
            raise ValueError("Invalid RemappedLinear dimensions or explicit precision")
        self.in_features, self.out_features, self.rank = in_features, out_features, rank
        self.tail_dtype, self.scale_dtype = tail_dtype, scale_dtype
        self.compute_dtype, self.bias_dtype = compute_dtype, bias_dtype
        self.tall = out_features >= in_features
        short = min(in_features, out_features)
        self.register_buffer("left_codes", torch.zeros((short, rank), dtype=torch.int8))
        self.register_buffer("right_codes", torch.zeros((rank, short), dtype=torch.int8))
        self.register_buffer("left_scale", torch.ones(rank, dtype=getattr(torch, scale_dtype)))
        self.register_buffer("right_scale", torch.ones(rank, dtype=getattr(torch, scale_dtype)))
        tail_shape = ((out_features-short, rank) if self.tall else (rank, in_features-short))
        self.register_buffer("tail", torch.zeros(tail_shape, dtype=getattr(torch, tail_dtype)))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=getattr(torch, bias_dtype)),
                                 requires_grad=False) if bias else None

    @classmethod
    def from_weight(cls, weight, rank, bias=None, *, tail_dtype="float16",
                    scale_dtype="float32", compute_dtype="float32"):
        if (not isinstance(weight, torch.Tensor) or weight.ndim != 2
                or weight.device.type != "cpu" or weight.dtype not in (torch.float32, torch.float64)
                or not bool(torch.isfinite(weight).all())
                or (bias is not None and (not isinstance(bias, torch.Tensor)
                    or bias.shape != (weight.shape[0],) or bias.dtype != weight.dtype
                    or bias.device.type != "cpu" or not bool(torch.isfinite(bias).all())))):
            raise NumericalDomainViolation("Remapping requires finite CPU weight and source-dtype bias")
        result = cls(weight.shape[1], weight.shape[0], rank, bias is not None,
                     tail_dtype=tail_dtype, scale_dtype=scale_dtype, compute_dtype=compute_dtype,
                     bias_dtype=str(weight.dtype)[6:])
        with torch.no_grad():
            u, s, vh = torch.linalg.svd(weight.double(), full_matrices=False)
            if result.tall:
                left, right = u[:, :rank] * s[:rank], vh[:rank]
                tail = left[weight.shape[1]:]
                left = left[:weight.shape[1]]
            else:
                left, right = u[:, :rank], s[:rank, None] * vh[:rank]
                tail = right[:, weight.shape[0]:]
                right = right[:, :weight.shape[0]]
            core = getattr(torch, compute_dtype)
            result.left_codes, result.left_scale = _encode(left, 0, getattr(torch, scale_dtype), core)
            result.right_codes, result.right_scale = _encode(right, 1, getattr(torch, scale_dtype), core)
            result.tail = _cast_checked(tail, getattr(torch, tail_dtype), "tail").clone(memory_format=torch.contiguous_format)
            if bias is not None:
                result.bias.copy_(bias)
            result.validate_state(result.state_dict())
        return result

    def decoded_factors(self):
        if self.compute_dtype == "float64":
            left = self.left_codes.to(torch.float64) * self.left_scale.to(torch.float64).unsqueeze(0)
            right = self.right_codes.to(torch.float64) * self.right_scale.to(torch.float64).unsqueeze(1)
            tail = self.tail.to(torch.float64)
        else:
            left = self.left_codes.to(torch.float32) * self.left_scale.to(torch.float32).unsqueeze(0)
            right = self.right_codes.to(torch.float32) * self.right_scale.to(torch.float32).unsqueeze(1)
            tail = self.tail.to(torch.float32)
        if self.tall:
            left = torch.cat((left, tail), dim=0)
        else:
            right = torch.cat((right, tail), dim=1)
        return left, right

    def forward(self, inputs):
        if inputs.shape[-1] != self.in_features or not inputs.is_floating_point():
            raise ValueError("RemappedLinear requires compatible floating inputs")
        if not bool(torch.isfinite(inputs).all()):
            raise ValueError("RemappedLinear refuses nonfinite inputs")
        left, right = self.decoded_factors()
        x = inputs.to(dtype=left.dtype)
        bias = self.bias
        y = F.linear(F.linear(x, right), left, None if bias is None else bias.to(dtype=left.dtype))
        result = y.to(dtype=inputs.dtype)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("RemappedLinear output overflow or nonfinite values")
        return result

    @torch.jit.unused
    def representation_config(self):
        self.validate_state(self.state_dict())
        return {"layer_type": "RemappedLinear", "version": 1,
                "in_features": self.in_features, "out_features": self.out_features,
                "rank": self.rank, "bias": self.bias is not None,
                "remapping": "symmetric_int8_v1", "left_dtype": "int8", "right_dtype": "int8",
                "tail_dtype": self.tail_dtype, "scale_dtype": self.scale_dtype,
                "compute_dtype": self.compute_dtype, "bias_dtype": self.bias_dtype}

    @torch.jit.unused
    def validate_state(self, state, prefix=""):
        short = min(self.in_features,self.out_features)
        expected = {"left_codes":((short,self.rank),torch.int8),
                    "right_codes":((self.rank,short),torch.int8),
                    "left_scale":((self.rank,),getattr(torch,self.scale_dtype)),
                    "right_scale":((self.rank,),getattr(torch,self.scale_dtype)),
                    "tail":(((self.out_features-short,self.rank) if self.tall else
                              (self.rank,self.in_features-short)),getattr(torch,self.tail_dtype))}
        if self.bias is not None:
            expected["bias"] = ((self.out_features,),getattr(torch,self.bias_dtype))
        storage_names = {}
        for name, (shape,dtype) in expected.items():
            value = state[prefix+name]
            if (not isinstance(value, torch.Tensor) or value.shape != shape
                    or value.dtype != dtype
                    or (value.is_floating_point() and not bool(torch.isfinite(value).all()))):
                raise ValueError("RemappedLinear state conflicts with declared precision or shape")
            storage = value.untyped_storage()
            if (not value.is_contiguous() or value.storage_offset()!=0
                    or storage.nbytes()!=value.numel()*value.element_size()):
                raise ValueError("RemappedLinear components require minimal contiguous owned storage")
            identity = (str(value.device),storage._cdata)
            if storage.nbytes() and identity in storage_names:
                raise ValueError("RemappedLinear components must not share storage")
            if storage.nbytes():
                storage_names[identity] = name
            if name.endswith("codes") and bool((value == -128).any()):
                raise ValueError("RemappedLinear signed codes must be in [-127,127]")
            if name.endswith("scale") and bool((value <= 0).any()):
                raise ValueError("RemappedLinear scales must be positive")
        core = getattr(torch, self.compute_dtype)
        for name in ("left_scale", "right_scale", "tail", "bias"):
            if prefix+name not in state:
                continue
            value = state[prefix+name].to(core)
            if not bool(torch.isfinite(value).all()) or (name.endswith("scale") and bool((value <= 0).any())):
                raise ValueError("RemappedLinear state overflows or underflows the declared compute dtype")
        left = state[prefix+"left_codes"].to(core)*state[prefix+"left_scale"].to(core).unsqueeze(0)
        right = state[prefix+"right_codes"].to(core)*state[prefix+"right_scale"].to(core).unsqueeze(1)
        tail = state[prefix+"tail"].to(core)
        if self.tall:
            left = torch.cat((left,tail),dim=0)
        else:
            right = torch.cat((right,tail),dim=1)
        if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
            raise ValueError("RemappedLinear decoded factor overflow")

    def _apply(self, fn, recurse=True):
        # Device transfer is supported. Module-wide precision conversion would
        # erase the declared independent scale/tail/bias precision; refuse it
        # before touching any real tensor, including half()/double()/to(dtype).
        for value in (*self.parameters(),*self.buffers()):
            probe = fn(torch.empty(0,device=value.device,dtype=value.dtype))
            if probe.dtype != value.dtype:
                raise ValueError("RemappedLinear precision is fixed; rebuild with explicit component dtypes")
        return super()._apply(fn,recurse=recurse)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        try:
            self.validate_state(state_dict, prefix)
        except (ValueError, KeyError, TypeError) as error:
            error_msgs.append(str(error))
            return
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
