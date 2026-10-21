"""Portable compact hybrid Linear operators without a hidden dense weight.

The N:M branch is a CPU reference using packed gathers and dot products. It is
not an accelerated sparse kernel. Neither forward builds a full residual/W.
"""
from __future__ import annotations

from collections.abc import Mapping
import torch
from torch import nn
from torch.nn import functional as F

from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear
from .specs import NMStructure, LoSparseSpec, SparseDomainError, plan_linear


def _factors(in_features, out_features, rank, *, dtype, device):
    if rank == 0:
        return None
    # Construction must not consume the application's RNG stream.
    with torch.random.fork_rng(devices=[]):
        result = FactorizedLinear(in_features, out_features, rank, bias=False,
                                  device=device, dtype=dtype)
    with torch.no_grad():
        result.left.weight.zero_()
        result.right.weight.zero_()
    return result


def _dtype(dtype):
    dtype = torch.float32 if dtype is None else dtype
    if dtype not in (torch.float32, torch.float64):
        raise SparseDomainError("unsupported_device_dtype", "Only CPU float32/float64 is admitted")
    return dtype


def _tensor(value, shape, name, *, dtype, device, integer=False):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
            or value.layout != torch.strided or value.device != torch.device(device)
            or value.dtype != (torch.int64 if integer else dtype)
            or (not integer and not bool(torch.isfinite(value).all()))):
        raise SparseDomainError("invalid_state", f"{name} has an invalid shape, layout, device, dtype or value")


def _config(config, layer_type, extra):
    keys = {"layer_type", "version", "in_features", "out_features", "rank", "bias", *extra}
    if (not isinstance(config, Mapping) or set(config) != keys
            or config["layer_type"] != layer_type or type(config["version"]) is not int
            or config["version"] != 1 or type(config["bias"]) is not bool):
        raise SparseDomainError("invalid_configuration", f"Invalid {layer_type} constructor description")


def _factor_state(config, state, dtype, device):
    rank = config["rank"]
    shapes = {"bias": (config["out_features"],)} if config["bias"] else {}
    if rank:
        shapes.update({"factors.left.weight": (config["out_features"], rank),
                       "factors.right.weight": (rank, config["in_features"])})
    for key, shape in shapes.items():
        if key not in state:
            raise SparseDomainError("invalid_state", f"Missing {key}")
        _tensor(state[key], shape, key, dtype=dtype, device=device)
    return set(shapes)


def _state_precision(state):
    if not isinstance(state, Mapping) or not state:
        raise SparseDomainError("invalid_state", "A nonempty tensor state mapping is required")
    values = [v for v in state.values() if isinstance(v, torch.Tensor) and v.is_floating_point()]
    if not values:
        raise SparseDomainError("invalid_state", "State requires floating-point values")
    reference = values[0]
    _dtype(reference.dtype)
    if reference.device.type != "cpu":
        raise SparseDomainError("unsupported_device_dtype", "Portable sparse state requires CPU")
    return reference.dtype, reference.device


class PackedNMLinear(nn.Module):
    """S + BA with exactly n packed slots per input-axis group of m.

    Zero numerical values may occupy slots; slot count describes support.
    ``values`` and absolute-column ``indices`` are ordinary persisted buffers.
    """

    def __init__(self, in_features, out_features, n=2, m=4, rank=0, bias=True,
                 *, device="cpu", dtype=None):
        super().__init__()
        dtype = _dtype(dtype)
        structure = NMStructure(n, m, rank)
        plan_linear(in_features, out_features, structure, dtype=str(dtype).split(".")[-1],
                    device=str(torch.device(device)), bias=bias)
        self.in_features, self.out_features = in_features, out_features
        self.n, self.m, self.rank = n, m, rank
        self.factors = _factors(in_features, out_features, rank, dtype=dtype, device=device)
        shape = (out_features, in_features // m, n)
        self.register_buffer("values", torch.zeros(shape, dtype=dtype, device=device))
        indices = torch.arange(in_features // m, device=device)[:, None] * m + torch.arange(n, device=device)
        self.register_buffer("indices", indices.expand(out_features, -1, -1).clone().long())
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device)) if bias else None

    @classmethod
    def from_tensors(cls, values, indices, left, right, *, in_features, bias=None, m=4):
        if not isinstance(values, torch.Tensor) or values.ndim != 3:
            raise SparseDomainError("invalid_state", "values must have shape [out, groups, n]")
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor) or left.ndim != 2 or right.ndim != 2:
            raise SparseDomainError("invalid_state", "Explicit left/right matrices, including rank-zero matrices, required")
        out_features, _, n = values.shape
        config = {"layer_type": "PackedNMLinear", "version": 1, "in_features": in_features,
                  "out_features": out_features, "n": n, "m": m, "rank": right.shape[0], "bias": bias is not None}
        state = {"values": values, "indices": indices}
        if right.shape[0]:
            state.update({"factors.left.weight": left, "factors.right.weight": right})
        elif left.shape != (out_features, 0) or right.shape != (0, in_features):
            raise SparseDomainError("invalid_state", "Invalid rank-zero factor shapes")
        if bias is not None:
            state["bias"] = bias
        cls.validate_checkpoint_state(config, state)
        result = cls.from_config(config, dtype=values.dtype, device=values.device)
        result.load_state_dict(state)
        return result

    def representation_config(self):
        return {"layer_type": "PackedNMLinear", "version": 1, "in_features": self.in_features,
                "out_features": self.out_features, "n": self.n, "m": self.m,
                "rank": self.rank, "bias": self.bias is not None}

    @classmethod
    def from_config(cls, config, *, device="cpu", dtype=None):
        _config(config, "PackedNMLinear", ("n", "m"))
        return cls(config["in_features"], config["out_features"], config["n"], config["m"],
                   config["rank"], config["bias"], device=device, dtype=dtype)

    @classmethod
    def validate_checkpoint_state(cls, config, state):
        _config(config, "PackedNMLinear", ("n", "m"))
        dtype, device = _state_precision(state)
        structure = NMStructure(config["n"], config["m"], config["rank"])
        plan_linear(config["in_features"], config["out_features"], structure,
                    dtype=str(dtype).split(".")[-1], device=str(device), bias=config["bias"])
        shape = (config["out_features"], config["in_features"] // config["m"], config["n"])
        for key in ("values", "indices"):
            if key not in state:
                raise SparseDomainError("invalid_state", f"Missing {key}")
            _tensor(state[key], shape, key, dtype=dtype, device=device, integer=key == "indices")
        indices = state["indices"]
        groups = torch.arange(shape[1], device=device)[None, :, None]
        if (bool((indices < 0).any()) or bool((indices >= config["in_features"]).any())
                or not bool((indices // config["m"] == groups).all())
                or (shape[2] > 1 and bool((indices.sort(-1).values.diff(dim=-1) == 0).any()))):
            raise SparseDomainError("invalid_support", "Indices must be distinct columns inside their declared N:M group")
        expected = _factor_state(config, state, dtype, device) | {"values", "indices"}
        if set(state) != expected:
            raise SparseDomainError("invalid_state", "Unexpected tensors in compact N:M state")

    def validate_state(self, state=None, prefix=""):
        selected = self.state_dict() if state is None else {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        self.validate_checkpoint_state(self.representation_config(), selected)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        self.validate_checkpoint_state(self.representation_config(), state_dict)
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def forward(self, x):
        # Per-output gather bounds the temporary to batch * groups * n. A
        # broadcast over all outputs would create a large dense activation copy.
        sparse = torch.cat([F.linear(x.index_select(-1, self.indices[row].reshape(-1)),
                                     self.values[row].reshape(1, -1))
                            for row in range(self.out_features)], dim=-1)
        result = sparse if self.factors is None else sparse + self.factors(x)
        return result if self.bias is None else result + self.bias


class IndexedResidualLinear(nn.Module):
    """BA + scatter(S_J), retaining the full output width for every k >= 0."""

    def __init__(self, in_features, out_features, rank, residual_count, bias=True,
                 *, device="cpu", dtype=None):
        super().__init__()
        dtype = _dtype(dtype)
        plan_linear(in_features, out_features, LoSparseSpec(rank, residual_count, 1),
                    dtype=str(dtype).split(".")[-1], device=str(torch.device(device)), bias=bias)
        self.in_features, self.out_features = in_features, out_features
        self.rank, self.residual_count = rank, residual_count
        self.factors = _factors(in_features, out_features, rank, dtype=dtype, device=device)
        self.residual_rows = nn.Parameter(torch.zeros((residual_count, in_features), dtype=dtype, device=device))
        self.register_buffer("row_indices", torch.arange(residual_count, dtype=torch.int64, device=device))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device)) if bias else None

    @classmethod
    def from_tensors(cls, left, right, residual_rows, row_indices, *, bias=None):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor) or left.ndim != 2 or right.ndim != 2:
            raise SparseDomainError("invalid_state", "Explicit factor matrices required")
        if not isinstance(residual_rows, torch.Tensor) or residual_rows.ndim != 2:
            raise SparseDomainError("invalid_state", "Residual rows must be a matrix")
        config = {"layer_type": "IndexedResidualLinear", "version": 1, "in_features": right.shape[1],
                  "out_features": left.shape[0], "rank": right.shape[0],
                  "residual_count": residual_rows.shape[0], "bias": bias is not None}
        state = {"residual_rows": residual_rows, "row_indices": row_indices}
        if right.shape[0]:
            state.update({"factors.left.weight": left, "factors.right.weight": right})
        elif left.shape != (config["out_features"], 0):
            raise SparseDomainError("invalid_state", "Invalid rank-zero left factor shape")
        if bias is not None:
            state["bias"] = bias
        cls.validate_checkpoint_state(config, state)
        result = cls.from_config(config, device=right.device, dtype=right.dtype)
        result.load_state_dict(state)
        return result

    def representation_config(self):
        return {"layer_type": "IndexedResidualLinear", "version": 1, "in_features": self.in_features,
                "out_features": self.out_features, "rank": self.rank,
                "residual_count": self.residual_count, "bias": self.bias is not None}

    @classmethod
    def from_config(cls, config, *, device="cpu", dtype=None):
        _config(config, "IndexedResidualLinear", ("residual_count",))
        return cls(config["in_features"], config["out_features"], config["rank"],
                   config["residual_count"], config["bias"], device=device, dtype=dtype)

    @classmethod
    def validate_checkpoint_state(cls, config, state):
        _config(config, "IndexedResidualLinear", ("residual_count",))
        dtype, device = _state_precision(state)
        plan_linear(config["in_features"], config["out_features"],
                    LoSparseSpec(config["rank"], config["residual_count"], 1),
                    dtype=str(dtype).split(".")[-1], device=str(device), bias=config["bias"])
        for key, shape in (("residual_rows", (config["residual_count"], config["in_features"])),
                           ("row_indices", (config["residual_count"],))):
            if key not in state:
                raise SparseDomainError("invalid_state", f"Missing {key}")
            _tensor(state[key], shape, key, dtype=dtype, device=device, integer=key == "row_indices")
        indices = state["row_indices"]
        if (bool(((indices < 0) | (indices >= config["out_features"])).any())
                or indices.unique().numel() != indices.numel()):
            raise SparseDomainError("invalid_support", "Residual row indices must be distinct full-output coordinates")
        expected = _factor_state(config, state, dtype, device) | {"residual_rows", "row_indices"}
        if set(state) != expected:
            raise SparseDomainError("invalid_state", "Unexpected tensors in compact indexed residual state")

    def validate_state(self, state=None, prefix=""):
        selected = self.state_dict() if state is None else {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        self.validate_checkpoint_state(self.representation_config(), selected)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        self.validate_checkpoint_state(self.representation_config(), state_dict)
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def forward(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features)) if self.factors is None else self.factors(x)
        if self.residual_count:
            result = result.index_add(-1, self.row_indices, F.linear(x, self.residual_rows))
        return result if self.bias is None else result + self.bias
