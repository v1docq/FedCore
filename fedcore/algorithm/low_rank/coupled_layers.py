"""Portable CPU reference attention and MLP with real compact inner axes.

This deliberately small runtime supports interleaved RoPE, causal attention,
boolean allow masks and shared KV groups. Cached keys are already rotated;
cached values have the actual compressed width. Forward returns a Tensor so
the modules compose with the existing method interpreter.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


def _positive(*values):
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("Reference dimensions must be positive integers")


def _coordinates(values, original_width, count, *, rope):
    if (len(values) != count or any(type(v) is not int or not 0 <= v < original_width for v in values)
            or len(set(values)) != len(values)):
        raise ValueError("Q/K coordinates must be distinct and fit the original head width")
    if rope and (count % 2 or any(values[i] % 2 or values[i + 1] != values[i] + 1
                                 for i in range(0, count, 2))):
        raise ValueError("Interleaved RoPE requires complete ordered coordinate pairs")


def apply_reference_rope(states, positions, coordinates, original_width, base):
    """Rotate using ORIGINAL frequencies, even when a later pair was selected."""
    pairs = coordinates[:, ::2].to(device=states.device, dtype=states.dtype)
    frequencies = base ** (-pairs / original_width)
    angles = positions.to(states.dtype)[:, None, :, None] * frequencies[None, :, None, :]
    c, s = angles.cos(), angles.sin()
    even, odd = states[..., ::2], states[..., 1::2]
    return torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2)


@dataclass(frozen=True)
class KVCache:
    keys: torch.Tensor  # [batch, KV groups, tokens, actual Q/K width]
    values: torch.Tensor  # [batch, KV groups, tokens, actual Value width]

    @property
    def length(self):
        return self.keys.shape[2]

    @property
    def tensor_bytes(self):
        return self.keys.numel() * self.keys.element_size() + self.values.numel() * self.values.element_size()


@dataclass(frozen=True)
class AttentionOutput:
    output: torch.Tensor
    logits: torch.Tensor  # unmasked, scaled bilinear logits
    probabilities: torch.Tensor
    cache: KVCache | None
    values: torch.Tensor
    contexts: torch.Tensor


class ReferenceAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, head_dim, *, qk_dim=None,
                 value_dim=None, head_map=None, group_coordinates=None, rope=True,
                 rope_base=10000.0, logits_scale=None, causal=True, bias=True,
                 device=None, dtype=None):
        super().__init__()
        qk_dim = head_dim if qk_dim is None else qk_dim
        value_dim = head_dim if value_dim is None else value_dim
        _positive(d_model, num_heads, num_kv_heads, head_dim, qk_dim, value_dim)
        if num_heads % num_kv_heads or num_kv_heads > num_heads:
            raise ValueError("Reference GQA requires equally sized query groups")
        if qk_dim > head_dim or value_dim > head_dim:
            raise ValueError("Inner reference widths cannot exceed original head width")
        if type(rope) is not bool or type(causal) is not bool or type(bias) is not bool:
            raise ValueError("Reference flags must be boolean")
        if rope and head_dim % 2:
            raise ValueError("Interleaved RoPE requires an even original head width")
        if type(rope_base) not in (int, float) or not math.isfinite(rope_base) or rope_base <= 1:
            raise ValueError("RoPE base must be finite and greater than one")
        head_map = tuple(i // (num_heads // num_kv_heads) for i in range(num_heads)) if head_map is None else tuple(head_map)
        if (len(head_map) != num_heads or any(type(v) is not int or not 0 <= v < num_kv_heads for v in head_map)
                or any(head_map.count(group) != num_heads // num_kv_heads for group in range(num_kv_heads))):
            raise ValueError("Explicit head map must cover equally sized KV groups")
        coordinates = (tuple(range(qk_dim)),) * num_kv_heads if group_coordinates is None else tuple(tuple(v) for v in group_coordinates)
        if len(coordinates) != num_kv_heads:
            raise ValueError("One coordinate system is required per shared KV group")
        for group in coordinates:
            _coordinates(group, head_dim, qk_dim, rope=rope)
        scale = head_dim ** -.5 if logits_scale is None else logits_scale
        if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("Original logits scale must be finite and positive")
        self.d_model = d_model; self.num_heads = num_heads; self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim; self.qk_dim = qk_dim; self.value_dim = value_dim
        self.rope = rope; self.rope_base = float(rope_base); self.logits_scale = float(scale); self.causal = causal
        self.register_buffer("head_map", torch.tensor(head_map, dtype=torch.long, device=device))
        self.register_buffer("group_coordinates", torch.tensor(coordinates, dtype=torch.long, device=device))
        self.q_proj = nn.Linear(d_model, num_heads * qk_dim, bias=bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(d_model, num_kv_heads * qk_dim, bias=bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(d_model, num_kv_heads * value_dim, bias=bias, device=device, dtype=dtype)
        self.o_proj = nn.Linear(num_heads * value_dim, d_model, bias=bias, device=device, dtype=dtype)

    def project_query_key(self, x, *, positions=None, offset=0):
        b, t, _ = x.shape
        if positions is None:
            positions = torch.arange(offset, offset + t, device=x.device)[None].expand(b, -1)
        if (not isinstance(positions, torch.Tensor) or positions.shape != (b, t)
                or positions.dtype not in (torch.int32, torch.int64) or positions.device != x.device
                or bool((positions < 0).any())):
            raise ValueError("Position IDs require a nonnegative integer [batch,tokens] tensor")
        q = self.q_proj(x).reshape(b, t, self.num_heads, self.qk_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(b, t, self.num_kv_heads, self.qk_dim).transpose(1, 2)
        if self.rope:
            q = apply_reference_rope(q, positions, self.group_coordinates[self.head_map], self.head_dim, self.rope_base)
            k = apply_reference_rope(k, positions, self.group_coordinates, self.head_dim, self.rope_base)
        return q, k

    def attention(self, x, *, padding_mask=None, attention_mask=None, positions=None, cache=None, use_cache=False):
        if (not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[-1] != self.d_model
                or not x.shape[0] or not x.shape[1] or x.dtype != self.q_proj.weight.dtype
                or x.device != self.q_proj.weight.device or not bool(torch.isfinite(x).all())):
            raise ValueError("Reference attention requires finite compatible [batch,tokens,d_model] input")
        b, t, _ = x.shape
        if type(use_cache) is not bool:
            raise ValueError("use_cache must be boolean")
        past = 0
        if cache is not None:
            if (not isinstance(cache, KVCache) or cache.keys.ndim != 4 or cache.values.ndim != 4
                    or cache.keys.shape[:2] != (b, self.num_kv_heads)
                    or cache.values.shape[:2] != (b, self.num_kv_heads)
                    or cache.keys.shape[2] != cache.values.shape[2]
                    or cache.keys.shape[3] != self.qk_dim or cache.values.shape[3] != self.value_dim
                    or any(v.dtype != x.dtype or v.device != x.device or not bool(torch.isfinite(v).all()) for v in (cache.keys, cache.values))):
                raise ValueError("Cache requires the exact compact reference KV layout")
            past = cache.length
        total = past + t
        if padding_mask is not None and (not isinstance(padding_mask, torch.Tensor)
                or padding_mask.dtype != torch.bool or padding_mask.shape != (b, total) or padding_mask.device != x.device):
            raise ValueError("padding_mask must be boolean [batch,total_tokens], True means valid")
        if attention_mask is not None and (not isinstance(attention_mask, torch.Tensor)
                or attention_mask.dtype != torch.bool or attention_mask.shape != (t, total) or attention_mask.device != x.device):
            raise ValueError("attention_mask must be boolean [query_tokens,total_tokens], True means allowed")
        q, k = self.project_query_key(x, positions=positions, offset=past)
        v = self.v_proj(x).reshape(b, t, self.num_kv_heads, self.value_dim).transpose(1, 2)
        if cache is not None:
            k = torch.cat((cache.keys, k), dim=2); v = torch.cat((cache.values, v), dim=2)
        logits = (q @ k[:, self.head_map].transpose(-2, -1)) * self.logits_scale
        allowed = torch.ones((b, 1, t, total), dtype=torch.bool, device=x.device)
        if self.causal:
            allowed = allowed & (torch.arange(total, device=x.device)[None, :] <= torch.arange(past, total, device=x.device)[:, None])[None, None]
        if padding_mask is not None:
            allowed = allowed & padding_mask[:, None, None, :]
        if attention_mask is not None:
            allowed = allowed & attention_mask[None, None]
        # Fully masked rows have zero probabilities and output contribution.
        masked = logits.masked_fill(~allowed, -torch.inf)
        has_keys = allowed.any(-1, keepdim=True)
        probabilities = torch.softmax(torch.where(has_keys, masked, torch.zeros_like(masked)), dim=-1)
        probabilities = torch.where(allowed, probabilities, torch.zeros_like(probabilities))
        contexts = probabilities @ v[:, self.head_map]
        output = self.o_proj(contexts.transpose(1, 2).reshape(b, t, self.num_heads * self.value_dim))
        return AttentionOutput(output, logits, probabilities, KVCache(k, v) if use_cache else None, v, contexts)

    def forward(self, x, **kwargs):
        return self.attention(x, **kwargs).output

    def representation_config(self):
        return {"layer_type": "ReferenceAttention", "version": 1, "d_model": self.d_model,
                "num_heads": self.num_heads, "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim, "qk_dim": self.qk_dim, "value_dim": self.value_dim,
                "head_map": self.head_map.tolist(), "group_coordinates": self.group_coordinates.tolist(),
                "rope": self.rope, "rope_base": self.rope_base, "logits_scale": self.logits_scale,
                "causal": self.causal, "bias": self.q_proj.bias is not None}


class ReferenceMLP(nn.Module):
    def __init__(self, d_model, intermediate_size, *, gated=True, activation="silu", bias=True,
                 channel_indices=None, device=None, dtype=None):
        super().__init__()
        _positive(d_model, intermediate_size)
        if type(gated) is not bool or type(bias) is not bool or activation not in ("silu", "gelu", "relu"):
            raise ValueError("Declare a supported reference MLP activation and boolean flags")
        indices = tuple(range(intermediate_size)) if channel_indices is None else tuple(channel_indices)
        if (len(indices) != intermediate_size or any(type(v) is not int or v < 0 for v in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError("Channel metadata must list distinct original channels")
        self.d_model = d_model; self.intermediate_size = intermediate_size
        self.gated = gated; self.activation = activation; self.channel_indices = indices
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=bias, device=device, dtype=dtype)
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=bias, device=device, dtype=dtype) if gated else None
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=bias, device=device, dtype=dtype)

    def features(self, x):
        activation = {"silu": F.silu, "gelu": F.gelu, "relu": F.relu}[self.activation]
        return activation(self.gate_proj(x)) * self.up_proj(x) if self.gated else activation(self.up_proj(x))

    def forward(self, x):
        return self.down_proj(self.features(x))

    def representation_config(self):
        return {"layer_type": "ReferenceMLP", "version": 1, "d_model": self.d_model,
                "intermediate_size": self.intermediate_size, "gated": self.gated,
                "activation": self.activation, "bias": self.up_proj.bias is not None,
                "channel_indices": list(self.channel_indices)}
