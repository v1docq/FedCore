"""Bounded causal Transformer and independently specified FlexiGPT transfer.

Selection uses the tail of a difference of rank-r reconstructions; adapter
initialization uses the rank-r approximation of the FULL weight difference.
The runtime preserves one trainable dense base across execution positions.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from fedcore.algorithm.low_rank.coupled_layers import ReferenceAttention, ReferenceMLP
from fedcore.algorithm.low_rank.structured_layers import ResidualLinear
from fedcore.algorithm.low_rank.structured_profiles import FactorPair
from .channels import validate_tensor_state, _indices


MAX_GRAPH_ELEMENTS = 1_000_000
MAX_GRAPH_DEPTH = 32
PROJECTIONS = ("attention.q_proj", "attention.k_proj", "attention.v_proj", "attention.o_proj",
               "mlp.up_proj", "mlp.down_proj")


def _bound(d, h, depth, heads, gated, bias, rank=0):
    if (any(type(v) is not int or v < 1 for v in (d, h, depth, heads))
            or d % heads or depth > MAX_GRAPH_DEPTH or type(gated) is not bool or type(bias) is not bool):
        raise ValueError("Bounded Transformer requires positive compatible dimensions and boolean flags")
    # Conservative upper bound includes a dense source, all adapters and norms.
    per_layer = 4*d*d + (3 if gated else 2)*d*h + 10*d + (2 if gated else 1)*h
    per_layer += rank*(8*d + (3 if gated else 2)*(d+h)) + 2*d
    if depth * per_layer > MAX_GRAPH_ELEMENTS:
        raise ValueError("Transformer persistent-element limit exceeded before allocation")


def _device_dtype(device, dtype):
    if (device is not None and torch.device(device).type != "cpu") or dtype not in (None, torch.float32, torch.float64):
        raise ValueError("Bounded Transformer requires CPU float32/float64")


class GammaNorm(nn.Module):
    def __init__(self, width, gamma, eps, *, device=None, dtype=None):
        super().__init__()
        self.width, self.eps = width, float(eps)
        self.gamma = nn.Parameter(torch.full((width,), float(gamma), device=device, dtype=dtype))

    def forward(self, x):
        # Population variance, normalized across the hidden dimension per token.
        return F.layer_norm(x, (self.width,), weight=self.gamma, bias=None, eps=self.eps)


class SparseReferenceBlock(nn.Module):
    def __init__(self, d_model, intermediate_size, num_heads, *, gated, activation, bias,
                 norm_eps, device=None, dtype=None):
        super().__init__()
        self.attention = ReferenceAttention(d_model, num_heads, num_heads, d_model//num_heads,
                                             rope=False, causal=True, bias=bias, device=device, dtype=dtype)
        self.mlp = ReferenceMLP(d_model, intermediate_size, gated=gated, activation=activation,
                                bias=bias, device=device, dtype=dtype)
        self.attention_norm = nn.LayerNorm(d_model, eps=norm_eps, device=device, dtype=dtype)
        self.mlp_norm = nn.LayerNorm(d_model, eps=norm_eps, device=device, dtype=dtype)
        self.attention_output_norm = nn.Identity()
        self.mlp_output_norm = nn.Identity()

    def forward(self, x, *, padding_mask=None, cache=None, use_cache=False):
        attended = self.attention.attention(self.attention_norm(x), padding_mask=padding_mask,
                                            cache=cache, use_cache=use_cache)
        x = x + self.attention_output_norm(attended.output)
        x = x + self.mlp_output_norm(self.mlp(self.mlp_norm(x)))
        return (x, attended.cache) if use_cache else x


class SparseReferenceTransformer(nn.Module):
    """Small explicit graph over existing ReferenceAttention and ReferenceMLP.

    Input/output are continuous hidden states [batch,tokens,d_model]. No claim
    about an arbitrary HuggingFace model, tokenization or a language-model head.
    """
    def __init__(self, d_model, intermediate_size, num_layers, *, num_heads=1, gated=True,
                 activation="silu", bias=True, norm_eps=1e-5, device=None, dtype=None):
        super().__init__()
        _device_dtype(device, dtype)
        _bound(d_model, intermediate_size, num_layers, num_heads, gated, bias)
        if (activation not in ("silu", "gelu", "relu") or type(norm_eps) not in (int, float)
                or not math.isfinite(norm_eps) or norm_eps <= 0):
            raise ValueError("Declare a supported activation and finite positive normalization epsilon")
        precision = torch.finfo(torch.get_default_dtype() if dtype is None else dtype)
        if norm_eps < precision.tiny or norm_eps > precision.max:
            raise ValueError("Normalization epsilon is outside the bounded output-precision domain")
        self.d_model, self.intermediate_size, self.num_layers = d_model, intermediate_size, num_layers
        self.num_heads, self.gated, self.activation = num_heads, gated, activation
        self.bias, self.norm_eps = bias, float(norm_eps)
        self.layers = nn.ModuleList([SparseReferenceBlock(d_model, intermediate_size, num_heads,
            gated=gated, activation=activation, bias=bias, norm_eps=norm_eps, device=device, dtype=dtype)
            for _ in range(num_layers)])

    def _input(self, x):
        parameter = next(self.parameters())
        if (not isinstance(x, torch.Tensor) or x.ndim != 3 or not x.shape[0] or not x.shape[1]
                or x.shape[-1] != self.d_model or x.device != parameter.device or x.dtype != parameter.dtype
                or not bool(torch.isfinite(x).all())):
            raise ValueError("Transformer requires finite compatible nonempty [batch,tokens,d_model] input")

    def forward(self, x, *, padding_mask=None, caches=None, use_cache=False):
        self._input(x)
        if type(use_cache) is not bool or (caches is not None and (not isinstance(caches, (tuple, list)) or len(caches) != len(self.layers))):
            raise ValueError("KV caches require one separate state per execution position")
        outputs = []
        for i, layer in enumerate(self.layers):
            result = layer(x, padding_mask=padding_mask, cache=None if caches is None else caches[i], use_cache=use_cache)
            if use_cache:
                x, cache = result
                outputs.append(cache)
            else:
                x = result
        return (x, tuple(outputs)) if use_cache else x

    def representation_config(self):
        return {"layer_type": "SparseReferenceTransformer", "version": 1,
                **{name: getattr(self, name) for name in ("d_model", "intermediate_size", "num_layers", "num_heads",
                                                        "gated", "activation", "bias", "norm_eps")}}

    @classmethod
    def from_config(cls, config, *, device=None, dtype=None):
        required = {"layer_type", "version", "d_model", "intermediate_size", "num_layers", "num_heads",
                    "gated", "activation", "bias", "norm_eps"}
        if (not isinstance(config, Mapping) or set(config) != required or config["layer_type"] != "SparseReferenceTransformer"
                or type(config["version"]) is not int or config["version"] != 1):
            raise ValueError("Exact version-1 SparseReferenceTransformer config required")
        with torch.random.fork_rng(devices=[]):
            return cls(**{k: v for k, v in config.items() if k not in ("layer_type", "version")}, device=device, dtype=dtype)

    def validate_state(self, state=None, prefix=""):
        validate_tensor_state(self, state, prefix)

    validate_checkpoint_state = validate_state


@dataclass(frozen=True)
class FlexiGPTSpec:
    retained_positions: tuple[int, ...] | None = (0,)
    rank: int = 1
    alpha: float = 1.
    gamma_init: float = .01
    norm_eps: float = 1e-5
    selection_aggregation: str = "sum_squared_tail"
    extension_sources: tuple[int, ...] = ()
    max_workspace_bytes: int = 64 * 1024**2
    retained_count: int | None = None

    def __post_init__(self):
        explicit = (isinstance(self.retained_positions, tuple) and bool(self.retained_positions)
                    and self.retained_count is None)
        automatic = (self.retained_positions is None and type(self.retained_count) is int and self.retained_count > 0)
        if (not (explicit or automatic)
                or not isinstance(self.extension_sources, tuple)
                or any(type(i) is not int or i < 0 for i in (*(self.retained_positions or ()), *self.extension_sources))
                or (explicit and len(set(self.retained_positions)) != len(self.retained_positions))
                or type(self.rank) is not int or self.rank < 1
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in (self.alpha, self.gamma_init, self.norm_eps))
                or self.alpha <= 0 or self.norm_eps <= 0 or self.gamma_init < 0
                or self.selection_aggregation != "sum_squared_tail"
                or type(self.max_workspace_bytes) is not int or self.max_workspace_bytes < 1):
            raise ValueError("Invalid explicit FlexiGPT reference profile")


def truncated_weight(weight, rank):
    u, s, vh = torch.linalg.svd(weight.double(), full_matrices=False)
    return ((u[:, :rank] * s[:rank]) @ vh[:rank]).to(weight)


def flexigpt_distance(weight, base, rank):
    if (weight.ndim != 2 or weight.shape != base.shape or type(rank) is not int
            or not 1 <= rank <= min(weight.shape)):
        raise ValueError("FlexiGPT matrix distance requires compatible shapes and admissible rank")
    delta = truncated_weight(weight, rank).double() - truncated_weight(base, rank).double()
    singular = torch.linalg.svdvals(delta)
    return float(singular[rank:].square().sum())


class FlexiGPTLinear(nn.Module):
    """Method-local scaled ResidualLinear; the base remains the same object."""
    def __init__(self, base, rank, alpha, *, factors=None, bias_delta=None):
        super().__init__()
        self.in_features, self.out_features = base.in_features, base.out_features
        self.rank, self.alpha, self.scale = rank, float(alpha), float(alpha)/rank
        if factors is None:
            # A nonzero right factor preserves a usable first gradient at zero delta.
            left = base.weight.new_zeros((base.out_features, rank))
            right = base.weight.new_zeros((rank, base.in_features))
            right[:, :rank] = torch.eye(rank, dtype=right.dtype, device=right.device)
            factors = FactorPair(left, right)
        self.inner = ResidualLinear(base, factors, freeze_base=False)
        self.bias_delta = None if base.bias is None else nn.Parameter(
            base.bias.new_zeros(base.out_features) if bias_delta is None else bias_delta.detach().clone())

    @property
    def weight(self):
        # ReferenceAttention reads only dtype/device metadata from this property.
        return self.inner.base.weight

    @property
    def bias(self):
        return self.inner.base.bias

    def forward(self, x):
        output = self.inner.base(x) + self.scale * self.inner.residual(x)
        return output if self.bias_delta is None else output + self.bias_delta


class FlexiGPTPosition(SparseReferenceBlock):
    pass


class FlexiGPTTransformer(SparseReferenceTransformer):
    def __init__(self, d_model, intermediate_size, source_depth, base_sources, position_sources, *,
                 retained_positions=(0,), rank=1, alpha=1., gamma_init=.01, num_heads=1,
                 gated=True, activation="silu", bias=True, norm_eps=1e-5, device=None, dtype=None):
        _device_dtype(device, dtype)
        _bound(d_model, intermediate_size, len(base_sources), num_heads, gated, bias, rank)
        if type(source_depth) is not int or source_depth < 1 or source_depth > len(base_sources):
            raise ValueError("FlexiGPT source depth must fit execution depth")
        retained = _indices(retained_positions, source_depth, "retained positions")
        if not retained or rank > min(d_model, intermediate_size):
            raise ValueError("FlexiGPT requires a retained dense base and shape-compatible adapter rank")
        FlexiGPTSpec(retained, rank, alpha, gamma_init, norm_eps)
        precision = torch.finfo(torch.get_default_dtype() if dtype is None else dtype)
        if not precision.tiny <= alpha/rank <= precision.max or gamma_init > precision.max:
            raise ValueError("FlexiGPT alpha/r or gamma is outside the bounded output-precision domain")
        if (not isinstance(base_sources, (tuple, list)) or not isinstance(position_sources, (tuple, list))
                or len(base_sources) != len(position_sources) or len(base_sources) < source_depth
                or any(type(i) is not int or i not in retained for i in base_sources)
                or any(type(i) is not int or not 0 <= i < source_depth for i in position_sources)
                or tuple(position_sources[:source_depth]) != tuple(range(source_depth))
                or any(base_sources[i] != i for i in retained)):
            raise ValueError("Invalid FlexiGPT base/position provenance maps")
        super().__init__(d_model, intermediate_size, len(base_sources), num_heads=num_heads, gated=gated,
                         activation=activation, bias=bias, norm_eps=norm_eps, device=device, dtype=dtype)
        self.source_depth, self.base_sources, self.position_sources = source_depth, tuple(base_sources), tuple(position_sources)
        self.retained_positions = retained
        self.rank, self.alpha, self.gamma_init = rank, float(alpha), float(gamma_init)
        names = self._projections()
        bases = {i: {name: self.layers[i].get_submodule(name) for name in names} for i in retained}
        for i, layer in enumerate(self.layers):
            if i in retained:
                continue
            for name in names:
                parent_name, child_name = name.rsplit(".", 1)
                parent = layer.get_submodule(parent_name)
                setattr(parent, child_name, FlexiGPTLinear(bases[self.base_sources[i]][name], rank, alpha))
            layer.attention_output_norm = GammaNorm(d_model, gamma_init, norm_eps, device=device, dtype=dtype)
            layer.mlp_output_norm = GammaNorm(d_model, gamma_init, norm_eps, device=device, dtype=dtype)
        self.register_buffer("_base_map", torch.tensor(base_sources, dtype=torch.int64, device=device))
        self.register_buffer("_position_map", torch.tensor(position_sources, dtype=torch.int64, device=device))

    def _projections(self):
        return PROJECTIONS + (("mlp.gate_proj",) if self.gated else ())

    def representation_config(self):
        return {"layer_type": "FlexiGPTTransformer", "version": 1,
                **{name: getattr(self, name) for name in ("d_model", "intermediate_size", "source_depth", "rank", "alpha",
                    "gamma_init", "num_heads", "gated", "activation", "bias", "norm_eps")},
                "base_sources": list(self.base_sources), "position_sources": list(self.position_sources),
                "retained_positions": list(self.retained_positions)}

    @classmethod
    def from_config(cls, config, *, device=None, dtype=None):
        required = {"layer_type", "version", "d_model", "intermediate_size", "source_depth", "rank", "alpha", "gamma_init",
                    "num_heads", "gated", "activation", "bias", "norm_eps", "base_sources", "position_sources", "retained_positions"}
        if (not isinstance(config, Mapping) or set(config) != required or config["layer_type"] != "FlexiGPTTransformer"
                or type(config["version"]) is not int or config["version"] != 1):
            raise ValueError("Exact version-1 FlexiGPTTransformer constructor config required")
        kwargs = {k: v for k, v in config.items() if k not in ("layer_type", "version")}
        for name in ("base_sources", "position_sources", "retained_positions"):
            if not isinstance(kwargs[name], (tuple, list)):
                raise ValueError("FlexiGPT provenance maps require sequences")
            kwargs[name] = tuple(kwargs[name])
        with torch.random.fork_rng(devices=[]):
            return cls(**kwargs, device=device, dtype=dtype)

    def validate_state(self, state=None, prefix=""):
        validate_tensor_state(self, state, prefix)
        if self._base_map.tolist() != list(self.base_sources) or self._position_map.tolist() != list(self.position_sources):
            raise ValueError("FlexiGPT immutable position/base maps changed")
        for i, layer in enumerate(self.layers):
            for name in self._projections():
                projection = layer.get_submodule(name)
                if i not in self.retained_positions:
                    expected = self.layers[self.base_sources[i]].get_submodule(name)
                    if type(projection) is not FlexiGPTLinear or projection.inner.base is not expected:
                        raise ValueError("FlexiGPT shared dense base alias topology changed")
                    if projection.inner.freeze_base:
                        raise ValueError("FlexiGPT source profile cannot declare a permanently frozen base")

    validate_checkpoint_state = validate_state


def block_influence(inputs, outputs):
    if inputs.shape != outputs.shape:
        raise ValueError("Block Influence compares corresponding residual stream states")
    left, right = inputs.double(), outputs.double()
    ln, rn = left.norm(dim=-1), right.norm(dim=-1)
    product = ln * rn
    cosine = (left*right).sum(-1) / product.clamp_min(torch.finfo(torch.float64).tiny)
    cosine = torch.where((ln == 0) & (rn == 0), torch.ones_like(cosine), cosine)
    cosine = torch.where((ln == 0) ^ (rn == 0), torch.zeros_like(cosine), cosine)
    score = float((1 - cosine.clamp(-1, 1)).mean())
    if not math.isfinite(score):
        raise ValueError("Block Influence exceeds the finite numerical domain")
    return score


def flexigpt(graph, inputs, spec=FlexiGPTSpec()):
    if type(graph) is not SparseReferenceTransformer or type(spec) is not FlexiGPTSpec:
        raise ValueError("FlexiGPT requires its explicit dense reference graph and FlexiGPTSpec")
    graph._input(inputs)
    if any(p.device.type != "cpu" or p.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(p).all()) for p in graph.parameters()):
        raise ValueError("FlexiGPT requires finite CPU float32/float64 weights")
    if any("forward" in layer.__dict__ or layer._forward_hooks or layer._forward_pre_hooks for layer in graph.modules()):
        raise ValueError("FlexiGPT source must be an unmodified explicit graph without hooks")
    if spec.norm_eps != graph.norm_eps:
        raise ValueError("Bounded FlexiGPT profile requires the declared source normalization epsilon")
    retained = None if spec.retained_positions is None else _indices(spec.retained_positions, graph.num_layers, "retained positions")
    if spec.retained_count is not None and spec.retained_count > graph.num_layers:
        raise ValueError("FlexiGPT retained count exceeds source depth")
    extensions = _indices(spec.extension_sources, graph.num_layers, "extension sources") if len(set(spec.extension_sources)) == len(spec.extension_sources) else tuple(spec.extension_sources)
    if any(type(i) is not int or not 0 <= i < graph.num_layers for i in extensions):
        raise ValueError("FlexiGPT extension source must identify an existing source position")
    if spec.rank > min(graph.d_model, graph.intermediate_size):
        raise ValueError("FlexiGPT rank exceeds a projection shape")
    precision = torch.finfo(inputs.dtype)
    if not precision.tiny <= spec.alpha/spec.rank <= precision.max or spec.gamma_init > precision.max:
        raise ValueError("FlexiGPT alpha/r or gamma is outside the bounded output-precision domain")
    depth = graph.num_layers + len(extensions)
    _bound(graph.d_model, graph.intermediate_size, depth, graph.num_heads, graph.gated, graph.bias, spec.rank)
    reserve = 8*(24*max(graph.d_model, graph.intermediate_size)**2 + depth*inputs.numel()*8
                 + 8*inputs.shape[0]*graph.num_heads*inputs.shape[1]**2)
    if reserve > spec.max_workspace_bytes:
        raise ValueError("FlexiGPT workspace limit exceeded before graph copying")
    names = PROJECTIONS + (("mlp.gate_proj",) if graph.gated else ())
    if any(type(layer) is not SparseReferenceBlock for layer in graph.layers):
        raise ValueError("FlexiGPT requires the original explicit reference block classes")
    if any(type(layer.get_submodule(name)) is not nn.Linear for layer in graph.layers for name in names):
        raise ValueError("FlexiGPT source projections must be unmodified dense Linear")
    parameters = tuple(graph.named_parameters(remove_duplicate=False))
    if len({id(parameter) for _, parameter in parameters}) != len(parameters):
        raise ValueError("Dense FlexiGPT source cannot contain undeclared parameter aliases")
    position_sources = tuple(range(graph.num_layers)) + tuple(extensions)
    influence, distances, mapping = [], {}, []
    with torch.no_grad():
        x = inputs.detach()
        for layer in graph.layers:
            y = layer(x)
            influence.append(block_influence(x, y))
            x = y
        if retained is None:
            ranked = sorted(range(graph.num_layers), key=lambda i: (-influence[i], i))
            retained = tuple(sorted(ranked[:spec.retained_count]))
        for position, source in enumerate(position_sources):
            if position in retained:
                mapping.append(position)
                continue
            candidates = {}
            for base in retained:
                terms = [flexigpt_distance(graph.layers[source].get_submodule(name).weight,
                                            graph.layers[base].get_submodule(name).weight, spec.rank) for name in names]
                candidates[base] = math.fsum(terms)
            chosen = min(candidates, key=lambda i: (candidates[i], i))
            mapping.append(chosen)
            distances[position] = candidates
        with torch.random.fork_rng(devices=[]):
            compact = FlexiGPTTransformer(graph.d_model, graph.intermediate_size, graph.num_layers,
                tuple(mapping), position_sources, retained_positions=retained, rank=spec.rank, alpha=spec.alpha,
                gamma_init=spec.gamma_init, num_heads=graph.num_heads, gated=graph.gated, activation=graph.activation,
                bias=graph.bias, norm_eps=spec.norm_eps, device=inputs.device, dtype=inputs.dtype)
        for i in retained:
            compact.layers[i].load_state_dict(graph.layers[i].state_dict())
        for i, source in enumerate(position_sources):
            target, original = compact.layers[i], graph.layers[source]
            if i in retained:
                continue
            target.attention_norm.load_state_dict(original.attention_norm.state_dict())
            target.mlp_norm.load_state_dict(original.mlp_norm.state_dict())
            for name in names:
                projection = target.get_submodule(name)
                source_linear = original.get_submodule(name)
                base_linear = projection.inner.base
                difference = source_linear.weight.double() - base_linear.weight.double()
                if not bool(torch.isfinite(difference).all()):
                    raise ValueError("FlexiGPT full weight difference exceeds the finite numerical domain")
                u, s, vh = torch.linalg.svd(difference, full_matrices=False)
                root = s[:spec.rank].sqrt()
                left = (u[:, :spec.rank]*root).to(inputs.dtype)/projection.scale
                right = (root[:, None]*vh[:spec.rank]).to(inputs.dtype)
                if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
                    raise ValueError("FlexiGPT scaled factors exceed the finite output-precision domain")
                # Exact zero differences retain one nonzero factor for training.
                if not bool(torch.count_nonzero(left)) and not bool(torch.count_nonzero(right)):
                    right[:, :spec.rank] = torch.eye(spec.rank, dtype=right.dtype)
                projection.inner.residual.left.weight.copy_(left)
                projection.inner.residual.right.weight.copy_(right)
                if projection.bias_delta is not None:
                    projection.bias_delta.copy_(source_linear.bias - base_linear.bias)
        compact.train(graph.training)
        compact.validate_state()
    unique = sum(p.numel() for p in compact.parameters())
    return compact, {"method": "flexigpt", "source_profile": "2501.14713v2-independent-bounded-v1",
                      "block_influence": influence, "zero_norm_rule": "both_zero_cosine_one_else_zero",
                      "retained_selection": "highest_block_influence_stable_position_ties" if spec.retained_count is not None else "explicit_positions",
                      "retained_positions": retained, "base_sources": tuple(mapping), "position_sources": position_sources,
                      "selection_aggregation": spec.selection_aggregation, "selection_squared_distances": distances,
                      "adapter_initialization": "rank_r_of_full_difference", "alpha_over_rank": spec.alpha/spec.rank,
                      "gamma_init": spec.gamma_init, "shared_base_trainable": True,
                      "unique_parameters": unique, "execution_depth": depth,
                      "cache_slots": depth, "activation_positions": depth, "workspace_reserved_bytes": reserve,
                      "recovery_required": True, "accelerated_kernel": False}
