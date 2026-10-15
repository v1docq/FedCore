"""Checkpoint-ready dedicated causal RMS Transformer for SliceGPT.

This is an exact architectural adapter, not a general Hugging Face adapter.
Each residual subblock can change coordinates and physical width. Attention
head/KV dimensions remain independent of the residual stream width.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

from .coupled_layers import KVCache, ReferenceAttention, ReferenceMLP


MAX_PERSISTENT_ELEMENTS = 1_000_000
MAX_RESIDUAL_BLOCKS = 128
MAX_REFERENCE_WIDTH = 1024
MAX_HEADS = 128
MAX_INTERMEDIATE_WIDTH = 4096


def _positive(*values):
    if any(type(v) is not int or v < 1 for v in values):
        raise ValueError("SliceGPT dimensions must be positive integers")


def _bound_dimensions(reference_width, num_heads=1, num_kv_heads=1, head_dim=1,
                      intermediate_size=1):
    _positive(reference_width, num_heads, num_kv_heads, head_dim, intermediate_size)
    if (reference_width > MAX_REFERENCE_WIDTH or max(num_heads, num_kv_heads) > MAX_HEADS
            or head_dim > MAX_REFERENCE_WIDTH or intermediate_size > MAX_INTERMEDIATE_WIDTH
            or num_kv_heads > num_heads or num_heads % num_kv_heads):
        raise ValueError("SliceGPT dimensions exceed the bounded reference architecture")


def _block_elements(kind, ki, ko, num_heads, num_kv_heads, head_dim,
                    intermediate_size, bias, fused_norms):
    norm = (0 if fused_norms else ki) + 1
    residual = ki * ko
    if kind == "attention":
        inner = (num_heads + 2 * num_kv_heads) * head_dim
        return (norm + residual + ki * inner + num_heads * head_dim * ko
                + (inner + ko if bias else 0) + num_heads + num_kv_heads * head_dim)
    return (norm + residual + 2 * intermediate_size * ki + intermediate_size * ko
            + (2 * intermediate_size + ko if bias else 0))


def _bound_elements(elements):
    if elements > MAX_PERSISTENT_ELEMENTS:
        raise ValueError("SliceGPT persistent tensors exceed the bounded reference resource limit")


def _dtype_name(dtype):
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("SliceGPT supports float32 and float64 only")
    return str(dtype).split(".")[-1]


def _config(config, layer_type, keys):
    if (not isinstance(config, dict) or set(config) != set(keys) | {"layer_type", "version"}
            or config.get("layer_type") != layer_type or type(config.get("version")) is not int
            or config["version"] != 1):
        raise ValueError(f"Invalid {layer_type} representation configuration")
    return {key: config[key] for key in keys}


def _dtype(value):
    if value not in ("float32", "float64"):
        raise ValueError("SliceGPT configuration dtype must be float32 or float64")
    return getattr(torch, value)


def _fixed_tensor(value, shape, *, dtype, device):
    # Validate the supplied description before any tensor allocation. In
    # particular, torch.tensor(huge_nested_list) must not be the shape validator.
    if isinstance(value, torch.Tensor):
        if tuple(value.shape) != tuple(shape) or value.layout != torch.strided:
            raise ValueError("SliceGPT fixed matrix has an invalid declared shape/layout")
    elif (not isinstance(value, (tuple, list)) or len(value) != shape[0]
            or any(not isinstance(row, (tuple, list)) or len(row) != shape[1] for row in value)
            or any(type(v) not in (int, float) or not math.isfinite(v) for row in value for v in row)):
        raise ValueError("SliceGPT fixed matrix description must have its exact finite numeric shape")
    try:
        result = torch.tensor(value, device=device, dtype=dtype)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("Invalid SliceGPT fixed matrix") from exc
    if tuple(result.shape) != tuple(shape) or not bool(torch.isfinite(result).all()):
        raise ValueError("SliceGPT fixed matrix must be finite with the declared shape")
    return result


def _validate_fixed(state, prefix, name, expected):
    actual = state.get(prefix + name)
    if (not isinstance(actual, torch.Tensor) or actual.shape != expected.shape
            or actual.dtype != expected.dtype or not torch.equal(actual.cpu(), expected.detach().cpu())):
        raise ValueError(f"SliceGPT checkpoint fixed-buffer mismatch: {prefix + name}")


def _validate_tensors(module, state, prefix):
    expected = module.state_dict()
    keys = {key[len(prefix):] for key in state if key.startswith(prefix)}
    if keys != set(expected):
        raise ValueError("SliceGPT checkpoint tensor paths do not match its full graph")
    for name, value in expected.items():
        actual = state[prefix + name]
        if (not isinstance(actual, torch.Tensor) or actual.shape != value.shape
                or actual.dtype != value.dtype
                or (actual.is_floating_point() and not bool(torch.isfinite(actual).all()))):
            raise ValueError(f"Invalid SliceGPT checkpoint tensor: {prefix + name}")


def _raw_module_paths(module):
    """Inspect registered children without executing custom traversal methods."""
    result = {}; seen = set(); pending = [("", module)]
    while pending:
        name, child = pending.pop()
        if not isinstance(child, nn.Module):
            raise ValueError("SliceGPT rejects non-module registered children")
        if id(child) in seen:
            raise ValueError("SliceGPT rejects shared or cyclic child modules")
        seen.add(id(child)); result[name] = child
        if any(callable(value) for value in vars(child).values()):
            raise ValueError("SliceGPT rejects instance callable overrides")
        if any(getattr(child, key, {}) for key in (
                "_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks",
                "_state_dict_hooks", "_state_dict_pre_hooks", "_load_state_dict_pre_hooks",
                "_load_state_dict_post_hooks")):
            raise ValueError("SliceGPT rejects hooks affecting its declared graph")
        for key, value in child._modules.items():
            if value is not None:
                pending.append((name + "." + key if name else key, value))
    return result


def _check_reconstruction(module, config):
    """Every executable child setting must be represented by the root config."""
    with torch.random.fork_rng(devices=[]):
        expected = type(module).from_config(config)
    actual_modules = _raw_module_paths(module)
    expected_modules = dict(nn.Module.named_modules(expected))
    if set(actual_modules) != set(expected_modules):
        raise ValueError("SliceGPT representation has an undeclared consumer graph")
    for name, target in expected_modules.items():
        actual = actual_modules[name]
        if type(actual) is not type(target):
            raise ValueError("SliceGPT representation has an unsupported child module")
        for key, value in vars(target).items():
            if not key.startswith("_") and key != "training":
                if key not in vars(actual) or vars(actual)[key] != value:
                    raise ValueError(f"SliceGPT representation omits changed child setting {name}.{key}")
        if set(actual._parameters) != set(target._parameters) or set(actual._buffers) != set(target._buffers):
            raise ValueError("SliceGPT representation has undeclared parameters or buffers")
    _validate_tensors(module, module.state_dict(), "")
    actual_state, expected_state = module.state_dict(), expected.state_dict()
    if set(actual_state) != set(expected_state):
        raise ValueError("SliceGPT representation tensor paths do not match its configuration")
    for name, value in expected_state.items():
        actual = actual_state[name]
        if actual.shape != value.shape or actual.dtype != value.dtype:
            raise ValueError("SliceGPT representation tensor shape/dtype mismatch")
    for name, value in expected.named_buffers():
        if not torch.equal(actual_state[name].cpu(), value):
            raise ValueError("SliceGPT representation fixed metadata disagrees with its configuration")
    if type(module) is SliceGPTModel:
        if module.tie_word_embeddings != (module.embedding.weight is module.lm_head.weight):
            raise ValueError("SliceGPT representation embedding/head sharing mismatch")
    # All other parameter aliases would introduce sharing absent from the config.
    def groups(value, *, buffers=False):
        result = {}
        iterator = value.named_buffers(remove_duplicate=False) if buffers else value.named_parameters(remove_duplicate=False)
        for name, parameter in iterator:
            key = (str(parameter.device), parameter.untyped_storage().data_ptr(), parameter.untyped_storage().nbytes())
            result.setdefault(key, []).append(name)
        return sorted(tuple(names) for names in result.values())
    if groups(module) != groups(expected):
        raise ValueError("SliceGPT representation contains undeclared parameter sharing")
    if groups(module, buffers=True) != groups(expected, buffers=True):
        raise ValueError("SliceGPT representation contains undeclared buffer sharing")


class SliceGPTRMSNorm(nn.Module):
    """RMS normalization with explicit original-width denominator after slicing.

    A reduced width k still divides squared norm by original width d. This is
    equivalent to RMSNorm_k with multiplier sqrt(d/k) and epsilon d/k, rather
    than silently changing the original network's normalization convention.
    """
    def __init__(self, width, *, reference_width=None, eps=1e-6, learnable_scale=True,
                 device=None, dtype=None):
        super().__init__()
        reference_width = width if reference_width is None else reference_width
        _positive(width, reference_width)
        _bound_dimensions(reference_width)
        if width > reference_width or type(learnable_scale) is not bool:
            raise ValueError("SliceGPT RMSNorm width must fit its reference width")
        if type(eps) not in (int, float) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("SliceGPT RMSNorm epsilon must be finite and positive")
        dtype = torch.get_default_dtype() if dtype is None else dtype
        _dtype_name(dtype)
        self.width = width; self.reference_width = reference_width; self.eps = float(eps)
        self.register_buffer("width_denominator", torch.tensor(float(reference_width), device=device, dtype=dtype))
        self.weight = nn.Parameter(torch.ones(width, device=device, dtype=dtype)) if learnable_scale else None

    def forward(self, x):
        if x.shape[-1] != self.width:
            raise ValueError("SliceGPT RMSNorm input width mismatch")
        result = x * torch.rsqrt(x.square().sum(-1, keepdim=True) / self.width_denominator + self.eps)
        return result if self.weight is None else result * self.weight

    def representation_config(self):
        config = {"layer_type": "SliceGPTRMSNorm", "version": 1, "width": self.width,
                "reference_width": self.reference_width, "eps": self.eps,
                "learnable_scale": self.weight is not None, "dtype": _dtype_name(self.width_denominator.dtype)}
        _check_reconstruction(self, config)
        return config

    @classmethod
    def from_config(cls, config, *, device=None, dtype=None):
        values = _config(config, cls.__name__, ("width", "reference_width", "eps", "learnable_scale", "dtype"))
        declared_dtype = _dtype(values["dtype"])
        if dtype is not None and dtype != declared_dtype:
            raise ValueError("SliceGPT checkpoint dtype conflicts with its configuration")
        values["dtype"] = declared_dtype; values["device"] = device
        return cls(**values)

    def validate_checkpoint_state(self, state, prefix=""):
        _validate_tensors(self, state, prefix)
        _validate_fixed(state, prefix, "width_denominator", self.width_denominator)


class SliceGPTBlock(nn.Module):
    """One pre-norm attention or gated MLP residual edge with row-vector R."""
    def __init__(self, kind, input_width, output_width, reference_width, *,
                 num_heads, num_kv_heads, head_dim, intermediate_size, bias=True,
                 rope=True, eps=1e-6, fused_norms=False, transition=None,
                 device=None, dtype=None):
        super().__init__()
        _positive(input_width, output_width, reference_width)
        _bound_dimensions(reference_width, num_heads, num_kv_heads, head_dim, intermediate_size)
        if max(input_width, output_width) > reference_width or kind not in ("attention", "mlp"):
            raise ValueError("Unsupported SliceGPT residual block")
        if any(type(v) is not bool for v in (bias, rope, fused_norms)):
            raise ValueError("SliceGPT flags must be boolean")
        if rope and head_dim % 2:
            raise ValueError("SliceGPT RoPE requires an even head dimension")
        _bound_elements(_block_elements(kind, input_width, output_width, num_heads,
            num_kv_heads, head_dim, intermediate_size, bias, fused_norms))
        dtype = torch.get_default_dtype() if dtype is None else dtype
        _dtype_name(dtype)
        self.kind = kind; self.input_width = input_width; self.output_width = output_width
        self.reference_width = reference_width; self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads; self.head_dim = head_dim
        self.intermediate_size = intermediate_size; self.bias = bias; self.rope = rope
        self.eps = float(eps); self.fused_norms = fused_norms
        self.norm = SliceGPTRMSNorm(input_width, reference_width=reference_width, eps=eps,
                                     learnable_scale=not fused_norms, device=device, dtype=dtype)
        if kind == "attention":
            self.consumer = ReferenceAttention(input_width, num_heads, num_kv_heads, head_dim,
                                                bias=bias, rope=rope, device=device, dtype=dtype)
            self.consumer.o_proj = nn.Linear(num_heads * head_dim, output_width, bias=bias, device=device, dtype=dtype)
        else:
            self.consumer = ReferenceMLP(input_width, intermediate_size, bias=bias, device=device, dtype=dtype)
            self.consumer.down_proj = nn.Linear(intermediate_size, output_width, bias=bias, device=device, dtype=dtype)
        matrix = (torch.eye(input_width, output_width, device=device, dtype=dtype) if transition is None
                  else _fixed_tensor(transition, (input_width, output_width), dtype=dtype, device=device))
        if transition is not None:
            # A.T B for orthonormal columns is a contraction. Full bases must
            # give an orthogonal matrix; this is a useful config semantic check.
            tolerance = 2e-5 if dtype == torch.float32 else 2e-10
            singular = torch.linalg.svdvals(matrix.detach().double())
            if (bool((singular > 1 + tolerance).any()) or
                    (input_width == output_width == reference_width
                     and not torch.allclose(singular, torch.ones_like(singular), atol=tolerance, rtol=tolerance))):
                raise ValueError("SliceGPT transition must be an orthogonal-basis contraction")
        self.register_buffer("transition", matrix)

    def apply_branch(self, x, *, cache=None, use_cache=False, **kwargs):
        normalized = self.norm(x)
        if self.kind == "attention":
            out = self.consumer.attention(normalized, cache=cache, use_cache=use_cache, **kwargs)
            return out.output, out.cache
        if cache is not None:
            raise ValueError("SliceGPT MLP has no KV cache")
        return self.consumer(normalized), None

    def forward(self, x, **kwargs):
        branch, _ = self.apply_branch(x, **kwargs)
        return x @ self.transition + branch

    def representation_config(self):
        config = {"layer_type": "SliceGPTBlock", "version": 1, "kind": self.kind,
                "input_width": self.input_width, "output_width": self.output_width,
                "reference_width": self.reference_width, "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads, "head_dim": self.head_dim,
                "intermediate_size": self.intermediate_size, "bias": self.bias, "rope": self.rope,
                "eps": self.eps, "fused_norms": self.fused_norms,
                "transition": self.transition.detach().cpu().tolist(), "dtype": _dtype_name(self.transition.dtype)}
        _check_reconstruction(self, config)
        return config

    @classmethod
    def from_config(cls, config, *, device=None, dtype=None):
        keys = ("kind", "input_width", "output_width", "reference_width", "num_heads", "num_kv_heads",
                "head_dim", "intermediate_size", "bias", "rope", "eps", "fused_norms", "transition", "dtype")
        values = _config(config, cls.__name__, keys)
        declared_dtype = _dtype(values["dtype"])
        if dtype is not None and dtype != declared_dtype:
            raise ValueError("SliceGPT checkpoint dtype conflicts with its configuration")
        values["dtype"] = declared_dtype; values["device"] = device
        return cls(**values)

    def validate_checkpoint_state(self, state, prefix=""):
        _validate_tensors(self, state, prefix)
        _validate_fixed(state, prefix, "transition", self.transition)
        self.norm.validate_checkpoint_state(state, prefix + "norm.")
        if self.kind == "attention":
            for name in ("head_map", "group_coordinates"):
                _validate_fixed(state, prefix + "consumer.", name, getattr(self.consumer, name))


@dataclass(frozen=True)
class SliceGPTOutput:
    logits: torch.Tensor
    caches: tuple[KVCache, ...]


class SliceGPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, num_kv_heads,
                 head_dim, intermediate_size, *, widths=None, tie_word_embeddings=True,
                 bias=True, rope=True, eps=1e-6, fused_norms=False, transitions=None,
                 head_bridge=None, device=None, dtype=None):
        super().__init__()
        _positive(vocab_size, d_model, num_layers, num_heads, num_kv_heads, head_dim, intermediate_size)
        _bound_dimensions(d_model, num_heads, num_kv_heads, head_dim, intermediate_size)
        if 2 * num_layers > MAX_RESIDUAL_BLOCKS:
            raise ValueError("SliceGPT depth exceeds the bounded reference architecture")
        if any(type(v) is not bool for v in (tie_word_embeddings, bias, rope, fused_norms)):
            raise ValueError("SliceGPT model flags must be boolean")
        if widths is not None and not isinstance(widths, (tuple, list)):
            raise ValueError("SliceGPT widths require an explicit finite list/tuple")
        widths = (d_model,) * (2 * num_layers + 1) if widths is None else tuple(widths)
        if len(widths) != 2 * num_layers + 1 or any(type(v) is not int or not 1 <= v <= d_model for v in widths):
            raise ValueError("SliceGPT widths must describe embedding and every residual output")
        if transitions is not None and (not isinstance(transitions, (tuple, list))
                                        or len(transitions) != 2 * num_layers):
            raise ValueError("SliceGPT requires one transition per residual edge")
        if rope and head_dim % 2:
            raise ValueError("SliceGPT RoPE requires an even head dimension")
        # Count persistent parameters and buffers, including shared embedding
        # only once, residual matrices, tied head bridge and attention metadata.
        # This pure integer admission check precedes every nn allocation and is
        # also reached by from_config when deserializing an untrusted descriptor.
        head_width = widths[0] if tie_word_embeddings else widths[-1]
        elements = vocab_size * widths[0] + widths[-1] * head_width + 1
        elements += 0 if fused_norms else widths[-1]
        elements += 0 if tie_word_embeddings else vocab_size * widths[-1]
        elements += vocab_size if bias else 0
        elements += sum(_block_elements("attention" if i % 2 == 0 else "mlp", ki, ko,
            num_heads, num_kv_heads, head_dim, intermediate_size, bias, fused_norms)
            for i, (ki, ko) in enumerate(zip(widths, widths[1:])))
        _bound_elements(elements)
        dtype = torch.get_default_dtype() if dtype is None else dtype
        _dtype_name(dtype)
        self.vocab_size = vocab_size; self.d_model = d_model; self.num_layers = num_layers
        self.num_heads = num_heads; self.num_kv_heads = num_kv_heads; self.head_dim = head_dim
        self.intermediate_size = intermediate_size; self.widths = widths
        self.tie_word_embeddings = tie_word_embeddings; self.bias = bias; self.rope = rope
        self.eps = float(eps); self.fused_norms = fused_norms
        self.embedding = nn.Embedding(vocab_size, widths[0], device=device, dtype=dtype)
        self.blocks = nn.ModuleList([
            SliceGPTBlock("attention" if i % 2 == 0 else "mlp", widths[i], widths[i + 1], d_model,
                num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
                intermediate_size=intermediate_size, bias=bias, rope=rope, eps=eps,
                fused_norms=fused_norms, transition=None if transitions is None else transitions[i],
                device=device, dtype=dtype) for i in range(2 * num_layers)])
        self.final_norm = SliceGPTRMSNorm(widths[-1], reference_width=d_model, eps=eps,
                                           learnable_scale=not fused_norms, device=device, dtype=dtype)
        head_width = widths[0] if tie_word_embeddings else widths[-1]
        self.lm_head = nn.Linear(head_width, vocab_size, bias=bias, device=device, dtype=dtype)
        if tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight
        bridge = (torch.eye(widths[-1], head_width, device=device, dtype=dtype) if head_bridge is None
                  else _fixed_tensor(head_bridge, (widths[-1], head_width), dtype=dtype, device=device))
        self.register_buffer("head_bridge", bridge)

    def _tokens(self, token_ids):
        if (not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 2 or not token_ids.numel()
                or token_ids.dtype not in (torch.int32, torch.int64)
                or token_ids.device != self.embedding.weight.device
                or bool((token_ids < 0).any()) or bool((token_ids >= self.vocab_size).any())):
            raise ValueError("SliceGPT token IDs require nonempty compatible integer [batch,tokens]")

    def causal_forward(self, token_ids, *, caches=None, use_cache=False, padding_mask=None,
                       attention_mask=None, positions=None):
        self._tokens(token_ids)
        if type(use_cache) is not bool or (caches is not None and len(caches) != self.num_layers):
            raise ValueError("SliceGPT requires one cache per attention block")
        x = self.embedding(token_ids); outputs = []
        for i, block in enumerate(self.blocks):
            cache = None if caches is None or i % 2 else caches[i // 2]
            branch, next_cache = block.apply_branch(x, cache=cache, use_cache=use_cache,
                padding_mask=padding_mask, attention_mask=attention_mask, positions=positions)
            x = x @ block.transition + branch
            if i % 2 == 0 and use_cache:
                outputs.append(next_cache)
        logits = self.lm_head(self.final_norm(x) @ self.head_bridge)
        return SliceGPTOutput(logits, tuple(outputs))

    def forward(self, token_ids, **kwargs):
        return self.causal_forward(token_ids, **kwargs).logits

    def representation_config(self):
        config = {"layer_type": "SliceGPTModel", "version": 1, "vocab_size": self.vocab_size,
                "d_model": self.d_model, "num_layers": self.num_layers, "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads, "head_dim": self.head_dim,
                "intermediate_size": self.intermediate_size, "widths": list(self.widths),
                "tie_word_embeddings": self.tie_word_embeddings, "bias": self.bias, "rope": self.rope,
                "eps": self.eps, "fused_norms": self.fused_norms,
                "transitions": [b.transition.detach().cpu().tolist() for b in self.blocks],
                "head_bridge": self.head_bridge.detach().cpu().tolist(),
                "dtype": _dtype_name(self.embedding.weight.dtype)}
        _check_reconstruction(self, config)
        return config

    @classmethod
    def from_config(cls, config, *, device=None, dtype=None):
        keys = ("vocab_size", "d_model", "num_layers", "num_heads", "num_kv_heads", "head_dim",
                "intermediate_size", "widths", "tie_word_embeddings", "bias", "rope", "eps",
                "fused_norms", "transitions", "head_bridge", "dtype")
        values = _config(config, cls.__name__, keys)
        declared_dtype = _dtype(values["dtype"])
        if dtype is not None and dtype != declared_dtype:
            raise ValueError("SliceGPT checkpoint dtype conflicts with its configuration")
        values["dtype"] = declared_dtype; values["device"] = device
        return cls(**values)

    def validate_checkpoint_state(self, state, prefix=""):
        # Check the live graph after a checkpoint loader applies its alias
        # topology. Shape/value validation alone cannot reject a K/V retie.
        # reconstruction checks state tensors directly and never calls this
        # validator, so this does not recurse.
        SliceGPTModel.representation_config(self)
        _validate_tensors(self, state, prefix)
        _validate_fixed(state, prefix, "head_bridge", self.head_bridge)
        self.final_norm.validate_checkpoint_state(state, prefix + "final_norm.")
        for i, block in enumerate(self.blocks):
            block.validate_checkpoint_state(state, prefix + f"blocks.{i}.")
        if self.tie_word_embeddings:
            embedding, head = state.get(prefix + "embedding.weight"), state.get(prefix + "lm_head.weight")
            if (not isinstance(embedding, torch.Tensor) or not isinstance(head, torch.Tensor)
                    or embedding.dtype != head.dtype or embedding.shape != head.shape
                    or not torch.equal(embedding, head)):
                raise ValueError("SliceGPT checkpoint has conflicting tied embedding/head weights")
