"""A small causal token Transformer with directly trained LED projections.

Paper: arXiv:1910.13923v3, eqs. (1), (3)-(5). Attention uses the standard
softmax(Q K.T / sqrt(d_k)) V, correcting the printed eq. (2). This bounded
reconstruction deliberately has no acoustic front end, encoder cross-attention,
dropout, beam search, CER implementation or claim to reproduce paper results.
"""
from dataclasses import asdict
import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .lrt_specs import (LRT, lrt_metadata_values, lrt_parameter_counts,
                        lrt_spec_from_config)


class LRTProfileError(ValueError):
    """The bounded from-scratch architecture contract was violated."""


def _runtime(device, dtype):
    try:
        device = torch.device("cpu" if device is None else device)
    except (TypeError, RuntimeError, ValueError) as error:
        raise LRTProfileError("LRT requires a CPU device") from error
    dtype = torch.float32 if dtype is None else dtype
    if device.type != "cpu" or device.index is not None or dtype not in (torch.float32, torch.float64):
        raise LRTProfileError("LRT supports CPU float32/float64 only")
    return device, dtype


def _initialize(module, initialization, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initializer = (nn.init.xavier_uniform_ if initialization == "xavier_uniform"
                   else nn.init.xavier_normal_)
    for child in module.modules():
        if isinstance(child, (nn.Linear, nn.Embedding)):
            initializer(child.weight, generator=generator)
            if isinstance(child, nn.Linear) and child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.LayerNorm):
            nn.init.ones_(child.weight)
            nn.init.zeros_(child.bias)


def _finite(tensor, name):
    if not bool(torch.isfinite(tensor).all()):
        raise LRTProfileError(f"{name} must contain finite values")


def _float_input(inputs, reference, width, name="inputs"):
    if (not isinstance(inputs, torch.Tensor) or inputs.ndim < 1
            or inputs.shape[-1] != width or any(size < 1 for size in inputs.shape)
            or inputs.device.type != "cpu" or inputs.dtype not in (torch.float32, torch.float64)
            or inputs.dtype != reference.dtype or inputs.device != reference.device):
        raise LRTProfileError(f"{name} require nonempty CPU floating tensors with matching dtype and width {width}")
    _finite(inputs, name)


class LEDLinear(nn.Module):
    """Two affine-compatible linear factors; no activation between E and D.

    Only D has a bias. In PyTorch storage the effective weight is D.weight @
    E.weight; forward never constructs that dense matrix.
    """
    def __init__(self, in_features, out_features, rank, bias=True, *,
                 initialization="xavier_uniform", seed=0, device="cpu", dtype=None):
        super().__init__()
        if (any(type(n) is not int or n < 1 for n in (in_features, out_features, rank))
                or rank > min(in_features, out_features) or type(bias) is not bool
                or in_features > 16384 or out_features > 16384
                or rank*(in_features+out_features) > 10_000_000):
            raise LRTProfileError("LED requires bounded positive dimensions and rank <= min(in_features,out_features)")
        if type(initialization) is not str or initialization not in ("xavier_uniform", "xavier_normal"):
            raise LRTProfileError("Unsupported LED initialization")
        if type(seed) is not int or seed < 0 or seed >= 2**63:
            raise LRTProfileError("LED seed must be a nonnegative signed 64-bit integer")
        device, dtype = _runtime(device, dtype)
        self.in_features, self.out_features, self.rank = in_features, out_features, rank
        self.initialization, self.seed = initialization, seed
        # nn.Linear constructors consume the CPU generator even though their
        # initial values are overwritten. Restore it on success and failure.
        with torch.random.fork_rng(devices=[]):
            self.encoder = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
            self.decoder = nn.Linear(rank, out_features, bias=bias, device=device, dtype=dtype)
            _initialize(self, initialization, seed)

    def forward(self, inputs):
        _float_input(inputs, self.encoder.weight, self.in_features)
        result = self.decoder(self.encoder(inputs))
        _finite(result, "LED outputs")
        return result

    def representation_config(self):
        return {"layer_type": "LEDLinear", "version": 1, "in_features": self.in_features,
                "out_features": self.out_features, "rank": self.rank,
                "bias": self.decoder.bias is not None,
                "initialization": self.initialization, "seed": self.seed}

    @classmethod
    def from_config(cls, config, *, device="cpu", dtype=None):
        names = {"layer_type", "version", "in_features", "out_features", "rank", "bias", "initialization", "seed"}
        if (type(config) is not dict or set(config) != names
                or config["layer_type"] != "LEDLinear" or type(config["version"]) is not int
                or config["version"] != 1):
            raise LRTProfileError("Invalid LED representation descriptor")
        return cls(**{key: value for key, value in config.items() if key not in ("layer_type", "version")},
                   device=device, dtype=dtype)


def _projection(inp, out, rank, spec, factorized, device, dtype):
    if factorized:
        return LEDLinear(inp, out, rank, spec.bias, initialization=spec.initialization,
                         seed=spec.seed, device=device, dtype=dtype)
    return nn.Linear(inp, out, bias=spec.bias, device=device, dtype=dtype)


def _attention_mask(inputs, attention_mask, key_padding_mask, causal):
    batch, length, _ = inputs.shape
    if type(causal) is not bool:
        raise LRTProfileError("causal must be a boolean")
    blocked = torch.zeros((batch, length, length), dtype=torch.bool, device=inputs.device)
    if causal:
        blocked |= torch.ones((length, length), dtype=torch.bool, device=inputs.device).triu(1)
    if attention_mask is not None:
        if (not isinstance(attention_mask, torch.Tensor) or attention_mask.dtype != torch.bool
                or attention_mask.device != inputs.device
                or tuple(attention_mask.shape) not in ((length, length), (batch, length, length))):
            raise LRTProfileError("attention_mask must be boolean [tokens,tokens] or [batch,tokens,tokens]; True blocks")
        blocked |= attention_mask
    if key_padding_mask is not None:
        if (not isinstance(key_padding_mask, torch.Tensor) or key_padding_mask.dtype != torch.bool
                or key_padding_mask.device != inputs.device or tuple(key_padding_mask.shape) != (batch, length)):
            raise LRTProfileError("key_padding_mask must be boolean [batch,tokens]; True blocks")
        blocked |= key_padding_mask.unsqueeze(1)
    if bool(blocked.all(dim=-1).any()):
        raise LRTProfileError("Every attention query must have at least one unmasked key")
    return blocked


def _forward_budget(spec, factorized, dtype, batch, length):
    """Conservative forward tensor reserve before B*T*T/embedding allocations.

    This is a capability guard, not a guarantee about allocator RSS. Optimizer
    state and the caller's inputs remain the training runner's responsibility.
    """
    from .lrt_profiles import plan_lrt
    plan = plan_lrt(spec, dtype=dtype, factorized=factorized)
    itemsize = 4 if dtype == torch.float32 else 8
    projected = spec.num_heads*(2*spec.d_k+spec.d_v)
    factor_width = (spec.num_heads*(spec.q_rank+spec.k_rank+spec.v_rank)
                    +spec.out_rank+spec.ffn_in_rank+spec.ffn_out_rank) if factorized else 0
    per_token = spec.num_layers*(12*spec.d_model+4*projected+4*spec.d_ff+4*factor_width)+4*spec.vocab_size
    reserve = plan.tensor_bytes+itemsize*batch*(length*per_token+8*spec.num_layers*spec.num_heads*length*length)
    if reserve > spec.max_peak_bytes:
        raise LRTProfileError("Forward tensor reserve exceeds max_peak_bytes for this batch/sequence")


class LRTAttention(nn.Module):
    """Per-head LED Q/K/V followed by standard bilinear attention.

    Returned probabilities have shape [batch, heads, tokens, tokens], irrespective
    of factor rank. The output projection preserves d_model, without a residual
    or normalization here; the enclosing Transformer block owns those operations.
    """
    def __init__(self, spec, *, factorized=True, device="cpu", dtype=None):
        super().__init__()
        if type(spec) is not LRT or type(factorized) is not bool:
            raise LRTProfileError("LRTAttention requires an explicit LRT spec and factorized boolean")
        # The same whole-model guard protects direct attention construction.
        from .lrt_profiles import plan_lrt
        device, dtype = _runtime(device, dtype)
        plan_lrt(spec, dtype=dtype, factorized=factorized)
        self.spec, self.factorized = spec, factorized
        with torch.random.fork_rng(devices=[]):
            self.q_proj = nn.ModuleList([_projection(spec.d_model, spec.d_k, spec.q_rank, spec,
                factorized, device, dtype) for _ in range(spec.num_heads)])
            self.k_proj = nn.ModuleList([_projection(spec.d_model, spec.d_k, spec.k_rank, spec,
                factorized, device, dtype) for _ in range(spec.num_heads)])
            self.v_proj = nn.ModuleList([_projection(spec.d_model, spec.d_v, spec.v_rank, spec,
                factorized, device, dtype) for _ in range(spec.num_heads)])
            self.out_proj = _projection(spec.num_heads*spec.d_v, spec.d_model, spec.out_rank,
                                        spec, factorized, device, dtype)
            _initialize(self, spec.initialization, spec.seed)

    def forward(self, inputs, *, attention_mask=None, key_padding_mask=None,
                causal=True, return_attention=False):
        reference = next(self.parameters())
        _float_input(inputs, reference, self.spec.d_model)
        if inputs.ndim != 3 or inputs.shape[1] > self.spec.max_seq_len:
            raise LRTProfileError("Attention inputs require [batch,tokens,d_model] within max_seq_len")
        if type(return_attention) is not bool:
            raise LRTProfileError("return_attention must be a boolean")
        _forward_budget(self.spec, self.factorized, reference.dtype, inputs.shape[0], inputs.shape[1])
        blocked = _attention_mask(inputs, attention_mask, key_padding_mask, causal)
        query = torch.stack([projection(inputs) for projection in self.q_proj], dim=1)
        key = torch.stack([projection(inputs) for projection in self.k_proj], dim=1)
        value = torch.stack([projection(inputs) for projection in self.v_proj], dim=1)
        scores = (query @ key.transpose(-1, -2))/math.sqrt(self.spec.d_k)
        _finite(scores, "Unmasked attention scores")
        probabilities = torch.softmax(scores.masked_fill(blocked.unsqueeze(1), -torch.inf), dim=-1)
        attended = probabilities @ value
        merged = attended.transpose(1, 2).reshape(inputs.shape[0], inputs.shape[1], -1)
        output = self.out_proj(merged)
        _finite(output, "Attention outputs")
        return (output, probabilities) if return_attention else output


class _LRTBlock(nn.Module):
    def __init__(self, spec, factorized, device, dtype):
        super().__init__()
        self.attention = LRTAttention(spec, factorized=factorized, device=device, dtype=dtype)
        self.attention_norm = nn.LayerNorm(spec.d_model, device=device, dtype=dtype)
        self.ffn_in = _projection(spec.d_model, spec.d_ff, spec.ffn_in_rank, spec, factorized, device, dtype)
        self.ffn_out = _projection(spec.d_ff, spec.d_model, spec.ffn_out_rank, spec, factorized, device, dtype)
        self.ffn_norm = nn.LayerNorm(spec.d_model, device=device, dtype=dtype)

    def forward(self, inputs, attention_mask, key_padding_mask):
        hidden = self.attention_norm(inputs+self.attention(inputs, attention_mask=attention_mask,
                                                         key_padding_mask=key_padding_mask))
        # This is the original FFN nonlinearity, between two LED units.
        return self.ffn_norm(hidden+self.ffn_out(F.relu(self.ffn_in(hidden))))


def _positions(spec, dtype):
    # Canonical float32 buffer is stable under float32/float64 .to() and reload.
    positions = torch.arange(spec.max_seq_len, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, spec.d_model, 2, dtype=torch.float32)
                            *(-math.log(10000.0)/spec.d_model))
    angles = positions*frequencies
    encoding = torch.empty(spec.max_seq_len, spec.d_model, dtype=torch.float32)
    encoding[:, 0::2] = angles.sin()
    encoding[:, 1::2] = angles[:, :spec.d_model//2].cos()
    return encoding.to(dtype=dtype)


class LRTTransformer(nn.Module):
    """Trainable causal token logits [batch,tokens,vocab], built from scratch."""
    def __init__(self, spec, *, factorized=True, device="cpu", dtype=None):
        super().__init__()
        if type(spec) is not LRT or type(factorized) is not bool:
            raise LRTProfileError("LRTTransformer requires an explicit LRT spec and factorized boolean")
        from .lrt_profiles import plan_lrt
        device, dtype = _runtime(device, dtype)
        plan_lrt(spec, dtype=dtype, factorized=factorized)
        self.spec, self.factorized = spec, factorized
        with torch.random.fork_rng(devices=[]):
            self.embedding = nn.Embedding(spec.vocab_size, spec.d_model, device=device, dtype=dtype)
            self.blocks = nn.ModuleList([_LRTBlock(spec, factorized, device, dtype)
                                        for _ in range(spec.num_layers)])
            self.output = nn.Linear(spec.d_model, spec.vocab_size, bias=spec.bias, device=device, dtype=dtype)
            self.register_buffer("position_encoding", _positions(spec, dtype))
            self.register_buffer("_architecture_metadata", torch.tensor(
                lrt_metadata_values(spec, factorized), dtype=torch.int64, device=device))
            _initialize(self, spec.initialization, spec.seed)
        if sum(parameter.numel() for parameter in self.parameters()) != lrt_parameter_counts(spec)[
                "factorized" if factorized else "dense"]:
            raise LRTProfileError("LRT constructor disagrees with its whole-model parameter plan")

    def forward(self, tokens, *, attention_mask=None, key_padding_mask=None):
        if (not isinstance(tokens, torch.Tensor) or tokens.ndim != 2
                or tokens.dtype != torch.int64 or tokens.device.type != "cpu"
                or any(size < 1 for size in tokens.shape) or tokens.shape[1] > self.spec.max_seq_len):
            raise LRTProfileError("LRT tokens require nonempty CPU int64 [batch,tokens] within max_seq_len")
        if self.embedding.weight.device != tokens.device or self.embedding.weight.dtype not in (torch.float32, torch.float64):
            raise LRTProfileError("LRT model must remain CPU float32/float64")
        if bool((tokens < 0).any()) or bool((tokens >= self.spec.vocab_size).any()):
            raise LRTProfileError("Token IDs must lie in [0,vocab_size)")
        _forward_budget(self.spec, self.factorized, self.embedding.weight.dtype, tokens.shape[0], tokens.shape[1])
        hidden = self.embedding(tokens)*math.sqrt(self.spec.d_model)+self.position_encoding[:tokens.shape[1]]
        for block in self.blocks:
            hidden = block(hidden, attention_mask, key_padding_mask)
        logits = self.output(hidden)
        _finite(logits, "LRT logits")
        return logits

    def representation_config(self):
        return {"layer_type": "LRTTransformer", "version": 1,
                "factorized": self.factorized, "spec": asdict(self.spec)}

    @classmethod
    def from_config(cls, config, *, device="cpu", dtype=None):
        if (type(config) is not dict or set(config) != {"layer_type", "version", "factorized", "spec"}
                or config["layer_type"] != "LRTTransformer" or type(config["version"]) is not int
                or config["version"] != 1 or type(config["factorized"]) is not bool):
            raise LRTProfileError("Invalid LRT architecture descriptor")
        spec = lrt_spec_from_config(config["spec"])
        return cls(spec, factorized=config["factorized"], device=device, dtype=dtype)

    def validate_checkpoint_state(self, state, prefix=""):
        """Reject missing/altered fixed buffers and incompatible trained weights."""
        if not isinstance(state, Mapping) or type(prefix) is not str:
            raise LRTProfileError("A checkpoint state mapping and string prefix are required")
        if any(type(key) is not str for key in state):
            raise LRTProfileError("Checkpoint tensor keys must be strings")
        self._validate_structure()
        expected = self.state_dict()
        selected = {key[len(prefix):]: value for key, value in state.items()
                    if type(key) is str and key.startswith(prefix)}
        if set(selected) != set(expected):
            raise LRTProfileError("Checkpoint keys must exactly match the declared LRT architecture")
        for key, tensor in selected.items():
            reference = expected[key]
            if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                    or tensor.device.type != "cpu" or tensor.shape != reference.shape
                    or tensor.dtype != reference.dtype):
                raise LRTProfileError(f"Checkpoint tensor shape/dtype/device mismatch: {prefix+key}")
            if tensor.is_floating_point():
                _finite(tensor, f"Checkpoint tensor {prefix+key}")
        metadata = torch.tensor(lrt_metadata_values(self.spec, self.factorized), dtype=torch.int64)
        if not torch.equal(selected["_architecture_metadata"], metadata):
            raise LRTProfileError("Checkpoint architecture metadata was altered")
        if not torch.equal(selected["position_encoding"], _positions(self.spec, self.embedding.weight.dtype)):
            raise LRTProfileError("Checkpoint positional encoding was altered")

    def _validate_structure(self):
        """Reject executable mutations that a state_dict cannot represent."""
        try:
            self._validate_structure_impl()
        except (AttributeError, TypeError) as error:
            raise LRTProfileError("Malformed live LRT architecture") from error

    def _validate_structure_impl(self):
        spec = self.spec
        if type(spec) is not LRT or type(self.factorized) is not bool:
            raise LRTProfileError("Invalid live LRT architecture metadata")
        if (set(self._modules) != {"embedding", "blocks", "output"}
                or set(self._buffers) != {"position_encoding", "_architecture_metadata"}
                or self._parameters or type(self.embedding) is not nn.Embedding
                or type(self.blocks) is not nn.ModuleList or len(self.blocks) != spec.num_layers
                or self.embedding.num_embeddings != spec.vocab_size or self.embedding.embedding_dim != spec.d_model
                or self.embedding.padding_idx is not None or self.embedding.max_norm is not None
                or self.embedding.norm_type != 2 or self.embedding.scale_grad_by_freq
                or self.embedding.sparse):
            raise LRTProfileError("Live LRT embedding/module structure disagrees with its descriptor")
        _runtime(self.embedding.weight.device, self.embedding.weight.dtype)
        linear_modules = []

        def linear(module, inp, out, bias):
            if (type(module) is not nn.Linear or module.in_features != inp or module.out_features != out
                    or module._modules or tuple(module.weight.shape) != (out, inp)
                    or (module.bias is not None) != bias
                    or (bias and tuple(module.bias.shape) != (out,))):
                raise LRTProfileError("Live Linear attributes disagree with the LRT descriptor")
            linear_modules.append(module)

        def projection(module, inp, out, rank):
            if self.factorized:
                if (type(module) is not LEDLinear or module.in_features != inp or module.out_features != out
                        or module.rank != rank or module.initialization != spec.initialization or module.seed != spec.seed
                        or set(module._modules) != {"encoder", "decoder"} or module._parameters or module._buffers):
                    raise LRTProfileError("Live LED topology disagrees with the LRT descriptor")
                linear(module.encoder, inp, rank, False)
                linear(module.decoder, rank, out, spec.bias)
            else:
                linear(module, inp, out, spec.bias)

        linear(self.output, spec.d_model, spec.vocab_size, spec.bias)
        for block in self.blocks:
            if (type(block) is not _LRTBlock or set(block._modules) != {
                    "attention", "attention_norm", "ffn_in", "ffn_out", "ffn_norm"}
                    or block._parameters or block._buffers):
                raise LRTProfileError("Live Transformer block disagrees with its descriptor")
            attention = block.attention
            if (type(attention) is not LRTAttention or attention.spec != spec
                    or attention.factorized != self.factorized or set(attention._modules) != {
                    "q_proj", "k_proj", "v_proj", "out_proj"} or attention._parameters or attention._buffers):
                raise LRTProfileError("Live attention topology disagrees with its descriptor")
            for modules, width, rank in ((attention.q_proj, spec.d_k, spec.q_rank),
                    (attention.k_proj, spec.d_k, spec.k_rank), (attention.v_proj, spec.d_v, spec.v_rank)):
                if type(modules) is not nn.ModuleList or len(modules) != spec.num_heads:
                    raise LRTProfileError("Live attention head count disagrees with its descriptor")
                for module in modules:
                    projection(module, spec.d_model, width, rank)
            projection(attention.out_proj, spec.num_heads*spec.d_v, spec.d_model, spec.out_rank)
            projection(block.ffn_in, spec.d_model, spec.d_ff, spec.ffn_in_rank)
            projection(block.ffn_out, spec.d_ff, spec.d_model, spec.ffn_out_rank)
            for norm in (block.attention_norm, block.ffn_norm):
                if (type(norm) is not nn.LayerNorm or norm.normalized_shape != (spec.d_model,)
                        or norm.eps != 1e-5 or not norm.elementwise_affine
                        or norm.weight is None or norm.bias is None or norm._modules):
                    raise LRTProfileError("Live LayerNorm attributes disagree with the LRT descriptor")
        if (len({id(module) for module in linear_modules}) != len(linear_modules)
                or sum(parameter.numel() for parameter in self.parameters()) != lrt_parameter_counts(spec)[
                    "factorized" if self.factorized else "dense"]):
            raise LRTProfileError("Unexpected module or parameter sharing in LRT")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.validate_checkpoint_state(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
