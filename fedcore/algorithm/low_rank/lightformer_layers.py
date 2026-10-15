"""Small pre-norm encoder/decoder runtime with complete group object sharing.

Only ordinary multi-head self/cross attention, ReLU FFNs and sinusoidal
positions are supported. Every application computes its own activations; no
KV cache is stored on a shared attention object.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .lightformer_specs import validate_groups
from .structured_layers import FactorizedLinear


# Fixed limits also protect standalone checkpoint construction, where a
# transfer plan is unavailable. Elements include dense fixed components,
# unique block parameters, positions and integer provenance/group buffers.
MAX_REFERENCE_ELEMENTS = 1_000_000
MAX_REFERENCE_DEPTH = 128
MAX_REFERENCE_POSITIONS = 4096


def _dimensions(*values):
    if any(type(v) is not int or v < 1 for v in values):
        raise ValueError("LightFormer dimensions must be positive integers")


def _reference_bound(src_vocab, tgt_vocab, d, f, enc_groups, dec_groups, rank, bias, positions):
    if max(len(enc_groups), len(dec_groups)) > MAX_REFERENCE_DEPTH or positions > MAX_REFERENCE_POSITIONS:
        raise ValueError("LightFormer bounded reference profile depth/position limit exceeded before allocation")
    enc, dec, b = max(enc_groups) + 1, max(dec_groups) + 1, int(bias)
    attention = (4 * d * d if rank is None else 8 * d * rank) + b * 4 * d
    ffn = (2 * d * f if rank is None else 2 * rank * (d + f)) + b * (d + f)
    fixed = (src_vocab + 2 * tgt_vocab) * d + b * tgt_vocab + 4 * d
    elements = fixed + enc * (attention + ffn + 4 * d) + dec * (2 * attention + ffn + 6 * d)
    elements += positions * d + len(enc_groups) + len(dec_groups) + enc + dec
    if elements > MAX_REFERENCE_ELEMENTS:
        raise ValueError("LightFormer bounded reference profile persistent-element limit exceeded before allocation")
    return elements


def _linear(in_features, out_features, rank, bias, device, dtype):
    if rank is None:
        return nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
    return FactorizedLinear(in_features, out_features, rank, bias=bias, device=device, dtype=dtype)


def _input(x, width, reference, name):
    if (not isinstance(x, torch.Tensor) or x.ndim != 3 or not x.shape[0] or not x.shape[1]
            or x.shape[-1] != width or x.dtype != reference.dtype or x.device != reference.device
            or not bool(torch.isfinite(x).all())):
        raise ValueError(f"{name} requires finite compatible [batch,tokens,d_model] states")


def _padding(mask, batch, tokens, device):
    if mask is not None and (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
            or mask.shape != (batch, tokens) or mask.device != device):
        raise ValueError("Padding mask requires boolean [batch,tokens], True means valid")


def _positions(max_positions, d_model, *, device, dtype):
    position = torch.arange(max_positions, device=device, dtype=torch.float64)[:, None]
    frequencies = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float64)
                            * (-math.log(10000.0) / d_model))
    positions = torch.zeros((max_positions, d_model), device=device, dtype=torch.float64)
    positions[:, 0::2] = torch.sin(position * frequencies)
    positions[:, 1::2] = torch.cos(position * frequencies[:d_model // 2])
    return positions.to(dtype=dtype)


class LightFormerAttention(nn.Module):
    def __init__(self, d_model, num_heads, *, rank=None, bias=True, device=None, dtype=None):
        super().__init__()
        _dimensions(d_model, num_heads)
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if rank is not None and (type(rank) is not int or not 1 <= rank <= d_model):
            raise ValueError("Attention rank must fit d_model")
        if type(bias) is not bool:
            raise ValueError("bias must be boolean")
        self.d_model, self.num_heads, self.rank, self.bias = d_model, num_heads, rank, bias
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, _linear(d_model, d_model, rank, bias, device, dtype))

    def forward(self, query, key_value=None, *, padding_mask=None, causal=False,
                cache=None, use_cache=False):
        if cache is not None or use_cache is not False:
            raise ValueError("LightFormer reference has no KV cache; shared weights do not share caches")
        if type(causal) is not bool:
            raise ValueError("causal must be boolean")
        reference = next(self.parameters())
        _input(query, self.d_model, reference, "Attention query")
        key_value = query if key_value is None else key_value
        _input(key_value, self.d_model, reference, "Attention key/value")
        b, t, _ = query.shape
        if key_value.shape[0] != b:
            raise ValueError("Attention query and memory batch sizes must agree")
        s = key_value.shape[1]
        _padding(padding_mask, b, s, query.device)
        if causal and s != t:
            raise ValueError("Causal reference attention requires equal query/key lengths")
        h, width = self.num_heads, self.d_model // self.num_heads
        q = self.q_proj(query).reshape(b, t, h, width).transpose(1, 2)
        k = self.k_proj(key_value).reshape(b, s, h, width).transpose(1, 2)
        v = self.v_proj(key_value).reshape(b, s, h, width).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(width)
        allowed = torch.ones((b, 1, t, s), dtype=torch.bool, device=query.device)
        if causal:
            allowed = allowed & torch.ones((t, s), dtype=torch.bool, device=query.device).tril()[None, None]
        if padding_mask is not None:
            allowed = allowed & padding_mask[:, None, None, :]
        has_keys = allowed.any(-1, keepdim=True)
        masked = scores.masked_fill(~allowed, -torch.inf)
        probabilities = torch.softmax(torch.where(has_keys, masked, torch.zeros_like(masked)), -1)
        probabilities = torch.where(allowed, probabilities, torch.zeros_like(probabilities))
        contexts = (probabilities @ v).transpose(1, 2).reshape(b, t, self.d_model)
        return self.o_proj(contexts)


class LightFormerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, *, rank=None, bias=True,
                 norm_eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.self_attention = LightFormerAttention(d_model, num_heads, rank=rank, bias=bias,
                                                   device=device, dtype=dtype)
        self.ffn_up = _linear(d_model, dim_feedforward, rank, bias, device, dtype)
        self.ffn_down = _linear(dim_feedforward, d_model, rank, bias, device, dtype)
        self.self_norm = nn.LayerNorm(d_model, eps=norm_eps, device=device, dtype=dtype)
        self.ffn_norm = nn.LayerNorm(d_model, eps=norm_eps, device=device, dtype=dtype)

    def forward(self, x, *, padding_mask=None):
        x = x + self.self_attention(self.self_norm(x), padding_mask=padding_mask)
        return x + self.ffn_down(F.relu(self.ffn_up(self.ffn_norm(x))))


class LightFormerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, *, rank=None, bias=True,
                 norm_eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.self_attention = LightFormerAttention(d_model, num_heads, rank=rank, bias=bias,
                                                   device=device, dtype=dtype)
        self.cross_attention = LightFormerAttention(d_model, num_heads, rank=rank, bias=bias,
                                                    device=device, dtype=dtype)
        self.ffn_up = _linear(d_model, dim_feedforward, rank, bias, device, dtype)
        self.ffn_down = _linear(dim_feedforward, d_model, rank, bias, device, dtype)
        self.self_norm = nn.LayerNorm(d_model, eps=norm_eps, device=device, dtype=dtype)
        self.cross_norm = nn.LayerNorm(d_model, eps=norm_eps, device=device, dtype=dtype)
        self.ffn_norm = nn.LayerNorm(d_model, eps=norm_eps, device=device, dtype=dtype)

    def forward(self, x, memory, *, padding_mask=None, memory_padding_mask=None):
        x = x + self.self_attention(self.self_norm(x), padding_mask=padding_mask, causal=True)
        x = x + self.cross_attention(self.cross_norm(x), memory, padding_mask=memory_padding_mask)
        return x + self.ffn_down(F.relu(self.ffn_up(self.ffn_norm(x))))


class LightFormerTransformer(nn.Module):
    """Dedicated reference graph; ``rank=None`` builds a dense source model.

    ``encoder_groups`` and ``decoder_groups`` enumerate actual applications.
    Repeated ids refer to the very same block, including its normalizations.
    Embeddings, final norms and output projection stay dense in this bounded
    profile; they are included in full unique-parameter resource accounting.
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, dim_feedforward,
                 encoder_groups, decoder_groups, *, rank=None, max_positions=128,
                 bias=True, norm_eps=1e-5, encoder_sources=None, decoder_sources=None,
                 device=None, dtype=None):
        super().__init__()
        _dimensions(src_vocab_size, tgt_vocab_size, d_model, num_heads, dim_feedforward, max_positions)
        if d_model % num_heads or type(bias) is not bool:
            raise ValueError("Declare divisible head dimensions and boolean bias")
        if (type(norm_eps) not in (float, int) or not math.isfinite(norm_eps) or norm_eps <= 0):
            raise ValueError("norm_eps must be finite and positive")
        if rank is not None and (type(rank) is not int or not 1 <= rank <= min(d_model, dim_feedforward)):
            raise ValueError("Compact rank must fit every projection")
        enc_count = validate_groups(encoder_groups, "encoder groups")
        dec_count = validate_groups(decoder_groups, "decoder groups")
        _reference_bound(src_vocab_size, tgt_vocab_size, d_model, dim_feedforward,
                         encoder_groups, decoder_groups, rank, bias, max_positions)
        if (device is not None and torch.device(device).type != "cpu") or (
                dtype is not None and dtype not in (torch.float32, torch.float64)):
            raise ValueError("LightFormer bounded reference profile requires CPU float32/float64")
        enc_sources = tuple(range(enc_count)) if encoder_sources is None else encoder_sources
        dec_sources = tuple(range(dec_count)) if decoder_sources is None else decoder_sources
        for sources, count in ((enc_sources, enc_count), (dec_sources, dec_count)):
            if (not isinstance(sources, tuple) or len(sources) != count
                    or any(type(i) is not int or i < 0 for i in sources)):
                raise ValueError("Source metadata requires one nonnegative source index per group")
        self.src_vocab_size, self.tgt_vocab_size = src_vocab_size, tgt_vocab_size
        self.d_model, self.num_heads, self.dim_feedforward = d_model, num_heads, dim_feedforward
        self.max_positions, self.rank, self.bias, self.norm_eps = max_positions, rank, bias, float(norm_eps)
        self.encoder_groups, self.decoder_groups = encoder_groups, decoder_groups
        self.encoder_sources, self.decoder_sources = enc_sources, dec_sources
        kwargs = dict(rank=rank, bias=bias, norm_eps=norm_eps, device=device, dtype=dtype)
        encoders = [LightFormerEncoderBlock(d_model, num_heads, dim_feedforward, **kwargs) for _ in range(enc_count)]
        decoders = [LightFormerDecoderBlock(d_model, num_heads, dim_feedforward, **kwargs) for _ in range(dec_count)]
        self.encoder_layers = nn.ModuleList([encoders[i] for i in encoder_groups])
        self.decoder_layers = nn.ModuleList([decoders[i] for i in decoder_groups])
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, device=device, dtype=dtype)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, device=device, dtype=dtype)
        self.encoder_norm = nn.LayerNorm(d_model, eps=norm_eps, device=device, dtype=dtype)
        self.decoder_norm = nn.LayerNorm(d_model, eps=norm_eps, device=device, dtype=dtype)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size, bias=bias, device=device, dtype=dtype)
        # Compute in float64 once; dtype conversion follows ordinary module rules.
        self.register_buffer("positions", _positions(max_positions, d_model, device=device,
                                                     dtype=self.src_embedding.weight.dtype))
        for name, values in (("_encoder_group_map", encoder_groups), ("_decoder_group_map", decoder_groups),
                             ("_encoder_source_map", enc_sources), ("_decoder_source_map", dec_sources)):
            self.register_buffer(name, torch.tensor(values, dtype=torch.long, device=device))

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        # Positions are mathematical constants; regenerate after dtype changes
        # rather than preserving rounding from a former float32 representation.
        if hasattr(self, "positions"):
            self.positions = _positions(self.max_positions, self.d_model,
                                        device=self.positions.device, dtype=self.positions.dtype)
        return result

    def _tokens(self, tokens, embedding, name):
        if (not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or not tokens.shape[0]
                or not 0 < tokens.shape[1] <= self.max_positions
                or tokens.dtype not in (torch.int32, torch.int64) or tokens.device != embedding.weight.device
                or bool((tokens < 0).any()) or bool((tokens >= embedding.num_embeddings).any())):
            raise ValueError(f"{name} requires bounded integer [batch,tokens] token ids")
        return embedding(tokens) * math.sqrt(self.d_model) + self.positions[:tokens.shape[1]]

    def encode(self, src_tokens, *, src_padding_mask=None):
        x = self._tokens(src_tokens, self.src_embedding, "Source")
        _padding(src_padding_mask, x.shape[0], x.shape[1], x.device)
        for layer in self.encoder_layers:
            x = layer(x, padding_mask=src_padding_mask)
        return self.encoder_norm(x)

    def decode_hidden(self, tgt_tokens, memory, *, tgt_padding_mask=None, src_padding_mask=None):
        x = self._tokens(tgt_tokens, self.tgt_embedding, "Target")
        _input(memory, self.d_model, self.src_embedding.weight, "Encoder memory")
        if x.shape[0] != memory.shape[0]:
            raise ValueError("Source/target batch sizes must agree")
        _padding(tgt_padding_mask, x.shape[0], x.shape[1], x.device)
        _padding(src_padding_mask, memory.shape[0], memory.shape[1], memory.device)
        for layer in self.decoder_layers:
            x = layer(x, memory, padding_mask=tgt_padding_mask, memory_padding_mask=src_padding_mask)
        return self.decoder_norm(x)

    def forward(self, src_tokens, tgt_tokens, *, src_padding_mask=None, tgt_padding_mask=None,
                cache=None, use_cache=False):
        if cache is not None or use_cache is not False:
            raise ValueError("LightFormer reference has no KV cache; weight sharing does not merge caches")
        memory = self.encode(src_tokens, src_padding_mask=src_padding_mask)
        hidden = self.decode_hidden(tgt_tokens, memory, tgt_padding_mask=tgt_padding_mask,
                                    src_padding_mask=src_padding_mask)
        return self.output_projection(hidden)

    def representation_config(self):
        return {"layer_type": "LightFormerTransformer", "version": 1,
                **{name: getattr(self, name) for name in ("src_vocab_size", "tgt_vocab_size", "d_model",
                   "num_heads", "dim_feedforward", "rank", "max_positions", "bias", "norm_eps")},
                **{name: list(getattr(self, name)) for name in ("encoder_groups", "decoder_groups",
                                                              "encoder_sources", "decoder_sources")}}

    @classmethod
    def from_config(cls, config, *, device=None, dtype=None):
        required = {"layer_type", "version", "src_vocab_size", "tgt_vocab_size", "d_model", "num_heads",
                    "dim_feedforward", "rank", "max_positions", "bias", "norm_eps", "encoder_groups",
                    "decoder_groups", "encoder_sources", "decoder_sources"}
        if (not isinstance(config, Mapping) or set(config) != required
                or config["layer_type"] != "LightFormerTransformer" or type(config["version"]) is not int
                or config["version"] != 1):
            raise ValueError("Exact version-1 LightFormerTransformer constructor config required")
        kwargs = {k: v for k, v in config.items() if k not in ("layer_type", "version")}
        for name in ("encoder_groups", "decoder_groups", "encoder_sources", "decoder_sources"):
            if not isinstance(kwargs[name], (tuple, list)):
                raise ValueError("LightFormer maps must be integer sequences")
            kwargs[name] = tuple(kwargs[name])
        with torch.random.fork_rng(devices=[]):
            return cls(**kwargs, device=device, dtype=dtype)

    def validate_checkpoint_state(self, state, prefix=""):
        if not isinstance(state, Mapping) or not isinstance(prefix, str):
            raise ValueError("LightFormer checkpoint requires a tensor state mapping")
        self._validate_topology()
        # Equal-valued clones are not complete group sharing.
        for stack in ("encoder", "decoder"):
            layers, groups = getattr(self, stack + "_layers"), getattr(self, stack + "_groups")
            for i, group in enumerate(groups):
                for j in range(i):
                    if (layers[i] is layers[j]) != (group == groups[j]):
                        raise ValueError("LightFormer complete block alias topology was changed")
        expected = self.state_dict()
        keys = {prefix + key for key in expected}
        if not keys.issubset(state) or (not prefix and set(state) != keys):
            raise ValueError("LightFormer checkpoint keys disagree with the declared topology")
        for key, value in expected.items():
            installed = state[prefix + key]
            if (not isinstance(installed, torch.Tensor) or installed.shape != value.shape
                    or installed.dtype != value.dtype or not bool(torch.isfinite(installed).all())):
                raise ValueError(f"LightFormer checkpoint tensor mismatch: {prefix + key}")
        fixed = {"_encoder_group_map": self.encoder_groups, "_decoder_group_map": self.decoder_groups,
                 "_encoder_source_map": self.encoder_sources, "_decoder_source_map": self.decoder_sources}
        for name, values in fixed.items():
            installed = state[prefix + name]
            if installed.tolist() != list(values):
                raise ValueError(f"LightFormer immutable map mismatch: {name}")
        positions = _positions(self.max_positions, self.d_model, device=self.positions.device,
                               dtype=self.positions.dtype)
        if not torch.equal(state[prefix + "positions"].cpu(), positions.cpu()):
            raise ValueError("LightFormer fixed sinusoidal position metadata mismatch")
        aliases = {}
        for name, parameter in self.named_parameters(remove_duplicate=False):
            old = aliases.setdefault(id(parameter), name)
            if not torch.equal(state[prefix + name], state[prefix + old]):
                raise ValueError(f"LightFormer shared parameter values disagree: {old}, {name}")

    def _validate_topology(self):
        """Check constructor semantics, including accidental cross-group ties."""
        _dimensions(self.src_vocab_size, self.tgt_vocab_size, self.d_model, self.num_heads,
                    self.dim_feedforward, self.max_positions)
        if self.d_model % self.num_heads or type(self.bias) is not bool:
            raise ValueError("LightFormer constructor dimensions/flags were changed")
        if type(self.norm_eps) not in (int, float) or not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("LightFormer norm_eps was changed")
        if self.rank is not None and (type(self.rank) is not int or not 1 <= self.rank <= min(self.d_model, self.dim_feedforward)):
            raise ValueError("LightFormer constructor rank was changed")
        validate_groups(self.encoder_groups, "encoder groups")
        validate_groups(self.decoder_groups, "decoder groups")
        _reference_bound(self.src_vocab_size, self.tgt_vocab_size, self.d_model, self.dim_feedforward,
                         self.encoder_groups, self.decoder_groups, self.rank, self.bias, self.max_positions)
        roots = {"encoder_layers", "decoder_layers", "src_embedding", "tgt_embedding",
                 "encoder_norm", "decoder_norm", "output_projection"}
        if set(self._modules) != roots or self._parameters:
            raise ValueError("LightFormer root topology was changed")
        if set(self._buffers) != {"positions", "_encoder_group_map", "_decoder_group_map",
                                  "_encoder_source_map", "_decoder_source_map"}:
            raise ValueError("LightFormer fixed buffer topology was changed")

        def linear(module, incoming, outgoing, *, factorized):
            if factorized:
                if (type(module) is not FactorizedLinear or module.in_features != incoming
                        or module.out_features != outgoing or module.rank != self.rank
                        or set(module._modules) != {"left", "right"} or module._parameters or module._buffers):
                    raise ValueError("LightFormer compact projection topology was changed")
                dense(module.right, incoming, self.rank, False)
                dense(module.left, self.rank, outgoing, self.bias)
            else:
                dense(module, incoming, outgoing, self.bias)

        def dense(module, incoming, outgoing, bias):
            if (type(module) is not nn.Linear or module.in_features != incoming or module.out_features != outgoing
                    or module.weight.shape != (outgoing, incoming) or (module.bias is not None) != bias
                    or (bias and module.bias.shape != (outgoing,)) or module._modules or module._buffers
                    or set(module._parameters) != {"weight", "bias"}):
                raise ValueError("LightFormer dense projection topology was changed")

        def norm(module):
            if (type(module) is not nn.LayerNorm or module.normalized_shape != (self.d_model,)
                    or module.eps != self.norm_eps or not module.elementwise_affine or module.bias is None
                    or module.weight.shape != (self.d_model,) or module.bias.shape != (self.d_model,)
                    or module._modules or module._buffers or set(module._parameters) != {"weight", "bias"}):
                raise ValueError("LightFormer normalization topology was changed")

        for name, vocab in (("src_embedding", self.src_vocab_size), ("tgt_embedding", self.tgt_vocab_size)):
            embedding = getattr(self, name)
            if (type(embedding) is not nn.Embedding or embedding.num_embeddings != vocab
                    or embedding.embedding_dim != self.d_model or embedding.weight.shape != (vocab, self.d_model)
                    or embedding.padding_idx is not None or embedding.max_norm is not None
                    or embedding.scale_grad_by_freq or embedding.sparse or embedding.norm_type != 2.0
                    or embedding._modules or embedding._buffers or set(embedding._parameters) != {"weight"}):
                raise ValueError("LightFormer embedding topology was changed")
        reference = self.src_embedding.weight
        if (self.positions.shape != (self.max_positions, self.d_model)
                or self.positions.dtype != reference.dtype or self.positions.device != reference.device):
            raise ValueError("LightFormer fixed positional buffer layout was changed")
        for name, values in (("_encoder_group_map", self.encoder_groups), ("_decoder_group_map", self.decoder_groups),
                             ("_encoder_source_map", self.encoder_sources), ("_decoder_source_map", self.decoder_sources)):
            buffer = getattr(self, name)
            if buffer.dtype != torch.long or buffer.shape != (len(values),) or buffer.device != reference.device:
                raise ValueError("LightFormer integer metadata buffer layout was changed")
        norm(self.encoder_norm)
        norm(self.decoder_norm)
        dense(self.output_projection, self.d_model, self.tgt_vocab_size, self.bias)
        for stack, block_type in (("encoder", LightFormerEncoderBlock), ("decoder", LightFormerDecoderBlock)):
            groups, sources = getattr(self, stack + "_groups"), getattr(self, stack + "_sources")
            count = validate_groups(groups, stack + " groups")
            if (not isinstance(sources, tuple) or len(sources) != count
                    or any(type(i) is not int or i < 0 for i in sources)):
                raise ValueError("LightFormer source provenance map was changed")
            layers = getattr(self, stack + "_layers")
            if type(layers) is not nn.ModuleList or len(layers) != len(groups) or layers._parameters or layers._buffers:
                raise ValueError("LightFormer application list topology was changed")
            names = {"self_attention", "ffn_up", "ffn_down", "self_norm", "ffn_norm"}
            if stack == "decoder":
                names |= {"cross_attention", "cross_norm"}
            for block in layers:
                if type(block) is not block_type or set(block._modules) != names or block._parameters or block._buffers:
                    raise ValueError("LightFormer complete block topology was changed")
                for name, module in block.named_children():
                    if name.endswith("attention"):
                        if (type(module) is not LightFormerAttention or module.d_model != self.d_model
                                or module.num_heads != self.num_heads or module.rank != self.rank or module.bias != self.bias
                                or set(module._modules) != {"q_proj", "k_proj", "v_proj", "o_proj"}
                                or module._parameters or module._buffers):
                            raise ValueError("LightFormer attention constructor semantics were changed")
                        for projection in module.children():
                            linear(projection, self.d_model, self.d_model, factorized=self.rank is not None)
                    elif name.endswith("norm"):
                        norm(module)
                    elif name == "ffn_up":
                        linear(module, self.d_model, self.dim_feedforward, factorized=self.rank is not None)
                    else:
                        linear(module, self.dim_feedforward, self.d_model, factorized=self.rank is not None)
        expected_ids, actual_ids, storage_owners = {}, {}, {}
        for name, parameter in self.named_parameters(remove_duplicate=False):
            parts = name.split(".")
            if parts[0] in ("encoder_layers", "decoder_layers"):
                stack = parts[0].split("_")[0]
                parts[1] = str(getattr(self, stack + "_groups")[int(parts[1])])
            canonical = ".".join(parts)
            if expected_ids.setdefault(canonical, id(parameter)) != id(parameter) or actual_ids.setdefault(id(parameter), canonical) != canonical:
                raise ValueError("LightFormer Parameter alias semantics disagree with complete group sharing")
            storage = (parameter.device, parameter.untyped_storage().data_ptr())
            if storage_owners.setdefault(storage, canonical) != canonical:
                raise ValueError("LightFormer independent Parameters cannot overlap one storage")
            if parameter.dtype != reference.dtype or parameter.device != reference.device:
                raise ValueError("LightFormer parameters require a uniform dtype/device")
        for module in self.modules():
            if (module._forward_hooks or module._forward_pre_hooks or module._backward_hooks
                    or "forward" in module.__dict__ or "__call__" in module.__dict__):
                raise ValueError("LightFormer hooks/custom forward overrides are unsupported")
