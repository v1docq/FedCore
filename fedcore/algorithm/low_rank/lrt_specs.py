"""Metadata for a bounded, from-scratch Low-Rank Transformer reconstruction.

The ranks are internal LED coordinates. They do not change head widths,
sequence attention matrix sizes or the dimensions of a hypothetical KV cache.
This token model is not the paper's speech-recognition system.
"""
from dataclasses import asdict, dataclass, fields

from .method_errors import MethodSpecError
from .plans import Goal


def _integer(value, name, minimum=1, maximum=None):
    if (type(value) is not int or value < minimum
            or (maximum is not None and value > maximum)):
        bound = f" <= {maximum}" if maximum is not None else ""
        raise MethodSpecError(f"{name} must be an integer >= {minimum}{bound}")


@dataclass(frozen=True)
class LRT:
    """Explicit architecture, initialization and seed; no source weights.

    Q/K/V use independent factors for each head, following paper eq. (3).
    The attention output and each of the two FFN projections have their own
    LED pair. Embeddings, output vocabulary projection and norms stay dense.
    Bounds deliberately describe small CPU token-model tests, not ASR scale.
    """
    vocab_size: int = 32
    d_model: int = 16
    num_heads: int = 2
    d_k: int = 8
    d_v: int = 8
    d_ff: int = 32
    num_layers: int = 1
    max_seq_len: int = 32
    q_rank: int = 2
    k_rank: int = 2
    v_rank: int = 2
    out_rank: int = 4
    ffn_in_rank: int = 4
    ffn_out_rank: int = 4
    bias: bool = True
    initialization: str = "xavier_uniform"
    seed: int = 0
    max_parameters: int = 1_000_000
    max_tensor_bytes: int = 16_000_000
    max_peak_bytes: int = 64_000_000

    def __post_init__(self):
        bounds = {"vocab_size": 8192, "d_model": 1024, "num_heads": 16,
                  "d_k": 1024, "d_v": 1024, "d_ff": 1024,
                  "num_layers": 8, "max_seq_len": 512}
        for name, maximum in bounds.items():
            _integer(getattr(self, name), name, maximum=maximum)
        for name, maximum in (
                ("q_rank", min(self.d_model, self.d_k)),
                ("k_rank", min(self.d_model, self.d_k)),
                ("v_rank", min(self.d_model, self.d_v)),
                ("out_rank", min(self.num_heads*self.d_v, self.d_model)),
                ("ffn_in_rank", min(self.d_model, self.d_ff)),
                ("ffn_out_rank", min(self.d_model, self.d_ff))):
            _integer(getattr(self, name), name, maximum=maximum)
        if type(self.bias) is not bool:
            raise MethodSpecError("LRT bias must be an explicit boolean")
        if type(self.initialization) is not str or self.initialization not in ("xavier_uniform", "xavier_normal"):
            raise MethodSpecError("Declare xavier_uniform or xavier_normal initialization")
        _integer(self.seed, "seed", minimum=0, maximum=2**63-1)
        _integer(self.max_parameters, "max_parameters", maximum=10_000_000)
        _integer(self.max_tensor_bytes, "max_tensor_bytes", maximum=256_000_000)
        _integer(self.max_peak_bytes, "max_peak_bytes", maximum=1_024_000_000)
        if lrt_parameter_counts(self)["factorized"] > self.max_parameters:
            raise MethodSpecError("LRT whole-model parameters exceed max_parameters")


def lrt_spec_from_config(options):
    """Strict JSON-safe description parsing before constructing any tensors."""
    names = {field.name for field in fields(LRT)}
    if type(options) is not dict or set(options) != names:
        raise MethodSpecError("LRT requires the complete, exact architecture description")
    return LRT(**options)


def lrt_parameter_counts(spec):
    """Exact whole-model counts including dense vocabulary, biases and norms."""
    if type(spec) is not LRT:
        raise MethodSpecError("A validated LRT description is required")
    m, h, k, v, f = spec.d_model, spec.num_heads, spec.d_k, spec.d_v, spec.d_ff
    common = 2*spec.vocab_size*m + (spec.vocab_size if spec.bias else 0)
    per_layer_bias = h*(2*k+v)+m+f+m if spec.bias else 0
    per_layer_common = per_layer_bias+4*m  # two affine LayerNorm modules
    factor_weights = (h*((m+k)*(spec.q_rank+spec.k_rank)+(m+v)*spec.v_rank)
                      +(h*v+m)*spec.out_rank+(m+f)*(spec.ffn_in_rank+spec.ffn_out_rank))
    dense_weights = h*m*(2*k+v)+h*v*m+2*m*f
    return {"factorized": common+spec.num_layers*(factor_weights+per_layer_common),
            "dense": common+spec.num_layers*(dense_weights+per_layer_common),
            "common_parameters": common+spec.num_layers*per_layer_common,
            "factorized_projection_weights_per_layer": factor_weights,
            "dense_projection_weights_per_layer": dense_weights}


def lrt_metadata_values(spec, factorized):
    """Fixed integer checkpoint passport; the full spec is in its descriptor."""
    return (1, int(factorized), *(getattr(spec, name) for name in (
        "vocab_size", "d_model", "num_heads", "d_k", "d_v", "d_ff",
        "num_layers", "max_seq_len", "q_rank", "k_rank", "v_rank", "out_rank",
        "ffn_in_rank", "ffn_out_rank")), int(spec.bias),
        int(spec.initialization == "xavier_normal"), spec.seed,
        spec.max_parameters, spec.max_tensor_bytes, spec.max_peak_bytes)


@dataclass(frozen=True)
class LRTBuildPlan:
    """No source checkpoint, observations, solver or live tensors are needed."""
    spec: LRT
    factorized: bool
    dtype: str
    parameter_count: int
    tensor_bytes: int
    dense_parameter_count: int
    dense_tensor_bytes: int
    build_peak_bytes_reserve: int
    goal: Goal = Goal.BUILD_ARCHITECTURE
    architecture: str = "causal_token_transformer_v1"
    schema_version: int = 1

    def config(self):
        return {"layer_type": "LRTTransformer", "version": 1,
                "factorized": self.factorized, "spec": asdict(self.spec)}
