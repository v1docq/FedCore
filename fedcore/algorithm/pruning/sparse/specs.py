"""Immutable contracts for the bounded sparse/hybrid CPU profiles.

Plans contain values and storage descriptions, never modules or live resources.
All weights use PyTorch [out, in]; N:M groups run along the input axis.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from fedcore.algorithm.low_rank.allocation import StorageCost, exact_unique_cost


class SparseDomainError(ValueError):
    """A predictable profile refusal with a machine-readable reason."""

    def __init__(self, code: str, message: str, details=()):
        self.code = code
        self.details = tuple(details)
        super().__init__(message)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise SparseDomainError("invalid_specification", f"{name} must be an integer >= {minimum}")


def _number(value, name, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise SparseDomainError("invalid_specification", f"{name} must be finite and {'positive' if positive else 'nonnegative'}")


def _budget(value):
    if value is not None:
        _integer(value, "budget_bytes")


@dataclass(frozen=True)
class NMStructure:
    n: int = 2
    m: int = 4
    rank: int = 0
    axis: str = "in_features"
    tail_policy: str = "reject"
    kernel: str = "portable_cpu"

    def __post_init__(self):
        _integer(self.n, "n", 1)
        _integer(self.m, "m", 1)
        _integer(self.rank, "rank")
        if self.n > self.m:
            raise SparseDomainError("invalid_specification", "n cannot exceed m")
        if self.axis != "in_features":
            raise SparseDomainError("unsupported_axis", "N:M groups require the PyTorch input axis")
        if self.tail_policy != "reject":
            raise SparseDomainError("unsupported_tail_policy", "Incomplete N:M groups are rejected")
        if self.kernel != "portable_cpu":
            raise SparseDomainError("unsupported_kernel", "Only the declared portable CPU kernel is admitted")


@dataclass(frozen=True)
class HASSLEFreeSpec:
    structure: NMStructure = NMStructure()
    outer_steps: int = 3
    inner_steps: int = 20
    learning_rate: float = 0.01
    damping: float = 0.01
    damping_rule: str = "mean_diagonal"
    normalization: str = "mean"
    coordinate_scaling: bool = True
    seed: int = 0
    max_input_features: int = 512
    budget_bytes: int | None = None

    def __post_init__(self):
        if not isinstance(self.structure, NMStructure):
            raise SparseDomainError("invalid_specification", "NMStructure required")
        _integer(self.outer_steps, "outer_steps", 1)
        _integer(self.inner_steps, "inner_steps")
        _number(self.learning_rate, "learning_rate", positive=True)
        _number(self.damping, "damping")
        _integer(self.seed, "seed")
        if self.seed >= 2**63:
            raise SparseDomainError("invalid_specification", "seed must be below 2**63")
        _integer(self.max_input_features, "max_input_features", 1)
        _budget(self.budget_bytes)
        if self.damping_rule not in ("mean_diagonal", "trace"):
            raise SparseDomainError("invalid_specification", "Declare mean_diagonal or trace damping")
        if self.normalization not in ("mean", "sum"):
            raise SparseDomainError("invalid_specification", "Declare mean or sum Gram normalization")
        if type(self.coordinate_scaling) is not bool:
            raise SparseDomainError("invalid_specification", "coordinate_scaling must be bool")


@dataclass(frozen=True)
class LoSparseSpec:
    rank: int
    residual_rows: int
    total_steps: int
    initial_scale: float = 1.0
    warmup_steps: int = 0
    final_step: int | None = None
    delta_t: int = 1
    ema_beta: float = 0.85
    allow_reentry: bool = False
    budget_bytes: int | None = None

    def __post_init__(self):
        _integer(self.rank, "rank")
        _integer(self.residual_rows, "residual_rows")
        _integer(self.total_steps, "total_steps", 1)
        _number(self.initial_scale, "initial_scale", positive=True)
        _integer(self.warmup_steps, "warmup_steps")
        _integer(self.delta_t, "delta_t", 1)
        _number(self.ema_beta, "ema_beta")
        _budget(self.budget_bytes)
        end = self.total_steps if self.final_step is None else self.final_step
        _integer(end, "final_step", 1)
        if not self.warmup_steps < end <= self.total_steps or self.ema_beta >= 1:
            raise SparseDomainError("invalid_specification", "Require warmup_steps < final_step <= total_steps and ema_beta < 1")
        if type(self.allow_reentry) is not bool:
            raise SparseDomainError("invalid_specification", "allow_reentry must be bool")


@dataclass(frozen=True)
class SparseLinearPlan:
    method: str
    in_features: int
    out_features: int
    rank: int
    dtype: str
    device: str
    bias: bool
    storages: tuple[StorageCost, ...]
    numerical_elements: int
    metadata_bytes: int
    artifact_bytes: int
    training_state: str
    kernel: str = "portable_cpu"
    orientation: str = "out_in"


def reject_method(method):
    if not isinstance(method, str):
        raise SparseDomainError("invalid_specification", "Method must be a string identifier")
    if method.lower().replace("-", "_") == "tedepr":
        raise SparseDomainError("insufficient_specification", "TEDEPR requires the missing full primary specification")


def plan_linear(in_features, out_features, spec, *, dtype="float32", device="cpu", bias=True, storage_prefix="linear"):
    """Validate dimensions/cost before cloning tensors, collecting data or RNG use."""
    _integer(in_features, "in_features", 1)
    _integer(out_features, "out_features", 1)
    if dtype not in ("float32", "float64") or device != "cpu":
        raise SparseDomainError("unsupported_device_dtype", "This profile admits CPU float32/float64 only")
    if type(bias) is not bool or not isinstance(storage_prefix, str) or not storage_prefix:
        raise SparseDomainError("invalid_specification", "Declare bias and stable storage prefix")
    precision = 4 if dtype == "float32" else 8
    if isinstance(spec, NMStructure):
        structure, method, budget = spec, "packed_nm", None
    elif isinstance(spec, HASSLEFreeSpec):
        structure, method, budget = spec.structure, "hassle_free", spec.budget_bytes
        if in_features > spec.max_input_features:
            raise SparseDomainError("unsupported_shape", "HASSLE-free exceeds its bounded full-Gram CPU profile")
    elif isinstance(spec, LoSparseSpec):
        structure, method, budget = None, "losparse", spec.budget_bytes
    else:
        raise SparseDomainError("invalid_specification", "A supported immutable sparse specification is required")
    rank = structure.rank if structure is not None else spec.rank
    if rank > min(in_features, out_features):
        raise SparseDomainError("invalid_rank", "Rank must fit both weight axes; zero means no factor branch")
    entries = [StorageCost(storage_prefix + ".left", out_features * rank, precision),
               StorageCost(storage_prefix + ".right", rank * in_features, precision)]
    if structure is not None:
        if in_features % structure.m:
            raise SparseDomainError("incomplete_nm_group", "Input width must be divisible by m")
        sparse_count = out_features * (in_features // structure.m) * structure.n
        entries += [StorageCost(storage_prefix + ".values", sparse_count, precision, "sparse_values"),
                    StorageCost(storage_prefix + ".indices", sparse_count, 8, "index")]
        training = "local_full_gram" if method == "hassle_free" else "inference"
    else:
        if spec.residual_rows > out_features:
            raise SparseDomainError("invalid_support", "Residual row count exceeds the full output width")
        entries += [StorageCost(storage_prefix + ".residual_rows", spec.residual_rows * in_features, precision, "residual"),
                    StorageCost(storage_prefix + ".row_indices", spec.residual_rows, 8, "index")]
        training = "dense_residual_and_dense_ema_until_finalize"
    if bias:
        entries.append(StorageCost(storage_prefix + ".bias", out_features, precision, "bias"))
    storages = tuple(entries)
    cost = exact_unique_cost(storages, "bytes")
    if budget is not None and cost > budget:
        raise SparseDomainError("budget_infeasible", "Final representation exceeds budget_bytes", (cost, budget))
    metadata = sum(s.numel * s.bytes_per_element for s in storages if s.category == "index")
    numbers = sum(s.numel for s in storages if s.category != "index")
    return SparseLinearPlan(method, in_features, out_features, rank, dtype, device, bias,
                            storages, numbers, metadata, cost, training)
