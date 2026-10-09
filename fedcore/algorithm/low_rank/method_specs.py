"""Immutable method descriptions for the explicit P2 CPU profiles.

Descriptions contain metadata only. Tensor observations, teachers, hooks and
optimizers belong to the interpreter. Paper heuristics and corrected formulas
are named rather than silently substituted for one another.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math


class MethodSpecError(ValueError):
    """An unsupported or ambiguous method description."""


def _nonnegative(value, name):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise MethodSpecError(f"{name} must be finite and nonnegative")


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise MethodSpecError(f"{name} must be a positive integer")


def _zero_policy(value):
    if value not in ("support_only", "preserve_nullspace", "error"):
        raise MethodSpecError("Declare support_only, preserve_nullspace or error for zero channels")


@dataclass(frozen=True)
class ASVD:
    abs_mode: str = "abs_mean"
    alpha: float = 0.5
    epsilon: float = 0.0
    zero_policy: str = "support_only"

    def __post_init__(self):
        if self.abs_mode not in ("abs_mean", "abs_max"):
            raise MethodSpecError("ASVD abs_mode must be abs_mean or abs_max")
        _nonnegative(self.alpha, "alpha")
        _nonnegative(self.epsilon, "epsilon")
        _zero_policy(self.zero_policy)


@dataclass(frozen=True)
class FWSVD:
    loss: str = "cross_entropy"
    reduction: str = "mean"
    epsilon: float = 0.0
    zero_policy: str = "support_only"
    max_examples: int = 4096

    def __post_init__(self):
        if self.loss not in ("cross_entropy", "mse") or self.reduction not in ("mean", "sum"):
            raise MethodSpecError("FWSVD requires a declared observed-label loss and reduction")
        _nonnegative(self.epsilon, "epsilon")
        _zero_policy(self.zero_policy)
        _positive_integer(self.max_examples, "max_examples")


@dataclass(frozen=True)
class AFM:
    """Affine PCA of centered outputs. GFM is a separate recovery decision."""


@dataclass(frozen=True)
class Bolaco:
    """Within-group covariance with observation or equal-group weighting."""
    group_weighting: str = "observations"

    def __post_init__(self):
        if self.group_weighting not in ("observations", "equal_groups"):
            raise MethodSpecError("Bolaco requires an explicit group weighting")


@dataclass(frozen=True)
class FLARSVD:
    estimator_version: str = "ledoit_wolf_corrected_v1"

    def __post_init__(self):
        if self.estimator_version != "ledoit_wolf_corrected_v1":
            raise MethodSpecError("Only the independently checked shrinkage formula is supported")


@dataclass(frozen=True)
class DRONE:
    rcond: float | None = None
    loss_growth_tolerance: float = 0.0

    def __post_init__(self):
        if self.rcond is not None:
            _nonnegative(self.rcond, "rcond")
        _nonnegative(self.loss_growth_tolerance, "loss_growth_tolerance")


@dataclass(frozen=True)
class SVDLLMV1:
    rcond: float | None = None
    ridge: float = 0.0

    def __post_init__(self):
        if self.rcond is not None:
            _nonnegative(self.rcond, "rcond")
        _nonnegative(self.ridge, "ridge")


@dataclass(frozen=True)
class SVDLLMV2:
    formula_version: str = "corrected_support_factors_v1"

    def __post_init__(self):
        if self.formula_version != "corrected_support_factors_v1":
            raise MethodSpecError("Literal erroneous printed V2 factors are not executable")


@dataclass(frozen=True)
class EoRA:
    base_kind: str = "dense"
    gram_update_version: str = "psd_second_moment_v1"

    def __post_init__(self):
        if self.base_kind != "dense":
            raise MethodSpecError("Packed or sparse EoRA bases require a separately validated runtime")
        if self.gram_update_version not in ("psd_second_moment_v1", "author_fixed_n_v1"):
            raise MethodSpecError("Declare the supported EoRA Gram interpretation")


@dataclass(frozen=True)
class MixedRank:
    residual_rank: int = 1
    metric: str = "normalized_output"
    zero_policy: str = "error"
    seed: int = 0

    def __post_init__(self):
        _positive_integer(self.residual_rank, "residual_rank")
        if self.metric not in ("normalized_output", "author_mean_input"):
            raise MethodSpecError("Mixed-rank author heuristic and normalized objective are distinct")
        if self.zero_policy not in ("error", "skip"):
            raise MethodSpecError("Mixed-rank zero denominators require error or skip")
        if type(self.seed) is not int or self.seed < 0:
            raise MethodSpecError("seed must be a nonnegative integer")


@dataclass(frozen=True)
class BasisSharing:
    metric_mode: str = "pooled_surrogate"
    proportional_scales: tuple[float, ...] = ()

    def __post_init__(self):
        if self.metric_mode not in ("pooled_surrogate", "equal_moments", "proportional_moments"):
            raise MethodSpecError("Declare the relation between layer input moments")
        if not isinstance(self.proportional_scales, tuple):
            raise MethodSpecError("proportional_scales must be an immutable tuple")
        for value in self.proportional_scales:
            _nonnegative(value, "proportional_scale")
            if value == 0:
                raise MethodSpecError("Proportional metric scales must be positive")
        if (self.metric_mode == "proportional_moments") != bool(self.proportional_scales):
            raise MethodSpecError("Proportional mode requires scales; other modes must not set scales")


@dataclass(frozen=True)
class GroupReduce:
    groups: tuple[tuple[int, ...], ...]
    ranks: tuple[int, ...]
    zero_policy: str = "project"
    transfer_steps: int = 0

    def __post_init__(self):
        if (not isinstance(self.groups, tuple) or not self.groups
                or any(not isinstance(group, tuple) or not group for group in self.groups)):
            raise MethodSpecError("GroupReduce requires immutable nonempty token groups")
        if not isinstance(self.ranks, tuple) or len(self.ranks) != len(self.groups):
            raise MethodSpecError("One rank is required per token group")
        flat = tuple(token for group in self.groups for token in group)
        if any(type(token) is not int or token < 0 for token in flat) or len(flat) != len(set(flat)):
            raise MethodSpecError("Token IDs must be nonnegative and belong to exactly one group")
        for rank in self.ranks:
            _positive_integer(rank, "group rank")
        if self.zero_policy not in ("project", "error"):
            raise MethodSpecError("GroupReduce zero-frequency rows require project or error")
        if type(self.transfer_steps) is not int or self.transfer_steps < 0:
            raise MethodSpecError("transfer_steps must be a nonnegative integer")


@dataclass(frozen=True)
class SVDLLMV5:
    adapter_rank: int = 1
    adapter_alpha: float = 1.0
    left_epochs: int = 1
    right_epochs: int = 1
    merge: bool = True

    def __post_init__(self):
        _positive_integer(self.adapter_rank, "adapter_rank")
        _nonnegative(self.adapter_alpha, "adapter_alpha")
        if self.adapter_alpha == 0:
            raise MethodSpecError("adapter_alpha must be positive")
        for name, value in (("left_epochs", self.left_epochs), ("right_epochs", self.right_epochs)):
            if type(value) is not int or value < 0:
                raise MethodSpecError(f"{name} must be a nonnegative integer")
        if type(self.merge) is not bool or not self.merge:
            raise MethodSpecError("The first v5 public profile exports merged factor adapters")


MethodSpec = ASVD | FWSVD | AFM | Bolaco | FLARSVD | DRONE | SVDLLMV1 | SVDLLMV2 | EoRA | MixedRank | BasisSharing | GroupReduce | SVDLLMV5


_METHOD_TYPES = {
    "asvd": ASVD, "fwsvd": FWSVD, "afm": AFM, "bolaco": Bolaco,
    "flar_svd": FLARSVD, "drone": DRONE, "svdllm_v1": SVDLLMV1,
    "svdllm_v2": SVDLLMV2, "eora": EoRA, "mixed_rank": MixedRank,
    "basis_sharing": BasisSharing, "groupreduce": GroupReduce, "svdllm_v5": SVDLLMV5,
}


def method_name(spec):
    for name, cls in _METHOD_TYPES.items():
        if type(spec) is cls:
            return name
    raise MethodSpecError("A validated explicit P2 method description is required")


def method_payload(spec):
    return {"method": method_name(spec), "version": 1, "options": asdict(spec)}


def parse_method(method, options=None, *, version=1):
    if type(version) is not int or version != 1 or method not in _METHOD_TYPES:
        raise MethodSpecError("Unsupported method or method version")
    cls = _METHOD_TYPES[method]
    if options is None:
        options = {}
    if not isinstance(options, dict) or set(options) - {field.name for field in fields(cls)}:
        raise MethodSpecError("Unknown method option or invalid options object")
    options = dict(options)
    if cls is BasisSharing and "proportional_scales" in options:
        if not isinstance(options["proportional_scales"], (tuple, list)):
            raise MethodSpecError("Expected proportional scale sequence")
        options["proportional_scales"] = tuple(options["proportional_scales"])
    if cls is GroupReduce:
        if not isinstance(options.get("groups"), (tuple, list)) or not isinstance(options.get("ranks"), (tuple, list)):
            raise MethodSpecError("Expected group and rank sequences")
        if any(not isinstance(group, (tuple, list)) for group in options["groups"]):
            raise MethodSpecError("Expected token groups")
        options["groups"] = tuple(tuple(group) for group in options["groups"])
        options["ranks"] = tuple(options["ranks"])
    try:
        return cls(**options)
    except TypeError as error:
        raise MethodSpecError(f"Missing or invalid {method} options") from error
