"""Metadata-only description of the bounded ARS structure search."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .method_errors import MethodSpecError


@dataclass(frozen=True)
class ARS:
    """Explicit ARS choices; the caller owns data, optimizer and update loop.

    ``retained_ratio`` targets whole-model parameter elements, including fixed
    parameters and biases. It is a soft loss target, not a feasibility guarantee.
    The paper freezes weights and exports a prefix; the default index-preserving
    export deliberately avoids that operator change. Both decisions are recorded.
    """

    updates: int = 1
    seed: int = 0
    loss: str = "mse"
    reduction: str = "mean"
    retained_ratio: float = 0.5
    budget_weight: float = 1.0
    alignment_weight: float = 1.0
    temperature_start: float = 0.4
    temperature_end: float = 0.4
    noise: str = "paper_single_gumbel"
    mask_bias: float = 3.0
    structural_trainable: bool = True
    weight_trainable: bool = False
    input_size: int = 32
    hidden_size: int = 64
    export_policy: str = "selected_indices"

    def __post_init__(self):
        for name in ("updates", "input_size", "hidden_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise MethodSpecError(f"ARS {name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise MethodSpecError("ARS seed must be an integer in [0, 2**63)")
        if self.loss not in ("mse", "cross_entropy") or self.reduction not in ("mean", "sum"):
            raise MethodSpecError("ARS requires an explicit observed-label loss and reduction")
        for name in ("retained_ratio", "budget_weight", "alignment_weight",
                     "temperature_start", "temperature_end", "mask_bias"):
            value = getattr(self, name)
            try:
                valid = type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                valid = False
            if not valid:
                raise MethodSpecError(f"ARS {name} must be finite")
        if not 0 < self.retained_ratio <= 1:
            raise MethodSpecError("ARS retained_ratio must be in (0, 1]")
        if min(self.budget_weight, self.alignment_weight) < 0:
            raise MethodSpecError("ARS loss weights must be nonnegative")
        if min(self.temperature_start, self.temperature_end) <= 0:
            raise MethodSpecError("ARS temperatures must be positive")
        if self.noise not in ("paper_single_gumbel", "binary_gumbel_difference"):
            raise MethodSpecError("Declare the paper's single Gumbel or binary Gumbel difference")
        if self.export_policy not in ("selected_indices", "paper_prefix"):
            raise MethodSpecError("Declare selected_indices or the operator-changing paper_prefix export")
        if type(self.structural_trainable) is not bool or type(self.weight_trainable) is not bool:
            raise MethodSpecError("ARS structural/weight trainability must be explicit booleans")
        if not (self.structural_trainable or self.weight_trainable):
            raise MethodSpecError("ARS search requires a trainable structure or selected weight factors")
