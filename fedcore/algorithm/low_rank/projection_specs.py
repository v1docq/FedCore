"""Immutable descriptions of the bounded P3 projection profiles."""
from dataclasses import dataclass
import math

from .method_errors import MethodSpecError


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise MethodSpecError(f"{name} must be an integer >= {minimum}")


def _number(value, name, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise MethodSpecError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")


@dataclass(frozen=True)
class ESPACE:
    criterion: str = "mse"
    zero_policy: str = "zero_contribution"
    loss: str = "mse"
    reduction: str = "mean"
    max_examples: int = 4096
    max_batches: int = 256
    updates: int = 0

    def __post_init__(self):
        if self.criterion not in ("mse", "nmse", "go_mse_bound", "go_normalized", "nl_first_order_bound", "nl_normalized"):
            raise MethodSpecError("Choose one of the six explicit ESPACE criteria")
        if self.zero_policy not in ("zero_contribution", "error"):
            raise MethodSpecError("Normalized zero vectors require zero_contribution or error")
        if self.loss not in ("mse", "cross_entropy") or self.reduction not in ("mean", "sum"):
            raise MethodSpecError("Declare observed-label loss and within-example reduction")
        _integer(self.max_examples, "max_examples")
        _integer(self.max_batches, "max_batches")
        _integer(self.updates, "updates", 0)


@dataclass(frozen=True)
class PELA:
    kappa: float = 2.0
    block_map: tuple[tuple[str, str], ...] = ()
    feature_normalization: str = "paper_sum_squared"
    infinity_norm: str = "max_element"
    regularizer: str = "published_subtract_epsilon"
    epsilon: float = 0.0
    feature_weight: float = 1.0
    regularizer_weight: float = 1.0
    updates: int = 1
    freeze: str = "factors_only"
    loss: str = "mse"
    reduction: str = "mean"

    def __post_init__(self):
        _number(self.kappa, "kappa", True)
        if (not isinstance(self.block_map, tuple) or not self.block_map
                or any(not isinstance(pair, tuple) or len(pair) != 2 or any(not isinstance(p, str) for p in pair) for pair in self.block_map)
                or len(set(s for s, _ in self.block_map)) != len(self.block_map)
                or len(set(t for _, t in self.block_map)) != len(self.block_map)):
            raise MethodSpecError("PELA requires a one-to-one immutable student/teacher block map")
        if self.feature_normalization not in ("paper_sum_squared", "per_element_mean"):
            raise MethodSpecError("Declare PELA feature normalization")
        if self.infinity_norm not in ("max_element", "max_row_sum"):
            raise MethodSpecError("Declare the infinity norm interpretation")
        if self.regularizer not in ("published_subtract_epsilon", "hinge_variant"):
            raise MethodSpecError("Published subtraction and hinge are distinct regularizers")
        if self.freeze not in ("factors_only", "all_student"):
            raise MethodSpecError("Declare factors_only or all_student training")
        if self.loss not in ("mse", "cross_entropy") or self.reduction not in ("mean", "sum"):
            raise MethodSpecError("Declare PELA's base observed-label loss")
        for name in ("epsilon", "feature_weight", "regularizer_weight"):
            _number(getattr(self, name), name)
        _integer(self.updates, "updates")


@dataclass(frozen=True)
class LASER:
    rho: float = 0.5

    def __post_init__(self):
        _number(self.rho, "rho")
        if self.rho >= 1:
            raise MethodSpecError("LASER rho is the retained rank fraction in [0,1)")


@dataclass(frozen=True)
class CTRAFM:
    field_ranks: tuple[tuple[str, int], ...]
    weighting: str = "uniform_rows"
    interaction_policy: str = "retain_decoder"
    nonlinear_variant: str = "linear_v1"
    updates: int = 0

    def __post_init__(self):
        if (not isinstance(self.field_ranks, tuple) or not self.field_ranks
                or any(not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str) or not item[0] for item in self.field_ranks)
                or len({name for name, _ in self.field_ranks}) != len(self.field_ranks)):
            raise MethodSpecError("CTRAFM requires immutable unique field/rank pairs")
        for _, rank in self.field_ranks:
            _integer(rank, "field rank")
        if self.weighting not in ("uniform_rows", "frequency"):
            raise MethodSpecError("Declare uniform vocabulary or frequency-weighted PCA")
        if self.interaction_policy not in ("retain_decoder", "refuse"):
            raise MethodSpecError("Interactions require a retained decoder or explicit refusal")
        if self.nonlinear_variant not in ("linear_v1", "relu_between_factors_v1"):
            raise MethodSpecError("Unknown CTR architecture variant")
        _integer(self.updates, "updates", 0)
        if self.nonlinear_variant != "linear_v1" and self.updates == 0:
            raise MethodSpecError("The nonlinear CTR variant requires explicit recovery updates")


@dataclass(frozen=True)
class CTRMLPAFM:
    """Centered affine MLP PCA and the separate author ReLU architecture."""
    nonlinear_variant: str = "linear_v1"
    updates: int = 0
    loss: str = "mse"
    reduction: str = "mean"

    def __post_init__(self):
        if self.nonlinear_variant not in ("linear_v1", "relu_between_factors_v1"):
            raise MethodSpecError("Unknown CTR MLP architecture variant")
        _integer(self.updates,"updates",0)
        if self.nonlinear_variant != "linear_v1" and self.updates == 0:
            raise MethodSpecError("The nonlinear CTR MLP architecture requires recovery updates")
        if self.loss not in ("mse","cross_entropy") or self.reduction not in ("mean","sum"):
            raise MethodSpecError("Declare the CTR MLP observed-label loss and reduction")
