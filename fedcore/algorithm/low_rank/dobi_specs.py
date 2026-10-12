"""Pure options for the bounded Dobi-SVD reconstruction; no tensor runtime."""
from dataclasses import dataclass
import math

from .method_errors import MethodSpecError


def _finite(value, name, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise MethodSpecError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")


@dataclass(frozen=True)
class DobiSVD:
    """Learn continuous activation cutoffs, then export fixed hard projectors.

    ``initial_fraction`` is a fraction of the maximum matrix rank.
    ``target_fraction`` is the soft two-factor weight-element budget, excluding
    unchanged biases. It is a penalty target, not an export-size guarantee.
    Near-spectrum clustering and uncentered direction IPCA are explicit
    FedCore reconstructions, not the paper's Taylor-backward/remapping code.
    """
    updates: int = 1
    seed: int = 0
    loss: str = "mse"
    reduction: str = "mean"
    target_fraction: float = 0.5
    initial_fraction: float = 0.5
    beta: float = 5.0
    budget_weight: float = 1.0
    spectrum_policy: str = "cluster_equal"
    spectrum_rtol: float = 1e-6
    spectrum_atol: float = 1e-12
    export_rounding: str = "floor"
    max_activation_rows: int = 4096
    max_calibration_batches: int = 64
    variant: str = "activation_tanh_ipca_v1"
    remapping: str = "none"

    def __post_init__(self):
        for name in ("updates", "max_activation_rows", "max_calibration_batches"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise MethodSpecError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise MethodSpecError("seed must be an integer in [0, 2**63)")
        if self.loss not in ("mse", "cross_entropy") or self.reduction not in ("mean", "sum"):
            raise MethodSpecError("Declare mse/cross_entropy and mean/sum reduction")
        for name in ("target_fraction", "initial_fraction"):
            _finite(getattr(self, name), name)
            if getattr(self, name) > 1:
                raise MethodSpecError(f"{name} must be in [0, 1]")
        _finite(self.beta, "beta", positive=True)
        for name in ("budget_weight", "spectrum_rtol", "spectrum_atol"):
            _finite(getattr(self, name), name)
        if self.spectrum_policy not in ("error", "cluster_equal"):
            raise MethodSpecError("Declare error or cluster_equal spectrum policy")
        if self.export_rounding not in ("floor", "ceil", "nearest"):
            raise MethodSpecError("Declare floor, ceil or nearest export rounding")
        if self.variant != "activation_tanh_ipca_v1":
            raise MethodSpecError("Unsupported Dobi-SVD variant")
        if self.remapping != "none":
            raise MethodSpecError("Mixed-precision remapping is a separate deferred representation")
