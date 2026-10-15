"""Explicit bounded SliceGPT architecture contract, separate from Linear SVD."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .method_errors import MethodSpecError


@dataclass(frozen=True)
class SliceGPT:
    """Sequential uncentered residual PCA for the dedicated RMS graph.

    ``widths`` describes embedding and every attention/MLP output boundary.
    Empty widths request an exact full-width rotation. No training, arbitrary
    Hugging Face graph adaptation, LayerNorm conversion or KV pruning occurs.
    """
    widths: tuple[int, ...] = ()
    variant: str = "sequential_residual_second_moment_v1"
    max_samples: int = 128
    max_tokens: int = 16384
    max_peak_bytes: int = 268435456
    verification_rtol: float | None = None
    verification_atol: float | None = None

    def __post_init__(self):
        if (not isinstance(self.widths, tuple)
                or any(type(v) is not int or v < 1 for v in self.widths)):
            raise MethodSpecError("SliceGPT widths require an immutable tuple of positive integers")
        if self.variant != "sequential_residual_second_moment_v1":
            raise MethodSpecError("Only the declared sequential residual second-moment variant is supported")
        for name in ("max_samples", "max_tokens", "max_peak_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise MethodSpecError(f"SliceGPT {name} must be a positive integer")
        for name in ("verification_rtol", "verification_atol"):
            value = getattr(self, name)
            if value is not None and (type(value) not in (int, float)
                                      or not math.isfinite(value) or value < 0):
                raise MethodSpecError(f"SliceGPT {name} must be finite and nonnegative")


@dataclass(frozen=True)
class SliceGPTPlan:
    architecture: str
    original_width: int
    widths: tuple[int, ...]
    boundaries: tuple[str, ...]
    source_fingerprint: str
    calibration_fingerprint: str | None
    calibration_samples: int
    calibration_tokens: int
    estimated_unique_parameters: int
    estimated_tensor_bytes: int
    estimated_floor_bytes: int
    max_peak_bytes: int
    exact_rotation_required: bool = True
    normalization: str = "sum_squared_over_original_width_plus_epsilon"
    tied_head_policy: str = "shared_compact_embedding_with_explicit_head_bridge"

    @property
    def parameter_count(self):
        return self.estimated_unique_parameters

    @property
    def tensor_bytes(self):
        return self.estimated_tensor_bytes

    def to_dict(self):
        return asdict(self)
