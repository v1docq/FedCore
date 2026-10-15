"""Immutable descriptions of bounded reference-architecture graph transforms.

Names intentionally distinguish paper objectives from reconstructed RoPE/GQA
extensions. There is no promise of arbitrary Hugging Face model support.
"""
from dataclasses import dataclass
import math
from .method_errors import MethodSpecError


def _number(value, name, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise MethodSpecError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")


def _indices(values, name):
    if (not isinstance(values, tuple) or any(type(v) is not int or v < 0 for v in values)
            or len(set(values)) != len(values)):
        raise MethodSpecError(f"{name} must be an immutable sequence of distinct nonnegative integers")


@dataclass(frozen=True)
class FlatValueOutput:
    observation_point: str = "value_projection"
    variant: str = "uncentered_group_pca_v1"

    def __post_init__(self):
        if self.observation_point not in ("value_projection", "post_softmax"):
            raise MethodSpecError("Declare the Value observation point before or after softmax")
        if self.variant != "uncentered_group_pca_v1":
            raise MethodSpecError("Unsupported FLAT Value variant")


@dataclass(frozen=True)
class MoDeValueOutput:
    variant: str = "thin_weighted_product_v1"

    def __post_init__(self):
        if self.variant != "thin_weighted_product_v1":
            raise MethodSpecError("Unsupported MoDeGPT product variant")


@dataclass(frozen=True)
class FlatMLPChannels:
    indices: tuple[int, ...] = ()
    ridge: float = 0.0
    rcond: float | None = None
    variant: str = "appendix_c_corrected_ls_v1"

    def __post_init__(self):
        _indices(self.indices, "MLP indices")
        _number(self.ridge, "LS ridge")
        if self.rcond is not None:
            _number(self.rcond, "LS rcond")
        if self.variant != "appendix_c_corrected_ls_v1":
            raise MethodSpecError("Unsupported FLAT MLP variant")


@dataclass(frozen=True)
class MoDeMLPChannels:
    indices: tuple[int, ...] = ()
    leverage_ridge: float = 1.0
    ridge: float = 0.0
    rcond: float | None = None
    variant: str = "type_i_selected_channels_ls_v1"

    def __post_init__(self):
        _indices(self.indices, "MLP indices")
        _number(self.leverage_ridge, "leverage ridge", positive=True)
        _number(self.ridge, "LS ridge")
        if self.rcond is not None:
            _number(self.rcond, "LS rcond")
        if self.variant != "type_i_selected_channels_ls_v1":
            raise MethodSpecError("Unsupported MoDeGPT MLP variant")


@dataclass(frozen=True)
class FlatQueryKey:
    """Reconstructed coordinate-energy baseline; absent from FLAT v1."""
    coordinates: tuple[int, ...] = ()
    variant: str = "reconstructed_rope_pair_energy_v1"

    def __post_init__(self):
        _indices(self.coordinates, "Q/K coordinates")
        if self.variant != "reconstructed_rope_pair_energy_v1":
            raise MethodSpecError("Unsupported reconstructed FLAT Q/K variant")


@dataclass(frozen=True)
class MoDeQueryKey:
    """Type-II norm selection with explicit pair/GQA constraints."""
    coordinates: tuple[int, ...] = ()
    variant: str = "reconstructed_type_ii_rope_pairs_v1"

    def __post_init__(self):
        _indices(self.coordinates, "Q/K coordinates")
        if self.variant != "reconstructed_type_ii_rope_pairs_v1":
            raise MethodSpecError("Unsupported pair-constrained MoDeGPT Q/K variant")


COUPLED_SPEC_TYPES = (FlatValueOutput, MoDeValueOutput, FlatMLPChannels,
                      MoDeMLPChannels, FlatQueryKey, MoDeQueryKey)
