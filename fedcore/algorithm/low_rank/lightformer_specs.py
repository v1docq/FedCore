"""Explicit architecture and source-layer choices for bounded LightFormer transfer.

The ACL 2023 paper shares contiguous encoder layers. Decoder sharing is an
explicit reference extension here, not a claim about the authors' experiments.
"""
from dataclasses import dataclass

from .method_errors import MethodSpecError


def validate_groups(groups, name):
    if (not isinstance(groups, tuple) or not groups
            or any(type(g) is not int or g < 0 for g in groups)):
        raise MethodSpecError(f"{name} must be a nonempty immutable group map")
    # Canonical contiguous groups make both depth and unique storage explicit.
    runs = tuple(g for i, g in enumerate(groups) if i == 0 or g != groups[i - 1])
    if runs != tuple(range(len(runs))):
        raise MethodSpecError(f"{name} must contain contiguous groups numbered from zero")
    return len(runs)


@dataclass(frozen=True)
class LightFormer:
    rank: int
    encoder_groups: tuple[int, ...]
    decoder_groups: tuple[int, ...]
    encoder_sources: tuple[int, ...]
    decoder_sources: tuple[int, ...]
    max_parameters: int = 1_000_000
    max_peak_bytes: int = 64 * 1024 * 1024
    variant: str = "bounded_encoder_decoder_svd_transfer_v1"

    def __post_init__(self):
        if type(self.rank) is not int or self.rank < 1:
            raise MethodSpecError("LightFormer rank must be a positive integer")
        for name in ("max_parameters", "max_peak_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise MethodSpecError(f"{name} must be a positive integer resource bound")
        for stack in ("encoder", "decoder"):
            count = validate_groups(getattr(self, stack + "_groups"), stack + " groups")
            sources = getattr(self, stack + "_sources")
            if (not isinstance(sources, tuple) or len(sources) != count
                    or any(type(i) is not int or i < 0 for i in sources)):
                raise MethodSpecError(f"{stack} requires one explicit source index per unique group")
        if self.variant != "bounded_encoder_decoder_svd_transfer_v1":
            raise MethodSpecError("Unsupported LightFormer variant")


@dataclass(frozen=True)
class LightFormerArchitecture:
    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int
    num_heads: int
    dim_feedforward: int
    max_positions: int
    bias: bool
    norm_eps: float


@dataclass(frozen=True)
class LightFormerPlan:
    architecture: LightFormerArchitecture
    spec: LightFormer
    source_fingerprint: str
    source_parameter_elements: int
    target_parameter_elements: int
    fixed_parameter_elements: int
    source_encoder_depth: int
    source_decoder_depth: int
    dtype: str
    target_tensor_bytes: int
    estimated_peak_bytes: int

    @property
    def parameter_count(self):
        return self.target_parameter_elements

    @property
    def tensor_bytes(self):
        return self.target_tensor_bytes
