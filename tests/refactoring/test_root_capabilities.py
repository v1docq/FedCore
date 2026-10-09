import json
from importlib import resources

import pytest

from fedcore.repository.capabilities import (
    SupportLevel, UnsupportedModeError, capabilities, capability_for,
    parse_catalog, require_supported, validate_requested_modes,
)


@pytest.mark.parametrize("mode", ["cuttlefish", "DynamicRankPruner", "activation_entropy", "custom_depth", "manifold_losses", "ImportancePreservingRankSelector"])
def test_experimental_modes_fail_with_structured_reason(mode):
    with pytest.raises(UnsupportedModeError) as error:
        require_supported(mode)
    assert error.value.code == "experimental_mode"
    assert error.value.support is SupportLevel.EXPERIMENTAL
    assert error.value.reason


def test_supported_rank_mode_and_unknown_mode_are_distinct():
    assert require_supported("low_rank.onetime").support is SupportLevel.SUPPORTED
    with pytest.raises(UnsupportedModeError) as error:
        require_supported("missing")
    assert error.value.code == "unsupported_mode"


def test_catalog_is_immutable_and_alias_resolves_same_value():
    assert isinstance(capabilities(), tuple)
    assert capability_for("cuttlefish") == capability_for("low_rank.dynamic")
    assert all(item.contract and item.checks for item in capabilities())


def test_nested_configuration_rejects_before_caller_effects():
    effects = []
    with pytest.raises(UnsupportedModeError):
        validate_requested_modes({"model": {"hooks": ["cuttlefish"]}})
        effects.append("start")
    assert effects == []
    validate_requested_modes({"quantization_type": "dynamic", "importance": "magnitude"})


def test_packaged_catalog_and_duplicate_alias_validation():
    payload = json.loads(resources.files("fedcore.repository").joinpath("data", "mode_capabilities.json").read_text())
    payload["modes"].append(dict(payload["modes"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        parse_catalog(payload)


def test_real_experimental_constructors_reject_before_base_initialization(monkeypatch):
    from fedcore.algorithm.low_rank.hooks import DynamicRankPruner
    from fedcore.algorithm.pruning.pruners import BasePruner
    from fedcore.algorithm.base_compression_model import BaseCompressionModel
    effects = []
    monkeypatch.setattr(BaseCompressionModel, '__init__', lambda *args, **kwargs: effects.append('registry'))
    with pytest.raises(UnsupportedModeError):
        BasePruner({'importance': 'custom_depth'})
    with pytest.raises(UnsupportedModeError):
        DynamicRankPruner(None, None)
    assert effects == []


def test_enum_hook_configuration_is_guarded():
    from fedcore.algorithm.low_rank.hooks import LRHooks
    with pytest.raises(UnsupportedModeError):
        validate_requested_modes({'hooks': [LRHooks.cuttlefish]})


def test_boolean_catalog_version_is_rejected():
    with pytest.raises(ValueError, match='version'):
        parse_catalog({'version': True, 'modes': []})


def test_flatllm_prototypes_fail_before_model_access():
    from external.flatllmcore.core.rank_allocation import ImportancePreservingRankSelector
    from external.flatllmcore.core.absorption import AbsorptionCompressor
    with pytest.raises(UnsupportedModeError):
        ImportancePreservingRankSelector({})
    with pytest.raises(UnsupportedModeError):
        AbsorptionCompressor.patch_compressed_layers(object(), [0])
