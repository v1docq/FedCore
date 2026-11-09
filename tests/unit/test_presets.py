import pytest

from fedcore.api.presets import CompressionPreset, get_preset


def _peft_config(api_config):
    return api_config.learning_config.peft_strategy_params[0]


def _fedot_config(api_config):
    return api_config.automl_config.fedot_config


def test_all_presets_are_available():
    assert {preset.value for preset in CompressionPreset} == {
        "low_rank",
        "pruning",
        "quantization",
    }


@pytest.mark.parametrize(
    "preset,peft_cls",
    [
        ("low_rank", "LowRank"),
        ("pruning", "Pruning"),
        ("quantization", "Quantization"),
    ],
)
def test_preset_builds_expected_peft_strategy(preset, peft_cls):
    api_config = get_preset(preset, initial_assumption="ResNet18")

    peft = _peft_config(api_config)
    assert type(peft).__name__ == peft_cls
    assert type(api_config).__name__ == "APIConfig"
    assert _fedot_config(api_config).initial_assumption == "ResNet18"
    assert api_config.learning_config.criterion == "cross_entropy"
    assert api_config.learning_config.learning_strategy == "from_checkpoint"


def test_quantization_preset_matches_example_config():
    api_config = get_preset(CompressionPreset.QUANTIZATION,
                            initial_assumption="ResNet18")

    fedot = _fedot_config(api_config)
    assert fedot.problem == "classification"
    assert fedot.metric == ["MulticlassAccuracy__10", "Latency", "ModelSize"]
    assert fedot.pop_size == 1
    assert fedot.timeout == 0.1

    peft = _peft_config(api_config)
    assert peft.quant_type == "static"
    assert peft.allow_emb is False
    assert peft.allow_conv is True


def test_pruning_preset_matches_example_config():
    api_config = get_preset(CompressionPreset.PRUNING,
                            initial_assumption="ResNet18")

    fedot = _fedot_config(api_config)
    assert fedot.metric == ["MulticlassAccuracy__10", "Latency", "ModelSize"]
    assert fedot.timeout == 1

    peft = _peft_config(api_config)
    assert peft.importance == "magnitude"
    assert peft.prune_each == 5
    assert peft.epochs == 9
    assert peft.eval_each == 5
    assert peft.pruning_ratio == 0.4


def test_low_rank_preset_matches_example_config():
    api_config = get_preset(CompressionPreset.LOW_RANK,
                            initial_assumption="BigMLP")

    fedot = _fedot_config(api_config)
    assert fedot.metric == ["BinaryAccuracy", "Latency", "ModelSize"]
    assert fedot.timeout == 0.1

    peft = _peft_config(api_config)
    assert peft.strategy == "explained_variance"
    assert peft.rank_prune_each == 1
    assert peft.non_adaptive_threshold == 0.7
    assert peft.epochs == 10
    assert peft.decomposer == "svd"
    assert peft.distortion_factor == 0.6
    assert peft.random_init == "normal"
    assert peft.power == 3


def test_pruning_smoke_mode_adjustments():
    api_config = get_preset(CompressionPreset.PRUNING,
                            initial_assumption="ResNet18", smoke=True)

    fedot = _fedot_config(api_config)
    assert fedot.timeout == 0.001
    assert fedot.n_jobs == 1

    peft = _peft_config(api_config)
    assert peft.epochs == 1
    assert peft.prune_each == 1
    assert peft.eval_each == 1


def test_overrides_are_applied():
    api_config = get_preset(
        "pruning",
        initial_assumption="ResNet18",
        fedot_overrides={"timeout": 5, "metric": ["accuracy"]},
        peft_overrides={"pruning_ratio": 0.2},
        learning_overrides={"criterion": "mse"},
    )

    assert _fedot_config(api_config).timeout == 5
    assert _fedot_config(api_config).metric == ["accuracy"]
    assert _peft_config(api_config).pruning_ratio == 0.2
    assert api_config.learning_config.criterion == "mse"


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown preset"):
        get_preset("distillation")


def test_unknown_override_field_raises():
    with pytest.raises(KeyError):
        get_preset("pruning", peft_overrides={"unknown_field": 1})
