"""PETRA named-profile training boundaries with real CPU optimizer updates."""
import json
import math

import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.factor_recovery import FactorLoRAStage
from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear, ResidualLinear
from fedcore.experiments.protocol import (
    CandidateSpec, ExperimentBundle, ExperimentProtocol, ProtocolError, TensorSplit,
)
from fedcore.experiments import runner


@pytest.fixture(autouse=True)
def four_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


def _case(*, training_label_shift=0.):
    model = nn.Sequential(nn.Linear(4, 4, dtype=torch.float64), nn.Tanh(),
                          nn.Linear(4, 2, dtype=torch.float64)).eval()
    generator = torch.Generator().manual_seed(382)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=torch.float64) * .5)
    splits = []
    for index, role in enumerate(("train", "validation", "calibration", "test")):
        inputs = torch.randn((12, 4), generator=generator, dtype=torch.float64) + index * .7
        targets = model(inputs).detach() + .2 + (training_label_shift if role == "train" else 0.)
        splits.append(TensorSplit(inputs, targets, tuple(f"{role}-{row}" for row in range(12))))
    bundle = ExperimentBundle(*splits, "regression", model)
    protocol = ExperimentProtocol(batch_size=4, baseline_epochs=0, finetune_epochs=1,
                                  learning_rate=.01, threads=4, seed=43)
    return model, bundle, protocol


def _unchanged(model, snapshot):
    assert model.state_dict().keys() == snapshot.keys()
    for name, value in model.state_dict().items():
        assert torch.equal(value, snapshot[name])


def test_petra_v5_trains_left_then_right_with_fresh_optimizers_on_train_only(monkeypatch):
    model, bundle, protocol = _case()
    snapshot = {name: value.clone() for name, value in model.state_dict().items()}
    optimizers, validated, calls = [], [], []
    real_adam = torch.optim.Adam
    real_validate = FactorLoRAStage.validate_optimizer
    real_train = runner.train_model

    def adam(parameters, *args, **kwargs):
        optimizer = real_adam(parameters, *args, **kwargs)
        assert not optimizer.state
        optimizers.append(optimizer)
        return optimizer

    def validate(stage, optimizer):
        expected = set(stage.trainable_names)
        live = {name for name, parameter in stage.model.named_parameters() if parameter.requires_grad}
        assert live == expected and live
        assert all(f".{stage.factor}.lora_" in name for name in live)
        real_validate(stage, optimizer)
        validated.append((stage.factor, optimizer, {id(p) for p in stage.trainable_parameters()}))

    def train(current, split, task, **kwargs):
        assert split is bundle.train
        frozen = {name: parameter.detach().clone() for name, parameter in current.named_parameters()
                  if not parameter.requires_grad}
        live = {name: parameter.detach().clone() for name, parameter in current.named_parameters()
                if parameter.requires_grad}
        evidence = real_train(current, split, task, **kwargs)
        for name, value in frozen.items():
            assert torch.equal(dict(current.named_parameters())[name], value)
        assert any(not torch.equal(dict(current.named_parameters())[name], value) for name, value in live.items())
        calls.append(evidence)
        return evidence

    monkeypatch.setattr(torch.optim, "Adam", adam)
    monkeypatch.setattr(FactorLoRAStage, "validate_optimizer", validate)
    monkeypatch.setattr(runner, "train_model", train)
    candidate = CandidateSpec("svdllm_v5", {"rank": 2, "target_paths": ["0", "2"],
        "method_options": {"adapter_rank": 1, "left_epochs": 1, "right_epochs": 2}})
    result, evidence = runner.apply_candidate(model, bundle, protocol, candidate)
    assert len(optimizers) == len(validated) == len(calls) == 2
    assert [item[0] for item in validated] == ["left", "right"]
    assert optimizers[0] is not optimizers[1]
    assert validated[0][2].isdisjoint(validated[1][2])
    assert [call["training_steps"] for call in calls] == [3, 6]
    stages = evidence["method_transform"]["layers"][-1]["factor_recovery_stages"]
    assert [stage["stage"] for stage in stages] == ["left", "right"]
    for stage in stages:
        assert stage["adapter"]["merge_policy"] == "merged_into_factors"
        assert stage["adapter"]["optimizer_reusable"] is False
        assert all(value > 0 for value in stage["adapter"]["delta_norms"].values())
        assert all(math.isfinite(value) for value in stage["training"]["train_loss"])
        assert stage["training"]["training_role"] == "train"
    assert all(type(layer.left) is nn.Linear and type(layer.right) is nn.Linear
               for layer in result.modules() if isinstance(layer, FactorizedLinear))
    assert all(not parameter.requires_grad for parameter in result.parameters())
    assert torch.isfinite(result(bundle.validation.x)).all()
    _unchanged(model, snapshot)
    assert all(parameter.grad is None for parameter in model.parameters())
    json.dumps(evidence, allow_nan=False)


def test_petra_afm_gfm_uses_fixed_teacher_features_and_ignores_train_task_labels(monkeypatch):
    model, bundle, protocol = _case()
    _, shifted_bundle, _ = _case(training_label_shift=1000.)
    snapshot = {name: value.clone() for name, value in model.state_dict().items()}
    teacher_observations, train_roles = [], []
    real_train = runner.train_model

    def train(current, split, task, **kwargs):
        assert split is bundle.train or split is shifted_bundle.train
        teacher = kwargs.get("teacher")
        assert teacher is not None and teacher is not current
        assert kwargs["feature_path"] == "2"
        teacher_snapshot = {name: value.clone() for name, value in teacher.state_dict().items()}
        before = {name: value.clone() for name, value in current.state_dict().items()}
        evidence = real_train(current, split, task, **kwargs)
        _unchanged(teacher, teacher_snapshot)
        assert any(not torch.equal(value, before[name]) for name, value in current.state_dict().items())
        assert all(parameter.grad is None for parameter in teacher.parameters())
        teacher_observations.append(runner.model_state_hash(teacher))
        train_roles.append(evidence["training_role"])
        return evidence

    def forbidden_task_loss(*args, **kwargs):
        raise AssertionError("GFM must use the declared fixed teacher feature objective")

    monkeypatch.setattr(runner, "train_model", train)
    monkeypatch.setattr(runner, "task_loss", forbidden_task_loss)
    candidate = CandidateSpec("afm", {"rank": 1, "target_paths": ["0", "2"],
                                       "finetune_epochs": 2, "gfm_feature_path": "2"})
    result, evidence = runner.apply_candidate(model, bundle, protocol, candidate)
    shifted, shifted_evidence = runner.apply_candidate(model, shifted_bundle, protocol, candidate)
    assert train_roles == ["train", "train"]
    assert teacher_observations == [runner.model_state_hash(model)] * 2
    assert evidence["objective"] == "GFM_late_feature_mse"
    assert evidence["teacher_checkpoint"] == runner.model_state_hash(model)
    assert evidence["teacher_feature_map"] == {"teacher": "2", "student": "2"}
    assert evidence["training_steps"] == 6
    assert evidence["train_loss"] == shifted_evidence["train_loss"]
    assert all(math.isfinite(value) for value in evidence["train_loss"])
    for name, value in result.state_dict().items():
        torch.testing.assert_close(value, shifted.state_dict()[name], rtol=0, atol=0)
    assert torch.isfinite(result(bundle.validation.x)).all()
    assert all(not layer._forward_hooks for layer in result.modules())
    _unchanged(model, snapshot)
    json.dumps(evidence, allow_nan=False)


def test_petra_eora_uses_explicit_pruned_base_and_trains_residual_on_train_only(monkeypatch):
    model, bundle, protocol = _case()
    snapshot = {name: value.clone() for name, value in model.state_dict().items()}
    calls = []
    real_train = runner.train_model

    def train(current, split, task, **kwargs):
        assert split is bundle.train
        residuals = [layer for layer in current.modules() if isinstance(layer, ResidualLinear)]
        frozen = [{name: value.clone() for name, value in layer.base.state_dict().items()} for layer in residuals]
        evidence = real_train(current, split, task, **kwargs)
        for layer, original in zip(residuals, frozen):
            _unchanged(layer.base, original)
            assert all(not parameter.requires_grad for parameter in layer.base.parameters())
        calls.append(evidence)
        return evidence

    monkeypatch.setattr(runner, "train_model", train)
    candidate = CandidateSpec("eora", {"rank": 1, "target_paths": ["0"], "finetune_epochs": 1,
        "base_candidate": {"method": "pruning", "parameters": {"amount": .3, "finetune_epochs": 0}}})
    result, evidence = runner.apply_candidate(model, bundle, protocol, candidate)
    assert [call["training_steps"] for call in calls] == [0, 3]
    assert evidence["base_candidate"]["pruned_weight_count"] > 0
    assert isinstance(result[0], ResidualLinear)
    assert bool((result[0].base.weight == 0).any())
    assert bool(torch.count_nonzero(result[0].residual.left.weight.grad))
    assert not torch.equal(result[0].base.weight, model[0].weight)
    assert evidence["training_steps"] == 3
    assert all(math.isfinite(value) for value in evidence["train_loss"])
    assert torch.isfinite(result(bundle.validation.x)).all()
    _unchanged(model, snapshot)


@pytest.mark.parametrize("parameters", [
    {"rank": 2, "target_paths": ["0"], "finetune_epochs": 1},
    {"rank": 2, "target_paths": ["0"], "gfm_feature_path": "2"},
])
def test_v5_refuses_ordinary_training_budget_before_optimizer_creation(monkeypatch, parameters):
    model, bundle, protocol = _case()
    def forbidden_optimizer(*args, **kwargs):
        raise AssertionError("Conflicting training budgets must refuse before optimizer creation")
    monkeypatch.setattr(torch.optim, "Adam", forbidden_optimizer)
    with pytest.raises(ProtocolError):
        runner.apply_candidate(model, bundle, protocol, CandidateSpec("svdllm_v5", parameters))


def test_v5_rejects_optimizer_retained_from_another_model():
    from fedcore.algorithm.low_rank.factor_recovery import prepare_factor_lora_stage
    from fedcore.algorithm.low_rank.method_execution import transform_method
    from fedcore.algorithm.low_rank.method_specs import SVDLLMV2
    model, bundle, _ = _case()
    stale = torch.optim.Adam(model.parameters(), lr=.01)
    compressed = transform_method(model, bundle.calibration.x, SVDLLMV2(), rank=2,
                                   target_paths=("0", "2")).model
    stage = prepare_factor_lora_stage(compressed, ("0", "2"), factor="left", rank=1)
    with pytest.raises(ValueError, match="stale|unauthorized"):
        stage.validate_optimizer(stale)


def test_petra_eora_refuses_implicit_or_unsupported_base_candidate():
    model, bundle, protocol = _case()
    for base in (None, {"method": "weighted_svd", "parameters": {"rank": 1}}):
        parameters = {"rank": 1, "target_paths": ["0"]}
        if base is not None:
            parameters["base_candidate"] = base
        with pytest.raises(ProtocolError, match="base_candidate"):
            runner.apply_candidate(model, bundle, protocol, CandidateSpec("eora", parameters))
