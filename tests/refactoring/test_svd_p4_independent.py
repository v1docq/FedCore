"""Independent P4 review: external oracles, counterexamples and public contracts.

These tests belong to the P4 reviewer, not to an implementation owner. Small
closed examples expose operator changes and protocol bypasses; covariance
eigendecomposition independently checks the SVD-based IPCA transition.
"""
from copy import deepcopy
import io
import math
import random
import types

import numpy as np
import pytest
import torch
from torch import nn
from hypothesis import given, settings, strategies as st

from fedcore.algorithm.low_rank.ars_profiles import gumbel_sigmoid_mask
from fedcore.algorithm.low_rank.dobi_profiles import (
    empty_direction_ipca, prepare_dobi, smooth_activation_truncation,
    update_direction_ipca,
)
from fedcore.algorithm.low_rank.learned_execution import LearnedProfileError
from fedcore.algorithm.low_rank.method_execution import plan_method, transform_method
from fedcore.algorithm.low_rank.method_specs import ARS, DobiSVD, RankDyna
from fedcore.algorithm.low_rank.rankdyna_profiles import prepare_rankdyna
from fedcore.algorithm.low_rank.rankdyna_specs import rankdyna_group_budget
from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear
from fedcore.experiments.protocol import TensorSplit
from fedcore.experiments.runner import train_model
from fedcore.tools.registry.checkpoint_manager import CheckpointManager


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


SPECS = (
    ARS(input_size=4, hidden_size=4, mask_bias=0),
    DobiSVD(max_activation_rows=16, max_calibration_batches=8),
    RankDyna(),
)


def _case():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(384)
        model = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 2)).double()
        data = torch.randn(6, 4, dtype=torch.float64)
    return model, data, model(data).detach() + 0.1


def _run_one(model, data, labels, objective, *, learning_rate=0.01):
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    with objective:
        loss = objective(model, data, labels)
        loss.backward()
        objective.before_step(model, optimizer)
        optimizer.step()
        objective.after_step(model, optimizer)
    return {"updates": 1, "training_role": "train"}


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("omission", ["backward", "before_step", "optimizer_step", "null_optimizer"])
def test_public_boundary_rejects_fabricated_optimizer_transition(spec, omission):
    source, data, labels = _case()
    snapshot = deepcopy(source.state_dict())

    def executor(model, stage, updates, objective):
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.01)
        with objective:
            loss = objective(model, data, labels)
            if omission == "null_optimizer":
                objective.after_step(model, None)
            else:
                if omission != "backward":
                    loss.backward()
                if omission != "before_step":
                    objective.before_step(model, optimizer)
                if omission != "optimizer_step":
                    optimizer.step()
                objective.after_step(model, optimizer)
        return {"updates": updates, "training_role": "train"}

    with pytest.raises(LearnedProfileError):
        transform_method(source, data, spec, target_paths=("0",), batch_size=2,
                         recovery_executor=executor)
    assert all(torch.equal(value, snapshot[name]) for name, value in source.state_dict().items())


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("modification", ["hook", "custom_forward", "child", "buffer", "extra_parameter"])
def test_public_plan_refuses_modified_linear_before_svd(spec, modification, monkeypatch):
    source, _, _ = _case()
    layer = source[0]
    if modification == "hook":
        layer.register_forward_hook(lambda module, args, result: 2 * result)
    elif modification == "custom_forward":
        layer.forward = types.MethodType(lambda self, x: 2 * nn.functional.linear(x, self.weight, self.bias), layer)
    elif modification == "child":
        layer.add_module("extra", nn.Identity())
    elif modification == "buffer":
        layer.register_buffer("extra", torch.tensor(1.0))
    else:
        layer.register_parameter("extra", nn.Parameter(torch.tensor(1.0)))

    def forbidden(*args, **kwargs):
        raise AssertionError("Refusal must precede SVD or user forward execution")

    monkeypatch.setattr(torch.linalg, "svd", forbidden)
    with pytest.raises(LearnedProfileError, match="unmodified standard Linear"):
        plan_method(source, spec, target_paths=("0",))


@pytest.mark.parametrize("spec", SPECS)
def test_functional_dense_weight_read_refuses_explicitly_and_preserves_source(spec):
    class FunctionalRead(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4, dtype=torch.float64)

        def forward(self, inputs):
            return nn.functional.linear(inputs, self.fc.weight, self.fc.bias)

    source = FunctionalRead()
    data = torch.eye(4, dtype=torch.float64)
    labels = source(data).detach()
    snapshot = deepcopy(source.state_dict())
    rng = torch.random.get_rng_state().clone()

    def executor(model, stage, updates, objective):
        return _run_one(model, data, labels, objective)

    with pytest.raises(LearnedProfileError):
        transform_method(source, data, spec, target_paths=("fc",), batch_size=2,
                         recovery_executor=executor)
    assert type(source.fc) is nn.Linear
    assert all(torch.equal(value, snapshot[name]) for name, value in source.state_dict().items())
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("noise", ["paper_single_gumbel", "binary_gumbel_difference"])
def test_ars_mask_exact_samples_and_surrogate_gradient_match_closed_formula(noise):
    logits = torch.tensor([-0.9, -0.1, 0.4, 1.3], dtype=torch.float64, requires_grad=True)
    seed, temperature, bias = 739, 0.7, -0.2
    expected_generator = torch.Generator().manual_seed(seed)

    def draw():
        uniform = torch.rand(logits.shape, dtype=logits.dtype, generator=expected_generator)
        eps = torch.finfo(logits.dtype).eps
        return -torch.log(-torch.log(uniform.clamp(eps, 1 - eps)))

    perturbation = draw()
    if noise == "binary_gumbel_difference":
        perturbation -= draw()
    probability = torch.sigmoid((logits.detach() + bias + perturbation) / temperature)
    rng = torch.random.get_rng_state().clone()
    actual = gumbel_sigmoid_mask(logits, temperature=temperature, bias=bias, noise=noise,
                                generator=torch.Generator().manual_seed(seed))
    assert torch.equal(actual.detach(), (probability >= 0.5).double())
    gradient, = torch.autograd.grad(actual.sum(), logits)
    torch.testing.assert_close(gradient, probability * (1 - probability) / temperature,
                               rtol=1e-14, atol=1e-15)
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("policy", ["selected_indices", "paper_prefix"])
def test_ars_nonprefix_export_has_the_declared_actual_operator(policy):
    source = nn.Sequential(nn.Linear(6, 6, bias=False, dtype=torch.float64))
    with torch.no_grad():
        source[0].weight.copy_(torch.diag(torch.arange(6, 0, -1, dtype=torch.float64)))
    data = torch.eye(6, dtype=torch.float64)
    spec = ARS(input_size=3, hidden_size=3, mask_bias=0, retained_ratio=0.75, export_policy=policy)

    def executor(model, stage, updates, objective):
        controller = objective.objective.hypernetwork
        with torch.no_grad():
            controller.heads[0].weight.zero_()
            controller.heads[0].bias.copy_(torch.tensor([-100., 100., -100., -100., 100., -100.]))
        return _run_one(model, data, torch.zeros_like(data), objective, learning_rate=0)

    result = transform_method(source, data, spec, target_paths=("0",), batch_size=2,
                              recovery_executor=executor)
    indices = [1, 4] if policy == "selected_indices" else [0, 1]
    expected = torch.zeros_like(source[0].weight)
    expected[indices, indices] = source[0].weight[indices, indices]
    torch.testing.assert_close(result.model(data), expected.T, rtol=0, atol=0)
    record = result.evidence["export"]["final_decisions"][0]
    assert record["selected_indices"] == [1, 4]
    assert record["exported_indices"] == indices
    assert record["selected_to_prefix_operator_squared_frobenius"] == 40
    assert record["export_changes_selected_operator"] is (policy == "paper_prefix")
    assert type(result.model[0]) is FactorizedLinear
    assert result.model[0].rank == 2
    assert sum(p.numel() for p in result.model[0].parameters()) == 24
    assert not any("hypernetwork" in name or "mask" in name for name in result.model.state_dict())


@pytest.mark.parametrize("shape", [(5, 4), (4, 5)])
def test_dobi_fixed_cluster_frechet_derivative_matches_directional_difference(shape):
    matrix = torch.zeros(shape, dtype=torch.float64)
    matrix[:4, :4] = torch.diag(torch.tensor([3., 2.9999, 0.2, 0.], dtype=torch.float64))
    matrix.requires_grad_()
    cutoff = torch.tensor(1.6, dtype=torch.float64, requires_grad=True)
    direction = torch.linspace(-0.35, 0.45, matrix.numel(), dtype=torch.float64).reshape(shape)
    upstream = direction.flip(0) + 0.3
    options = {"beta": 0.8, "spectrum_rtol": 0.01, "spectrum_atol": 0.001}
    result = smooth_activation_truncation(matrix, cutoff, **options)
    da, dk = torch.autograd.grad((result * upstream).sum(), (matrix, cutoff))
    epsilon = 2e-6
    plus = smooth_activation_truncation(matrix.detach() + epsilon * direction,
                                       cutoff.detach() + epsilon * 0.27, **options)
    minus = smooth_activation_truncation(matrix.detach() - epsilon * direction,
                                        cutoff.detach() - epsilon * 0.27, **options)
    finite_difference = ((plus - minus) * upstream).sum() / (2 * epsilon)
    torch.testing.assert_close((da * direction).sum() + dk * 0.27, finite_difference,
                               rtol=2e-6, atol=2e-8)
    assert bool(torch.isfinite(da).all()) and bool(torch.isfinite(dk))


def test_dobi_ipca_matches_independent_covariance_model_and_is_not_global_pca():
    e1, e2 = torch.tensor([[1., 0.]], dtype=torch.float64), torch.tensor([[0., 1.]], dtype=torch.float64)
    blocks = (e1, e1, e2, e2, e2)
    state = empty_direction_ipca(2, 1)
    covariance = torch.zeros(2, 2, dtype=torch.float64)
    for block in blocks:
        state = update_direction_ipca(state, block)
        values, vectors = torch.linalg.eigh(covariance + block.T @ block)
        principal = vectors[:, -1:]
        covariance = values[-1] * (principal @ principal.T)
        torch.testing.assert_close(state.components.T @ torch.diag(state.singular_values.square()) @ state.components,
                                   covariance, rtol=0, atol=1e-14)
    forward_projector = state.components.T @ state.components
    reverse = empty_direction_ipca(2, 1)
    for block in reversed(blocks):
        reverse = update_direction_ipca(reverse, block)
    reverse_projector = reverse.components.T @ reverse.components
    assert torch.equal(forward_projector, torch.diag(torch.tensor([1., 0.], dtype=torch.float64)))
    assert torch.equal(reverse_projector, torch.diag(torch.tensor([0., 1.], dtype=torch.float64)))
    global_covariance = sum((block.T @ block for block in blocks), torch.zeros_like(covariance))
    assert float(torch.trace(forward_projector @ global_covariance)) == 2
    assert float(torch.trace(reverse_projector @ global_covariance)) == 3
    assert state.rows_seen == 5 and len(state.batch_hashes) == 5


@settings(max_examples=50, deadline=None, database=None, derandomize=True)
@given(initial=st.integers(1, 100), warmup=st.integers(0, 7), adjustment=st.integers(1, 12),
       cooldown=st.integers(0, 7), tenths=st.integers(0, 10))
def test_rankdyna_phase_consistent_cubic_matches_separate_reference(initial, warmup, adjustment, cooldown, tenths):
    horizon = warmup + adjustment + cooldown
    spec = RankDyna(updates=horizon, warmup_steps=warmup, cooldown_steps=cooldown,
                    target_fraction=tenths / 10)
    target = math.ceil(initial * (tenths / 10))
    actual = [rankdyna_group_budget(initial, t, spec) for t in range(horizon + 1)]
    reference = []
    for step in range(horizon + 1):
        if step <= warmup:
            reference.append(initial)
        elif step >= horizon - cooldown:
            reference.append(target)
        else:
            remaining = 1 - (step - warmup) / adjustment
            reference.append(target + math.ceil((initial - target) * remaining**3))
    assert actual == reference
    assert all(a >= b for a, b in zip(actual, actual[1:]))


def test_rankdyna_two_removals_keep_original_ids_and_exact_multigroup_adamw_state():
    source = nn.Linear(4, 4, bias=False, dtype=torch.float64)
    spec = RankDyna(updates=2, beta=0.25, target_fraction=0.5)
    objective = prepare_rankdyna(source, ("",), spec)
    layer = objective.model
    with torch.no_grad():
        layer.s.copy_(torch.tensor([-0.2, 5., -3., 1.], dtype=torch.float64))
    optimizer = torch.optim.AdamW([
        {"params": [layer.u], "lr": 0.01},
        {"params": [layer.s], "lr": 0.02},
        {"params": [layer.vh], "lr": 0.03},
    ], amsgrad=True, weight_decay=0.05)
    initial_ids = tuple(layer.active_ids)
    expected_ema = dict.fromkeys(initial_ids, 0.0)
    data = torch.eye(4, dtype=torch.float64)
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        objective(layer, data, torch.ones_like(data)).backward()
        # Independently prescribe nonuniform group scores in original-ID space.
        layer.u.grad.zero_()
        layer.vh.grad.zero_()
        importance = {0: 1.0, 1: 6.0, 2: 3.0, 3: 8.0} if step == 1 else {0: 0.0, 1: 0.0, 2: 9.0, 3: 4.0}
        layer.s.grad.copy_(torch.tensor([importance[i.initial_column] for i in layer.active_ids], dtype=torch.float64)
                           / layer.s.detach().abs())
        old_ids = tuple(layer.active_ids)
        for identity in old_ids:
            expected_ema[identity] = 0.25 * expected_ema[identity] + 0.75 * importance[identity.initial_column]
        width = 3 if step == 1 else 2
        selected = set(sorted(old_ids, key=lambda i: (-expected_ema[i], i))[:width])
        positions = torch.tensor([n for n, identity in enumerate(old_ids) if identity in selected])
        old_parameters = {name: getattr(layer, name) for name in ("u", "s", "vh")}
        objective.before_step(layer, optimizer)
        for group in objective.state.groups:
            assert group.ema == pytest.approx(expected_ema[group.identity], rel=1e-14, abs=1e-14)
        optimizer.step()
        values = {name: parameter.detach().clone() for name, parameter in old_parameters.items()}
        states = {name: {key: value.clone() for key, value in optimizer.state[p].items()}
                  for name, p in old_parameters.items()}
        objective.after_step(layer, optimizer)
        assert layer.active_ids == tuple(identity for identity in old_ids if identity in selected)
        for name, axis in (("u", 1), ("s", 0), ("vh", 0)):
            parameter = getattr(layer, name)
            assert parameter is optimizer.param_groups[("u", "s", "vh").index(name)]["params"][0]
            assert old_parameters[name] not in optimizer.state
            torch.testing.assert_close(parameter, values[name].index_select(axis, positions), rtol=0, atol=0)
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                torch.testing.assert_close(optimizer.state[parameter][key], states[name][key].index_select(axis, positions),
                                           rtol=0, atol=0)
            assert optimizer.state[parameter]["step"].item() == step
        assert set(layer.active_ids).isdisjoint(objective.state.removed_ids)
        assert set(layer.active_ids) | set(objective.state.removed_ids) == set(initial_ids)
        assert {id(p) for p in optimizer.state} == {id(p) for p in layer.parameters()}
    assert tuple(i.initial_column for i in layer.active_ids) == (2, 3)
    assert tuple(identity.initial_column for identity in objective.state.removed_ids) == (0, 1)
    assert layer.s[0] < 0 < layer.s[1]
    # Negative and unsorted trainable S are valid; export preserves the sum.
    before = layer(data).detach()
    objective.finalize(layer, data, 2)
    torch.testing.assert_close(objective.model(data), before, rtol=1e-13, atol=1e-13)
@pytest.mark.parametrize("spec", SPECS)
def test_exception_restores_all_global_rngs_source_and_objective_context(spec):
    source, data, labels = _case()
    snapshot = deepcopy(source.state_dict())
    source_flags = tuple(p.requires_grad for p in source.parameters())
    torch_state, python_state, numpy_state = torch.random.get_rng_state().clone(), random.getstate(), np.random.get_state()
    captured = []

    def executor(model, stage, updates, objective):
        captured.append(objective)
        with objective:
            objective(model, data[:2], labels[:2]).backward()
            torch.rand(3)
            random.random()
            np.random.random(3)
            raise RuntimeError("independent-review-stop")

    with pytest.raises(RuntimeError, match="independent-review-stop"):
        transform_method(source, data, spec, target_paths=("0",), batch_size=2, recovery_executor=executor)
    assert torch.equal(torch_state, torch.random.get_rng_state())
    assert python_state == random.getstate()
    current_numpy = np.random.get_state()
    assert numpy_state[0] == current_numpy[0] and np.array_equal(numpy_state[1], current_numpy[1])
    assert numpy_state[2:] == current_numpy[2:]
    assert source_flags == tuple(p.requires_grad for p in source.parameters())
    assert all(torch.equal(value, snapshot[name]) for name, value in source.state_dict().items())
    assert captured and not captured[0].active
    delegate = captured[0].objective
    if isinstance(spec, ARS):
        assert not delegate.active and all(layer._mask is None for layer in delegate.layers)
    if isinstance(spec, DobiSVD):
        assert not delegate.active
        assert all(not layer._forward_hooks and not layer._forward_pre_hooks for layer in delegate.layers.values())


def test_dobi_failed_calibration_removes_hooks_and_restores_export_flags():
    model = nn.Sequential(nn.Linear(4, 4, dtype=torch.float64), nn.Tanh()).train()
    objective = prepare_dobi(model, ("0",), DobiSVD(max_activation_rows=4))
    target = objective.layers["0"]
    with pytest.raises(ValueError, match="ResourceLimitExceeded"):
        objective.finalize(objective.model, torch.ones(2, 5, 4, dtype=torch.float64), 1)
    assert not target.export_mode
    assert not target._forward_hooks and not target._forward_pre_hooks
    assert objective.model.training and objective.model[1].training
    assert not objective.finalized


@pytest.mark.parametrize("spec", SPECS)
def test_real_runner_reservations_freeze_and_portable_checkpoint(spec):
    source, data, labels = _case()
    split = TensorSplit(data, labels, tuple(f"independent-train-{i}" for i in range(len(data))))
    frozen = deepcopy(source[2].state_dict())

    def executor(model, stage, updates, objective):
        record = train_model(model, split, "regression", epochs=1, batch_size=2,
                             learning_rate=0.01, seed=18, batch_objective=objective, max_updates=updates)
        return dict(record, updates=record["training_steps"])

    result = transform_method(source, data, spec, target_paths=("0",), batch_size=2, recovery_executor=executor)
    for name, value in frozen.items():
        assert torch.equal(value, result.model[2].state_dict()[name])
    evidence = result.evidence
    resources, training = evidence["resources"], evidence["training"]
    assert training["training_steps"] == 1
    assert evidence["structure_training"]["observed_optimizer_steps"] == 1
    assert resources["gradient_reserve_bytes"] >= training["gradient_tensor_bytes_peak"]
    assert resources["adam_moments_reserve_bytes"] + resources["adam_step_scalar_reserve_bytes"] >= training["optimizer_state_bytes_peak"]
    assert all(type(value) is int and value >= 0 for value in resources["method_tensor_reservations"].values())
    assert "not hard-bounded" in resources["scope"]
    assert evidence["parameters_after"] == sum(p.numel() for p in result.model.parameters())
    manager = CheckpointManager("unused-in-memory-review", auto_cleanup=False)
    restored = manager.deserialize_from_bytes(manager.serialize_to_bytes(result.model)).eval()
    torch.testing.assert_close(restored(data), result.model.eval()(data), rtol=0, atol=0)
    assert restored._fedcore_method_evidence == evidence
    assert not any("hypernetwork" in name or ".k" in name or "singular_values" in name for name in restored.state_dict())
    buffer = io.BytesIO()
    torch.jit.save(torch.jit.trace(restored, data[:1]), buffer)
    buffer.seek(0)
    deployed = torch.jit.load(buffer)
    torch.testing.assert_close(deployed(data), restored(data), rtol=0, atol=0)
