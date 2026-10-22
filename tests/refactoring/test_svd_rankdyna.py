"""RankDyna pure transitions, real Adam migration and portable export."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fedcore.algorithm.low_rank.method_errors import MethodSpecError
from fedcore.algorithm.low_rank.rankdyna_specs import (
    RankDyna, RankDynaID, initial_rankdyna_state, update_rankdyna_ema,
    rankdyna_group_budget, advance_rankdyna)
from fedcore.algorithm.low_rank.rankdyna_profiles import (
    RankDynaLinear, RankDynaProfileError, prepare_rankdyna, estimate_rankdyna_bytes)
from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear
from fedcore.algorithm.low_rank.learned_layers import ZeroLinear


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def linear(width=4, bias=True, dtype=torch.float64):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(53)
        return nn.Linear(width, width, bias=bias, dtype=dtype)


def rows(width=4):
    return torch.randn(7, width, dtype=torch.float64, generator=torch.Generator().manual_seed(19))


def optimizer(objective, *, lr=0.01, amsgrad=False):
    return torch.optim.Adam((p for p in objective.model.parameters() if p.requires_grad),
                            lr=lr, amsgrad=amsgrad)


def actual_step(objective, opt, data, target):
    opt.zero_grad(set_to_none=True)
    loss = objective(objective.model, data, target)
    loss.backward()
    objective.before_step(objective.model, opt)
    opt.step()
    objective.after_step(objective.model, opt)
    return float(loss.detach())


@pytest.mark.parametrize("changes", [
    {"updates": 0}, {"updates": True}, {"seed": -1}, {"seed": 2**63},
    {"target_fraction": -0.01}, {"target_fraction": 1.01}, {"target_fraction": float("nan")},
    {"beta": 1}, {"beta": -1}, {"warmup_steps": 1}, {"cooldown_steps": 1},
    {"prune_interval": 0}, {"parameter_budget": -1}, {"parameter_budget": True},
    {"freeze": "all"}, {"loss": "unobserved"}, {"reduction": "none"},
])
def test_invalid_explicit_specs_refused(changes):
    with pytest.raises(MethodSpecError):
        RankDyna(**changes)


@pytest.mark.parametrize("initial", [1, 2, 3, 8, 17])
@pytest.mark.parametrize("fraction", [0, 0.17, 0.5, 1])
def test_cubic_schedule_has_monotone_phase_boundaries(initial, fraction):
    spec = RankDyna(updates=12, warmup_steps=3, cooldown_steps=2, target_fraction=fraction)
    budgets = [rankdyna_group_budget(initial, step, spec) for step in range(13)]
    assert budgets[:4] == [initial]*4
    assert budgets[10:] == [math.ceil(initial*fraction)]*3
    assert all(left >= right for left, right in zip(budgets, budgets[1:]))
    with pytest.raises(MethodSpecError):
        rankdyna_group_budget(initial, 13, spec)


def test_immutable_ema_ties_original_ids_and_irreversible_model():
    state = initial_rankdyna_state((("z", 2, 7), ("a", 2, 3)))
    old = state
    state = update_rankdyna_ema(state, tuple((identity, 2.0) for identity in state.active_ids), 0.25)
    assert [g.ema for g in state.groups] == [1.5]*4
    assert [g.ema for g in old.groups] == [0]*4
    with pytest.raises(FrozenInstanceError):
        state.completed_updates = 10
    spec = RankDyna(updates=3, target_fraction=0.25, prune_interval=9)
    # The adjustment boundary must run even when the pruning interval misses it.
    first = advance_rankdyna(state, spec)
    second = advance_rankdyna(first.state, spec)
    last = advance_rankdyna(second.state, spec)
    assert not first.scheduled and not second.scheduled and last.scheduled
    assert last.state.active_ids == (RankDynaID("a", 0),)
    assert set(last.state.active_ids).isdisjoint(last.state.removed_ids)
    assert set(last.state.active_ids) | set(last.state.removed_ids) == set(old.active_ids)
    with pytest.raises(MethodSpecError, match="active"):
        update_rankdyna_ema(last.state, ((RankDynaID("z", 0), 3),), 0.9)
    with pytest.raises(MethodSpecError):
        advance_rankdyna(last.state, spec)


def test_weighted_parameter_extension_does_not_redefine_paper_group_target():
    state = initial_rankdyna_state((("wide", 2, 20), ("small", 2, 4)))
    scores = ((RankDynaID("wide", 0), 4), (RankDynaID("wide", 1), 3),
              (RankDynaID("small", 0), 2), (RankDynaID("small", 1), 1))
    state = update_rankdyna_ema(state, scores, 0)
    paper = advance_rankdyna(state, RankDyna(target_fraction=0.75))
    extra = advance_rankdyna(state, RankDyna(target_fraction=0.75, parameter_budget=25))
    assert paper.scheduled_group_budget == extra.scheduled_group_budget == 3
    assert len(paper.state.groups) == 3
    # Deliberately greedy by sensitivity, not optimal by value/cost ratio.
    assert extra.state.active_ids == (RankDynaID("wide", 0),)
    assert extra.scheduled_parameter_budget == 25
    assert sum(g.exported_elements for g in extra.state.groups) <= 25


def test_shape_only_reservations_before_svd_and_prepare_atomic_failure(monkeypatch):
    model = nn.Sequential(linear(), nn.Tanh(), linear())
    state = deepcopy(model.state_dict())
    flags = [p.requires_grad for p in model.parameters()]
    def fail(*_, **__):
        raise RuntimeError("SVD allocation sentinel")
    monkeypatch.setattr(torch.linalg, "svd", fail)
    reservation = estimate_rankdyna_bytes(model, ("0", "2"), RankDyna())
    assert all(type(value) is int and value >= 0 for value in reservation.values())
    assert reservation["factor_parameters_bytes"] == 2*4*(4+4+1)*8
    assert reservation["removal_copy_bytes"] >= 4*reservation["factor_parameters_bytes"]
    with pytest.raises(RuntimeError, match="sentinel"):
        prepare_rankdyna(model, ("0", "2"), RankDyna())
    assert type(model[0]) is nn.Linear and type(model[2]) is nn.Linear
    assert flags == [p.requires_grad for p in model.parameters()]
    for key, value in model.state_dict().items():
        assert torch.equal(value, state[key])


def test_owned_initialization_reconstructs_and_observed_updates_train_all_factors():
    source = nn.Sequential(linear(), nn.Tanh(), nn.LayerNorm(4, dtype=torch.float64), linear())
    snapshot = deepcopy(source.state_dict())
    rng = torch.random.get_rng_state().clone()
    objective = prepare_rankdyna(deepcopy(source), ("0", "3"), RankDyna(updates=4, target_fraction=0.5))
    data, target = rows(), torch.zeros(7, 4, dtype=torch.float64)
    torch.testing.assert_close(objective.model(data), source(data), rtol=1e-12, atol=1e-12)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(not p.requires_grad for p in objective.model[2].parameters())
    assert not objective.model[0].bias.requires_grad
    opt = optimizer(objective)
    before = {name: p.detach().clone() for name, p in objective.model.named_parameters() if p.requires_grad}
    opt.zero_grad(set_to_none=True)
    objective(objective.model, data, target).backward()
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in objective.model.parameters() if p.requires_grad)
    objective.before_step(objective.model, opt)
    opt.step()
    assert all(not torch.equal(before[name], p) for name, p in objective.model.named_parameters() if p.requires_grad)
    objective.after_step(objective.model, opt)
    counts = [len(objective.state.groups)]
    for _ in range(3):
        actual_step(objective, opt, data, target)
        counts.append(len(objective.state.groups))
    assert all(a >= b for a, b in zip(counts, counts[1:]))
    assert counts[-1] == math.ceil(objective.state.initial_groups*0.5)
    assert objective.evidence["peak_optimizer_state_bytes"] >= objective.evidence["optimizer_state_bytes"] > 0
    json.dumps(objective.evidence, allow_nan=False)
    for key, value in source.state_dict().items():
        assert torch.equal(value, snapshot[key])


def test_true_group_sensitivity_uses_absolute_values_before_summation_and_zero_based_ema():
    objective = prepare_rankdyna(linear(), ("",), RankDyna(updates=2, target_fraction=1, beta=0.5))
    layer = objective.model
    with torch.no_grad():
        layer.u.copy_(torch.arange(-8, 8, dtype=torch.float64).reshape(4, 4))
        layer.s.copy_(torch.tensor([-3., 1., -2., 0.5], dtype=torch.float64))
        layer.vh.copy_(torch.arange(-5, 11, dtype=torch.float64).reshape(4, 4))
    layer.u.grad = torch.tensor([[1., -1., 1., -1.]]*4, dtype=torch.float64)
    layer.s.grad = torch.tensor([2., -3., 4., -5.], dtype=torch.float64)
    layer.vh.grad = torch.tensor([[1., -1., 1., -1.]]*4, dtype=torch.float64)
    expected = [sum(abs(float(layer.u[row, col])*float(layer.u.grad[row, col])) for row in range(4))
                + abs(float(layer.s[col])*float(layer.s.grad[col]))
                + sum(abs(float(layer.vh[col, row])*float(layer.vh.grad[col, row])) for row in range(4))
                for col in range(4)]
    opt = optimizer(objective)
    objective.before_step(layer, opt)
    assert [g.ema for g in objective.state.groups] == [0.5*x for x in expected]
    opt.step()
    objective.after_step(layer, opt)
    assert layer.active_ids == tuple(RankDynaID("", i) for i in range(4))


@pytest.mark.parametrize("adamw", [False, True])
def test_nonprefix_physical_removal_migrates_adam_moments_gradients_and_live_references(adamw):
    objective = prepare_rankdyna(linear(), ("",), RankDyna(updates=4, target_fraction=0.25, beta=0))
    layer = objective.model
    opt_type = torch.optim.AdamW if adamw else torch.optim.Adam
    opt = opt_type([layer.u, layer.s, layer.vh], lr=0.01, amsgrad=True)
    layer.u.grad = torch.zeros_like(layer.u)
    layer.vh.grad = torch.zeros_like(layer.vh)
    layer.s.grad = torch.tensor([4., 0., 3., 2.], dtype=torch.float64)/layer.s.detach().abs()
    old = {name: getattr(layer, name) for name in ("u", "s", "vh")}
    objective.before_step(layer, opt)
    opt.step()
    expected_values = {name: parameter.detach().clone() for name, parameter in old.items()}
    expected_gradients = {name: parameter.grad.clone() for name, parameter in old.items()}
    expected_state = {name: {key: value.clone() for key, value in opt.state[parameter].items()}
                      for name, parameter in old.items()}
    objective.after_step(layer, opt)
    retained = torch.tensor([0, 2, 3])
    assert layer.rank == 3 and layer.active_ids == tuple(RankDynaID("", i) for i in retained.tolist())
    for name, axis in (("u", 1), ("s", 0), ("vh", 0)):
        new = getattr(layer, name)
        assert new is not old[name] and old[name] not in opt.state
        torch.testing.assert_close(new, expected_values[name].index_select(axis, retained), rtol=0, atol=0)
        torch.testing.assert_close(new.grad, expected_gradients[name].index_select(axis, retained), rtol=0, atol=0)
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            torch.testing.assert_close(opt.state[new][key], expected_state[name][key].index_select(axis, retained), rtol=0, atol=0)
        assert torch.equal(opt.state[new]["step"], expected_state[name]["step"])
    assert {id(p) for group in opt.param_groups for p in group["params"]} == {id(layer.u), id(layer.s), id(layer.vh)}
    values = {name: getattr(layer, name).detach().clone() for name in old}
    actual_step(objective, opt, rows(), torch.ones(7, 4, dtype=torch.float64))
    # U/V had zero gradient at the first step. They must receive a real update
    # through the optimizer's new Parameter references at the second step.
    for name in ("u", "vh"):
        new = getattr(layer, name)
        retained_positions = [retained.tolist().index(i.initial_column) for i in layer.active_ids]
        axis = 1 if name == "u" else 0
        assert not torch.equal(new, values[name].index_select(axis, torch.tensor(retained_positions)))
    assert RankDynaID("", 1) not in layer.active_ids


def test_hook_order_and_optimizer_boundary_fail_explicitly():
    objective = prepare_rankdyna(linear(), ("",), RankDyna())
    opt = optimizer(objective)
    with pytest.raises(RankDynaProfileError, match="gradient"):
        objective.before_step(objective.model, opt)
    objective(objective.model, rows(), torch.zeros(7, 4, dtype=torch.float64)).backward()
    objective.before_step(objective.model, opt)
    with pytest.raises(RankDynaProfileError, match="before_step"):
        objective.before_step(objective.model, opt)
    with pytest.raises(RankDynaProfileError, match="actual Adam"):
        objective.after_step(objective.model, opt)
    opt.step()
    objective.after_step(objective.model, opt)
    with pytest.raises(RankDynaProfileError, match="matching"):
        objective.after_step(objective.model, opt)


def test_migration_rejection_does_not_commit_partial_factor_or_optimizer_replacements():
    objective = prepare_rankdyna(nn.Sequential(linear(), linear()), ("0", "1"),
                                RankDyna(target_fraction=0.5, beta=0))
    opt = optimizer(objective)
    for layer in objective.model:
        layer.u.grad = torch.zeros_like(layer.u)
        layer.vh.grad = torch.zeros_like(layer.vh)
        layer.s.grad = torch.tensor([10., 9., 1., 0.], dtype=torch.float64)/layer.s.detach().abs()
    objective.before_step(objective.model, opt)
    opt.step()
    opt.state[objective.model[1].vh]["exp_avg"] = torch.zeros(1, dtype=torch.float64)
    old_parameters = tuple(objective.model.parameters())
    old_group = tuple(opt.param_groups[0]["params"])
    with pytest.raises(RankDynaProfileError, match="Unsupported Adam tensor"):
        objective.after_step(objective.model, opt)
    assert all(a is b for a, b in zip(old_parameters, objective.model.parameters()))
    assert all(a is b for a, b in zip(old_group, opt.param_groups[0]["params"]))
    assert objective.state.completed_updates == 0
    assert all(layer.rank == 4 for layer in objective.model)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_free_nonorthogonal_unsorted_factors_export_exact_operator_and_bias(dtype):
    objective = prepare_rankdyna(linear(dtype=dtype), ("",), RankDyna(target_fraction=1))
    layer = objective.model
    with torch.no_grad():
        layer.u.copy_(torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=dtype))
        layer.s.copy_(torch.tensor([-2., 0.5, 3., -1.], dtype=dtype))
        layer.vh.copy_(torch.tensor([[1, 2, 3, 4], [1, 0, 2, 1], [0, 1, 0, 1], [2, 1, 3, 2]], dtype=dtype))
    data = rows().to(dtype)
    opt = optimizer(objective)
    actual_step(objective, opt, data, torch.zeros(7, 4, dtype=dtype))
    before = layer(data).detach()
    left = (layer.u.detach()*layer.s.detach()[None, :]).clone()
    right = layer.vh.detach().clone()
    bias = layer.bias.detach().clone()
    rng = torch.random.get_rng_state().clone()
    export = objective.finalize(layer, data, 3)
    compact = objective.model
    assert type(compact) is FactorizedLinear and compact.rank == 4
    assert not any(name.endswith(".s") or name == "s" for name, _ in compact.named_parameters())
    torch.testing.assert_close(compact.left.weight, left, rtol=0, atol=0)
    torch.testing.assert_close(compact.right.weight, right, rtol=0, atol=0)
    torch.testing.assert_close(compact.left.bias, bias, rtol=0, atol=0)
    torch.testing.assert_close(compact(data), before, rtol=1e-5 if dtype == torch.float32 else 1e-12,
                               atol=1e-5 if dtype == torch.float32 else 1e-12)
    assert export["rank_upper_bounds"] == {"": 4}
    assert objective.evidence["ordered_singular_spectrum"] is False
    assert torch.equal(rng, torch.random.get_rng_state())
    json.dumps(objective.evidence, allow_nan=False)


def test_rank_zero_survives_cooldown_then_exports_bias_only_portable_operator():
    objective = prepare_rankdyna(linear(), ("",),
                                RankDyna(updates=3, cooldown_steps=1, target_fraction=0))
    opt = optimizer(objective)
    data = rows()
    bias = objective.model.bias.detach().clone()
    for _ in range(3):
        actual_step(objective, opt, data, torch.zeros(7, 4, dtype=torch.float64))
    assert objective.model.rank == 0
    assert all(p.numel() == 0 for p in (objective.model.u, objective.model.s, objective.model.vh))
    objective.finalize(objective.model, data, 2)
    assert type(objective.model) is ZeroLinear
    assert list(objective.model.state_dict()) == ["bias", "_anchor"]
    assert sum(p.numel() for p in objective.model.parameters()) == 4
    torch.testing.assert_close(objective.model(data), bias.expand(7, -1), rtol=0, atol=0)


def test_explicit_cross_entropy_and_selected_bias_training():
    objective = prepare_rankdyna(linear(), ("",), RankDyna(
        target_fraction=1, freeze="factors_and_bias", loss="cross_entropy", reduction="sum"))
    labels = torch.tensor([0, 1, 2, 3, 0, 2, 1])
    data = rows()
    assert objective.model.bias.requires_grad
    torch.testing.assert_close(objective(objective.model, data, labels),
                               F.cross_entropy(objective.model(data), labels, reduction="sum"))
    bias = objective.model.bias.detach().clone()
    actual_step(objective, optimizer(objective), data, labels)
    assert not torch.equal(bias, objective.model.bias)


def test_selected_aliases_and_storage_ties_refused_before_any_mutation():
    first = linear()
    alias = nn.ModuleDict({"a": first, "b": first})
    with pytest.raises(RankDynaProfileError, match="aliases"):
        prepare_rankdyna(alias, ("a",), RankDyna())
    assert alias["a"] is alias["b"] and type(alias["a"]) is nn.Linear
    other = linear()
    other.weight = first.weight
    tied = nn.Sequential(first, other)
    with pytest.raises(RankDynaProfileError, match="tied"):
        prepare_rankdyna(tied, ("0", "1"), RankDyna())
    assert type(tied[0]) is nn.Linear and tied[0].weight is tied[1].weight
