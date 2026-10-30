"""Independent algebra, seed isolation and the bounded ARS lifecycle."""
from copy import deepcopy
import json
import math
import subprocess
import sys

import pytest
import torch
from torch import nn
from hypothesis import given, settings, strategies as st

from fedcore.algorithm.low_rank.ars_specs import ARS
from fedcore.algorithm.low_rank.ars_profiles import (
    ARSHypernetwork, ARSMaskedLinear, ars_temperature, estimate_ars_bytes,
    gumbel_sigmoid_mask, prepare_ars,
)
from fedcore.algorithm.low_rank.learned_layers import ZeroLinear
from fedcore.algorithm.low_rank.method_errors import MethodSpecError
from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear


def _diagonal(values=(9., 5., 2., 1.), *, bias=True):
    layer = nn.Linear(len(values), len(values), bias=bias, dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(torch.diag(torch.tensor(values, dtype=torch.float64)))
        if bias:
            layer.bias.zero_()
    return nn.Sequential(layer)


def _force_logits(objective, values):
    with torch.no_grad():
        for head, logits in zip(objective.hypernetwork.heads, values):
            head.weight.zero_()
            head.bias.copy_(torch.tensor(logits, dtype=head.bias.dtype) - objective.spec.mask_bias)


def _train(objective, inputs, labels, *, lr=.01):
    optimizer = torch.optim.SGD([p for p in objective.model.parameters() if p.requires_grad], lr=lr)
    with objective:
        for _ in range(objective.spec.updates):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(objective.model, inputs, labels)
            loss.backward()
            objective.before_step(objective.model, optimizer)
            optimizer.step()
            objective.after_step(objective.model, optimizer)
    return optimizer


def test_ars_spec_import_is_metadata_only():
    code = ('import sys; from fedcore.algorithm.low_rank.ars_specs import ARS; '
            'assert "torch" not in sys.modules; assert ARS().noise == "paper_single_gumbel"')
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("options", [
    {"updates": 0}, {"updates": True}, {"seed": -1}, {"seed": 2**63},
    {"retained_ratio": 0}, {"retained_ratio": 1.1}, {"temperature_start": 0},
    {"temperature_end": math.inf}, {"alignment_weight": -1}, {"mask_bias": math.nan},
    {"noise": "gumbel"}, {"export_policy": "prefix"}, {"loss": "pseudo_labels"},
    {"structural_trainable": 1}, {"structural_trainable": False, "weight_trainable": False},
])
def test_ars_rejects_ambiguous_or_invalid_description(options):
    with pytest.raises(MethodSpecError):
        ARS(**options)


@pytest.mark.parametrize("noise", ["paper_single_gumbel", "binary_gumbel_difference"])
def test_gumbel_mask_binary_forward_and_independent_straight_through_derivative(noise):
    logits = torch.tensor([-.7, .2, 1.1], dtype=torch.float64, requires_grad=True)
    seed, temperature, bias = 102, .7, .1
    # Independent formula, with a separately seeded generator and no production helper.
    generator = torch.Generator().manual_seed(seed)
    uniform = torch.rand(3, generator=generator, dtype=torch.float64)
    gumbel = -torch.log(-torch.log(uniform))
    if noise == "binary_gumbel_difference":
        other = torch.rand(3, generator=generator, dtype=torch.float64)
        gumbel = gumbel + torch.log(-torch.log(other))
    probability = torch.sigmoid((logits.detach() + bias + gumbel) / temperature)
    coefficients = torch.tensor([2., -3., 5.], dtype=torch.float64)
    mask = gumbel_sigmoid_mask(logits, temperature=temperature, bias=bias, noise=noise,
                              generator=torch.Generator().manual_seed(seed))
    assert set(mask.detach().tolist()) <= {0., 1.}
    torch.testing.assert_close(mask.detach(), (probability >= .5).double(), rtol=0, atol=0)
    (mask * coefficients).sum().backward()
    torch.testing.assert_close(logits.grad, coefficients * probability * (1 - probability) / temperature,
                               rtol=1e-13, atol=1e-14)


def test_temperature_schedule_is_driven_by_updates_and_has_exact_endpoints():
    spec = ARS(updates=3, temperature_start=1., temperature_end=.2)
    assert [ars_temperature(spec, i) for i in range(4)] == [1., .6, .2, .2]
    assert ars_temperature(ARS(updates=1, temperature_start=1., temperature_end=.2), 0) == 1.


def test_literal_loss_components_match_independent_diagonal_reference():
    model = _diagonal()
    spec = ARS(mask_bias=0., retained_ratio=.5, budget_weight=2., alignment_weight=.25,
               input_size=2, hidden_size=2)
    objective = prepare_ars(model, ("0",), spec)
    _force_logits(objective, [(-1000., 1000., -1000., 1000.)])
    _train(objective, torch.eye(4, dtype=torch.float64), torch.zeros(4, 4, dtype=torch.float64))
    row = objective.evidence["history"][0]
    # Original 16 weights + 4 immutable biases; masked factors cost 8*2 + 4.
    assert row["search_parameter_elements"] == 20
    assert row["task_loss"] == pytest.approx(26 / 16)
    assert row["budget_loss"] == pytest.approx(math.log(20 / 10))
    assert row["alignment_loss"] == pytest.approx(9**2 + 1**2)
    assert row["total_loss"] == pytest.approx(26 / 16 + 2 * math.log(2) + .25 * 82)


def test_shared_gru_receives_gradient_and_optimizer_changes_structure_only():
    model = nn.Sequential(nn.Linear(3, 4, dtype=torch.float64), nn.Tanh(),
                          nn.Linear(4, 2, dtype=torch.float64))
    original = deepcopy(model)
    spec = ARS(updates=2, seed=42, mask_bias=0., temperature_start=1., temperature_end=.5,
               retained_ratio=.8, alignment_weight=.1, input_size=3, hidden_size=4)
    objective = prepare_ars(model, ("0", "2"), spec)
    factors = [(layer.left.detach().clone(), layer.right.detach().clone()) for layer in objective.layers]
    controller = {name: p.detach().clone() for name, p in objective.hypernetwork.named_parameters()}
    assert objective.hypernetwork.gru.bidirectional
    assert sum(isinstance(module, ARSHypernetwork) for module in model.modules()) == 1
    x = torch.arange(18, dtype=torch.float64).reshape(6, 3) / 18
    _train(objective, x, torch.zeros(6, 2, dtype=torch.float64))
    assert any(not torch.equal(controller[name], p) for name, p in objective.hypernetwork.named_parameters())
    assert any(p.grad is not None and bool(torch.count_nonzero(p.grad))
               for p in objective.hypernetwork.gru.parameters())
    for layer, (left, right) in zip(objective.layers, factors):
        torch.testing.assert_close(layer.left, left, rtol=0, atol=0)
        torch.testing.assert_close(layer.right, right, rtol=0, atol=0)
        assert not layer.left.requires_grad and not layer.right.requires_grad
    assert all(p.grad is None for p in original.parameters())
    assert objective.evidence["completed_updates"] == 2


def test_structure_and_weight_trainability_have_separate_boundaries():
    objective = prepare_ars(_diagonal(), ("0",), ARS(structural_trainable=False,
        weight_trainable=True, input_size=2, hidden_size=2, mask_bias=1000.))
    frozen = {name: p.detach().clone() for name, p in objective.hypernetwork.named_parameters()}
    before = objective.layers[0].left.detach().clone()
    _train(objective, torch.eye(4, dtype=torch.float64), torch.zeros(4, 4, dtype=torch.float64))
    assert not torch.equal(before, objective.layers[0].left)
    assert all(not p.requires_grad and p.grad is None and torch.equal(p, frozen[name])
               for name, p in objective.hypernetwork.named_parameters())


def test_seed_determinism_and_global_rng_isolation_through_prepare_search_and_export():
    original = _diagonal()
    before = {name: value.clone() for name, value in original.state_dict().items()}
    x = torch.eye(4, dtype=torch.float64)
    records = []
    states = []
    torch.manual_seed(12345)
    ambient = torch.random.get_rng_state().clone()
    for _ in range(2):
        objective = prepare_ars(deepcopy(original), ("0",), ARS(seed=31, mask_bias=0.,
            temperature_start=1., temperature_end=1., input_size=2, hidden_size=3))
        _train(objective, x, torch.zeros_like(x))
        records.append(objective.finalize(objective.model))
        states.append({name: value.clone() for name, value in objective.model.state_dict().items()})
        assert torch.equal(ambient, torch.random.get_rng_state())
    assert records[0] == records[1]
    for name, value in states[0].items():
        torch.testing.assert_close(value, states[1][name], rtol=0, atol=0)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    json.dumps(records, allow_nan=False)


@pytest.mark.parametrize("policy, expected", [
    ("selected_indices", [0., 5., 0., 1.]), ("paper_prefix", [9., 5., 0., 0.]),
])
def test_nonprefix_export_preserves_indices_and_prefix_explicitly_changes_operator(policy, expected):
    original = _diagonal()
    objective = prepare_ars(deepcopy(original), ("0",), ARS(export_policy=policy,
        input_size=2, hidden_size=2))
    _force_logits(objective, [(-1000., 1000., -1000., 1000.)])
    x = torch.eye(4, dtype=torch.float64)
    _train(objective, x, torch.zeros_like(x))
    record = objective.finalize(objective.model)
    layer = objective.model[0]
    assert type(layer) is FactorizedLinear and layer.rank == 2
    assert layer.left.weight.shape == (4, 2) and layer.right.weight.shape == (2, 4)
    assert layer.left.weight.untyped_storage().nbytes() == 4 * 2 * 8
    assert layer.right.weight.untyped_storage().nbytes() == 2 * 4 * 8
    torch.testing.assert_close(objective.model(x), torch.diag(torch.tensor(expected, dtype=torch.float64)))
    decision = record["final_decisions"][0]
    assert decision["selected_indices"] == [1, 3]
    assert decision["selected_to_prefix_operator_squared_frobenius"] == pytest.approx(82.)
    assert decision["export_changes_selected_operator"] == (policy == "paper_prefix")
    assert record["hypernetwork_removed"]
    assert not any(isinstance(module, (ARSHypernetwork, ARSMaskedLinear)) for module in objective.model.modules())


@pytest.mark.parametrize("keep", [False, True])
def test_zero_and_full_rank_export_root_are_portable_and_preserve_bias(keep):
    original = _diagonal((3., 2.), bias=True)[0]
    with torch.no_grad():
        original.bias.copy_(torch.tensor([.2, -.4], dtype=torch.float64))
    objective = prepare_ars(deepcopy(original), ("",), ARS(input_size=2, hidden_size=2))
    _force_logits(objective, [(1000., 1000.) if keep else (-1000., -1000.)])
    x = torch.eye(2, dtype=torch.float64)
    _train(objective, x, torch.zeros_like(x))
    record = objective.finalize(objective.model)
    result = objective.model
    assert type(result) is (FactorizedLinear if keep else ZeroLinear)
    expected = original(x) if keep else original.bias.expand(2, -1)
    torch.testing.assert_close(result(x), expected)
    assert record["final_parameter_elements"] == (10 if keep else 2)
    assert record["final_budget_satisfied"] == (not keep)
    if not keep:
        assert not any(name.endswith("weight") for name, _ in result.named_parameters())
    assert result.rank == (2 if keep else 0)


def test_estimate_precedes_allocation_and_matches_persistent_tensor_components():
    model = _diagonal()
    spec = ARS(input_size=2, hidden_size=3)
    estimate = estimate_ars_bytes(model, ("0",), spec)
    objective = prepare_ars(model, ("0",), spec)
    layer = objective.layers[0]
    size = lambda tensor: tensor.numel() * tensor.element_size()
    assert estimate["factor_parameters_bytes"] == sum(size(p) for p in (layer.left, layer.right, layer.bias))
    assert estimate["singular_value_buffers_bytes"] == size(layer.singular_values)
    assert estimate["fixed_hypernetwork_input_bytes"] == size(objective.hypernetwork.z)
    assert estimate["hypernetwork_parameters_bytes"] == sum(size(p) for p in objective.hypernetwork.parameters())
    assert all(type(value) is int and value >= 0 for value in estimate.values())


def test_context_clears_masks_and_pending_graph_on_failure_without_hooks():
    objective = prepare_ars(_diagonal(), ("0",), ARS(input_size=2, hidden_size=2))
    with pytest.raises(ValueError, match="abort"):
        with objective:
            objective(objective.model, torch.eye(4, dtype=torch.float64), torch.zeros(4, 4, dtype=torch.float64))
            raise ValueError("abort")
    assert not objective.active and objective._pending is None
    assert all(layer._mask is None and not layer._forward_hooks for layer in objective.layers)
    with pytest.raises(ValueError, match="all declared updates"):
        objective.finalize(objective.model)


def test_invalid_targets_fail_before_replacement_or_trainability_mutation():
    model = _diagonal()
    original = model[0]
    with pytest.raises(ValueError, match="immutable"):
        prepare_ars(model, ("0",), ARS(retained_ratio=.01))
    assert model[0] is original and all(p.requires_grad for p in model.parameters())
    model.add_module("alias", original)
    with pytest.raises(ValueError, match="aliases"):
        prepare_ars(model, ("0",), ARS())
    assert model[0] is original


@given(st.integers(min_value=1, max_value=5), st.integers(min_value=1, max_value=5))
@settings(max_examples=12, deadline=None)
def test_full_selected_factor_operator_matches_independent_dense_reference(out_features, in_features):
    # This checks reconstruction across rectangular and rank-one boundaries,
    # not any quality or speed statement about learned rank selection.
    generator = torch.Generator().manual_seed(901)
    matrix = torch.randn((out_features, in_features), generator=generator, dtype=torch.float64)
    layer = nn.Linear(in_features, out_features, bias=False, dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(matrix)
    objective = prepare_ars(nn.Sequential(layer), ("0",), ARS(retained_ratio=1.,
        input_size=2, hidden_size=2, structural_trainable=False, weight_trainable=True))
    x = torch.randn((3, in_features), generator=generator, dtype=torch.float64)
    torch.testing.assert_close(objective.model(x), x @ matrix.T, rtol=1e-12, atol=1e-12)
