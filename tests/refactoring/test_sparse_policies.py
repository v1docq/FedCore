"""Enumeration and independent spectral tests of the PruneNet correction."""
from copy import deepcopy
from itertools import permutations

import pytest
import torch
from torch.nn import functional as F

from fedcore.algorithm.low_rank.coupled_layers import ReferenceMLP
from fedcore.algorithm.pruning.sparse.channels import SparseMLP
from fedcore.algorithm.pruning.sparse.policies import (
    PruneNetSpec, WeightPolicy, OrderedAction, ordered_log_probability,
    sample_ordered_action, future_costs, reinforce_cost_loss, ks_distance,
    spectral_cost, train_prunenet_policy, prunenet,
)


@pytest.fixture(autouse=True)
def cpu_scope():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(153012)
        yield
    torch.set_num_threads(previous)


def test_ordered_probability_enumeration_normalizes_and_has_correct_expected_gradient():
    logits = torch.tensor([-.8, .4, 1.1], dtype=torch.float64, requires_grad=True)
    noise = torch.tensor([.2, .7, .4], dtype=torch.float64)
    orders = tuple(permutations(range(3), 2))
    logs = tuple(ordered_log_probability(logits, noise, order) for order in orders)
    probabilities = tuple(value.exp() for value in logs)
    torch.testing.assert_close(sum(probabilities), torch.tensor(1., dtype=torch.float64), rtol=1e-14, atol=1e-14)
    weights = torch.sigmoid(logits + torch.logit(noise))
    for order, probability in zip(orders, probabilities):
        a, b = order
        expected = weights[a]/weights.sum() * weights[b]/(weights.sum()-weights[a])
        torch.testing.assert_close(probability, expected, rtol=1e-14, atol=1e-14)
    costs = (0., .2, .9, .1, .4, .7)
    exact_objective = sum(p*c for p, c in zip(probabilities, costs))
    exact_gradient, = torch.autograd.grad(exact_objective, logits, retain_graph=True)
    score_objective = sum(p.detach()*c*log for p, c, log in zip(probabilities, costs, logs))
    score_gradient, = torch.autograd.grad(score_objective, logits)
    torch.testing.assert_close(score_gradient, exact_gradient, rtol=1e-12, atol=1e-12)
    # Gradient descent has the direction of decreasing expected cost.
    next_logits = (logits.detach() - .001*score_gradient).requires_grad_()
    next_cost = sum(ordered_log_probability(next_logits, noise, order).exp()*c for order, c in zip(orders, costs))
    assert float(next_cost) < float(exact_objective)


def test_sampling_reuses_noise_and_order_and_handles_boundary_probabilities():
    logits = torch.tensor([-1000., 0., 1000.], dtype=torch.float64, requires_grad=True)
    generator = torch.Generator().manual_seed(80)
    action = sample_ordered_action(logits, 2, generator=generator)
    assert len(action.order) == len(set(action.order)) == 2
    exact = ordered_log_probability(logits, action.noise, action.order)
    assert torch.equal(exact, action.log_probability)
    assert torch.isfinite(exact)
    exact.backward()
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError, match="strictly inside"):
        ordered_log_probability(logits, torch.zeros(3, dtype=torch.float64), (0, 1))


def test_finite_returns_include_future_costs_and_loss_sign():
    assert future_costs((1., 2., 3.), .5) == (2.75, 3.5, 3.)
    logs = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (-.2, -.3, -.4)]
    actions = [OrderedAction((0,), torch.tensor([.5]), log, torch.zeros(1)) for log in logs]
    loss = reinforce_cost_loss(actions, (1., 2., 3.), .5)
    loss.backward()
    assert [float(log.grad) for log in logs] == [2.75, 1.75, .75]
    with pytest.raises(ValueError):
        future_costs((float("inf"),), .99)


def test_policy_is_paper_left_projection_without_dense_h_squared_temporary():
    policy = WeightPolicy(3, 2, generator=torch.Generator().manual_seed(14))
    weight = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    expected = policy.projection @ (weight.detach() @ policy.intermediate.T)
    torch.testing.assert_close(policy(weight), expected, rtol=1e-14, atol=1e-14)
    policy(weight).sum().backward()
    assert weight.grad is None
    assert sum(p.numel() for p in policy.parameters()) == 3*2+3


def test_spectra_have_min_shape_length_and_ks_does_not_add_absent_zeros():
    tall = torch.tensor([[1., 0.], [0., 2.], [1., 1.], [2., 1.]], dtype=torch.float64)
    original = torch.linalg.svdvals(tall)
    selected = torch.linalg.svdvals(tall[[1]])
    assert len(original) == 2 and len(selected) == 1
    thresholds = sorted(set(original.tolist()+selected.tolist()))
    independent = max(abs(sum(v <= z for v in original.tolist())/2 - sum(v <= z for v in selected.tolist())) for z in thresholds)
    assert spectral_cost(tall, (1,)) == independent
    assert ks_distance(torch.tensor([1., 1.]), torch.tensor([1.])) == 0.


def test_zero_ks_is_not_a_functional_quality_guarantee():
    model = ReferenceMLP(2, 2, gated=False, activation="relu", bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.up_proj.weight.copy_(torch.eye(2))
        model.down_proj.weight.copy_(torch.eye(2))
    assert spectral_cost(model.up_proj.weight, (0,)) == spectral_cost(model.up_proj.weight, (1,)) == 0.
    x = torch.tensor([[1., 0.]], dtype=torch.float64)
    first = F.linear(F.relu(F.linear(x, model.up_proj.weight[[0]])), model.down_proj.weight[:, [0]])
    second = F.linear(F.relu(F.linear(x, model.up_proj.weight[[1]])), model.down_proj.weight[:, [1]])
    assert not torch.equal(first, second)


def test_real_policy_steps_freeze_source_and_export_only_compact_saved_indices():
    model = ReferenceMLP(4, 7, dtype=torch.float64)
    before, rng = deepcopy(model.state_dict()), torch.get_rng_state().clone()
    compact, evidence = prunenet(model, PruneNetSpec(keep_count=3, episodes=4, seed=11))
    assert evidence["policy_changed"] is True
    assert evidence["optimizer_steps"] == 4
    assert evidence["original_spectrum_lengths"] == (4,)
    assert evidence["calibration_used"] is False and evidence["policy_exported"] is False
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert torch.equal(rng, torch.get_rng_state())
    assert not any(isinstance(layer, WeightPolicy) for layer in compact.modules())
    x = torch.randn(3, 4, dtype=torch.float64)
    selected = list(evidence["selected_current_indices"])
    hidden = F.silu(F.linear(x, model.gate_proj.weight[selected], model.gate_proj.bias[selected]))
    hidden = hidden*F.linear(x, model.up_proj.weight[selected], model.up_proj.bias[selected])
    expected = F.linear(hidden, model.down_proj.weight[:, selected], model.down_proj.bias)
    torch.testing.assert_close(compact(x), expected, rtol=1e-13, atol=1e-13)
    restored = SparseMLP.from_config(compact.representation_config(), dtype=torch.float64)
    restored.load_state_dict(compact.state_dict())
    assert torch.equal(restored(x), compact(x))


def test_shared_policy_has_future_returns_across_multiple_frozen_layer_states():
    models = [ReferenceMLP(3, 5, gated=False, dtype=torch.float64) for _ in range(3)]
    policy, evidence = train_prunenet_policy(models, PruneNetSpec(keep_count=2, episodes=2, discount=.5))
    for episode in evidence["history"]:
        costs = episode["costs"]
        expected = tuple(sum(.5**(j-i)*costs[j] for j in range(i, 3)) for i in range(3))
        assert episode["future_costs"] == expected
    assert all(p.grad is None for module in models for p in module.parameters())
    assert all(p.grad is not None for p in policy.parameters())


def test_weight_only_policy_rejects_budget_and_cardinality_before_allocating():
    model = ReferenceMLP(4, 7, dtype=torch.float64)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="workspace"):
        prunenet(model, PruneNetSpec(keep_count=2, max_workspace_bytes=1))
    with pytest.raises(ValueError, match="widths"):
        prunenet(model, PruneNetSpec(keep_count=8))
    assert torch.equal(rng, torch.get_rng_state())
