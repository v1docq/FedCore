"""Per-example analytic gradient references and collector isolation contracts."""
from dataclasses import replace
import random

import numpy as np
import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.approximation import NumericalDomainViolation
from fedcore.algorithm.low_rank.statistics import StatisticsPassport
from fedcore.algorithm.low_rank.statistical_collectors import collect_empirical_gradient_squares
from fedcore.algorithm.low_rank.statistical_profiles import GradientCollectionMetadata, solve_fwsvd


def passport(**updates):
    return replace(StatisticsPassport("checkpoint-hash", "graph-hash", "predecessor-hash", "weight_gradient", "train-data", data_role="train"), **updates)


def collect(model, batches, *, reduction="mean", max_examples=10, max_batches=10, stamp=None, loss_fn=None):
    return collect_empirical_gradient_squares(
        model, ("",), batches, loss_fn or (lambda prediction, target: (prediction - target).square()),
        stamp or passport(), GradientCollectionMetadata("unreduced_squared_error_v1", "observed-targets", reduction=reduction),
        max_examples=max_examples, max_batches=max_batches)[""]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("reduction,denominator", [("mean", 2), ("sum", 1)])
def test_individual_fisher_matches_manual_outer_product_gradients(dtype, reduction, denominator):
    layer = nn.Linear(3, 2, bias=True, dtype=dtype)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1., 2., -1.], [3., -2., .5]], dtype=dtype))
        layer.bias.copy_(torch.tensor([.2, -.4], dtype=dtype))
    x = torch.tensor([[1., 2., -1.], [2., -1., 3.], [-1., 2., 0.]], dtype=dtype)
    targets = torch.tensor([[0., 1.], [1., -2.], [2., 0.]], dtype=dtype)
    stats = collect(layer, [(x[:2], targets[:2]), (x[2:], targets[2:])], reduction=reduction)
    # Analytic derivative of sum/mean squared output error. No autograd or
    # production reduction helper participates in this numerical reference.
    residual = x.double() @ layer.weight.detach().double().T + layer.bias.detach().double() - targets.double()
    individual = np.stack([np.outer(2 * residual[i].numpy() / denominator, x[i].double().numpy())
                           for i in range(len(x))])
    expected = np.mean(individual ** 2, axis=0)
    np.testing.assert_allclose(stats.mean_square.numpy(), expected, rtol=3e-7, atol=1e-12)
    assert stats.passport.count == 3
    assert stats.metadata.reduction == reduction and stats.metadata.labels_id == "observed-targets"
    assert stats.collection_manifest["backward_passes"] == 3
    assert stats.collection_manifest["observed_batches"] == 2
    assert stats.collection_manifest["loss_reduction_scope"] == "within_individual_example"
    assert stats.passport.checkpoint_id == "checkpoint-hash"
    assert solve_fwsvd(layer.weight.detach(), stats, 1).manifest["aggregation"] == "sum_over_output_axis_dim0"


def test_g_and_negative_g_do_not_cancel_in_individual_gradient_squares():
    layer = nn.Linear(1, 1, bias=False).double()
    layer.weight.data.zero_()
    x = torch.ones((2, 1), dtype=torch.float64)
    y = torch.tensor([[1.], [-1.]], dtype=torch.float64)
    stats = collect(layer, [(x, y)])
    assert stats.mean_square.item() == 4.
    # Batch mean gradient vanishes for the same data.
    batch_loss = (layer(x) - y).square().mean()
    assert torch.autograd.grad(batch_loss, layer.weight)[0].item() == 0


def assert_numpy_rng_equal(left, right):
    assert left[0] == right[0] and left[2:] == right[2:]
    np.testing.assert_array_equal(left[1], right[1])


@pytest.mark.parametrize("fail", [False, True])
def test_original_weights_buffers_mixed_modes_grad_objects_and_rng_survive(fail):
    model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2), nn.Dropout(.5), nn.Linear(2, 1)).double()
    model.train()
    model[1].eval()
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, .25)
    original_modes = [item.training for item in model.modules()]
    original_parameters = tuple(model.parameters())
    original_grads = tuple(parameter.grad for parameter in original_parameters)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    python_before, numpy_before, torch_before = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    calls = []

    def loss(prediction, labels):
        random.random()
        np.random.random()
        torch.rand(1)
        calls.append(1)
        if fail:
            raise RuntimeError("intentional loss failure")
        return (prediction - labels).square()

    arguments = (model, ("0", "3"), [(torch.ones((3, 2), dtype=torch.float64), torch.ones((3, 1), dtype=torch.float64))],
                 loss, passport(), GradientCollectionMetadata("loss", "labels"))
    if fail:
        with pytest.raises(RuntimeError, match="intentional"):
            collect_empirical_gradient_squares(*arguments, max_examples=2, max_batches=1)
    else:
        results = collect_empirical_gradient_squares(*arguments, max_examples=2, max_batches=1)
        assert set(results) == {"0", "3"}
        assert all(value.collection_manifest["backward_passes"] == 2 for value in results.values())
    assert [item.training for item in model.modules()] == original_modes
    assert all(before is after for before, after in zip(original_parameters, model.parameters()))
    assert all(parameter.grad is gradient for parameter, gradient in zip(model.parameters(), original_grads))
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, state[name])
    for gradient in original_grads:
        assert bool((gradient == .25).all())
    assert random.getstate() == python_before
    assert_numpy_rng_equal(np.random.get_state(), numpy_before)
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    assert all(len(item._forward_hooks) == 0 and len(item._backward_hooks) == 0 for item in model.modules())


def test_collection_bounds_do_not_pull_an_extra_batch_and_mask_before_forward():
    layer = nn.Linear(1, 1, bias=False).double()
    layer.weight.data.zero_()
    read_batches = []

    def batches():
        for index in range(10):
            read_batches.append(index)
            yield (torch.ones((3, 1), dtype=torch.float64), torch.ones((3, 1), dtype=torch.float64),
                   torch.tensor([False, True, True]))

    stats = collect(layer, batches(), max_examples=3, max_batches=10, stamp=passport(mask_id="skip-first"))
    assert read_batches == [0, 1]
    assert stats.passport.count == 3
    assert stats.collection_manifest["masked_examples"] == 2
    read_batches.clear()
    limited = collect(layer, batches(), max_examples=30, max_batches=1, stamp=passport(mask_id="skip-first"))
    assert read_batches == [0] and limited.passport.count == 2


def test_original_frozen_parameters_remain_frozen_and_grad_none():
    layer = nn.Linear(2, 1, bias=False).double().requires_grad_(False)
    stats = collect(layer, [(torch.ones((2, 2), dtype=torch.float64), torch.ones((2, 1), dtype=torch.float64))])
    assert stats.mean_square.shape == (1, 2)
    assert layer.weight.requires_grad is False and layer.weight.grad is None


@pytest.mark.parametrize("max_examples,max_batches", [(0, 1), (1, 0), (True, 2), (-1, 2)])
def test_collection_requires_explicit_positive_bounds(max_examples, max_batches):
    with pytest.raises(NumericalDomainViolation, match="limits"):
        collect(nn.Linear(1, 1), [], max_examples=max_examples, max_batches=max_batches)


def test_zero_observation_mass_and_test_labels_refuse():
    layer = nn.Linear(1, 1)
    with pytest.raises(NumericalDomainViolation, match="No unmasked"):
        collect(layer, [(torch.ones(2, 1), torch.ones(2, 1), torch.zeros(2, dtype=torch.bool))],
                stamp=passport(mask_id="mask"))
    with pytest.raises(ValueError, match="test/validation"):
        collect(layer, [], stamp=passport(data_role="test"))


def test_invalid_loss_and_batch_have_explicit_refusal():
    layer = nn.Linear(1, 1)
    with pytest.raises(NumericalDomainViolation, match="differentiable"):
        collect(layer, [(torch.ones(1, 1), torch.ones(1, 1))], loss_fn=lambda output, labels: torch.tensor(0.))
    with pytest.raises(NumericalDomainViolation, match="matching batch"):
        collect(layer, [(torch.ones(2, 1), torch.ones(1, 1))])
