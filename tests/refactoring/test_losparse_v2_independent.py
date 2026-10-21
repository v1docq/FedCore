"""Independent derivative oracle for LoSparse's numerical repair."""
import pytest
import torch
from torch import nn

from fedcore.algorithm.pruning.sparse.linear import prepare_losparse, finalize_losparse
from fedcore.algorithm.pruning.sparse.specs import LoSparseSpec


@pytest.mark.parametrize("rank", [0, 2])
@pytest.mark.parametrize("rows", [0, 3, 6])
@pytest.mark.parametrize("scale", [.3, 2.])
@pytest.mark.parametrize("bias", [False, True])
def test_finalized_operator_matches_completed_training_exactly(rank, rows, scale, bias):
    torch.manual_seed(97)
    layer = nn.Linear(8, 6, bias=bias)
    module = prepare_losparse(layer, LoSparseSpec(rank, rows, 8, initial_scale=scale))
    assert module.residual.dtype == torch.float64
    assert module.output_dtype == torch.float32
    with torch.no_grad():
        module.support.zero_()
        module.support[:rows] = True
        module.completed_steps.fill_(8)
        module.residual.add_(.0000123456789)
    x = torch.randn(4, 8)
    expected = module(x)
    final = finalize_losparse(module)
    assert all(value.dtype == torch.float32 for value in final.parameters())
    assert final.residual_count == rows
    assert not any("ema" in name or name == "residual" for name in final.state_dict())
    torch.testing.assert_close(final(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("scale", [.3, 2.])
@pytest.mark.parametrize("rank", [0, 2])
def test_initial_full_support_recovers_live_source_weight(scale, rank):
    torch.manual_seed(11)
    layer = nn.Linear(8, 6)
    module = prepare_losparse(layer, LoSparseSpec(rank, 3, 8, initial_scale=scale))
    x = torch.randn(100, 8)
    torch.testing.assert_close(module(x), layer(x), rtol=0, atol=0)
    assert set(module.state_dict()) == ({"residual", "bias", "support", "importance_ema", "completed_steps"}
                                      | ({"factors.left.weight", "factors.right.weight"} if rank else set()))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('reentry', [False, True])
@pytest.mark.parametrize('full_support', [False, True])
def test_training_gradients_match_explicit_matrix_derivatives(dtype, reentry, full_support):
    """Forward must retain gradients of both factors, residual and input."""
    generator = torch.Generator().manual_seed(901)
    layer = nn.Linear(8, 6, dtype=dtype)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(6, 8, generator=generator, dtype=dtype))
        layer.bias.copy_(torch.randn(6, generator=generator, dtype=dtype))
    module = prepare_losparse(layer, LoSparseSpec(
        2, 3, 4, initial_scale=.3, allow_reentry=reentry))
    with torch.no_grad():
        module.support.copy_(torch.tensor([True] * 6 if full_support else [True, False, True, False, True, False]))
    x = torch.randn(5, 8, generator=generator, dtype=dtype, requires_grad=True)
    upstream = torch.randn(5, 6, generator=generator, dtype=dtype)
    left = module.factors.left.weight.detach().double()
    right = module.factors.right.weight.detach().double()
    mask = module.support[:, None].double()
    effective = left @ right + .3 * module.residual.detach().double() * mask
    weight_gradient = upstream.double().T @ x.detach().double()
    expected = {
        'left': weight_gradient @ right.T,
        'right': left.T @ weight_gradient,
        'residual': .3 * weight_gradient * (1 if reentry else mask),
        'input': upstream.double() @ effective,
        'bias': upstream.double().sum(0),
    }
    output = module(x)
    (output * upstream).sum().backward()
    tolerance = dict(rtol=3e-5, atol=3e-6) if dtype == torch.float32 else dict(rtol=1e-12, atol=1e-12)
    actual = {
        'left': module.factors.left.weight.grad,
        'right': module.factors.right.weight.grad,
        'residual': module.residual.grad,
        'input': x.grad,
        'bias': module.bias.grad,
    }
    torch.testing.assert_close(output.double(), x.detach().double() @ effective.T + module.bias.detach().double(), **tolerance)
    for name, gradient in actual.items():
        assert gradient is not None
        torch.testing.assert_close(gradient.double(), expected[name], **tolerance)


def test_training_state_owns_no_live_source_weight():
    layer = nn.Linear(8, 6, dtype=torch.float64)
    module = prepare_losparse(layer, LoSparseSpec(2, 3, 4))
    x = torch.arange(24, dtype=torch.float64).reshape(3, 8) / 10
    expected = module(x).detach().clone()
    with torch.no_grad():
        layer.weight.fill_(200)
        layer.bias.fill_(-70)
    torch.testing.assert_close(module(x), expected, rtol=0, atol=0)
    source_storages = {value.untyped_storage().data_ptr() for value in layer.parameters()}
    assert not source_storages.intersection(value.untyped_storage().data_ptr() for value in module.parameters())
