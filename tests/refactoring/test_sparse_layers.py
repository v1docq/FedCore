from copy import deepcopy
import io
import pytest
import torch
from torch.nn import functional as F

from fedcore.algorithm.pruning.sparse.layers import PackedNMLinear, IndexedResidualLinear
from fedcore.algorithm.pruning.sparse.specs import SparseDomainError


def tensors(out=5, width=8, rank=2, dtype=torch.float64):
    generator = torch.Generator().manual_seed(31)
    return (torch.randn(out, rank, generator=generator, dtype=dtype),
            torch.randn(rank, width, generator=generator, dtype=dtype),
            torch.randn(out, generator=generator, dtype=dtype))


@pytest.mark.parametrize("shape", [(8,), (3, 8), (2, 3, 8), (0, 8)])
@pytest.mark.parametrize("rank", [0, 2])
def test_packed_nm_matches_independent_dense_and_input_gradient(shape, rank):
    left, right, bias = tensors(rank=rank)
    indices = torch.tensor([[[0, 2], [5, 7]]] * 5)
    values = torch.arange(20, dtype=torch.float64).reshape(5, 2, 2) / 10
    module = PackedNMLinear.from_tensors(values, indices, left, right, in_features=8, bias=bias)
    sparse = torch.zeros(5, 8, dtype=torch.float64)
    for row in range(5):
        for group in range(2):
            for slot in range(2):
                sparse[row, indices[row, group, slot]] = values[row, group, slot]
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    actual = module(x)
    expected = F.linear(expected_x, sparse + left @ right, bias)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, expected_x.grad)
    assert not any(tuple(t.shape) == (5, 8) for t in module.state_dict().values())
    assert module.values.numel() + left.numel() + right.numel() == 20 + 13 * rank


@pytest.mark.parametrize("support", [[], [4, 1], [0, 1, 2, 3, 4]])
@pytest.mark.parametrize("rank", [0, 2])
def test_indexed_residual_full_width_and_zero_sum_row(support, rank):
    left, right, bias = tensors(rank=rank)
    rows = torch.zeros(len(support), 8, dtype=torch.float64)
    if support:
        rows[0, :2] = torch.tensor([1., -1.])
    indices = torch.tensor(support, dtype=torch.int64)
    module = IndexedResidualLinear.from_tensors(left, right, rows, indices, bias=bias)
    full = torch.zeros(5, 8, dtype=torch.float64)
    full[indices] = rows
    for shape in ((2, 8), (1, 3, 8), (4, 2, 8)):
        x = torch.randn(shape, dtype=torch.float64)
        torch.testing.assert_close(module(x), F.linear(x, left @ right + full, bias))
        assert module(x).shape[-1] == 5
    assert module.residual_rows.numel() == len(support) * 8
    assert set(module.state_dict()) == ({"residual_rows", "row_indices", "bias"}
            | ({"factors.left.weight", "factors.right.weight"} if rank else set()))


@pytest.mark.parametrize("kind", ["nm", "indexed"])
def test_roundtrip_and_serialized_storage_is_compact(kind):
    left, right, bias = tensors()
    if kind == "nm":
        module = PackedNMLinear.from_tensors(torch.ones(5, 2, 2, dtype=torch.float64),
                    torch.tensor([[[0, 3], [4, 6]]] * 5), left, right, in_features=8, bias=bias)
    else:
        module = IndexedResidualLinear.from_tensors(left, right, torch.ones(2, 8, dtype=torch.float64),
                                                   torch.tensor([3, 0]), bias=bias)
    stream = io.BytesIO()
    torch.save({"config": module.representation_config(), "state": module.state_dict()}, stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=True)
    restored = type(module).from_config(saved["config"], dtype=torch.float64)
    restored.load_state_dict(saved["state"])
    x = torch.randn(3, 8, dtype=torch.float64)
    torch.testing.assert_close(module(x), restored(x), rtol=0, atol=0)
    expected_bytes = sum(t.numel() * t.element_size() for t in module.state_dict().values())
    assert expected_bytes == sum(t.untyped_storage().nbytes() for t in restored.state_dict().values())


@pytest.mark.parametrize("mutation", ["duplicate", "wrong_group", "nonfinite", "extra_weight"])
def test_bad_checkpoint_refused_before_mutating_live_module(mutation):
    module = PackedNMLinear(8, 5, rank=2, dtype=torch.float64)
    before = deepcopy(module.state_dict())
    state = deepcopy(before)
    if mutation == "duplicate":
        state["indices"][0, 0, 1] = state["indices"][0, 0, 0]
    elif mutation == "wrong_group":
        state["indices"][0, 0, 0] = 6
    elif mutation == "nonfinite":
        state["values"][0, 0, 0] = torch.nan
    else:
        state["weight"] = torch.ones(5, 8)
    with pytest.raises(SparseDomainError):
        module.load_state_dict(state)
    for name, tensor in before.items():
        assert torch.equal(module.state_dict()[name], tensor)


def test_prefixed_state_and_invalid_indexed_map():
    module = IndexedResidualLinear(8, 5, 1, 2, dtype=torch.float64)
    state = {"layer." + k: v for k, v in module.state_dict().items()}
    module.validate_state(state, prefix="layer.")
    state["layer.row_indices"] = torch.tensor([2, 2])
    with pytest.raises(SparseDomainError) as error:
        module.validate_state(state, prefix="layer.")
    assert error.value.code == "invalid_support"
