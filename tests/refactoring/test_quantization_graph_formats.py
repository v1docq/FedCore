import math
import pytest
import torch
from torch import nn
from fedcore.algorithm.quantization.tensor_formats import (Grid, Rounding, QuantizedTensorSpec,
    PackedTensor, pack_codes, unpack_codes)
from fedcore.algorithm.quantization.portable_layers import PortableQuantizedFactorLinear
from fedcore.algorithm.quantization.graph_specs import *
from fedcore.algorithm.quantization.graph_execution import *


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("shape", [(1, 1), (3, 5), (8, 7)])
def test_pack_independent_bit_reference(bits, shape):
    codes = torch.arange(math.prod(shape)).reshape(shape) % (2**bits)
    actual = pack_codes(codes, bits)
    reference = bytearray((codes.numel()*bits+7)//8)
    for i, code in enumerate(codes.flatten().tolist()):
        reference[i*bits//8] |= code << (i*bits%8)
    assert actual.tolist() == list(reference)
    assert torch.equal(unpack_codes(actual, bits, codes.numel()), codes.flatten())


@pytest.mark.parametrize("grid", list(Grid))
@pytest.mark.parametrize("bits", [2, 4, 8])
def test_grid_and_zero_range(grid, bits):
    spec = QuantizedTensorSpec(bits, grid=grid)
    z = PackedTensor.from_tensor(torch.zeros(3, 5), spec)
    assert torch.equal(z.decode(), torch.zeros(3, 5, dtype=torch.float64))
    x = torch.tensor([[-2., -0.3, 0., .1, 2.]])
    q = PackedTensor.from_tensor(x, spec)
    assert (q.decode()-x).abs().max() <= q.scale.max()/2+1e-14
    assert q.packed.numel() == (x.numel()*bits+7)//8
    assert QuantizedTensorSpec.parse(spec.payload()) == spec


def test_arbitrary_factors_are_encoded_without_svd():
    l = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    r = torch.tensor([[2., 0., 3., 8.], [1., 7., 3., 2.]])
    layer = PortableQuantizedFactorLinear.from_factors(l, r, torch.ones(3),
        left_spec=QuantizedTensorSpec(4), right_spec=QuantizedTensorSpec(8))
    x = torch.randn(2, 4)
    expected = x.double() @ layer.right.decode().T @ layer.left.decode().T + 1
    torch.testing.assert_close(layer(x), expected.float())


def test_two_linear_export_source_unchanged_and_g5_inventory():
    torch.manual_seed(10)
    source = nn.Sequential(nn.Linear(7, 6), nn.ReLU(), nn.Linear(6, 3))
    before = {k: v.clone() for k, v in source.state_dict().items()}
    request = QuantizationGraphRequest((OperatorRecipe("0", TransformSpec(Method.PTQ, weight=QuantizedTensorSpec(4))),
        OperatorRecipe("2", TransformSpec(Method.DSVD, rank=2, left=QuantizedTensorSpec(2), right=QuantizedTensorSpec(8)))), max_peak_bytes=12*1024**3)
    x = torch.randn(4, 7)
    result = compress_quantization_graph(source, request, calibration=x)
    assert isinstance(result, ExportVerifiedArtifact), result
    assert result.storage_bytes == sum(v.untyped_storage().nbytes() for v in result.model.state_dict().values())
    for key, value in before.items():
        assert torch.equal(source.state_dict()[key], value)
    assert result.model[0].weight.spec.bits == 4
    assert result.model[2].left.spec.bits == 2
    from fedcore.tools.registry.checkpoint_manager import CheckpointManager
    loaded = CheckpointManager(None).deserialize_from_bytes(result.checkpoint_bytes)
    assert torch.equal(loaded(x), result.model(x))


def test_invalid_plan_rejected_before_forward():
    source = nn.Sequential(nn.Linear(4, 3))
    source.register_forward_pre_hook(lambda *args: pytest.fail("unexpected forward"))
    result = check_quantization_graph(source, QuantizationGraphRequest((OperatorRecipe("0", TransformSpec(Method.DSVD, rank=9)),)))
    assert isinstance(result, ValidationFailure)


def test_corrupt_padding_and_scale_rejected():
    q = PackedTensor.from_tensor(torch.ones(1, 1), QuantizedTensorSpec(2))
    state = q.state_dict()
    state["packed"] = torch.tensor([255], dtype=torch.uint8)
    with pytest.raises(ValueError):
        q.validate_state(state)
    with pytest.raises(ValueError):
        PackedTensor.from_tensor(torch.ones(2, 3), scale=torch.ones(3, dtype=torch.float64))
