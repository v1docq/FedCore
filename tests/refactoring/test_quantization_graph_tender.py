import pytest
import torch
from torch import nn
from fedcore.algorithm.quantization.tender import TenderLinear
from fedcore.algorithm.quantization.graph_execution import *
from fedcore.algorithm.quantization.graph_specs import *


def test_tender_integer_and_dequantized_group_references_roundtrip():
    rows = torch.tensor([[-8., -1., -.01,3.], [8.,1.,.01,3.], [4.,.2,.005,3.]],dtype=torch.float64)
    weight = torch.tensor([[.1,-.2,.3,.4],[-.7,.2,.1,-.1]],dtype=torch.float64)
    bias = torch.tensor([1.,-2.],dtype=torch.float64)
    layer = TenderLinear.calibrate(weight,bias,rows,activation_bits=4)
    assert (layer.exponents[:-1]-layer.exponents[1:]).max() > 1  # skipped levels
    xc,wc = layer.activation_codes(rows),layer.weight.codes()
    integer = torch.zeros(len(rows),2,dtype=torch.int64)
    dequantized = torch.zeros(len(rows),2,dtype=torch.float64)
    final = int(layer.exponents[-1])
    for g,e in enumerate(layer.exponents.tolist()):
        ids = layer.activation_indices[layer.group_ptr[g]:layer.group_ptr[g+1]]
        p = xc[:,ids]@wc[:,ids].T
        integer += p*2**(e-final)
        dequantized += p.double()*(layer.activation_scale[0]*2**e)*layer.weight.scale
    assert torch.equal(integer,layer.integer_accumulator(rows))
    torch.testing.assert_close(layer(rows),dequantized+layer.weight.decode()@layer.centers+bias,atol=1e-12,rtol=1e-12)
    source = nn.Sequential(nn.Linear(4,2).double())
    source[0].weight.data.copy_(weight); source[0].bias.data.copy_(bias)
    artifact = compress_quantization_graph(source,QuantizationGraphRequest((OperatorRecipe("0",TransformSpec(Method.TENDER,activation_bits=4)),)),calibration=rows)
    assert isinstance(artifact,ExportVerifiedArtifact),artifact
    torch.testing.assert_close(artifact.model(rows),layer(rows),atol=0,rtol=0)
    assert artifact.model[0].config()["profile"] == "fixed_global_channel_linear"
    assert artifact.model[0](rows[:1].reshape(1,1,4)).shape == (1,1,2)


def test_zero_channel_ranges_and_center_identity():
    rows = torch.tensor([[3.,4.],[3.,4.]],dtype=torch.float64)
    weight = torch.tensor([[.5,-.25]],dtype=torch.float64)
    layer = TenderLinear.calibrate(weight,None,rows)
    assert torch.equal(layer.activation_codes(rows),torch.zeros(2,2,dtype=torch.int64))
    torch.testing.assert_close(layer(rows),rows@layer.weight.decode().T)
    with pytest.raises(ValueError):
        layer(torch.ones(2,3))


def test_tender_max_anchor_and_exact_half_threshold():
    maxima = torch.tensor([10.,5.,3.,1.,0.],dtype=torch.float64)
    layer = TenderLinear.calibrate(torch.ones(1,5,dtype=torch.float64),None,torch.stack((-maxima,maxima)),activation_bits=4)
    actual = torch.zeros(5,dtype=torch.float64)
    for g,e in enumerate(layer.exponents.tolist()):
        ids = layer.activation_indices[layer.group_ptr[g]:layer.group_ptr[g+1]]
        actual[ids] = layer.activation_scale[0]*2**e
    torch.testing.assert_close(actual,torch.tensor([10/7,5/7,5/7,1.25/7,1.25/7],dtype=torch.float64))
    # A minimum-anchored profile would give scale 8/7 to the first channel.
    assert float(actual[0]) != 8/7


def test_overflow_is_checked_before_torch_int64_wrap():
    rows = torch.tensor([[-1.,-1.,-1.],[1.,1.,1.]],dtype=torch.float64)
    layer = TenderLinear.calibrate(torch.ones(1,3,dtype=torch.float64),None,rows,accumulator_bits=16)
    with pytest.raises(OverflowError,match="intermediate"):
        layer(rows)


@pytest.mark.parametrize("corruption", ["indices","empty","center"])
def test_corrupt_tender_metadata_refused(corruption):
    layer = TenderLinear.calibrate(torch.ones(2,3,dtype=torch.float64),None,torch.randn(8,3,dtype=torch.float64))
    state = {key:value.clone() for key,value in layer.state_dict().items()}
    if corruption == "indices":state["weight_indices"][0] = state["weight_indices"][1]
    if corruption == "empty":state["group_ptr"][1] = 0
    if corruption == "center":state["center_correction"][0] += 1
    with pytest.raises(ValueError):layer.validate_state(state)
