"""Bounded sequential Transformer FFN profile; attention is not factorized."""
from copy import deepcopy
import pytest
import torch
from torch import nn
from fedcore.algorithm.low_rank.method_execution import transform_method
from fedcore.algorithm.low_rank.method_specs import DRONE,SVDLLMV1,SVDLLMV2


@pytest.mark.parametrize('spec',[DRONE(),SVDLLMV1(),SVDLLMV2()])
def test_transformer_current_graph_ffn(spec,tmp_path):
    torch.manual_seed(64)
    model=nn.Sequential(nn.TransformerEncoderLayer(d_model=8,nhead=2,
        dim_feedforward=12,dropout=0,batch_first=True)).double().eval()
    original=deepcopy(model.state_dict())
    calibration=torch.randn(6,3,8,dtype=torch.double)
    result=transform_method(model,calibration,spec,rank=2,
        target_paths=('0.linear1','0.linear2'),batch_size=2)
    layers=result.evidence['layers']
    assert layers[0]['predecessor_version']!=layers[1]['predecessor_version']
    assert layers[0]['observations']==layers[1]['observations']==18
    assert result.evidence['parameters_after']<result.evidence['parameters_before']
    assert torch.isfinite(result.model(calibration)).all()
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,original[key],rtol=0,atol=0)
    # Export the actual sequential FFN graph, including unchanged attention.
    assert model[0].activation_relu_or_gelu == 1
    assert result.model[0].activation_relu_or_gelu == 0
    with torch.no_grad():
        traced=torch.jit.trace(result.model.eval(),calibration[:1])
    artifact=tmp_path/'transformer.pt'
    torch.jit.save(traced,str(artifact))
    restored=torch.jit.load(str(artifact))
    torch.testing.assert_close(restored(calibration[:1]),result.model(calibration[:1]))
