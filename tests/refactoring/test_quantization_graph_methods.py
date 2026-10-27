from dataclasses import replace
import pytest
import torch
from torch import nn
from fedcore.algorithm.quantization.graph_specs import *
from fedcore.algorithm.quantization.tensor_formats import *
from fedcore.algorithm.quantization.matrix_transforms import *
from fedcore.algorithm.quantization.graph_execution import *


def lplr_spec(method=Method.LPLR, rank=3):
    return TransformSpec(method, rank=rank, comparison_rank=2,
        left=QuantizedTensorSpec(4, rounding=Rounding.STOCHASTIC, grid=Grid.ENDPOINT),
        right=QuantizedTensorSpec(8, rounding=Rounding.STOCHASTIC, grid=Grid.ENDPOINT), seed=17)


@pytest.mark.parametrize("method", [Method.LPLR, Method.LPLR_SVD])
@pytest.mark.parametrize("zero", [False, True])
def test_lplr_projection_quantized_design_and_reproducible_roundtrip(method, zero):
    torch.manual_seed(12)
    w = torch.randn(6, 5, dtype=torch.float64) if not zero else torch.zeros(6,5,dtype=torch.float64)
    spec = lplr_spec(method)
    result = lplr(w, spec, torch.Generator().manual_seed(spec.seed))
    repeated = lplr(w, spec, torch.Generator().manual_seed(spec.seed))
    assert torch.equal(result.left.packed, repeated.left.packed)
    assert torch.equal(result.right.packed, repeated.right.packed)
    expected = torch.linalg.pinv(result.left.decode(), rcond=spec.rcond)@w
    torch.testing.assert_close(result.fitted_right, expected, atol=1e-12, rtol=1e-12)
    assert result.diagnostics["factor_width_m"] == 3
    assert result.diagnostics["comparison_rank_k"] == 2
    if method is Method.LPLR:
        generator = torch.Generator().manual_seed(spec.seed)
        projection = torch.randn(5,3,dtype=torch.float64,generator=generator)/(3**.5)
        assert torch.equal(result.projection, projection)
        expected_left = PackedTensor.from_tensor(w@projection, spec.left, generator=generator)
        assert torch.equal(result.left.packed, expected_left.packed)
    source = nn.Sequential(nn.Linear(5,6).double())
    source[0].weight.data.copy_(w)
    artifact = compress_quantization_graph(source, QuantizationGraphRequest((OperatorRecipe("0", spec),)), calibration=torch.randn(4,5,dtype=torch.float64))
    assert isinstance(artifact, ExportVerifiedArtifact), artifact
    assert torch.equal(artifact.model[0].left.packed, result.left.packed)


def test_scalar_ldlq_reference_transfers_error():
    w = torch.tensor([[.4, .2, .8], [-.1, .6, -.9]],dtype=torch.float64)
    h = torch.tensor([[2., .7, .3], [.7, 1.5, .8], [.3, .8, 2.]],dtype=torch.float64)
    spec = QuantizedTensorSpec(2, axis=0)
    result, diagnostics = scalar_ldlq(w,h,spec,generator=torch.Generator().manual_seed(1))
    chol = torch.linalg.cholesky(h)
    l = chol/chol.diagonal()[None,:]
    expected = torch.zeros_like(w)
    for j in reversed(range(3)):
        for row in range(2):
            adjusted = float(w[row,j]+sum((w[row,i]-expected[row,i])*l[i,j] for i in range(j+1,3)))
            expected[row,j] = max(-1, min(1, round(adjusted/float(result.scale[row]))))*result.scale[row]
    assert torch.equal(result.decode(), expected)
    assert sum(diagnostics["feedback_squared"]) > 0
    plain = PackedTensor.from_tensor(w,spec)
    assert not torch.equal(plain.packed,result.packed)
    reconstructed, d = scalar_ldl(h,1e-10)
    torch.testing.assert_close(reconstructed@torch.diag(d)@reconstructed.T,h)


@pytest.mark.parametrize("singular", [False, True])
def test_caldera_best_bias_bytes_and_export(singular):
    torch.manual_seed(3)
    source = nn.Sequential(nn.Linear(5,4).double())
    rows = torch.randn(20,5,dtype=torch.float64)
    if singular:
        rows[:,2:] = 0
    spec = TransformSpec(Method.CALDERA,rank=2,left=QuantizedTensorSpec(4),right=QuantizedTensorSpec(4),weight=QuantizedTensorSpec(2,axis=0))
    result = compress_quantization_graph(source,QuantizationGraphRequest((OperatorRecipe("0",spec),)),calibration=rows)
    assert isinstance(result,ExportVerifiedArtifact),result
    details = result.evidence["operators"][0]
    assert details["best_error"] <= details["initial_error"]
    assert details["best_error"] == min(details["candidate_errors"])
    layer = result.model[0]
    expected = rows@(layer.base.weight.decode()+layer.correction.left.decode()@layer.correction.right.decode()).T+source[0].bias
    torch.testing.assert_close(layer(rows),expected,atol=1e-12,rtol=1e-12)
    assert details["metric_support"] == (2 if singular else 5)


def test_endpoint_grid_has_exact_number_of_nodes_and_saturation():
    spec = QuantizedTensorSpec(2,bound=1.,rounding=Rounding.STOCHASTIC,grid=Grid.ENDPOINT)
    values = torch.tensor([[-2., -1., -1/3, 1/3, 1., 2.]],dtype=torch.float64)
    q = PackedTensor.from_tensor(values,spec,generator=torch.Generator().manual_seed(0))
    assert q.codes().tolist() == [[0,0,1,2,3,3]]
    torch.testing.assert_close(q.decode(),values.clamp(-1,1))
