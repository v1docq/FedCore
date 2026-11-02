import sys,json,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch,numpy as np
from torch import nn
from dataclasses import replace
from fedcore.algorithm.quantization.tensor_formats import *
from fedcore.algorithm.quantization.graph_specs import *
from fedcore.algorithm.quantization.graph_execution import *
from fedcore.algorithm.quantization.matrix_transforms import scalar_ldlq
from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear
from fedcore.algorithm.low_rank.structured_profiles import FactorPair
torch.set_num_threads(2)
checks=[]
def run(name,fn):
 try:fn();checks.append({'name':name,'passed':True})
 except Exception as e:checks.append({'name':name,'passed':False,'error':traceback.format_exc()})
def endpoint_zero_axis():
 x=torch.tensor([[0.,-1.],[0.,1.]],dtype=torch.float64)
 q=PackedTensor.from_tensor(x,QuantizedTensorSpec(2,axis=1,grid=Grid.ENDPOINT))
 assert torch.equal(q.decode(),x)
def metric_scale():
 w=torch.tensor([[.4,.2,.8],[-.1,.6,-.9]],dtype=torch.float64)
 h=torch.tensor([[2.,.7,.3],[.7,1.5,.8],[.3,.8,2.]],dtype=torch.float64)
 a,_=scalar_ldlq(w,h,QuantizedTensorSpec(2,axis=0),generator=torch.Generator().manual_seed(2))
 b,_=scalar_ldlq(w,h*1e-18,QuantizedTensorSpec(2,axis=0),generator=torch.Generator().manual_seed(2))
 assert torch.equal(a.packed,b.packed)
def malformed():
 m=nn.Sequential(nn.Linear(3,2))
 for r in [QuantizationGraphRequest((OperatorRecipe('0',TransformSpec(Method.PTQ)),),target=None),QuantizationGraphRequest(({},))]:
  assert isinstance(check_quantization_graph(m,r),ValidationFailure)
 assert isinstance(check_quantization_graph(m,QuantizationGraphRequest((OperatorRecipe('0',TransformSpec(Method.PTQ)),)),calibration={}),ValidationFailure)
def root_scope():
 m=FactorizedLinear.from_factors(FactorPair(torch.ones(3,2),torch.ones(2,4),{}),None)
 r=QuantizationGraphRequest((OperatorRecipe('',TransformSpec(Method.FACTORS)),OperatorRecipe('left',TransformSpec(Method.PTQ))))
 assert isinstance(check_quantization_graph(m,r),ValidationFailure)
def numpy_prefix():
 w=np.array([[.4,.8,-.2,1.2],[.9,-.3,.7,.1],[-1.,.25,.15,.6]],dtype=np.float64)
 m=nn.Sequential(nn.Linear(4,3,bias=False).double());m[0].weight.data.copy_(torch.from_numpy(w))
 x=torch.tensor([[1.,2.,-1.,.5]],dtype=torch.float64)
 spec=TransformSpec(Method.SVD,rank=1,prequantize=QuantizedTensorSpec(2))
 state=torch.random.get_rng_state().clone()
 out=compress_quantization_graph(m,QuantizationGraphRequest((OperatorRecipe('0',spec),)),calibration=x)
 assert type(out) is ExportVerifiedArtifact,out
 assert torch.equal(state,torch.random.get_rng_state())
 scale=np.max(np.abs(w));q=np.clip(np.rint(w/scale),-1,1)*scale
 u,s,v=np.linalg.svd(q,full_matrices=False);expected=(u[:,:1]*s[:1])@v[:1]
 np.testing.assert_allclose(out.model(x).detach().numpy(),x.numpy()@expected.T,rtol=1e-10,atol=1e-10)
for name,fn in [('endpoint_zero_channel',endpoint_zero_axis),('ldlq_metric_scale_invariance',metric_scale),('typed_invalid_inputs',malformed),('root_child_overlap',root_scope),('numpy_graph_prefix_and_rng',numpy_prefix)]:run(name,fn)
report={'checks':checks,'passed':all(c['passed'] for c in checks),'scope':'Independent small CPU numerical/boundary probes; no quality/latency study'}
output=Path(__file__).resolve().parent/'results'/'root_adversarial_check.json'
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2));sys.exit(0 if report['passed'] else 1)
