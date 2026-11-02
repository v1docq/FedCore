import json, math
from pathlib import Path
import torch
from fedcore.algorithm.quantization.tensor_formats import *
from fedcore.algorithm.quantization.matrix_transforms import lplr,input_second_moment,caldera
from fedcore.algorithm.quantization.portable_layers import *
from fedcore.algorithm.quantization.graph_specs import *
torch.set_num_threads(1)
checks=[]
def rec(name,value,**detail):checks.append({"name":name,"passed":bool(value),**detail})
# Exact grid, saturation, homogeneity and independent bit reconstruction.
for bits in (2,4,8):
    spec=QuantizedTensorSpec(bits,bound=1.,grid=Grid.ENDPOINT,rounding=Rounding.STOCHASTIC)
    nodes=torch.linspace(-1.,1.,2**bits,dtype=torch.float64).unsqueeze(0)
    q=PackedTensor.from_tensor(nodes,spec,generator=torch.Generator().manual_seed(5))
    rec(f"endpoint_nodes_{bits}",bool(torch.allclose(q.decode(),nodes,atol=2e-16,rtol=0)))
    val=torch.tensor([[-2.,2.]],dtype=torch.float64)
    sat=PackedTensor.from_tensor(val,spec,generator=torch.Generator().manual_seed(5))
    rec(f"endpoint_saturation_{bits}",sat.codes().tolist()==[[0,2**bits-1]])
    raw=q.codes().flatten().tolist()
    packed=bytearray((len(raw)*bits+7)//8)
    for i,c in enumerate(raw):packed[i*bits//8]|=c<<(i*bits%8)
    rec(f"packing_reference_{bits}",list(packed)==q.packed.tolist())
for axis in (None,0,1):
    for grid in Grid:
        spec=QuantizedTensorSpec(4,axis=axis,rounding=Rounding.STOCHASTIC,grid=grid)
        w=torch.tensor([[0.,0.,0.],[.4,-.2,.8]],dtype=torch.float64)
        q=PackedTensor.from_tensor(w,spec,generator=torch.Generator().manual_seed(10))
        scaled=PackedTensor.from_tensor(w*2.**20,spec,generator=torch.Generator().manual_seed(10))
        rec(f"quantization_homogeneity_{grid.value}_{axis}",torch.equal(q.packed,scaled.packed) and torch.equal(q.decode()*2.**20,scaled.decode()))
        z=PackedTensor.from_tensor(torch.zeros(3,2,dtype=torch.float64),spec,generator=torch.Generator().manual_seed(10))
        rec(f"quantization_zero_{grid.value}_{axis}",bool(torch.equal(z.decode(),torch.zeros(3,2,dtype=torch.float64))))
# Stochastic unbiasedness at a fixed unsaturated scalar, 20000 independent draws.
n=20000;v=.15
spec=QuantizedTensorSpec(2,bound=1.,grid=Grid.ENDPOINT,rounding=Rounding.STOCHASTIC)
q=PackedTensor.from_tensor(torch.full((1,n),v,dtype=torch.float64),spec,generator=torch.Generator().manual_seed(71))
mean=float(q.decode().mean());se=(2./3)/2/math.sqrt(n)
rec("stochastic_mean_fixed_value",abs(mean-v)<6*se,mean=mean,target=v,six_standard_error_bound=6*se)
# LPLR independent projection and pseudoinverse laws across zero/rank deficient/general.
for method in (Method.LPLR,Method.LPLR_SVD):
    for kind in ("zero","rank1","general"):
        gen=torch.Generator().manual_seed(7)
        w=torch.randn(5,4,dtype=torch.float64,generator=gen)
        if kind=="zero":w.zero_()
        if kind=="rank1":w=w[:,:1]@torch.tensor([[1.,2.,-1.,.5]],dtype=torch.float64)
        factor=QuantizedTensorSpec(4,rounding=Rounding.STOCHASTIC,grid=Grid.ENDPOINT)
        spec=TransformSpec(method,rank=3,comparison_rank=1,left=factor,right=factor,seed=33)
        result=lplr(w,spec,torch.Generator().manual_seed(spec.seed))
        reference=torch.linalg.pinv(result.left.decode(),rcond=spec.rcond)@w
        rec(f"lplr_fit_{method.value}_{kind}",bool(torch.allclose(result.fitted_right,reference,atol=1e-12,rtol=1e-12)))
# Sum representation: one final cast, bias after branch cancellation.
base=PortableQuantizedLinear.from_weight(torch.tensor([[2.**30]],dtype=torch.float64),torch.tensor([1.],dtype=torch.float64))
correction=PortableQuantizedFactorLinear.from_factors(torch.tensor([[-2.**15]],dtype=torch.float64),torch.tensor([[2.**15]],dtype=torch.float64))
x=torch.ones(1,1,dtype=torch.float32)
rec("sum_float32_bias",float(QuantizedSumLinear(base,correction)(x)[0,0])==1.)
base=PortableQuantizedLinear.from_weight(torch.tensor([[2.**80]],dtype=torch.float64))
correction=PortableQuantizedFactorLinear.from_factors(torch.tensor([[-2.**40]],dtype=torch.float64),torch.tensor([[2.**40]],dtype=torch.float64))
x=torch.full((1,1),2.**60,dtype=torch.float32)
rec("sum_false_overflow",float(QuantizedSumLinear(base,correction)(x)[0,0])==0.)
out={"passed":sum(c["passed"] for c in checks),"total":len(checks),"checks":checks}
Path(__file__).with_name("verify_formats_lplr_results.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps({"passed":out["passed"],"total":out["total"],"failures":[c for c in checks if not c["passed"]]},indent=2))
