"""Independent small CPU references; raw outputs are regenerated beside this script."""
import json, subprocess, types, sys
from pathlib import Path
import torch
from fedcore.algorithm.quantization.matrix_transforms import scalar_ldl,scalar_ldlq
from fedcore.algorithm.quantization.tensor_formats import QuantizedTensorSpec,PackedTensor
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parent
REPO=Path(__file__).resolve().parents[4]
HEAD="50f1cdf51b9f65ea8a5d2b868b7ed46a35738f4c"
source=subprocess.check_output(["git","show",f"{HEAD}:fedcore/algorithm/quantization/matrix_transforms.py"],cwd=REPO)
(ROOT/"baseline_matrix_transforms.py.txt").write_bytes(source)
baseline=types.ModuleType("fedcore.algorithm.quantization.baseline_matrix_transforms")
baseline.__package__="fedcore.algorithm.quantization"
sys.modules[baseline.__name__]=baseline
exec(compile(source,str(ROOT/"baseline_matrix_transforms.py.txt"),"exec"),baseline.__dict__)
checks=[]
def check(name,h,expected_failure=False):
    try:
        l,d=scalar_ldl(h,1e-10)
        scaled=l*d.sqrt()
        err=float((scaled@scaled.T-h).abs().max())/max(float(h.abs().max()),1e-320)
        good=not expected_failure and bool(torch.isfinite(l).all()) and bool(torch.isfinite(d).all()) and err<1e-12
        result={"passed":good,"relative_backward_error":err,"minimum_d":float(d.min())}
    except Exception as e:
        result={"passed":expected_failure,"error":f"{type(e).__name__}: {e}"}
    checks.append({"name":name,**result})
    return result
for exponent in [-150,-100,-50,-12,-6,-4,0,4,6,12,50,100,150]:
    v=torch.tensor([10.**exponent,1.],dtype=torch.float64)
    for reverse in [False,True]:
        vv=v.flip(0) if reverse else v
        check(f"rank1_coordinate_1e{exponent}_reverse{reverse}",torch.outer(vv,vv))
for scale in [1e-200,1e-100,1.,1e100,1e200]:
    v=torch.tensor([1e-6,1.,.3],dtype=torch.float64)
    check(f"rank1_metric_scale_{scale}",torch.outer(v,v)*scale)
check("zero_metric",torch.zeros(3,3,dtype=torch.float64))
check("leading_zero_coordinate",torch.diag(torch.tensor([0.,1.,2.],dtype=torch.float64)))
check("negative_metric",torch.tensor([[1.,2.],[2.,1.]],dtype=torch.float64),True)
check("non_symmetric",torch.tensor([[1.,.2],[.1,1.]],dtype=torch.float64),True)
# Deterministic PSD probes, well-conditioned retained support.
for seed in range(20):
    gen=torch.Generator().manual_seed(seed)
    for rank in [1,2,5]:
        b=torch.randn(5,rank,dtype=torch.float64,generator=gen)
        scale=torch.tensor([1e-6,1e-3,1.,1e3,1e6],dtype=torch.float64)
        b=b*scale[:,None]
        h=b@b.T
        check(f"gram_seed_{seed}_rank_{rank}",h)
# Independent Cholesky/coordinate reference for SPD scalar LDLQ.
ldlq_checks=[]
for seed in range(20):
    gen=torch.Generator().manual_seed(seed)
    b=torch.randn(4,4,dtype=torch.float64,generator=gen)
    h=b@b.T+.2*torch.eye(4,dtype=torch.float64)
    w=torch.randn(3,4,dtype=torch.float64,generator=gen)
    spec=QuantizedTensorSpec(4,axis=0,bound=4.)
    q,_=scalar_ldlq(w,h,spec,generator=gen)
    chol=torch.linalg.cholesky(h)
    unit=chol/chol.diagonal()[None,:]
    ref=torch.zeros_like(w)
    for j in range(3,-1,-1):
        for i in range(3):
            t=float(w[i,j])+sum(float(w[i,k]-ref[i,k])*float(unit[k,j]) for k in range(j+1,4))
            ref[i,j]=max(-7,min(7,round(t/(4./7))))*(4./7)
    ldlq_checks.append({"seed":seed,"passed":bool(torch.equal(ref,q.decode())),"max_error":float((ref-q.decode()).abs().max())})
# Baseline wrong-support demonstration and current result.
a=1e-12;h=torch.tensor([[a*a,a],[a,1.]],dtype=torch.float64)
w=torch.tensor([[4e11,.5]],dtype=torch.float64);spec=QuantizedTensorSpec(2,axis=0,bound=1e12)
old,_=baseline.scalar_ldlq(w,h,spec,generator=torch.Generator().manual_seed(0))
new,_=scalar_ldlq(w,h,spec,generator=torch.Generator().manual_seed(0))
direction=torch.tensor([[a],[1.]],dtype=torch.float64)
loss=lambda q:float(((w-q)@direction).square().sum())
result={"baseline_head":HEAD,"ldl_checks":checks,"ldl_passed":sum(x["passed"] for x in checks),"ldl_total":len(checks),
"ldlq_checks":ldlq_checks,"ldlq_passed":sum(x["passed"] for x in ldlq_checks),
"baseline_silent_support":{"baseline_q":old.decode().tolist(),"new_q":new.decode().tolist(),"baseline_loss":loss(old.decode()),"new_loss":loss(new.decode())}}
(ROOT/"verify_ldl_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps({"ldl_passed":result["ldl_passed"],"ldl_total":result["ldl_total"],"ldl_failures":[x for x in checks if not x["passed"]],"ldlq_passed":result["ldlq_passed"],"ldlq_total":len(ldlq_checks),"ldlq_failures":[x for x in ldlq_checks if not x["passed"]],"baseline_silent_support":result["baseline_silent_support"]},indent=2))
