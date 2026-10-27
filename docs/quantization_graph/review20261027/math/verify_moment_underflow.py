"""Targeted regression for the representable subnormal second moment."""
import json,math,hashlib
from fractions import Fraction
from pathlib import Path
import torch
from fedcore.algorithm.quantization.matrix_transforms import input_second_moment
from fedcore.algorithm.low_rank.approximation import prepare_weighted_metric
from fedcore.algorithm.low_rank.plans import MetricPolicy
torch.set_num_threads(1)
checks=[]
def record(name,passed,**details):checks.append({"name":name,"passed":bool(passed),**details})
small=2e-162
rows=torch.full((4,1),small,dtype=torch.float64)
before=rows.clone()
h=input_second_moment(rows,1)
raw=rows.T@rows/len(rows)
normalized=(rows/math.sqrt(len(rows))).T@(rows/math.sqrt(len(rows)))
correct=float(Fraction.from_float(small)**2)
record("tiny_mean_correct_rounding",float(h[0,0])==correct and correct>0.,actual=h.tolist(),exact_fraction_rounded=correct,previous_predivision=normalized.tolist(),legacy_raw=raw.tolist())
record("input_rows_unchanged",torch.equal(rows,before))
rows=torch.tensor([[1.,small],[1.,-small],[1.,small],[1.,-small]],dtype=torch.float64)
h=input_second_moment(rows,2)
metric=prepare_weighted_metric(h,MetricPolicy(ridge=0.,rcond=0.))
record("mixed_source_small_before_filter_rcond0",float(h[1,1])==correct and int(metric.support.sum())==2,actual=h.tolist(),eigenvalues=metric.eigenvalues.tolist(),supported=int(metric.support.sum()),rcond=0.)
rows=torch.tensor([[1e154,small],[1e154,-small],[1e154,small],[1e154,-small]],dtype=torch.float64)
raw_gram=rows.T@rows
h=input_second_moment(rows,2)
good=bool(torch.isfinite(h).all()) and float(h[0,0])==1e308 and float(h[1,1])==correct and torch.equal(h[torch.isfinite(raw_gram)],(raw_gram/len(rows))[torch.isfinite(raw_gram)])
record("overflow_and_tiny_same_gram",good,raw_gram=raw_gram.tolist(),actual=h.tolist())
minimum=float.fromhex("0x0.0000000000001p-1022")
rows=torch.tensor([[1e154,0.],[minimum,1e154],[0.,0.],[0.,0.]],dtype=torch.float64)
h=input_second_moment(rows,2)
raw=rows.T@rows/len(rows)
record("minimum_input_cross_preserved",torch.equal(h,raw) and float(h[0,1])>0.,actual=h.tolist(),legacy_raw=raw.tolist())
rows=torch.tensor([[1e154,1e154],[1e154,-1e154]],dtype=torch.float64)
h=input_second_moment(rows,2)
record("representable_large_mean",bool(torch.isfinite(h).all()) and bool(torch.equal(h,torch.diag(torch.full((2,),1e308,dtype=torch.float64)))),actual=h.tolist())
try:
    input_second_moment(torch.tensor([[1e308],[-1e308]],dtype=torch.float64),1)
    refused=False
except ValueError as e:refused="overflow" in str(e).lower()
record("unrepresentable_mean_refused",refused)
ordinary=True
for seed in range(20):
    x=torch.randn(8,4,dtype=torch.float64,generator=torch.Generator().manual_seed(seed))
    ordinary=ordinary and torch.equal(input_second_moment(x,4),x.T@x/len(x))
record("finite_legacy_entries_bitwise_20_matrices",ordinary)
path=Path(__file__).resolve()
out={"passed":sum(c["passed"] for c in checks),"total":len(checks),"checks":checks,"sha256_matrix_transforms":hashlib.sha256(Path("fedcore/algorithm/quantization/matrix_transforms.py").read_bytes()).hexdigest()}
path.with_name("verify_moment_underflow_results.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out,indent=2))
raise SystemExit(0 if out["passed"]==out["total"] else 1)
