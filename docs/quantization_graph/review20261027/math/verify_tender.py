"""Independent Tender boundary, integer recurrence, and float64-scale checks."""
import json, math
from decimal import Decimal, localcontext
from pathlib import Path
import torch
from fedcore.algorithm.quantization.tender import TenderLinear
from fedcore.algorithm.quantization.tensor_formats import PackedTensor
torch.set_num_threads(1)
checks=[]
def record(name,passed,**info):
    checks.append({"name":name,"passed":bool(passed),**info})
def channel_scales(layer):
    result=torch.empty(layer.in_features,dtype=torch.float64)
    for g,e in enumerate(layer.exponents.tolist()):
        ids=layer.activation_indices[layer.group_ptr[g]:layer.group_ptr[g+1]]
        result[ids]=layer.activation_scale[0]*2**e
    return result
for maximum in [49.,98.,103.,1e-300,1e-309,1e-310,1e308]:
    rows=torch.tensor([[-maximum],[0.],[maximum]],dtype=torch.float64)
    l=TenderLinear.calibrate(torch.ones(1,1,dtype=torch.float64),None,rows)
    expected=maximum/127
    record(f"maximum_anchor_{maximum}",float(channel_scales(l)[0])==expected and l.activation_codes(rows).flatten().tolist()==[-127,0,127],actual_scale=float(channel_scales(l)[0]),expected_scale=expected,codes=l.activation_codes(rows).flatten().tolist())
# Compare classification to exact binary-float inequalities with Python Decimal.
for maximum in [1.,49.,103.,1e-200,1e200]:
    values=[maximum,0.]
    for level in [1,2,7,20,59,60]:
        threshold=maximum/(2.**level)
        values.extend([math.nextafter(threshold,0.),threshold,math.nextafter(threshold,math.inf)])
    # Level-60 lower neighbour is intentionally outside reference dynamic range.
    values=[v for v in values if v==0. or v>=maximum/(2.**60)]
    ranges=torch.tensor(values,dtype=torch.float64)
    rows=torch.stack((-ranges,ranges))
    l=TenderLinear.calibrate(torch.ones(1,len(values),dtype=torch.float64),None,rows)
    actual=channel_scales(l)
    expected_depth=[]
    for v in values:
        expected_depth.append(max([0]+[k for k in range(1,61) if v>0. and v<=maximum/(2.**k)]))
    deepest=max(expected_depth)
    expected_depth=[deepest if v==0. else d for v,d in zip(values,expected_depth)]
    expected=torch.tensor([maximum/127/(2.**d) for d in expected_depth],dtype=torch.float64)
    record(f"threshold_nextafter_{maximum}",bool(torch.allclose(actual,expected,atol=0.,rtol=3e-16)),max_relative_error=float(((actual-expected).abs()/expected).max()))
# Direct integer group sum using Python unbounded integers, then Decimal physical units.
gen=torch.Generator().manual_seed(12)
rows=torch.randn(6,5,dtype=torch.float64,generator=gen)*torch.tensor([1.,.1,.001,.5,0.],dtype=torch.float64)
w=torch.randn(3,5,dtype=torch.float64,generator=gen)
l=TenderLinear.calibrate(w,None,rows,activation_bits=4)
xc=l.activation_codes(rows).tolist();wc=l.weight.codes().tolist()
expected=[]
last=int(l.exponents[-1])
for row in xc:
    outputs=[]
    for weight in wc:
        total=0
        for g,e in enumerate(l.exponents.tolist()):
            ids=l.activation_indices[l.group_ptr[g]:l.group_ptr[g+1]].tolist()
            total+=sum(row[i]*weight[i] for i in ids)*(2**(e-last))
        outputs.append(total)
    expected.append(outputs)
record("integer_group_sum",expected==l.integer_accumulator(rows).tolist(),reference=expected)
# Overflow-free dequantization with ordinary calibrated metadata.
rows=torch.tensor([[-1e308],[1e308]],dtype=torch.float64)
l=TenderLinear.calibrate(torch.tensor([[1e-308]],dtype=torch.float64),None,rows)
actual=l(rows)
reference=rows@l.weight.decode().T
record("triple_product_false_overflow",bool(torch.allclose(actual,reference,rtol=1e-14,atol=0)),actual=actual.tolist(),reference=reference.tolist())
# Reverse direction: scales-first would underflow despite a representable result.
l=TenderLinear(2,1,2,False,activation_bits=8)
l.weight=PackedTensor.from_tensor(torch.tensor([[1e-170,127e-170]],dtype=torch.float64),l.weight.spec)
l.activation_scale=torch.tensor([1e-170],dtype=torch.float64)
l.exponents=torch.tensor([60,0],dtype=torch.int64)
l.activation_indices=torch.tensor([0,1],dtype=torch.int64);l.weight_indices=l.activation_indices.clone();l.group_ptr=torch.tensor([0,1,2],dtype=torch.int64)
l.validate_state(l.state_dict())
inputs=torch.tensor([[1e-170*(2**60),0.]],dtype=torch.float64)
acc=l.integer_accumulator(inputs)
with localcontext() as context:
    context.prec=100
    target=float(Decimal(int(acc[0,0]))*Decimal.from_float(float(l.activation_scale[0]))*Decimal.from_float(float(l.weight.scale[0])))
actual=float(l(inputs)[0,0])
record("triple_product_false_underflow",actual==target and actual!=0.,actual=actual,decimal_reference=target,integer=int(acc[0,0]))
rows=torch.tensor([[-1e308],[1e308]],dtype=torch.float64)
l=TenderLinear.calibrate(torch.ones(1,1,dtype=torch.float64),None,rows)
actual=l(rows)
record("large_finite_final_result",bool(torch.equal(actual,rows)),actual=actual.tolist(),reference=rows.tolist())
l=TenderLinear.calibrate(torch.full((1,1),2.,dtype=torch.float64),None,rows)
try:
    l(rows)
    refused=False
except ValueError as error:
    refused="overflow" in str(error).lower()
record("true_float64_overflow_refused",refused)
out={"passed":sum(c["passed"] for c in checks),"total":len(checks),"checks":checks}
Path(__file__).with_name("verify_tender_results.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out,indent=2))
