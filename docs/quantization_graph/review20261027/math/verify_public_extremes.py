import json,torch
from pathlib import Path
from fedcore.algorithm.quantization.graph_execution import compress_quantization_graph,ExportVerifiedArtifact
from fedcore.algorithm.quantization.graph_specs import TransformSpec,Method,QuantizationGraphRequest,OperatorRecipe
from fedcore.algorithm.quantization.tender import TenderLinear
from fedcore.algorithm.quantization.matrix_transforms import input_second_moment
torch.set_num_threads(1)
checks=[]
for name,w,x in [("large_output",1.,1e308),("subnormal_output",1.,1e-309),("large_activation_small_weight",1e-308,1e308),("small_activation_large_weight",1e308,1e-308)]:
    rows=torch.tensor([[-x],[0.],[x]],dtype=torch.float64)
    model=torch.nn.Sequential(torch.nn.Linear(1,1,bias=False).double())
    model[0].weight.data.fill_(w)
    artifact=compress_quantization_graph(model,QuantizationGraphRequest((OperatorRecipe("0",TransformSpec(Method.TENDER)),)),calibration=rows)
    result={"name":name,"public_type":type(artifact).__name__}
    if isinstance(artifact,ExportVerifiedArtifact):
        layer=artifact.model[0]
        actual=artifact.model(rows)
        expected=rows@layer.weight.decode().T
        eps=abs(float(expected[-1,0]))*1e-13
        result.update(passed=bool(torch.allclose(actual,expected,atol=max(eps,5e-324),rtol=1e-13)),actual=actual.tolist(),expected=expected.tolist())
    else:result.update(passed=False,error=str(artifact))
    checks.append(result)
rows=torch.tensor([[1e154,1e154],[1e154,-1e154]],dtype=torch.float64)
moment=input_second_moment(rows,2)
model=torch.nn.Sequential(torch.nn.Linear(2,2,bias=False).double())
model[0].weight.data.copy_(torch.tensor([[1e-154,.3e-154],[.2e-154,-.8e-154]],dtype=torch.float64))
artifact=compress_quantization_graph(model,QuantizationGraphRequest((OperatorRecipe("0",TransformSpec(Method.CALDERA,rank=1)),)),calibration=rows)
checks.append({"name":"representable_large_mean_moment","passed":bool(torch.isfinite(moment).all()) and isinstance(artifact,ExportVerifiedArtifact),"moment":moment.tolist(),"public_type":type(artifact).__name__})
out={"passed":sum(c["passed"] for c in checks),"total":len(checks),"checks":checks}
Path(__file__).with_name("verify_public_extremes_results.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out,indent=2))
