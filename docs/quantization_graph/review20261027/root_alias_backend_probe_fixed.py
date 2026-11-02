import json
from pathlib import Path
import torch
from torch import nn
from fedcore.algorithm.quantization.program import quantize,QuantizationProgram
from fedcore.algorithm.quantization.graph_specs import QuantizationGraphRequest,OperatorRecipe,TransformSpec,Method
from fedcore.algorithm.quantization.training import BackendStageSpec
from fedcore.algorithm.quantization.backend import RegisteredBackendLinear

layer=nn.Linear(3,3)
model=nn.Sequential(layer,nn.ReLU(),layer)
x=torch.randn(6,3)
results={}
for path in ('0','2'):
    request=QuantizationProgram(QuantizationGraphRequest((OperatorRecipe(path,TransformSpec(Method.DENSE)),)),backend=BackendStageSpec('static',(path,)))
    artifact=quantize(model,request,calibration=x)
    results[path]={'result':type(artifact).__name__,'failure':str(artifact) if not hasattr(artifact,'model') else None}
    if hasattr(artifact,'model'):
        results[path]['quantized_operator_count']=sum(type(m) is RegisteredBackendLinear for m in artifact.model.modules())
        results[path]['operators']=[type(m).__name__ for m in artifact.model.operators.values()]
print(json.dumps(results,indent=2))
Path(__file__).with_suffix('.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
