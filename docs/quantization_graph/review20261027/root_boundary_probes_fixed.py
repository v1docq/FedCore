from types import SimpleNamespace
from unittest.mock import patch
import json
from pathlib import Path
import torch
from torch import nn
from fedcore.algorithm.quantization import program as p
from fedcore.algorithm.quantization.graph_specs import QuantizationGraphRequest,OperatorRecipe,TransformSpec,Method
from fedcore.algorithm.quantization.checkpoint import validate_descriptor_budget

results={}
rss=[100*1024**2]
original=p.compress_quantization_graph
def after_compression(*args,**kwargs):
    artifact=original(*args,**kwargs)
    rss[0]=2*1024**3
    return artifact
model=nn.Sequential(nn.Linear(2,2))
x=torch.ones(2,2)
request=QuantizationGraphRequest((OperatorRecipe('0',TransformSpec(Method.PTQ)),),max_peak_bytes=1024**3)
with patch('psutil.Process',lambda *a,**k:SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=rss[0]))),patch.object(p,'compress_quantization_graph',after_compression):
    artifact=p.quantize(model,p.QuantizationProgram(request),calibration=x)
    results['program_request_peak_limit']={'returned':type(artifact).__name__,'declared_limit':request.max_peak_bytes,'final_rss':rss[0]}

description={'type':'QuantizationGraph','schema':'quantization_graph_v1','kind':'RegisteredBackendGraph','config':{
    'instructions':[{'name':'x','op':'placeholder','target':'x','args':[]},{'name':'output','op':'output','target':'output','args':[{'node':'x'}]}],
    'operators':{},'constants':{'constant_0':{'shape':[2],'dtype':'float64'}}}}
try:
    validate_descriptor_budget(description,max_bytes=1)
    results['backend_constant_allocation_budget']={'returned':'accepted','declared_bytes':16,'budget':1}
except ValueError as error:results['backend_constant_allocation_budget']={'returned':'rejected','error':str(error)}
print(json.dumps(results,indent=2))
Path(__file__).with_suffix('.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
