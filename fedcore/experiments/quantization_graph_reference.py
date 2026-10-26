"""Small reproducible CPU example for all methods, training and graph search.

Run: python -m fedcore.experiments.quantization_graph_reference --training --search
No dataset download, GPU, external service or large experiment is involved.
"""
import argparse
import json
import torch
from torch import nn

from fedcore.algorithm.quantization import (quantize,QuantizationGraphRequest,OperatorRecipe,TransformSpec,
    Method,QuantizedTensorSpec,Grid,Rounding,ExportVerifiedArtifact,QuantizationProgram,RecoveryStageSpec,
    BackendStageSpec,ProgramArtifact,training_identity,prepare_quantization_catalogue,
    allocate_quantization_catalogue,assess_quantization_selection,prepare_quantization_space,
    run_quantization_search,GrammarSearchConfig)


def run_reference(*,training=False,search=False):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        model=nn.Sequential(nn.Linear(8,6),nn.Tanh(),nn.Linear(6,4)).eval()
        x=torch.randn(12,8);target=model(x).detach()+.05
    grid4=QuantizedTensorSpec(4,rounding=Rounding.STOCHASTIC,grid=Grid.ENDPOINT)
    grid8=QuantizedTensorSpec(8,rounding=Rounding.STOCHASTIC,grid=Grid.ENDPOINT)
    methods={
        'weight_ptq':TransformSpec(Method.PTQ,weight=QuantizedTensorSpec(4)),
        'real_svd':TransformSpec(Method.SVD,rank=3),
        'dsvd':TransformSpec(Method.DSVD,rank=3,left=QuantizedTensorSpec(4),right=QuantizedTensorSpec(8)),
        'lplr':TransformSpec(Method.LPLR,rank=3,comparison_rank=2,left=grid4,right=grid8,seed=5),
        'lplr_svd':TransformSpec(Method.LPLR_SVD,rank=3,comparison_rank=2,left=grid4,right=grid8,seed=5),
        'caldera':TransformSpec(Method.CALDERA,rank=2,weight=QuantizedTensorSpec(4),left=QuantizedTensorSpec(4),right=QuantizedTensorSpec(8),outer_iterations=1,inner_iterations=1),
        'tender':TransformSpec(Method.TENDER,weight=QuantizedTensorSpec(8,axis=0),activation_bits=8),
    }
    report={'profile':'CPU Linear reference','methods':{},'claims':'Functional small-matrix verification; no speed or task-quality claim'}
    for name,spec in methods.items():
        artifact=quantize(model,QuantizationGraphRequest((OperatorRecipe('0',spec),)),calibration=x)
        if type(artifact) is not ExportVerifiedArtifact:raise RuntimeError(f'{name}: {artifact}')
        report['methods'][name]={'tensor_bytes':artifact.storage_bytes,'file_bytes':len(artifact.checkpoint_bytes),
            'mean_squared_error':float((artifact.model(x)-model(x)).square().mean())}
    mixed=QuantizationGraphRequest((OperatorRecipe('0',methods['lplr']),OperatorRecipe('2',methods['tender'])))
    artifact=quantize(model,mixed,calibration=x)
    if type(artifact) is not ExportVerifiedArtifact:raise RuntimeError(artifact)
    report['mixed']={'tensor_bytes':artifact.storage_bytes,'file_bytes':len(artifact.checkpoint_bytes),'methods':['lplr m3 B4/B8','tender W8A8']}
    catalogue=prepare_quantization_catalogue(model,{'0':(TransformSpec(Method.DENSE),methods['lplr']),
        '2':(methods['weight_ptq'],methods['caldera'],methods['tender'])},calibration=x)
    selection=allocate_quantization_catalogue(catalogue,4096)
    assessed=assess_quantization_selection(model,catalogue,selection,calibration=x,
        evaluate=lambda loaded:float((loaded(x)-target).square().mean()),evaluation_context='example-selection-toy-v1')
    if not hasattr(assessed,'whole_model_quality'):raise RuntimeError(assessed)
    report['catalogue']={'physical_budget':4096,'selected_bytes':assessed.artifact.storage_bytes,
        'local_surrogate':assessed.local_surrogate,'whole_model_error':assessed.whole_model_quality}
    if training:
        batches=((x,target),)
        recoverable=QuantizationGraphRequest((OperatorRecipe('0',methods['lplr']),))
        recovery=quantize(model,QuantizationProgram(recoverable,recovery=RecoveryStageSpec(('0',),steps=1,seed=7),
            training_fingerprint=training_identity(batches)),calibration=x,training_batches=batches)
        if type(recovery) is not ProgramArtifact:raise RuntimeError(recovery)
        dense=QuantizationGraphRequest((OperatorRecipe('0',TransformSpec(Method.DENSE)),OperatorRecipe('2',TransformSpec(Method.DENSE))))
        qat=quantize(model,QuantizationProgram(dense,backend=BackendStageSpec('qat',('0',),training_steps=1,seed=7),
            training_fingerprint=training_identity(batches)),calibration=x,training_batches=batches)
        if type(qat) is not ProgramArtifact:raise RuntimeError(qat)
        report['training']={'recovery_updates':recovery.evidence['resources']['actual_training_steps'],
            'qat_updates':qat.evidence['resources']['actual_training_steps'],'qat_tensor_bytes':qat.storage_bytes}
    if search:
        space=prepare_quantization_space(model,{'0':(methods['lplr'],methods['caldera'],methods['tender']),
            '2':(methods['weight_ptq'],methods['dsvd'])},calibration=x,evaluation_context='example-selection-toy-v1')
        result=run_quantization_search(model,space,GrammarSearchConfig(population_size=3,generations=1,max_evaluations=5,max_seconds=30,max_proposals=12),
            calibration=x,evaluate=lambda loaded:float((loaded(x)-target).square().mean()))
        if not result.get('finite_archive'):raise RuntimeError(result)
        report['search']={'engine':result['engine'],'evaluations':result['charged_evaluations'],'stop_reason':result['stop_reason'],
                          'archive_size':len(result['finite_archive'])}
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',action='store_true');parser.add_argument('--search',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run_reference(training=args.training,search=args.search),indent=2,allow_nan=False))


if __name__=='__main__':main()
