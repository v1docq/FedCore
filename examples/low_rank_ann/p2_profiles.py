"""CPU mechanism demonstrations, not reproductions of paper quality results.

python -m examples.low_rank_ann.p2_profiles --output local-p2-demo
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import torch
from torch import nn

from fedcore.algorithm.low_rank.method_execution import transform_method
from fedcore.algorithm.low_rank.method_specs import (ASVD,FWSVD,AFM,Bolaco,FLARSVD,
    DRONE,SVDLLMV1,SVDLLMV2,SVDLLMV5,EoRA,MixedRank,BasisSharing,GroupReduce,method_name)
from fedcore.tools.registry.checkpoint_manager import CheckpointManager
from fedcore.tools.export import export_model


def run(output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(2026)
    calibration=torch.randn(24,8,dtype=torch.double)
    validation=torch.randn(7,8,dtype=torch.double)
    train=torch.randn(12,8,dtype=torch.double)
    original=nn.Sequential(nn.Linear(8,8),nn.Tanh(),nn.Linear(8,4)).double().eval()
    reports=[]
    specs=(ASVD(),FWSVD(loss='mse'),AFM(),Bolaco(),FLARSVD(),DRONE(),
           SVDLLMV1(),SVDLLMV2(),SVDLLMV5(),EoRA(),MixedRank())
    for spec in (*specs,BasisSharing(),GroupReduce(((0,2,4),(1,3,5)),(2,2))):
        model,inputs,holdout=original,calibration,validation
        options={'rank':2,'target_paths':('0','2')}
        if isinstance(spec,FWSVD):
            options['labels']=torch.zeros(24,4,dtype=torch.double)
        if isinstance(spec,Bolaco):
            options['group_labels']=torch.arange(24)%3
        if isinstance(spec,EoRA):
            options['base_model']=deepcopy(original)
            with torch.no_grad():
                for layer in options['base_model'].modules():
                    if type(layer) is nn.Linear:
                        layer.weight.mul_(.8)
        if isinstance(spec,SVDLLMV5):
            def recover(current,side,epochs,stage):
                optimizer=torch.optim.Adam(stage.trainable_parameters(),lr=.01)
                stage.validate_optimizer(optimizer)
                losses=[]
                with torch.no_grad():
                    targets=original(train)
                for _ in range(epochs):
                    optimizer.zero_grad()
                    loss=(current(train)-targets).square().mean()
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()))
                return {'training_role':'train','training_steps':len(losses),'loss':losses}
            options['recovery_executor']=recover
        if isinstance(spec,BasisSharing):
            model=nn.Sequential(nn.Linear(8,8),nn.Tanh(),nn.Linear(8,8)).double().eval()
        if isinstance(spec,GroupReduce):
            embedding=nn.Embedding(6,4).double()
            head=nn.Linear(4,6,bias=False).double()
            head.weight=embedding.weight
            model=nn.Sequential(embedding,head).eval()
            inputs=torch.tensor([[0,1,1,2,3,4,5]])
            holdout=torch.arange(6).reshape(2,3)
            options={'target_paths':('0','1')}
        result=transform_method(model,inputs,spec,batch_size=8,**options)
        root=output/method_name(spec)
        root.mkdir(exist_ok=True)
        manager=CheckpointManager(str(root/'registry'),auto_cleanup=False)
        checkpoint=root/'checkpoint.pt'
        manager.save_to_file(manager.serialize_to_bytes(result.model),str(checkpoint))
        restored=manager.load_from_file(str(checkpoint)).eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(holdout),result.model.eval()(holdout))
            error=float((restored(holdout)-model(holdout)).square().mean())
        artifact=export_model(restored,'torchscript',root/'compressed.pt',holdout[:1])
        with artifact.open('rb') as stream:
            deployed=torch.jit.load(stream)
        torch.testing.assert_close(deployed(holdout[:1]),restored(holdout[:1]))
        report=dict(result.evidence,validation_output_mse=error,
                    artifact=str(artifact),status='roundtrip_verified')
        (root/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        reports.append({'method':method_name(spec),'status':report['status'],
                        'parameters':report['parameters_after'],'tensor_bytes':report['tensor_bytes_after']})
    (output/'summary.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    return reports


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(run(args.output)))


if __name__=='__main__':
    main()
