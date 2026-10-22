"""Small CPU mechanism demonstrations, distinct from paper-quality experiments.

python -m examples.low_rank_ann.p4_profiles --output local-p4-demo
"""
import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn

from fedcore.algorithm.low_rank.method_specs import ARS,DobiSVD,RankDyna
from fedcore.algorithm.low_rank.method_execution import transform_method
from fedcore.experiments.protocol import TensorSplit
from fedcore.experiments.runner import train_model
from fedcore.tools.registry.checkpoint_manager import CheckpointManager
from fedcore.tools.export import export_model


def run(output):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261012)
        source = nn.Sequential(nn.Linear(6,6),nn.Tanh(),nn.Linear(6,3)).double().eval()
        train = torch.randn(12,6,dtype=torch.double)
        calibration = torch.randn(12,6,dtype=torch.double)
        validation = torch.randn(5,6,dtype=torch.double)
    split = TensorSplit(train,source(train).detach()+.15,tuple(f'train-{i}' for i in range(len(train))))
    def recover(model,stage,updates,objective):
        evidence = train_model(model,split,'regression',epochs=math.ceil(updates/3),
            batch_size=4,learning_rate=.005,seed=23,batch_objective=objective,max_updates=updates)
        return dict(evidence,updates=evidence['training_steps'],stage=stage)

    cases = (
        ('ars_indices',ARS(updates=3,input_size=4,hidden_size=4,mask_bias=0)),
        ('ars_prefix',ARS(updates=3,input_size=4,hidden_size=4,mask_bias=0,export_policy='paper_prefix')),
        ('ars_zero',ARS(updates=2,input_size=4,hidden_size=4,mask_bias=-30)),
        ('dobi_ipca',DobiSVD(updates=3,max_activation_rows=32,max_calibration_batches=8)),
        ('dobi_full',DobiSVD(updates=1,initial_fraction=1.,target_fraction=1.,budget_weight=0,
                            export_rounding='ceil',max_activation_rows=32,max_calibration_batches=8)),
        ('rankdyna_groups',RankDyna(updates=3)),
        ('rankdyna_weighted_extension',RankDyna(updates=3,parameter_budget=24)),
    )
    reports = []
    for name,spec in cases:
        result = transform_method(source,calibration,spec,target_paths=('0',),batch_size=4,
                                  recovery_executor=recover)
        root = output/name
        root.mkdir(exist_ok=True)
        manager = CheckpointManager(str(root/'registry'),auto_cleanup=False)
        checkpoint = root/'checkpoint.pt'
        manager.save_to_file(manager.serialize_to_bytes(result.model),str(checkpoint))
        restored = manager.load_from_file(str(checkpoint)).eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(validation),result.model.eval()(validation))
            error = float((restored(validation)-source(validation)).square().mean())
        artifact = export_model(restored,'torchscript',root/'deployed.pt',validation[:1])
        with artifact.open('rb') as stream:
            deployed = torch.jit.load(stream)
        torch.testing.assert_close(deployed(validation[:1]),restored(validation[:1]))
        report = dict(result.evidence,case=name,validation_output_mse=error,
            checkpoint_bytes=checkpoint.stat().st_size,artifact_bytes=artifact.stat().st_size,
            export_contract='fixed declared tensor input; dynamic shapes and masks are unverified',
            status='checkpoint_and_fixed_input_export_verified')
        (root/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        reports.append({'case':name,'status':report['status'],'parameters':report['parameters_after'],
                        'tensor_bytes':report['tensor_bytes_after'],'updates':spec.updates,
                        'validation_output_mse':error})
    (output/'summary.json').write_text(json.dumps(reports,indent=2,allow_nan=False),encoding='utf-8')
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(run(parser.parse_args().output)))


if __name__ == '__main__':
    main()
