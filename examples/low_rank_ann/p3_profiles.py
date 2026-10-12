"""Bounded CPU mechanism checks; no paper-scale quality claims.

python -m examples.low_rank_ann.p3_profiles --output local-p3-demo
"""
import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn

from fedcore.algorithm.low_rank.method_execution import transform_method
from fedcore.algorithm.low_rank.method_specs import (
    ESPACE, PELA, LASER, CTRAFM, CTRMLPAFM, FlatValueOutput, MoDeValueOutput,
    FlatMLPChannels, MoDeMLPChannels, FlatQueryKey, MoDeQueryKey,
)
from fedcore.algorithm.low_rank.coupled_layers import ReferenceAttention, ReferenceMLP
from fedcore.algorithm.low_rank.projection_layers import CTRFieldGraph, CTRLinearSum
from fedcore.algorithm.low_rank.projection_execution import (
    DiagnoseIntervention, diagnose_laser, export_laser_compact,
)
from fedcore.experiments.measurement import tensor_state_bytes
from fedcore.experiments.protocol import TensorSplit
from fedcore.experiments.runner import train_model
from fedcore.tools.export import export_model
from fedcore.tools.registry.checkpoint_manager import CheckpointManager


class CTRExportView(nn.Module):
    """Explicit single-tensor deployment view preserving BOTH CTR outputs."""
    def __init__(self, graph):
        super().__init__()
        self.graph = graph

    def forward(self, tokens):
        output = self.graph(tokens)
        return output['linear'], output['interactions']


def _mse(actual, expected):
    if isinstance(actual, dict):
        return {name:_mse(actual[name],expected[name]) for name in actual}
    return float((actual-expected).double().square().mean())


def _write(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def run(output):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20261012)
    baseline = nn.Sequential(nn.Linear(8,8),nn.Tanh(),nn.Linear(8,4)).double().eval()
    calibration = torch.randn(16,8,dtype=torch.double)
    holdout = torch.randn(5,8,dtype=torch.double)
    train = torch.randn(12,8,dtype=torch.double)
    split = TensorSplit(train,baseline(train).detach()+.1,tuple(f'train-{i}' for i in range(12)))

    def recover(current, stage, updates, objective):
        evidence = train_model(current,split,'regression',epochs=math.ceil(updates/3),
            batch_size=4,learning_rate=.01,seed=23,batch_objective=objective,max_updates=updates)
        return dict(evidence,updates=evidence['training_steps'],stage=stage)

    reports = []
    def record(name, model, inputs, validation, spec, **options):
        transformed = transform_method(model,inputs,spec,batch_size=4,**options)
        root = output/name
        root.mkdir(exist_ok=True)
        manager = CheckpointManager(str(root/'registry'),auto_cleanup=False)
        checkpoint = root/'checkpoint.pt'
        manager.save_to_file(manager.serialize_to_bytes(transformed.model),str(checkpoint))
        restored = manager.load_from_file(str(checkpoint)).eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(validation),transformed.model.eval()(validation))
            error = _mse(restored(validation),model.eval()(validation))
        is_ctr = type(restored) is CTRFieldGraph
        deployable = CTRExportView(restored) if is_ctr else restored
        example = validation[:1]
        artifact = export_model(deployable,'torchscript',root/'deployed.pt',example)
        with artifact.open('rb') as stream:
            deployed = torch.jit.load(stream)
        torch.testing.assert_close(deployed(example),deployable(example))
        report = dict(transformed.evidence,case=name,validation_output_mse=error,
            checkpoint_bytes=checkpoint.stat().st_size,artifact_bytes=artifact.stat().st_size,
            export_view='linear_and_interactions_tuple' if is_ctr else 'single_tensor_forward',
            status='checkpoint_and_fixed_input_export_verified')
        if type(model) is ReferenceAttention:
            before = model.attention(validation,use_cache=True)
            after = restored.attention(validation,use_cache=True)
            report['kv_cache_bytes_before'] = before.cache.tensor_bytes
            report['kv_cache_bytes_after'] = after.cache.tensor_bytes
            report['validation_logits_mse'] = _mse(after.logits,before.logits)
        _write(root/'report.json',report)
        reports.append({'case':name,'status':report['status'],
            'parameters':report['parameters_after'],'tensor_bytes':tensor_state_bytes(restored)})
        return transformed,root

    for criterion in ('mse','nmse','go_mse_bound','go_normalized','nl_first_order_bound','nl_normalized'):
        record('espace_'+criterion,baseline,calibration,holdout,
            ESPACE(criterion=criterion,updates=2,max_examples=16,max_batches=4),rank=2,
            target_paths=('0','2'),labels=baseline(calibration).detach()+.2,recovery_executor=recover)
    record('pela',baseline,calibration,holdout,
        PELA(kappa=1.5,block_map=(('0','0'),('2','2')),updates=2),
        target_paths=('0','2'),recovery_executor=recover)
    dense,root = record('laser',baseline,calibration,holdout,LASER(rho=.25),target_paths=('0',))
    diagnosis = diagnose_laser(baseline,tuple(DiagnoseIntervention('0',rank,'baseline-v1',
        'validation-v1','output_mse') for rank in (1,2)),
        lambda candidate:float((candidate(holdout)-baseline(holdout)).square().mean()))
    selected_rank = diagnosis.selected.intervention.rank
    selected_dense = transform_method(baseline,calibration,LASER(rho=selected_rank/8),target_paths=('0',))
    compact,economics = export_laser_compact(selected_dense.model,'0',selected_rank)
    torch.testing.assert_close(compact(holdout),selected_dense.model(holdout))
    compact_artifact = export_model(compact.eval(),'torchscript',root/'selected_compact.pt',holdout[:1])
    _write(root/'diagnosis.json',{'selected_rank':diagnosis.selected.intervention.rank,
        'isolated_candidates':[{'rank':c.intervention.rank,'quality':c.task_quality,
                               'bytes':c.tensor_bytes,'baseline':c.baseline_id} for c in diagnosis.candidates],
        'selected_compact_export':dict(economics,artifact_bytes=compact_artifact.stat().st_size)})
    for variant in ('linear_v1','relu_between_factors_v1'):
        record('ctr_mlp_'+variant,baseline,calibration,holdout,
            CTRMLPAFM(nonlinear_variant=variant,updates=2),rank=2,
            target_paths=('0','2'),recovery_executor=recover)

    attention = ReferenceAttention(8,2,1,4,dtype=torch.double).eval()
    mlp = ReferenceMLP(8,12,gated=True,activation='silu',dtype=torch.double).eval()
    tokens = torch.randn(8,4,8,dtype=torch.double)
    validation = torch.randn(3,4,8,dtype=torch.double)
    for name,spec in (('flat_vo',FlatValueOutput()),('modegpt_vo',MoDeValueOutput()),
            ('flat_qk',FlatQueryKey()),('modegpt_qk',MoDeQueryKey()),
            ('flat_mlp',FlatMLPChannels()),('modegpt_mlp',MoDeMLPChannels())):
        model = mlp if 'mlp' in name else attention
        record(name,model,tokens,validation,spec,rank=2,target_paths=('',))

    fields = {'user':nn.Embedding(12,6,dtype=torch.double),
              'item':nn.Embedding(10,6,dtype=torch.double)}
    graph = CTRFieldGraph(fields,CTRLinearSum({'user':6,'item':6},2,dtype=torch.double),
                          (('user','item'),)).eval()
    categories = torch.tensor([[0,1],[3,5],[8,2],[11,9]],dtype=torch.long)
    record('ctr_fields',graph,categories,categories,
        CTRAFM((('user',2),('item',2))))
    _write(output/'summary.json',reports)
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(run(parser.parse_args().output)))


if __name__ == '__main__':
    main()
