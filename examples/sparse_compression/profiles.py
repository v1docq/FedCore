"""Offline CPU mechanisms, checkpoint parity and actual artifact measurements.

Run against an installed wheel: python profiles.py --output /absolute/results
No model or data download. This is a mechanism experiment, not paper quality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import torch
from torch import nn

from fedcore.algorithm.pruning.sparse.execution import transform_sparse, ChannelsSpec
from fedcore.algorithm.pruning.sparse.specs import HASSLEFreeSpec, LoSparseSpec, NMStructure
from fedcore.algorithm.pruning.sparse.linear import LoSparseTrainingCallback
from fedcore.algorithm.pruning.sparse.channels import SoLASpec
from fedcore.algorithm.pruning.sparse.architectures import SparseReferenceTransformer, FlexiGPTSpec
from fedcore.algorithm.pruning.sparse.policies import PruneNetSpec
from fedcore.algorithm.pruning.sparse.moe import ReferenceMoE, MoECompressionSpec
from fedcore.algorithm.low_rank.coupled_layers import ReferenceMLP
from fedcore.experiments.measurement import measure_artifact, tensor_state_bytes, predict_loaded_artifact
from fedcore.experiments.protocol import TensorSplit
from fedcore.experiments.runner import train_model
from fedcore.tools.registry.checkpoint_manager import CheckpointManager


def run(output, seed=17):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    manager = CheckpointManager(root)
    x_cal, x_train, x_test = (torch.randn(16, 8) for _ in range(3))
    linear = nn.Linear(8, 8).eval()
    ffn = ReferenceMLP(8, 8, gated=True).eval()
    transformer = SparseReferenceTransformer(8, 8, 2, num_heads=2).eval()
    moe = ReferenceMoE(8, 8, 3, top_k=1).eval()
    # Acceptance thresholds are declared before candidates are evaluated.
    limits = {'max_relative_error': 2.0, 'useful_if': 'quality_pass and tensor_state_bytes < baseline_tensor_state_bytes',
              'speedup_claim': False}
    cases = []
    cases.append(('hassle_sparse_only', linear, x_cal, x_test,
        lambda: transform_sparse(linear, 'hassle_free', HASSLEFreeSpec(structure=NMStructure(rank=0), outer_steps=1, inner_steps=2), calibration=x_cal)))
    cases.append(('hassle_full_hybrid', linear, x_cal, x_test,
        lambda: transform_sparse(linear, 'hassle_free', HASSLEFreeSpec(structure=NMStructure(rank=1), outer_steps=2, inner_steps=4), calibration=x_cal)))
    diagonal_inputs = torch.diag(x_cal.square().mean(0).sqrt()) * (8**0.5)
    cases.append(('hassle_diagonal_hybrid', linear, diagonal_inputs, x_test,
        lambda: transform_sparse(linear, 'hassle_free', HASSLEFreeSpec(structure=NMStructure(rank=1), outer_steps=2, inner_steps=4), calibration=diagonal_inputs)))
    def train_losparse(graph, modules, spec):
        torch.testing.assert_close(graph(x_train), linear(x_train), rtol=1e-5, atol=1e-6)
        y = linear(x_train).detach()
        split = TensorSplit(x_train, y, tuple('train-'+str(i) for i in range(len(x_train))))
        callback = LoSparseTrainingCallback(modules, lambda model, x, y: (model(x)-y).square().mean())
        train_model(graph, split, 'regression', epochs=spec.total_steps, batch_size=len(split.x),
            learning_rate=1e-3, seed=seed, batch_objective=callback)
    for name, rows, warmup in [('losparse_rank_only', 0, 0), ('losparse_immediate', 2, 0), ('losparse_gradual', 2, 1)]:
        spec = LoSparseSpec(1, rows, 3, warmup_steps=warmup)
        cases.append((name, linear, x_cal, x_test, lambda spec=spec: transform_sparse(linear, 'losparse', spec, train=train_losparse)))
    cases.append(('sola', ffn, x_cal, x_test, lambda: transform_sparse(ffn, 'sola', SoLASpec(prime_count=2, rank=1), calibration=x_cal)))
    cases.append(('sola_exact_partition', ffn, x_cal, x_test,
        lambda: transform_sparse(ffn, 'partition_mlp', ChannelsSpec((0, 1)))))
    cases.append(('sola_uniform_rank', ffn, x_cal, x_test,
        lambda: transform_sparse(ffn, 'sola', SoLASpec(prime_count=0, rank=1), calibration=x_cal)))
    cases.append(('prunenet', ffn, x_cal, x_test, lambda: transform_sparse(ffn, 'prunenet', PruneNetSpec(keep_count=4, episodes=2))))
    random_channels = tuple(torch.randperm(8)[:4].tolist())
    norm_channels = tuple(ffn.up_proj.weight.norm(dim=1).argsort(descending=True)[:4].tolist())
    cases.append(('prunenet_random_control', ffn, x_cal, x_test,
        lambda: transform_sparse(ffn, 'slice_mlp', ChannelsSpec(random_channels))))
    cases.append(('prunenet_row_norm_control', ffn, x_cal, x_test,
        lambda: transform_sparse(ffn, 'slice_mlp', ChannelsSpec(norm_channels))))
    tx_cal, tx_test = torch.randn(4, 3, 8), torch.randn(4, 3, 8)
    cases.append(('flexigpt', transformer, tx_cal, tx_test, lambda: transform_sparse(transformer, 'flexigpt', FlexiGPTSpec((0,), rank=1), calibration=tx_cal)))
    for name, ranks in [('moe_remove', None), ('moe_combine', (1, 1))]:
        spec = MoECompressionSpec((0, 2), ranks)
        cases.append((name, moe, x_cal, x_test, lambda spec=spec: transform_sparse(moe, 'moe_i2', spec, calibration=x_cal)))
    cases.append(('moe_factor_only', moe, x_cal, x_test,
        lambda: transform_sparse(moe, 'moe_i2', MoECompressionSpec((0, 1, 2), (1, 1, 1)), calibration=x_cal)))
    records = []
    for name, baseline, calibration, evaluation, execute in cases:
        start = time.perf_counter()
        result = execute()
        preparation = time.perf_counter()-start
        compact = result.model.eval()
        checkpoint = manager.serialize_to_bytes(compact)
        (root/(name+'.checkpoint.pt')).write_bytes(checkpoint)
        restored = manager.deserialize_from_bytes(checkpoint).eval()
        with torch.no_grad():
            expected, actual = baseline(evaluation), compact(evaluation)
            torch.testing.assert_close(restored(evaluation), actual, rtol=0, atol=0)
            relative = float(torch.linalg.vector_norm((expected-actual).double())/torch.linalg.vector_norm(expected.double()).clamp_min(1e-12))
        measured = measure_artifact(compact, evaluation[:2], root/(name+'.pt'), repeats=3, warmup=1)
        record = {'name': name, 'source': result.evidence, 'preparation_seconds': preparation,
            'baseline_tensor_state_bytes': tensor_state_bytes(baseline), 'tensor_state_bytes': tensor_state_bytes(compact),
            'checkpoint_file_bytes': len(checkpoint), 'checkpoint_parity': True, 'relative_error': relative,
            'quality_pass': relative <= limits['max_relative_error'], 'artifact': measured}
        record['useful'] = record['quality_pass'] and record['tensor_state_bytes'] < record['baseline_tensor_state_bytes']
        if measured['status'] == 'succeeded':
            actual_export = predict_loaded_artifact(measured['artifact'], evaluation[:2])
            torch.testing.assert_close(actual_export, compact(evaluation[:2]).detach(), rtol=1e-4, atol=1e-5)
        records.append(record)
    report = {'version': 1, 'seed': seed, 'device': 'cpu', 'dtype': 'float32',
        'data_roles': {'calibration': 'independent synthetic Gaussian seed stream', 'train': 'independent synthetic Gaussian with dense teacher outputs',
                       'evaluation': 'independent synthetic Gaussian; never used for selection or training'},
        'thresholds': limits, 'profiles': records,
        'limitations': ['synthetic mechanisms do not establish real-model quality', 'no generation or KV-cache measurements',
            'RSS snapshots are not peak memory', 'portable N:M reference makes no acceleration claim',
            'paper ablation matrix and real pretrained quality remain separate experiments'],
        'tedepr': {'status': 'insufficient_specification', 'reason': 'Full primary algorithm specification unavailable'}}
    (root/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    print(json.dumps({'report': str(Path(args.output).resolve()/'report.json'),
                      'profiles': len(run(args.output, args.seed)['profiles'])}))
