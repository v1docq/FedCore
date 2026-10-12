"""Small, offline Dobi remapping demonstrations; no LLM quality claims.

python -m examples.low_rank_ann.dobi_remapping --output local-dobi-remapping
"""
import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn

from fedcore.algorithm.low_rank.low_rank_opt import LowRankModel
from fedcore.algorithm.low_rank.method_specs import DobiSVD
from fedcore.experiments.protocol import TensorSplit
from fedcore.experiments.runner import train_model
from fedcore.tools.export import export_model
from fedcore.tools.registry.checkpoint_manager import CheckpointManager


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    reports = []
    # Wide/tall cases exercise the two different partial-quantization layouts.
    cases = (
        ('wide_none', 48, 32, 'none', 'float16', .6),
        ('wide_fp16', 48, 32, 'symmetric_int8_v1', 'float16', .6),
        ('wide_bf16', 48, 32, 'symmetric_int8_v1', 'bfloat16', .6),
        ('tall_fp16', 32, 48, 'symmetric_int8_v1', 'float16', .6),
        ('square_fp16', 32, 32, 'symmetric_int8_v1', 'float16', .6),
        ('full_remap', 48, 32, 'symmetric_int8_v1', 'float16', 1.),
        ('zero_remap', 48, 32, 'symmetric_int8_v1', 'float16', 0.),
        ('wide_byte_budget', 48, 32, 'symmetric_int8_v1', 'float16', .6),
    )
    for name, in_features, out_features, remapping, tail_dtype, fraction in cases:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(135)
            source = nn.Sequential(nn.Linear(in_features, out_features), nn.Tanh(),
                                   nn.Linear(out_features, 5)).float().eval()
            train = torch.randn(96, in_features)
            calibration = torch.randn(64, in_features)
            validation = torch.randn(7, in_features)
        original = {key: value.clone() for key, value in source.state_dict().items()}
        split = TensorSplit(train, source(train).detach() + .05,
                            tuple(f'train-{i}' for i in range(len(train))))

        def recover(model, stage, updates, objective):
            evidence = train_model(model, split, 'regression', epochs=math.ceil(updates / 3),
                                   batch_size=32, learning_rate=.001, seed=135,
                                   batch_objective=objective, max_updates=updates)
            return dict(evidence, updates=evidence['training_steps'], stage=stage)

        options = {}
        if name == 'wide_byte_budget':
            nominal = next(report for report in reports if report['case'] == 'wide_fp16')
            options['tensor_byte_budget'] = nominal['tensor_bytes_after'] - 100
        spec = DobiSVD(updates=2, initial_fraction=fraction, target_fraction=fraction,
                       budget_weight=0, export_rounding='ceil' if fraction == 1 else 'floor',
                       max_activation_rows=32, max_calibration_batches=8, remapping=remapping,
                       remap_tail_dtype=tail_dtype)
        result = LowRankModel({'device': 'cpu'}).compress_profile(source, calibration, spec,
                    target_paths=('0',), batch_size=32, recovery_executor=recover, **options)
        root = output / name
        root.mkdir(exist_ok=True)
        manager = CheckpointManager(str(root / 'registry'), auto_cleanup=False)
        checkpoint = root / 'checkpoint.pt'
        manager.save_to_file(manager.serialize_to_bytes(result.model), str(checkpoint))
        restored = manager.load_from_file(str(checkpoint)).eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(validation), result.model.eval()(validation),
                                       atol=0, rtol=0)
            mse = float((restored(validation) - source(validation)).square().mean())
        artifact = export_model(restored, 'torchscript', root / 'deployed.pt', validation[:1])
        with artifact.open('rb') as stream:
            deployed = torch.jit.load(stream)
        torch.testing.assert_close(deployed(validation[:1]), restored(validation[:1]),
                                   atol=0, rtol=0)
        assert all(torch.equal(value, source.state_dict()[key]) for key, value in original.items())
        actual_bytes = sum(value.numel() * value.element_size()
                           for value in restored.state_dict().values())
        assert actual_bytes == result.evidence['tensor_bytes_after']
        if options:
            assert actual_bytes <= options['tensor_byte_budget']
        report = dict(result.evidence, case=name, validation_output_mse=mse,
                      requested_tensor_byte_budget=options.get('tensor_byte_budget'),
                      independently_counted_tensor_bytes=actual_bytes,
                      checkpoint_file_bytes=checkpoint.stat().st_size,
                      torchscript_file_bytes=artifact.stat().st_size,
                      export_contract='fixed declared CPU tensor input',
                      status='checkpoint_and_fixed_input_export_verified')
        (root / 'evidence.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        reports.append(report)
    destination = output / 'dobi-remapping-results.json'
    destination.write_text(json.dumps(reports, indent=2, allow_nan=False), encoding='utf-8')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(run(parser.parse_args().output))
