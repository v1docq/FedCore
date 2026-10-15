"""Small CPU round trip; a mechanism demonstration, not a quality benchmark.

Run from the repository root:
python -m examples.low_rank_ann.weighted_profile --output local-weighted-demo
"""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from fedcore.algorithm.low_rank.execution import transform_weighted
from fedcore.algorithm.low_rank.plans import MetricPolicy
from fedcore.tools.registry.checkpoint_manager import CheckpointManager
from fedcore.tools.export import export_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(2026)
    model = nn.Sequential(nn.Linear(16, 12), nn.Tanh(), nn.Linear(12, 8)).double().eval()
    calibration = torch.randn(32, 16, dtype=torch.double)
    validation = torch.randn(8, 16, dtype=torch.double)
    result = transform_weighted(model, calibration, rank_ratio=.5, batch_size=8,
                                policy=MetricPolicy(ridge=0.0, nullspace_policy='support_only'))
    manager = CheckpointManager(str(args.output / 'registry'), auto_cleanup=False)
    checkpoint = args.output / 'compressed-checkpoint.pt'
    manager.save_to_file(manager.serialize_to_bytes(result.model), str(checkpoint))
    restored = manager.load_from_file(str(checkpoint)).eval()
    artifact = export_model(restored, 'torchscript', args.output / 'compressed.pt', validation[:1])
    with artifact.open('rb') as stream:
        deployed = torch.jit.load(stream)
    with torch.no_grad():
        torch.testing.assert_close(restored(validation), result.model(validation))
        torch.testing.assert_close(deployed(validation[:1]), restored(validation[:1]))
        report = dict(result.evidence, validation_output_mse=float((model(validation) - restored(validation)).square().mean()),
                      checkpoint=str(checkpoint), artifact=str(artifact), status='roundtrip_verified')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'parameters_before': report['parameters_before'],
                      'parameters_after': report['parameters_after'], 'report': str(args.output / 'report.json')}))


if __name__ == '__main__':
    main()
