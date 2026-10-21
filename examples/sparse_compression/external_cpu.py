"""One offline external CPU v4 request through the isolated worker."""
import argparse
import json
from pathlib import Path
import sys
import torch
from torch import nn

from fedcore.algorithm.pruning.sparse.specs import HASSLEFreeSpec, NMStructure
from fedcore.external_runtime.client import compress
from fedcore.external_runtime.contracts import Resources, SparseOptions


def run(output):
    torch.manual_seed(17)
    torch.set_num_threads(1)
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
    calibration, validation = torch.randn(16, 8), torch.randn(16, 8)
    root = Path(output).resolve()
    result = compress(model, validation[:2], (validation, model(validation).detach()),
        jobs_root=root, method='hassle_free',
        sparse_options=SparseOptions(HASSLEFreeSpec(structure=NMStructure(rank=1),
            outer_steps=2, inner_steps=4, seed=17)),
        calibration=(calibration, model(calibration).detach()), max_relative_error=2.0,
        resources=Resources(repetitions=3, timeout_seconds=120), python_executable=sys.executable)
    (root/'external-report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    if result['status'] != 'succeeded':
        raise RuntimeError(result.get('error', result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({'status': report['status'], 'job_directory': report['job_directory'],
        'artifact': report['artifact'], 'request_version': report['request_version']}))
