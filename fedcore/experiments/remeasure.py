"""Repeat deployment measurements of the SAME verified artifacts, without fit.

Each session runs in a fresh Python process. This estimates runtime variation;
it is not an independent repeat of model training or a search by latency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from fedcore.tools.atomic_json import write_atomic_json
from .measurement import file_hash, load_artifact, _call_loaded, _percentile, _synchronize
from .protocol import ProtocolError, TensorSplit, tensor_hash
from .runner import environment_manifest


def _session(manifest_path, output_path, *, batch_sizes, repeats, warmup):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    replay = manifest['replay_files']['data.pt']
    if file_hash(replay['path']) != replay['sha256']:
        raise ProtocolError('Measurement replay data hash mismatch')
    data = torch.load(replay['path'], map_location='cpu', weights_only=True)['validation']
    role = manifest['data']['roles']['validation']
    if 'files' in data:
        files = data['files']
        split = TensorSplit.from_npy(files['x']['path'], files['y']['path'], tuple(role['ids']),
            expected_hashes=(files['x']['sha256'], files['y']['sha256']))
        x = split.x
    else:
        x = data['x']
    if tensor_hash(x) != role['x_sha256']:
        raise ProtocolError('Validation inputs differ from the original run')
    records = []
    previous_threads = torch.get_num_threads()
    threads = manifest['protocol']['threads']
    torch.set_num_threads(threads)
    try:
        for record in manifest['candidates']:
            if record['status'] != 'succeeded':
                continue
            artifact = record['measurement']['artifact']
            prediction_info = record['validation_predictions']
            if file_hash(prediction_info['path']) != prediction_info['sha256']:
                raise ProtocolError('Validation prediction hash mismatch')
            predictions = torch.load(prediction_info['path'], map_location='cpu', weights_only=True)
            if predictions['ids'] != role['ids']:
                raise ProtocolError('Validation prediction order differs')
            loaded = load_artifact(artifact)
            for batch_size in batch_sizes:
                if batch_size > len(x):
                    records.append({'candidate_id': record['candidate_id'], 'batch_size': batch_size,
                                    'status': 'unsupported', 'reason': 'Validation role is smaller than batch'})
                    continue
                sample = x[:batch_size]
                device_sample = sample.to(artifact['device'])
                times, full_times = [], []
                with torch.inference_mode():
                    actual = _call_loaded(loaded, artifact, device_sample).detach().cpu()
                    torch.testing.assert_close(actual, predictions['predictions'][:batch_size], rtol=1e-4, atol=1e-5)
                    for _ in range(warmup):
                        _call_loaded(loaded, artifact, device_sample)
                    _synchronize(artifact['device'])
                    for _ in range(repeats):
                        _synchronize(artifact['device'])
                        start = time.perf_counter()
                        _call_loaded(loaded, artifact, device_sample)
                        _synchronize(artifact['device'])
                        times.append((time.perf_counter() - start) * 1000)
                    for _ in range(repeats):
                        _synchronize(artifact['device'])
                        start = time.perf_counter()
                        _call_loaded(loaded, artifact, sample.to(artifact['device'])).detach().cpu()
                        _synchronize(artifact['device'])
                        full_times.append((time.perf_counter() - start) * 1000)
                records.append({'candidate_id': record['candidate_id'], 'method': record['configuration']['method'],
                    'status': 'succeeded', 'artifact_sha256': artifact['sha256'],
                    'file_bytes': Path(artifact['path']).stat().st_size, 'device': artifact['device'],
                    'format': artifact['format'], 'batch_size': batch_size, 'threads': threads,
                    'input_shape': list(sample.shape), 'dtype': str(sample.dtype),
                    'repeats': repeats, 'warmup': warmup, 'prediction_parity': 'checked',
                    'raw_inference_ms': times, 'raw_full_call_ms': full_times,
                    'latency_p50_ms': _percentile(times, .5), 'latency_p95_ms': _percentile(times, .95),
                    'throughput_samples_per_second': batch_size * repeats * 1000 / sum(times),
                    'full_call_p50_ms': _percentile(full_times, .5),
                    'scope': 'loaded artifact; full call includes tensor transfers, excludes disk I/O'})
    finally:
        torch.set_num_threads(previous_threads)
    result = {'environment': environment_manifest(), 'records': records,
              'claim': 'one fresh-process timing session; no training/search superiority claim'}
    write_atomic_json(output_path, result)
    return result


def remeasure(manifest_path, output_dir, *, sessions=3, batch_sizes=(1, 32), repeats=100, warmup=10):
    if not batch_sizes or len(set(batch_sizes)) != len(batch_sizes) or any(type(n) is not int or n < 1 for n in (sessions, repeats, *batch_sizes)) or type(warmup) is not int or warmup < 0:
        raise ProtocolError('Positive session/repeat/batch counts and nonnegative warmup required')
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise ProtocolError('Measurement output directory must be empty')
    path = Path(manifest_path).resolve()
    plan = {'manifest': str(path), 'manifest_sha256': file_hash(path), 'sessions': sessions,
            'batch_sizes': list(batch_sizes), 'repeats': repeats, 'warmup': warmup,
            'policy': 'fresh sequential process per session; same artifact hashes; validation parity',
            'environment': environment_manifest(), 'claim': 'runtime replicates, not independent training repeats'}
    write_atomic_json(root / 'measurement_plan.json', plan)
    results = []
    for index in range(sessions):
        output = root / f'session-{index}.json'
        command = [sys.executable, '-m', 'fedcore.experiments.remeasure', '--worker',
                   '--manifest', str(path), '--output-dir', str(output), '--repeats', str(repeats),
                   '--warmup', str(warmup), '--batch-sizes', *(str(n) for n in batch_sizes)]
        completed = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if completed.returncode:
            write_atomic_json(root / 'failure.json', {'session': index, 'returncode': completed.returncode,
                'stderr': completed.stderr[-4000:], 'status': 'failed'})
            raise RuntimeError('Measurement session failed; see failure.json')
        result = json.loads(output.read_text(encoding='utf-8'))
        if result['environment']['source_sha256'] != plan['environment']['source_sha256'] or file_hash(path) != plan['manifest_sha256']:
            raise ProtocolError('Source or run manifest changed during frozen measurement')
        results.append({'session': index, 'path': str(output), 'sha256': file_hash(output)})
    write_atomic_json(root / 'measurement_summary.json', {'status': 'completed', 'sessions': results,
        'claim': plan['claim'], 'memory': 'not remeasured; original process snapshots are not model-exclusive peaks'})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sessions', type=int, default=3)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 32])
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        _session(args.manifest, args.output_dir, batch_sizes=args.batch_sizes, repeats=args.repeats, warmup=args.warmup)
    else:
        remeasure(args.manifest, args.output_dir, sessions=args.sessions,
                  batch_sizes=tuple(args.batch_sizes), repeats=args.repeats, warmup=args.warmup)


if __name__ == '__main__':
    main()
