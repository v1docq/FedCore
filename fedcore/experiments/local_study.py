"""Prospective, bounded local replication series; no confirmatory claim.

Run ``python -m fedcore.experiments.local_study --group comparisons``.
The plan freezes candidates and repeats before any final test scores exist.
Failures are retained and make the CLI return nonzero; failed seeds are never
silently replaced. Groups execute sequentially; other system work is not locked.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import math
from pathlib import Path
from time import perf_counter

import torch

from fedcore.tools.atomic_json import write_atomic_json
from .protocol import CandidateSpec, ExperimentProtocol, ProtocolError
from .runner import ExperimentRunner, environment_manifest
from .scenarios import build_tabular, build_ts_regression, build_open_sequence


def summarize_pairs(manifests):
    """Descriptive within-seed differences; failed/missing pairs stay explicit."""
    seen = set()
    groups = {}
    for manifest in manifests:
        dataset = manifest['data']['metadata'].get('scenario', 'unknown')
        seed = manifest['protocol']['seed']
        key = dataset, seed
        if key in seen:
            raise ProtocolError('Duplicate dataset/seed cannot be independent repeats')
        seen.add(key)
        records = manifest.get('candidates', ())
        baseline = next((r for r in records if r['configuration']['method'] == 'baseline'
                         and r['status'] == 'succeeded'), None)
        for record in records:
            method = record['configuration']['method']
            if method == 'baseline':
                continue
            group = groups.setdefault((dataset, method), {'paired_seeds': [], 'missing_seeds': [],
                'validation_differences': [], 'file_size_change_percent': [], 'latency_change_percent': []})
            if baseline is None or record['status'] != 'succeeded':
                group['missing_seeds'].append(seed)
                continue
            left, right = baseline['validation'], record['validation']
            if left['metric'] != right['metric'] or left['direction'] != right['direction']:
                raise ProtocolError('Paired quality metrics differ')
            difference = right['value'] - left['value']
            if not math.isfinite(difference):
                raise ProtocolError('Nonfinite paired difference')
            group['paired_seeds'].append(seed)
            group['validation_differences'].append(difference)
            for field, output in (('file_bytes', 'file_size_change_percent'),
                                  ('latency_p50_ms', 'latency_change_percent')):
                initial = baseline['measurement']['metrics'].get(field)
                final = record['measurement']['metrics'].get(field)
                group[output].append((final / initial - 1) * 100 if initial and final is not None else None)
            group['metric'], group['direction'] = left['metric'], left['direction']
    summaries = []
    for (dataset, method), group in sorted(groups.items()):
        values = group['validation_differences']
        mean = sum(values) / len(values) if values else None
        sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else None
        summaries.append({'dataset': dataset, 'method': method, **group,
                          'mean_validation_difference': mean, 'sample_sd': sd,
                          'inference': 'descriptive paired pilot; repeat count/power not established'})
    return summaries


def run_local_study(root, *, group='comparisons', seeds=(41, 42, 43, 44, 45),
                    baseline_epochs=20, finetune_epochs=2, ts_data=None):
    root = Path(root)
    if group not in ('comparisons', 'mechanisms', 'sequence'):
        raise ProtocolError('Unknown local study group')
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ProtocolError('Declare unique nonnegative repeat seeds')
    if type(baseline_epochs) is not int or baseline_epochs < 1 or type(finetune_epochs) is not int or finetune_epochs < 1:
        raise ProtocolError('These local studies require actual training and finetuning')
    if (root / 'study_plan.json').exists():
        raise ProtocolError('A prospective plan already exists; use a new directory')
    environment = environment_manifest()
    candidates = {
        'tabular': (CandidateSpec('baseline'), CandidateSpec('train'),
            CandidateSpec('svd', {'threshold': .5, 'strategy': 'quantile'}),
            CandidateSpec('structural_pruning', {'pruning_ratio': .3}),
            CandidateSpec('ptq'), CandidateSpec('qat')),
        'ts_regression': (CandidateSpec('baseline'), CandidateSpec('train'),
            CandidateSpec('svd', {'threshold': .5, 'strategy': 'quantile'})),
    }
    plan = {'group': group, 'seeds': list(seeds), 'baseline_epochs': baseline_epochs,
        'finetune_epochs': finetune_epochs, 'device': 'cpu', 'threads': 1,
        'batch_size': 64, 'learning_rate': .001,
        'measurement_repeats': 10 if group == 'mechanisms' else 30,
        'warmup': 3 if group == 'mechanisms' else 10,
        'system_load': 'not monitored; one series at a time does not establish system idleness',
        'candidate_plan': {name: [c.to_dict() for c in cs] for name, cs in candidates.items()},
        'rank_ablation': {'layer': '0', 'ranks': [5, 10, 15], 'ridge': 1e-6},
        'repeat_status': 'pilot; confirmatory repeat count requires paired variance',
        'quality_tolerance': {'classification_absolute_loss': .05, 'regression_absolute_MSE': .05},
        'baseline_gate': {'tabular_validation_accuracy_min': .90,
                          'ts': 'no published strong-baseline claim; local implementation pilot'},
        'scope': 'real local replicates; no H1-H4 superiority claim',
        'environment': environment}
    write_atomic_json(root / 'study_plan.json', plan)
    manifests, runs, failures = [], [], []
    started = perf_counter()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for seed in seeds:
            protocol = ExperimentProtocol(seed=seed, baseline_epochs=baseline_epochs,
                finetune_epochs=finetune_epochs, batch_size=64, learning_rate=.001,
                measurement_repeats=plan['measurement_repeats'], warmup=plan['warmup'], repeat_seeds=tuple(seeds),
                repeat_rationale=plan['repeat_status'])
            items = ('tabular', 'ts_regression') if group == 'comparisons' else (group,)
            for item in items:
                if environment_manifest()['source_sha256'] != environment['source_sha256']:
                    raise RuntimeError('Source changed during the frozen local study')
                directory = root / item / f'seed{seed}'
                preparation_started = perf_counter()
                try:
                    if group == 'comparisons':
                        bundle = build_tabular(seed) if item == 'tabular' else build_ts_regression(
                            Path(ts_data or 'datasets/time_series_regression/multi_dim/AppliancesEnergy'), seed)
                        preparation = perf_counter() - preparation_started
                        p = replace(protocol, minimum_baseline_quality=.9) if item == 'tabular' else replace(
                            protocol, quality_scale=max(float(bundle.train.y.float().var(unbiased=False)), 1e-8))
                        from .baseline import prepare_baseline
                        _, training = prepare_baseline(bundle, p, root / 'shared_baselines' / item / f'seed{seed}',
                                                       environment=environment)
                        manifest = ExperimentRunner(bundle, p, directory,
                            external_preparation_seconds=preparation,
                            baseline_checkpoint=training['checkpoint']).run(candidates[item])
                        manifests.append(manifest)
                        failed = [r['candidate_id'] for r in manifest['candidates'] if r['status'] != 'succeeded']
                        if failed:
                            failures.append({'group': item, 'seed': seed, 'reason': 'candidate failures', 'ids': failed})
                    elif group == 'sequence':
                        from examples.petra.sequence import run_comparison
                        manifest = run_comparison(build_open_sequence(seed), protocol, directory)
                        if manifest['status'] != 'completed':
                            failures.append({'group': item, 'seed': seed, 'reason': manifest['status']})
                    else:
                        from .pilot import ablation_pilot
                        # The helper fixes the same recipe and records each suite's baseline separately.
                        ablation_pilot(directory, seed=seed, baseline_epochs=baseline_epochs,
                                       finetune_epochs=finetune_epochs, ranks=(5, 10, 15),
                                       repeat_seeds=tuple(seeds))
                    runs.append({'group': item, 'seed': seed, 'status': 'completed', 'path': str(directory)})
                except Exception as error:
                    failure = {'group': item, 'seed': seed, 'status': 'failed',
                               'error_type': type(error).__name__, 'reason': str(error), 'path': str(directory)}
                    failures.append(failure)
                    runs.append(failure)
                print(f'{item} seed={seed}: {runs[-1]["status"]}', flush=True)
                write_atomic_json(root / 'study_summary.json', {'status': 'running', 'runs': runs, 'failures': failures})
    finally:
        torch.set_num_threads(previous_threads)
    summary = {'status': 'completed' if not failures else 'completed_with_failures',
        'runs': runs, 'failures': failures, 'wall_seconds': perf_counter() - started,
        'paired_results': summarize_pairs(manifests), 'claim': plan['scope']}
    write_atomic_json(root / 'study_summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=('comparisons', 'mechanisms', 'sequence'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[41, 42, 43, 44, 45])
    parser.add_argument('--baseline-epochs', type=int, default=20)
    parser.add_argument('--finetune-epochs', type=int, default=2)
    parser.add_argument('--ts-data', type=Path)
    args = parser.parse_args(argv)
    summary = run_local_study(args.output_dir, group=args.group, seeds=tuple(args.seeds),
        baseline_epochs=args.baseline_epochs, finetune_epochs=args.finetune_epochs, ts_data=args.ts_data)
    return 0 if summary['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
