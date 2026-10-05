"""Bounded real-data PETRA pilots; no confirmatory superiority claim.

Run from the checkout: python -m fedcore.experiments.pilot --group search
The default search charges at most six evaluations, including the baseline,
per method/seed/task. An indivisible operation can overrun the wall-time cap.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import torch

from .protocol import CandidateSpec, ExperimentProtocol
from .runner import ExperimentRunner, environment_manifest
from .scenarios import build_cv_digits, build_tabular


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def search_pilot(root, *, seeds=(41, 42, 43), device='cpu', max_evaluations=6,
                 max_seconds=60., baseline_epochs=15):
    from .search import SearchConfig, compare_search_runs
    space = (
        CandidateSpec('train', {'epochs': 1}),
        *(CandidateSpec('svd', {'rank_ratio': ratio, 'finetune_epochs': 1})
          for ratio in (.3, .5, .7)),
        *(CandidateSpec('structural_pruning', {'pruning_ratio': ratio, 'finetune_epochs': 1})
          for ratio in (.2, .4)),
        CandidateSpec('pruning', {'amount': .3, 'finetune_epochs': 1}),
    )
    plan = {'scope': 'pilot; two real tasks, three paired seeds, three methods',
            'seeds': list(seeds), 'device': device, 'max_evaluations': max_evaluations,
            'max_seconds': max_seconds, 'baseline_epochs': baseline_epochs,
            'baseline_charged': True, 'final_test_charged_separately': True,
            'space': [item.to_dict() for item in space],
            'primary_comparison': 'validation archive hypervolume versus random at equal full budget',
            'quality_tolerance': .10, 'hypervolume_reference': [2., 2.],
            'quality_scale': 1., 'cost_scale': 1048576.,
            'repeat_count_status': 'pilot only; power and confirmatory count not established',
            'runtime_policy': 'one interpreter; method order rotates by seed; per-run work charged, process startup excluded',
            'environment': environment_manifest()}
    _save(root / 'search_plan.json', plan)
    manifests, failures = [], []
    frozen_source = plan['environment']['source_sha256']
    for task, builder, minimum in (('tabular', build_tabular, .90), ('digits', build_cv_digits, .85)):
        for seed_index, seed in enumerate(seeds):
            preparation_started = perf_counter()
            bundle = builder(seed)
            preparation_seconds = perf_counter() - preparation_started
            protocol = ExperimentProtocol(seed=seed, device=device, baseline_epochs=baseline_epochs,
                finetune_epochs=1, batch_size=64, learning_rate=.001, quality_tolerance=.10,
                minimum_baseline_quality=minimum, measurement_repeats=10, warmup=3,
                repeat_seeds=tuple(seeds), primary_comparison=plan['primary_comparison'],
                repeat_rationale=plan['repeat_count_status'])
            methods = ('random', 'evolution', 'bayesian')
            offset = seed_index % len(methods)
            for method in methods[offset:] + methods[:offset]:
                directory = root / task / f'seed{seed}-{method}'
                runner = ExperimentRunner(bundle, protocol, directory,
                                          external_preparation_seconds=preparation_seconds)
                if runner.environment['source_sha256'] != frozen_source:
                    raise RuntimeError('Source changed during the frozen search comparison')
                config = SearchConfig(method=method, seed=seed, max_evaluations=max_evaluations,
                                      max_seconds=max_seconds, population_size=2)
                try:
                    manifest = runner.search(space, config)
                    manifests.append(manifest)
                    print(json.dumps({'task': task, 'seed': seed, 'method': method,
                                      'status': manifest['search']['status'],
                                      'evaluations': manifest['search']['charged_evaluations']}), flush=True)
                except Exception as error:
                    failures.append({'task': task, 'seed': seed, 'method': method,
                                     'error_type': type(error).__name__, 'reason': str(error),
                                     'manifest': str(directory / 'manifest.json')})
                    print(json.dumps(failures[-1]), flush=True)
    summary = compare_search_runs(manifests)
    summary.update(failures=failures, planned_runs=2 * len(seeds) * 3,
                   completed_manifests=len(manifests), claim='H4 remains unconfirmed')
    _save(root / 'search_comparison.json', summary)
    return summary


def ablation_pilot(root, *, seed=42, baseline_epochs=20):
    from .ablations import (RegularizationVariant, run_order_ablation, run_rank_ablation,
                           run_regularization_ablation, run_training_cost_controls,
                           run_validity_ablation)
    from fedcore.algorithm.quantization.quantizers import validate_quantization_request
    bundle = build_tabular(seed)
    protocol = ExperimentProtocol(seed=seed, baseline_epochs=baseline_epochs, finetune_epochs=2,
        batch_size=64, learning_rate=.001, measurement_repeats=10, warmup=3,
        repeat_seeds=(seed,), repeat_rationale='One real-data implementation pilot; no population inference')
    plan = {'protocol': protocol.to_dict(), 'dataset': bundle.metadata['dataset'],
            'seed': seed, 'layer': '0', 'ranks': [8, 15], 'ridge': 1e-6,
            'claim': 'Single implementation pilot; empirical superiority not established',
            'environment': environment_manifest()}
    _save(root / 'ablation_plan.json', plan)
    results = {}
    results['rank'] = run_rank_ablation(bundle, protocol, root / 'rank', layer_path='0', ranks=(8, 15), ridge=1e-6)
    variants = (RegularizationVariant('none', 0.), RegularizationVariant('hoyer', .001),
                RegularizationVariant('orthogonal', .001, 'rank'),
                RegularizationVariant('orthogonal', .015, 'rank_squared'),
                RegularizationVariant('norm', .001))
    results['regularization'] = run_regularization_ablation(bundle, protocol, root / 'regularization',
                                                          variants=variants, layer_path='0', rank=15)
    results['order'] = run_order_ablation(bundle, protocol, root / 'order', rank=1)
    results['training'] = run_training_cost_controls(bundle, protocol, root / 'training', lora_rank=8)
    proposals = (CandidateSpec('ptq', {'mode': 'dynamic', 'backend': 'fbgemm'}),
                 CandidateSpec('ptq', {'mode': 'dynamic', 'backend': 'none'}))
    def validator(candidate):
        try:
            validate_quantization_request(bundle.original_model, bundle.calibration.x[:1],
                candidate.parameters['mode'], candidate.parameters['backend'], torch.qint8)
            return True
        except ValueError:
            return False
    results['validity'] = run_validity_ablation(bundle, replace(protocol, finetune_epochs=0),
                                              root / 'validity', proposals=proposals, validator=validator)
    _save(root / 'ablation_summary.json', {'experiments': list(results), 'claim': plan['claim']})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=('search', 'ablations'), required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('work/petra_pilot'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--seeds', type=int, nargs='+', default=[41, 42, 43])
    parser.add_argument('--baseline-epochs', type=int, default=15)
    parser.add_argument('--max-evaluations', type=int, default=6)
    parser.add_argument('--max-seconds', type=float, default=60.)
    args = parser.parse_args(argv)
    if args.baseline_epochs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Positive baseline epochs and unique seeds are required')
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        if args.group == 'search':
            search_pilot(args.output_dir, seeds=tuple(args.seeds), device=args.device,
                         max_evaluations=args.max_evaluations, max_seconds=args.max_seconds,
                         baseline_epochs=args.baseline_epochs)
        else:
            if args.device != 'cpu':
                parser.error('The paired QAT/float training ablations use CPU')
            ablation_pilot(args.output_dir, seed=args.seeds[0], baseline_epochs=args.baseline_epochs)
    finally:
        torch.set_num_threads(threads)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
