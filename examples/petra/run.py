"""Train a common real-data baseline and compare isolated compression candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from fedcore.experiments.scenarios import (
    ScenarioUnavailable, build_cv_digits, build_cv_dataset, build_tabular, build_ts_regression,
    build_ts_classification, build_forecasting, build_open_sequence, build_financial,
    build_language_model,
)


def main(argv=None, *, default_scenario=None, default_methods=None, default_cv_dataset="digits"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default=default_scenario or "tabular",
                        choices=("cv", "tabular", "ts_regression", "ts_classification", "forecasting", "open_sequence", "financial", "lm"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-dataset", choices=("digits", "cifar10", "imagenette"), default=default_cv_dataset)
    parser.add_argument("--architecture", choices=("resnet18", "resnet50"), default="resnet18")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--max-samples", type=int, help="Explicit real-data pilot subset; omitted means full dataset")
    parser.add_argument("--storage-dir", type=Path,
                        help="File-backed CV tensors; default is output-dir/input_storage")
    parser.add_argument("--max-materialized-mib", type=int, default=256,
                        help="Declared in-memory CV materialization limit")
    parser.add_argument("--epochs", type=int, default=20, help="Actual common baseline training epochs")
    parser.add_argument("--baseline-checkpoint", type=Path,
                        help="Verified baseline.json; all original training settings must match")
    parser.add_argument("--finetune-epochs", type=int, default=3, help="Equal candidate update budget")
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("work/petra_results"))
    parser.add_argument("--methods", nargs="+", default=default_methods or ["baseline", "train", "svd"],
                        choices=("baseline", "train", "pruning", "structural_pruning", "svd", "ptq", "qat"))
    parser.add_argument("--rank-ratio", type=float, default=.5)
    parser.add_argument("--pruning-ratio", type=float, default=.3)
    parser.add_argument("--quality-tolerance", type=float, default=.05)
    parser.add_argument("--quality-scale", type=float,
                        help="Frozen objective scale; default regression/forecasting MSE uses train target variance, other tasks 1")
    parser.add_argument("--cost-scale", type=float, default=1048576.)
    parser.add_argument("--hypervolume-reference", type=float, nargs=2, default=[2., 2.])
    parser.add_argument("--minimum-baseline-quality", type=float,
                        help="Freeze an acceptable validation accuracy or maximum loss before running")
    parser.add_argument("--measurement-repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--data", type=Path, help="Dataset directory, forecasting CSV, financial NPZ, or LM document JSON")
    parser.add_argument("--access-manifest", type=Path)
    parser.add_argument("--context", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=6)
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be positive: these examples require a trained common baseline")
    if args.max_materialized_mib < 1:
        parser.error("--max-materialized-mib must be positive")
    if not 0 < args.rank_ratio <= 1 or not 0 < args.pruning_ratio < 1:
        parser.error("--rank-ratio must be in (0,1]; --pruning-ratio must be in (0,1)")
    preparation_started = perf_counter()
    try:
        if args.scenario == "cv":
            if args.cv_dataset == "digits":
                bundle = build_cv_digits(args.seed)
            else:
                if args.data is None:
                    raise ScenarioUnavailable("CIFAR10/ImageNette requires --data with local original data")
                bundle = build_cv_dataset(args.cv_dataset, args.data, architecture=args.architecture, seed=args.seed,
                                          resolution=args.resolution, max_samples=args.max_samples, dataset_revision=args.dataset_revision,
                                          storage_dir=args.storage_dir or args.output_dir / "input_storage",
                                          max_materialized_bytes=args.max_materialized_mib * 1024 * 1024)
        elif args.scenario == "tabular":
            bundle = build_tabular(args.seed)
        elif args.scenario == "open_sequence":
            bundle = build_open_sequence(args.seed)
        elif args.scenario == "ts_regression":
            bundle = build_ts_regression(args.data or Path("datasets/time_series_regression/multi_dim/AppliancesEnergy"), args.seed)
        elif args.scenario == "ts_classification":
            bundle = build_ts_classification(args.data or Path("datasets/time_series_classification/one_dim/CinCECGTorso"), args.seed)
        elif args.scenario == "forecasting":
            if args.data is None:
                raise ScenarioUnavailable("--data must point to the original UCI energydata_complete.csv")
            bundle = build_forecasting(args.data, context=args.context, horizon=args.horizon, seed=args.seed)
        elif args.scenario == "financial":
            if args.data is None or args.access_manifest is None:
                raise ScenarioUnavailable("Financial data requires --data and --access-manifest; Alpha/Age access is not restored")
            bundle = build_financial(args.data, args.access_manifest, args.seed)
        else:
            if args.data is None:
                raise ScenarioUnavailable("LM requires --data with licensed independent documents, source and revision")
            bundle = build_language_model(args.data, args.seed)
    except (ScenarioUnavailable, ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    from fedcore.experiments import CandidateSpec, ExperimentProtocol, ExperimentRunner
    quality_scale = args.quality_scale
    if quality_scale is None:
        quality_scale = max(float(bundle.train.y.float().var(unbiased=False)), 1e-8) if bundle.task in ("regression", "forecasting") else 1.
    protocol = ExperimentProtocol(seed=args.seed, batch_size=args.batch_size,
                                  baseline_epochs=args.epochs, finetune_epochs=args.finetune_epochs,
                                  learning_rate=args.lr, quality_tolerance=args.quality_tolerance,
                                  minimum_baseline_quality=args.minimum_baseline_quality,
                                  quality_scale=quality_scale, cost_scale=args.cost_scale,
                                  hypervolume_reference=tuple(args.hypervolume_reference), repeat_seeds=(args.seed,),
                                  device=args.device, measurement_repeats=args.measurement_repeats, warmup=args.warmup)
    # quantile keeps ceil(ratio * original rank); explained_variance is a different policy.
    candidates = [CandidateSpec(method, {"threshold": args.rank_ratio, "strategy": "quantile"} if method == "svd" else
                              {"amount": args.pruning_ratio} if method == "pruning" else
                              {"pruning_ratio": args.pruning_ratio} if method == "structural_pruning" else {}) for method in args.methods]
    preparation_seconds = perf_counter() - preparation_started
    result = ExperimentRunner(bundle, protocol, args.output_dir,
                              baseline_checkpoint=args.baseline_checkpoint,
                              external_preparation_seconds=preparation_seconds).run(candidates)
    # The full record is saved by the runner. Keep CLI output concise and secret-free.
    print(json.dumps({"output_dir": str(args.output_dir), "scenario": args.scenario,
                      "seed": args.seed, "manifest_status": result.get("status", "recorded")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
