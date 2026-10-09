"""Separate open sequence experiment: frozen learned features + fixed LightGBM.

This is supervised digits-row sequence classification, not restored CoLES,
Alpha/Age data, or a financial result. LightGBM is an explicit optional dependency.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch

from fedcore.experiments import CandidateSpec, ExperimentProtocol
from fedcore.experiments.measurement import measure_artifact
from fedcore.experiments.runner import apply_candidate, environment_manifest, model_state_hash, train_model
from fedcore.experiments.scenarios import build_open_sequence, ScenarioUnavailable


def statistical_features(x):
    # No learned feature statistics and no labels are used by this control.
    return torch.cat((x.mean(-1), x.std(-1, unbiased=False), x.amin(-1), x.amax(-1), x[:, :, -1]), dim=1)


def _metrics(target, labels, probabilities):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
    result = {"accuracy": float(accuracy_score(target, labels)),
              "balanced_accuracy": float(balanced_accuracy_score(target, labels)),
              "macro_f1": float(f1_score(target, labels, average="macro", zero_division=0))}
    if len(np.unique(target)) == 2:
        result["roc_auc"] = float(roc_auc_score(target, probabilities[:, 1]))
    return result


def run_comparison(bundle, protocol, output_dir: Path):
    try:
        from lightgbm import LGBMClassifier
    except ImportError as error:
        raise ScenarioUnavailable("Install LightGBM explicitly for the frozen sequence downstream comparison") from error
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = copy.deepcopy(bundle.original_model)
    # Avoid nested huge BLAS/OpenMP thread pools in this small offline example.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(protocol.threads)
    try:
        training = train_model(baseline, bundle.train, bundle.task, epochs=protocol.baseline_epochs,
                               batch_size=protocol.batch_size, learning_rate=protocol.learning_rate,
                               device=protocol.device, seed=protocol.seed)
        original_hash = model_state_hash(baseline)
        frozen_candidates = (CandidateSpec("baseline"), CandidateSpec("train"),
                             CandidateSpec("svd", {"threshold": .5, "strategy": "quantile"}))
        lgbm_parameters = {"n_estimators": 50, "num_leaves": 15, "max_depth": 4, "learning_rate": .05,
                           "random_state": protocol.seed, "n_jobs": 1, "verbosity": -1,
                           "deterministic": True, "force_col_wise": True}
        manifest = {"version": 1, "status": "running", "task": bundle.manifest(), "protocol": protocol.to_dict(),
                    "environment": environment_manifest(),
                    "historical_tables_restored": False, "encoder_type": "supervised SeriesCNN, not CoLES",
                    "trained_baseline_sha256": original_hash, "baseline_training": training,
                    "downstream_parameters": lgbm_parameters, "lightgbm_version": importlib.metadata.version("lightgbm"),
                    "fixed_candidate_plan": [candidate.to_dict() for candidate in frozen_candidates], "results": []}
        prepared = []
        for candidate in frozen_candidates:
            model, evidence = apply_candidate(baseline, bundle, protocol, candidate)
            encoder = copy.deepcopy(model.features).cpu().eval()
            # Weights are frozen before downstream training. Neither validation nor test labels train the encoder.
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)
            prepared.append((candidate.method, encoder, model_state_hash(encoder), evidence))
        prepared.append(("statistical_features", None, None, {"implementation": "mean/std/min/max/last; no learned encoder"}))
        if model_state_hash(baseline) != original_hash:
            raise RuntimeError("A candidate mutated the shared baseline")
        # Candidate plan and LightGBM parameters are fixed above; no test-adaptive selection.
        manifest["selection_frozen"] = True
        for name, encoder, encoder_hash, evidence in prepared:
            def encode(x):
                with torch.inference_mode():
                    return (statistical_features(x) if encoder is None else encoder(x)).numpy()
            train_x = encode(bundle.train.x)
            downstream = LGBMClassifier(**lgbm_parameters).fit(train_x, bundle.train.y.numpy())
            validation_x, test_x = encode(bundle.validation.x), encode(bundle.test.x)
            result = {"method": name, "status": "succeeded", "encoder_sha256": encoder_hash,
                      "evidence": evidence, "validation": _metrics(bundle.validation.y.numpy(), downstream.predict(validation_x), downstream.predict_proba(validation_x)),
                      "test": _metrics(bundle.test.y.numpy(), downstream.predict(test_x), downstream.predict_proba(test_x))}
            if encoder is not None:
                result["encoder_measurement"] = measure_artifact(encoder, bundle.validation.x[:1], output_dir / name / "encoder",
                                                                 device="cpu", repeats=protocol.measurement_repeats, warmup=protocol.warmup)
                if result["encoder_measurement"]["status"] != "succeeded":
                    result.update(status=result["encoder_measurement"]["status"],
                                  reason=result["encoder_measurement"].get("reason", "Encoder artifact measurement failed"))
                if model_state_hash(encoder) != encoder_hash:
                    raise RuntimeError("Frozen encoder weights changed during downstream evaluation")
            model_path = output_dir / f"{name}_lightgbm.txt"
            downstream.booster_.save_model(str(model_path))
            sample = bundle.validation.x[:1]
            for _ in range(protocol.warmup):
                downstream.predict(encode(sample))
            latencies = []
            for _ in range(protocol.measurement_repeats):
                started = time.perf_counter()
                downstream.predict(encode(sample))
                latencies.append((time.perf_counter() - started) * 1000)
            result["pipeline_measurement"] = {"device": "cpu", "batch_size": 1, "scope": "preprocessed tensor -> encoder/statistics -> LightGBM labels",
                                               "latency_samples_ms": latencies, "latency_p50_ms": float(np.median(latencies)),
                                               "latency_p95_ms": float(np.quantile(latencies, .95)), "downstream_file_bytes": model_path.stat().st_size}
            manifest["results"].append(result)
        manifest["status"] = "completed" if all(result["status"] == "succeeded" for result in manifest["results"]) else "completed_with_failures"
        (output_dir / "sequence_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return manifest
    finally:
        torch.set_num_threads(previous_threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("work/open_sequence"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("A genuinely trained common encoder requires --epochs > 0")
    protocol = ExperimentProtocol(seed=args.seed, baseline_epochs=args.epochs, finetune_epochs=args.finetune_epochs,
                                  batch_size=64, learning_rate=.001)
    manifest = run_comparison(build_open_sequence(args.seed), protocol, args.output_dir)
    print(json.dumps({"status": manifest["status"], "output_dir": str(args.output_dir), "historical_tables_restored": False}))


if __name__ == "__main__":
    main()
