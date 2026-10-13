"""Recompute inspectable compression comparisons from per-seed raw records.

Uses only the standard library. Runtime observations and analytical MAC counts
remain separate; seed spread is descriptive, not a confidence interval.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def analytical_macs(architecture, ranks=None):
    """Per-example multiply-accumulates; excludes bias, activations and loading."""
    ranks = ranks or {}
    if architecture == "mlp":
        operators = {"0": (128, 30, 1), "2": (64, 128, 1), "4": (2, 64, 1)}
    elif architecture == "cnn":
        operators = {"features.0": (16, 9, 64), "features.2": (16, 144, 64),
                     "classifier.0": (64, 1024, 1), "classifier.2": (10, 64, 1)}
    else:
        raise ValueError("Unknown fixed comparison architecture")
    if set(ranks) - set(operators):
        raise ValueError("Unknown analytical target")
    total = 0
    for path, (outputs, inputs, applications) in operators.items():
        if path in ranks:
            rank = ranks[path]
            if type(rank) is not int or not 1 <= rank <= min(outputs, inputs):
                raise ValueError("Rank outside declared operator dimensions")
            total += rank * (outputs + inputs) * applications
        else:
            total += outputs * inputs * applications
    return total


def describe(values):
    """Paired seed observations, without a significance or population claim."""
    values = list(values)
    if not values:
        return {"n": 0, "mean": None, "sample_sd": None, "min": None, "max": None}
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Non-finite analysis input")
    return {"n": len(values), "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values)}


def paired_delta(candidate, baseline, field="accuracy"):
    if (candidate["architecture"], candidate["seed"]) != (baseline["architecture"], baseline["seed"]):
        raise ValueError("A comparison must use its own paired baseline")
    return candidate[field] - baseline[field]


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})


def analyze(input_dir, output_dir):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = _read(input_dir / "protocol.json")
    rows, layers, resources, source_files = [], [], [], {}
    for path in sorted(input_dir.glob("*-seed-*/run.json")):
        run = _read(path)
        source_files[path.relative_to(input_dir).as_posix()] = _sha(path)
        directory = path.parent
        resource_file = directory / "resources.json"
        if resource_file.exists():
            resources.append({"architecture": run["architecture"], "seed": run["seed"],
                              "resource_observations": _read(resource_file)})
        baseline = run["baseline"]
        for record in (baseline, *run["records"]):
            key = f"{run['architecture']}-s{run['seed']}-{record['case_id']}"
            row = {"row_id": key, "architecture": run["architecture"], "seed": run["seed"],
                   "case_id": record["case_id"], "method": record["method"],
                   "cohort": record["cohort"], "level": record.get("level"),
                   "ranks": record.get("ranks", {}), "status": record["status"],
                   "reason": record.get("reason"), "baseline_hash": run["baseline_hash"],
                   "baseline_gate_passed": run["gate"]["passed"]}
            if record["status"] == "succeeded":
                expected_hash = run["baseline_hash"]
                if (not isinstance(expected_hash, str) or not expected_hash
                        or record.get("source_baseline_hash") != expected_hash):
                    raise ValueError("Candidate and trained baseline hashes disagree")
                storage = record["storage"]
                row.update(tensor_bytes=storage["tensor_bytes"],
                           tensor_retained_fraction=storage["tensor_bytes"] / baseline["storage"]["tensor_bytes"],
                           tensor_saved_percent=100 * (1 - storage["tensor_bytes"] / baseline["storage"]["tensor_bytes"]),
                           artifact_bytes=storage["artifact_file_bytes"],
                           artifact_retained_fraction=storage["artifact_file_bytes"] / baseline["storage"]["artifact_file_bytes"],
                           descriptor_bytes=storage["artifact_descriptor_json_bytes"],
                           checkpoint_bytes=record["checkpoint"]["file_bytes"],
                           checkpoint_metadata_json_bytes=record["checkpoint"]["metadata_json_bytes"],
                           compression_seconds=record.get("compression_seconds"),
                           total_candidate_seconds=record.get("total_candidate_seconds"),
                           analytical_macs=analytical_macs(run["architecture"], record.get("ranks")),
                           analytical_mac_fraction=analytical_macs(run["architecture"], record.get("ranks")) /
                               analytical_macs(run["architecture"]))
                for role in ("train", "calibration", "validation", "test"):
                    quality = record["quality"][role]
                    row.update({f"{role}_accuracy": quality["value"],
                                f"{role}_macro_f1": quality["macro_f1"],
                                f"{role}_cross_entropy": quality["cross_entropy"]})
                row["accuracy"] = row["test_accuracy"]
                if record["method"] != "baseline":
                    row["accuracy_delta_pp"] = 100 * (
                        record["quality"]["test"]["value"] - baseline["quality"]["test"]["value"])
                    row["cross_entropy_delta"] = (
                        record["quality"]["test"]["cross_entropy"] - baseline["quality"]["test"]["cross_entropy"])
                    row["macro_f1_delta"] = (
                        record["quality"]["test"]["macro_f1"] - baseline["quality"]["test"]["macro_f1"])
                    row["test_logit_relative_mse"] = record["quality"]["test"]["logit_error"]["relative_mse"]
                    row["test_teacher_kl"] = record["quality"]["test"]["teacher_to_student_kl"]
                    if storage["tensor_bytes"] != record["budget"]["whole_model_tensor_byte_budget"]:
                        raise ValueError("Measured complete storage differs from declared budget")
                    for role, operators in record["isolated_layer_errors"].items():
                        for operator, error in operators.items():
                            layers.append({"row_id": key, "architecture": run["architecture"], "seed": run["seed"],
                                           "cohort": record["cohort"], "level": record["level"], "method": record["method"],
                                           "operator": operator, "role": role, **error})
                if record.get("timing_job"):
                    timing_file = directory / record["case_id"] / "timing.json"
                    if timing_file.exists():
                        timing = _read(timing_file)
                        raw = timing["raw_inference_ms"]
                        if len(raw) != plan["timing"]["repeats"]:
                            raise ValueError("Unexpected timing sample count")
                        if not math.isclose(statistics.median(raw), timing["metrics"]["latency_p50_ms"],
                                            rel_tol=1e-12, abs_tol=1e-12):
                            raise ValueError("Stored latency median differs from raw observations")
                        row.update(timing_samples=len(raw), **timing["metrics"])
                        source_files[timing_file.relative_to(input_dir).as_posix()] = _sha(timing_file)
            rows.append(row)
    baselines = {(row["architecture"], row["seed"]): row for row in rows if row["method"] == "baseline"}
    groups = {}
    for row in rows:
        if row["method"] == "baseline":
            continue
        group_key = (row["architecture"], row["cohort"], row["level"], row["method"])
        group = groups.setdefault(group_key, [])
        group.append(row)
        if row["status"] != "succeeded":
            continue
        baseline = baselines[row["architecture"], row["seed"]]
        if not math.isclose(row["accuracy_delta_pp"], 100 * paired_delta(row, baseline), abs_tol=1e-12):
            raise ValueError("Incorrect paired quality delta")
        if "latency_p50_ms" in row and "latency_p50_ms" in baseline:
            row["observed_latency_ratio"] = row["latency_p50_ms"] / baseline["latency_p50_ms"]
            # A conditional scenario from the observed median, not a measured
            # payback or a universal deployment-speed claim.
            saving = (baseline["latency_p50_ms"] - row["latency_p50_ms"]) / 1000
            row["conditional_break_even_batch_calls"] = (
                math.ceil(row["total_candidate_seconds"] / saving) if saving > 0 else None)
    summaries = []
    fields = ("accuracy_delta_pp", "cross_entropy_delta", "macro_f1_delta", "tensor_saved_percent",
              "artifact_retained_fraction", "compression_seconds", "total_candidate_seconds",
              "test_logit_relative_mse", "test_teacher_kl", "latency_p50_ms", "latency_p95_ms",
              "observed_latency_ratio", "analytical_mac_fraction")
    for key, records in sorted(groups.items()):
        success = [row for row in records if row["status"] == "succeeded"]
        summaries.append({"architecture": key[0], "cohort": key[1], "level": key[2], "method": key[3],
                          "succeeded_seeds": [row["seed"] for row in success],
                          "failed_seeds": [row["seed"] for row in records if row["status"] != "succeeded"],
                          "tensor_bytes": sorted({row["tensor_bytes"] for row in success}),
                          **{field: describe(row[field] for row in success if row.get(field) is not None)
                             for field in fields}})
    result = {"schema": 1, "study": plan["study"], "rows": rows, "paired_summaries": summaries,
              "baselines": list(baselines.values()), "layers": layers,
              "source_files_sha256": source_files, "protocol_sha256": _sha(input_dir / "protocol.json"),
              "analysis_script_sha256": _sha(Path(__file__)),
              "counts": {"baselines": len(baselines), "candidates": len(rows) - len(baselines),
                         "succeeded_candidates": sum(row["method"] != "baseline" and row["status"] == "succeeded"
                                                     for row in rows),
                         "timing_sessions": sum("timing_samples" in row for row in rows)},
              "definitions": {"accuracy_delta_pp": "100 * (candidate test accuracy - paired baseline test accuracy)",
                  "tensor_storage": "All unique physical parameter/buffer storage, including untouched operators and bias",
                  "equal_budget": "Equal whole-model tensor storage; coincides with equal rank in these biased float32 factors",
                  "analytical_macs": "Per-example multiply-accumulates from fixed operator dimensions/ranks; excludes bias and activations",
                  "seed_spread": "Sample standard deviation and observed range across 3 paired seeds, not confidence intervals",
                  "construction_cost": plan.get("construction_timing_scope", "Observed fixed-order portfolio execution"),
                  "conditional_break_even": "ceil(total measured candidate creation/validation/export seconds / observed batch median time saving); conditional scenario, not measured payback",
                  "memory": "Sampled recursive tree RSS may double-count shared pages; not an exact allocation or resident peak"}}
    (output_dir / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                             encoding="utf-8")
    _csv(output_dir / "candidates.csv", rows)
    _csv(output_dir / "layers.csv", layers)
    _csv(output_dir / "paired.csv", summaries)
    (output_dir / "resource-observations.json").write_text(
        json.dumps(resources, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.input, args.output)
    print(json.dumps(result["counts"]))
