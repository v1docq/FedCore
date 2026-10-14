"""Recompute descriptive pilot comparisons without importing the ML runtime."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     indent=2, allow_nan=False) + "\n").encode("utf-8"))


def describe(values):
    values = list(values)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Timing ratios require finite positive observations")
    return {"n": len(values), "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def quality_row(checkpoint, baseline, dense):
    row = {"tensor_bytes": checkpoint["storage"]["tensor_bytes"],
           "tensor_retained_fraction": checkpoint["storage"]["tensor_bytes"] / baseline["storage"]["tensor_bytes"],
           "macs": checkpoint["macs"]["per_sample"],
           "mac_retained_fraction": checkpoint["macs"]["per_sample"] / baseline["macs"]["per_sample"],
           "artifact_file_bytes": checkpoint["artifact_file_bytes"]}
    for role in ("validation", "test"):
        q = checkpoint["quality"][role]
        row.update({f"{role}_accuracy": q["value"],
                    f"{role}_delta_baseline_pp": 100 * (q["value"] - baseline["quality"][role]["value"]),
                    f"{role}_delta_dense_control_pp": 100 * (q["value"] - dense["quality"][role]["value"]),
                    f"{role}_macro_f1": q["macro_f1"], f"{role}_ce": q["cross_entropy"],
                    f"{role}_logit_relative_mse": q["logit_error"]["relative_mse"]})
    return row


def timing_analysis(sessions):
    grouped = {}
    for session in sessions:
        job = session["job"]
        key = (job["kind"], job["subject"] if job["kind"] == "artifact" else "conv2",
               job["device"], job["batch"], job["block"])
        label = job["side"] if job["kind"] == "artifact" else job["subject"]
        group = grouped.setdefault(key, {})
        if label in group:
            raise ValueError("Duplicate session within one block")
        group[label] = session
    block_rows = []
    for (kind, subject, device, batch, block), group in sorted(grouped.items()):
        comparisons = (("baseline", "candidate"),) if kind == "artifact" else (
            ("original_dense", "factor_r8"), ("materialized_r8", "factor_r8"),
            ("original_dense", "materialized_r8"))
        for reference, candidate in comparisons:
            row = {"kind": kind, "subject": subject, "device": device,
                   "batch": batch, "block": block, "comparison": f"{reference}/{candidate}"}
            a, b = group.get(reference), group.get(candidate)
            row["valid"] = a is not None and b is not None and a["status"] == b["status"] == "succeeded"
            row["reasons"] = {label: group[label].get("noise_reasons", []) for label in (reference, candidate) if label in group}
            if row["valid"]:
                if a["input_sha256"] != b["input_sha256"]:
                    raise ValueError("A paired comparison requires identical inputs")
                for scope, field in (("wall", "wall"), ("full_path", "full_path"), ("device", "cuda_event")):
                    if a[field] is not None and b[field] is not None:
                        for percentile in ("p50_ms", "p95_ms"):
                            value_a, value_b = a[field][percentile], b[field][percentile]
                            if min(value_a, value_b) <= 0 or not all(map(math.isfinite, (value_a, value_b))):
                                raise ValueError("Invalid timing summary")
                            row[f"{scope}_{percentile}_reference"] = value_a
                            row[f"{scope}_{percentile}_candidate"] = value_b
                            row[f"{scope}_{percentile}_ratio"] = value_a / value_b
                throughput_a = statistics.median(x["samples_per_second"] for x in a["blocked_throughput"])
                throughput_b = statistics.median(x["samples_per_second"] for x in b["blocked_throughput"])
                row["throughput_candidate_over_reference"] = throughput_b / throughput_a
            block_rows.append(row)
    groups = {}
    for row in block_rows:
        key = row["kind"], row["subject"], row["device"], row["batch"], row["comparison"]
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, rows in sorted(groups.items()):
        good = [row for row in rows if row["valid"]]
        summary = dict(zip(("kind", "subject", "device", "batch", "comparison"), key))
        summary.update(planned_blocks=len(rows), valid_blocks=len(good), invalid_blocks=len(rows)-len(good))
        ratio_fields = sorted({field for row in good for field in row if field.endswith("_ratio")})
        ratio_fields += ["throughput_candidate_over_reference"] if good else []
        for field in ratio_fields:
            summary[field] = describe(row[field] for row in good if field in row)
        summary["statistical_scope"] = "descriptive within-machine pilot; three blocks, one trained seed; no confidence interval"
        summaries.append(summary)
    return block_rows, summaries


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
                         if isinstance(value, (dict, list)) else value for key, value in row.items()})
    Path(path).write_bytes(buffer.getvalue().encode("utf-8"))


def analyze(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    protocol = read(source / "protocol.json")
    transforms = read(source / "transforms.json")
    recovery = read(source / "recovery.json")
    timing = read(source / "timing.json")
    ledger = read(source / "ledger.json")
    planned_sessions = {job["session_id"] for job in protocol["timing_schedule"]}
    observed_sessions = [row["job"]["session_id"] for row in timing["sessions"]]
    if len(observed_sessions) != 132 or len(set(observed_sessions)) != 132 or set(observed_sessions) != planned_sessions:
        raise ValueError("The complete pilot requires all 132 predeclared sessions")
    curves = recovery["curves"]
    if len(curves) != 10 or recovery["student_updates"] != 4000 or recovery["teacher_forwards"] != 2000:
        raise ValueError("The complete pilot requires ten recovery curves and the fixed work counts")
    dense = {curve["objective"]: {row["update"]: row for row in curve["checkpoints"]}
             for curve in curves if curve["name"] == "dense"}
    baseline = dense["ce"][0]
    transform_rows = []
    for record in transforms["rows"]:
        row = {"case_id": record["spec"]["case_id"], "status": record["status"], "spec": record["spec"]}
        if record["status"] == "succeeded":
            row.update(quality_row(record, baseline, baseline))
            row["isolated_error"] = record["isolated_error"]
            row["propagated_error"] = record["propagated_error"]
        else:
            row.update(reason=record.get("reason"), error_type=record.get("error_type"))
        transform_rows.append(row)
    recovery_rows = []
    for curve in curves:
        for checkpoint in curve["checkpoints"]:
            row = {"name": curve["name"], "objective": curve["objective"], "update": checkpoint["update"]}
            row.update(quality_row(checkpoint, baseline, dense[curve["objective"]][checkpoint["update"]]))
            row["model_hash"] = checkpoint["model_hash"]
            recovery_rows.append(row)
    block_rows, timing_summaries = timing_analysis(timing["sessions"])
    result = {"schema": 1, "analysis_script_git_lf_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
              "inputs_sha256": {name: digest(source/name) for name in ("protocol.json", "transforms.json", "recovery.json", "timing.json", "ledger.json")},
              "scientific_source_sha256": protocol["source"]["sha256"],
              "baseline": baseline, "selected_weighted": transforms["selected_weighted"],
              "transforms": transform_rows, "recovery": recovery_rows,
              "timing_blocks": block_rows, "timing_summaries": timing_summaries,
              "counts": {"transform_attempts": len(transform_rows), "curves": len(curves),
                         "student_updates": recovery["student_updates"], "teacher_forwards": recovery["teacher_forwards"],
                         "timing_sessions": len(timing["sessions"]),
                         "valid_timing_sessions": sum(row["status"] == "succeeded" for row in timing["sessions"]),
                         "wall_calls": sum(len(row["raw_synchronized_wall_ms"]) for row in timing["sessions"]),
                         "full_path_calls": sum(len(row["raw_full_path_ms"]) for row in timing["sessions"]),
                         "cuda_event_calls": sum(len(row["raw_cuda_event_ms"]) for row in timing["sessions"]),
                         "charged_compute_seconds": ledger["charged_seconds"]},
              "limits": {"quality_test": "seen development test", "timing": "descriptive pilot only", "cluster_used": False}}
    write_json(destination/"analysis.json", result)
    for name, rows in (("conv.csv", transform_rows), ("recovery.csv", recovery_rows),
                       ("timing-blocks.csv", block_rows), ("timing-summary.csv", timing_summaries)):
        write_csv(destination/name, rows)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.input, args.output)
    print(json.dumps(result["counts"], sort_keys=True))
