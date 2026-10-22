"""Portable, standard-library-only reconstruction of the third SVD packet.

Ratios describe temporal blocks in one process; no independence or population
confidence interval is asserted. A failed control invalidates a whole profile.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

GRAPHS = ("svd_factor", "svd_materialized", "weighted_factor", "weighted_materialized", "dense")
METRICS = ("wall_ms", "full_path_ms", "cuda_event_ms", "throughput_seconds")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")


def metrics_for(device):
    return METRICS if str(device).startswith("cuda") else tuple(m for m in METRICS if m != "cuda_event_ms")


def values(side, metric, stage, device):
    raw = side.get(metric)
    expected = 1 if metric == "throughput_seconds" else (10 if stage == "pilot" else 20)
    if not isinstance(raw, list) or len(raw) != expected:
        raise ValueError(f"Missing or incomplete {metric}: expected {expected} observations")
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in raw):
        raise ValueError(f"Nonpositive/nonfinite raw {metric}")
    return raw


def unit_ratios(record, stage, device):
    unit = record["unit"]
    if sorted(unit["order"]) != ["a", "b"] or set(record.get("sides", {})) != {"a", "b"}:
        raise ValueError("Both recorded sides and one complete AB/BA order are required")
    left, right = record["sides"]["a"], record["sides"]["b"]
    if left.get("input_sha256") is None or left.get("input_sha256") != right.get("input_sha256"):
        raise ValueError("The exact input tensor must match on both sides")
    for side in (left, right):
        if not side.get("source_sha256"):
            raise ValueError("Source artifact identity unavailable")
        if not str(device).startswith("cuda") and side.get("cuda_event_ms") != []:
            raise ValueError("CPU CUDA-event series must explicitly be unavailable (empty)")
    if unit["kind"] == "control" and left["source_sha256"] != right["source_sha256"]:
        raise ValueError("Identical-graph control has different artifacts")
    return {metric: statistics.median(values(left, metric, stage, device)) /
                    statistics.median(values(right, metric, stage, device))
            for metric in metrics_for(device)}


def describe(raw):
    return {"n": len(raw), "median": statistics.median(raw) if raw else None,
            "min": min(raw) if raw else None, "max": max(raw) if raw else None}


def control_gate(records, stage, device):
    """Fixed complete-set operational gate, never select passing metric subsets."""
    if stage not in ("pilot", "main"):
        raise ValueError("Unknown stage")
    n, minimum = (8, 7) if stage == "pilot" else (20, 18)
    controls = [r for r in records if r.get("unit", {}).get("kind") == "control"]
    expected_ids = {(graph, block) for graph in GRAPHS for block in range(n)}
    keys = [(r["unit"]["subject"], r["unit"]["block"]) for r in controls]
    reasons, checks = [], []
    if len(keys) != len(set(keys)) or set(keys) != expected_ids:
        reasons.append("Control set is missing, duplicated, or outside the frozen matrix")
    for graph in GRAPHS:
        group = [r for r in controls if r["unit"]["subject"] == graph]
        parsed = []
        for record in group:
            try:
                if record.get("status") != "succeeded":
                    raise ValueError("Control status is not succeeded")
                unit = record["unit"]
                if unit["models"] != {"a": graph, "b": graph}:
                    raise ValueError("Control graph labels differ")
                parsed.append((unit["order"], unit_ratios(record, stage, device)))
            except (ValueError, KeyError, TypeError) as error:
                reasons.append(f"{graph}: {error}")
        if len(parsed) != n or sum(order == ["a", "b"] for order, _ in parsed) != n // 2:
            reasons.append(f"{graph}: incomplete or unbalanced AB/BA blocks")
            continue
        for metric in metrics_for(device):
            ratios = [r[metric] for _, r in parsed]
            ab = [r[metric] for order, r in parsed if order == ["a", "b"]]
            ba = [r[metric] for order, r in parsed if order == ["b", "a"]]
            center, within = statistics.median(ratios), sum(.85 <= x <= 1.15 for x in ratios)
            order_ratio = statistics.median(ab) / statistics.median(ba)
            passed = .95 <= center <= 1.05 and within >= minimum and .90 <= order_ratio <= 1.10
            checks.append({"graph": graph, "metric": metric, **describe(ratios),
                           "ratios": ratios, "within_15_percent": within,
                           "ab_median": statistics.median(ab), "ba_median": statistics.median(ba),
                           "ab_over_ba": order_ratio, "passed": passed})
            if not passed:
                reasons.append(f"{graph}/{metric}: fixed operational tolerance failed")
    return {"passed": not reasons, "reasons": reasons, "checks": checks,
            "planned_controls": 5 * n, "observed_controls": len(controls),
            "scope": "pragmatic 5% median/15% block/10% order tolerances, not a significance test"}


def validate_records(protocol, job, records):
    """Bind raw records to frozen order, graph bytes, input and complete schedule."""
    planned = protocol["schedules"][job["stage"]]
    found = [r.get("unit", {}).get("unit_id") for r in records]
    if len(found) != len(set(found)):
        raise ValueError("Duplicate raw unit")
    by_id = {unit["unit_id"]: unit for unit in planned}
    if found != [unit["unit_id"] for unit in planned[:len(found)]]:
        raise ValueError("Raw records are not a chronological prefix of the frozen schedule")
    if any(r.get("unit") != by_id.get(r.get("unit", {}).get("unit_id")) for r in records):
        raise ValueError("Raw unit differs from frozen schedule")
    expected_input = protocol["prepared"]["input"]["tensor_sha256_by_batch"][str(job["batch"])]
    complete = set(found) == set(by_id)
    for record in records:
        status = record.get("status")
        if status not in ("succeeded", "incomplete"):
            raise ValueError("Unknown raw unit status")
        if status == "succeeded":
            unit_ratios(record, job["stage"], job["device"])
        else:
            complete = False
            if not set(record.get("sides", {})) <= {"a", "b"}:
                raise ValueError("Unknown partial side")
        for side, raw in record["sides"].items():
            source = protocol["prepared"]["graphs"][record["unit"]["models"][side]]
            if raw["source_sha256"] != source["sha256"] or raw["input_sha256"] != expected_input:
                raise ValueError("Raw source/input identity differs from frozen protocol")
            if status == "incomplete":
                for metric in METRICS:
                    series = raw.get(metric)
                    expected = 1 if metric == "throughput_seconds" else (10 if job["stage"] == "pilot" else 20)
                    if metric == "cuda_event_ms" and not job["device"].startswith("cuda"):
                        expected = 0
                    if (not isinstance(series, list) or len(series) > expected or any(
                        isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in series)):
                        raise ValueError("Malformed raw partial timing series")
    return complete


def validate_provenance(result, result_path):
    """Verify sibling resource evidence without opening original absolute paths."""
    successful = result.get("status") == "succeeded"
    resource_record = result.get("resources")
    if not resource_record:
        if successful:
            raise ValueError("Successful result lacks resources proof")
        return
    resource_path = Path(result_path).parent / "resources.json"
    if hashlib.sha256(resource_path.read_bytes()).hexdigest() != resource_record.get("sha256"):
        raise ValueError("Sibling resources digest differs")
    resource = read(resource_path)
    if not successful:
        return
    job, admission = result["job"], result.get("admission", {})
    if (admission.get("status") != "admitted" or admission.get("session_id") != job["session_id"]
            or admission.get("protocol_sha256") != job["protocol_sha256"]
            or admission.get("before_worker_spawn") is not True):
        raise ValueError("Successful result has missing or foreign pre-worker admission")
    post = resource.get("post_exit", {})
    if (resource.get("admission") != admission or resource.get("returncode") != 0
            or resource.get("violation") is not None or post.get("status") != "admitted"
            or post.get("worker_exited") is not True):
        raise ValueError("Successful result has failed resource/post-exit proof")


def analyze(root, destination):
    root, destination = Path(root), Path(destination)
    if destination.exists():
        raise ValueError("Analysis destination must be fresh")
    protocol, run = read(root / "protocol.json"), read(root / "run.json")
    expected_hash = hashlib.sha256(json.dumps({k: v for k, v in protocol.items() if k != "protocol_sha256"},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if protocol.get("protocol_sha256") != expected_hash:
        raise ValueError("Frozen protocol hash differs")
    if run.get("protocol_sha256") != expected_hash:
        raise ValueError("Run has a different protocol identity")
    expected_profiles = {(p["device"], p["batch"]) for p in protocol["profiles"]}
    observed_profiles = [(p["device"], p["batch"]) for p in run["profiles"]]
    if (len(observed_profiles) != len(set(observed_profiles)) or set(observed_profiles) != expected_profiles
            or expected_profiles != {("cpu", 1), ("cpu", 64), ("cuda:0", 1), ("cuda:0", 64)}):
        raise ValueError("Run does not contain exactly the four frozen profiles")
    destination.mkdir(parents=True)
    profiles, rows = [], []
    for entry in run["profiles"]:
        profile = {"device": entry["device"], "batch": entry["batch"], "stages": {}}
        gates = {}
        if set(entry["stages"]) - {"pilot", "main"} or "pilot" not in entry["stages"]:
            raise ValueError("Unknown stage or missing pilot result")
        for stage in ("pilot", "main"):
            if stage not in entry["stages"]:
                continue
            result_record = entry["stages"][stage]
            result_path = (root / result_record["relative_path"]).resolve()
            if not result_path.is_relative_to(root.resolve()):
                raise ValueError("Result reference escapes the portable archive")
            result = read(result_path)
            if hashlib.sha256(result_path.read_bytes()).hexdigest() != result_record["sha256"]:
                raise ValueError("Worker result digest differs")
            validate_provenance(result, result_path)
            job, records = result["job"], result.get("records", [])
            if (job["protocol_sha256"] != protocol["protocol_sha256"] or job["stage"] != stage
                    or job["device"] != entry["device"] or job["batch"] != entry["batch"]):
                raise ValueError("Worker result has a different frozen profile identity")
            expected_profile = {"dtype": "float32", "threads": 1, "runtime": "torch.jit", "profiler": False,
                                "tf32": False, "cudnn_benchmark": False, "cudnn_deterministic": True}
            if result["status"] == "succeeded" and result.get("profile") != expected_profile:
                raise ValueError("Successful worker has different effective precision/runtime flags")
            complete = validate_records(protocol, job, records) if records else False
            gate = control_gate(records, stage, job["device"])
            valid = result["status"] == "succeeded" and complete and gate["passed"]
            if stage == "main":
                valid = valid and gates.get("pilot", False)
            gates[stage] = valid
            profile["stages"][stage] = {"status": result["status"], "complete": complete,
                "gate": gate, "ranking_eligible": valid and stage == "main"}
            for record in records:
                if record["unit"]["kind"] != "comparison" or record.get("status") != "succeeded":
                    continue
                for metric, ratio in unit_ratios(record, stage, job["device"]).items():
                    rows.append({"device": job["device"], "batch": job["batch"], "stage": stage,
                        "subject": record["unit"]["subject"], "block": record["unit"]["block"],
                        "order": "".join(record["unit"]["order"]), "metric": metric, "a_over_b": ratio,
                        "profile_eligible": valid})
        profiles.append(profile)
    summaries = []
    keys = sorted({(r["device"], r["batch"], r["subject"], r["metric"]) for r in rows})
    for device, batch, subject, metric in keys:
        group = [r for r in rows if (r["device"], r["batch"], r["subject"], r["metric"]) == (device, batch, subject, metric)]
        summaries.append({"device": device, "batch": batch, "subject": subject, "metric": metric,
            **describe([r["a_over_b"] for r in group]), "ranking_eligible": all(r["profile_eligible"] for r in group),
            "ab": describe([r["a_over_b"] for r in group if r["order"] == "ab"]),
            "ba": describe([r["a_over_b"] for r in group if r["order"] == "ba"])})
    analysis = {"schema": 3, "protocol_sha256": expected_hash, "profiles": profiles, "comparisons": summaries,
        "ratio_meaning": "a/b latency (throughput uses elapsed seconds for equal work); >1 favors b",
        "scope": "descriptive temporal blocks, one process/profile/stage, one trained seed; no iid or population CI",
        "prepared": protocol["prepared"], "new_training_updates": 0}
    write(destination / "analysis.json", analysis)
    for filename, content in (("blocks.csv", rows), ("comparisons.csv", summaries)):
        buffer = io.StringIO(newline="")
        keys = list(dict.fromkeys(k for row in content for k in row))
        writer = csv.DictWriter(buffer, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in content:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})
        (destination / filename).write_text(buffer.getvalue(), encoding="utf-8")
    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root, args.destination)
