"""Recompute the second SVD packet from inspectable data, without the ML runtime."""
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


def median_interval(values, minimum_coverage=.95):
    """Conservative order-statistic interval; observations are paired block ratios.

    Its coverage assumes independent observations from a stationary distribution
    with a common median. Serial measurements on one device may violate that
    assumption; this is not a confidence interval across trained models.
    """
    values = sorted(values)
    n = len(values)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Block ratios must be finite and positive")
    if not 0 < minimum_coverage < 1:
        raise ValueError("Coverage must lie strictly between zero and one")
    candidates = [(k, 1 - 2 * sum(math.comb(n, i) for i in range(k)) / 2 ** n)
                  for k in range(1, (n + 1) // 2 + 1)]
    candidates = [(k, coverage) for k, coverage in candidates
                  if coverage >= minimum_coverage]
    if not candidates:
        return None
    k, coverage = candidates[-1]
    return {"lower": values[k - 1], "upper": values[n - k],
            "order_from_each_tail": k, "minimum_coverage": minimum_coverage,
            "binomial_coverage": coverage,
            "assumptions": "independent stationary paired blocks, common median",
            "scope": "timing on this machine; not across trained seeds"}


def describe(values, *, interval=True):
    values = list(values)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Timing ratios must be finite and positive")
    return {"n": len(values), "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "median_interval": median_interval(values) if interval else None}


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True,
                                        allow_nan=False)
                         if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})
    Path(path).write_bytes(buffer.getvalue().encode("utf-8"))


def check_identity(protocol, session, *, profiler=False):
    """Bind outcome labels to the frozen physical source, input and observers."""
    job = session["job"]
    prepared = protocol["prepared"]
    if job["kind"] == "conv":
        group = prepared["conv_groups"][job["subject"]]
        source, inputs = group["representations"][job["side"]], group["input"]
    else:
        source_key = prepared["comparisons"][job["subject"]][job["side"]]
        source, inputs = prepared["sources"][source_key], prepared["model_input"]
    expected = {"source_artifact_sha256": source["sha256"],
                "input_file_sha256": inputs["sha256"],
                "protocol_sha256": protocol["protocol_sha256"],
                "input_tensor_sha256": inputs["tensor_sha256_by_batch"][str(job["batch"])],
                "profiler": profiler}
    if any(job.get(key) != value for key, value in expected.items()):
        raise ValueError("Outcome source, input, protocol or profiler identity differs")
    if session["status"] == "succeeded":
        if session.get("input_sha256") != expected["input_tensor_sha256"]:
            raise ValueError("Observed input tensor differs from its frozen batch")
        if session.get("source_artifact_sha256") != expected["source_artifact_sha256"]:
            raise ValueError("Observed model differs from its frozen source")
        admission, post = session.get("admission", {}), session.get("post_exit_observation", {})
        if (admission.get("status") != "admitted" or
                admission.get("before_worker_spawn") is not True or
                admission.get("session_id") != job["session_id"] or
                admission.get("protocol_sha256") != protocol["protocol_sha256"] or
                post.get("status") != "admitted" or post.get("worker_exited") is not True or
                post.get("worker_returncode") != 0):
            raise ValueError("Successful observation lacks pre-spawn/post-exit admission proof")
        profile = session.get("profile", {})
        if (profile.get("profiler") != profiler or profile.get("precision") != "float32" or
                profile.get("threads") != 1 or profile.get("runtime") != "torch.jit" or
                profile.get("matmul_tf32") is not False or profile.get("cudnn_tf32") is not False or
                profile.get("cudnn_benchmark") is not False or
                profile.get("cudnn_deterministic") is not True or
                profile.get("batch_shape_checked") is not True):
            raise ValueError("Successful observation differs from frozen execution settings")


def timing_analysis(protocol, sessions):
    planned = {job["session_id"]: job for job in protocol["timing_schedule"]}
    if len(planned) != len(protocol["timing_schedule"]):
        raise ValueError("Duplicate planned session")
    observed = [session["job"]["session_id"] for session in sessions]
    if len(observed) != len(set(observed)) or set(observed) != set(planned):
        raise ValueError("Every planned session requires exactly one retained outcome")
    grouped = {}
    for session in sessions:
        job = session["job"]
        if any(job[key] != value for key, value in planned[job["session_id"]].items()):
            raise ValueError("Observed session differs from the frozen schedule")
        check_identity(protocol, session)
        group = grouped.setdefault(job["block_id"], {})
        if job["side"] in group:
            raise ValueError("Duplicate side within a paired block")
        group[job["side"]] = session
    rows = []
    for block_id, group in sorted(grouped.items()):
        first_job = next(iter(group.values()))["job"]
        kind = first_job["kind"]
        comparisons = (("baseline", "candidate"),) if kind != "conv" else (
            ("original_dense", "factor_r8"), ("materialized_r8", "factor_r8"),
            ("original_dense", "materialized_r8"))
        # Conv representation identifiers can carry ranks other than eight.
        if kind == "conv":
            factors = [side for side in group if side.startswith("factor")]
            dense = [side for side in group if side.startswith("materialized")]
            if len(factors) != 1 or len(dense) != 1 or "original_dense" not in group:
                raise ValueError("Conv block must retain all three representations")
            comparisons = (("original_dense", factors[0]), (dense[0], factors[0]),
                           ("original_dense", dense[0]))
        for reference, candidate in comparisons:
            a, b = group.get(reference), group.get(candidate)
            row = {"block_id": block_id, "kind": kind, "subject": first_job["subject"],
                   "device": first_job["device"], "batch": first_job["batch"],
                   "block": first_job["block"],
                   "comparison": f"{reference}/{candidate}",
                   "reference_session": a["job"]["session_id"] if a else None,
                   "candidate_session": b["job"]["session_id"] if b else None,
                   "valid": a is not None and b is not None and
                            a["status"] == b["status"] == "succeeded",
                   "statuses": {side: session["status"] for side, session in group.items()},
                   "reasons": {side: session.get("noise_reasons", [])
                               for side, session in group.items()}}
            if kind == "conv":
                # The protocol admits a Conv comparison only as a complete
                # three-representation block, even when two sides succeeded.
                row["valid"] = row["valid"] and len(group) == 3 and all(
                    session["status"] == "succeeded" for session in group.values())
            if row["valid"]:
                if a["job"]["input_file_sha256"] != b["job"]["input_file_sha256"]:
                    raise ValueError("A paired comparison requires identical input files")
                if a.get("input_sha256") != b.get("input_sha256"):
                    raise ValueError("A paired comparison requires identical input tensors")
                if any(session.get("profiler") or session["job"].get("profiler") or
                       session.get("profile", {}).get("profiler") for session in (a, b)):
                    raise ValueError("Profiled observations cannot enter latency analysis")
                for scope, field in (("wall", "wall"), ("full_path", "full_path"),
                                     ("device", "cuda_event")):
                    if a.get(field) is not None and b.get(field) is not None:
                        for percentile in ("p50_ms", "p95_ms"):
                            value_a, value_b = a[field][percentile], b[field][percentile]
                            if min(value_a, value_b) <= 0 or not all(map(math.isfinite, (value_a, value_b))):
                                raise ValueError("Timing summary must be finite and positive")
                            row[f"{scope}_{percentile}_reference"] = value_a
                            row[f"{scope}_{percentile}_candidate"] = value_b
                            row[f"{scope}_{percentile}_ratio"] = value_a / value_b
                throughput_a = statistics.median(x["samples_per_second"] for x in a["blocked_throughput"])
                throughput_b = statistics.median(x["samples_per_second"] for x in b["blocked_throughput"])
                if min(throughput_a, throughput_b) <= 0 or not all(map(math.isfinite, (throughput_a, throughput_b))):
                    raise ValueError("Throughput must be finite and positive")
                row["throughput_candidate_over_reference"] = throughput_b / throughput_a
            rows.append(row)
    groups = {}
    for row in rows:
        key = row["kind"], row["subject"], row["device"], row["batch"], row["comparison"]
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, blocks in sorted(groups.items()):
        good = [row for row in blocks if row["valid"]]
        summary = dict(zip(("kind", "subject", "device", "batch", "comparison"), key))
        summary.update(planned_blocks=len(blocks), valid_blocks=len(good),
                       invalid_blocks=len(blocks) - len(good))
        fields = sorted({key for row in good for key in row if key.endswith("_ratio")})
        if good:
            fields.append("throughput_candidate_over_reference")
        for field in fields:
            summary[field] = describe((row[field] for row in good if field in row),
                                      interval=key[0] != "conv")
        summary["statistical_scope"] = (
            "three-block descriptive kernel workload" if key[0] == "conv" else
            "paired timing blocks on one machine; independent stationary block assumption; "
            "pointwise intervals, no simultaneous coverage; environment-filtered complete pairs")
        summaries.append(summary)
    return rows, summaries


def profiling_analysis(protocol, sessions):
    planned = {job["session_id"]: job for job in protocol["profile_schedule"]}
    observed = [session["job"]["session_id"] for session in sessions]
    if len(planned) != len(protocol["profile_schedule"]) or len(observed) != len(set(observed)) or set(observed) != set(planned):
        raise ValueError("Every planned profile requires exactly one retained outcome")
    rows = []
    for session in sessions:
        job = session["job"]
        if any(job[key] != value for key, value in planned[job["session_id"]].items()):
            raise ValueError("Observed profile differs from frozen schedule")
        check_identity(protocol, session, profiler=True)
        row = {key: job[key] for key in ("session_id", "subject", "device", "batch", "side")}
        row.update(status=session["status"], diagnostic_only=True,
                   reasons=session.get("noise_reasons", []))
        if session["status"] == "succeeded":
            if not job.get("profiler") or not session.get("profile", {}).get("profiler"):
                raise ValueError("A profile must carry the profiler marker")
            operators = session["operator_rows"]
            top = sorted(operators, key=lambda operator: operator["self_cpu_time_us"], reverse=True)[:8]
            row.update(profile_calls=job["profile_calls"],
                       conv2d_operator_calls=sum(operator["calls"] for operator in operators
                                                 if operator["name"] == "aten::conv2d"),
                       cuda_operator_time_present=any(operator["self_cuda_time_us"] > 0
                                                      for operator in operators),
                       self_cpu_time_us=sum(operator["self_cpu_time_us"] for operator in operators),
                       self_cuda_time_us=sum(operator["self_cuda_time_us"] for operator in operators),
                       cpu_memory_operator_positive_bytes=sum(max(0, operator["cpu_memory_bytes"])
                                                              for operator in operators),
                       cuda_memory_operator_positive_bytes=sum(max(0, operator["cuda_memory_bytes"])
                                                               for operator in operators),
                       trace=session["trace"], top_cpu_operators=top)
        rows.append(row)
    return rows


def check_raw_timing(protocol, sessions):
    """Verify raw call counts and summaries, including retained noisy sessions."""
    settings = protocol["timing"]
    for session in sessions:
        job = session["job"]
        if (job["repeats"] != settings["individual_repeats"] or
                job["throughput_blocks"] != settings["throughput_blocks"] or
                job["throughput_calls"] != settings["calls_per_throughput_block"]):
            raise ValueError("Timing work count differs from frozen protocol")
        if session["status"] not in ("succeeded", "invalid_noise"):
            continue
        for raw_field, summary_field in (("raw_synchronized_wall_ms", "wall"),
                                         ("raw_full_path_ms", "full_path"),
                                         ("raw_cuda_event_ms", "cuda_event")):
            raw = session[raw_field]
            expected_calls = job["repeats"] if summary_field != "cuda_event" or job["device"].startswith("cuda") else 0
            if len(raw) != expected_calls or any(not math.isfinite(value) or value <= 0 for value in raw):
                raise ValueError("Retained raw observations differ from frozen work count")
            summary = session[summary_field]
            if not raw:
                if summary is not None:
                    raise ValueError("CPU observation cannot supply a CUDA event summary")
                continue
            values = sorted(raw)
            position = (len(values) - 1) * .95
            low, high = math.floor(position), math.ceil(position)
            p95 = values[low] + (values[high] - values[low]) * (position - low)
            expected = {"calls": len(raw), "p50_ms": statistics.median(raw),
                        "p95_ms": p95, "min_ms": min(raw), "max_ms": max(raw)}
            if any(not math.isclose(summary[key], value, rel_tol=1e-12, abs_tol=1e-12)
                   for key, value in expected.items()):
                raise ValueError("Timing summary differs from retained individual calls")
        blocks = session["blocked_throughput"]
        if len(blocks) != job["throughput_blocks"]:
            raise ValueError("Throughput block count differs")
        for block in blocks:
            if block["calls"] != job["throughput_calls"] or block["seconds"] <= 0:
                raise ValueError("Throughput work count differs")
            expected_rate = job["batch"] * block["calls"] / block["seconds"]
            if not math.isclose(block["samples_per_second"], expected_rate, rel_tol=1e-12):
                raise ValueError("Throughput rate differs from counted samples and elapsed time")


def analyze(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    protocol = read(source / "protocol.json")
    timing = read(source / "timing.json")
    profiling = read(source / "profiling.json")
    ledger = read(source / "ledger.json")
    if protocol["schema"] != 2 or any(item["protocol_sha256"] != protocol["protocol_sha256"]
                                      for item in (timing, profiling, ledger)):
        raise ValueError("Every stage must belong to the same frozen protocol")
    if protocol["training"]["new_student_updates"] or protocol["training"]["new_teacher_forwards"] or ledger["new_training_updates"]:
        raise ValueError("This packet is inference-only")
    check_raw_timing(protocol, timing["sessions"])
    block_rows, summaries = timing_analysis(protocol, timing["sessions"])
    profiles = profiling_analysis(protocol, profiling["sessions"])
    counts = {"timing_sessions": len(timing["sessions"]),
              "timing_session_statuses": dict(sorted({status: sum(row["status"] == status for row in timing["sessions"])
                                                      for status in {row["status"] for row in timing["sessions"]}}.items())),
              "profile_sessions": len(profiles),
              "profile_session_statuses": dict(sorted({status: sum(row["status"] == status for row in profiles)
                                                       for status in {row["status"] for row in profiles}}.items())),
              "wall_calls": sum(len(row.get("raw_synchronized_wall_ms", [])) for row in timing["sessions"]),
              "full_path_calls": sum(len(row.get("raw_full_path_ms", [])) for row in timing["sessions"]),
              "cuda_event_calls": sum(len(row.get("raw_cuda_event_ms", [])) for row in timing["sessions"]),
              "throughput_blocks": sum(len(row.get("blocked_throughput", [])) for row in timing["sessions"]),
              "throughput_calls": sum(block["calls"] for row in timing["sessions"] for block in row.get("blocked_throughput", [])),
              "charged_compute_seconds": ledger["charged_seconds"],
              "new_student_updates": 0, "new_teacher_forwards": 0}
    result = {"schema": 2, "analysis_script_git_lf_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
              "inputs_sha256": {name: digest(source / name) for name in
                                ("protocol.json", "timing.json", "profiling.json", "ledger.json")},
              "protocol_sha256": protocol["protocol_sha256"],
              "scientific_source_sha256": protocol["source"]["sha256"],
              "counts": counts, "comparisons": protocol["comparisons"],
              "timing_blocks": block_rows, "timing_summaries": summaries,
              "profiling": profiles,
              "recovered_checkpoints": protocol["prepared"]["recovered_checkpoints"],
              "conv_groups": protocol["prepared"]["conv_groups"],
              "limits": {"quality": "previous one-seed data; seen development test; no new quality evidence",
                         "intervals": "pointwise paired timing intervals conditional on stationarity/independence; no simultaneous guarantee",
                         "profiling": "diagnostic operator totals; inclusive memory/time may overlap; not unprofiled latency or pure kernel attribution",
                         "synthetic": "kernel workloads with constructed factors, no trained network quality claim",
                         "cluster_used": False}}
    write_json(destination / "analysis.json", result)
    write_csv(destination / "timing-blocks.csv", block_rows)
    write_csv(destination / "timing-summary.csv", summaries)
    write_csv(destination / "profiling.csv", profiles)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.input, args.output)
    print(json.dumps(result["counts"], sort_keys=True))
