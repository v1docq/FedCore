"""Pure, standard-library analysis of the frozen issue-99 package artifacts.

This module never imports a model runtime, reconstructs a model or opens test
data. Missing/aborted jobs stay visible; inconsistent pairs are reported rather
than replaced. All times use recorded completion timestamps and the parent's
worker-launch clock. Writing/publication remains the caller's responsibility.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import product
import json
import math
from pathlib import Path
from statistics import median


REFERENCE = (2.0, 1.05)
REFERENCE_AREA = 2.1
DENSE_BYTES = 49_416
BYTE_CAPS = (14_824, 29_649, 42_003)
COMMON_BYTES = (12_488, 22_792, 39_112)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _point(value):
    if len(value) != 2 or not all(_finite(item) for item in value):
        raise ValueError("A finite two-dimensional point is required")
    return float(value[0]), float(value[1])


def pareto_points(points):
    """Unique finite nondominated minimization points, sorted by first axis."""
    ordered = sorted({_point(value) for value in points})
    best_y, frontier = math.inf, []
    for x, y in ordered:
        if y < best_y:
            frontier.append((x, y))
            best_y = y
    return tuple(frontier)


def hypervolume_2d(points, reference=REFERENCE):
    """Area of the union [x,ref_x] × [y,ref_y], clipped to the reference.

    A point on or beyond either reference coordinate contributes zero. No
    observed value changes the reference. The sweep uses horizontal strips;
    duplicates and dominated points cannot increase the area.
    """
    reference_x, reference_y = _point(reference)
    area, best_y = 0.0, reference_y
    for x, y in pareto_points(points):
        if x < reference_x and y < best_y:
            area += (reference_x - x) * (best_y - y)
            best_y = y
    return area


def normalized_hypervolume(points):
    return hypervolume_2d(points) / REFERENCE_AREA


def evaluation_point(row):
    """Return the recorded normalized finite fitness, without inventing loss."""
    if row.get("status") != "finite":
        return None
    quality, cost = row.get("quality"), row.get("cost")
    if not _finite(quality) or not _finite(cost):
        return None
    return float(quality), float(cost)


def _ordered(rows):
    return sorted(rows, key=lambda row: (row.get("completed_at", math.inf),
        row.get("global_attempt", math.inf), row.get("solution_id", "")))


def _within(row, deadline):
    completed = row.get("completed_at")
    return _finite(completed) and (deadline is None or completed <= deadline)


def finite_evaluations(rows, *, deadline=None, accepted=False):
    return [row for row in _ordered(rows) if evaluation_point(row) is not None
            and _within(row, deadline) and (not accepted or row.get("quality_accepted") is True)]


def _frontier(rows):
    frontier = set(pareto_points(evaluation_point(row) for row in rows))
    # Different recipes with equal nondominated fitness remain separate. A
    # repeated evaluation of the same recipe appears once, at its first end.
    unique = {}
    for row in _ordered(rows):
        if evaluation_point(row) in frontier:
            unique.setdefault((row.get("solution_id"), evaluation_point(row)), row)
    return sorted(unique.values(), key=lambda row: (*evaluation_point(row), row.get("solution_id", "")))


def _summary(row):
    if row is None:
        return None
    return {key: row.get(key) for key in ("solution_id", "workflow_fingerprint", "quality", "cost",
        "tensor_bytes", "metrics", "quality_accepted", "completed_at", "global_attempt")}


def _hv(rows):
    return hypervolume_2d(evaluation_point(row) for row in rows)


def select_at_caps(rows, caps=BYTE_CAPS):
    """Validation-accepted minimum CE, then actual bytes, then canonical ID."""
    accepted = [row for row in rows if row.get("quality_accepted") is True
                and evaluation_point(row) is not None and type(row.get("tensor_bytes")) is int]
    result = []
    for cap in caps:
        feasible = [row for row in accepted if row["tensor_bytes"] <= cap]
        selected = min(feasible, key=lambda row: (row["quality"], row["tensor_bytes"], row["solution_id"])) if feasible else None
        result.append({"byte_cap": cap, "selected": _summary(selected)})
    return result


def virtual_stop_diagnostic(rows, *, applicable=True, reason=None, parent_started_at=None,
                            initial=12, patience=6, tolerance=1e-6):
    """Inspect a completed trajectory; this never stops or reruns computation."""
    if not applicable:
        return {"status": "not_applicable", "reason": reason,
                "initial_finite": initial, "patience": patience, "tolerance": tolerance}
    unique, seen = [], set()
    for row in _ordered(rows):
        if evaluation_point(row) is not None and row.get("solution_id") not in seen:
            unique.append(row)
            seen.add(row.get("solution_id"))
    prefix, previous, flat, stop = [], 0.0, 0, None
    for index, row in enumerate(unique, 1):
        prefix.append(row)
        current = _hv(prefix)
        if index > initial:
            flat = 0 if current - previous > tolerance else flat + 1
            if flat >= patience:
                stop = index
                break
        previous = current
    full_hv = _hv(unique)
    result = {"status": "triggered" if stop is not None else "not_triggered",
        "initial_finite": initial, "patience": patience, "tolerance": tolerance,
        "finite_unique_evaluations": len(unique), "full_hypervolume": full_hv,
        "stop_after_finite": stop, "actual_time_saved": None,
        "interpretation": "trajectory prefix diagnostic; no measured time reduction"}
    if stop is not None:
        row = unique[stop - 1]
        result.update(prefix_hypervolume=current, missed_hypervolume=full_hv - current,
            later_finite_evaluations=len(unique) - stop, stopped_at=row.get("completed_at"),
            stopped_after_parent_seconds=None if parent_started_at is None else row["completed_at"] - parent_started_at,
            stop_global_attempt=row.get("global_attempt"))
    return result


def _describe(values):
    values = [value for value in values if _finite(value)]
    return {"count": len(values), "median": median(values) if values else None,
            "minimum": min(values) if values else None, "maximum": max(values) if values else None}


def _ratio(numerator, denominator):
    return numerator / denominator if _finite(numerator) and _finite(denominator) and denominator > 0 else None


class _Reader:
    def __init__(self):
        self.issues = []

    def issue(self, code, path, **details):
        self.issues.append({"code": code, "path": str(path), **details})

    def json(self, path):
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if type(value) is not dict:
                raise ValueError("A package metadata/result object is required")
            return value
        except (OSError, ValueError) as error:
            self.issue("invalid_json", path, message=str(error))
            return None

    def lines(self, path):
        if not path.exists():
            return []
        result = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            self.issue("unreadable_jsonl", path, message=str(error))
            return result
        for index, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if type(value) is not dict:
                    raise ValueError("An event must be an object")
                result.append(value)
            except ValueError as error:
                self.issue("invalid_jsonl_line", path, line=index, message=str(error))
        return result


def _merge_rows(first, second):
    rows = {}
    for row in first + second:
        key = (row.get("global_attempt"), row.get("solution_id"), row.get("started_at"))
        rows[key] = row
    return _ordered(list(rows.values()))


def _load_jobs(output, reader, protocol, ledger):
    planned = {job["job"]: job for job in protocol.get("jobs", [])}
    resources = {row["job"]: row for row in ledger.get("jobs", [])}
    directories = {path.name: path for path in (output / "jobs").iterdir() if path.is_dir()} if (output / "jobs").exists() else {}
    jobs = []
    for name in sorted(set(planned) | set(resources) | set(directories)):
        directory = output / "jobs" / name
        job = {**planned.get(name, {}), **(reader.json(directory / "job.json") or {})}
        result = reader.json(directory / "result.json") or {}
        resource = reader.json(directory / "resources.json") or resources.get(name, {})
        rows = _merge_rows(result.get("evaluations", []),
                           [row for row in reader.lines(directory / "events.jsonl") if row.get("event") == "evaluation"])
        for row in rows:
            if row.get("status") == "finite" and evaluation_point(row) is None:
                reader.issue("invalid_finite_fitness", directory, global_attempt=row.get("global_attempt"))
        jobs.append({"name": name, "job": job, "result": result, "resource": resource,
            "kind": result.get("kind", job.get("kind", resource.get("kind"))),
            "rows": rows, "trace": reader.lines(directory / "search-trace.jsonl"),
            "composition": result.get("rows") or reader.lines(directory / "composition-events.jsonl"),
            "requests": result.get("requests") or reader.lines(directory / "cache-requests.jsonl"),
            "order": result.get("rows") or reader.lines(directory / "order-events.jsonl")})
    return jobs


def _run_analysis(item, oracle, attempts):
    job, result, rows, trace = item["job"], item["result"], item["rows"], item["trace"]
    info = {**job, **result}
    deadline = result.get("deadline")
    if deadline is None and info.get("kind") == "search" and _finite(job.get("worker_started_at")):
        deadline = job["worker_started_at"] + (60 if info.get("budget") == "attempts" else 30)
        if _finite(job.get("package_deadline")):
            deadline = min(deadline, job["package_deadline"])
    finite = finite_evaluations(rows, deadline=deadline)
    accepted = [row for row in finite if row.get("quality_accepted") is True]
    all_finite = finite_evaluations(rows)
    search = result.get("search", {})
    proposals = [row for row in trace if row.get("event") == "proposal"]
    cache_events = [row for row in trace if row.get("event") == "cache_observation"]
    catalog_size = 15 if info.get("mode") == "R" else 80
    completed_successfully = result.get("status") == "succeeded"
    is_oracle = item["kind"] == "oracle"
    oracle_complete = bool(oracle and oracle["complete"])
    full_attempts = [row for row in attempts if row.get("job") == item["name"] and row.get("purpose") == "full_validation"
        and (not row.get("protocol_sha256") or not job.get("protocol_sha256") or row["protocol_sha256"] == job["protocol_sha256"])]
    if info.get("mode") == "E":
        observed_proposals, observed_basis = len(cache_events), "grammar_cache_observation_events"
    elif full_attempts:
        observed_proposals, observed_basis = len(full_attempts), "recorded_full_validation_starts"
    else:
        observed_proposals, observed_basis = len(rows), "completed_evaluation_rows"
    parent_started = job.get("worker_started_at", item["resource"].get("resources", {}).get("started_at"))
    hv, accepted_hv = _hv(finite), _hv(accepted)
    row = {"job": item["name"], "kind": item["kind"],
        "status": result.get("status", "partial" if rows or trace or "worker_started_at" in job else "unexecuted"),
        "model_seed": info.get("model_seed"), "search_seed": info.get("search_seed"),
        "mode": info.get("mode"), "budget": info.get("budget"), "deadline": deadline,
        "worker_started_at": parent_started, "stop_reason": search.get("stop_reason"),
        "engine": search.get("engine"), "catalog_size": catalog_size,
        "complete": completed_successfully and (len({value["solution_id"] for value in finite}) == 80 if is_oracle else True),
        "charged_evaluations": search.get("charged_evaluations", len(full_attempts) or len(rows)),
        "full_evaluations_started": len(full_attempts) if attempts else None,
        "full_evaluations_completed": len(rows), "finite_completed": len(all_finite),
        "finite_within_deadline": len(finite), "finite_after_deadline": len(all_finite) - len(finite),
        "accepted_within_deadline": len(accepted),
        "failure_counts": dict(Counter(value.get("status", "unknown") for value in rows if value.get("status") != "finite")),
        "created_proposals": search.get("created_proposals", len(proposals)),
        "observed_proposals": observed_proposals, "observed_proposals_basis": observed_basis,
        "cache_hits": search.get("cache_hits", sum(value.get("cache_hit") is True for value in cache_events)),
        "repeated_failures": search.get("repeated_failures", sum(value.get("repeated_failure") is True for value in cache_events)),
        "component_reads": search.get("component_reads", sum(value.get("event") == "fitness_read" for value in trace)),
        "technical_component_reads": sum(value.get("event") == "fitness_read" and value.get("technical_read") is True for value in trace),
        "unique_proposed": len({value.get("canonical_solution_id") for value in proposals}),
        "unique_finite": len({value.get("solution_id") for value in finite}),
        "catalog_coverage": len({value.get("solution_id") for value in finite}) / catalog_size,
        "hypervolume": hv, "normalized_hypervolume": hv / REFERENCE_AREA,
        "accepted_hypervolume": accepted_hv, "accepted_normalized_hypervolume": accepted_hv / REFERENCE_AREA,
        "oracle_complete": oracle_complete,
        "oracle_hypervolume": oracle.get("hypervolume") if oracle else None,
        "gap_to_oracle": oracle["hypervolume"] - hv if oracle_complete else None,
        "accepted_gap_to_oracle": oracle["accepted_hypervolume"] - accepted_hv if oracle_complete else None,
        "resources": item["resource"].get("resources", {}), "selections": select_at_caps(finite)}
    row["repeated_proposals"] = max(0, row["created_proposals"] - row["unique_proposed"])
    cumulative, curve = [], []
    for index, value in enumerate(_ordered(rows), 1):
        admitted = evaluation_point(value) is not None and _within(value, deadline)
        if admitted:
            cumulative.append(value)
        current = _hv(cumulative)
        accepted_current = _hv([entry for entry in cumulative if entry.get("quality_accepted") is True])
        completed = value.get("completed_at")
        curve.append({"completed_attempt": index, "global_attempt": value.get("global_attempt"),
            "solution_id": value.get("solution_id"), "status": value.get("status"),
            "completed_at": completed,
            "elapsed_parent_seconds": completed - parent_started if _finite(completed) and _finite(parent_started) else None,
            "included_in_deadline_archive": admitted, "hypervolume": current,
            "normalized_hypervolume": current / REFERENCE_AREA, "accepted_hypervolume": accepted_current,
            "gap_to_oracle": oracle["hypervolume"] - current if oracle_complete else None})
    row["curves"] = curve
    applicable = info.get("budget") != "attempts" and info.get("mode") != "R"
    row["virtual_stop"] = virtual_stop_diagnostic(finite, applicable=applicable,
        reason="12-attempt budget cannot reach 12+6" if info.get("budget") == "attempts" else "ready catalog has only 15 solutions",
        parent_started_at=parent_started)
    if row["status"] == "unexecuted":
        row["virtual_stop"] = {"status": "unexecuted"}
    frontier = {"job": item["name"], "deadline": deadline,
        "all": [_summary(value) for value in _frontier(finite)],
        "accepted": [_summary(value) for value in _frontier(accepted)],
        "all_including_late": [_summary(value) for value in _frontier(all_finite)]}
    return row, frontier, finite


def _search_pairs(runs, finite_by_job):
    groups = defaultdict(dict)
    for row in runs:
        if row["kind"] == "search":
            groups[(row["model_seed"], row["budget"], row["search_seed"])][row["mode"]] = row
    paired, common = [], []
    for (model, budget, seed), modes in sorted(groups.items()):
        for first, second in (("R", "T"), ("T", "E")):
            left, right = modes.get(first), modes.get(second)
            complete = bool(left and right and left["status"] == right["status"] == "succeeded")
            paired.append({"model_seed": model, "budget": budget, "search_seed": seed,
                "comparison": first + second, "complete": complete,
                "left_job": left["job"] if left else None, "right_job": right["job"] if right else None,
                "delta_hypervolume_second_minus_first": right["hypervolume"] - left["hypervolume"] if complete else None,
                "delta_accepted_hypervolume_second_minus_first": right["accepted_hypervolume"] - left["accepted_hypervolume"] if complete else None})
        for byte_count in COMMON_BYTES:
            selections = {}
            for mode in ("R", "T"):
                rows = finite_by_job.get(modes.get(mode, {}).get("job"), [])
                feasible = [row for row in rows if row.get("tensor_bytes") == byte_count]
                selections[mode] = _summary(min(feasible, key=lambda row: (row["quality"], row["solution_id"]))) if feasible else None
            left, right = selections["R"], selections["T"]
            common.append({"model_seed": model, "budget": budget, "search_seed": seed,
                "tensor_bytes": byte_count, "R": left, "T": right,
                "quality_scope": "all finite validation outcomes at exact equal physical bytes",
                "delta_quality_T_minus_R": right["quality"] - left["quality"] if left and right else None,
                "delta_CE_T_minus_R": (right["quality"] - left["quality"]) * math.log(2) if left and right else None})
    return paired, common


def _admissibility(jobs):
    item = next((item for item in jobs if item["kind"] == "admissibility"), None)
    rows = item["composition"] if item else []
    ordinary = [row for row in rows if not row.get("schema_negative", not row.get("case_id", "").startswith("word-"))]
    negative = [row for row in rows if row not in ordinary]
    mismatches = [row for row in rows if row.get("contract_accepted") != row.get("production_accepted")
        or (row.get("contract_accepted") != row.get("execution_accepted")
            and (row.get("reference_rejection") or {}).get("category", "structural") == "structural")]
    failures = [row for row in rows if row.get("reference_rejection") and row["reference_rejection"].get("category") != "structural"]
    return {"status": item["result"].get("status", "partial") if item else "unexecuted",
        "expected_word_requests": 155, "expected_schema_negatives": 8,
        "predicted_accepted_words": 72, "predicted_rejected_words": 83,
        "observed_word_requests": len(ordinary), "observed_schema_negatives": len(negative),
        "accepted_words": sum(row.get("contract_accepted") is True for row in ordinary),
        "rejected_words": sum(row.get("contract_accepted") is False for row in ordinary),
        "structural_decision_disagreements": mismatches, "nonstructural_failures": failures,
        "rows": rows}


def _timing_row(row):
    return {key: row.get(key) for key in ("case_id", "accepted", "output_identity", "tensor_bytes",
        "elapsed_seconds", "check_seconds", "execution_seconds", "stage_timings",
        "decompositions_started", "completed_steps", "rejection", "production_rejection")}


def _early_late_pairs(jobs):
    grouped = defaultdict(dict)
    for item in jobs:
        if item["kind"] == "early_late":
            grouped[item["job"].get("stream_seed", item["result"].get("stream_seed"))][item["job"].get("branch", item["result"].get("branch"))] = item
    pairs = []
    expected_ids = {"word-" + "".join(word) for length in (1, 2, 3)
                    for word in product("PLQDR", repeat=length)} | {
        "rank-zero", "rank-too-large", "duplicate-mask", "out-of-range-mask",
        "compute-dtype", "unknown-scope", "overlapping-scopes", "onnx"}
    for seed, branches in sorted(grouped.items()):
        a, b = branches.get("A"), branches.get("B")
        a_rows, b_rows = (a["composition"] if a else []), (b["composition"] if b else [])
        a_ids, b_ids = [row.get("case_id") for row in a_rows], [row.get("case_id") for row in b_rows]
        by_a, by_b = {row["case_id"]: row for row in a_rows}, {row["case_id"]: row for row in b_rows}
        shared, measurements, mismatches = sorted(set(by_a) & set(by_b)), [], []
        for identity in shared:
            left, right = by_a[identity], by_b[identity]
            same = (left.get("accepted") == right.get("accepted")
                and left.get("output_identity") == right.get("output_identity")
                and left.get("tensor_bytes") == right.get("tensor_bytes"))
            if not same or left.get("accepted") and left.get("output_identity") is None:
                mismatches.append(identity)
            measurements.append({"case_id": identity, "A": _timing_row(left), "B": _timing_row(right), "parity": same})
        finished = bool(a and b and a["result"].get("status") == b["result"].get("status") == "succeeded")
        shape_valid = len(a_rows) == len(b_rows) == 163 and set(by_a) == set(by_b) == expected_ids
        complete = finished and shape_valid
        if finished and not shape_valid:
            mismatches.append("frozen_request_catalog")
        order_equal = a_ids == b_ids
        if complete and not order_equal:
            mismatches.append("request_order")
        seconds_a, seconds_b = (a["result"].get("stream_seconds") if a else None), (b["result"].get("stream_seconds") if b else None)
        subsets = {}
        for accepted in (True, False):
            relevant = [pair for pair in measurements if pair["A"].get("accepted") is accepted and pair["B"].get("accepted") is accepted]
            first = sum(pair["A"].get("elapsed_seconds", 0) or 0 for pair in relevant)
            second = sum(pair["B"].get("elapsed_seconds", 0) or 0 for pair in relevant)
            subsets["accepted" if accepted else "rejected"] = {"count": len(relevant), "A_seconds": first,
                "B_seconds": second, "early_over_late": _ratio(first, second)}
        pairs.append({"stream_seed": seed, "complete": complete, "valid": False if mismatches else (True if complete else None),
            "request_order_equal": order_equal, "A_requests": len(a_rows), "B_requests": len(b_rows),
            "mismatches": mismatches, "A_stream_seconds": seconds_a, "B_stream_seconds": seconds_b,
            "early_over_late": _ratio(seconds_a, seconds_b), "late_over_early": _ratio(seconds_b, seconds_a),
            "A_decompositions_started": sum(row.get("decompositions_started", 0) for row in a_rows),
            "B_decompositions_started": sum(row.get("decompositions_started", 0) for row in b_rows),
            "subsets": subsets, "measurements": measurements})
    return pairs


def _cache_pairs(jobs):
    grouped = defaultdict(dict)
    for item in jobs:
        if item["kind"] == "cache":
            info = {**item["job"], **item["result"]}
            grouped[(info.get("model_seed"), info.get("stream_seed"))][info.get("branch")] = item
    pairs = []
    for (model, seed), branches in sorted(grouped.items()):
        a, b = branches.get("A"), branches.get("B")
        first, second = (a["requests"] if a else []), (b["requests"] if b else [])
        mismatches = []
        for index, (left, right) in enumerate(zip(first, second)):
            fields = [key for key in ("solution_id", "status", "quality", "tensor_bytes") if left.get(key) != right.get(key)]
            if fields:
                mismatches.append({"index": index, "fields": fields})
        for branch, requests in (("A", first), ("B", second)):
            for index, row in enumerate(requests):
                if row.get("cache_hit") and row.get("status") != "finite":
                    mismatches.append({"branch": branch, "index": index, "fields": ["nonfinite_cache_hit"]})
        counts_a, counts_b = Counter(row.get("solution_id") for row in first), Counter(row.get("solution_id") for row in second)
        complete = bool(a and b and len(first) == len(second) == 48 and len(counts_a) == len(counts_b) == 12
            and set(counts_a.values()) == set(counts_b.values()) == {4}
            and a["result"].get("status") == b["result"].get("status") == "succeeded")
        if a and b and a["result"].get("status") == b["result"].get("status") == "succeeded" and not complete:
            mismatches.append({"fields": ["frozen_48_request_stream"]})
        if complete:
            for branch, item, requests in (("A", a, first), ("B", b, second)):
                expected_evaluations = len(requests) - sum(row.get("cache_hit") is True for row in requests)
                if len(item["rows"]) != expected_evaluations:
                    mismatches.append({"branch": branch, "fields": ["full_evaluation_total"],
                        "expected": expected_evaluations, "observed": len(item["rows"])})
        seconds_a, seconds_b = (a["result"].get("stream_seconds") if a else None), (b["result"].get("stream_seconds") if b else None)
        pairs.append({"model_seed": model, "stream_seed": seed, "complete": complete,
            "valid": False if mismatches else (True if complete else None), "mismatches": mismatches,
            "A_requests": len(first), "B_requests": len(second), "distinct_A": len(counts_a), "distinct_B": len(counts_b),
            "A_full_evaluations": len(a["rows"]) if a else 0, "B_full_evaluations": len(b["rows"]) if b else 0,
            "A_cache_hits": sum(row.get("cache_hit") is True and row.get("status") == "finite" for row in first),
            "B_cache_hits": sum(row.get("cache_hit") is True and row.get("status") == "finite" for row in second),
            "A_stream_seconds": seconds_a, "B_stream_seconds": seconds_b,
            "enabled_over_disabled": _ratio(seconds_a, seconds_b),
            "A_status_counts": dict(Counter(row.get("status") for row in first)),
            "B_status_counts": dict(Counter(row.get("status") for row in second)),
            "requests_A": first, "requests_B": second})
    return pairs


def _order_pairs(jobs):
    grouped = defaultdict(dict)
    for item in jobs:
        if item["kind"] == "order":
            for row in item["order"]:
                grouped[(row.get("model_seed"), row.get("pair"))][row.get("word")] = row
    pairs = []
    for (model, name), words in sorted(grouped.items()):
        first, second = ("PLD", "LDP") if name.startswith("mask-") else ("LDQD", "QDLD")
        a, b = words.get(first), words.get(second)
        same_bytes = bool(a and b and a.get("tensor_bytes") == b.get("tensor_bytes") == DENSE_BYTES)
        deltas = {key: b["metrics"][key] - a["metrics"][key] for key in ("cross_entropy", "accuracy", "macro_f1")} if a and b else None
        pairs.append({"model_seed": model, "pair": name, "first_word": first, "second_word": second,
            "complete": bool(a and b), "equal_dense_bytes": same_bytes,
            "metrics_second_minus_first": deltas, "first": a, "second": b})
    return pairs


def resource_lineage(output, reader):
    """Each original execution contributes resources once, despite copied data."""
    seen, revisions, owned_jobs = set(), [], {}
    directory = Path(output).resolve()
    while directory not in seen:
        seen.add(directory)
        protocol = reader.json(directory / "protocol.json") or {}
        ledger = reader.json(directory / "resource-ledger.json") or {}
        revisions.append({"directory": str(directory), "protocol_sha256": protocol.get("protocol_sha256"),
            "source_build": protocol.get("git_revision"), "status": ledger.get("status"),
            "cumulative_elapsed_seconds": ledger.get("elapsed_seconds"), "mlp_attempts": ledger.get("mlp_attempts"),
            "operator_stop": ledger.get("operator_stop")})
        for item in ledger.get("jobs", []):
            origin = str(Path(item.get("origin_execution", directory)).resolve())
            owned_jobs.setdefault((origin, item.get("job")), {"origin_execution": origin, **item})
        prior = protocol.get("resumed_admission")
        if not prior:
            break
        directory = Path(prior["directory"]).resolve()
    return revisions, list(owned_jobs.values())


def build_analysis(output):
    """Read recorded artifacts and return an inspectable, JSON-safe analysis."""
    output, reader = Path(output), _Reader()
    protocol = reader.json(output / "protocol.json") or {}
    ledger = reader.json(output / "resource-ledger.json") or {}
    attempts = reader.lines(output / "mlp-attempts.jsonl")
    jobs = _load_jobs(output, reader, protocol, ledger)
    oracle = {}
    for item in jobs:
        if item["kind"] == "oracle":
            rows = finite_evaluations(item["rows"])
            accepted = [row for row in rows if row.get("quality_accepted") is True]
            seed = item["result"].get("model_seed", item["job"].get("model_seed"))
            oracle[seed] = {"hypervolume": _hv(rows), "accepted_hypervolume": _hv(accepted),
                "complete": item["result"].get("status") == "succeeded" and len({row["solution_id"] for row in rows}) == 80}
    runs, frontiers, finite_by_job = [], [], {}
    for item in jobs:
        if item["kind"] in ("oracle", "search"):
            seed = item["result"].get("model_seed", item["job"].get("model_seed"))
            row, frontier, finite = _run_analysis(item, oracle.get(seed), attempts)
            runs.append(row)
            frontiers.append(frontier)
            finite_by_job[item["name"]] = finite
    paired, common = _search_pairs(runs, finite_by_job)
    early_late, cache, order = _early_late_pairs(jobs), _cache_pairs(jobs), _order_pairs(jobs)
    admission = _admissibility(jobs)
    if admission["structural_decision_disagreements"] or admission["nonstructural_failures"]:
        reader.issue("admissibility_disagreement_or_failure", "admissibility")
    paired_groups = defaultdict(list)
    for row in paired:
        paired_groups[(row["model_seed"], row["budget"], row["comparison"])].append(row["delta_hypervolume_second_minus_first"])
    paired_summary = [{"model_seed": key[0], "budget": key[1], "comparison": key[2],
        "delta_hypervolume_second_minus_first": _describe(values),
        "replication_unit": "search seeds within one fixed trained model"}
        for key, values in sorted(paired_groups.items())]
    for family, pairs in (("early_late", early_late), ("cache", cache)):
        for pair in pairs:
            if pair.get("valid") is False:
                reader.issue("paired_result_mismatch", family, model_seed=pair.get("model_seed"),
                    stream_seed=pair.get("stream_seed"), mismatches=pair.get("mismatches"))
    for pair in order:
        if pair["complete"] and not pair["equal_dense_bytes"]:
            reader.issue("order_storage_mismatch", pair["pair"], model_seed=pair["model_seed"])
    resources = [item["resource"] for item in jobs if item["resource"]]
    revisions, all_resources = resource_lineage(output, reader)
    missing = [item["name"] for item in jobs if not item["result"]]
    failures = [item["name"] for item in jobs if item["result"].get("status") == "failed"]
    status = "inconsistent" if reader.issues else ("complete" if ledger.get("status") == "succeeded" and not missing else "partial")
    return {"schema": 1, "status": status, "protocol_sha256": protocol.get("protocol_sha256"),
        "hypervolume_reference": list(REFERENCE), "normalization_area": REFERENCE_AREA,
        "historical_data_limitation": protocol.get("inference"),
        "search_runs": [row for row in runs if row["kind"] == "search"],
        "oracle_runs": [row for row in runs if row["kind"] == "oracle"],
        "frontiers": frontiers, "paired": paired, "paired_summary_by_model": paired_summary,
        "common_byte_pairs": common,
        "admissibility": admission, "early_late_pairs": early_late,
        "early_late_ratio_summary": _describe(pair["early_over_late"] for pair in early_late if pair["valid"] is True),
        "cache_pairs": cache, "cache_ratio_summary": _describe(pair["enabled_over_disabled"] for pair in cache if pair["valid"] is True),
        "order_pairs": order, "resources": {"ledger": ledger, "jobs": resources,
            "execution_lineage": revisions, "all_original_jobs": all_resources,
            "recorded_mlp_attempts": len(attempts),
            "attempts_by_purpose": dict(Counter(row.get("purpose", "unknown") for row in attempts)),
            "maximum_worker_tree_rss_bytes": max((row.get("resources", {}).get("peak_tree_rss_bytes", 0) for row in all_resources), default=0)},
        "final_test": [{"job": item["name"], **item["result"]} for item in jobs if item["kind"] == "finalize"],
        "missing_jobs": missing, "failed_jobs": failures, "issues": reader.issues}
