"""Additional independent stdlib checks; reads evidence without executing models."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median
import zipfile


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(row) for row in path.read_text(encoding="utf-8").splitlines() if row] if path.exists() else []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(run, archive=None):
    analysis, protocol = read(run / "analysis.json"), read(run / "protocol.json")
    ledger = read(run / "resource-ledger.json")
    stopped = read(run / "stopped-resource-ledger-1.json")
    findings, intervals, evaluations, details = [], [], [], []

    def check(condition, code, **context):
        if not condition:
            findings.append(dict(code=code, **context))

    for name, saved_driver in (("matrix-continuation-protocol.json", "continue-matrix-driver.py"),
                               ("remaining-matrices-protocol.json", "continue-remaining-matrices-driver.py")):
        amendment = read(run / name)["amendment"]
        check(digest(run / saved_driver) == amendment["driver_sha256"], "continuation_driver_digest", protocol=name)
        check(amendment["original_matrix_started_at"] == 1942.1477145
              and amendment["original_matrix_deadline"] == 3742.1477145
              and amendment["original_package_started_at"] == protocol["package_started_at"],
              "continuation_original_clocks", protocol=name)
    for saved in (run / "stopped-matrix-sources-1").iterdir():
        if saved.is_file():
            check(digest(saved) == digest(run / "matrix-sources" / saved.name),
                  "preserved_matrix_source_bytes", path=saved.name)

    known_protocols = {protocol["protocol_sha256"]: protocol}
    for name in ("matrix-continuation-protocol.json", "remaining-matrices-protocol.json"):
        continuation = read(run / name)
        known_protocols[continuation["protocol_sha256"]] = continuation
    index = {row["job"]: row for row in analysis["search_runs"]}
    raw_search = {}
    resource_records, matrix_streams = {}, []
    matrix_charges = {row["global_attempt"]: row for row in lines(run / "matrix-attempts.jsonl")}
    for path in sorted((run / "jobs").glob("*/job.json")):
        job = read(path)
        record = read(path.with_name("resources.json"))
        resource = record["resources"]
        label = job["job"]
        resource_records[label] = record
        bound = known_protocols.get(job["protocol_sha256"])
        check(bound is not None, "job_protocol", job=label)
        check(bound is not None and job["source_sha256"] == bound["source_sha256"], "job_source", job=label)
        check(job["package_deadline"] == protocol["package_started_at"] + 7200, "job_package_deadline", job=label)
        if label == "matrix-32":
            check(record["status"] == "monitoring_interrupted" and record["monitoring_complete"] is False,
                  "size32_monitoring_status")
            check(all(resource[field] is None for field in ("completed_at", "elapsed_seconds", "peak_tree_rss_bytes",
                  "minimum_available_ram_bytes", "samples")), "size32_unknown_resources")
            end = record["child_absence_confirmed_at"]
        else:
            check(resource["peak_tree_rss_bytes"] <= 8 * 1024**3, "rss_limit", job=label)
            check(resource["minimum_available_ram_bytes"] >= 4 * 1024**3, "ram_reserve", job=label)
            end = resource["completed_at"]
        intervals.append((resource["started_at"], end, label))
        if job["kind"].startswith("matrix"):
            for charge in (row for row in matrix_charges.values() if row["job"] == label):
                cutoff = min(job["compute_admission_deadline"], job.get("request_admission_deadline", job["admission_deadline"]))
                check(charge["started_at"] < cutoff, "matrix_late_admission", job=label,
                      attempt=charge["global_attempt"])
            for event in lines(path.with_name("matrix-events.jsonl")):
                charge = matrix_charges[event["global_attempt"]]
                check(all(charge[field] == event[field] for field in ("n", "word", "arm")),
                      "matrix_event_charge", job=label, attempt=event["global_attempt"])
            matrix_streams.extend(dict(row, job=label) for row in lines(path.with_name("matrix-streams.jsonl")))
        events = lines(path.with_name("events.jsonl"))
        evaluations.extend(events)
        if events:
            baseline = read(run / "sources" / str(job["split_seed"]) / "manifest.json")["baseline_validation"]
            for row in events:
                if row.get("status") == "finite":
                    accepted = (row["metrics"]["cross_entropy"] <= baseline["cross_entropy"] + .05
                                and row["metrics"]["accuracy"] >= baseline["accuracy"] - .02)
                    check(row["quality_accepted"] == accepted, "quality_acceptance", job=label,
                          attempt=row["global_attempt"])
        if job["kind"] != "search":
            continue
        record = index[label]
        trace = lines(path.with_name("search-trace.jsonl"))
        proposals = sum(row.get("event") == "proposal" for row in trace)
        hits = sum(row.get("event") == "cache_observation" and row.get("cache_hit") is True for row in trace)
        check(proposals == record["created_proposals"], "proposal_count", job=label)
        check(hits == record["cache_hits"], "cache_count", job=label)
        cap = 15 if job["mode"] == "R" else 24 if job["budget"] == "attempts" else 440
        proposal_cap = ((192 if job["budget"] == "attempts" else 1024) if job["mode"] == "E"
                        else 15 if job["mode"] == "R" else 440)
        check(record["attempts"] <= cap, "search_attempt_limit", job=label)
        check(proposals <= proposal_cap, "proposal_limit", job=label)
        identities = [row["solution_id"] for row in events]
        check(len(identities) == len(set(identities)), "duplicate_full_evaluation", job=label)
        raw_search[label] = (job, events)
        details.append(dict(job=label, proposals=proposals, cache_hits=hits,
                            attempts=record["attempts"], on_time=record["on_time"], late=record["late"]))
    for previous, following in zip(sorted(intervals), sorted(intervals)[1:]):
        check(previous[1] <= following[0], "overlapping_workers", previous=previous[2], following=following[2])
    check(ledger["started_at"] == stopped["started_at"] == protocol["package_started_at"], "continuation_origin")
    check(ledger["historical_stop_reason"] == stopped["stop_reason"], "preserved_failure_reason")
    check(ledger["jobs"][:len(stopped["jobs"])] == stopped["jobs"], "preserved_old_jobs")
    stopped2 = read(run / "stopped-resource-ledger-2.json")
    check(ledger["jobs"][:len(stopped2["jobs"])] == stopped2["jobs"], "preserved_second_stop_jobs")
    check(protocol["limits"]["matrix_request_peak_bytes"] == 1024**3, "approved_matrix_cap")
    stage = next(row for row in ledger["stages"] if row["stage"] == "matrix")
    check(stage["started_at"] == 1942.1477145 and stage["deadline"] == 3742.1477145,
          "original_matrix_window")
    check(stage["completed_at"] <= stage["deadline"], "matrix_stage_deadline")

    for actual in analysis["common_bytes"]:
        for mode in "RT":
            name = "search-%s-%s-%s-%s" % (actual["model_seed"], actual["budget"], actual["search_seed"], mode)
            job, events = raw_search[name]
            expected = min((row["metrics"]["cross_entropy"] for row in events if row["status"] == "finite"
                            and row["tensor_bytes"] == actual["tensor_bytes"]
                            and row["completed_at"] <= job["admission_deadline"]), default=None)
            check(expected == actual["CE_" + mode], "common_bytes", job=name, tensor_bytes=actual["tensor_bytes"])

    selected, unique_report = analysis["test_selected"], []
    for seed in (19901, 19911, 19921):
        rows = [row for row in selected if row["model_seed"] == seed]
        groups = {}
        for row in rows:
            groups.setdefault(row["solution_id"], []).append(row)
        for identity, duplicates in groups.items():
            for field in ("test_CE", "test_accuracy", "test_macro_F1", "tensor_bytes", "checkpoint_sha256", "global_attempt"):
                check(len({row[field] for row in duplicates}) == 1, "logical_duplicate_disagreement",
                      model_seed=seed, solution_id=identity, field=field)
        unique = [duplicates[0] for duplicates in groups.values()]
        unique_report.append(dict(model_seed=seed, logical_choices=len(rows), distinct_model_solutions=len(unique),
            test_accuracy_range=[min(row["test_accuracy"] for row in unique), max(row["test_accuracy"] for row in unique)],
            test_CE_range=[min(row["test_CE"] for row in unique), max(row["test_CE"] for row in unique)],
            oracle_validation_selected=[row for row in rows if row["job"].startswith("oracle-")]))
    frozen = read(run / "selected.json")
    check(len(frozen["selected"]) == len(selected) == 171, "logical_selection_count")
    check(sum(row["distinct_model_solutions"] for row in unique_report) == 55, "distinct_selection_count")
    matrix_results = []
    for n in (8, 32, 128, 512, 1024):
        observed = [row for row in matrix_streams if row["n"] == n]
        check(len(observed) == (105 if n == 1024 else 107), "completed_stream_count", n=n)
        record = resource_records[f"matrix-{n}"]
        monitored = (record["status"] in ("succeeded", "partial")
                     and record.get("monitoring_complete") is not False
                     and record["resources"]["completed_at"] is not None
                     and record["resources"]["peak_tree_rss_bytes"] is not None)
        for invalid in (0, 2, 4, 6):
            cell = [row for row in observed if row["invalid_count"] == invalid]
            timed = {(row["block"], row["arm"]): row for row in cell if row["group"] == "timed"}
            controls = {(row["pair"], row["label"]): row for row in cell if row["group"] == "control"}
            blocks = [block for block in range(6) if all((block, arm) in timed for arm in "ABC")]
            bb = [controls[pair, "B1"]["primary_seconds"] / controls[pair, "B2"]["primary_seconds"]
                  for pair in range(4) if all((pair, label) in controls for label in ("B1", "B2"))]
            bb_ok = len(bb) == 4 and .95 <= median(bb) <= 1.05 and all(.85 <= value <= 1.15 for value in bb)
            for arm in "AC":
                ratios, before, after = [], [], []
                for block in blocks:
                    numerator, denominator = timed[block, arm], timed[block, "B"]
                    ratio = numerator["primary_seconds"] / denominator["primary_seconds"]
                    ratios.append(ratio)
                    (before if numerator["started_at"] < denominator["started_at"] else after).append(ratio)
                order_ratio = median(before) / median(after) if len(before) == len(after) == 3 else None
                baseline = median(timed[block, "B"]["primary_seconds"] for block in blocks) if blocks else None
                eligible = bool(monitored and len(blocks) == 6 and bb_ok and order_ratio is not None
                                and .9 <= order_ratio <= 1.1 and baseline >= .05)
                win = bool(eligible and median(ratios) <= .9 and sum(value < 1 for value in ratios) >= 5)
                actual = next(row for row in analysis["matrix_summary"] if row["n"] == n
                              and row["invalid_count"] == invalid and row["arm"] == arm)
                check(actual["eligible"] == eligible and actual["observed_win"] == win,
                      "matrix_claim", n=n, invalid_count=invalid, arm=arm)
                check(actual["median_ratio"] == median(ratios) and actual["order_ratio"] == order_ratio,
                      "matrix_statistic", n=n, invalid_count=invalid, arm=arm)
                matrix_results.append(dict(n=n, invalid_count=invalid, arm=arm, timed_blocks=len(blocks),
                    control_pairs=len(bb), median_ratio=median(ratios), order_ratio=order_ratio,
                    bb_pass=bb_ok, median_b_seconds=baseline, resource_eligible=monitored,
                    eligible=eligible, observed_win=win))
    if archive is not None:
        manifest = read(archive / "artifact-manifest.json")
        with zipfile.ZipFile(archive / "raw-evidence.zip") as zipped:
            archived = {row["path"]: row for row in manifest if row["archived"]}
            check(set(zipped.namelist()) == set(archived), "archive_member_set")
            for name, row in archived.items():
                check(hashlib.sha256(zipped.read(name)).hexdigest() == row["sha256"], "archive_member_digest", path=name)
    return dict(schema="issue99-local2-additional-audit-v1", status="passed" if not findings else "failed",
        findings=findings, scope="stdlib metadata and arithmetic; no tensor deserialization or model execution",
        inputs_sha256={name: digest(run / name) for name in ("analysis.json", "protocol.json", "resource-ledger.json",
                                                            "stopped-resource-ledger-1.json", "selected.json")},
        package_status=ledger["status"], package_elapsed_seconds=ledger["elapsed_seconds"],
        original_clock=ledger["started_at"], stopped_elapsed_seconds=stopped["elapsed_seconds"],
        counters_before=[stopped["mlp_attempts"], stopped["matrix_attempts"]],
        counters_after=[ledger["mlp_attempts"], ledger["matrix_attempts"]],
        worker_intervals=len(intervals), evaluation_status_counts=dict(Counter(row["status"] for row in evaluations)),
        baselines=analysis["baselines"], search_runs=details, search_summary=analysis["search_summary"],
        paired=analysis["paired"], common_bytes=analysis["common_bytes"], selected_models=unique_report,
        matrix_streams=len(matrix_streams), matrix=matrix_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    result = audit(args.run, args.archive)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "findings", "counters_after")}))
    raise SystemExit(0 if result["status"] == "passed" else 1)
