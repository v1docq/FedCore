"""Independent stdlib-only replay of local2 evidence; never loads a tensor.

Usage: python -B verify_run.py --run PACKAGE --report REPORT.json
Missing observations remain missing. This checks file/metadata claims and does
not replace numerical admission of the runtime.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import struct

SIZES = (8, 32, 128, 512, 1024)
COUNTS = (0, 2, 4, 6)
ORDERS = ("ABC", "BCA", "CAB", "ACB", "CBA", "BAC")
CAPS = (20671, 41342, 58568)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(value) for value in Path(path).read_text(encoding="utf-8").splitlines()
            if value.strip()] if Path(path).exists() else []


def independent_hv(points):
    # Integrate vertical strips at every observed x rather than using the
    # production Pareto-frontier sweep. No clipping changes stored observations.
    points = [(x, y) for x, y in points if x < 2 and y < 1.05]
    edges = sorted({x for x, _ in points} | {2.})
    return sum((right-left) * max([0.] + [1.05-y for x, y in points if x <= left])
               for left, right in zip(edges, edges[1:])) / 2.1


def matrix_claims(streams):
    claims = []
    for n in SIZES:
        for invalid in COUNTS:
            cell = [row for row in streams if row["n"] == n and row["invalid_count"] == invalid
                    and row.get("request_count") == 8]
            timed = {(row["block"], row["arm"]): row for row in cell if row["group"] == "timed"}
            controls = {(row["pair"], row["label"]): row for row in cell if row["group"] == "control"}
            complete = [block for block in range(6) if all((block, arm) in timed for arm in "ABC")]
            bb = [controls[pair, "B1"]["primary_seconds"] / controls[pair, "B2"]["primary_seconds"]
                  for pair in range(4) if all((pair, label) in controls for label in ("B1", "B2"))]
            bb_ok = len(bb) == 4 and .95 <= median(bb) <= 1.05 and all(.85 <= ratio <= 1.15 for ratio in bb)
            for arm in "AC":
                ratios = [timed[block, arm]["primary_seconds"] / timed[block, "B"]["primary_seconds"] for block in complete]
                differences = [timed[block, arm]["primary_seconds"] - timed[block, "B"]["primary_seconds"] for block in complete]
                before = [ratio for block, ratio in zip(complete, ratios) if ORDERS[block].index(arm) < ORDERS[block].index("B")]
                after = [ratio for block, ratio in zip(complete, ratios) if ORDERS[block].index(arm) > ORDERS[block].index("B")]
                order_ratio = median(before)/median(after) if len(before) == len(after) == 3 else None
                baseline = median([timed[block, "B"]["primary_seconds"] for block in complete]) if complete else None
                eligible = len(complete) == 6 and bb_ok and order_ratio is not None and .9 <= order_ratio <= 1.1 and baseline >= .05
                claims.append(dict(n=n, invalid_count=invalid, arm=arm, complete_blocks=complete,
                    ratios=ratios, median_ratio=median(ratios) if ratios else None,
                    absolute_seconds=differences, bb_ratios=bb, bb_pass=bb_ok,
                    order_ratio=order_ratio, median_b_seconds=baseline, eligible=eligible,
                    observed_win=bool(eligible and median(ratios) <= .9 and sum(value < 1 for value in ratios) >= 5)))
    return claims


def audit(output, repo=None):
    output = Path(output).resolve()
    protocol, ledger = read(output/"protocol.json"), read(output/"resource-ledger.json")
    findings = []
    def check(condition, code, **details):
        if not condition:
            findings.append(dict(code=code, **details))
    payload = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    check(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() == protocol["protocol_sha256"], "protocol_digest")
    frozen_source_digests = {}
    if repo is not None:
        repo = Path(repo).resolve()
        for relative, expected in protocol["source_files"].items():
            path = repo / relative
            actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() if path.exists() else None
            frozen_source_digests[relative] = actual
            check(actual == expected, "executed_source_digest", path=relative)
        source_digest = hashlib.sha256(json.dumps(protocol["source_files"], sort_keys=True).encode()).hexdigest()
        check(source_digest == protocol["source_sha256"], "source_manifest_digest")
        plan = repo / "docs/svd_compression/PLAN_ISSUE_99_LOCAL_2.md"
        check(plan.exists() and sha(plan) == protocol["approved_plan_sha256"], "approved_plan_digest")
    known_protocols = {protocol["protocol_sha256"]: protocol}
    parent_source = "fedcore/experiments/compression_search_local2.py"
    immutable_fields = ("limits", "stage_seconds", "search_jobs", "matrix_schedule",
                        "package_started_at", "approved_plan_sha256", "environment")
    first_continuation = None
    for name, snapshot in (("matrix-continuation-protocol.json", "continued-parent-source.py"),
                           ("remaining-matrices-protocol.json", "continued-parent-source-2.py")):
        path = output/name
        if not path.exists():
            continue
        continuation = read(path)
        payload = {key: value for key, value in continuation.items() if key != "protocol_sha256"}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        check(digest == continuation.get("protocol_sha256"), "continuation_protocol_digest", protocol=name)
        source_files = continuation.get("source_files", {})
        source_digest = hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest()
        check(source_digest == continuation.get("source_sha256"), "continuation_source_manifest_digest", protocol=name)
        check(continuation.get("original_protocol_sha256") == protocol["protocol_sha256"],
              "continuation_original_protocol", protocol=name)
        for field in immutable_fields:
            check(field in continuation and continuation[field] == protocol[field],
                  "continuation_changed_frozen_field", protocol=name, field=field)
        changed = sorted(relative for relative in set(protocol["source_files"]) | set(source_files)
                         if protocol["source_files"].get(relative) != source_files.get(relative))
        check(changed == [parent_source], "continuation_source_change_scope", protocol=name, paths=changed)
        saved_parent = output/snapshot
        check(saved_parent.exists(), "continuation_parent_snapshot_missing", protocol=name, path=snapshot)
        if saved_parent.exists():
            actual = hashlib.sha256(saved_parent.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            check(actual == source_files.get(parent_source), "continuation_parent_source_digest", protocol=name, path=snapshot)
        if repo is not None:
            for relative, expected in source_files.items():
                if relative != parent_source:
                    check(frozen_source_digests.get(relative) == expected,
                          "continuation_executed_source_digest", protocol=name, path=relative)
        if name == "matrix-continuation-protocol.json":
            first_continuation = continuation
        else:
            check(first_continuation is not None, "continuation_previous_protocol_missing",
                  protocol=name, path="matrix-continuation-protocol.json")
            check(first_continuation is not None and continuation.get("previous_protocol_sha256") == first_continuation.get("protocol_sha256"),
                  "continuation_previous_protocol", protocol=name)
        declared = continuation.get("protocol_sha256")
        check(declared not in known_protocols, "duplicate_protocol_digest", protocol=name)
        known_protocols[declared] = continuation
    check(ledger["started_at"] == protocol["package_started_at"], "package_clock_origin")
    check(ledger.get("elapsed_seconds", 0) <= 7205, "package_clock_limit", elapsed=ledger.get("elapsed_seconds"))
    mlp, matrix, updates = (lines(output/name) for name in
        ("mlp-attempts.jsonl", "matrix-attempts.jsonl", "training-updates.jsonl"))
    for rows, key, limit, label in ((mlp, "attempt", 10300, "mlp"), (matrix, "global_attempt", 4350, "matrix")):
        check([row[key] for row in rows] == list(range(1, len(rows)+1)), "noncontiguous_attempts", family=label)
        check(len(rows) <= limit, "attempt_limit", family=label, count=len(rows))
        for row in rows:
            check(row["started_at"] < ledger["started_at"] + 6900, "late_attempt", family=label, attempt=row[key])
    check(ledger.get("mlp_attempts") == len(mlp), "mlp_ledger_total")
    check(ledger.get("matrix_attempts") == len(matrix), "matrix_ledger_total")
    starts = [row for row in updates if row["event"] == "update_started"]
    ends = [row for row in updates if row["event"] == "update_completed"]
    check([row["update"] for row in starts] == list(range(1, len(starts)+1)), "training_update_order")
    check(len(starts) <= 2040 and len(ends) <= len(starts), "training_update_limit")
    check(len({row["update"] for row in ends}) == len(ends), "duplicate_update_completion")
    all_rows, streams, summaries, jobs, rows_by_job = [], [], [], {}, {}
    charges = {row["attempt"]: row for row in mlp}
    observed_svd = svd_completed = forward_starts = forward_ends = 0
    for directory in sorted((output/"jobs").iterdir()):
        if not directory.is_dir() or not (directory/"job.json").exists():
            continue
        job = read(directory/"job.json")
        jobs[job["job"]] = job
        job_protocol = known_protocols.get(job.get("protocol_sha256"))
        check(job_protocol is not None, "unknown_job_protocol", job=job["job"], protocol_sha256=job.get("protocol_sha256"))
        if job_protocol is not None:
            check(job.get("source_sha256") == job_protocol.get("source_sha256"),
                  "job_source_protocol_disagreement", job=job["job"])
        result = read(directory/"result.json") if (directory/"result.json").exists() else None
        resource = read(directory/"resources.json") if (directory/"resources.json").exists() else {}
        events = [row for row in lines(directory/"events.jsonl") if row.get("event") == "evaluation"]
        rows_by_job[job["job"]] = events
        all_rows.extend(events)
        catalog = {row["solution_id"]: row["genotype"] for row in read(directory/"catalog.json")} if (directory/"catalog.json").exists() else {}
        for row in events:
            charge = charges.get(row["global_attempt"], {})
            check(charge.get("job") == job["job"] and charge.get("solution_id") == row["solution_id"], "evaluation_without_charge", job=job["job"], attempt=row["global_attempt"])
            if row.get("status") != "finite":
                continue
            genotype = catalog.get(row["solution_id"])
            check(genotype is not None, "unknown_solution", job=job["job"])
            if genotype:
                expected_bytes = 4 * (17226 + sum(192 * gene["choice"]["rank"] - 8192 for gene in genotype["genes"]))
                check(row["tensor_bytes"] == row["expected_tensor_bytes"] == expected_bytes, "storage_formula", job=job["job"], attempt=row["global_attempt"])
            check(math.isclose(row["quality"], row["metrics"]["cross_entropy"] / math.log(10), rel_tol=1e-13, abs_tol=1e-13), "ce_normalization", job=job["job"])
            check(math.isclose(row["cost"], row["tensor_bytes"]/68904, rel_tol=1e-13, abs_tol=1e-13), "cost_normalization", job=job["job"])
            check(row["tensor_bytes"] <= 65458 and row.get("inventory_verified_after_reload") is True, "storage_inventory", job=job["job"])
            artifact = row.get("artifact_path")
            if artifact:
                check(Path(artifact).exists() and sha(artifact) == row["checkpoint_sha256"], "artifact_digest", path=artifact)
        if job["kind"] in ("search", "oracle"):
            deadline = (result or {}).get("deadline", job["admission_deadline"])
            finite = sorted([row for row in events if row.get("status") == "finite" and row["completed_at"] <= deadline], key=lambda row: row["completed_at"])
            picks = []
            for cap in CAPS:
                available = [row for row in finite if row["quality_accepted"] and row["tensor_bytes"] <= cap]
                best = min(available, key=lambda row: (row["metrics"]["cross_entropy"], row["tensor_bytes"], row["solution_id"])) if available else None
                picks.append(dict(byte_cap=cap, solution_id=best["solution_id"] if best else None))
            charged = sum(row["job"] == job["job"] and row["purpose"] == "full_validation" for row in mlp)
            check(charged >= len(events), "completed_without_started_attempt", job=job["job"])
            on_time = sum(row["completed_at"] <= deadline for row in events)
            catalog_size = 15 if job.get("mode") == "R" else 440
            exhausted = len({row["solution_id"] for row in events if row["completed_at"] <= deadline}) == catalog_size
            valid_resource = resource.get("status") == "succeeded" and not (result or {}).get("partial")
            main_complete = bool(valid_resource and ((on_time >= min(24, catalog_size) or exhausted)
                if job.get("budget") == "attempts" else (deadline >= job["worker_started_at"] + 30 - 1e-6 or exhausted)))
            summaries.append(dict(job=job["job"], kind=job["kind"], model_seed=job["split_seed"],
                mode=job.get("mode"), budget=job.get("budget"), search_seed=job.get("search_seed"),
                charged_attempts=charged, on_time=on_time, main_complete=main_complete,
                completed_rows=len(events), incomplete_attempts=charged-len(events),
                on_time_finite=len(finite), late_finite=sum(row.get("status") == "finite" and row["completed_at"] > deadline for row in events),
                normalized_hypervolume=independent_hv([(row["quality"], row["cost"]) for row in finite]),
                full_oracle=bool(job["kind"] == "oracle" and resource.get("status") == "succeeded"
                    and (result or {}).get("status") == "succeeded" and not (result or {}).get("partial")
                    and (result or {}).get("complete_catalog") is True
                    and len(events) == len(finite) == 440
                    and len({row["solution_id"] for row in finite}) == 440),
                independent_selections=picks))
        monitored = (resource.get("status") in ("succeeded", "partial")
            and resource.get("resources", {}).get("peak_tree_rss_bytes") is not None
            and resource.get("stop") in (None, "hard_deadline", "request_hard_deadline"))
        streams.extend(dict(row, job=job["job"], resource_status=resource.get("status", "missing"),
                            resource_eligible=monitored) for row in lines(directory/"matrix-streams.jsonl"))
        matrix_events = lines(directory/"matrix-events.jsonl")
        for row in matrix_events:
            expected = int(row["word"] == "LD" or row["arm"] in ("B", "production"))
            check(row["observed_svd_starts"] == expected, "svd_law", job=job["job"], attempt=row["global_attempt"])
            check(row["accepted"] == (row["word"] == "LD"), "matrix_admission_law", job=job["job"])
        for stream in lines(directory/"matrix-streams.jsonl"):
            child_rows = [row for row in matrix_events if row.get("stream_id") == stream["stream_id"]]
            check(len(child_rows) == stream["request_count"] == 8, "matrix_stream_count", stream=stream["stream_id"])
            check(math.isclose(sum(row["call_seconds"] for row in child_rows), stream["primary_seconds"], rel_tol=1e-13, abs_tol=1e-13), "matrix_primary_timer", stream=stream["stream_id"])
        telemetry = directory/"svd-telemetry.bin"
        if telemetry.exists():
            value = telemetry.read_bytes()
            check(len(value) == 64, "telemetry_size", job=job["job"])
            if len(value) == 64:
                started, completed, _, _ = struct.unpack_from("<QQdd", value)
                observed_svd += started
                svd_completed += completed
                check(completed <= started and started-completed <= 1, "telemetry_svd_order", job=job["job"])
                check(sum(row["observed_svd_starts"] for row in matrix_events) <= started, "telemetry_understates_rows", job=job["job"])
        calls = lines(directory/"forward-calls.jsonl")
        fs = sum(row["event"] == "forward_started" for row in calls)
        fe = sum(row["event"] == "forward_completed" for row in calls)
        forward_starts += fs
        forward_ends += fe
        check(fe <= fs, "forward_completion_exceeds_start", job=job["job"])
    source_report = []
    for source_dir in sorted((output/"sources").iterdir()) if (output/"sources").exists() else ():
        if not source_dir.is_dir() or not (source_dir/"roles.json").exists():
            continue
        passport = read(source_dir/"roles.json")
        identifiers = []
        for role, size in (("train", 1078), ("validation", 269), ("calibration", 180), ("test", 270)):
            record = passport["roles"][role]
            ids = record["manifest"]["ids"]
            identifiers.extend(ids)
            check(len(ids) == size and sum(record["class_counts"].values()) == size, "role_size", source=source_dir.name, role=role)
            check(sha(source_dir/record["file"]) == record["file_sha256"], "role_file_digest", source=source_dir.name, role=role)
        check(len(identifiers) == len(set(identifiers)) == 1797 and set(identifiers) == {f"sklearn-digits:{i}" for i in range(1797)}, "role_disjoint_cover", source=source_dir.name)
        for old_seed, historical in passport["historical_manifests"].items():
            check(sha(historical["path"]) == historical["sha256"], "historical_manifest_digest", seed=old_seed)
            old = read(historical["path"])["data"]["roles"]
            for role, record in passport["roles"].items():
                for other, old_record in old.items():
                    expected = len(set(record["manifest"]["ids"]) & set(old_record["ids"]))
                    check(passport["historical_intersections"][old_seed][role][other] == expected, "historical_intersection", source=source_dir.name, role=role, other=other)
        manifest = read(source_dir/"manifest.json") if (source_dir/"manifest.json").exists() else None
        if manifest:
            check(sha(source_dir/"baseline.pt") == manifest["checkpoint_sha256"], "baseline_checkpoint_digest", source=source_dir.name)
            check(manifest["training"]["training_steps"] == 680 and len(manifest["training"]["train_loss"]) == 680, "training_steps", source=source_dir.name)
        source_report.append(dict(split_seed=passport["split_seed"], initial_state_sha256=passport["initial_state_sha256"], trained=manifest is not None))
    check(len({row["initial_state_sha256"] for row in source_report}) <= 1, "nonshared_initialization")
    selection = read(output/"selected.json") if (output/"selected.json").exists() else None
    if selection:
        check(len(selection["selected"]) <= 171, "logical_selection_limit")
        expected = {row["job"]: row for row in summaries}
        for row in selection["selected"]:
            picks = expected[row["job"]]["independent_selections"]
            chosen = next(item for item in picks if item["byte_cap"] == row["byte_cap"])
            check(chosen["solution_id"] == row["solution_id"], "selection_disagreement", job=row["job"], cap=row["byte_cap"])
        for directory in (output/"jobs").iterdir():
            if (directory/"job.json").exists():
                job = read(directory/"job.json")
                if job["kind"] == "finalize":
                    check(job.get("selections_sha256", job.get("selected_sha256")) == sha(output/"selected.json"), "selection_digest_before_test", job=job["job"])
                    check(job["worker_started_at"] >= selection["frozen_at"], "test_before_selection_freeze", job=job["job"])
    claims = matrix_claims(streams)
    for claim in claims:
        observed = [stream for stream in streams if stream["n"] == claim["n"]]
        monitored = bool(observed) and all(stream["resource_eligible"] for stream in observed)
        claim["resource_eligible"] = monitored
        claim["timing_controls_eligible"] = claim["eligible"]
        claim["eligible"] = claim["eligible"] and monitored
        claim["observed_win"] = claim["observed_win"] and monitored
    analysis = read(output/"analysis.json") if (output/"analysis.json").exists() else None
    analysis_checked = analysis is not None
    if analysis is not None:
        def close(actual, expected):
            return actual is expected if actual is None or expected is None else math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
        resources = analysis["resources"]
        for key, expected in (("mlp_ledger_count", len(mlp)), ("matrix_ledger_count", len(matrix)),
                ("training_updates_started", len(starts)), ("training_updates_completed", len(ends)),
                ("observed_svd_starts", observed_svd), ("observed_svd_completions", svd_completed)):
            check(resources.get(key) == expected, "analysis_resource_disagreement", field=key, expected=expected)
        indexed = {row["job"]: row for row in summaries}
        for actual in analysis["search_runs"]:
            expected = indexed[actual["job"]]
            for field, source in (("attempts", "charged_attempts"), ("completed_attempts", "completed_rows"),
                    ("incomplete_attempts", "incomplete_attempts"), ("on_time", "on_time"),
                    ("main_complete", "main_complete")):
                check(actual.get(field) == expected[source], "analysis_search_count_disagreement", job=actual["job"], field=field)
            check(close(actual["normalized_hypervolume"], expected["normalized_hypervolume"]), "analysis_hv_disagreement", job=actual["job"])
        for actual in analysis["oracles"]:
            expected = indexed[actual["job"]]
            check(actual["complete"] == expected["full_oracle"], "analysis_oracle_completeness", job=actual["job"])
            check(close(actual["partial_archive_hypervolume"], expected["normalized_hypervolume"]), "analysis_oracle_hv", job=actual["job"])
            check(close(actual["normalized_hypervolume"], expected["normalized_hypervolume"] if expected["full_oracle"] else None), "analysis_oracle_full_hv", job=actual["job"])
        matrix_index = {(row["n"], row["invalid_count"], row["arm"]): row for row in claims}
        for actual in analysis["matrix_summary"]:
            expected = matrix_index[actual["n"], actual["invalid_count"], actual["arm"]]
            for field in ("eligible", "observed_win", "bb_pass", "resource_eligible", "timing_controls_eligible"):
                check(actual.get(field) == expected[field], "analysis_matrix_gate", n=actual["n"], invalid_count=actual["invalid_count"], arm=actual["arm"], field=field)
            for field in ("median_ratio", "order_ratio", "median_b_seconds"):
                check(close(actual[field], expected[field]), "analysis_matrix_statistic", n=actual["n"], invalid_count=actual["invalid_count"], arm=actual["arm"], field=field)
            check(actual["pairs"] == len(expected["complete_blocks"]), "analysis_matrix_pairs", n=actual["n"], arm=actual["arm"])
        stream_index = {(row["job"], row["stream_id"]): row for row in streams}
        check(len(analysis["matrix_streams"]) == len(streams), "analysis_matrix_stream_count")
        for actual in analysis["matrix_streams"]:
            expected = stream_index.get((actual.get("job"), actual["stream_id"]))
            check(expected is not None, "analysis_unknown_matrix_stream", job=actual.get("job"), stream=actual["stream_id"])
            if expected is not None:
                for field in ("resource_status", "resource_eligible"):
                    check(actual.get(field) == expected[field], "analysis_matrix_stream_resource",
                          job=actual.get("job"), stream=actual["stream_id"], field=field)
        search_index = {(row["model_seed"], row["budget"], row["search_seed"], row["mode"]): row for row in summaries if row["kind"] == "search"}
        for actual in analysis["paired"]:
            first, second = actual["comparison"].split("→")
            prefix = actual["model_seed"], actual["budget"], actual["search_seed"]
            a, b = (search_index.get(prefix + (mode,)) for mode in (first, second))
            complete = bool(a and b and a["main_complete"] and b["main_complete"])
            expected = b["normalized_hypervolume"] - a["normalized_hypervolume"] if complete else None
            check(actual["complete"] == complete and close(actual["delta_hv"], expected), "analysis_pair_disagreement", comparison=actual["comparison"], model_seed=actual["model_seed"], budget=actual["budget"], search_seed=actual["search_seed"])
    return dict(schema="independent-local2-audit-v1", package=str(output), status="passed" if not findings else "failed",
        findings=findings, source_scope="stdlib file and arithmetic audit; tensor numerical checks remain separate",
        resources=dict(mlp_actions=len(mlp), matrix_requests=len(matrix), training_updates_started=len(starts),
            training_updates_completed=len(ends), observed_svd_starts=observed_svd,
            observed_svd_completed=svd_completed, forward_starts=forward_starts, forward_completions=forward_ends,
            mlp_purposes=dict(Counter(row["purpose"] for row in mlp))),
        source_passports=source_report, trajectories=summaries, matrix=claims, analysis_checked=analysis_checked,
        logical_selections=len(selection["selected"]) if selection else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3],
                        help="Checkout containing the executed source files; defaults to this repository")
    args = parser.parse_args()
    result = audit(args.run, args.repo)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "resources", "findings")}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
