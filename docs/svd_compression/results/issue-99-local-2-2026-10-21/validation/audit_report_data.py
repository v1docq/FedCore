"""Independent report-row and retrospective plateau audit, with stdlib only."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(value) for value in path.read_text(encoding="utf-8").splitlines() if value]


def raw_hv(rows):
    points = [(row["quality"], row["cost"]) for row in rows if row["quality"] < 2 and row["cost"] < 1.05]
    edges = sorted({point[0] for point in points} | {2.})
    return sum((b-a) * max([0.] + [1.05-y for x, y in points if x <= a]) for a, b in zip(edges, edges[1:]))


def audit(run, archive, data_path):
    analysis, data = read(run / "analysis.json"), read(data_path)
    queries, findings, plateaus = data["queries"], [], []

    def check(condition, code, **extra):
        if not condition:
            findings.append(dict(code=code, **extra))

    def close(a, b):
        return a is b if a is None or b is None else math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)

    for name in ("baselines", "search_summary", "search_runs", "curves", "paired", "common_bytes", "oracles",
                 "frontiers", "test_selected", "matrix_summary", "matrix_streams", "jobs"):
        expected, actual = analysis[name], queries[name]["rows"]
        check(len(expected) == len(actual), "query_row_count", query=name)
        for source, shown in zip(expected, actual):
            for key, value in source.items():
                if isinstance(value, (dict, list, tuple)):
                    continue
                expected_value = str(value) if key == "model_seed" and value is not None else value
                check(shown.get(key) == expected_value, "query_scalar", query=name, field=key)
    for name, query in queries.items():
        check(bool(query.get("source", {}).get("evidenceFlow")), "missing_query_provenance", query=name)
        path = archive / (name + ".csv")
        check(path.exists(), "missing_csv", query=name)
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as stream:
                check(len(list(csv.DictReader(stream))) == len(query["rows"]), "csv_row_count", query=name)

    plateau_rows = {row["job"]: row for row in queries["plateau"]["rows"]}
    for search in analysis["search_runs"]:
        actual = search["virtual_stop"]
        shown = plateau_rows[search["job"]]
        check(all(shown.get(key) == value for key, value in actual.items()), "plateau_query_copy", job=search["job"])
        if search["mode"] == "R":
            check(actual["status"] == "not_applicable", "plateau_R_applicability", job=search["job"])
            plateaus.append(dict(job=search["job"], status="not_applicable"))
            continue
        folder = run / "jobs" / search["job"]
        job = read(folder / "job.json")
        rows = sorted([row for row in lines(folder / "events.jsonl") if row["status"] == "finite"
                       and row["completed_at"] <= job["admission_deadline"]],
                      key=lambda row: (row["completed_at"], row["global_attempt"], row["solution_id"]))
        seen, unique = set(), []
        for row in rows:
            if row["solution_id"] not in seen:
                seen.add(row["solution_id"])
                unique.append(row)
        stop, previous, quiet = None, 0., 0
        for count in range(1, len(unique)+1):
            current = raw_hv(unique[:count])
            if count > 12:
                quiet = 0 if current-previous > 1e-6 else quiet+1
                if quiet == 6:
                    stop = count
                    break
            previous = current
        status = "triggered" if stop else "not_triggered"
        full = raw_hv(unique)
        check(actual["status"] == status and actual["stop_after_finite"] == stop, "plateau_stop", job=search["job"])
        check(actual["finite_unique_evaluations"] == len(unique), "plateau_finite_count", job=search["job"])
        check(close(actual["full_hypervolume"], full), "plateau_full_hv", job=search["job"])
        check(actual["actual_time_saved"] is None, "plateau_unmeasured_savings", job=search["job"])
        if stop:
            check(close(actual["prefix_hypervolume"], current) and close(actual["missed_hypervolume"], full-current),
                  "plateau_prefix_hv", job=search["job"])
            check(actual["stop_global_attempt"] == unique[stop-1]["global_attempt"], "plateau_global_attempt", job=search["job"])
        plateaus.append(dict(job=search["job"], status=status, stop_after_finite=stop))

    stream_index = {(row["n"], row["invalid_count"], row["block"], row["arm"]): row
                    for row in analysis["matrix_streams"] if row["group"] == "timed"}
    for shown in queries["matrix_pairs"]["rows"]:
        a, b = (stream_index[shown["n"], shown["invalid_count"], shown["block"], arm] for arm in (shown["arm"], "B"))
        check(close(shown["ratio"], a["primary_seconds"]/b["primary_seconds"]), "pair_ratio")
        check(shown["before_b"] == (a["started_at"] < b["started_at"]), "pair_order")
        check(shown["resource_eligible"] == (a["resource_eligible"] and b["resource_eligible"]), "pair_resource")
    controls = {(row["n"], row["invalid_count"], row["pair"], row["label"]): row
                for row in analysis["matrix_streams"] if row["group"] == "control"}
    for shown in queries["matrix_controls"]["rows"]:
        b1, b2 = (controls[shown["n"], shown["invalid_count"], shown["pair"], label]["primary_seconds"] for label in ("B1", "B2"))
        check(close(shown["ratio"], b1/b2) and shown["B1_seconds"] == b1 and shown["B2_seconds"] == b2, "control_ratio")
    check(len(queries["matrix_pairs"]["rows"]) == 240, "matrix_pairs_count")
    check(len(queries["matrix_controls"]["rows"]) == 79, "matrix_controls_count")
    return dict(schema="issue99-local2-report-data-audit-v1", status="passed" if not findings else "failed",
                findings=findings, data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                data_id=data.get("id"), buildStatus=data.get("buildStatus"),
                query_counts={name: len(query["rows"]) for name, query in queries.items()},
                plateau_counts=dict(Counter(row["status"] for row in plateaus)), plateau=plateaus)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "archive", "data", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.archive, args.data)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "findings", "query_counts", "plateau_counts")}))
    raise SystemExit(0 if result["status"] == "passed" else 1)
