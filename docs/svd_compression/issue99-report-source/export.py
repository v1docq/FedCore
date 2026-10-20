"""Export recorded issue-99 evidence; performs no model execution or fitting."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zipfile

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from fedcore.experiments.compression_search_analysis import build_analysis


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def scalar_rows(rows):
    return [{key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list, tuple))
             else value for key, value in row.items()} for row in rows]


def write_csv(path, rows):
    rows = scalar_rows(rows)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    analysis = build_analysis(args.run)
    save(output / "analysis.json", analysis)
    summary = []
    for model in (41, 42, 43):
        for budget in ("attempts", "time"):
            for mode in "RTE":
                rows = [row for row in analysis["search_runs"] if row["model_seed"] == model
                        and row["budget"] == budget and row["mode"] == mode]
                if not rows:
                    continue
                finite = [row["normalized_hypervolume"] for row in rows]
                summary.append({"model_seed": model, "budget": budget, "mode": mode,
                    "n": len(rows), "succeeded": sum(row["status"] == "succeeded" for row in rows),
                    "median_hv": statistics.median(finite), "min_hv": min(finite), "max_hv": max(finite),
                    "median_evaluations": statistics.median(row["finite_within_deadline"] for row in rows),
                    "median_parent_seconds": statistics.median(row["resources"].get("elapsed_seconds", 0) for row in rows),
                    "median_accepted_hv": statistics.median(row["accepted_normalized_hypervolume"] for row in rows)})
    curves, frontiers = [], []
    for run in analysis["search_runs"]:
        curves.extend({"model_seed": run["model_seed"], "search_seed": run["search_seed"],
            "mode": run["mode"], "budget": run["budget"], **row} for row in run["curves"])
    for frontier in analysis["frontiers"]:
        for row in frontier["all"]:
            frontiers.append({"job": frontier["job"], **row,
                "KiB": row["tensor_bytes"] / 1024, "CE": row["quality"] * 0.6931471805599453})
    tests = [{"model_seed": job["model_seed"], **row,
              "test_CE": row["test"]["cross_entropy"], "test_accuracy": row["test"]["accuracy"]}
             for job in analysis["final_test"] for row in job.get("selected", [])]
    baselines = [{"model_seed": job["model_seed"], **job["baseline_test"]}
                 for job in analysis["final_test"] if job.get("baseline_test")]
    resources = analysis["resources"]
    tables = {"search_summary": summary, "search_runs": analysis["search_runs"],
        "paired": analysis["paired"], "common_bytes": analysis["common_byte_pairs"],
        "curves": curves, "frontiers": frontiers, "early_late": analysis["early_late_pairs"],
        "cache": analysis["cache_pairs"], "order": analysis["order_pairs"],
        "test_selected": tests, "baseline_test": baselines,
        "resources": resources["execution_lineage"], "admissibility": [analysis["admissibility"]]}
    for name, rows in tables.items():
        write_csv(output / (name + ".csv"), analysis["admissibility"]["rows"] if name == "admissibility" else rows)
    save(output / "frontiers.json", analysis["frontiers"])
    source_manifests = []
    for seed in (41, 42, 43):
        path = args.run / "jobs" / f"oracle-{seed}" / "source-manifest.json"
        if path.exists():
            source_manifests.append(json.loads(path.read_text(encoding="utf-8")))
    save(output / "source-manifest.json", source_manifests)
    catalog = args.run / "jobs/oracle-41/catalog.json"
    if catalog.exists():
        (output / "catalog.json").write_bytes(catalog.read_bytes())
    for name in ("protocol.json", "resource-ledger.json", "mlp-attempts.jsonl", "selected.json", "reused-evidence.json"):
        source = args.run / name
        if source.exists():
            (output / name).write_bytes(source.read_bytes())
    # Preserve original metadata/event bytes for all revisions. Binary model
    # artifacts remain in their recorded local directories with SHA-256 IDs.
    manifest = []
    with zipfile.ZipFile(output / "raw-evidence.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for revision in reversed(resources["execution_lineage"]):
            directory = Path(revision["directory"])
            for path in sorted(directory.rglob("*")):
                if not path.is_file() or path.suffix not in (".json", ".jsonl", ".log") or "tmp" in path.relative_to(directory).parts:
                    continue
                relative = path.relative_to(directory)
                # Copied witnesses are already archived under their origin.
                reused = json.loads((directory / "reused-evidence.json").read_text(encoding="utf-8")) if (directory / "reused-evidence.json").exists() else {}
                if len(relative.parts) > 1 and relative.parts[0] == "jobs" and relative.parts[1] in reused:
                    continue
                member = directory.name + "/" + relative.as_posix()
                raw = path.read_bytes()
                archive.writestr(member, raw)
                manifest.append({"archive_member": member, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    save(output / "raw-evidence-manifest.json", manifest)
    now = datetime.now(timezone.utc).isoformat()
    queries = {}
    for name, rows in tables.items():
        reviewed = [{key: value for key, value in row.items() if not isinstance(value, (dict, list, tuple))}
                    for row in rows]
        if name == "order":
            for flat, original in zip(reviewed, rows):
                flat["delta_CE"] = (original.get("metrics_second_minus_first") or {}).get("cross_entropy")
        if name == "cache":
            for flat in reviewed:
                flat["pair_label"] = f"{flat['model_seed']}/{flat['stream_seed']}"
        if name == "common_bytes":
            for flat, original in zip(reviewed, rows):
                for mode in "RT":
                    flat["CE_" + mode] = ((original.get(mode) or {}).get("metrics") or {}).get("cross_entropy")
        queries[name] = {"rows": reviewed, "source": {
            "label": "FedCore issue #99: recorded local CPU package",
            "files": ["analysis.json", name + ".csv", "raw-evidence.zip"], "executedAt": now,
            "evidenceFlow": [{"title": "Frozen experiment protocol", "detail": "protocol.json; source build " + str(json.loads((args.run / "protocol.json").read_text(encoding="utf-8"))["git_revision"])},
                {"title": "Recorded executions", "detail": "raw-evidence.zip retains all three execution revisions; transferred witnesses retain their original timestamps and hashes."},
                {"title": "Analysis", "detail": "compression_search_analysis.build_analysis(run_directory), then issue99-report-source/export.py; no model reruns."}],
            "metricDefinitions": [{"label": "HV", "definition": "Minimization hypervolume: CE/ln(2), tensor_bytes/49416, fixed reference (2,1.05); displayed normalized HV is divided by 2.1."},
                {"label": "Replication", "definition": "Three trained models; three nested search seeds within each. Historical validation and test were already exposed."}],
            "caveats": ["Historical engineering control; no fresh generalization claim.", "Constructed rejection and cache-repeat shares; no native int8/GPU speed claim."]},
            "methods": [{"language": "python", "code": "build_analysis(run_directory); see docs/svd_compression/issue99-report-source/export.py for the exact transformation."}]}
    snapshot = {"surface": "report", "title": "FedCore: проверка графового поиска сжатия",
        "generatedAt": now, "buildStatus": "creating", "status": "observed",
        "report": {"asOf": "2026-10-20"}, "filters": [], "queries": queries}
    if (output / "reviewed-snapshot.json").exists():
        old_snapshot = json.loads((output / "reviewed-snapshot.json").read_text(encoding="utf-8"))
        if old_snapshot.get("id"):
            snapshot["id"] = old_snapshot["id"]
    save(output / "reviewed-snapshot.json", snapshot)
    hashes = [{"file": path.name, "bytes": path.stat().st_size,
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
              for path in sorted(output.iterdir()) if path.is_file() and path.name != "manifest.json"]
    save(output / "manifest.json", hashes)
    print(json.dumps({"status": analysis["status"], "search_runs": len(summary),
        "attempts": resources["recorded_mlp_attempts"], "elapsed_seconds": resources["ledger"].get("elapsed_seconds")}))


if __name__ == "__main__":
    main()
