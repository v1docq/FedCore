"""Descriptive report from preserved pilot records; no adaptive computation."""
import json
from pathlib import Path

from fedcore.experiments.quantization_pilot_guard import atomic_json
from fedcore.experiments.quantization_pilot_protocol import quantile, file_hash, search_execution_summary


def analyze(plan, root):
    root = Path(root)
    closed = json.loads((root / "closed_test.json").read_text()) if (root / "closed_test.json").exists() else {"jobs": {}}
    rows = []
    for job in plan.payload["jobs"]:
        path = root / "jobs" / (job+".json")
        if not path.exists():
            rows.append({"job": job, "status": "missing"}); continue
        result = json.loads(path.read_text())
        artifact = result.get("artifact", {})
        engine = search_execution_summary(result)
        status = "partial_artifact" if artifact and engine["search_completed"] is False else result["status"]
        rows.append({"job": job, "status": status, "recorded_job_status": result["status"],
                     "artifact_status": "verified_finite_roundtrip" if artifact else "absent", **engine,
                     "search_requested_evaluations": plan.payload["search"]["evaluations"] if engine["search_status"] is not None else None,
                     "physical_tensor_bytes": artifact.get("physical_tensor_bytes"),
                     "file_bytes": artifact.get("file_bytes"), "budget_bytes": result.get("budget_bytes"),
                     "within_budget": result.get("within_budget"), "wall_seconds": result.get("wall_seconds"),
                     "selection": result.get("selection"), "closed_test": closed["jobs"].get(job),
                     "failure": result.get("failure", result.get("exception", engine["search_reason"])), "result_sha256": file_hash(path)})
    profiles = [json.loads(p.read_text()) for p in sorted((root / "timing").glob("*.json"))] if (root / "timing").exists() else []
    timing = []
    for job in plan.payload["jobs"]:
        for batch in (1, 128):
            for threads in (1, 8):
                relevant = [p for p in profiles if p["job"] == job and p["batch"] == batch and p["threads"] == threads]
                aa = [p for p in profiles if p["job"] == "dense32" and p.get("mode") == "aa" and p["batch"] == batch and p["threads"] == threads]
                ab = [p for p in relevant if p.get("mode") == "ab"]
                single = [p for p in relevant if p.get("mode") == "single"]
                complete = (len(aa) == 5 and all(p.get("aa_passed") for p in aa) and len(ab) == 5
                            and all(p["summary"]["p95_eligible"] and p["control_summary"]["p95_eligible"] for p in ab))
                timing.append({"job": job, "batch": batch, "threads": threads, "aa_processes": len(aa), "ab_processes": len(ab),
                               "single_processes": len(single), "strict_speed_claim_eligible": complete,
                               "median_speedup": quantile([p["median_time_ratio_A_over_B"] for p in ab], .5) if complete else None,
                               "raw_profile_files": [str(p.relative_to(root)) for p in sorted((root / "timing").glob(f"{job}-*-b{batch}-t{threads}-r*.json"))]})
    guards = [json.loads(path.read_text()) for path in sorted((root / "guards").glob("*.json"))] if (root / "guards").exists() else []
    guards = [record for record in guards if record.get("schema") == "pr166_guard_v1"]
    report = {"schema": "pr166_pilot_report_v1", "manifest_sha256": plan.fingerprint, "quality_storage": rows,
              "resource_accounting": {"guard_receipts": len(guards),
                  "receipts_including_registered_controller_rss": sum(bool(record.get("registered_controller_identities")) for record in guards),
                  "receipts_excluding_controller_rss": sum(not record.get("registered_controller_identities") for record in guards),
                  "historical_limit": "Receipts without a registered controller cover guardian plus workers; controller RSS is excluded"},
              "timing": timing, "claims": ["Descriptive L0 technical pilot on one untrained controlled-spectrum teacher",
              "Synthetic CE and agreement measure teacher fidelity; no applied task-quality or search superiority inference",
              "No speedup claim without five complete A/A and paired AB/BA process repetitions in the same regime",
              "Sampled family RSS can miss peaks between samples; kernel private-commit protection is a separate metric"],
              "closed_test_opened": (root / "CLOSED_TEST_OPENED.json").exists()}
    atomic_json(root / "REPORT.json", report)
    lines = ["# PR166 local CPU pilot", "", "This is a descriptive technical pilot with synthetic teacher labels.", "",
             "| Job | Artifact / job status | Search status | Charged evaluations | Tensor bytes | File bytes | Selection CE | Closed-test agreement |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        selection, test = row.get("selection") or {}, row.get("closed_test") or {}
        lines.append("| "+" | ".join(str(v) if v is not None else "unknown" for v in (row["job"], row["status"], row.get("search_status"),
                     row.get("search_charged_evaluations"), row.get("physical_tensor_bytes"),
                     row.get("file_bytes"), selection.get("cross_entropy"), test.get("teacher_agreement")))+" |")
    failed_engines = [row for row in rows if row.get("search_completed") is False]
    if failed_engines:
        lines += ["", "Search engine failures (valid partial artifacts remain available):", ""]
        for row in failed_engines:
            lines.append(f"- {row['job']}: {row.get('search_reason')}; {row.get('search_charged_evaluations')} charged evaluations, "
                         f"{row.get('search_variation_candidates_assessed')} variation candidates assessed.")
    lines += ["", "All failures and partial profiles remain in the run directory. See REPORT.json for strict timing eligibility."]
    (root / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report
