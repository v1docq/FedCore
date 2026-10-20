"""Export recorded local-2 evidence with no model imports, inference or fitting."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import zipfile

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from fedcore.experiments.compression_search_analysis import normalized_hypervolume, pareto_points
from fedcore.experiments.compression_search_matrix import summarize_matrix


def load(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()] if Path(path).exists() else []


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def hv(rows, accepted=False):
    return normalized_hypervolume([(row["quality"],row["cost"]) for row in rows
        if row.get("status") == "finite" and (not accepted or row.get("quality_accepted"))])


def frontier(rows):
    points = set(pareto_points((row["quality"], row["cost"]) for row in rows if row.get("status") == "finite"))
    return [row for row in rows if row.get("status") == "finite" and (row["quality"],row["cost"]) in points]


def build_analysis(run):
    run = Path(run)
    protocol, ledger = load(run/"protocol.json"), load(run/"resource-ledger.json")
    charges = Counter(row["job"] for row in lines(run/"mlp-attempts.jsonl") if row.get("purpose") == "full_validation")
    search, curves, oracles, frontiers, finalized, jobs, matrix_streams = [], [], [], [], [], [], []
    evaluations_by_job = {}
    for path in sorted((run/"jobs").glob("*/job.json")):
        job = load(path)
        result = load(path.parent/"result.json", {})
        resource = load(path.parent/"resources.json", {})
        jobs.append(dict(job=job["job"],kind=job["kind"],status=resource.get("status", "missing"),
            elapsed_seconds=resource.get("resources",{}).get("elapsed_seconds"),
            peak_rss_bytes=resource.get("resources",{}).get("peak_tree_rss_bytes"),
            stop=resource.get("stop"),error=resource.get("error")))
        if job["kind"] == "matrix":
            matrix_streams.extend(lines(path.parent/"matrix-streams.jsonl"))
        if job["kind"] == "finalize":
            finalized.append(dict(result, model_seed=job["split_seed"]))
        if job["kind"] not in ("search", "oracle"):
            continue
        # Event files survive a hard stop even if result.json was never written.
        all_rows = lines(path.parent/"events.jsonl")
        all_rows.sort(key=lambda row: (row["completed_at"],row["global_attempt"]))
        deadline = min(job["admission_deadline"], result.get("deadline",job["admission_deadline"]))
        rows = [row for row in all_rows if row["completed_at"] <= deadline]
        evaluations_by_job[job["job"]] = rows
        finite = [row for row in rows if row["status"] == "finite"]
        accepted = [row for row in finite if row["quality_accepted"]]
        common = dict(job=job["job"],model_seed=job["split_seed"],mode=job.get("mode", "oracle"),
            budget=job.get("budget", "oracle"),search_seed=job.get("search_seed"))
        for row in frontier(finite):
            frontiers.append(dict(common, solution_id=row["solution_id"],CE=row["metrics"]["cross_entropy"],
                accuracy=row["metrics"]["accuracy"],tensor_bytes=row["tensor_bytes"],KiB=row["tensor_bytes"]/1024,
                quality_accepted=row["quality_accepted"]))
        if job["kind"] == "oracle":
            complete = (resource.get("status") == "succeeded" and result.get("status") == "succeeded"
                and result.get("complete_catalog") is True and len(finite) == 440
                and len({row["solution_id"] for row in finite}) == 440)
            oracles.append(dict(common,attempts=charges[job["job"]],completed_attempts=len(all_rows),
                incomplete_attempts=charges[job["job"]]-len(all_rows),on_time=len(rows),finite=len(finite),complete=complete,
                normalized_hypervolume=hv(rows) if complete else None,
                accepted_normalized_hypervolume=hv(rows,True) if complete else None,
                partial_archive_hypervolume=hv(rows),accepted=len(accepted)))
            continue
        catalog_size = 15 if job["mode"] == "R" else 440
        exhausted = len({row["solution_id"] for row in rows}) == catalog_size
        resource_ok = resource.get("status") == "succeeded" and not result.get("partial")
        if job["budget"] == "attempts":
            main_complete = resource_ok and (len(rows) >= min(24,catalog_size) or exhausted)
        else:
            full_window = job["admission_deadline"] >= job["worker_started_at"]+30-1e-6
            main_complete = resource_ok and (full_window or exhausted)
        trace = lines(path.parent/"search-trace.jsonl")
        summary = result.get("search") or {}
        created = summary.get("created_proposals")
        if created is None and trace:
            created = sum(row.get("event") == "proposal" for row in trace)
        cache_hits = summary.get("cache_hits")
        if cache_hits is None and trace:
            hits = [row for row in trace if row.get("event") == "cache_observation" and row.get("cache_hit") is True]
            cache_hits = len(hits)
        record = dict(common,attempts=charges[job["job"]],completed_attempts=len(all_rows),
            incomplete_attempts=charges[job["job"]]-len(all_rows),on_time=len(rows),finite=len(finite),accepted=len(accepted),
            late=len(all_rows)-len(rows),main_complete=main_complete,catalog_exhausted=exhausted,
            status=resource.get("status", "missing"),stop_reason=result.get("stop_reason",resource.get("stop")),
            normalized_hypervolume=hv(rows),accepted_normalized_hypervolume=hv(rows,True),
            created_proposals=created,cache_hits=cache_hits,
            trace_events=len(trace),parent_seconds=resource.get("resources",{}).get("elapsed_seconds"),
            peak_rss_bytes=resource.get("resources",{}).get("peak_tree_rss_bytes"))
        search.append(record)
        prefix = []
        for index, row in enumerate(rows,1):
            prefix.append(row)
            curves.append(dict(common,attempt=index,elapsed_parent_seconds=row["completed_at"]-job["worker_started_at"],
                normalized_hypervolume=hv(prefix),accepted_normalized_hypervolume=hv(prefix,True)))
    for row in search:
        oracle = next((item for item in oracles if item["model_seed"] == row["model_seed"] and item["complete"]),None)
        row["oracle_gap"] = oracle["normalized_hypervolume"]-row["normalized_hypervolume"] if oracle else None
        row["accepted_oracle_gap"] = oracle["accepted_normalized_hypervolume"]-row["accepted_normalized_hypervolume"] if oracle else None
    pairs, common_bytes = [], []
    index = {(row["model_seed"],row["budget"],row["search_seed"],row["mode"]):row for row in search}
    for model in (19901,19911,19921):
        for budget in ("attempts","time"):
            for seed in (11,22,33):
                for first,second in (("R","T"),("T","E")):
                    a,b = (index.get((model,budget,seed,mode)) for mode in (first,second))
                    complete = bool(a and b and a["main_complete"] and b["main_complete"])
                    pairs.append(dict(model_seed=model,budget=budget,search_seed=seed,comparison=first+"→"+second,
                        complete=complete,delta_hv=b["normalized_hypervolume"]-a["normalized_hypervolume"] if complete else None,
                        delta_accepted_hv=b["accepted_normalized_hypervolume"]-a["accepted_normalized_hypervolume"] if complete else None))
                a,b = (index.get((model,budget,seed,mode)) for mode in ("R","T"))
                for count in (15656,42280):
                    rows_a = evaluations_by_job.get(a["job"],[]) if a else []
                    rows_b = evaluations_by_job.get(b["job"],[]) if b else []
                    ce = lambda rows:min((row["metrics"]["cross_entropy"] for row in rows
                        if row.get("status") == "finite" and row["tensor_bytes"] == count),default=None)
                    ce_a,ce_b=ce(rows_a),ce(rows_b)
                    complete = bool(a and b and a["main_complete"] and b["main_complete"])
                    common_bytes.append(dict(model_seed=model,budget=budget,search_seed=seed,tensor_bytes=count,
                        main_complete=complete,R_attempts=a["on_time"] if a else None,T_attempts=b["on_time"] if b else None,
                        CE_R=ce_a,CE_T=ce_b,
                        delta_CE_T_minus_R=ce_b-ce_a if complete and ce_a is not None and ce_b is not None else None))
    summaries = []
    for model in (19901,19911,19921):
        for budget in ("attempts","time"):
            for mode in "RTE":
                values = [row for row in search if row["model_seed"] == model and row["budget"] == budget and row["mode"] == mode and row["main_complete"]]
                summaries.append(dict(model_seed=model,budget=budget,mode=mode,complete_repeats=len(values),
                    median_hv=statistics.median(row["normalized_hypervolume"] for row in values) if values else None,
                    min_hv=min((row["normalized_hypervolume"] for row in values),default=None),
                    max_hv=max((row["normalized_hypervolume"] for row in values),default=None),
                    median_accepted_hv=statistics.median(row["accepted_normalized_hypervolume"] for row in values) if values else None,
                    median_attempts=statistics.median(row["on_time"] for row in values) if values else None))
    baselines=[]
    for seed in (19901,19911,19921):
        source=load(run/"sources"/str(seed)/"manifest.json")
        if source:
            final=next((row for row in finalized if row["model_seed"] == seed),{})
            baselines.append(dict(model_seed=seed,validation_CE=source["baseline_validation"]["cross_entropy"],
                validation_accuracy=source["baseline_validation"]["accuracy"],
                test_CE=final.get("baseline_test",{}).get("cross_entropy"),
                test_accuracy=final.get("baseline_test",{}).get("accuracy"),
                training_steps=source["training"]["training_steps"],initial_state_sha256=source["initial_state_sha256"]))
    selected=[]
    for result in finalized:
        for row in result.get("selected",[]):
            selected.append(dict(row,model_seed=result["model_seed"],
                test_CE=row["test"]["cross_entropy"],test_accuracy=row["test"]["accuracy"],
                test_macro_F1=row["test"]["macro_f1"],storage_fraction=row["tensor_bytes"]/68904))
    updates=lines(run/"training-updates.jsonl")
    matrix_events=[row for path in (run/"jobs").glob("*/matrix-events.jsonl") for row in lines(path)]
    return dict(schema="issue99-local2-analysis-v1",code_revision=protocol["git_revision"],utc=protocol["utc"],
        resources=dict(ledger,training_updates_started=sum(row["event"] == "update_started" for row in updates),
            training_updates_completed=sum(row["event"] == "update_completed" for row in updates),
            mlp_ledger_count=len(lines(run/"mlp-attempts.jsonl")),matrix_ledger_count=len(lines(run/"matrix-attempts.jsonl")),
            observed_svd_starts=sum((row.get("svd_telemetry") or {}).get("started",0) for row in ledger["jobs"]),
            observed_svd_completions=sum((row.get("svd_telemetry") or {}).get("completed",0) for row in ledger["jobs"]),
            max_tree_rss_bytes=max((row.get("resources",{}).get("peak_tree_rss_bytes",0) for row in ledger["jobs"]),default=0)),
        baselines=baselines,search_runs=search,search_summary=summaries,curves=curves,oracles=oracles,
        paired=pairs,common_bytes=common_bytes,frontiers=frontiers,test_selected=selected,
        matrix_summary=summarize_matrix(matrix_streams),matrix_streams=matrix_streams,
        matrix_events_count=len(matrix_events),jobs=jobs,
        data_scope="1797 previously exposed Digits images; three overlapping new partitions; common initialization101")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    analysis=build_analysis(args.run)
    save(args.run/"analysis.json",analysis)
    save(args.output/"analysis.json",analysis)
    # Preserve original event bytes. Binary tensor artifacts stay local; their
    # SHA-256 witnesses remain in manifests and the artifact inventory.
    files=[]
    with zipfile.ZipFile(args.output/"raw-evidence.zip","w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in sorted(args.run.rglob("*")):
            if not path.is_file() or "tmp" in path.relative_to(args.run).parts:
                continue
            relative=path.relative_to(args.run).as_posix()
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            archived=path.suffix in (".json",".jsonl",".log",".bin") and relative != "analysis.json"
            files.append(dict(path=relative,bytes=path.stat().st_size,sha256=digest,archived=archived))
            if archived:
                archive.write(path,relative)
    save(args.output/"artifact-manifest.json",files)
    for name in ("protocol.json","resource-ledger.json","selected.json"):
        if (args.run/name).exists():
            shutil.copyfile(args.run/name,args.output/name)
    print(json.dumps(dict(status=analysis["resources"]["status"],searches=len(analysis["search_runs"]),
        matrix_cells=len(analysis["matrix_summary"]),mlp=analysis["resources"]["mlp_ledger_count"],
        matrix=analysis["resources"]["matrix_ledger_count"],files=len(files))))


if __name__ == "__main__":
    main()
