"""Serial, bounded issue-99 local package; the parent imports no ML runtime.

Run ``python -m fedcore.experiments.compression_search_package --output DIR
--historical-root DIR``. Tests and environment installation precede this run.
Every child has an absolute deadline, a resource watchdog and an append-only
attempt ledger. A failed admission or unknown execution error stops the package.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

MODULE = "fedcore.experiments.compression_search_package"
MAX_SECONDS = 7200
NO_NEW_AFTER_SECONDS = 6900
ATTEMPT_LIMIT = 3300
REPO = Path(__file__).resolve().parents[2]
MODULE_STARTED_AT = time.perf_counter()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_hashes():
    return {path.relative_to(REPO).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for path in sorted((REPO / "fedcore").rglob("*.py"))}


def count_attempts(output):
    path = Path(output) / "mlp-attempts.jsonl"
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0


def _run_job(output, job, protocol, package_start):
    import psutil
    if source_hashes() != protocol["source_files"]:
        raise RuntimeError("Source changed after protocol freeze")
    if time.perf_counter() - package_start >= NO_NEW_AFTER_SECONDS or count_attempts(output) >= ATTEMPT_LIMIT:
        return {"job": job["job"], "status": "package_budget", "resources": {}}
    directory = Path(output) / "jobs" / job["job"]
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    job.update(output=str(Path(output).resolve()), directory=str(directory.resolve()),
        historical_root=protocol["historical_root"], repo=str(REPO),
        protocol_sha256=protocol["protocol_sha256"], source_sha256=protocol["source_sha256"],
        worker_started_at=started, package_deadline=package_start + MAX_SECONDS,
        compute_admission_deadline=package_start + NO_NEW_AFTER_SECONDS)
    write_json(directory / "job.json", job)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2", HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", PYTHONUTF8="1",
        PYTHONHASHSEED="0", TEMP=str(Path(output).resolve() / "tmp"), TMP=str(Path(output).resolve() / "tmp"))
    peak, minimum_ram, samples, stop = 0, None, 0, None
    with (directory / "stdout.log").open("w", encoding="utf-8") as stdout, (directory / "stderr.log").open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen([sys.executable, "-m", MODULE, "--worker", str(directory / "job.json")],
            cwd=REPO, env=env, stdout=stdout, stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        root = psutil.Process(process.pid)
        while process.poll() is None:
            now = time.perf_counter()
            owned = [root]
            try:
                owned = [root] + root.children(recursive=True)
                rss = sum(item.memory_info().rss for item in owned if item.is_running())
            except psutil.NoSuchProcess:
                rss = 0
            except psutil.AccessDenied:
                rss, stop = 0, "resource_monitor_access_denied"
            available = psutil.virtual_memory().available
            peak, samples = max(peak, rss), samples + 1
            minimum_ram = available if minimum_ram is None else min(minimum_ram, available)
            if rss > 8 * 1024**3:
                stop = "rss_limit"
            elif available < 4 * 1024**3:
                stop = "ram_reserve"
            elif now >= package_start + MAX_SECONDS:
                stop = "package_time_limit"
            elif now - started >= job.get("hard_seconds", 300):
                stop = "worker_hard_deadline"
            if stop:
                for child in reversed(owned):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                process.wait(timeout=10)
                break
            time.sleep(.1)
    result_path = directory / "result.json"
    result = read_json(result_path) if result_path.exists() else None
    status = "succeeded" if process.returncode == 0 and result and result.get("status") == "succeeded" else "failed"
    record = {"job": job["job"], "kind": job["kind"], "status": status, "returncode": process.returncode,
        "stop": stop, "result_path": str(result_path), "resources": {"started_at": started,
            "completed_at": time.perf_counter(), "elapsed_seconds": time.perf_counter() - started,
            "peak_tree_rss_bytes": peak, "minimum_available_ram_bytes": minimum_ram, "samples": samples},
        "error": None if status == "succeeded" else (result or {}).get("error", "Worker ended without a successful result")}
    write_json(directory / "resources.json", record)
    return record


def package_jobs():
    """Static job order; no observed quality or speed changes this schedule."""
    jobs = [{"job": f"admission-{i}", "kind": "admission", "model_seed": 41, "hard_seconds": 150} for i in range(3)]
    jobs.append({"job": "admissibility", "kind": "admissibility", "hard_seconds": 300})
    for seed, branches in zip(range(9911, 9917), (("A", "B"), ("B", "A"), ("B", "A"), ("A", "B"), ("A", "B"), ("B", "A"))):
        for branch in branches:
            jobs.append({"job": f"early-late-{seed}-{branch}", "kind": "early_late", "stream_seed": seed, "branch": branch, "hard_seconds": 150})
    jobs.extend({"job": f"oracle-{seed}", "kind": "oracle", "model_seed": seed, "hard_seconds": 300} for seed in (41, 42, 43))
    for seed in (41, 42, 43):
        for budget in ("attempts", "time"):
            for search_seed, order in ((11, "RTE"), (22, "TER"), (33, "ERT")):
                for mode in order:
                    jobs.append({"job": f"search-{seed}-{budget}-{search_seed}-{mode}", "kind": "search",
                        "model_seed": seed, "search_seed": search_seed, "budget": budget, "mode": mode,
                        "hard_seconds": 75 if budget == "attempts" else 45})
    for seed in (41, 42, 43):
        jobs.append({"job": f"order-{seed}", "kind": "order", "model_seed": seed, "hard_seconds": 150})
        for stream_seed in (9921, 9922, 9923):
            index = (seed - 41) * 3 + stream_seed - 9921
            for branch in ("AB" if index % 2 == 0 else "BA"):
                jobs.append({"job": f"cache-{seed}-{stream_seed}-{branch}", "kind": "cache", "model_seed": seed,
                    "stream_seed": stream_seed, "branch": branch, "hard_seconds": 150})
    return jobs


def run_package(output, historical_root, *, resume_from=None):
    import psutil
    started = MODULE_STARTED_AT
    lineage = None
    if resume_from is not None:
        prior = Path(resume_from).resolve()
        previous_protocol, previous_ledger = read_json(prior / "protocol.json"), read_json(prior / "resource-ledger.json")
        # This narrowly permits fixing admission instrumentation before any
        # scientific series ran. It cannot replay selected experiments freely.
        if (previous_ledger.get("status") != "stopped" or not previous_ledger.get("jobs")
                or any(item.get("kind") != "admission" for item in previous_ledger["jobs"])):
            raise RuntimeError("Only a preserved admission-only stop may resume with a corrected loader")
        plan_digest = hashlib.sha256((REPO / "docs/svd_compression/PLAN_ISSUE_99_LOCAL_1.md").read_bytes()).hexdigest()
        if previous_protocol.get("plan_sha256") != plan_digest or previous_protocol.get("historical_root") != str(Path(historical_root).resolve()):
            raise RuntimeError("Resume must preserve the approved plan and exact historical source")
        started = previous_ledger["started_at"]
        if type(started) not in (int, float) or not 0 <= started <= MODULE_STARTED_AT:
            raise RuntimeError("Prior monotonic budget origin is not valid in this boot")
        lineage = {"directory": str(prior), "protocol_sha256": previous_protocol["protocol_sha256"],
            "ledger_sha256": hashlib.sha256((prior / "resource-ledger.json").read_bytes()).hexdigest(),
            "source_build": previous_protocol.get("git_revision"), "original_budget_origin": started,
            "reason": "Restore archived int64 label preprocessing; no source/split/rank/criterion change",
            "prior_attempts": count_attempts(prior), "new_free_budget": False}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "tmp").mkdir()
    if lineage is not None:
        (output / "mlp-attempts.jsonl").write_bytes((Path(lineage["directory"]) / "mlp-attempts.jsonl").read_bytes())
    sources = source_hashes()
    protocol = {"schema": 1, "study": "issue99-local1", "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
            capture_output=True, text=True).stdout.strip(),
        "resumed_admission": lineage,
        "plan_sha256": hashlib.sha256((REPO / "docs/svd_compression/PLAN_ISSUE_99_LOCAL_1.md").read_bytes()).hexdigest(),
        "historical_root": str(Path(historical_root).resolve()), "source_files": sources,
        "source_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "hardware": {"processor": platform.processor(), "logical_cpus": psutil.cpu_count(),
                "physical_cpus": psutil.cpu_count(logical=False), "total_ram_bytes": psutil.virtual_memory().total},
            "packages": {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()}},
        "resources": {"max_seconds": MAX_SECONDS, "no_new_after_seconds": NO_NEW_AFTER_SECONDS,
            "max_mlp_attempts": ATTEMPT_LIMIT, "workers": 1, "math_threads": 2, "interop_threads": 1,
            "max_tree_rss_bytes": 8 * 1024**3, "min_available_ram_bytes": 4 * 1024**3},
        "selection": {"byte_caps": [14824, 29649, 42003], "quality_rule": "CE<=M0+.05 and accuracy>=M0-.02",
            "fitness": "CE/ln(2), actual tensor bytes/49416", "hypervolume_reference": [2, 1.05]},
        "sampling": {"models": [41, 42, 43], "search_seeds": [11, 22, 33], "catalog_sizes": [15, 80],
            "searches": 54, "initial_prefix": "four common rank8 weighted/ASVD/AFM solutions"},
        "inference": "Historical engineering control; previously exposed validation/test; no fresh quality claim",
        "jobs": package_jobs()}
    protocol["protocol_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    write_json(output / "protocol.json", protocol)
    ledger = {"status": "running", "started_at": started, "protocol_sha256": protocol["protocol_sha256"], "jobs": []}
    write_json(output / "resource-ledger.json", ledger)
    try:
        for job in protocol["jobs"]:
            record = _run_job(output, dict(job), protocol, started)
            ledger["jobs"].append(record)
            ledger.update(elapsed_seconds=time.perf_counter() - started, mlp_attempts=count_attempts(output))
            write_json(output / "resource-ledger.json", ledger)
            print(json.dumps({"job": record["job"], "status": record["status"], "elapsed_seconds": round(ledger["elapsed_seconds"], 2),
                "attempts": ledger["mlp_attempts"]}), flush=True)
            if record["status"] != "succeeded":
                ledger["status"] = "stopped"
                ledger["stop_reason"] = record.get("error") or record["status"]
                break
        else:
            selections = freeze_selections(output)
            write_json(output / "selected.json", selections)
            for seed in (41, 42, 43):
                job = {"job": f"finalize-{seed}", "kind": "finalize", "model_seed": seed, "hard_seconds": 300,
                    "selections_sha256": hashlib.sha256((output / "selected.json").read_bytes()).hexdigest()}
                record = _run_job(output, job, protocol, started)
                ledger["jobs"].append(record)
                write_json(output / "resource-ledger.json", ledger)
                if record["status"] != "succeeded":
                    ledger.update(status="stopped", stop_reason=record.get("error") or record["status"])
                    break
            else:
                ledger["status"] = "succeeded"
    except BaseException as error:
        ledger.update(status="stopped", stop_reason=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        ledger.update(elapsed_seconds=time.perf_counter() - started, mlp_attempts=count_attempts(output), completed_at=time.perf_counter())
        write_json(output / "resource-ledger.json", ledger)
    return ledger


def completed_rows(output, job):
    path = Path(output) / "jobs" / job / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line] if path.exists() else []


def freeze_selections(output):
    selected = []
    for path in sorted((Path(output) / "jobs").glob("*/result.json")):
        result = read_json(path)
        if result.get("kind") not in ("oracle", "search"):
            continue
        rows = [row for row in result.get("evaluations", []) if row.get("status") == "finite"
            and row.get("quality_accepted") and row["completed_at"] <= result.get("deadline", float("inf"))]
        for cap in (14824, 29649, 42003):
            feasible = [row for row in rows if row["tensor_bytes"] <= cap]
            chosen = min(feasible, key=lambda row: (row["quality"], row["tensor_bytes"], row["solution_id"])) if feasible else None
            selected.append({"job": result["job"], "model_seed": result["model_seed"], "byte_cap": cap,
                "solution_id": None if chosen is None else chosen["solution_id"],
                "validation_quality": None if chosen is None else chosen["quality"]})
    return {"frozen_at": time.perf_counter(), "rule": "validation only; CE then bytes then canonical ID", "selected": selected}


def worker(job_path):
    # No heavy imports before the parent's clock has started.
    job = read_json(job_path)
    directory = Path(job["directory"])
    try:
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        import random
        import numpy as np
        def rng_signature():
            numpy_state = np.random.get_state()
            numpy_bytes = numpy_state[1].tobytes() + repr((numpy_state[0], *numpy_state[2:])).encode()
            return {"python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
                "numpy": hashlib.sha256(numpy_bytes).hexdigest(),
                "torch": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}
        rng_before = rng_signature()
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            result = execute_job(job)
        finally:
            # Includes first-import effects from optional libraries. Numerical
            # callbacks also own their narrower independently tested isolation.
            inner_changed = rng_signature() != rng_before
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
        rng_after = rng_signature()
        if rng_before != rng_after:
            raise RuntimeError("A package job changed caller-global RNG state")
        result["rng_verification"] = {"before": rng_before, "after": rng_after, "unchanged": True,
            "job_body_or_first_import_changed_state": inner_changed, "policy": "restore job including first imports; evaluator separately isolated"}
        result.update(status="succeeded", job=job["job"], kind=job["kind"], completed_at=time.perf_counter())
        write_json(directory / "result.json", result)
        return 0
    except BaseException as error:
        write_json(directory / "result.json", {"status": "failed", "kind": job["kind"], "job": job["job"],
            "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
        traceback.print_exc()
        return 1


def _source_evaluator(job, catalog):
    from .compression_search_evaluation import AttemptLedger, FullEvaluator, load_source, prepare_space
    from .protocol import stable_hash
    ledger = AttemptLedger(Path(job["output"]) / "mlp-attempts.jsonl", job["job"], job["protocol_sha256"],
        admission_deadline=job["compute_admission_deadline"])
    ledger.charge("source_baseline_validation", f"M0-{job['model_seed']}")
    source = load_source(job["model_seed"], job["historical_root"], job["repo"])
    space = prepare_space(source, catalog)
    context = stable_hash({"source": source.manifest["normalized_identity"],
        "calibration": source.calibration.manifest(), "validation": source.validation.manifest(),
        "metrics": "numpy-and-torch-stable-CE-v1;macroF1-v1", "quality_rule": "CE+.05,acc-.02",
        "source_sha256": job["source_sha256"], "environment": read_json(Path(job["output"]) / "protocol.json")["environment"],
        "rng_policy": "save/restore Python NumPy Torch global; local search RNG separately"})
    evaluator = FullEvaluator(source, space, ledger, Path(job["directory"]) / "events.jsonl",
        context_sha256=context, job_id=job["job"])
    write_json(Path(job["directory"]) / "source-manifest.json", source.manifest)
    return source, space, evaluator


def _toy():
    import torch
    from torch import nn
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9900)
        return nn.Sequential(nn.Linear(8, 8, dtype=torch.float64)).eval()


def execute_job(job):
    from .compression_search_protocol import (READY_CATALOG, TUNED_CATALOG, G5_CASES, G5_NEGATIVE_CASES,
        ATTEMPT_BUDGET, TIME_BUDGET, INITIAL_GENOTYPES, canonical_solution_id, random_search_stream,
        admissibility_stream, cache_replay_stream, catalog_by_id)
    from .compression_search_evaluation import (PackageFatal, compile_checked, append_event, predict, metrics,
        isolated_rng, historical_splits, roundtrip)
    kind = job["kind"]
    if kind in ("admissibility", "early_late"):
        return composition_job(job)
    source, space, evaluator = _source_evaluator(job, READY_CATALOG if job.get("mode") == "R" else TUNED_CATALOG)
    if kind == "admission":
        from fedcore.algorithm.low_rank.grammar_search import GrammarFitness, GrammarSearchConfig, run_grammar_search
        for genotype in (INITIAL_GENOTYPES[0], TUNED_CATALOG.genotypes[2], TUNED_CATALOG.genotypes[3]):
            evaluator(compile_checked(genotype, space))
        trace_path = Path(job["directory"]) / "search-trace.jsonl"
        search = run_grammar_search(space, GrammarSearchConfig(seed=9900, max_evaluations=4, max_seconds=45., max_proposals=32),
            evaluator, fitness_context=evaluator.context_sha256, initial_genotypes=INITIAL_GENOTYPES,
            absolute_deadline=min(job["package_deadline"], job["worker_started_at"] + 120),
            trace_callback=lambda event: append_event(trace_path, event))
        if search["status"] != "succeeded" or len(evaluator.rows) < 7:
            raise PackageFatal(f"Real optimizer admission failed: {search['status']}")
        toy_result = composition_job(dict(job, kind="admissibility", admission_only=True))
        return {"model_seed": source.seed, "evaluations": evaluator.rows,
            "engine": search["engine"], "composition": toy_result, "torch_threads": __import__("torch").get_num_threads()}
    if kind == "oracle":
        from fedcore.algorithm.low_rank.compression_grammar import genotype_payload
        evaluator.artifact_dir = Path(job["directory"]) / "artifacts"
        write_json(Path(job["directory"]) / "catalog.json", [{"solution_id": canonical_solution_id(value), "genotype": genotype_payload(value)} for value in TUNED_CATALOG.genotypes])
        for genotype in TUNED_CATALOG.genotypes:
            if time.perf_counter() >= job["package_deadline"]:
                raise PackageFatal("Package deadline before oracle completion")
            evaluator(compile_checked(genotype, space))
        return {"model_seed": source.seed, "evaluations": evaluator.rows}
    if kind == "search":
        from fedcore.algorithm.low_rank.grammar_search import GrammarSearchConfig, run_grammar_search
        budget = ATTEMPT_BUDGET if job["budget"] == "attempts" else TIME_BUDGET
        deadline = min(job["package_deadline"], job["worker_started_at"] + budget.max_seconds)
        trace_path = Path(job["directory"]) / "search-trace.jsonl"
        if job["mode"] == "E":
            search = run_grammar_search(space, GrammarSearchConfig(seed=job["search_seed"], population_size=4,
                generations=64, max_evaluations=budget.max_evaluations, max_seconds=budget.max_seconds,
                max_proposals=budget.max_proposals), evaluator, fitness_context=evaluator.context_sha256,
                initial_genotypes=INITIAL_GENOTYPES, absolute_deadline=deadline,
                trace_callback=lambda event: append_event(trace_path, event))
            if search["status"] not in ("succeeded", "no_valid_fitness"):
                raise PackageFatal(f"Optimizer shell failed: {search.get('reason')}")
            search = {key: value for key, value in search.items() if key not in ("best_genotypes", "finite_archive", "trace")}
        else:
            catalog = READY_CATALOG if job["mode"] == "R" else TUNED_CATALOG
            reason, proposals = "catalog_exhausted", 0
            for genotype in random_search_stream(catalog, job["search_seed"]):
                if len(evaluator.rows) >= budget.max_evaluations:
                    reason = "evaluation_budget"
                    break
                if time.perf_counter() >= deadline:
                    reason = "time_budget"
                    break
                if proposals >= budget.max_proposals:
                    reason = "proposal_budget"
                    break
                proposals += 1
                append_event(trace_path, {"event": "proposal", "proposal_id": proposals,
                    "canonical_solution_id": canonical_solution_id(genotype), "operator": "initial" if proposals <= 4 else "uniform_remaining",
                    "timestamp": time.perf_counter()})
                evaluator(compile_checked(genotype, space))
            search = {"engine": "frozen uniform remainder after common prefix", "stop_reason": reason,
                "charged_evaluations": len(evaluator.rows), "created_proposals": proposals, "cache_hits": 0}
        return {"model_seed": source.seed, "search_seed": job["search_seed"], "budget": job["budget"], "mode": job["mode"],
            "deadline": deadline, "search": search, "evaluations": evaluator.rows}
    if kind == "cache":
        memo, requests, stream_started = {}, [], time.perf_counter()
        evaluator.artifact_dir = Path(job["directory"]) / "artifacts"
        enabled = job["branch"] == "A"
        for index, genotype in enumerate(cache_replay_stream(job["stream_seed"])):
            workflow = compile_checked(genotype, space)
            key = (workflow.fingerprint, evaluator.context_sha256)
            cache_hit = enabled and key in memo
            request_started = time.perf_counter()
            if cache_hit:
                row = memo[key]
                if hashlib.sha256(Path(row["artifact_path"]).read_bytes()).hexdigest() != row["checkpoint_sha256"]:
                    raise PackageFatal("Cached artifact reference was changed or forged")
            else:
                evaluator(workflow)
                row = evaluator.rows[-1]
                if enabled and row["status"] == "finite":
                    memo[key] = dict(row)
            request = {"index": index, "solution_id": canonical_solution_id(genotype), "cache_hit": cache_hit,
                "status": row["status"], "quality": row.get("quality"), "tensor_bytes": row.get("tensor_bytes"),
                "started_at": request_started, "completed_at": time.perf_counter()}
            append_event(Path(job["directory"]) / "cache-requests.jsonl", request)
            requests.append(request)
        return {"model_seed": source.seed, "stream_seed": job["stream_seed"], "branch": job["branch"],
            "cache_enabled": enabled, "stream_seconds": time.perf_counter() - stream_started,
            "requests": requests, "evaluations": evaluator.rows}
    if kind == "order":
        return order_job(job, source, evaluator.ledger)
    if kind == "finalize":
        path = Path(job["output"]) / "selected.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != job["selections_sha256"]:
            raise PackageFatal("Selected IDs changed before test opening")
        selected = read_json(path)
        splits, _ = historical_splits(source.seed, job["repo"], include_test=True)
        test = splits["test"]
        evaluator.ledger.charge("final_baseline_test", f"M0-{source.seed}")
        baseline_test = metrics(predict(source.model, test.x), test.y)
        lookup, rows = catalog_by_id(), []
        evaluator.artifact_dir = Path(job["directory"]) / "artifacts"
        for choice in selected["selected"]:
            if choice["model_seed"] != source.seed or choice["solution_id"] is None:
                continue
            evaluator(compile_checked(lookup[choice["solution_id"]], space))
            if evaluator.rows[-1]["status"] != "finite":
                raise PackageFatal("Previously selected solution failed its final reconstruction")
            row = dict(choice, test=metrics(predict(evaluator.last_model, test.x), test.y),
                tensor_bytes=evaluator.rows[-1]["tensor_bytes"], artifact_path=evaluator.rows[-1]["artifact_path"],
                checkpoint_sha256=evaluator.rows[-1]["checkpoint_sha256"])
            if abs(evaluator.rows[-1]["quality"] - choice["validation_quality"]) > 1e-7:
                raise PackageFatal("Selected model validation changed during final rebuild")
            rows.append(row)
        return {"model_seed": source.seed, "selected": rows, "evaluations": evaluator.rows,
            "baseline_test": baseline_test, "selection_frozen_at": selected["frozen_at"], "historical_test": True}
    raise PackageFatal(f"Unknown package job {kind}")


def composition_job(job):
    import torch
    from fedcore.algorithm.low_rank.compression_composition import prepare_composition, execute_composition, CompositionSuccess
    from fedcore.algorithm.low_rank.plans import ValidationFailure
    from .compression_composition_reference import inspect_contract, execute_reference, assert_reference_parity
    from .compression_search_protocol import G5_CASES, G5_NEGATIVE_CASES, admissibility_stream
    from .compression_search_evaluation import PackageFatal, append_event, model_identity, isolated_rng
    model = _toy()
    original = model_identity(model)
    probe = (torch.arange(8, dtype=torch.float64) / 8).repeat(2, 1)
    write_json(Path(job["directory"]) / "toy-source.json", {"identity": original,
        "weight": model[0].weight.detach().tolist(), "bias": model[0].bias.detach().tolist(),
        "probe": probe.tolist(), "output": model(probe).detach().tolist(), "seed": 9900,
        "reference_step_probe": (torch.arange(16, dtype=torch.float64).reshape(2, 8) / 8).tolist(),
        "reference_parity_probe": (torch.arange(24, dtype=torch.float64).reshape(3, 8) / 8).tolist(),
        "dtype": "float64", "matrix_shape": [8, 8]})
    cases = (G5_CASES[1],) if job.get("admission_only") else G5_CASES + G5_NEGATIVE_CASES
    if job["kind"] == "early_late":
        cases = admissibility_stream(job["stream_seed"])
    rows, stream_started = [], time.perf_counter()
    for case in cases:
        if time.perf_counter() >= job["compute_admission_deadline"]:
            raise PackageFatal("No new toy computation after the package's 115-minute deadline")
        start = time.perf_counter()
        if job["kind"] == "admissibility":
            contract = inspect_contract(case.request, model)
            checked = prepare_composition(model, case.request)
            permitted = not isinstance(checked, ValidationFailure)
            actual = execute_reference(model, case.request)
            row = {"case_id": case.case_id, "word": case.word, "schema_negative": case.schema_negative,
                "contract_accepted": contract.accepted, "production_accepted": permitted,
                "execution_accepted": actual.accepted, "production_rejection": str(checked) if not permitted else None,
                "reference_rejection": None if actual.rejection is None else asdict(actual.rejection)}
            structural_mismatch = contract.accepted != permitted or (actual.rejection is not None
                and actual.rejection.category == "structural" and contract.accepted)
            nonstructural_failure = actual.rejection is not None and actual.rejection.category != "structural"
            if structural_mismatch or nonstructural_failure:
                append_event(Path(job["directory"]) / "composition-events.jsonl", row)
                category = "structural_admissibility_disagreement" if structural_mismatch else actual.rejection.category
                raise PackageFatal(f"{category} in {case.case_id}: {row}")
            if permitted:
                production = execute_composition(model, checked)
                if not isinstance(production, CompositionSuccess):
                    row["production_execution_failure"] = str(production)
                    append_event(Path(job["directory"]) / "composition-events.jsonl", row)
                    raise PackageFatal(f"Prepared valid chain failed execution: {case.case_id}: {production}")
                assert_reference_parity(actual, production.model, case.request)
                if production.storage_bytes != actual.storage_bytes:
                    raise PackageFatal("Reference and loaded production tensor storage differ")
                row.update(tensor_bytes=production.storage_bytes, checkpoint_bytes=len(production.checkpoint_bytes),
                    checkpoint_sha256=hashlib.sha256(production.checkpoint_bytes).hexdigest(),
                    final_states=[(path, asdict(state)) for path, state in actual.final_states])
        else:
            early = job["branch"] == "A"
            check_start = time.perf_counter()
            checked = prepare_composition(model, case.request) if early else None
            check_seconds = time.perf_counter() - check_start if early else 0.
            if early and isinstance(checked, ValidationFailure):
                row = {"case_id": case.case_id, "accepted": False, "completed_steps": 0,
                    "decompositions_started": 0, "check_seconds": check_seconds, "execution_seconds": 0.,
                    "production_rejection": str(checked), "output_identity": None,
                    "tensor_bytes": None, "final_states": [], "events": []}
            else:
                actual = execute_reference(model, case.request)
                row = {"case_id": case.case_id, "accepted": actual.accepted, "completed_steps": actual.completed_steps,
                    "decompositions_started": actual.decompositions_started, "check_seconds": check_seconds,
                    "execution_seconds": actual.timings.total_seconds, "stage_timings": asdict(actual.timings),
                    "output_identity": model_identity(actual.model) if actual.accepted else None,
                    "tensor_bytes": actual.storage_bytes,
                    "final_states": [(path, asdict(state)) for path, state in actual.final_states],
                    "events": [asdict(event) for event in actual.events],
                    "rejection": None if actual.rejection is None else asdict(actual.rejection)}
        row["elapsed_seconds"] = time.perf_counter() - start
        rows.append(row)
        append_event(Path(job["directory"]) / "composition-events.jsonl", row)
        if (row.get("rejection") or {}).get("category") in ("resource", "numerical", "shell"):
            raise PackageFatal(f"Nonstructural failure during paired stream: {row['rejection']}")
    if model_identity(model) != original:
        raise PackageFatal("Composition series mutated its source")
    return {"rows": rows, "stream_seconds": time.perf_counter() - stream_started,
        "stream_seed": job.get("stream_seed"), "branch": job.get("branch")}


def order_job(job, source, ledger):
    import torch
    from fedcore.algorithm.low_rank.compression_composition import (CompositionRecipe, CompositionRequest,
        PruneMask, LowRank, MaterializeDense, QuantizeRemapped, prepare_composition, execute_composition, CompositionSuccess)
    from .compression_composition_reference import execute_reference, assert_reference_parity
    from .compression_search_evaluation import PackageFatal, isolated_rng, predict, metrics, append_event, model_identity
    original, rows = model_identity(source.model), []
    weight = source.model[2].weight.detach()
    masks = {p: tuple(sorted(range(weight.numel()), key=lambda i: (float(weight.flatten()[i].abs()), i))[:int(weight.numel() * p)]) for p in (.25, .5)}
    d, q = MaterializeDense(), QuantizeRemapped("float32", "float32", "float32")
    cases = []
    for p, mask in masks.items():
        for rank in (8, 16):
            cases.extend([(f"mask-{p}-{rank}", "PLD", (PruneMask(mask), LowRank(rank), d), p, rank),
                (f"mask-{p}-{rank}", "LDP", (LowRank(rank), d, PruneMask(mask)), p, rank)])
    for rank in (8, 16):
        cases.extend([(f"packing-{rank}", "LDQD", (LowRank(rank), d, q, d), None, rank),
            (f"packing-{rank}", "QDLD", (q, d, LowRank(rank), d), None, rank)])
    for pair, word, steps, sparsity, rank in cases:
        attempt = ledger.charge("order", f"{pair}-{word}")
        started = time.perf_counter()
        request = CompositionRequest((CompositionRecipe("2", steps),), max_storage_bytes=49416, max_peak_bytes=8 * 1024**3)
        checked = prepare_composition(source.model, request)
        production = execute_composition(source.model, checked)
        if not isinstance(production, CompositionSuccess):
            raise PackageFatal(f"Order comparison refused: {pair}-{word}: {production}")
        reference = execute_reference(source.model, request)
        assert_reference_parity(reference, production.model, request, rtol=2e-4, atol=2e-5)
        if production.storage_bytes != 49416:
            raise PackageFatal("Order comparison did not end with matched dense bytes")
        layer = production.model[2]
        spectrum = torch.linalg.svdvals(layer.weight.detach().double())
        row = {"pair": pair, "word": word, "model_seed": source.seed, "mask_fraction": sparsity, "rank": rank,
            "global_attempt": attempt, "started_at": started, "completed_at": time.perf_counter(),
            "tensor_bytes": production.storage_bytes, "metrics": metrics(predict(production.model, source.validation.x), source.validation.y),
            "zero_fraction": float((layer.weight == 0).double().mean()), "singular_values": spectrum.tolist(),
            "checkpoint_bytes": len(production.checkpoint_bytes), "elapsed_seconds": time.perf_counter() - started}
        rows.append(row)
        append_event(Path(job["directory"]) / "order-events.jsonl", row)
    if model_identity(source.model) != original:
        raise PackageFatal("Order series mutated M0")
    source.calibration.verify_integrity()
    source.validation.verify_integrity()
    return {"model_seed": source.seed, "rows": rows, "masks_sha256": {str(p): hashlib.sha256(json.dumps(mask).encode()).hexdigest() for p, mask in masks.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--historical-root", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    if not args.output or not args.historical_root:
        parser.error("--output and --historical-root are required")
    ledger = run_package(args.output, args.historical_root, resume_from=args.resume_from)
    print(json.dumps({key: ledger.get(key) for key in ("status", "stop_reason", "elapsed_seconds", "mlp_attempts")}), flush=True)
    return 0 if ledger["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
