"""Fresh-process Digits jobs. Importing this entry module uses only stdlib.

The supervising parent starts the common clock before importing the numerical
implementation. Every job owns its model, data views, statistics and caches.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_dir(job):
    return Path(job["output"]) / "sources" / str(job["split_seed"])


def _prepare(job):
    import numpy as np
    import torch
    from sklearn.datasets import load_digits
    from .compression_search_digits_protocol import ROLE_SIZES, SPLIT_SEEDS, INITIALIZATION_SEED, protocol_metadata
    from .compression_search_digits_evaluation import check_deadline, fresh_model
    from .compression_search_evaluation import PackageFatal, model_identity
    from .protocol import TensorSplit, stable_hash, tensor_hash
    from .scenarios import split_indices
    check_deadline(job)
    data = load_digits()
    x = data.images.reshape(len(data.images), 64).astype(np.float32) / np.float32(16.)
    y = data.target.astype(np.int64)
    digest = lambda value: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    metadata = {"dataset": "sklearn.load_digits", "samples": 1797, "classes": list(range(10)),
        "raw_features_sha256": digest(data.data), "raw_labels_sha256": digest(data.target),
        "transformed_features_sha256": digest(x), "transformed_labels_sha256": digest(y),
        "preprocessing": "fixed row-major flatten to 64 float32 features divided by 16; no fitted statistics",
        "evidence_scope": "same previously exposed Digits source; new roles are not external independent evidence"}
    if x.shape != (1797, 64) or y.shape != (1797,):
        raise PackageFatal("Unexpected sklearn Digits source dimensions")
    historical = {}
    for seed in (41, 42, 43):
        path = Path(job["repo"]) / f"docs/svd_compression/results/local-2026-10-13/runs/cnn-seed-{seed}.json"
        old = read_json(path)
        old_data = old["data"]
        if (old_data["metadata"]["data_sha256"] != metadata["raw_features_sha256"]
                or old_data["metadata"]["target_sha256"] != digest(y)):
            raise PackageFatal("Historical DigitsCNN and current raw source disagree")
        historical[seed] = {"path": str(path), "sha256": _sha(path), "roles": old_data["roles"]}
    initial = fresh_model(job)
    identity = model_identity(initial)
    passports = []
    for seed in SPLIT_SEEDS:
        check_deadline(job)
        directory = Path(job["output"]) / "sources" / str(seed)
        directory.mkdir(parents=True, exist_ok=False)
        indices = split_indices(y, seed)
        ids = {role: tuple(f"sklearn-digits:{int(i)}" for i in rows) for role, rows in indices.items()}
        flattened = [value for values in ids.values() for value in values]
        if (len(set(flattened)) != 1797 or set(flattened) != {f"sklearn-digits:{i}" for i in range(1797)}
                or {role: len(values) for role, values in ids.items()} != ROLE_SIZES):
            raise PackageFatal("Digits roles overlap or do not exactly cover the source")
        intersections = {}
        for old_seed, old in historical.items():
            old_ids = {role: set(record["ids"]) for role, record in old["roles"].items()}
            if all(set(ids[role]) == old_ids[role] for role in ids):
                raise PackageFatal("New Digits role assignment equals a historical assignment")
            intersections[str(old_seed)] = {role: {other: len(set(values) & old_values)
                for other, old_values in old_ids.items()} for role, values in ids.items()}
        roles = {}
        for role, rows in indices.items():
            split = TensorSplit(torch.from_numpy(x[rows]), torch.from_numpy(y[rows]), ids[role])
            path = directory / f"role-{role}.pt"
            torch.save({"x": split.x, "y": split.y}, path)
            roles[role] = {"file": path.name, "file_sha256": _sha(path), "manifest": split.manifest(),
                "class_counts": {str(category): int((y[rows] == category).sum()) for category in range(10)}}
        torch.save(initial.state_dict(), directory / "initialization.pt")
        passport = {"split_seed": seed, "stratification_seeds": [seed, seed+1, seed+2],
            "roles": roles, "metadata": metadata, "historical_intersections": intersections,
            "historical_manifests": {str(value): {key: record[key] for key in ("path", "sha256")} for value, record in historical.items()},
            "initialization_seed": INITIALIZATION_SEED, "initial_identity": identity,
            "initial_state_sha256": stable_hash({name: tensor_hash(value) for name, value in initial.state_dict().items()}),
            "initial_checkpoint_sha256": _sha(directory / "initialization.pt")}
        write_json(directory / "roles.json", passport)
        passports.append({"split_seed": seed, "roles_sha256": _sha(directory / "roles.json"),
            "initial_identity": identity, "initial_state_sha256": passport["initial_state_sha256"]})
    write_json(Path(job["output"]) / "sources" / "dataset-manifest.json", {**metadata, "splits": passports})
    write_json(Path(job["directory"]) / "digits-protocol.json", protocol_metadata())
    return {"prepared": passports, "dataset": metadata, "dataset_manifest_sha256": _sha(Path(job["output"]) / "sources" / "dataset-manifest.json")}


def _train(job):
    import torch
    from torch import nn
    from .compression_search_digits_evaluation import (
        DigitsLedger, check_deadline, count_forward_batches, fresh_model, load_role)
    from .compression_search_digits_protocol import UPDATES_PER_MODEL, UPDATE_LIMIT, DENSE_TENSOR_BYTES
    from .compression_search_evaluation import PackageFatal, append_event, model_identity, metrics, predict
    from .protocol import stable_hash, tensor_hash
    from .runner import train_model
    directory = _source_dir(job)
    passport = read_json(directory / "roles.json")
    if _sha(directory / "initialization.pt") != passport["initial_checkpoint_sha256"]:
        raise PackageFatal("Initial source checkpoint changed before training")
    check_deadline(job)
    model = fresh_model(job)
    model.load_state_dict(torch.load(directory / "initialization.pt", map_location="cpu", weights_only=True), strict=True)
    if model_identity(model) != passport["initial_identity"]:
        raise PackageFatal("M0 does not start from the shared initialization")
    # Only train is in scope during the fixed training call.
    train = load_role(directory, "train")
    updates_path = Path(job["output"]) / "training-updates.jsonl"
    old = [json.loads(line) for line in updates_path.read_text(encoding="utf-8").splitlines() if line] if updates_path.exists() else []
    charged = sum(row["event"] == "update_started" for row in old)
    if charged + UPDATES_PER_MODEL > UPDATE_LIMIT or any(row.get("split_seed") == job["split_seed"] for row in old):
        raise PackageFatal("Fixed training budget would repeat a model or exceed 2040 updates")
    optimizer_parameters = {}
    def validate_optimizer(optimizer):
        group = optimizer.param_groups[0]
        expected = {"lr": .001, "weight_decay": 0, "betas": (.9, .999), "eps": 1e-8, "amsgrad": False}
        if any(group[key] != value for key, value in expected.items()):
            raise PackageFatal("Actual Adam configuration differs from the frozen training protocol")
        for key, value in group.items():
            if key != "params":
                optimizer_parameters[key] = value
    class TrainingObjective:
        evidence = {"objective": "cross_entropy", "training_role": "train", "shuffle_seed": 101}
        used = 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def __call__(self, current, x, y):
            check_deadline(job)
            return nn.functional.cross_entropy(current(x), y.long())
        def before_step(self, current, optimizer):
            check_deadline(job)
            if self.used >= UPDATES_PER_MODEL:
                raise PackageFatal("Training exceeded its fixed 680 updates")
            self.used += 1
            append_event(updates_path, {"event": "update_started", "update": charged+self.used,
                "model_update": self.used, "job": job["job"], "split_seed": job["split_seed"],
                "started_at": time.perf_counter(), "protocol_sha256": job["protocol_sha256"]})
        def after_step(self, current, optimizer):
            append_event(updates_path, {"event": "update_completed", "update": charged+self.used,
                "model_update": self.used, "job": job["job"], "split_seed": job["split_seed"],
                "completed_at": time.perf_counter(), "protocol_sha256": job["protocol_sha256"]})
    objective = TrainingObjective()
    with count_forward_batches(Path(job["directory"]) / "forward-events.jsonl", job_id=job["job"],
            purpose="fixed_training", solution_id=f"M0-{job['split_seed']}"):
        training = train_model(model, train, "classification", epochs=40, batch_size=64,
            learning_rate=.001, device="cpu", seed=101, optimizer_validator=validate_optimizer,
            batch_objective=objective, max_updates=680, drop_last=False)
    train.verify_integrity()
    if (training["training_steps"] != 680 or objective.used != 680
            or not all(math.isfinite(value) for value in training["train_loss"])
            or not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values())):
        raise PackageFatal("Digits training did not complete exactly 680 finite updates")
    path = directory / "baseline.pt"
    torch.save(model.state_dict(), path)
    ledger = DigitsLedger(Path(job["output"]) / "mlp-attempts.jsonl", job)
    attempt = ledger.charge("post_training_baseline_validation", f"M0-{job['split_seed']}")
    validation_started = time.perf_counter()
    validation = load_role(directory, "validation")
    restored = fresh_model(job)
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    with count_forward_batches(Path(job["directory"]) / "forward-events.jsonl", job_id=job["job"],
            purpose="post_training_baseline_validation", solution_id=f"M0-{job['split_seed']}"):
        before, after = predict(model, validation.x), predict(restored, validation.x)
    if not torch.equal(before, after) or model_identity(model) != model_identity(restored):
        raise PackageFatal("Strict M0 reload changed output or model identity")
    measured = metrics(after, validation.y)
    append_event(Path(job["directory"]) / "baseline-events.jsonl", {"event": "baseline_validation_completed",
        "job": job["job"], "model_seed": job["split_seed"], "global_attempt": attempt,
        "started_at": validation_started, "completed_at": time.perf_counter(), "metrics": measured})
    manifest = {"split_seed": job["split_seed"], "seed": job["split_seed"],
        "source_path": str(path), "checkpoint_sha256": _sha(path), "checkpoint_file_bytes": path.stat().st_size,
        "roles_sha256": _sha(directory / "roles.json"), "tensor_bytes": DENSE_TENSOR_BYTES,
        "normalized_identity": model_identity(restored), "normalized_constructor": "exact nn.Sequential 64-128-64-10 ReLU",
        "initial_identity": passport["initial_identity"], "initial_state_sha256": passport["initial_state_sha256"],
        "final_state_sha256": stable_hash({name: tensor_hash(value) for name, value in restored.state_dict().items()}),
        "initialization_seed": 101, "train_shuffle_seed": 101, "training": training,
        "optimizer_parameters": optimizer_parameters, "baseline_validation": measured,
        "strict_reload_verified": True, "validation_gate_passed": measured["accuracy"] >= .9,
        "protocol_sha256": job["protocol_sha256"], "source_sha256": job["source_sha256"]}
    write_json(directory / "manifest.json", manifest)
    return {"split_seed": job["split_seed"], "model_seed": job["split_seed"],
        "baseline_validation": measured, "training_steps": 680, "initial_state_sha256": passport["initial_state_sha256"],
        "validation_gate_passed": measured["accuracy"] >= .9, "source_manifest_sha256": _sha(directory / "manifest.json")}


def _evaluate_job(job):
    from .compression_search_digits_protocol import (
        ADMISSION_GENOTYPES, ATTEMPT_BUDGET, INITIAL_GENOTYPES, READY_CATALOG,
        TIME_BUDGET, TUNED_CATALOG, canonical_solution_id, random_search_stream)
    from .compression_search_digits_evaluation import (
        DigitsDeadline, check_deadline, effective_deadline, source_evaluator)
    from .compression_search_evaluation import PackageFatal, append_event, compile_checked
    from fedcore.algorithm.low_rank.compression_grammar import genotype_payload
    from fedcore.algorithm.low_rank.grammar_search import GrammarSearchConfig, run_grammar_search
    kind = job["kind"]
    catalog = READY_CATALOG if job.get("mode") == "R" else TUNED_CATALOG
    deadline = effective_deadline(job)
    if kind == "search":
        budget = ATTEMPT_BUDGET if job["budget"] == "attempts" else TIME_BUDGET
        deadline = min(deadline, job["worker_started_at"] + budget.max_seconds)
    bounded = dict(job, admission_deadline=deadline)
    source, space, evaluator = source_evaluator(bounded, catalog)
    write_json(Path(job["directory"]) / "source-manifest.json", source.manifest)
    write_json(Path(job["directory"]) / "catalog.json", [{"solution_id": canonical_solution_id(value),
        "genotype": genotype_payload(value)} for value in catalog.genotypes])
    trace_path = Path(job["directory"]) / "search-trace.jsonl"
    if kind == "finalize":
        return _finalize(bounded, source, space, evaluator)
    search, reason = None, "catalog_exhausted"
    try:
        if kind == "admission":
            for genotype in ADMISSION_GENOTYPES:
                check_deadline(bounded)
                evaluator(compile_checked(genotype, space))
                if evaluator.rows[-1]["status"] != "finite":
                    raise PackageFatal("A direct admission profile failed numerically")
            config = GrammarSearchConfig(seed=9900, population_size=4, generations=64,
                mutation_only=False, max_evaluations=12, max_seconds=max(.001, deadline-time.perf_counter()), max_proposals=48)
            search = run_grammar_search(space, config, evaluator, fitness_context=evaluator.context_sha256,
                initial_genotypes=INITIAL_GENOTYPES, absolute_deadline=deadline,
                trace_callback=lambda event: append_event(trace_path, event))
            populations = [event for event in search["trace"] if event["event"] == "population"]
            if (search["status"] != "succeeded" or not 4 <= search["charged_evaluations"] <= 12
                    or not any(event["generation"] >= 2 and event["size"] > 0 for event in populations)
                    or any(event["size"] == 0 for event in populations)):
                raise PackageFatal(f"Real Digits optimizer admission failed: {search.get('reason')}")
            reason = search["stop_reason"]
        elif kind == "oracle":
            evaluator.artifact_dir = Path(job["directory"]) / "artifacts"
            for genotype in TUNED_CATALOG.genotypes:
                check_deadline(bounded)
                evaluator(compile_checked(genotype, space))
        elif kind == "search" and job["mode"] == "E":
            config = GrammarSearchConfig(seed=job["search_seed"], population_size=4, generations=64,
                mutation_only=False, max_evaluations=budget.max_evaluations, max_seconds=budget.max_seconds,
                max_proposals=budget.max_proposals)
            search = run_grammar_search(space, config, evaluator, fitness_context=evaluator.context_sha256,
                initial_genotypes=INITIAL_GENOTYPES, absolute_deadline=deadline,
                trace_callback=lambda event: append_event(trace_path, event))
            if search["status"] not in ("succeeded", "no_valid_fitness"):
                raise PackageFatal(f"Digits optimizer shell failed: {search.get('reason')}")
            reason = search["stop_reason"]
        elif kind == "search":
            proposals = 0
            for genotype in random_search_stream(catalog, job["search_seed"]):
                if len(evaluator.rows) >= min(budget.max_evaluations, len(catalog.genotypes)):
                    reason = "catalog_exhausted" if len(evaluator.rows) == len(catalog.genotypes) else "evaluation_budget"
                    break
                check_deadline(bounded)
                proposals += 1
                append_event(trace_path, {"event": "proposal", "proposal_id": proposals,
                    "canonical_solution_id": canonical_solution_id(genotype), "timestamp": time.perf_counter(),
                    "operator": "initial" if proposals <= 4 else "uniform_remaining"})
                evaluator(compile_checked(genotype, space))
            search = {"engine": "frozen uniform remainder after common prefix", "stop_reason": reason,
                "charged_evaluations": len(evaluator.rows), "created_proposals": proposals, "cache_hits": 0}
        else:
            raise PackageFatal(f"Unknown Digits evaluation job: {kind}")
    except DigitsDeadline:
        reason = "time_budget"
    if search is not None:
        search = {key: value for key, value in search.items() if key not in ("best_genotypes", "finite_archive", "trace")}
        search["stop_reason"] = reason
    return {"model_seed": source.seed, "split_seed": source.seed, "search_seed": job.get("search_seed"),
        "budget": job.get("budget"), "mode": job.get("mode"), "deadline": deadline,
        "search": search, "stop_reason": reason, "evaluations": evaluator.rows,
        "complete_catalog": kind == "oracle" and len(evaluator.rows) == 440,
        "source_manifest_sha256": _sha(_source_dir(job) / "manifest.json"),
        "evaluation_context_sha256": evaluator.context_sha256}


def _finalize(job, source, space, evaluator):
    from .compression_search_digits_protocol import catalog_by_id
    from .compression_search_digits_evaluation import DigitsDeadline, count_forward_batches, load_role
    from .compression_search_evaluation import PackageFatal, append_event, compile_checked, metrics, predict
    path = Path(job["output"]) / "selected.json"
    expected = job.get("selections_sha256", job.get("selected_sha256"))
    if not expected or _sha(path) != expected:
        raise PackageFatal("Selected IDs changed before test opening")
    selected = read_json(path)
    if len(selected["selected"]) > 171 or selected["frozen_at"] > time.perf_counter():
        raise PackageFatal("Invalid immutable selection receipt")
    choices = [row for row in selected["selected"] if row["model_seed"] == source.seed and row["solution_id"] is not None]
    lookup = catalog_by_id()
    if any(row["solution_id"] not in lookup for row in choices):
        raise PackageFatal("Selected ID is outside the frozen catalog")
    attempt = evaluator.ledger.charge("final_baseline_test", f"M0-{source.seed}")
    baseline_started = time.perf_counter()
    test = load_role(_source_dir(job), "test")
    with count_forward_batches(Path(job["directory"]) / "forward-events.jsonl", job_id=job["job"],
            purpose="final_baseline_test", solution_id=f"M0-{source.seed}"):
        baseline_test = metrics(predict(source.model, test.x), test.y)
    append_event(Path(job["directory"]) / "baseline-events.jsonl", {"event": "baseline_test_completed",
        "job": job["job"], "model_seed": source.seed, "global_attempt": attempt,
        "started_at": baseline_started, "completed_at": time.perf_counter(), "metrics": baseline_test,
        "selections_sha256": expected})
    evaluator.artifact_dir = Path(job["directory"]) / "artifacts"
    rows, memo, partial = [], {}, False
    for choice in choices:
        if _sha(path) != expected:
            raise PackageFatal("Immutable selections changed during test evaluation")
        identity = choice["solution_id"]
        if identity not in memo:
            try:
                evaluator(compile_checked(lookup[identity], space))
            except DigitsDeadline:
                partial = True
                break
            row = evaluator.rows[-1]
            if row["status"] != "finite" or abs(row["quality"] - choice["validation_quality"]) > 1e-7:
                raise PackageFatal("Selected solution failed its final validation reconstruction")
            if row["tensor_bytes"] != choice["tensor_bytes"]:
                raise PackageFatal("Selected solution storage changed in final reconstruction")
            with count_forward_batches(Path(job["directory"]) / "forward-events.jsonl", job_id=job["job"],
                    purpose="selected_test", solution_id=identity):
                measured = metrics(predict(evaluator.last_model, test.x), test.y)
            completion = {"event": "selected_action_completed", "job": job["job"],
                "solution_id": identity, "global_attempt": row["global_attempt"],
                "started_at": row["started_at"], "completed_at": time.perf_counter(), "test": measured,
                "selections_sha256": expected}
            append_event(Path(job["directory"]) / "selected-actions.jsonl", completion)
            memo[identity] = {"test": measured, "tensor_bytes": row["tensor_bytes"],
                "artifact_path": row["artifact_path"], "checkpoint_sha256": row["checkpoint_sha256"],
                "global_attempt": row["global_attempt"], "validation_quality": row["quality"],
                "full_action_completed_at": completion["completed_at"]}
        verified = memo[identity]
        if abs(verified["validation_quality"] - choice["validation_quality"]) > 1e-7 or verified["tensor_bytes"] != choice["tensor_bytes"]:
            raise PackageFatal("Two logical choices disagree on their shared reconstruction")
        rows.append({**choice, **verified})
    test.verify_integrity()
    if _sha(path) != expected:
        raise PackageFatal("Immutable selections changed after test evaluation")
    return {"model_seed": source.seed, "split_seed": source.seed, "selected": rows,
        "evaluations": evaluator.rows, "baseline_test": baseline_test,
        "selection_frozen_at": selected["frozen_at"], "selections_sha256": expected,
        "logical_selections": len(rows), "physical_reconstructions": len(memo),
        "independent_external_test": False, "partial": partial,
        "stop_reason": "time_budget" if partial else "selected_complete"}


def run_job(job):
    """Execute one job; the fresh-process parent owns Torch backend settings."""
    from .compression_search_digits_evaluation import DigitsDeadline, check_deadline, rng_signature
    from .compression_search_evaluation import isolated_rng, PackageFatal
    if job["kind"] != "prepare" and job["split_seed"] not in (19901, 19911, 19921):
        raise PackageFatal("Invalid frozen Digits split seed")
    if job.get("model_seed", job.get("split_seed")) != job.get("split_seed"):
        raise PackageFatal("Digits model seed denotes its split; initialization is separately fixed at 101")
    before = rng_signature()
    try:
        with isolated_rng():
            check_deadline(job)
            if job["kind"] == "prepare":
                result = _prepare(job)
            elif job["kind"] == "train":
                result = _train(job)
            else:
                result = _evaluate_job(job)
    except DigitsDeadline as error:
        result = {"partial": True, "stop_reason": "time_budget", "reason": str(error),
            "model_seed": job.get("split_seed"), "split_seed": job.get("split_seed"),
            "evaluations": [], "deadline": job.get("admission_deadline")}
    if rng_signature() != before:
        raise PackageFatal("Digits job changed caller-global RNG state")
    result["rng_verification"] = {"before": before, "after": rng_signature(), "unchanged": True}
    return result
