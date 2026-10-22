"""Frozen local SVD comparison using FedCore's existing transforms and exports.

This is a small trained-model comparison, not a reproduction of article scores.
Run ``python -m fedcore.experiments.svd_comparison --output PATH``. Training,
compression and artifact timing are sequential child processes. All candidates
are frozen before the first baseline, and all outcomes (including failures and
baseline gate failures) are retained. Timing repetitions are not training seeds.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

import torch
from torch import nn

from fedcore.tools.atomic_json import write_atomic_json
from . import measurement, runner, scenarios
from .protocol import CandidateSpec, ExperimentProtocol, ProtocolError, json_value, stable_hash

SEEDS = (41, 42, 43)
METHODS = ("svd", "weighted_svd", "asvd", "afm")
RSS_LIMIT = 12 * 1024**3
OS_RESERVE = 4 * 1024**3
COMPUTE_SECONDS = 8 * 60 * 60
THREADS = 4
TIMING_THREADS = 1
BATCH = 64
REPEATS = 200
WARMUP = 20
MODULE = "fedcore.experiments.svd_comparison"


def build_bundle(architecture, seed):
    if architecture == "mlp":
        return scenarios.build_tabular(seed)
    if architecture == "cnn":
        return scenarios.build_cv_digits(seed)
    raise ProtocolError("Unknown comparison architecture")


def exact_storage(model):
    """Unique full Parameter/buffer storage; no model metadata or process RSS."""
    groups = {}
    for kind, tensors in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, value in tensors:
            storage = value.untyped_storage()
            key = str(value.device), storage._cdata
            entry = groups.setdefault(key, {"bytes": storage.nbytes(), "elements": storage.nbytes() // value.element_size(),
                                            "kind": kind, "names": []})
            entry["names"].append(name)
    return {"tensor_bytes": sum(item["bytes"] for item in groups.values()),
            "stored_elements": sum(item["elements"] for item in groups.values()),
            "parameter_bytes": sum(item["bytes"] for item in groups.values() if item["kind"] == "parameter"),
            "buffer_bytes": sum(item["bytes"] for item in groups.values() if item["kind"] == "buffer"),
            "unique_storages": list(groups.values()),
            "scope": "complete unique parameter and buffer tensor storage; metadata/file/RSS separate"}


def expected_factor_storage(model, ranks):
    """Independent closed-form cost for these untied biased channel operators."""
    total = exact_storage(model)["tensor_bytes"]
    target = {}
    for path, rank in ranks.items():
        layer = model.get_submodule(path)
        if type(layer) not in (nn.Linear, nn.Conv2d) or getattr(layer, "groups", 1) != 1:
            raise ProtocolError("Budget arithmetic supports untied Linear and groups=1 channel Conv2d")
        m, n = layer.weight.shape[0], math.prod(layer.weight.shape[1:])
        if type(rank) is not int or not 1 <= rank <= min(m, n):
            raise ProtocolError("Budget rank exceeds actual operator dimensions")
        bias = layer.bias.numel() if layer.bias is not None else 0
        # AFM would add an intercept to bias-free layers. Exclude that unequal
        # representation from this particular equal-rank/equal-budget design.
        if layer.bias is None:
            raise ProtocolError("This comparison requires existing biases for equal AFM storage")
        old = (m * n + bias) * layer.weight.element_size()
        new = (rank * (m + n) + bias) * layer.weight.element_size()
        target[path] = {"matrix_shape": [m, n], "rank": rank, "dense_bytes": old, "factor_bytes": new,
                        "bias_elements": bias, "dtype": str(layer.weight.dtype)}
        total += new - old
    return {"whole_model_tensor_byte_budget": total, "targets": target,
            "untouched_tensor_bytes": exact_storage(model)["tensor_bytes"] - sum(v["dense_bytes"] for v in target.values())}


def frozen_cases(architecture):
    if architecture == "mlp":
        levels = ({"2": rank} for rank in (8, 16, 24))
    elif architecture == "cnn":
        levels = ({"classifier.0": rank} for rank in (8, 16, 32))
    else:
        raise ProtocolError("Unknown comparison architecture")
    result = [{"cohort": "common_linear", "level": level, "method": method, "ranks": ranks}
              for level, ranks in enumerate(levels) for method in METHODS]
    if architecture == "cnn":
        for level, vector in enumerate(((1, 4, 8), (2, 8, 16), (3, 12, 32))):
            ranks = dict(zip(("features.0", "features.2", "classifier.0"), vector))
            result.extend({"cohort": "joint_channel_conv_linear", "level": level, "method": method,
                           "ranks": ranks} for method in ("svd", "weighted_svd"))
    return tuple(dict(item, case_id=stable_hash(item)[:16]) for item in result)


def candidate_spec(case):
    ranks = case["ranks"]
    common = {"target_paths": list(ranks), "finetune_epochs": 0}
    if case["method"] == "svd":
        return CandidateSpec("svd", dict(common, ranks=ranks))
    if case["method"] == "weighted_svd":
        # The existing weighted interpreter collects all selected statistics on
        # its input graph. Explicit one-target chain steps instead establish the
        # declared current-graph order for this joint convolution cohort.
        steps = tuple(CandidateSpec("weighted_svd", {"target_paths": [path], "rank": rank,
            "ridge": 0., "rcond": None, "nullspace_policy": "support_only", "finetune_epochs": 0,
            "max_workspace_bytes": 256 * 1024**2, "max_peak_bytes": RSS_LIMIT}) for path, rank in ranks.items())
        return steps[0] if len(steps) == 1 else CandidateSpec("chain", chain=steps)
    options = {"abs_mode": "abs_mean", "alpha": .5, "epsilon": 0., "zero_policy": "support_only"}
    return CandidateSpec(case["method"], dict(common, ranks=ranks,
        **({"method_options": options} if case["method"] == "asvd" else {})))


def source_identity():
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip()
    sources = list((root / "fedcore").rglob("*.py")) + [root / "tests/refactoring/test_svd_comparison.py"]
    hashes = {str(path.relative_to(root)).replace("\\", "/"):
              hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
              for path in sorted(sources) if path.is_file()}
    return {"parent_head": revision, "git_lf_source_hashes": hashes,
            "git_lf_source_sha256": stable_hash(hashes), "normalization": "CRLF to LF before SHA256"}


def _prior_compute_ledger(prior_runs):
    entries = []
    for prior in prior_runs:
        prior = Path(prior).resolve()
        files, seconds = {}, 0.
        for filename, key in (("execution.json", "architecture_workers"), ("timing-execution.json", "workers")):
            path = prior / filename
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                seconds += sum(item["resources"]["elapsed_seconds"] for item in payload[key])
                files[filename] = measurement.file_hash(path)
        entries.append({"directory": str(prior), "charged_worker_wall_seconds": seconds, "evidence_sha256": files,
                        "status": "retained prior attempt; instrumentation superseded, no test-driven model selection"})
    return entries


def predeclare_plan(output, *, prior_runs=()):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "protocol.json"
    if path.exists():
        raise ProtocolError("Prospective protocol exists; use a new output directory")
    architectures = {}
    for name in ("mlp", "cnn"):
        example = build_bundle(name, SEEDS[0])
        cases = tuple(dict(case, candidate=candidate_spec(case).to_dict(),
                           budget=expected_factor_storage(example.original_model, case["ranks"]))
                      for case in frozen_cases(name))
        architectures[name] = {"cases": cases, "baseline_epochs": 80 if name == "mlp" else 30,
            "minimum_validation_accuracy": .90 if name == "mlp" else .93,
            "model": str(example.original_model), "dense_storage": exact_storage(example.original_model)}
    prior_ledger = _prior_compute_ledger(prior_runs)
    prior_seconds = sum(entry["charged_worker_wall_seconds"] for entry in prior_ledger)
    if prior_seconds >= COMPUTE_SECONDS:
        raise ProtocolError("Prior attempts exhausted the global computational allowance")
    plan = {"schema": 1, "study": "svd-local-comparison-2026-10-13", "seeds": SEEDS,
        "source": source_identity(), "environment": runner.environment_manifest(), "architectures": architectures,
        "training": {"optimizer": "Adam", "learning_rate": .001, "batch_size": BATCH,
                     "dtype": "float32", "threads": THREADS, "recovery_updates": 0,
                     "checkpoint_selection": "fixed final epoch; no validation/test selection"},
        "data": {"split": "60/15/10/15 train/validation/calibration/test stratified by seed",
                 "preprocessing": "MLP fitted on train only; Digits fixed division by16",
                 "exact_feature_duplicate_check": {"BreastCancer": {"rows": 569, "unique_rows": 569},
                                                   "Digits": {"rows": 1797, "unique_rows": 1797},
                                                   "method": "np.unique(original bundled data, axis=0); root read-only verification"},
                 "selection": "all frozen candidates evaluated; test never changes protocol or model"},
        "matching": {"equal_rank": "same stored rank vector and same operators within cohort",
            "equal_budget": "complete unique model tensor bytes, including all biases, buffers and untouched tensors",
            "redundancy": "all selected operators have bias and two factors; equal rank and equal tensor-budget cases coincide",
            "not_matched": "serialized artifact and descriptor bytes; measured separately"},
        "weighted_policy": {"ridge": 0., "rcond": None, "effective_rcond": "float64 epsilon * input width",
            "nullspace_policy": "support_only", "known_limitation": "singular calibration moments do not constrain unseen nullspace action",
            "joint_collection": "features.0 then features.2 then classifier.0; current graph after each replacement"},
        "local_error": "direct isolated layer MSE on frozen baseline inputs, distinct from propagated activation drift",
        "supplementary_work_count": "root may report analytic MAC estimates excluding bias/activation/normalization; not measured time or extra success criterion",
        "construction_timing_scope": "fixed method order within architecture worker; first import/cache overhead charged to actual first consumer; no isolated per-method cold-cost ranking",
        "timing": {"runtime": "TorchScript reloaded in a fresh isolated subprocess per artifact",
                   "batch_size": BATCH, "dtype": "float32", "threads": TIMING_THREADS,
                   "warmup": WARMUP, "repeats": REPEATS, "statistical_unit": "one timing session per artifact; training seeds are separate",
                   "speed_claims": "none under background load; raw session observations only"},
        "resources": {"compute_wall_seconds": COMPUTE_SECONDS, "process_rss_limit_bytes": RSS_LIMIT,
                      "minimum_available_ram_bytes": OS_RESERVE, "max_threads": THREADS,
                      "prior_compute_ledger": prior_ledger, "prior_charged_worker_wall_seconds": prior_seconds,
                      "remaining_at_release_seconds": COMPUTE_SECONDS - prior_seconds,
                      "memory_monitor": "all owned recursive child processes; conservative12GiB cap on both individual RSS and tree RSS sum; shared pages may be counted repeatedly in sum",
                      "cpu_monitor": "recursive CPU-time delta in core-percent units; inactive OS threads are counted separately from configured computation threads",
                      "processes": "one computational worker at a time; CPU only; no cluster"},
        "failure_policy": "retain every candidate and baseline gate failure; no seed/ridge/dtype/rank substitution",
        "inference": "descriptive paired local experiment with3 seeds, not universal article reproduction"}
    write_atomic_json(path, json_value(plan))
    return plan


def _predictions(model, split):
    return runner._predictions(model, split, BATCH, "cpu")


def direct_error(expected, actual):
    if expected.shape != actual.shape or not bool(torch.isfinite(actual).all()):
        raise ProtocolError("Output error requires aligned finite tensors")
    residual = actual.double() - expected.double()
    mse, power = float(residual.square().mean()), float(expected.double().square().mean())
    return {"mse": mse, "relative_mse": mse / power if power else None,
            "reference_power": power, "max_abs_error": float(residual.abs().max()), "elements": expected.numel()}


def _reference_observations(model, split, paths):
    observed = {path: [] for path in paths}
    handles = [model.get_submodule(path).register_forward_hook(
        lambda layer, args, output, name=path: observed[name].append((args[0].detach().clone(), output.detach().clone())))
        for path in paths]
    try:
        _predictions(model, split)
    finally:
        for handle in handles:
            handle.remove()
    return {path: (torch.cat([v[0] for v in rows]), torch.cat([v[1] for v in rows]))
            for path, rows in observed.items()}


def isolated_errors(model, observations):
    result = {}
    with torch.inference_mode():
        for path, (x, expected) in observations.items():
            actual = torch.cat([model.get_submodule(path)(batch) for batch in x.split(BATCH)])
            result[path] = direct_error(expected, actual)
    return result


def propagated_errors(model, split, reference):
    """Same role through the changed graph, including predecessor drift."""
    changed = _reference_observations(model, split, tuple(reference))
    return {path: {"input_error": direct_error(reference[path][0], changed[path][0]),
                   "output_error": direct_error(reference[path][1], changed[path][1])} for path in reference}


def deployed_ranks(model, ranks):
    """Numerical rank of real factors multiplied in float64, rtol1e-10."""
    result = {}
    for path, requested in ranks.items():
        layer = model.get_submodule(path)
        left, right = layer.U.detach().double(), layer.Vh.detach().double()
        if layer.S is not None:
            left = left * layer.S.detach().double().unsqueeze(-2)
        value = left @ right
        numerical = int(torch.linalg.matrix_rank(value, rtol=1e-10, atol=0).item())
        result[path] = {"stored_rank": layer.U.shape[-1], "numerical_rank": numerical,
                       "rank_rtol": 1e-10, "rank_atol": 0, "requested_rank": requested}
        if layer.U.shape[-1] != requested or numerical != requested:
            raise ProtocolError(f"Actual rank differs from matched target at {path}: {result[path]}")
    return result


def _checkpoint_check(model, path, example):
    torch.save(model.state_dict(), path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    restored = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in restored.parameters():
            parameter.zero_()
    restored.load_state_dict(state, strict=True)
    with torch.inference_mode():
        torch.testing.assert_close(restored(example), model(example), rtol=0, atol=0)
    metadata = getattr(model.state_dict(), "_metadata", {})
    return {"path": str(path.resolve()), "sha256": measurement.file_hash(path), "file_bytes": path.stat().st_size,
            "reload_correct": True, "reload_scope": "strict state_dict reload into compressed architecture template",
            "representation_metadata": json_value(metadata),
            "metadata_json_bytes": len(json.dumps(json_value(metadata), sort_keys=True, separators=(",", ":")).encode())}


def _quality_and_error(model, bundle, baseline_outputs):
    result = {}
    for role in ("train", "validation", "calibration", "test"):
        split = getattr(bundle, role)
        output = _predictions(model, split)
        quality = runner._quality(output, split, bundle.task)
        if baseline_outputs is not None:
            teacher = baseline_outputs[role]
            quality["logit_error"] = direct_error(teacher, output)
            log_p, log_q = teacher.double().log_softmax(-1), output.double().log_softmax(-1)
            quality["teacher_to_student_kl"] = max(0., float((log_p.exp() * (log_p - log_q)).sum(-1).mean()))
        result[role] = quality
    return result


def _export_record(model, bundle, directory, record):
    directory.mkdir(parents=True, exist_ok=True)
    example = bundle.validation.x[:BATCH]
    record["checkpoint"] = _checkpoint_check(model, directory / "state_dict.pt", example)
    exported = measurement.measure_artifact(model, example, directory / "model", repeats=1, warmup=0,
                                             threads=TIMING_THREADS)
    record["export"] = exported
    if exported["status"] != "succeeded":
        raise ProtocolError("Artifact export/reload parity failed: " + exported.get("reason", "unknown"))
    # Every validation/test sample is evaluated on the reloaded executable;
    # equality is checked against eager inference before a timing job is created.
    loaded = measurement.load_artifact(exported["artifact"])
    parity = {}
    for role in ("validation", "test"):
        split = getattr(bundle, role)
        with torch.inference_mode():
            actual = torch.cat([measurement._call_loaded(loaded, exported["artifact"], batch)
                                for batch in split.x.split(BATCH)])
        expected = _predictions(model, split)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        parity[role] = {"quality": runner._quality(actual, split, bundle.task), "error": direct_error(expected, actual)}
    record["export_reload"] = {"all_validation_test_parity": True, "roles": parity}
    input_path = directory / "timing-input.pt"
    torch.save(example.clone(), input_path)
    descriptor = {"artifact": exported["artifact"], "input_path": str(input_path.resolve()),
                  "input_sha256": measurement.file_hash(input_path), "output": str((directory / "timing.json").resolve())}
    write_atomic_json(directory / "timing-job.json", descriptor)
    record["timing_job"] = str((directory / "timing-job.json").resolve())
    record["storage"]["artifact_file_bytes"] = exported["metrics"]["file_bytes"]
    record["storage"]["artifact_descriptor_json_bytes"] = len(json.dumps(exported["artifact"], sort_keys=True).encode())
    sidecar = Path(exported["artifact"]["path"]).with_suffix(".pt.json")
    record["artifact_sidecar"] = {"path": str(sidecar), "sha256": measurement.file_hash(sidecar),
                                  "file_bytes": sidecar.stat().st_size}
    record["storage"]["artifact_sidecar_file_bytes"] = sidecar.stat().st_size


def architecture_worker(job):
    architecture, seed = job["architecture"], job["seed"]
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(job["protocol"]).read_text(encoding="utf-8"))
    if source_identity() != plan["source"]:
        raise ProtocolError("Executed source differs from frozen Git-LF snapshot")
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    bundle = build_bundle(architecture, seed)
    config = plan["architectures"][architecture]
    protocol = ExperimentProtocol(seed=seed, batch_size=BATCH, baseline_epochs=config["baseline_epochs"],
        finetune_epochs=0, learning_rate=.001, measurement_repeats=REPEATS, warmup=WARMUP, threads=THREADS)
    baseline = copy.deepcopy(bundle.original_model)
    started = time.perf_counter()
    training = runner.train_model(baseline, bundle.train, bundle.task, epochs=protocol.baseline_epochs,
        batch_size=BATCH, learning_rate=.001, seed=seed)
    baseline.eval()
    baseline_hash = runner.model_state_hash(baseline)
    validation = runner.quality_metrics(baseline, bundle.validation, bundle.task, BATCH)
    gate = validation["value"] >= config["minimum_validation_accuracy"]
    manifest = {"architecture": architecture, "seed": seed, "protocol_sha256": measurement.file_hash(job["protocol"]),
        "data": bundle.manifest(), "training": training, "baseline_hash": baseline_hash,
        "gate": {"passed": gate, "metric": "validation_accuracy", "minimum": config["minimum_validation_accuracy"],
                 "observed": validation["value"], "seed_replacement_allowed": False}, "records": []}
    baseline_outputs = {role: _predictions(baseline, getattr(bundle, role)) for role in
                        ("train", "validation", "calibration", "test")}
    baseline_record = {"case_id": "baseline", "method": "baseline", "cohort": "baseline", "seed": seed,
        "architecture": architecture, "status": "succeeded", "quality": _quality_and_error(baseline, bundle, None),
        "storage": exact_storage(baseline), "model_hash": baseline_hash, "source_baseline_hash": baseline_hash}
    try:
        _export_record(baseline, bundle, directory / "baseline", baseline_record)
    except Exception as error:
        baseline_record.update(status="failed", error_type=type(error).__name__, reason=str(error), traceback=traceback.format_exc())
    manifest["baseline"] = baseline_record
    write_atomic_json(directory / "run.json", manifest)
    all_paths = tuple(dict.fromkeys(path for case in config["cases"] for path in case["ranks"]))
    references = {role: _reference_observations(baseline, getattr(bundle, role), all_paths)
                  for role in ("calibration", "validation")} if gate else {}
    for case in config["cases"]:
        record = dict(case, seed=seed, architecture=architecture, source_baseline_hash=baseline_hash,
                      recovery={"updates": 0, "seconds": 0., "scope": "primary zero-recovery series"})
        destination = directory / case["case_id"]
        destination.mkdir(parents=True, exist_ok=True)
        tick = time.perf_counter()
        if not gate:
            record.update(status="blocked_by_baseline_gate", reason="Predeclared baseline quality gate failed")
        else:
            try:
                candidate = CandidateSpec.from_dict(case["candidate"])
                model, transform = runner.apply_candidate(baseline, bundle, protocol, candidate)
                model.eval()
                record["compression_seconds"] = time.perf_counter() - tick
                record["transform"] = transform
                record["source_unchanged"] = runner.model_state_hash(baseline) == baseline_hash
                if not record["source_unchanged"]:
                    raise ProtocolError("Candidate mutated the common trained baseline")
                record["actual_ranks"] = deployed_ranks(model, case["ranks"])
                record["storage"] = exact_storage(model)
                if record["storage"]["tensor_bytes"] != case["budget"]["whole_model_tensor_byte_budget"]:
                    raise ProtocolError("Actual whole-model tensor storage differs from frozen budget")
                record["quality"] = _quality_and_error(model, bundle, baseline_outputs)
                record["isolated_layer_errors"] = {role: isolated_errors(model, {path: references[role][path]
                    for path in case["ranks"]}) for role in references}
                record["propagated_layer_errors"] = {role: propagated_errors(model, getattr(bundle, role),
                    {path: references[role][path] for path in case["ranks"]}) for role in references}
                record["model_hash"] = runner.model_state_hash(model)
                _export_record(model, bundle, destination, record)
                record["status"] = "succeeded"
                del model
            except Exception as error:
                record.update(status="failed", error_type=type(error).__name__, reason=str(error), traceback=traceback.format_exc())
        record["total_candidate_seconds"] = time.perf_counter() - tick
        write_atomic_json(destination / "record.json", record)
        manifest["records"].append(record)
        manifest["elapsed_seconds"] = time.perf_counter() - started
        write_atomic_json(directory / "run.json", manifest)
        print(json.dumps({"architecture": architecture, "seed": seed, "case_id": case["case_id"], "status": record["status"]}), flush=True)
    return manifest


def timing_worker(job):
    import psutil
    torch.set_num_threads(TIMING_THREADS)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if measurement.file_hash(job["input_path"]) != job["input_sha256"]:
        raise ProtocolError("Timing input hash mismatch")
    x = torch.load(job["input_path"], weights_only=True, map_location="cpu")
    before_load = time.perf_counter()
    loaded = measurement.load_artifact(job["artifact"])
    load_seconds = time.perf_counter() - before_load
    before_cpu = psutil.cpu_percent(interval=.25)
    raw = []
    with torch.inference_mode():
        for _ in range(WARMUP):
            measurement._call_loaded(loaded, job["artifact"], x)
        for _ in range(REPEATS):
            tick = time.perf_counter_ns()
            measurement._call_loaded(loaded, job["artifact"], x)
            raw.append((time.perf_counter_ns() - tick) / 1e6)
    after_cpu = psutil.cpu_percent(interval=.25)
    result = {"status": "succeeded", "artifact": job["artifact"], "process_id": os.getpid(),
        "profile": {"runtime": "torch.jit", "runtime_version": str(torch.__version__), "threads": TIMING_THREADS,
            "batch_size": len(x), "input_shape": list(x.shape), "dtype": str(x.dtype),
            "warmup": WARMUP, "repeats": REPEATS, "fresh_process": True, "concurrency": 1,
            "statistical_unit": "one process timing session, not independent training repeats"},
        "raw_inference_ms": raw, "metrics": {"latency_p50_ms": statistics.median(raw),
            "latency_p95_ms": measurement._percentile(raw, .95), "throughput_samples_s": len(x) * 1000 / statistics.mean(raw),
            "load_seconds": load_seconds}, "system_cpu_percent_before": before_cpu,
        "system_cpu_percent_after": after_cpu, "speed_claim": "no inference beyond observed local session latency",
        "cpu_rss_after_bytes": psutil.Process().memory_info().rss,
        "rss_scope": "process snapshot; sampled process maximum is in parent resources.json"}
    write_atomic_json(job["output"], result)
    return result


def _owned_process_sample(root_process):
    """Include venv launchers and their real interpreters without conflating RSS."""
    import psutil
    processes = [root_process]
    try:
        processes += root_process.children(recursive=True)
    except psutil.NoSuchProcess:
        pass
    result = []
    for process in processes:
        try:
            cpu = process.cpu_times()
            result.append({"pid": process.pid, "parent_pid": process.ppid(), "created": process.create_time(),
                           "rss_bytes": process.memory_info().rss, "cpu_seconds": cpu.user + cpu.system,
                           "os_threads": process.num_threads()})
        except psutil.NoSuchProcess:
            continue
    return result


def _terminate_owned_tree(root_process):
    """Terminate descendants before the launcher so no interpreter is orphaned."""
    import psutil
    try:
        children = root_process.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    targets = list(reversed(children)) + [root_process]
    for process in targets:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    # Popen.wait owns reaping the direct worker and its raw OS return code.
    # psutil.wait_procs on that same PID can reap it first on POSIX, after which
    # Popen may see ECHILD and synthesize a zero return code.
    psutil.wait_procs(children, timeout=10)
    for process in targets:
        try:
            if process.is_running():
                process.kill()
        except psutil.NoSuchProcess:
            pass


def monitored_process(command, directory, deadline):
    """One worker; sample RSS and reserve; retain capped/failed process evidence."""
    import psutil
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    resources, violation = [], None
    started = time.perf_counter()
    if time.monotonic() >= deadline:
        result = {"returncode": 1, "violation": "series compute wall cap reached before worker start",
                  "samples": [], "elapsed_seconds": 0., "sampled_maximum_rss_bytes": 0,
                  "sample_interval_seconds": .1, "scope": "worker was not started"}
        write_atomic_json(directory / "resources.json", result)
        return result
    env = dict(os.environ, OMP_NUM_THREADS=str(THREADS), MKL_NUM_THREADS=str(THREADS), OPENBLAS_NUM_THREADS=str(THREADS),
               NUMEXPR_NUM_THREADS=str(THREADS), BLIS_NUM_THREADS=str(THREADS))
    with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env)
        sampled = psutil.Process(process.pid)
        prior_cpu, prior_tick = {}, time.monotonic()
        while process.poll() is None:
            try:
                memory = psutil.virtual_memory()
                owned = _owned_process_sample(sampled)
                tick = time.monotonic()
                current_cpu = {(item["pid"], item["created"]): item["cpu_seconds"] for item in owned}
                delta = sum(max(0., seconds - prior_cpu[key]) for key, seconds in current_cpu.items() if key in prior_cpu)
                tree_cpu = 100 * delta / (tick - prior_tick) if tick > prior_tick else 0.
                prior_cpu, prior_tick = current_cpu, tick
                rss = max((item["rss_bytes"] for item in owned), default=0)
                tree_rss = sum(item["rss_bytes"] for item in owned)
                resources.append({"elapsed_monotonic": tick, "rss_bytes": rss,
                                  "individual_maximum_rss_bytes": rss,
                                  "tree_sum_rss_bytes": tree_rss,
                                  "tree_cpu_core_percent": tree_cpu, "owned_processes": owned,
                                  "system_available_bytes": memory.available,
                                  "system_cpu_percent": psutil.cpu_percent(interval=None)})
                if rss > RSS_LIMIT:
                    violating = [item["pid"] for item in owned if item["rss_bytes"] > RSS_LIMIT]
                    violation = f"owned process RSS cap exceeded: PIDs {violating}"
                elif tree_rss > RSS_LIMIT:
                    violation = "owned process-tree RSS cap exceeded (conservative sum; shared pages may be counted repeatedly)"
                elif memory.available < OS_RESERVE:
                    violation = "minimum OS RAM reserve breached"
                elif time.monotonic() >= deadline:
                    violation = "series compute wall cap reached"
                if violation:
                    _terminate_owned_tree(sampled)
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(.1)
        returncode = process.wait(timeout=30)
    result = {"returncode": returncode, "violation": violation, "samples": resources,
              "elapsed_seconds": time.perf_counter() - started,
              "sampled_maximum_rss_bytes": max((r["rss_bytes"] for r in resources), default=0),
              "sampled_maximum_individual_rss_bytes": max((r["rss_bytes"] for r in resources), default=0),
              "sampled_maximum_tree_sum_rss_bytes": max((r["tree_sum_rss_bytes"] for r in resources), default=0),
              "sampled_maximum_tree_cpu_core_percent": max((r["tree_cpu_core_percent"] for r in resources), default=0),
              "sample_interval_seconds": .1,
              "scope": "maximum sampled individual RSS includes actual recursive interpreters; separate tree sum may double-count shared pages; neither is exact allocator peak"}
    write_atomic_json(directory / "resources.json", result)
    return result


def consolidate_results(output):
    """Inspectable rows and paired seed summaries, without significance claims."""
    output = Path(output)
    runs, rows = [], []
    groups = {}
    for path in sorted(output.glob("*-seed-*/run.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        runs.append({"architecture": run["architecture"], "seed": run["seed"], "gate": run["gate"],
                     "baseline_hash": run["baseline_hash"], "run_path": str(path.resolve()),
                     "training_seconds": run.get("training", {}).get("training_seconds"),
                     "training_steps": run.get("training", {}).get("training_steps")})
        baseline = run["baseline"]
        for source in (baseline, *run["records"]):
            row = dict(source)
            if row.get("timing_job"):
                timing_path = Path(row["timing_job"]).parent / "timing.json"
                if timing_path.exists():
                    row["timing"] = json.loads(timing_path.read_text(encoding="utf-8"))
                resource_path = timing_path.parent / "timing-process/resources.json"
                if resource_path.exists():
                    resource = json.loads(resource_path.read_text(encoding="utf-8"))
                    row["timing_resources"] = {key: value for key, value in resource.items() if key != "samples"}
            rows.append(row)
            if row["method"] == "baseline":
                continue
            key = row["architecture"], row["cohort"], row["level"], row["method"]
            group = groups.setdefault(key, {"paired_seeds": [], "failed_seeds": [], "test_accuracy_deltas": [],
                                           "test_macro_f1_deltas": [], "test_cross_entropy_deltas": [],
                                           "tensor_bytes": [], "compression_seconds": []})
            if row["status"] != "succeeded" or baseline["status"] != "succeeded":
                group["failed_seeds"].append(row["seed"])
                continue
            group["paired_seeds"].append(row["seed"])
            for metric, field in (("value", "test_accuracy_deltas"), ("macro_f1", "test_macro_f1_deltas"),
                                  ("cross_entropy", "test_cross_entropy_deltas")):
                group[field].append(row["quality"]["test"][metric] - baseline["quality"]["test"][metric])
            group["tensor_bytes"].append(row["storage"]["tensor_bytes"])
            group["compression_seconds"].append(row["compression_seconds"])
            row["baseline_differences"] = {"test_accuracy": group["test_accuracy_deltas"][-1],
                "test_macro_f1": group["test_macro_f1_deltas"][-1], "test_cross_entropy": group["test_cross_entropy_deltas"][-1],
                "tensor_byte_ratio": row["storage"]["tensor_bytes"] / baseline["storage"]["tensor_bytes"],
                "artifact_file_byte_ratio": row["storage"]["artifact_file_bytes"] / baseline["storage"]["artifact_file_bytes"]}
        # Timing baselines may be read later in the loop: bind via the row map.
    baseline_rows = {(row["architecture"], row["seed"]): row for row in rows if row["method"] == "baseline"}
    for row in rows:
        baseline = baseline_rows.get((row["architecture"], row["seed"]))
        if row["method"] != "baseline" and row.get("timing") and baseline and baseline.get("timing"):
            row.setdefault("baseline_differences", {})["observed_latency_ratio"] = (
                row["timing"]["metrics"]["latency_p50_ms"] / baseline["timing"]["metrics"]["latency_p50_ms"])
    summaries = []
    for key, group in sorted(groups.items()):
        summaries.append(dict(zip(("architecture", "cohort", "level", "method"), key), **group,
            descriptive={field: {"mean": statistics.mean(group[field]) if group[field] else None,
                                 "sample_sd": statistics.stdev(group[field]) if len(group[field]) > 1 else None}
                         for field in ("test_accuracy_deltas", "test_macro_f1_deltas", "test_cross_entropy_deltas")},
            inference="paired3-seed descriptive comparison; timings are not independent seed repetitions"))
    summary = {"schema": 1, "protocol": str((output / "protocol.json").resolve()), "runs": runs,
        "rows": rows, "paired_summaries": summaries, "counts": {
            "baselines": sum(row["method"] == "baseline" for row in rows),
            "candidates": sum(row["method"] != "baseline" for row in rows),
            "succeeded_candidates": sum(row["method"] != "baseline" and row["status"] == "succeeded" for row in rows),
            "timing_sessions": sum("timing" in row for row in rows)}}
    write_atomic_json(output / "summary.json", summary)
    return summary


def _retain_worker_failure(directory, job, config, resources):
    path = Path(directory) / "run.json"
    run = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "architecture": job["architecture"], "seed": job["seed"], "baseline_hash": None,
        "gate": {"passed": None, "reason": "worker failed before establishing baseline gate"},
        "baseline": {"architecture": job["architecture"], "seed": job["seed"], "case_id": "baseline",
                     "cohort": "baseline", "method": "baseline", "status": "worker_failed"}, "records": []}
    existing = {record["case_id"] for record in run["records"]}
    for case in config["cases"]:
        if case["case_id"] not in existing:
            run["records"].append(dict(case, architecture=job["architecture"], seed=job["seed"],
                status="not_completed_worker_failure", reason=resources.get("violation") or "worker exited before candidate completed"))
    run["worker_failure"] = {key: value for key, value in resources.items() if key != "samples"}
    write_atomic_json(path, run)


def _worker_succeeded(resources):
    """A resource violation is failure even when the OS process exits zero."""
    return resources["returncode"] == 0 and not resources["violation"]


def run_series(output, *, timing_only=False, prior_runs=()):
    output = Path(output).resolve()
    protocol_path = output / "protocol.json"
    if timing_only and prior_runs:
        raise ProtocolError("Prior attempts must be charged at initial protocol release")
    plan = (json.loads(protocol_path.read_text(encoding="utf-8")) if timing_only else
            predeclare_plan(output, prior_runs=prior_runs))
    if source_identity() != plan["source"]:
        raise ProtocolError("Source changed after the prospective protocol snapshot")
    previous_execution = output / "execution.json"
    previous_timing = output / "timing-execution.json"
    timings = json.loads(previous_timing.read_text(encoding="utf-8"))["workers"] if previous_timing.exists() else []
    used_seconds = 0.
    if timing_only and previous_execution.exists():
        used_seconds = sum(item["resources"]["elapsed_seconds"] for item in
                           json.loads(previous_execution.read_text(encoding="utf-8"))["architecture_workers"])
    if timing_only:
        used_seconds += sum(item["resources"]["elapsed_seconds"] for item in timings)
    allowance = plan.get("resources", {}).get("remaining_at_release_seconds", COMPUTE_SECONDS)
    deadline = time.monotonic() + max(0., allowance - used_seconds)
    computational = []
    if not timing_only:
        for architecture in ("mlp", "cnn"):
            for seed in SEEDS:
                directory = output / f"{architecture}-seed-{seed}"
                job = {"architecture": architecture, "seed": seed, "directory": str(directory),
                       "protocol": str(protocol_path)}
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / "job.json"
                write_atomic_json(path, job)
                evidence = monitored_process([sys.executable, "-m", MODULE, "--architecture-job", str(path)], directory, deadline)
                computational.append(dict(job, resources=evidence))
                write_atomic_json(output / "execution.json", {"architecture_workers": computational})
                if not _worker_succeeded(evidence):
                    _retain_worker_failure(directory, job, plan["architectures"][architecture], evidence)
                if evidence["violation"]:
                    consolidate_results(output)
                    return 1
    # Root explicitly coordinates this boundary: inference starts only when
    # --timing-only is separately invoked after training/compression completes.
    if not timing_only:
        consolidate_results(output)
        return 0 if all(_worker_succeeded(item["resources"]) for item in computational) else 1
    for path in sorted(output.glob("*-seed-*/*/timing-job.json")):
        job = json.loads(path.read_text(encoding="utf-8"))
        timing_output = Path(job["output"])
        if timing_output.exists():
            completed = json.loads(timing_output.read_text(encoding="utf-8"))
            if completed.get("status") == "succeeded" and completed.get("artifact") == job["artifact"]:
                continue
        attempt = 1 + sum(item["job"] == str(path) for item in timings)
        process_dir = path.parent / "timing-process"
        evidence = monitored_process([sys.executable, "-m", MODULE, "--timing-job", str(path)],
                                     process_dir / f"attempt-{attempt}", deadline)
        write_atomic_json(process_dir / "resources.json", evidence)
        timings.append({"job": str(path), "resources": evidence})
        write_atomic_json(output / "timing-execution.json", {"workers": timings})
        if evidence["violation"]:
            return 1
    consolidate_results(output)
    return 0 if all(_worker_succeeded(item["resources"]) for item in timings) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--prior-run", type=Path, action="append", default=[])
    parser.add_argument("--architecture-job", type=Path)
    parser.add_argument("--timing-job", type=Path)
    options = parser.parse_args()
    if options.architecture_job:
        architecture_worker(json.loads(options.architecture_job.read_text(encoding="utf-8")))
    elif options.timing_job:
        timing_worker(json.loads(options.timing_job.read_text(encoding="utf-8")))
    elif options.output:
        raise SystemExit(run_series(options.output, timing_only=options.timing_only, prior_runs=options.prior_run))
    else:
        parser.error("Supply --output or one worker job")


if __name__ == "__main__":
    main()
