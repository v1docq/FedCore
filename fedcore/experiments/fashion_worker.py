"""Guarded Fashion-MNIST M0 and paired grammar-search worker.

Invoke with python -m fedcore.experiments.fashion_worker COMMAND --manifest ...
--run-root ... --limits ... beneath quantization_pilot_guard.py only.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import traceback

if "--run-root" in sys.argv:
    _at = sys.argv.index("--run-root") + 1
    if _at < len(sys.argv):
        os.environ["FEDCORE_MODEL_REGISTRY_PATH"] = str(Path(sys.argv[_at]).resolve()/"registry")

import torch
from torch import nn
from torch.nn import functional as F

from .fashion_protocol import (FashionPlan, PATHS, RAW_FILES, MD5, CONTROLS, SEARCH_JOBS,
    default_manifest, stratified_partition, stratified_subset, classify_search_artifact,
    select_archive_candidates, source_code_hashes)
from .quantization_pilot_protocol import canonical_bytes, digest, file_hash, HARD_RSS, search_execution_summary
from .quantization_pilot_guard import Limits, atomic_json
from .quantization_pilot_worker import (jsonable, append_json, checkpoint, load_artifact,
    require_guard, prefix_sample, unique_initial)
from fedcore.algorithm.quantization import (quantize, QuantizationGraphRequest, OperatorRecipe,
    TransformSpec, Method, QuantizedTensorSpec, Grid, Rounding, ExportVerifiedArtifact,
    ProgramArtifact, QuantizationProgram, BackendStageSpec, ResourceLedger, StageBudgets)


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def runtime_info():
    return {"torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "device": "cpu", "torch_version": str(torch.__version__)}


def verify_code(root):
    frozen = json.loads((root/"code.lock.json").read_text(encoding="utf-8"))
    if frozen["files"] != source_code_hashes():
        raise ValueError("Experiment source changed after protocol freeze")
    return file_hash(root/"code.lock.json")


def verified_raw(plan, root, name, *, allow_test=False):
    if name not in RAW_FILES:
        raise ValueError("Unknown raw data file")
    if name.startswith("t10k") and not allow_test:
        raise PermissionError("Test IDX cannot be opened before selection lock")
    if name.startswith("t10k") and not (root/"selection.lock.json").exists():
        raise PermissionError("Test IDX requires selection lock")
    path = root/"data"/"raw"/name
    if file_hash(path) != plan.payload["raw_sha256"][name]:
        raise ValueError("Raw SHA256 differs from protocol: " + name)
    h = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    if h.hexdigest() != MD5[name]:
        raise ValueError("Raw MD5 differs from canonical Fashion-MNIST: " + name)
    return path


def read_idx(path, *, images, expected_rows):
    with gzip.open(path, "rb") as stream:
        raw = stream.read()
    header = 16 if images else 8
    expected_magic = 2051 if images else 2049
    if len(raw) < header:
        raise ValueError("Truncated IDX header")
    magic, rows = struct.unpack(">II", raw[:8])
    if magic != expected_magic or rows != expected_rows:
        raise ValueError("Unexpected IDX type or row count")
    if images and struct.unpack(">II", raw[8:16]) != (28, 28):
        raise ValueError("Expected 28x28 Fashion-MNIST images")
    if len(raw) != header + rows*(784 if images else 1):
        raise ValueError("IDX payload length mismatch")
    result = torch.frombuffer(bytearray(raw[header:]), dtype=torch.uint8).clone()
    return result.reshape(rows, 784) if images else result.long()


def features(images, ids):
    return images[ids].to(torch.float32).div_(255.)


def class_counts(labels, ids):
    return torch.bincount(labels[ids], minlength=10).tolist()


def prepare(plan, root):
    verify_code(root)
    if (root/"prepared.json").exists():
        raise FileExistsError("Preparation cannot be replaced")
    images = read_idx(verified_raw(plan, root, RAW_FILES[0]), images=True, expected_rows=60000)
    labels = read_idx(verified_raw(plan, root, RAW_FILES[1]), images=False, expected_rows=60000)
    roles = stratified_partition(labels.tolist())
    roles["C"] = stratified_subset(roles["calibration"], labels.tolist(), 256, seed=20261028)
    roles["Vsearch"] = stratified_subset(roles["search"], labels.tolist(), 512, seed=20261029)
    data = {"images": images, "labels": labels,
            "indices": {role: torch.tensor(ids, dtype=torch.int64) for role, ids in roles.items()}}
    data_path = root/"data.pt"
    torch.save(data, data_path)
    summary = {"manifest_sha256": plan.fingerprint, "code_lock_sha256": verify_code(root),
        "data_sha256": file_hash(data_path), "images_sha256": tensor_hash(images), "labels_sha256": tensor_hash(labels),
        "closed_test_opened": False, "roles": {role: {"rows": len(ids), "ids_sha256": digest(ids),
        "ids": ids, "class_counts": class_counts(labels, ids)} for role, ids in roles.items()},
        "overlap_audit": "fit/stop/calibration/search/pick are exhaustive disjoint; C and Vsearch are declared subsets",
        "torch_version": str(torch.__version__), "python_version": sys.version, "runtime": runtime_info()}
    atomic_json(root/"prepared.json", summary)
    return summary


def load_data(plan, root):
    verify_code(root)
    summary = json.loads((root/"prepared.json").read_text())
    if summary["manifest_sha256"] != plan.fingerprint or file_hash(root/"data.pt") != summary["data_sha256"]:
        raise ValueError("Frozen manifest or training data changed")
    return summary, torch.load(root/"data.pt", map_location="cpu", weights_only=True)


def evaluate(model, images, labels, ids, *, reference_predictions=None, batch_size=128,
             deadline=None, retain_predictions=False):
    """Ground-truth CE/accuracy. Teacher agreement has a separate explicit field."""
    predictions, truth = [], []
    ce_sum = 0.
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("Full evaluation exceeded frozen deadline")
            selected = ids[start:start+batch_size]
            logits = model(features(images, selected))
            targets = labels[selected]
            if logits.shape != (len(selected), 10) or not bool(torch.isfinite(logits).all()):
                raise ValueError("Model produced invalid logits")
            ce_sum += float(F.cross_entropy(logits.double(), targets, reduction="sum"))
            predictions.append(logits.argmax(1)); truth.append(targets)
    predicted, target = torch.cat(predictions), torch.cat(truth)
    correct = predicted == target
    result = {"cross_entropy": ce_sum/len(ids), "accuracy": float(correct.double().mean()),
              "examples": len(ids), "class_counts": torch.bincount(target, minlength=10).tolist(),
              "confusion": torch.bincount(target*10+predicted, minlength=100).reshape(10, 10).tolist(),
              "meaning": "official ground-truth classification"}
    if reference_predictions is not None:
        if len(reference_predictions) != len(predicted):
            raise ValueError("Reference predictions must cover the identical ordered examples")
        result["m0_agreement"] = float((predicted == torch.as_tensor(reference_predictions)).double().mean())
    if retain_predictions:
        result.update(predictions=predicted.tolist(), correct=correct.tolist())
    return result


def make_source():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261027)
        return nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))


def train(plan, root, limits):
    path = root/"trained.json"
    if path.exists() or (root/"training.started.json").exists():
        raise FileExistsError("M0 training cannot restart or replace a result")
    summary, data = load_data(plan, root)
    atomic_json(root/"training.started.json", {"manifest_sha256": plan.fingerprint, "started_unix": time.time(),
                "prepared_sha256": file_hash(root/"prepared.json")})
    training = plan.payload["training"]
    deadline = time.perf_counter()+min(training["max_seconds"], limits.remaining()-3)
    model = make_source()
    optimizer = torch.optim.Adam(model.parameters(), lr=training["lr"])
    generator = torch.Generator().manual_seed(20261027)
    ids = data["indices"]["fit"]; best_ce = math.inf; best_state = None; history = []
    start = time.perf_counter()
    completed = True
    for epoch in range(training["epochs"]):
        model.train(); order = ids[torch.randperm(len(ids), generator=generator)]
        total_loss = 0.; rows = 0
        for selected in order.split(training["batch_size"]):
            if time.perf_counter() >= deadline:
                completed = False; break
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(features(data["images"], selected)), data["labels"][selected])
            if not bool(torch.isfinite(loss)):
                raise ValueError("M0 training loss became nonfinite")
            loss.backward(); optimizer.step()
            total_loss += float(loss.detach())*len(selected); rows += len(selected)
        if not completed:
            break
        quality = evaluate(model, data["images"], data["labels"], data["indices"]["stop"], deadline=deadline)
        record = {"epoch": epoch+1, "fit_ce": total_loss/rows, "stop": quality, "elapsed_seconds": time.perf_counter()-start}
        history.append(record); append_json(root/"training.history.jsonl", record)
        if quality["cross_entropy"] < best_ce:
            best_ce = quality["cross_entropy"]; best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            best_epoch = epoch+1
            _, best_metadata = checkpoint(model.eval(), root, features(data["images"], data["indices"]["C"][:4]))
            atomic_json(root/"training.best.json", {"manifest_sha256": plan.fingerprint, "epoch": best_epoch,
                "stop": quality, "source": best_metadata, "history_rows": len(history),
                "prepared_sha256": file_hash(root/"prepared.json")})
    if best_state is None:
        raise TimeoutError("No completed M0 epoch; the fixed training run cannot be restarted")
    model.load_state_dict(best_state); model.eval()
    loaded, metadata = checkpoint(model, root, features(data["images"], data["indices"]["C"][:4]))
    quality = (evaluate(loaded, data["images"], data["labels"], data["indices"]["stop"], deadline=deadline)
        if time.perf_counter() < deadline else history[best_epoch-1]["stop"])
    accepted = completed and len(history) == training["epochs"] and quality["accuracy"] >= training["minimum_stop_accuracy"]
    result = {"manifest_sha256": plan.fingerprint, "prepared_sha256": file_hash(root/"prepared.json"),
        "code_lock_sha256": verify_code(root), "data_sha256": summary["data_sha256"], "source": metadata,
        "budget_bytes": math.floor(metadata["physical_tensor_bytes"]*.5), "selected_epoch": best_epoch,
        "epochs_completed": len(history), "history": history, "stop": quality,
        "status": "accepted" if accepted else "rejected", "fixed_training_completed": completed,
        "wall_seconds": time.perf_counter()-start, "runtime": runtime_info()}
    atomic_json(path, result)
    return result


def load_prepared(plan, root):
    summary, data = load_data(plan, root)
    trained = json.loads((root/"trained.json").read_text())
    if (trained["status"] != "accepted" or trained["manifest_sha256"] != plan.fingerprint
            or trained["prepared_sha256"] != file_hash(root/"prepared.json") or trained["data_sha256"] != summary["data_sha256"]):
        raise ValueError("M0 is rejected or provenance changed")
    source = load_artifact(root, trained["source"])
    return trained, source, data


def recipe(method, rank=None):
    q4 = QuantizedTensorSpec(4)
    endpoint = QuantizedTensorSpec(4, rounding=Rounding.STOCHASTIC, grid=Grid.ENDPOINT)
    if method == "dense": return TransformSpec(Method.DENSE)
    if method == "ptq4": return TransformSpec(Method.PTQ, weight=QuantizedTensorSpec(4, axis=0))
    if method == "svd": return TransformSpec(Method.SVD, rank=rank)
    if method == "dsvd4": return TransformSpec(Method.DSVD, rank=rank, left=q4, right=q4)
    if method in ("lplr4", "lplr_svd4"):
        return TransformSpec(Method.LPLR if method == "lplr4" else Method.LPLR_SVD,
            rank=rank, comparison_rank=rank, left=endpoint, right=endpoint, seed=0)
    if method == "caldera4":
        return TransformSpec(Method.CALDERA, rank=rank, weight=q4, left=q4, right=q4,
            outer_iterations=1, inner_iterations=1, ridge=.0001, seed=0)
    if method == "tender_w4a8":
        return TransformSpec(Method.TENDER, weight=QuantizedTensorSpec(4, axis=0), activation_bits=8)
    raise ValueError("Unknown finite Fashion recipe")


def alternatives(plan):
    result = {}
    for index, path in enumerate(PATHS):
        choices = [recipe("dense"), recipe("ptq4")]
        for rank in plan.payload["search"]["ranks"][index]:
            choices.extend(recipe(method, rank) for method in ("svd", "dsvd4", "lplr4", "lplr_svd4", "caldera4"))
        if path == "4": choices.append(recipe("tender_w4a8"))
        result[path] = tuple(choices)
    return result


def data_identity(data):
    return {"images": tensor_hash(data["images"]), "labels": tensor_hash(data["labels"]),
            "indices": {name: tensor_hash(ids) for name, ids in data["indices"].items()}}


def seeded_request(workflow):
    """Execution randomness is a pure function of the whole canonical genotype."""
    from fedcore.algorithm.quantization.grammar import genotype_payload
    identity = digest(genotype_payload(workflow.genotype))
    seed = int(identity[:15], 16)
    request = replace(workflow.request, recipes=tuple(replace(item, transform=replace(item.transform, seed=seed))
                      for item in workflow.request.recipes))
    return identity, seed, request


def run_search(plan, root, job, source, data, budget, ledger, seconds, limits):
    from fedcore.algorithm.quantization.grammar import (prepare_quantization_space, QUANTIZATION_GRAMMAR_ADAPTER,
        source_identity, calibration_identity, genotype_payload)
    from fedcore.algorithm.quantization.cache import DependencyCache
    from fedcore.algorithm.low_rank.grammar_search import run_grammar_search, GrammarFitness, GrammarSearchConfig
    from fedcore.algorithm.low_rank.grammar_execution import ResourceFailure, NumericalFailure
    local_deadline = time.perf_counter()+seconds
    calibration = features(data["images"], data["indices"]["C"])
    context = digest({"manifest": plan.fingerprint, "source": source_identity(source), "data": data_identity(data),
                      "trained_sha256": file_hash(root/"trained.json"), "fitness_role": "Vsearch512_ground_truth"})
    space = prepare_quantization_space(source, alternatives(plan), calibration=calibration, evaluation_context=context,
        max_storage_bytes=budget, max_peak_bytes=HARD_RSS, prequantize_bits=(2, 4), training_choices=(None,))
    if not hasattr(space, "domains"):
        return None, {"space_failure": jsonable(space), "search": {"status": "failed", "reason": "space preparation"}}
    frozen_data = data_identity(data); cache = DependencyCache(max_bytes=plan.payload["search"]["cache_bytes"])
    attempts = []
    def context_guard():
        if (source_identity(source) != space.source or calibration_identity(calibration).fingerprint != space.calibration
                or data_identity(data) != frozen_data):
            return ResourceFailure("StaleContext", "M0, calibration, labels, role IDs or training data changed")
        return None
    def assessment(workflow):
        payload = genotype_payload(workflow.genotype)
        identity, seed, request = seeded_request(workflow)
        started = time.perf_counter()
        record = {"workflow_fingerprint": workflow.fingerprint, "canonical_solution_id": identity,
            "genotype": payload, "execution_seed": seed, "execution_request": jsonable(request)}
        try:
            ledger.checkpoint("search_candidate")
            outcome = quantize(source, request, calibration=calibration, ledger=ledger, cache=cache)
            if type(outcome) is not ExportVerifiedArtifact:
                record.update(status="failed", failure=jsonable(outcome)); return outcome
            loaded, metadata = checkpoint(outcome.model, root, calibration[:4])
            quality = evaluate(loaded, data["images"], data["labels"], data["indices"]["Vsearch"], deadline=local_deadline)
            feasible = metadata["physical_tensor_bytes"] <= budget
            record.update(status="succeeded" if feasible else "infeasible", artifact=metadata, search=quality)
            ledger.checkpoint("search_candidate_complete")
            return GrammarFitness(quality["cross_entropy"], float(metadata["physical_tensor_bytes"]), feasible)
        except Exception as error:
            record.update(status="failed", failure={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
            return getattr(error, "failure", NumericalFailure(type(error).__name__, str(error)))
        finally:
            record["seconds"] = time.perf_counter()-started
            attempts.append(record); append_json(root/"jobs"/(job+".attempts.jsonl"), record)
    seed = int(job.rsplit("_", 1)[1]); random_panel = job.startswith("random_")
    initial = unique_initial(space, 24 if random_panel else 4, seed)
    adapter = replace(QUANTIZATION_GRAMMAR_ADAPTER, generate=prefix_sample, context_guard=context_guard)
    config = GrammarSearchConfig(seed=seed, population_size=24 if random_panel else 4,
        generations=1 if random_panel else 24, max_evaluations=24,
        max_seconds=max(.001, local_deadline-time.perf_counter()), max_proposals=128)
    start = time.perf_counter()
    result = run_grammar_search(space, config, assessment, fitness_context=context, grammar_adapter=adapter,
        initial_genotypes=initial, strict_deadline=True,
        trace_callback=lambda event: append_json(root/"jobs"/(job+".trace.jsonl"), event))
    search_resources = ledger.evidence()
    atomic_json(root/"jobs"/(job+".search.json"), {"search": jsonable(result), "context": context,
        "initial_genotypes": [genotype_payload(item) for item in initial], "frozen_data_identity": frozen_data,
        "resources": search_resources})
    archive = {item["workflow_fingerprint"] for item in result.get("finite_archive", []) if item["quality_accepted"]}
    for item in attempts:
        item["finite_archive_member"] = item["workflow_fingerprint"] in archive
    shortlist = select_archive_candidates(attempts, 4)
    picked = []; pick_failures = []
    pick_seconds = min(plan.payload["pick_seconds"], limits.remaining()-3)
    if pick_seconds <= 0:
        raise TimeoutError("Shared deadline exhausted before Vpick")
    pick_deadline = time.perf_counter()+pick_seconds
    pick_ledger = ResourceLedger(StageBudgets(max_seconds=pick_seconds, max_training_steps=0,
        max_stages=16, max_rss_bytes=HARD_RSS))
    for candidate in shortlist:
        try:
            pick_ledger.checkpoint("pick_candidate")
            loaded = load_artifact(root, candidate["artifact"])
            quality = evaluate(loaded, data["images"], data["labels"], data["indices"]["pick"], deadline=pick_deadline)
            picked.append({"canonical_solution_id": candidate["canonical_solution_id"], "artifact": candidate["artifact"],
                           "search": candidate["search"], "pick": quality})
            del loaded; gc.collect()
        except Exception as error:
            pick_failures.append({"canonical_solution_id": candidate["canonical_solution_id"],
                "type": type(error).__name__, "message": str(error)})
        atomic_json(root/"jobs"/(job+".pick.json"), {"candidates": picked, "failures": pick_failures,
            "resources": pick_ledger.evidence()})
    selected = min(picked, key=lambda item: (item["pick"]["cross_entropy"], item["artifact"]["physical_tensor_bytes"],
        item["canonical_solution_id"])) if picked else None
    return selected["artifact"] if selected else None, {"search": jsonable(result), "attempts": attempts,
        "pick_candidates": picked, "pick_failures": pick_failures, "pick_resources": pick_ledger.evidence(),
        "search_resources": search_resources,
        "selected": selected, "context": context,
        "initial_genotypes": [genotype_payload(item) for item in initial], "frozen_data_identity": frozen_data,
        "full_search_and_pick_seconds": time.perf_counter()-start,
        "cache": {"bytes": cache.bytes, "max_bytes": cache.max_bytes, "events": cache.events},
        "random_semantics": "24 unique initial draws; same first four as Evo; identical engine, complete prefix sampler and assessment" if random_panel else "GOLEM evolution from four shared initial draws"}


def run_job(plan, root, job, limits):
    if job not in plan.payload["jobs"]:
        raise ValueError("Undeclared job")
    path = root/"jobs"/(job+".json"); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or (root/"selection.lock.json").exists():
        raise FileExistsError("Jobs cannot be replaced or added after selection freeze")
    result = {"job": job, "manifest_sha256": plan.fingerprint, "status": "started", "started_unix": time.time(),
              "runtime": runtime_info()}
    atomic_json(path, result)
    seconds = min(plan.payload["search_seconds"] if job in SEARCH_JOBS else plan.payload["job_seconds"], limits.remaining()-3)
    if seconds <= 0: raise TimeoutError("Shared deadline exhausted")
    ledger = ResourceLedger(StageBudgets(max_seconds=seconds, max_training_steps=0, max_stages=8192, max_rss_bytes=HARD_RSS))
    start = time.perf_counter()
    try:
        trained, source, data = load_prepared(plan, root)
        result["trained_sha256"] = file_hash(root/"trained.json")
        budget = trained["budget_bytes"]; calibration = features(data["images"], data["indices"]["C"])
        metadata = None; model = None
        if job == "dense32":
            metadata = trained["source"]
        elif job in SEARCH_JOBS:
            metadata, evidence = run_search(plan, root, job, source, data, budget, ledger, seconds-(time.perf_counter()-start), limits)
            result.update(evidence)
        else:
            if job == "static_int8":
                request = QuantizationProgram(QuantizationGraphRequest(tuple(OperatorRecipe(p, recipe("dense")) for p in PATHS),
                    max_peak_bytes=HARD_RSS), backend=BackendStageSpec("static", PATHS, seed=20261027))
            else:
                transforms = ([recipe("ptq4"), recipe("svd", 32), recipe("lplr4", 8)] if job == "manual_hetero" else
                              [recipe("svd", rank) for rank in (64, 32, 4)] if job == "svd" else [recipe("ptq4") for _ in PATHS])
                request = QuantizationGraphRequest(tuple(OperatorRecipe(path, spec) for path, spec in zip(PATHS, transforms)),
                    max_storage_bytes=budget, max_peak_bytes=HARD_RSS)
            compressed = quantize(source, request, calibration=calibration, ledger=ledger)
            if type(compressed) in (ExportVerifiedArtifact, ProgramArtifact):
                model = compressed.model
                result["method_evidence"] = jsonable(compressed.evidence) if hasattr(compressed, "evidence") else {}
            else:
                result["failure"] = jsonable(compressed)
        if model is not None:
            _, metadata = checkpoint(model, root, calibration[:4])
        if metadata is not None:
            result.update(artifact=metadata, budget_bytes=budget, baseline_budget_exempt=job == "dense32",
                within_budget=metadata["physical_tensor_bytes"] <= budget)
            if job in SEARCH_JOBS:
                result["selection"] = result["selected"]["search"]
            else:
                loaded = load_artifact(root, metadata)
                result["selection"] = evaluate(loaded, data["images"], data["labels"], data["indices"]["Vsearch"],
                    deadline=start+seconds)
            result["status"] = (classify_search_artifact(result["search"], metadata) if job in SEARCH_JOBS else
                "succeeded" if result["within_budget"] or job == "dense32" else "infeasible")
            result.update(search_execution_summary(result))
        else:
            result["status"] = "failed"
    except Exception as error:
        result.update(status="failed", exception={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
    result["resources"] = result.get("search_resources", ledger.evidence())
    result["wall_seconds"] = time.perf_counter()-start
    atomic_json(path, jsonable(result))
    return result


def record_abort(plan, root, job, guard_report):
    if job not in plan.payload["jobs"] or (root/"selection.lock.json").exists():
        raise ValueError("Only an unfinished declared job can be finalized")
    receipt_path = Path(guard_report).resolve()
    if not receipt_path.is_relative_to(root.resolve()):
        raise ValueError("Guard report must belong to run root")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") not in ("aborted", "worker_failed"):
        raise ValueError("Failed guardian receipt required")
    path = root/"jobs"/(job+".json")
    previous = json.loads(path.read_text()) if path.exists() else {"job": job, "manifest_sha256": plan.fingerprint}
    if previous.get("status") not in (None, "started"):
        raise ValueError("Terminal job cannot be replaced")
    result = {**previous, "status": "aborted", "guard_receipt": str(receipt_path.relative_to(root)),
        "guard_receipt_sha256": file_hash(receipt_path), "failure": receipt.get("reason")}
    atomic_json(path, result)
    return result


def freeze_selection(plan, root, limits):
    path = root/"selection.lock.json"
    if path.exists(): raise FileExistsError("Selection was already frozen")
    trained, source, data = load_prepared(plan, root)
    records = []
    for job in plan.payload["jobs"]:
        result_path = root/"jobs"/(job+".json")
        record = json.loads(result_path.read_text())
        if record["manifest_sha256"] != plan.fingerprint or record["status"] == "started":
            raise ValueError("Every declared job needs a matching terminal record")
        item = {"job": job, "status": record["status"], "result_sha256": file_hash(result_path), "artifact": record.get("artifact")}
        if item["artifact"] is not None:
            model = load_artifact(root, item["artifact"])
            # Use actual inputs and the closed-test batch size, including decode.
            probe_ids = data["indices"]["C"][:128]
            with torch.inference_mode(): model(features(data["images"], probe_ids))
            start = time.perf_counter()
            with torch.inference_mode():
                for _ in range(2): model(features(data["images"], probe_ids))
            predicted = (time.perf_counter()-start)/2*math.ceil(10000/128)
            item["preflight"] = {"batch_rows": 128, "timed_batches": 2, "predicted_full_seconds": predicted,
                "cap_seconds": plan.payload["test_model_seconds"],
                "eligible": predicted*plan.payload["preflight_safety_factor"] <= plan.payload["test_model_seconds"]}
            if not item["preflight"]["eligible"]:
                item["ineligible_reason"] = "full_test_inference_cost"
            del model; gc.collect()
        records.append(item)
        if limits.remaining() <= 5: raise TimeoutError("Shared deadline exhausted during test preflight")
    value = {"manifest_sha256": plan.fingerprint, "trained_sha256": file_hash(root/"trained.json"),
        "prepared_sha256": file_hash(root/"prepared.json"), "code_lock_sha256": verify_code(root),
        "frozen_unix": time.time(), "jobs": records, "test_rows": 10000}
    atomic_json(path, value)
    return value


def closed_test(plan, root, limits):
    verify_code(root)
    frozen = json.loads((root/"selection.lock.json").read_text())
    if (frozen["manifest_sha256"] != plan.fingerprint or frozen["trained_sha256"] != file_hash(root/"trained.json")
            or frozen["prepared_sha256"] != file_hash(root/"prepared.json")):
        raise ValueError("Frozen selection provenance changed")
    for record in frozen["jobs"]:
        if file_hash(root/"jobs"/(record["job"]+".json")) != record["result_sha256"]:
            raise ValueError("Job result changed after selection lock")
    marker = root/"CLOSED_TEST_OPENED.json"
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"opened_unix": time.time(), "selection_sha256": file_hash(root/"selection.lock.json")}, stream)
    # This is the only entry point that can decode test images or labels.
    images = read_idx(verified_raw(plan, root, RAW_FILES[2], allow_test=True), images=True, expected_rows=10000)
    labels = read_idx(verified_raw(plan, root, RAW_FILES[3], allow_test=True), images=False, expected_rows=10000)
    ids = torch.arange(10000); baseline = None
    result = {"manifest_sha256": plan.fingerprint, "selection_sha256": file_hash(root/"selection.lock.json"),
        "images_sha256": tensor_hash(images), "labels_sha256": tensor_hash(labels),
        "test_ids_sha256": tensor_hash(ids), "class_counts": class_counts(labels, ids), "jobs": {}, "status": "partial",
        "runtime": runtime_info()}
    atomic_json(root/"closed_test.json", result)
    for record in frozen["jobs"]:
        if record["artifact"] is None:
            result["jobs"][record["job"]] = {"status": record["status"]}
        elif not record["preflight"]["eligible"]:
            result["jobs"][record["job"]] = {"status": "ineligible", "reason": "full_test_inference_cost"}
        else:
            try:
                model = load_artifact(root, record["artifact"])
                start = time.perf_counter()
                quality = evaluate(model, images, labels, ids, reference_predictions=baseline, retain_predictions=True,
                    deadline=start+min(plan.payload["test_model_seconds"], limits.remaining()-2))
                quality.update(status="succeeded", wall_seconds=time.perf_counter()-start)
                result["jobs"][record["job"]] = quality
                if record["job"] == "dense32": baseline = quality["predictions"]
                del model; gc.collect()
            except Exception as error:
                result["jobs"][record["job"]] = {"status": "failed", "type": type(error).__name__, "message": str(error)}
        atomic_json(root/"closed_test.json", result)
        if limits.remaining() <= 2: break
    result["status"] = "complete" if len(result["jobs"]) == len(frozen["jobs"]) else "partial"
    atomic_json(root/"closed_test.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("manifest", "prepare", "train", "job", "record-abort", "freeze", "closed-test", "analyze"))
    parser.add_argument("--manifest", required=True); parser.add_argument("--run-root", required=True); parser.add_argument("--limits", required=True)
    parser.add_argument("--source-commit"); parser.add_argument("--job"); parser.add_argument("--guard-report")
    args = parser.parse_args(); root = Path(args.run_root).resolve(); require_guard(root)
    limits = Limits.read(args.limits)
    if limits.remaining() <= 0: raise TimeoutError("Shared absolute deadline exhausted")
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    if args.command == "manifest":
        hashes = json.loads((root/"data"/"raw"/"SHA256.json").read_text(encoding="utf-8-sig"))
        plan = FashionPlan.parse(default_manifest(args.source_commit, hashes))
        if Path(args.manifest).exists() or (root/"code.lock.json").exists():
            raise FileExistsError("Protocol freeze cannot be replaced")
        atomic_json(args.manifest, plan.payload)
        atomic_json(root/"code.lock.json", {"source_commit": args.source_commit, "files": source_code_hashes()})
        result = {"manifest_sha256": plan.fingerprint, "code_lock_sha256": verify_code(root)}
    else:
        plan = FashionPlan.read(args.manifest)
        if args.command == "prepare": result = prepare(plan, root)
        elif args.command == "train": result = train(plan, root, limits)
        elif args.command == "job": result = run_job(plan, root, args.job, limits)
        elif args.command == "record-abort": result = record_abort(plan, root, args.job, args.guard_report)
        elif args.command == "freeze": result = freeze_selection(plan, root, limits)
        elif args.command == "closed-test": result = closed_test(plan, root, limits)
        else:
            from .fashion_analysis import analyze
            result = analyze(plan, root)
    print(json.dumps(jsonable(result), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
