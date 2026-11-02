"""Bounded PR166 CPU pilot worker; run only beneath quantization_pilot_guard.py.

Commands: manifest, prepare, job, freeze, closed-test, timing, analyze.
Only the finite source model and recipes below are accepted; JSON is never code.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass, replace
from enum import Enum
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

# Registry writes made by existing BaseQuantizer remain inside the capped run.
# This happens before any FedCore import, including module execution with -m.
if "--run-root" in sys.argv:
    _root_index = sys.argv.index("--run-root") + 1
    if _root_index < len(sys.argv):
        os.environ["FEDCORE_MODEL_REGISTRY_PATH"] = str(Path(sys.argv[_root_index]).resolve() / "registry")

import psutil
import torch
from torch import nn
from torch.nn import functional as F

from fedcore.experiments.quantization_pilot_protocol import (
    HARD_RSS, PATHS, PilotPlan, canonical_bytes, default_manifest, digest, file_hash, timing_summary, search_execution_summary)
from fedcore.experiments.quantization_pilot_guard import Limits, atomic_json
from fedcore.algorithm.quantization import (
    quantize, QuantizationGraphRequest, OperatorRecipe, TransformSpec, Method,
    QuantizedTensorSpec, Grid, Rounding, ExportVerifiedArtifact, ProgramArtifact,
    QuantizationProgram, BackendStageSpec, ResourceLedger, StageBudgets,
    prepare_quantization_catalogue, allocate_quantization_catalogue, assess_quantization_selection)
from fedcore.algorithm.low_rank.candidate_storage import capture_inventory, inventory_cost
from fedcore.tools.registry.checkpoint_manager import CheckpointManager


def jsonable(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return jsonable(asdict(value))
    if type(value) in (tuple, list):
        return [jsonable(v) for v in value]
    if type(value) is dict:
        return {str(k): jsonable(v) for k, v in value.items()}
    if value is None or type(value) in (str, int, bool, float):
        if type(value) is float and not math.isfinite(value):
            raise ValueError("Nonfinite result cannot enter the experiment record")
        return value
    raise TypeError("Unsupported result metadata: " + type(value).__name__)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(jsonable(value), allow_nan=False) + "\n")
        stream.flush()


def make_source(plan):
    """Controlled fast singular spectrum, frozen teacher; no training claim."""
    g = torch.Generator().manual_seed(plan.payload["model"]["seed"])
    dimensions = plan.payload["model"]["dimensions"]
    layers = []
    for index, (ins, outs) in enumerate(zip(dimensions, dimensions[1:])):
        layer = nn.Linear(ins, outs)
        width = min(ins, outs)
        u = torch.linalg.qr(torch.randn(outs, width, generator=g, dtype=torch.float64), mode="reduced").Q
        v = torch.linalg.qr(torch.randn(ins, width, generator=g, dtype=torch.float64), mode="reduced").Q
        spectrum = torch.exp(-torch.arange(width, dtype=torch.float64) / plan.payload["model"]["singular_decay"])
        with torch.no_grad():
            layer.weight.copy_((u * spectrum) @ v.T)
            layer.bias.copy_(torch.randn(outs, generator=g) * .005)
        layers.append(layer)
        if index < len(dimensions) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers).eval()


def make_inputs(plan, role):
    item = plan.payload["data"][role]
    return torch.randn(item["rows"], plan.payload["model"]["dimensions"][0],
                       generator=torch.Generator().manual_seed(item["seed"]))


def forward(model, x):
    first = next(model.parameters(), None)
    if first is not None and first.dtype is torch.float64:
        x = x.double()
    return model(x)


def evaluate(model, x, target_logits, *, batch_size=128):
    logits = []
    with torch.inference_mode():
        for batch in x.split(batch_size):
            logits.append(forward(model, batch).double())
    actual = torch.cat(logits)
    if actual.shape != target_logits.shape or not bool(torch.isfinite(actual).all()):
        raise ValueError("Loaded model produced invalid logits")
    labels = target_logits.argmax(1)
    return {"cross_entropy": float(F.cross_entropy(actual, labels)),
            "teacher_agreement": float((actual.argmax(1) == labels).double().mean()),
            "logits_mse": float((actual - target_logits.double()).square().mean()),
            "examples": len(x), "meaning": "synthetic frozen-teacher fidelity"}


def checkpoint(model, root, probe):
    manager = CheckpointManager(str(root), auto_cleanup=False)
    started = time.perf_counter()
    blob = manager.serialize_to_bytes(model=model)
    sha = hashlib.sha256(blob).hexdigest()
    path = root / "cas" / (sha + ".pt")
    if not path.exists():
        manager.save_to_file(blob, str(path), cleanup_after_save=False)
    loaded = manager.load_from_file(str(path), device="cpu").eval()
    with torch.inference_mode():
        before, after = forward(model, probe).double(), forward(loaded, probe).double()
    if not bool(torch.isfinite(before).all() & torch.isfinite(after).all()) or not torch.allclose(before, after, rtol=1e-7, atol=1e-9):
        raise ValueError("Checkpoint round-trip changed finite model output")
    original_bytes = inventory_cost(capture_inventory(model), "bytes")
    loaded_bytes = inventory_cost(capture_inventory(loaded), "bytes")
    if original_bytes != loaded_bytes:
        raise ValueError("Checkpoint round-trip changed unique physical storage bytes")
    return loaded, {"sha256": sha, "checkpoint": str(path.relative_to(root)), "file_bytes": len(blob),
                    "physical_tensor_bytes": loaded_bytes, "roundtrip_max_abs": float((before-after).abs().max()),
                    "save_load_seconds": time.perf_counter()-started}


def recipe(method, index=0):
    q4 = QuantizedTensorSpec(4)
    endpoint = QuantizedTensorSpec(4, rounding=Rounding.STOCHASTIC, grid=Grid.ENDPOINT)
    rank = (8, 8, 4)[index]
    if method == "dense": return TransformSpec(Method.DENSE)
    if method == "ptq4": return TransformSpec(Method.PTQ, weight=QuantizedTensorSpec(4, axis=0))
    if method == "svd": return TransformSpec(Method.SVD, rank=rank)
    if method == "dsvd4": return TransformSpec(Method.DSVD, rank=rank*2, left=q4, right=q4)
    if method in ("lplr4", "lplr_svd4"):
        return TransformSpec(Method.LPLR if method == "lplr4" else Method.LPLR_SVD,
                             rank=rank*2, comparison_rank=rank*2, left=endpoint, right=endpoint, seed=37)
    if method == "caldera4":
        return TransformSpec(Method.CALDERA, rank=rank, weight=q4, left=q4, right=q4,
                             outer_iterations=1, inner_iterations=1, seed=37)
    if method == "tender_w4a8": return TransformSpec(Method.TENDER, weight=QuantizedTensorSpec(4, axis=0), activation_bits=8)
    raise ValueError("Unknown finite recipe")


def request(method, budget=None):
    return QuantizationGraphRequest(tuple(OperatorRecipe(path, recipe(method, i)) for i, path in enumerate(PATHS)),
                                    max_storage_bytes=budget, max_peak_bytes=HARD_RSS)


def recover_real_factor_control(packed, batches, ledger):
    """Same decoded factors and LoRA SGD updates as recovery, kept in float."""
    from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear
    from fedcore.algorithm.low_rank.structured_profiles import FactorPair
    from fedcore.algorithm.low_rank.factor_recovery import prepare_factor_lora_stage, finish_factor_lora_stage
    from fedcore.algorithm.low_rank.topology import module_paths, replace_modules_atomically
    replacements = {}
    for path in PATHS:
        layer = module_paths(packed)[path]
        replacements[path] = FactorizedLinear.from_factors(FactorPair(layer.left.decode().float(), layer.right.decode().float(),
                                                       {"method": "explicit_decode_control"}), layer.bias)
    work = replace_modules_atomically(packed, replacements)
    evidence = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(919)
        for factor in ("left", "right"):
            ledger.checkpoint("factor_control_prepare")
            stage = prepare_factor_lora_stage(work, PATHS, factor=factor, rank=1)
            optimizer = torch.optim.SGD(stage.trainable_parameters(), lr=.01)
            stage.validate_optimizer(optimizer)
            for step in range(4):
                ledger.checkpoint("factor_control_train", training_steps=1)
                x, y = batches[step % len(batches)]
                optimizer.zero_grad(); loss = (stage.model(x)-y).square().mean()
                if not bool(torch.isfinite(loss)): raise ValueError("Nonfinite factor control loss")
                loss.backward(); optimizer.step(); ledger.complete_training_step("factor_control_train")
            work, details = finish_factor_lora_stage(stage)
            evidence.append(jsonable(details))
    return work.eval(), evidence


def prepare(plan, root):
    if (root / "prepared.json").exists():
        raise FileExistsError("Preparation cannot overwrite existing provenance")
    source = make_source(plan)
    calibration = make_inputs(plan, "calibration")
    _, source_meta = checkpoint(source, root, calibration[:4])
    data = {}
    for role in ("training", "calibration", "selection"):
        x = make_inputs(plan, role)
        with torch.inference_mode():
            logits = source(x).detach()
        data[role] = {"x": x, "logits": logits}
    path = root / "data.pt"
    torch.save(data, path)
    summary = {"manifest_sha256": plan.fingerprint, "source": source_meta, "data_sha256": file_hash(path),
               "budget_bytes": math.floor(source_meta["physical_tensor_bytes"] * plan.payload["budget_fraction"]),
               "roles": plan.payload["data"], "closed_test_opened": False,
               "torch_version": str(torch.__version__), "python_version": sys.version, "platform": sys.platform,
               "split_provenance": {role: {"features_sha256": hashlib.sha256(item["x"].numpy().tobytes()).hexdigest(),
                                         "logits_sha256": hashlib.sha256(item["logits"].numpy().tobytes()).hexdigest(),
                                         "id": plan.payload["data"][role]["id"]} for role, item in data.items()}}
    atomic_json(root / "prepared.json", summary)
    return summary


def load_prepared(plan, root):
    summary = json.loads((root / "prepared.json").read_text())
    if summary["manifest_sha256"] != plan.fingerprint or file_hash(root / "data.pt") != summary["data_sha256"]:
        raise ValueError("Manifest or data changed after preparation")
    source = load_artifact(root, summary["source"])
    data = torch.load(root / "data.pt", map_location="cpu", weights_only=True)
    return summary, source, data


def load_artifact(root, metadata):
    path = (root / metadata["checkpoint"]).resolve()
    if not path.is_relative_to(root.resolve()) or file_hash(path) != metadata["sha256"]:
        raise ValueError("Artifact path or SHA256 mismatch")
    return CheckpointManager(str(root), auto_cleanup=False).load_from_file(str(path), device="cpu").eval()


def prefix_sample(space, *, seed=0):
    """Uniform per-gene base choices and compatible None/2/4 prefixes."""
    from fedcore.algorithm.quantization.grammar import QuantizationGene, QuantizationGenotype, normalize_genotype
    from fedcore.algorithm.quantization.graph_specs import recipe_graph
    rng = random.Random(seed)
    genes = []
    for domain in space.domains:
        spec = rng.choice(domain.choices)
        if spec.method not in (Method.DENSE, Method.FACTORS, Method.TENDER):
            bits = rng.choice((None,) + space.prequantize_bits)
            spec = replace(spec, prequantize=None if bits is None else QuantizedTensorSpec(bits))
        genes.append(QuantizationGene(domain.aliases, spec, recipe_graph(spec)))
    return normalize_genotype(QuantizationGenotype(tuple(genes)), space)


def unique_initial(space, count, seed):
    from fedcore.algorithm.quantization.grammar import genotype_payload
    result, seen = [], set()
    for attempt in range(1024):
        item = prefix_sample(space, seed=seed+attempt)
        identity = digest(genotype_payload(item))
        if identity not in seen:
            seen.add(identity); result.append(item)
        if len(result) == count:
            return tuple(result)
    raise ValueError("Finite space could not supply unique initial proposals")


def run_search(plan, root, job, source, data, budget, ledger, seconds):
    from fedcore.algorithm.quantization.grammar import (prepare_quantization_space, QUANTIZATION_GRAMMAR_ADAPTER,
        compile_genotype, source_identity, calibration_identity, genotype_payload)
    from fedcore.algorithm.quantization.cache import DependencyCache
    from fedcore.algorithm.low_rank.grammar_search import run_grammar_search, GrammarFitness, GrammarSearchConfig
    from fedcore.algorithm.low_rank.grammar_execution import ResourceFailure
    alternatives = {path: tuple(recipe(method, i) for method in ("dense", "ptq4", "svd", "dsvd4", "lplr4", "lplr_svd4", "caldera4", "tender_w4a8"))
                    for i, path in enumerate(PATHS)}
    calibration = data["calibration"]["x"]
    context = digest({"manifest": plan.fingerprint, "role": "selection", "source": source_identity(source)})
    space = prepare_quantization_space(source, alternatives, calibration=calibration, evaluation_context=context,
                                       max_storage_bytes=budget, max_peak_bytes=HARD_RSS, prequantize_bits=(2, 4), training_choices=(None,))
    if not hasattr(space, "domains"):
        return None, {"space_failure": jsonable(space)}
    cache = DependencyCache(max_bytes=plan.payload["search"]["cache_bytes"])
    assessed = {}; attempts = []
    def data_identity():
        return {role: {key: hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()
                       for key, value in item.items()} for role, item in data.items()}
    frozen_data_identity = data_identity()
    def context_guard():
        if (source_identity(source) != space.source or calibration_identity(calibration).fingerprint != space.calibration
                or data_identity() != frozen_data_identity):
            return ResourceFailure("StaleContext", "Source, calibration, selection or training provenance changed")
        return None
    def assessment(workflow):
        ledger.checkpoint("search_candidate")
        start = time.perf_counter()
        result = quantize(source, workflow.request, calibration=calibration, ledger=ledger, cache=cache)
        record = {"workflow_fingerprint": workflow.fingerprint, "genotype": genotype_payload(workflow.genotype)}
        if type(result) is not ExportVerifiedArtifact:
            record.update(status="failed", failure=jsonable(result), seconds=time.perf_counter()-start)
            attempts.append(record); append_json(root / "jobs" / (job+".attempts.jsonl"), record)
            return result
        loaded, metadata = checkpoint(result.model, root, calibration[:4])
        quality = evaluate(loaded, data["selection"]["x"], data["selection"]["logits"])
        feasible = metadata["physical_tensor_bytes"] <= budget
        record.update(status="succeeded" if feasible else "infeasible", artifact=metadata, selection=quality,
                      seconds=time.perf_counter()-start)
        attempts.append(record); append_json(root / "jobs" / (job+".attempts.jsonl"), record)
        if feasible:
            assessed[workflow.fingerprint] = metadata
        ledger.checkpoint("search_candidate_complete")
        return GrammarFitness(quality["cross_entropy"], float(metadata["physical_tensor_bytes"]), feasible)
    seed = int(job.rsplit("_", 1)[1]); random_panel = job.startswith("random")
    count = 8 if random_panel else 4
    adapter = replace(QUANTIZATION_GRAMMAR_ADAPTER, generate=prefix_sample, context_guard=context_guard)
    config = GrammarSearchConfig(seed=seed, population_size=count, generations=1 if random_panel else 8,
                                 max_evaluations=8, max_seconds=seconds, max_proposals=64)
    start = time.perf_counter()
    result = run_grammar_search(space, config, assessment, fitness_context=context, grammar_adapter=adapter,
                               initial_genotypes=unique_initial(space, count, seed), strict_deadline=True,
                               trace_callback=lambda event: append_json(root / "jobs" / (job+".trace.jsonl"), event))
    archive = [item for item in result.get("finite_archive", []) if item["quality_accepted"] and item["workflow_fingerprint"] in assessed]
    winner = min(archive, key=lambda item: (item["quality"], item["cost"])) if archive else None
    metadata = assessed[winner["workflow_fingerprint"]] if winner else None
    return metadata, {"search": jsonable(result), "attempts": attempts, "selected": winner,
                      "full_search_seconds": time.perf_counter()-start,
                      "cache": {"bytes": cache.bytes, "max_bytes": cache.max_bytes, "events": cache.events},
                      "random_semantics": "same GOLEM assessment, K unique initial draws including prefix grid, no adaptive variation" if random_panel else "GOLEM evolutionary variation, common prefix sampler"}


def run_job(plan, root, job, limits):
    if job not in plan.payload["jobs"]:
        raise ValueError("Undeclared job")
    path = root / "jobs" / (job+".json"); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or (root / "selection.lock.json").exists():
        raise FileExistsError("Jobs cannot be overwritten or added after selection freeze")
    start = time.perf_counter()
    result = {"job": job, "manifest_sha256": plan.fingerprint, "status": "started", "started_unix": time.time()}
    atomic_json(path, result)
    cap = plan.payload["search_seconds"] if job.startswith(("random_", "evo_")) else plan.payload["job_seconds"]
    seconds = min(cap, limits.remaining() - 2)
    if seconds <= 0:
        raise TimeoutError("Shared deadline exhausted")
    ledger = ResourceLedger(StageBudgets(max_seconds=seconds, max_training_steps=8 if job.startswith("recovery_") else 0,
                                         max_stages=4096, max_rss_bytes=HARD_RSS))
    try:
        summary, source, data = load_prepared(plan, root)
        budget = summary["budget_bytes"]; calibration = data["calibration"]["x"]
        metadata = None; artifact = None
        if job in ("dense32", "dense64"):
            artifact = source if job == "dense32" else source.double()
        elif job.startswith(("random_", "evo_")):
            metadata, evidence = run_search(plan, root, job, source, data, budget, ledger, seconds)
            result.update(evidence)
        elif job == "exact_catalogue":
            alternatives = {p: tuple(recipe(m, i) for m in ("dense", "ptq4", "svd")) for i, p in enumerate(PATHS)}
            catalogue = prepare_quantization_catalogue(source, alternatives, calibration=calibration, max_peak_bytes=HARD_RSS,
                                                       max_options=256, max_catalogue_bytes=128*1024**2, ledger=ledger)
            if not hasattr(catalogue, "entries"):
                result["failure"] = jsonable(catalogue)
            else:
                result["catalogue"] = jsonable(catalogue)
                selection = allocate_quantization_catalogue(catalogue, budget)
                assessed = assess_quantization_selection(source, catalogue, selection, calibration=calibration,
                            evaluate=lambda m: evaluate(m, data["selection"]["x"], data["selection"]["logits"])["cross_entropy"],
                            evaluation_context=plan.fingerprint+":selection", ledger=ledger)
                if hasattr(assessed, "artifact"):
                    artifact = assessed.artifact.model
                    result["local_surrogate"] = assessed.local_surrogate
                    result["whole_model_selection_ce"] = assessed.whole_model_quality
                    result["allocation"] = jsonable(selection)
                else:
                    result["failure"] = jsonable(assessed)
        else:
            batches = ()
            if job == "manual_hetero":
                req = QuantizationGraphRequest(tuple(OperatorRecipe(p, recipe(m, i)) for i, (p, m) in enumerate(zip(PATHS, ("ptq4", "svd", "lplr4")))),
                                               max_storage_bytes=budget, max_peak_bytes=HARD_RSS)
            elif job == "static_int8":
                # Intermediate Dense must not be subjected to the final cap.
                req = QuantizationProgram(request("dense"), backend=BackendStageSpec("static", PATHS, seed=37))
            elif job.startswith("recovery_"):
                from fedcore.algorithm.quantization import RecoveryStageSpec, training_identity
                x = data["training"]["x"][:64]; y = data["training"]["logits"][:64]
                batches = ((x, y),)
                base = request("lplr4", budget)
                # Same decoded initial factors, LoRA stages, data and updates;
                # the real-factor control omits final packed re-encoding.
                req = (QuantizationProgram(base, recovery=RecoveryStageSpec(PATHS, steps=4, seed=919), training_fingerprint=training_identity(batches))
                       if job == "recovery_lplr" else base)
                result["training_role"] = "logits MSE; two factor stages x four SGD updates"
            else:
                req = request(job, budget)
            compressed = quantize(source, req, calibration=calibration, training_batches=batches, ledger=ledger)
            if type(compressed) in (ExportVerifiedArtifact, ProgramArtifact):
                artifact = compressed.model; result["method_evidence"] = jsonable(compressed.evidence) if hasattr(compressed, "evidence") else {}
                if job == "recovery_factor_control":
                    artifact, result["factor_control_stages"] = recover_real_factor_control(artifact, batches, ledger)
            else:
                result["failure"] = jsonable(compressed)
        if artifact is not None:
            loaded, metadata = checkpoint(artifact, root, calibration[:4])
        elif metadata is not None:
            loaded = load_artifact(root, metadata)
        if metadata is not None:
            result["artifact"] = metadata
            result["artifact_status"] = "verified_finite_roundtrip"
            result["selection"] = evaluate(loaded, data["selection"]["x"], data["selection"]["logits"])
            result["budget_bytes"] = budget
            result["baseline_budget_exempt"] = job in ("dense32", "dense64")
            result["within_budget"] = metadata["physical_tensor_bytes"] <= budget
            result["status"] = "succeeded" if result["within_budget"] or result["baseline_budget_exempt"] else "infeasible"
            result.update(search_execution_summary(result))
            if result["search_completed"] is False and result["status"] == "succeeded":
                result["status"] = "partial_artifact"
        else:
            result["status"] = "failed"
    except Exception as error:
        result.update(status="failed", exception={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
    result["resources"] = ledger.evidence(); result["wall_seconds"] = time.perf_counter()-start
    atomic_json(path, jsonable(result))
    return result


def freeze_selection(plan, root):
    path = root / "selection.lock.json"
    if path.exists():
        raise FileExistsError("Selection was already frozen")
    records = []
    for job in plan.payload["jobs"]:
        result_path = root / "jobs" / (job+".json")
        if not result_path.exists():
            raise ValueError("Every declared job needs a preserved terminal record: " + job)
        record = json.loads(result_path.read_text())
        if record["status"] == "started":
            raise ValueError("Interrupted jobs need explicit guard-failure finalization before freeze")
        records.append({"job": job, "result_sha256": file_hash(result_path), "status": record["status"],
                        "artifact": record.get("artifact")})
    value = {"manifest_sha256": plan.fingerprint, "frozen_unix": time.time(), "jobs": records}
    atomic_json(path, value)
    return value


def record_abort(plan, root, job, guard_report):
    if job not in plan.payload["jobs"] or (root / "selection.lock.json").exists():
        raise ValueError("Only a declared unfinished job can be finalized")
    receipt_path = Path(guard_report).resolve()
    if not receipt_path.is_relative_to(root.resolve()):
        raise ValueError("Guard receipt must belong to this run")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") not in ("aborted", "worker_failed"):
        raise ValueError("A failed guardian receipt is required")
    path = root / "jobs" / (job+".json")
    previous = json.loads(path.read_text()) if path.exists() else {"job": job, "manifest_sha256": plan.fingerprint}
    if previous.get("status") not in (None, "started"):
        raise ValueError("Terminal job results cannot be replaced")
    result = {**previous, "status": "aborted", "guard_receipt": str(receipt_path.relative_to(root)),
              "guard_receipt_sha256": file_hash(receipt_path), "failure": receipt.get("reason"),
              "guard_wall_seconds": receipt.get("wall_seconds"), "guard_returncode": receipt.get("returncode")}
    atomic_json(path, result)
    return result


def closed_test(plan, root):
    frozen = json.loads((root / "selection.lock.json").read_text())
    if frozen["manifest_sha256"] != plan.fingerprint:
        raise ValueError("Selection manifest changed")
    marker = root / "CLOSED_TEST_OPENED.json"
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"opened_unix": time.time(), "selection_sha256": file_hash(root / "selection.lock.json")}, stream)
    summary = json.loads((root / "prepared.json").read_text())
    source = load_artifact(root, summary["source"])
    x = make_inputs(plan, "closed_test")
    with torch.inference_mode(): target = source(x).detach()
    result = {"manifest_sha256": plan.fingerprint, "selection_sha256": file_hash(root / "selection.lock.json"),
              "features_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(), "jobs": {}}
    del source
    for record in frozen["jobs"]:
        result_path = root / "jobs" / (record["job"]+".json")
        if file_hash(result_path) != record["result_sha256"]:
            raise ValueError("Job result changed after freeze")
        if record["artifact"] is None:
            result["jobs"][record["job"]] = {"status": record["status"]}
        else:
            model = load_artifact(root, record["artifact"])
            result["jobs"][record["job"]] = evaluate(model, x, target)
            del model; gc.collect()
        atomic_json(root / "closed_test.json", result)
    return result


def timing(plan, root, job, batch, threads, repeat, seconds, mode="single"):
    if not (root / "selection.lock.json").exists():
        raise ValueError("Timing requires frozen selection")
    if batch not in (1, 128) or threads not in (1, 8) or repeat not in range(5):
        raise ValueError("Timing regime outside frozen protocol")
    torch.set_num_threads(threads)
    if mode not in ("single", "aa", "ab"):
        raise ValueError("Unknown timing mode")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= plan.payload["job_seconds"]:
        raise ValueError("Timing must fit the frozen per-job cap")
    result_path = root / "timing" / f"{job}-{mode}-b{batch}-t{threads}-r{repeat}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists(): raise FileExistsError("Timing repeats cannot be replaced")
    record = json.loads((root / "jobs" / (job+".json")).read_text())
    import_rss = psutil.Process().memory_info().rss
    start_load = time.perf_counter(); model = load_artifact(root, record["artifact"])
    loaded_rss = psutil.Process().memory_info().rss; load_seconds = time.perf_counter()-start_load
    comparator = None
    if mode in ("aa", "ab"):
        baseline = record if mode == "aa" else json.loads((root / "jobs" / "dense32.json").read_text())
        comparator = load_artifact(root, baseline["artifact"])
    x = torch.randn(batch, 64, generator=torch.Generator().manual_seed(990+batch))
    deadline = time.monotonic() + seconds; observed = []; control_observed = []; pairs = []; warmups = 0
    order_rng = random.Random(1701+repeat)
    with torch.inference_mode():
        cold_start = time.perf_counter(); y = forward(model, x); cold = time.perf_counter()-cold_start
        if not bool(torch.isfinite(y).all()): raise ValueError("Nonfinite timed output")
        if mode == "aa" and not torch.equal(y, forward(comparator, x)):
            raise ValueError("A/A loaded copies disagree")
        for _ in range(20):
            if time.monotonic() >= deadline: break
            forward(model, x)
            if comparator is not None: forward(comparator, x)
            warmups += 1
        if warmups == 20:
            for _ in range(200):
                if time.monotonic() >= deadline: break
                order = ("A", "B") if order_rng.randrange(2) else ("B", "A")
                if comparator is None: order = ("B",)
                pair = {"order": list(order), "started_unix": time.time()}
                for label in order:
                    first = time.perf_counter(); forward(model if label == "B" else comparator, x)
                    elapsed = time.perf_counter()-first
                    (observed if label == "B" else control_observed).append(elapsed)
                    pair[label] = elapsed
                pairs.append(pair)
    result = {"job": job, "mode": mode, "batch": batch, "threads": threads, "repeat": repeat, "warmups_completed": warmups,
              "cold_seconds": cold, "load_seconds": load_seconds, "rss_after_import": import_rss,
              "rss_after_load": loaded_rss, "rss_load_delta": loaded_rss-import_rss,
              "timing_semantics": "single-request wall times, all per-forward decode/prepack included; RSS observer runs in guardian",
              "summary": timing_summary(observed) if observed else {"observed_requests": 0, "p95_eligible": False, "p95_seconds": None}}
    if comparator is not None:
        result["control_summary"] = timing_summary(control_observed) if control_observed else {"observed_requests": 0, "p95_eligible": False}
        result["paired_requests"] = pairs
        if observed and control_observed:
            ratio = result["control_summary"]["p50_seconds"] / result["summary"]["p50_seconds"]
            result["median_time_ratio_A_over_B"] = ratio
            # Order sensitivity is reported, never repaired by convenient reruns.
            order_groups = {}
            for first in ("A", "B"):
                group = [p for p in pairs if p["order"][0] == first]
                if group:
                    from fedcore.experiments.quantization_pilot_protocol import quantile
                    order_groups[first+"_first"] = quantile([p["A"] for p in group], .5) / quantile([p["B"] for p in group], .5)
            result["order_group_ratios"] = order_groups
            if mode == "aa":
                result["aa_passed"] = (.95 <= ratio <= 1.05 and len(observed) == 200
                    and len(order_groups) == 2 and all(.95 <= v <= 1.05 for v in order_groups.values()))
    atomic_json(result_path, result)
    return result


def require_guard(root):
    lock = json.loads((root / ".numerical-worker.lock").read_text())
    parents = {p.pid: p for p in psutil.Process().parents()}
    if lock["pid"] not in parents or parents[lock["pid"]].create_time() != lock["create_time"]:
        raise RuntimeError("Worker must run below the live resource guardian owning this run root")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("manifest", "prepare", "job", "record-abort", "freeze", "closed-test", "timing", "analyze"))
    parser.add_argument("--manifest", required=True); parser.add_argument("--run-root", required=True); parser.add_argument("--limits", required=True)
    parser.add_argument("--job"); parser.add_argument("--source-commit"); parser.add_argument("--include-recovery", action="store_true")
    parser.add_argument("--guard-report")
    parser.add_argument("--batch", type=int, default=1); parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=0); parser.add_argument("--seconds", type=float, default=120.)
    parser.add_argument("--profile-mode", choices=("single", "aa", "ab"), default="single")
    args = parser.parse_args(); root = Path(args.run_root).resolve(); require_guard(root)
    limits = Limits.read(args.limits)
    if limits.remaining() <= 0: raise TimeoutError("Shared absolute deadline exhausted")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if args.command == "manifest":
        plan = PilotPlan.parse(default_manifest(args.source_commit, include_recovery=args.include_recovery))
        if Path(args.manifest).exists(): raise FileExistsError("Manifest cannot be replaced")
        atomic_json(args.manifest, plan.payload); result = {"manifest_sha256": plan.fingerprint}
    else:
        plan = PilotPlan.read(args.manifest)
        if args.command == "prepare": result = prepare(plan, root)
        elif args.command == "job": result = run_job(plan, root, args.job, limits)
        elif args.command == "record-abort": result = record_abort(plan, root, args.job, args.guard_report)
        elif args.command == "freeze": result = freeze_selection(plan, root)
        elif args.command == "closed-test": result = closed_test(plan, root)
        elif args.command == "timing": result = timing(plan, root, args.job, args.batch, args.threads, args.repeat, min(args.seconds, limits.remaining()-2), args.profile_mode)
        else:
            from fedcore.experiments.quantization_pilot_analysis import analyze
            result = analyze(plan, root)
    print(json.dumps(jsonable(result), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
