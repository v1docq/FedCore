"""Prospective seed-42 local pilot: timing, Conv diagnostics and recovery.

The immutable protocol is written after code validation and before transforms.
The ordinary FedCore runner owns transforms and Adam.  This module owns the
finite plan, validation-only policy decisions, process budget and raw archive.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedcore.tools.atomic_json import write_atomic_json
from fedcore.tools.export import export_model
from . import measurement, runner, scenarios, svd_comparison as historical
from .protocol import (CandidateSpec, ExperimentProtocol, ProtocolError, TensorSplit,
                       role_content_identity, stable_hash, tensor_hash)
from .svd_local_timing import background_snapshot, timing_worker

MODULE = "fedcore.experiments.svd_local_package"
SEED = 42
CPU_THREADS = 8
RSS_LIMIT = 12*1024**3
OS_RESERVE = 6*1024**3
GPU_LIMIT = 14*1024**3
COMPUTE_SECONDS = 8*3600
HISTORICAL_CASES = ("33b7f7fce9b5ae84", "5567f927693d1a20", "d3b23a546f08c955", "6cd867aa41f7cb95")
CONV_PATHS = ("features.0", "features.2")
OBSERVED_PATHS = (*CONV_PATHS, "classifier.0")


@dataclass(frozen=True)
class TimingSession:
    session_id: str
    kind: str
    subject: str
    device: str
    batch: int
    block: int
    side: str
    order: int


@dataclass(frozen=True)
class TransformSpec:
    case_id: str
    method: str
    path: str
    rank: int
    calibration_n: int = 180
    alpha: float = 0.
    nullspace: str = "support_only"
    first_rank: int | None = None
    linear_rank: int | None = None
    statistics_graph: str = "original"


def timing_schedule(seed=SEED):
    """132 sessions; AB/BA counts differ by at most one in each 3-block set."""
    rng = random.Random(seed)
    sessions = []
    for device in ("cpu", "cuda:0"):
        for batch in (1, 64):
            for candidate_index, subject in enumerate(HISTORICAL_CASES):
                orders = [0, 1, (candidate_index+batch) % 2]
                rng.shuffle(orders)
                for block, order in enumerate(orders):
                    for position, side in enumerate(("baseline", "candidate") if order == 0 else ("candidate", "baseline")):
                        identity = f"artifact-{subject}-{device.replace(':','')}-b{batch}-k{block}-{side}"
                        sessions.append(TimingSession(identity, "artifact", subject, device, batch, block, side, position))
            for block in range(3):
                if block == 0:
                    representation_order = ["original_dense", "factor_r8", "materialized_r8"]
                    rng.shuffle(representation_order)
                representations = representation_order[block:] + representation_order[:block]
                for position, subject in enumerate(representations):
                    identity = f"conv2-{subject}-{device.replace(':','')}-b{batch}-k{block}"
                    sessions.append(TimingSession(identity, "conv2", subject, device, batch, block, "representation", position))
    if len(sessions) != 132 or len({item.session_id for item in sessions}) != 132:
        raise ProtocolError("Timing schedule count/identity invariant failed")
    return tuple(sessions)


def initial_transform_plan():
    return tuple(TransformSpec(f"single-{method}-{path.replace('.','')}-r{rank}", method, path, rank)
                 for path, ranks in (("features.0", (1, 3, 5)), ("features.2", (4, 8, 12)))
                 for rank in ranks for method in ("svd", "weighted_svd"))


def select_validation_policy(rows):
    """Validation accuracy, CE, N, alpha and ID form a finite total order."""
    successful = [row for row in rows if row.get("status") == "succeeded"]
    if not successful:
        raise ProtocolError("No successful validation policy; do not substitute an unplanned model")
    def key(row):
        quality = row["quality"]["validation"]
        accuracy, ce = quality["value"], quality["cross_entropy"]
        if not all(math.isfinite(value) for value in (accuracy, ce)):
            raise ProtocolError("Policy selection requires finite validation metrics")
        spec = row["spec"]
        return -accuracy, ce, spec["calibration_n"], spec["alpha"], spec["case_id"]
    return min(successful, key=key)


def nested_calibration(bundle, n):
    if n not in (32, 90, 180) or len(bundle.calibration.ids) != 180:
        raise ProtocolError("Only the frozen 32/90/180 prefixes of the historical pool are allowed")
    pool = bundle.calibration
    split = TensorSplit(pool.x[:n], pool.y[:n], pool.ids[:n])
    return replace(bundle, calibration=split)


def ridge_from_inputs(layer, inputs, alpha):
    """lambda=alpha trace(E[xx^T])/d; padded Conv windows are correlated."""
    if type(alpha) not in (float, int) or not math.isfinite(alpha) or alpha < 0:
        raise ProtocolError("A finite nonnegative ridge alpha is required")
    if isinstance(layer, nn.Conv2d):
        rows = nn.functional.unfold(inputs.double(), layer.kernel_size, dilation=layer.dilation,
                                    padding=layer.padding, stride=layer.stride).transpose(1, 2).reshape(-1, math.prod(layer.weight.shape[1:]))
    elif isinstance(layer, nn.Linear):
        rows = inputs.double().reshape(-1, layer.in_features)
    else:
        raise ProtocolError("Ridge scale supports Conv2d and Linear")
    trace = float(rows.square().sum()/len(rows))
    return float(alpha*trace/rows.shape[1]), {"alpha": float(alpha), "second_moment_trace": trace,
            "dimension": rows.shape[1], "images": len(inputs), "correlated_windows": len(rows),
            "window_independence_claim": False, "formula": "lambda = alpha * trace(C) / d"}


def planned_batches(split, updates=400, batch_size=64, seed=SEED):
    """Mirror the runner's DataLoader RNG, with all updates exactly 64 samples."""
    indices = torch.arange(len(split.x))
    loader = DataLoader(TensorDataset(indices), batch_size=batch_size, shuffle=True,
                        drop_last=True, generator=torch.Generator().manual_seed(seed))
    rows = []
    while len(rows) < updates:
        for (batch,) in loader:
            rows.append(batch.tolist())
            if len(rows) == updates:
                break
    return rows


def _cpu_state_hash(model):
    return runner.model_state_hash(copy.deepcopy(model).cpu())


def rng_snapshot():
    return {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


class RecoveryObjective:
    """One Adam lifetime, frozen independent logit teacher and checkpoints."""
    def __init__(self, teacher=None, checkpoint_callback=None, *, alpha=.5, temperature=2.,
                 expected_batch_hashes=()):
        if not (0 <= alpha <= 1 and temperature > 0 and math.isfinite(temperature)):
            raise ProtocolError("Invalid CE/KL objective coefficients")
        self.teacher = teacher
        self.callback = checkpoint_callback
        self.alpha = alpha
        self.temperature = temperature
        self.expected_batches = tuple(expected_batch_hashes)
        self.updates = 0
        self.teacher_forwards = 0
        self.samples = 0
        self.batch_hashes = []
        self.optimizer_identity = None
        self.teacher_initial = None
        self.evidence = {}

    def __enter__(self):
        if self.teacher is not None:
            self.teacher.eval().requires_grad_(False)
            self.teacher_initial = _cpu_state_hash(self.teacher)
        return self

    def __call__(self, model, x, y):
        if self.teacher is model:
            raise ProtocolError("Teacher and student must be independent")
        batch_hash = stable_hash({"x": tensor_hash(x), "y": tensor_hash(y)})
        if self.expected_batches and (self.updates >= len(self.expected_batches) or batch_hash != self.expected_batches[self.updates]):
            raise ProtocolError("Training batch order differs from frozen schedule")
        self.batch_hashes.append(batch_hash)
        self.samples += len(x)
        logits = model(x)
        ce = nn.functional.cross_entropy(logits, y.long())
        if self.teacher is None:
            return ce
        with torch.no_grad():
            target = self.teacher(x)
        self.teacher_forwards += 1
        t = self.temperature
        kl = nn.functional.kl_div((logits/t).log_softmax(-1), (target/t).softmax(-1), reduction="batchmean")
        return (1-self.alpha)*ce + self.alpha*t*t*kl

    def after_step(self, model, optimizer):
        if self.optimizer_identity is None:
            self.optimizer_identity = id(optimizer)
        if id(optimizer) != self.optimizer_identity:
            raise ProtocolError("Recovery must retain one optimizer lifetime")
        if self.teacher is not None:
            teacher_parameters = {id(p) for p in self.teacher.parameters()}
            if any(id(p) in teacher_parameters for group in optimizer.param_groups for p in group["params"]):
                raise ProtocolError("Teacher parameters entered the student optimizer")
        self.updates += 1
        if next(model.parameters()).device.type == "cuda":
            if max(torch.cuda.memory_allocated(),torch.cuda.memory_reserved()) > GPU_LIMIT:
                raise ProtocolError("Approved 14GiB GPU process allocator budget exceeded")
        if self.callback is not None and self.updates in (100, 400):
            self.callback(model, optimizer, self.updates)

    def __exit__(self, kind, error, traceback_):
        unchanged = True
        if self.teacher is not None:
            unchanged = (self.teacher_initial == _cpu_state_hash(self.teacher)
                         and not self.teacher.training
                         and all(p.grad is None and not p.requires_grad for p in self.teacher.parameters()))
            if not unchanged and kind is None:
                raise ProtocolError("Frozen teacher weight/buffer/mode/gradient invariant failed")
        self.evidence = {"objective": "CE" if self.teacher is None else "CE+logit_KL",
                         "alpha": 0 if self.teacher is None else self.alpha, "temperature": self.temperature,
                         "formula": "(1-alpha)*CE + alpha*T^2*KL(teacher || student)",
                         "teacher_forwards": self.teacher_forwards, "visited_samples": self.samples,
                         "teacher_initial_hash": self.teacher_initial, "teacher_unchanged": unchanged,
                         "updates": self.updates, "batch_hashes": self.batch_hashes,
                         "one_continuous_adam": True}
        return False


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _baseline(historical_dir):
    run = _read(Path(historical_dir)/"cnn-seed-42/run.json")
    bundle = scenarios.build_cv_digits(SEED)
    if role_content_identity(bundle.manifest()["roles"]) != role_content_identity(run["data"]["roles"]):
        raise ProtocolError("Historical data IDs/x/y hashes changed")
    checkpoint = run["baseline"]["checkpoint"]
    if measurement.file_hash(checkpoint["path"]) != checkpoint["sha256"]:
        raise ProtocolError("Historical baseline checkpoint changed")
    model = bundle.original_model
    model.load_state_dict(torch.load(checkpoint["path"], map_location="cpu", weights_only=True), strict=True)
    model.eval()
    if runner.model_state_hash(model) != run["baseline_hash"]:
        raise ProtocolError("Historical baseline state identity changed")
    return bundle, model, run


def source_manifest():
    root = Path(__file__).resolve().parents[2]
    paths = list((root/"fedcore").rglob("*.py")) + list((root/"tests/refactoring").glob("test_svd_local_package*.py"))
    hashes = {str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()
              for path in sorted(paths)}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    return {"git_head": revision, "git_lf_source_hashes": hashes, "sha256": stable_hash(hashes),
            "normalization": "CRLF to LF; includes untracked executed modules and package tests"}


def freeze_protocol(output, historical_dir):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/"protocol.json").exists():
        raise ProtocolError("Protocol exists; preserve it and use its declared stage commands")
    torch.set_num_threads(CPU_THREADS)
    bundle, model, run = _baseline(historical_dir)
    state_path = output/"historical-baseline-state.pt"
    shutil.copy2(run["baseline"]["checkpoint"]["path"], state_path)
    initial_rng = output/"initial-rng.pt"
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.save(rng_snapshot(), initial_rng)
    batches = planned_batches(bundle.train)
    batch_hashes = [stable_hash({"x": tensor_hash(bundle.train.x[index]), "y": tensor_hash(bundle.train.y[index])}) for index in batches]
    data_path = output/"data.pt"
    torch.save({role: {"x": getattr(bundle,role).x, "y": getattr(bundle,role).y} for role in ("train","validation","calibration","test")}, data_path)
    historical_artifacts = {"baseline": run["baseline"]["export"]["artifact"]}
    records = {row["case_id"]:row for row in run["records"]}
    historical_artifacts.update({key: records[key]["export"]["artifact"] for key in HISTORICAL_CASES})
    for artifact in historical_artifacts.values():
        if measurement.file_hash(artifact["path"]) != artifact["sha256"]:
            raise ProtocolError("Historical artifact changed")
    outputs = {role: tensor_hash(historical._predictions(model, getattr(bundle,role))) for role in ("validation","test")}
    protocol = {"schema":1,"study":"FedCore seed42 first local package 2026-10-14", "seed":SEED,
                "source":source_manifest(),"environment":runner.environment_manifest(),
                "historical": {"directory":str(Path(historical_dir).resolve()),"run_sha256":measurement.file_hash(Path(historical_dir)/"cnn-seed-42/run.json"),
                               "baseline_hash":run["baseline_hash"],"checkpoint":run["baseline"]["checkpoint"],"artifacts":historical_artifacts,
                               "optimizer_state":"not archived in historical series; recovery uses one new Adam from checkpoint0",
                               "historical_rng":"not archived; new deterministic recovery RNG frozen separately"},
                "data":bundle.manifest(), "development_test":"historical seen test; development data, no independent confirmation claim",
                "input_files":{"baseline": {"path":str(state_path),"sha256":measurement.file_hash(state_path)},
                               "data":{"path":str(data_path),"sha256":measurement.file_hash(data_path)},
                               "rng":{"path":str(initial_rng),"sha256":measurement.file_hash(initial_rng)}},
                "baseline_output_sha256":outputs, "timing_schedule":[asdict(item) for item in timing_schedule()],
                "timing":{"sessions":132,"warmup":100,"individual_repeats":500,"throughput_blocks":5,"calls_per_throughput_block":100,
                          "threads":1,"runtime":"torch.jit","dtype":"float32","matmul_tf32":False,"cudnn_tf32":False,
                          "cudnn_benchmark":False,"cudnn_deterministic":True,"profiler":False,
                          "passes":"synchronized wall forward; CPU-device-CPU full path; CUDA events; separate blocked throughput",
                          "first_call_scope":"first reloaded-graph call after exporter prepared device/context; not cold process/device",
                          "unit":"3 paired blocks; 500 calls are within-session observations","noise_guard":{"maximum_background_cpu_percent":25,
                          "maximum_background_gpu_percent":10,"post_GPU_idle_observation_seconds":1.2},
                          "micro_control":"materialized SAME approximate Conv2 W with same bias; representation control, not compressed budget",
                          "baseline_spread":"report full range of 3 paired-block baseline medians; no final acceleration claim"},
                "transforms":{"maximum_attempts":27,"initial":[asdict(item) for item in initial_transform_plan()],
                              "nested_N":[32,90,180],"N_source":"same existing calibration pool, prefix order unchanged",
                              "ridge_alpha_grid":[1e-6,1e-4,1e-2],"ridge_formula":"alpha*trace(C)/d",
                              "selection_order":["validation accuracy descending","validation CE ascending","N ascending","alpha ascending","case_id lexicographic"],
                              "N_selection":"lambda0 Conv2 r8 only; select before ridge trials",
                              "alpha_selection":"at chosenN include lambda0; best nonzero separately determines otherN diagnostics; no reselection from those diagnostics",
                              "nullspace":"Linear r8 N90/180 preserve_nullspace, retain unsupported rank-overflow without hidden rank increase",
                              "joint_controls":"2 protected-first Conv2+ordinary Linear16; 4 first SVD3 original/current graph weighted Conv2 r8 with/without same Linear16",
                              "main_models":"only second Conv r8 SVD/selected weighted, each with/without ordinary Linear16; first Conv dense"},
                "recovery":{"curves":10,"student_updates_per_curve":400,"total_student_updates":4000,"teacher_batch_forwards":2000,
                            "objectives":["CE","CE+logit_KL"],"alpha":.5,"temperature":2.,"optimizer":"Adam","lr":1e-4,
                            "batch_size":64,"drop_last":True,"checkpoints":[0,100,400],"batch_indices":batches,"batch_hashes":batch_hashes,
                            "batch_order_sha256":stable_hash(batches),"all_live_student_parameters_trainable":True,
                            "teacher":"independent frozen eval baseline; unchanged weight and buffer hash",
                            "latency_after_recovery":"none; export/reload smoke only, outside pilot timing claim"},
                "resources":{"cpu_threads":CPU_THREADS,"tree_rss_bytes":RSS_LIMIT,"os_reserve_bytes":OS_RESERVE,
                             "gpu_process_allocator_bytes":GPU_LIMIT,"minimum_gpu_physical_free_before_training_bytes":2*1024**3,
                             "compute_wall_seconds":COMPUTE_SECONDS,"preflight_charged_seconds":300.,
                             "preflight_scope":"conservative300s allowance covers all recorded prefreeze tests, their failed attempts, imports and baseline verification; not measured scientific wall time",
                             "scope":"all package computational attempts; stages sequential; no cluster"},
                "failure":{"technical_retries_per_job":1,"retain_failed_attempts":True,"noise_invalid_sessions_retained_no_retry":True,
                           "no_seed_replacement":True,"no_rank_dtype_device_substitution":True}}
    protocol["protocol_sha256"] = stable_hash(protocol)
    write_atomic_json(output/"protocol.json", protocol)
    write_atomic_json(output/"ledger.json", {"schema":1,"protocol_sha256":protocol["protocol_sha256"],"attempts":[],"charged_seconds":300.,
                                           "preflight_charged_seconds":300.,"transform_attempts":0})
    return protocol


def verify_protocol(output):
    protocol = _read(Path(output)/"protocol.json")
    expected = protocol.pop("protocol_sha256")
    if stable_hash(protocol) != expected:
        raise ProtocolError("Frozen protocol was modified")
    protocol["protocol_sha256"] = expected
    if source_manifest()["sha256"] != protocol["source"]["sha256"]:
        raise ProtocolError("Source changed after freeze; preserve protocol and explicitly document lineage")
    for item in protocol["input_files"].values():
        if measurement.file_hash(item["path"]) != item["sha256"]:
            raise ProtocolError("Frozen package input changed")
    return protocol


def _transform(baseline, bundle, spec):
    selected_bundle = nested_calibration(bundle,spec.calibration_n)
    source = baseline
    parts = []
    if spec.first_rank is not None:
        source, evidence = runner.apply_candidate(baseline,selected_bundle,ExperimentProtocol(seed=SEED,batch_size=64,finetune_epochs=0),
            CandidateSpec("svd",{"target_paths":["features.0"],"rank":spec.first_rank,"finetune_epochs":0}))
        parts.append(evidence)
    statistics_source = baseline if spec.statistics_graph == "original" else source
    if spec.method == "svd":
        candidate = CandidateSpec("svd",{"target_paths":[spec.path],"rank":spec.rank,"finetune_epochs":0})
        transformed, evidence = runner.apply_candidate(source,selected_bundle,ExperimentProtocol(seed=SEED,batch_size=64,finetune_epochs=0),candidate)
        ridge = None
    else:
        inputs = historical._reference_observations(statistics_source, selected_bundle.calibration,(spec.path,))[spec.path][0]
        ridge, scale = ridge_from_inputs(statistics_source.get_submodule(spec.path),inputs,spec.alpha)
        candidate = CandidateSpec("weighted_svd",{"target_paths":[spec.path],"rank":spec.rank,"ridge":ridge,
                                  "nullspace_policy":spec.nullspace,"rcond":None,"finetune_epochs":0,
                                  "max_peak_bytes":RSS_LIMIT,"max_workspace_bytes":256*1024**2})
        transformed, evidence = runner.apply_candidate(statistics_source, selected_bundle, ExperimentProtocol(seed=SEED,batch_size=64,finetune_epochs=0),candidate)
        evidence["ridge_scale"] = scale
        if source is not statistics_source:
            # Collect/solve on the original graph, then install its actual
            # second-layer factors into the already changed first-layer graph.
            combined = copy.deepcopy(source)
            parent, name = spec.path.rsplit(".",1)
            combined.get_submodule(parent)._modules[name] = copy.deepcopy(transformed.get_submodule(spec.path))
            transformed = combined
    parts.append(evidence)
    if spec.linear_rank is not None:
        transformed, evidence = runner.apply_candidate(transformed,selected_bundle,ExperimentProtocol(seed=SEED,batch_size=64,finetune_epochs=0),
            CandidateSpec("svd",{"target_paths":["classifier.0"],"rank":spec.linear_rank,"finetune_epochs":0}))
        parts.append(evidence)
    return transformed, {"steps":parts,"lambda":ridge,"calibration":selected_bundle.calibration.manifest(),
                         "statistics_graph":runner.model_state_hash(statistics_source)}


def analytic_macs(model):
    """Analytic multiply-accumulates, excluding biases/activations/transfers."""
    total, operators = 0, {}
    from fedcore.models.network_impl.decomposed_layers import IDecomposed
    for path, layer in model.named_modules():
        if type(layer) is nn.Linear or isinstance(layer, IDecomposed) and isinstance(layer,nn.Linear):
            work = layer.in_features*layer.out_features if type(layer) is nn.Linear else layer.U.shape[-1]*(layer.in_features+layer.out_features)
        elif type(layer) is nn.Conv2d or isinstance(layer, IDecomposed) and isinstance(layer,nn.Conv2d):
            width = layer.in_channels*math.prod(layer.kernel_size)
            work = 64*(width*layer.out_channels if type(layer) is nn.Conv2d else layer.U.shape[-1]*(width+layer.out_channels))
        else:
            continue
        operators[path] = work
        total += work
    return {"per_sample":total,"operators":operators,"scope":"8x8 Digits; analytic MACs, not measured runtime"}


def factor_signature(model):
    from fedcore.models.network_impl.decomposed_layers import IDecomposed
    return {path: layer.representation_metadata() for path,layer in model.named_modules() if isinstance(layer,IDecomposed)}


def export_checkpoint(model, directory, bundle, baseline_outputs, *, optimizer=None, update=0):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    local = copy.deepcopy(model).cpu().eval()
    example = bundle.validation.x[:64]
    state = directory/"state_dict.pt"
    torch.save(local.state_dict(),state)
    template = directory/"architecture.pt"
    torch.save(local, template)
    archive = directory/"training-state.pt"
    torch.save({"update":update,"optimizer":optimizer.state_dict() if optimizer is not None else None,
                "rng":rng_snapshot(),"model_sha256":runner.model_state_hash(local)},archive)
    start = time.perf_counter()
    artifact_path = export_model(local,"torchscript",directory/"model",example)
    artifact = {"path":str(artifact_path.resolve()),"sha256":measurement.file_hash(artifact_path),"format":"torchscript","device":"cpu","threads":1}
    loaded = measurement.load_artifact(artifact)
    quality, parity = {}, {}
    for role in ("validation","test"):
        split = getattr(bundle,role)
        expected = historical._predictions(local,split)
        with torch.inference_mode():
            actual = torch.cat([loaded(batch) for batch in split.x.split(64)])
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
        quality[role] = runner._quality(actual,split,bundle.task)
        quality[role]["logit_error"] = historical.direct_error(baseline_outputs[role],actual)
        parity[role] = historical.direct_error(expected,actual)
    return {"update":update,"model_hash":runner.model_state_hash(local),"quality":quality,
            "storage":historical.exact_storage(local),"factors":factor_signature(local),"macs":analytic_macs(local),
            "checkpoint":{"path":str(state),"sha256":measurement.file_hash(state)},
            "architecture":{"path":str(template),"sha256":measurement.file_hash(template),
                            "scope":"trusted own-package architecture with weights; restore without recomputing transforms"},
            "training_state":{"path":str(archive),"sha256":measurement.file_hash(archive)},
            "artifact":artifact,"artifact_file_bytes":artifact_path.stat().st_size,"export_reload_parity":parity,
            "export_smoke_seconds":time.perf_counter()-start,"latency_measurement":False,
            "test_role_label":"seen historical development test"}


def transforms_worker(output):
    output = Path(output)
    protocol = verify_protocol(output)
    if _read(output/"ledger.json")["transform_attempts"] or (output/"transforms-progress.json").exists():
        raise ProtocolError("A started transform packet cannot be retried or overwritten; partial attempts count against27")
    torch.set_num_threads(CPU_THREADS)
    bundle, baseline, _ = _baseline(protocol["historical"]["directory"])
    before_hash = runner.model_state_hash(baseline)
    baseline_outputs = {role:historical._predictions(baseline,getattr(bundle,role)) for role in ("validation","test")}
    observations = {role:historical._reference_observations(baseline,getattr(bundle,role),OBSERVED_PATHS) for role in ("validation","test")}
    rows, models = [], {}
    def attempt(spec, build=None):
        ledger = _read(output/"ledger.json")
        if ledger["transform_attempts"] >= 27:
            raise ProtocolError("Transform attempt cap reached")
        directory = output/"transforms"/spec.case_id
        if directory.exists():
            raise ProtocolError("Prior scientific transform outputs must be retained")
        ledger["transform_attempts"] += 1
        write_atomic_json(output/"ledger.json",ledger)
        start = time.perf_counter()
        row = {"spec":asdict(spec),"status":"started","attempt_number":len(rows)+1}
        rows.append(row)
        write_atomic_json(output/"transforms-progress.json",{"rows":rows})
        try:
            model, evidence = _transform(baseline,bundle,spec) if build is None else build()
            row.update(export_checkpoint(model,directory,bundle,baseline_outputs),transform=evidence,status="succeeded")
            ranks = {spec.path:spec.rank}
            if spec.first_rank is not None:
                ranks["features.0"] = spec.first_rank
            if spec.linear_rank is not None:
                ranks["classifier.0"] = spec.linear_rank
            row["rank_budget"] = historical.expected_factor_storage(baseline,ranks)
            row["deployed_ranks"] = historical.deployed_ranks(model,ranks)
            if row["storage"]["tensor_bytes"] != row["rank_budget"]["whole_model_tensor_byte_budget"]:
                raise ProtocolError("Actual complete factor storage differs from declared rank budget")
            row["isolated_error"] = {role:historical.isolated_errors(model,observed) for role,observed in observations.items()}
            row["propagated_error"] = {role:historical.propagated_errors(model,getattr(bundle,role),observed) for role,observed in observations.items()}
            models[spec.case_id] = model
        except Exception as error:
            row.update(status="unsupported" if isinstance(error,(ValueError,runner.UnsupportedOperation)) else "failed",
                       error_type=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
        row["transform_wall_seconds"] = time.perf_counter()-start
        directory.mkdir(parents=True,exist_ok=True)
        write_atomic_json(directory/"result.json",row)
        write_atomic_json(output/"transforms-progress.json",{"rows":rows})
        if runner.model_state_hash(baseline) != before_hash:
            raise ProtocolError("Baseline mutated during transform")
        return row
    for spec in initial_transform_plan():
        attempt(spec)
    base_weighted = next(row for row in rows if row["spec"]["case_id"] == "single-weighted_svd-features2-r8")
    calibration_rows = [base_weighted]
    for n in (32,90):
        calibration_rows.append(attempt(TransformSpec(f"conv2-weighted-r8-N{n}","weighted_svd","features.2",8,n)))
    selected_n = select_validation_policy(calibration_rows)["spec"]["calibration_n"]
    ridge_rows = [next(row for row in calibration_rows if row["spec"]["calibration_n"]==selected_n)]
    for alpha in (1e-6,1e-4,1e-2):
        ridge_rows.append(attempt(TransformSpec(f"conv2-weighted-r8-N{selected_n}-a{alpha:g}","weighted_svd","features.2",8,selected_n,alpha)))
    chosen = select_validation_policy(ridge_rows)
    chosen_nonzero = select_validation_policy(ridge_rows[1:])
    for n in (32,90,180):
        if n != selected_n:
            attempt(TransformSpec(f"conv2-weighted-r8-N{n}-a{chosen_nonzero['spec']['alpha']:g}","weighted_svd","features.2",8,n,chosen_nonzero["spec"]["alpha"]))
    for n in (90,180):
        attempt(TransformSpec(f"linear-nullspace-r8-N{n}","weighted_svd","classifier.0",8,n,0.,"preserve_nullspace"))
    selected_spec = TransformSpec(**chosen["spec"])
    main = {"conv2_svd": "single-svd-features2-r8", "conv2_weighted":chosen["spec"]["case_id"]}
    for name, key in tuple(main.items()):
        source = models[key]
        spec = replace(selected_spec if name.endswith("weighted") else TransformSpec("","svd","features.2",8),
                       case_id=f"{name}-linear16",linear_rank=16)
        def add_linear(source=source):
            return runner.apply_candidate(source,bundle,ExperimentProtocol(seed=SEED,batch_size=64,finetune_epochs=0),
                       CandidateSpec("svd",{"target_paths":["classifier.0"],"rank":16,"finetune_epochs":0}))
        result = attempt(spec,add_linear)
        if result["status"]=="succeeded":
            main[name+"_linear16"] = spec.case_id
    for statistics_graph in ("original","current"):
        for linear in (None,16):
            attempt(replace(selected_spec,case_id=f"order-first3-{statistics_graph}-linear{linear or 'dense'}",
                            first_rank=3,linear_rank=linear,statistics_graph=statistics_graph))
    if len(rows)!=27:
        raise ProtocolError("Declared transform count differs")
    # Conv2 representation control uses the very same r8 factor approximation.
    from fedcore.algorithm.low_rank.reassembly.decomposed_recreation import to_standard_module
    factor = models["single-svd-features2-r8"].get_submodule("features.2")
    dense = to_standard_module(factor)
    inputs = observations["validation"]["features.2"][0][:64]
    micro = output/"micro-sources"
    micro.mkdir(parents=True,exist_ok=True)
    input_path = micro/"input.pt"
    torch.save(inputs,input_path)
    micro_artifacts = {}
    with torch.inference_mode():
        torch.testing.assert_close(factor(inputs),dense(inputs),rtol=1e-4,atol=1e-5)
        for name, layer in (("original_dense",baseline.get_submodule("features.2")),("factor_r8",factor),("materialized_r8",dense)):
            path = export_model(layer,"torchscript",micro/name,inputs)
            micro_artifacts[name] = {"path":str(path.resolve()),"sha256":measurement.file_hash(path),"tensor_bytes":measurement.tensor_state_bytes(layer)}
    summary = {"schema":1,"protocol_sha256":protocol["protocol_sha256"],"rows":rows,"transform_attempts":len(rows),
               "selected_N":selected_n,"selected_weighted":chosen["spec"],"selected_nonzero_diagnostic":chosen_nonzero["spec"],
               "main_models":main,"micro_sources":micro_artifacts,"micro_input":{"path":str(input_path.resolve()),"sha256":measurement.file_hash(input_path)},
               "micro_representation_parity":True,"baseline_unchanged":True}
    write_atomic_json(output/"transforms.json",summary)
    return summary


def recovery_worker(output, name, objective_name):
    output = Path(output)
    protocol = verify_protocol(output)
    torch.set_num_threads(CPU_THREADS)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise ProtocolError("Approved GPU recovery unavailable; no implicit CPU fallback")
    bundle, baseline, _ = _baseline(protocol["historical"]["directory"])
    transforms = _read(output/"transforms.json")
    if name == "dense":
        student = copy.deepcopy(baseline)
        spec = None
    else:
        key = transforms["main_models"][name]
        record = next(row for row in transforms["rows"] if row["spec"]["case_id"]==key)
        spec = TransformSpec(**record["spec"])
        architecture = record["architecture"]
        if measurement.file_hash(architecture["path"]) != architecture["sha256"]:
            raise ProtocolError("Trusted compressed architecture archive changed")
        student = torch.load(architecture["path"],map_location="cpu",weights_only=False)
        if measurement.file_hash(record["checkpoint"]["path"]) != record["checkpoint"]["sha256"]:
            raise ProtocolError("Compressed start checkpoint changed")
        student.load_state_dict(torch.load(record["checkpoint"]["path"],map_location="cpu",weights_only=True),strict=True)
        if runner.model_state_hash(student) != record["model_hash"]:
            raise ProtocolError("Compressed recovery start differs")
    directory = output/"recovery"/f"{name}-{objective_name}"
    if directory.exists() and any(directory.iterdir()):
        raise ProtocolError("Started recovery curve cannot be retried or overwrite checkpoints")
    directory.mkdir(parents=True,exist_ok=True)
    baseline_outputs = {role:historical._predictions(baseline,getattr(bundle,role)) for role in ("validation","test")}
    signature = factor_signature(student)
    ranks = {path:metadata["shapes"]["U"][-1] for path,metadata in signature.items()}
    state0 = runner.model_state_hash(student)
    checkpoints = []
    def checkpoint(model,optimizer,update):
        mode = model.training
        # Export validation must preserve training RNG and all live ranks.
        with torch.random.fork_rng(devices=[0]):
            row = export_checkpoint(model,directory/f"update-{update}",bundle,baseline_outputs,optimizer=optimizer,update=update)
        model.train(mode)
        if row["factors"] != signature:
            raise ProtocolError("Recovery changed graph or stored ranks")
        row["deployed_ranks"] = historical.deployed_ranks(copy.deepcopy(model).cpu(),ranks)
        checkpoints.append(row)
        write_atomic_json(directory/"checkpoints.json",{"checkpoints":checkpoints})
    teacher = copy.deepcopy(baseline).to("cuda:0").eval().requires_grad_(False) if objective_name=="ce_kl" else None
    student.requires_grad_(True)
    student.to("cuda:0")
    free_bytes,total_bytes = torch.cuda.mem_get_info()
    if free_bytes < protocol["resources"]["minimum_gpu_physical_free_before_training_bytes"]:
        raise ProtocolError("Less than 2GiB physical GPU headroom before recovery")
    torch.manual_seed(SEED)
    random.seed(SEED)
    checkpoint(student,None,0)
    objective = RecoveryObjective(teacher,checkpoint,expected_batch_hashes=protocol["recovery"]["batch_hashes"])
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    training = runner.train_model(student,bundle.train,bundle.task,epochs=25,batch_size=64,learning_rate=1e-4,
                                 device="cuda:0",seed=SEED,batch_objective=objective,max_updates=400,drop_last=True)
    torch.cuda.synchronize()
    result = {"name":name,"objective":objective_name,"status":"succeeded","spec":asdict(spec) if spec else None,
              "start_model_hash":state0,"training":training,"checkpoints":checkpoints,
              "training_and_checkpoint_wall_seconds":time.perf_counter()-start,
              "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(),"cuda_peak_reserved_bytes":torch.cuda.max_memory_reserved(),
              "teacher_tensor_bytes":measurement.tensor_state_bytes(teacher) if teacher is not None else 0,
              "physical_gpu_free_before_training_bytes":free_bytes,"physical_gpu_total_bytes":total_bytes,
              "gpu_budget_enforced_each_update":True,
              "teacher_unchanged":objective.evidence["teacher_unchanged"],"factors_unchanged":True,
              "role_scope":"validation + seen development test; no independent test claim"}
    if training["training_steps"]!=400 or objective.teacher_forwards != (400 if teacher is not None else 0):
        raise ProtocolError("Recovery update/teacher-forward count violated")
    write_atomic_json(directory/"result.json",result)
    return result


def monitored_job(output, job_id, arguments):
    """Own only this subprocess tree; charge every attempt, including failures."""
    import psutil
    output = Path(output)
    protocol = verify_protocol(output)
    ledger = _read(output/"ledger.json")
    if job_id=="transforms" and ledger["transform_attempts"]:
        raise ProtocolError("No retry of an already started transform packet; retain every attempt within27")
    if job_id.startswith("recovery-"):
        science = output/"recovery"/job_id.removeprefix("recovery-")
        if science.exists() and any(science.iterdir()):
            raise ProtocolError("No retry of started recovery; retained partial checkpoints must not be overwritten")
    remaining = COMPUTE_SECONDS-ledger["charged_seconds"]
    if remaining <= 0:
        raise ProtocolError("Cumulative 8h package budget exhausted")
    directory = output/"processes"/job_id
    existing = [row for row in ledger["attempts"] if row["job_id"]==job_id]
    if len(existing)>=2:
        raise ProtocolError("At most one retained technical retry is permitted")
    attempt = len(existing)+1
    directory = directory/f"attempt-{attempt}"
    directory.mkdir(parents=True,exist_ok=True)
    cmd = [sys.executable,"-m",MODULE,"--output",str(output),*arguments]
    env = dict(os.environ,OMP_NUM_THREADS=str(CPU_THREADS),MKL_NUM_THREADS=str(CPU_THREADS),
               OPENBLAS_NUM_THREADS=str(CPU_THREADS),NUMEXPR_NUM_THREADS=str(CPU_THREADS),PYTHONUTF8="1")
    before = background_snapshot()
    if before["available_ram_bytes"] < OS_RESERVE:
        raise ProtocolError("Less than 6GiB available RAM before worker start")
    samples, violation = [], None
    start = time.perf_counter()
    with (directory/"stdout.log").open("wb") as stdout, (directory/"stderr.log").open("wb") as stderr:
        process = subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env=env)
        owned = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                tree = historical._owned_process_sample(owned)
                available = psutil.virtual_memory().available
                rss = sum(row["rss_bytes"] for row in tree)
                sample = {"seconds":time.perf_counter()-start,"tree_sum_rss_bytes":rss,"available_ram_bytes":available,"owned_processes":tree}
                samples.append(sample)
                if rss>RSS_LIMIT:
                    violation = "recursive RSS exceeds 12GiB"
                elif available<OS_RESERVE:
                    violation = "OS available reserve below 6GiB"
                elif time.perf_counter()-start>=remaining:
                    violation = "cumulative 8h computational budget reached"
                if violation:
                    historical._terminate_owned_tree(owned)
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(.2)
        code = process.wait(timeout=30)
    seconds = time.perf_counter()-start
    record = {"job_id":job_id,"attempt":attempt,"command":cmd,"pid":process.pid,"returncode":code,"violation":violation,
              "elapsed_seconds":seconds,"maximum_recursive_rss_bytes":max((s["tree_sum_rss_bytes"] for s in samples),default=0),
              "background_before":before,"samples":samples,"directory":str(directory)}
    write_atomic_json(directory/"resources.json",record)
    # The transform worker updates its scientific attempt counter while this
    # parent monitors it.  Read the latest ledger rather than losing that field.
    ledger = _read(output/"ledger.json")
    ledger["attempts"].append({key:value for key,value in record.items() if key!="samples"})
    ledger["charged_seconds"] += seconds
    write_atomic_json(output/"ledger.json",ledger)
    if code!=0 or violation:
        raise RuntimeError(f"Retained computational attempt failed: {job_id}, OS code {code}, {violation}")
    return record


def run_timing(output):
    output = Path(output)
    protocol = verify_protocol(output)
    transforms = _read(output/"transforms.json")
    model_input = output/"model-timing-input.pt"
    data = torch.load(output/"data.pt",map_location="cpu",weights_only=True)
    if not model_input.exists():
        torch.save(data["validation"]["x"][:64],model_input)
    results = []
    for session in protocol["timing_schedule"]:
        directory = output/"timing"/session["session_id"]
        result_path = directory/"result.json"
        source = (protocol["historical"]["artifacts"]["baseline" if session["side"]=="baseline" else session["subject"]]
                  if session["kind"]=="artifact" else transforms["micro_sources"][session["subject"]])
        input_record = {"path":str(model_input),"sha256":measurement.file_hash(model_input)} if session["kind"]=="artifact" else transforms["micro_input"]
        job = dict(session,source_artifact=source["path"],source_artifact_sha256=source["sha256"],input_path=input_record["path"],
                   input_file_sha256=input_record["sha256"],directory=str(directory),repeats=500,warmup=100,gpu_allocator_limit_bytes=GPU_LIMIT)
        if result_path.exists():
            prior = _read(result_path)
            if stable_hash(prior["job"]) != stable_hash(job) or prior["source_artifact_sha256"] != source["sha256"]:
                raise ProtocolError("Resumed timing result/job identity differs")
            if measurement.file_hash(prior["artifact"]["path"]) != prior["artifact"]["sha256"]:
                raise ProtocolError("Resumed timing artifact changed")
            results.append(prior)
            continue
        if directory.exists() and (directory/"model.pt").exists():
            raise ProtocolError("Partial timing export retained; retry needs a new explicit attempt directory")
        directory.mkdir(parents=True,exist_ok=True)
        write_atomic_json(directory/"job.json",job)
        monitored_job(output,session["session_id"],["--worker","timing","--job",str(directory/"job.json")])
        results.append(_read(result_path))
        write_atomic_json(output/"timing-progress.json",{"completed":len(results),"planned":132,"valid":sum(r["status"]=="succeeded" for r in results)})
    write_atomic_json(output/"timing.json",{"schema":1,"protocol_sha256":protocol["protocol_sha256"],"sessions":results,
                                           "statistical_unit":"3 paired blocks; no training repetitions","pilot_only":True})


def run_recovery(output):
    output = Path(output)
    names = ("conv2_svd","conv2_weighted","conv2_svd_linear16","conv2_weighted_linear16","dense")
    results = []
    for name in names:
        for objective in ("ce","ce_kl"):
            path = output/"recovery"/f"{name}-{objective}"/"result.json"
            if not path.exists():
                monitored_job(output,f"recovery-{name}-{objective}",["--worker","recovery","--name",name,"--objective",objective])
            row = _read(path)
            if row["name"] != name or row["objective"] != objective or row["training"]["training_steps"]!=400:
                raise ProtocolError("Resumed recovery result identity differs")
            for checkpoint in row["checkpoints"]:
                if measurement.file_hash(checkpoint["checkpoint"]["path"]) != checkpoint["checkpoint"]["sha256"]:
                    raise ProtocolError("Resumed recovery checkpoint changed")
            results.append(row)
            write_atomic_json(output/"recovery-progress.json",{"completed":len(results),"planned":10})
    write_atomic_json(output/"recovery.json",{"schema":1,"curves":results,"student_updates":sum(r["training"]["training_steps"] for r in results),
                                             "teacher_forwards":sum(r["training"]["recovery_objective"]["teacher_forwards"] for r in results)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--historical")
    parser.add_argument("--stage",choices=("freeze","transforms","timing","recovery"))
    parser.add_argument("--worker",choices=("transforms","timing","recovery"))
    parser.add_argument("--job")
    parser.add_argument("--name")
    parser.add_argument("--objective",choices=("ce","ce_kl"))
    args = parser.parse_args()
    if args.worker=="transforms":
        transforms_worker(args.output)
    elif args.worker=="timing":
        verify_protocol(args.output)
        timing_worker(_read(args.job))
    elif args.worker=="recovery":
        recovery_worker(args.output,args.name,args.objective)
    elif args.stage=="freeze":
        if not args.historical:
            parser.error("--historical is required for freeze")
        freeze_protocol(args.output,args.historical)
    elif args.stage=="transforms":
        monitored_job(args.output,"transforms",["--worker","transforms"])
    elif args.stage=="timing":
        run_timing(args.output)
    elif args.stage=="recovery":
        run_recovery(args.output)
    else:
        parser.error("one explicit --stage or --worker required")


if __name__=="__main__":
    main()
