"""Finite third local packet: exact-factor Conv2 materialization, inference only.

Freeze once after code review. Every profile has its own prospective control
gate; a failed profile is retained and skipped, without replacement or retries.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch
from torch import nn

from fedcore.tools.atomic_json import write_atomic_json
from fedcore.tools.export import export_model
from . import measurement, runner
from .protocol import ProtocolError, stable_hash, tensor_hash
from .svd_comparison import exact_storage
from .svd_second_package import (AdmissionPolicy, checked_file, read_json, recovered_checkpoint,
                                 wait_for_quiet, external_snapshot)
from .svd_third_timing import timing_worker
from fedcore.tools.svd_third_analysis import GRAPHS, control_gate, validate_records

MODULE = "fedcore.experiments.svd_third_package"
PROFILES = (("cpu", 1), ("cpu", 64), ("cuda:0", 1), ("cuda:0", 64))
COMPUTE_SECONDS = 7200.
QUIET_SECONDS = 600.
RSS_LIMIT = 12 * 1024**3
RAM_RESERVE = 6 * 1024**3
GPU_FREE = 2 * 1024**3


def schedule(stage):
    if stage not in ("pilot", "main"):
        raise ProtocolError("Unknown prospective stage")
    count = 8 if stage == "pilot" else 20
    pairs = [("control", name, name, name) for name in GRAPHS]
    if stage == "main":
        pairs += [("comparison", f"{name}_factor_vs_materialized", f"{name}_factor", f"{name}_materialized")
                  for name in ("svd", "weighted")]
        pairs += [("comparison", f"dense_vs_{name}", "dense", name) for name in GRAPHS[:4]]
    units = []
    for block in range(count):
        rotation = block % len(pairs)
        ordered = pairs[rotation:] + pairs[:rotation]
        for position, (kind, subject, left, right) in enumerate(ordered):
            units.append({"unit_id": f"{stage}-{subject}-k{block}", "kind": kind, "subject": subject,
                          "block": block, "position": position,
                          "order": ["a", "b"] if block % 2 == 0 else ["b", "a"],
                          "models": {"a": left, "b": right}})
    return units


def materialize_conv2(model):
    """Return one dense Conv2 made from the same factors; all other tensors kept."""
    from fedcore.algorithm.low_rank.reassembly.decomposed_recreation import to_standard_module
    from fedcore.models.network_impl.decomposed_layers import DecomposedConv2d, DecomposedLinear
    conv = model.get_submodule("features.2")
    if not isinstance(conv, DecomposedConv2d) or conv.U is None or conv.U.shape[-1] != 8:
        raise ProtocolError("Expected retained recovered Conv2 rank8 factors")
    if type(model.get_submodule("features.0")) is not nn.Conv2d:
        raise ProtocolError("Conv1 must remain dense")
    linear = [m for m in model.modules() if isinstance(m, DecomposedLinear)]
    if len(linear) != 1 or linear[0].U is None or linear[0].U.shape[-1] != 16:
        raise ProtocolError("Expected retained Linear16 factors")
    before = runner.model_state_hash(model)
    result = copy.deepcopy(model).eval()
    with torch.random.fork_rng(devices=[]):
        dense = to_standard_module(conv).eval()
    result.features[2] = dense
    for key, value in model.state_dict().items():
        if not key.startswith("features.2.") and not torch.equal(value, result.state_dict()[key]):
            raise ProtocolError("A non-Conv2 tensor changed")
    if before != runner.model_state_hash(model):
        raise ProtocolError("Materialization mutated source model")
    return result


def artifact_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": measurement.file_hash(path), "bytes": path.stat().st_size}


def source_manifest():
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "fedcore").rglob("*.py"))
    paths += list((root / "tests/refactoring").glob("test_svd_third*.py"))
    paths += [root / "scripts/analyze_svd_third_package.py"]
    paths += list((root / "fedcore/experiments").glob("svd_third*.json"))
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
              for p in sorted(paths)}
    return {"sha256": stable_hash(hashes), "lf_source_hashes": hashes,
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                       capture_output=True, check=True).stdout.strip(),
            "normalization": "CRLF to LF; all fedcore Python, third tests/analyzer/spec; docs excluded"}


def prepare_worker(job):
    """CPU-only recovered-model construction and already-known quality parity."""
    from .svd_second_timing import require_admission, precision_profile
    require_admission(job)
    precision_profile(1)
    output, first = Path(job["output"]), Path(job["first"])
    protocol = read_json(first / "protocol.json")
    data_record = protocol["input_files"]["data"]
    checked_file(data_record, root=first)
    data = torch.load(data_record["path"], map_location="cpu", weights_only=True)
    sources = output / "sources"
    sources.mkdir()
    torch.save(data["validation"]["x"][:64].clone(), sources / "input.pt")
    x = data["validation"]["x"][:64]
    graphs, lineage, parity = {}, {}, {}
    for name, recovered_name in (("svd", "conv2_svd_linear16"), ("weighted", "conv2_weighted_linear16"), ("dense", "dense")):
        model, checkpoint = recovered_checkpoint(first, recovered_name)
        lineage[recovered_name] = checkpoint
        variants = {"dense": model} if name == "dense" else {
            f"{name}_factor": model, f"{name}_materialized": materialize_conv2(model)}
        if name != "dense":
            factor, materialized = variants[f"{name}_factor"], variants[f"{name}_materialized"]
            conv, dense_conv = factor.features[2], materialized.features[2]
            left = conv.U.detach() if conv.S is None else conv.U.detach() * conv.S.detach().unsqueeze(-2)
            independent = (left @ conv.Vh.detach()).reshape_as(dense_conv.weight)
            torch.testing.assert_close(independent, dense_conv.weight, rtol=0, atol=0)
            torch.testing.assert_close(conv.bias, dense_conv.bias, rtol=0, atol=0)
            parity[name] = {"composed_weight_sha256": tensor_hash(independent), "materialized_weight_sha256": tensor_hash(dense_conv.weight),
                            "source_model_sha256": runner.model_state_hash(factor), "splits": {}}
        loaded = {}
        for key, variant in variants.items():
            expected_bytes = 274920 if key == "dense" else (82408 if key.endswith("materialized") else 78312)
            storage = exact_storage(variant)
            if storage["tensor_bytes"] != expected_bytes:
                raise ProtocolError("Prospective complete tensor storage differs")
            path = export_model(variant, "torchscript", sources / key, x[:1])
            with path.open("rb") as stream:
                loaded[key] = torch.jit.load(stream, map_location="cpu").eval()
            if measurement.tensor_state_bytes(loaded[key]) != expected_bytes:
                raise ProtocolError("Restored artifact storage differs")
            graphs[key] = {**artifact_record(path), "tensor_bytes": expected_bytes,
                           "model_state_sha256": runner.model_state_hash(variant), "known_quality": {}}
        for split in ("validation", "test"):
            inputs, labels = data[split]["x"], data[split]["y"]
            outputs = {}
            with torch.inference_mode():
                for key, variant in variants.items():
                    eager = torch.cat([variant(part) for part in inputs.split(64)])
                    restored = torch.cat([loaded[key](part) for part in inputs.split(64)])
                    torch.testing.assert_close(restored, eager, rtol=1e-4, atol=1e-5)
                    outputs[key] = restored
                    graphs[key]["known_quality"][split] = {"n": len(labels),
                        "correct": int((restored.argmax(1) == labels).sum()),
                        "accuracy": float((restored.argmax(1) == labels).float().mean()),
                        "cross_entropy": float(torch.nn.functional.cross_entropy(restored, labels)),
                        "eager_restored_max_abs": float((eager - restored).abs().max()),
                        "input_sha256": tensor_hash(inputs), "labels_sha256": tensor_hash(labels)}
            if name != "dense":
                a, b = outputs[f"{name}_factor"], outputs[f"{name}_materialized"]
                torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-5)
                if not torch.equal(a.argmax(1), b.argmax(1)):
                    raise ProtocolError("Known split decisions differ after Conv2 materialization")
                parity[name]["splits"][split] = {"max_abs": float((a - b).abs().max()), "same_decisions": True}
    input_record = artifact_record(sources / "input.pt")
    input_record["tensor_sha256_by_batch"] = {str(batch): tensor_hash(x[:batch]) for batch in (1, 64)}
    result = {"schema": 3, "status": "succeeded", "job": job, "prepared": {
        "graphs": graphs, "input": input_record, "parity": parity, "recovered": lineage,
        "lineage": {"recovery": artifact_record(first / "recovery.json"),
                    "first_protocol": artifact_record(first / "protocol.json"), "data": data_record},
        "new_training_updates": 0, "scope": "known validation/devtest only; numerical parity, no new quality proof"}}
    write_atomic_json(Path(job["directory"]) / "worker-result.json", result)
    return result


def observe_diagnostics():
    """No CUDA API; active utilization includes our worker and is diagnostic."""
    sample = external_snapshot()
    fields = ("pstate", "clocks.current.sm", "clocks.current.memory", "power.draw", "temperature.gpu")
    try:
        raw = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=" + ",".join(fields),
            "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True, timeout=5).stdout.strip()
        sample["diagnostics"] = {key: (None if value.strip() in ("N/A", "[N/A]") else value.strip())
                                  for key, value in zip(fields, raw.split(","))}
    except (OSError, subprocess.SubprocessError) as error:
        sample["diagnostics_error"] = type(error).__name__
    return sample


def stop_owned(owner, worker):
    """Reap the exact worker and its descendants, including parent-exit races."""
    import psutil
    try:
        children = owner.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for process in [*reversed(children), owner]:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(children, timeout=2)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    if worker.poll() is None:
        worker.kill()
    return worker.wait(timeout=30)


def monitored(output, job, ledger):
    """One admitted process, bounded resources, no retry of a started attempt."""
    import psutil
    output, directory = Path(output), Path(job["directory"])
    if directory.exists():
        raise ProtocolError("Attempt directory exists; retries are forbidden")
    directory.mkdir(parents=True)
    write_atomic_json(directory / "job.json", job)
    attempt = {"session_id": job["session_id"], "stage": job["stage"], "status": "admission_pending"}
    ledger["attempts"].append(attempt)
    write_atomic_json(output / "ledger.json", ledger)
    remaining = COMPUTE_SECONDS - ledger["charged_seconds"]
    wait_remaining = QUIET_SECONDS - ledger["extra_noise_wait_seconds"]
    if remaining <= 0 or wait_remaining <= 0:
        result = {"job": job, "status": "not_run_budget", "records": []}
    else:
        admission = wait_for_quiet(job["device"], AdmissionPolicy(timeout_seconds=min(60., wait_remaining)))
        admission.update(session_id=job["session_id"], protocol_sha256=job["protocol_sha256"], before_worker_spawn=True)
        write_atomic_json(directory / "admission.json", admission)
        ledger["admission_wait_seconds"] += admission["elapsed_seconds"]
        ledger["extra_noise_wait_seconds"] += admission["extra_wait_seconds"]
        if admission["status"] != "admitted":
            result = {"job": job, "status": "admission_failed", "records": [], "admission": admission}
        else:
            command = [sys.executable, "-m", MODULE, "--worker-job", str(directory / "job.json")]
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONUTF8="1")
            samples, violation, start = [], None, time.perf_counter()
            with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
                worker = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env)
                attempt.update(status="worker_started", worker_pid=worker.pid, command=command)
                write_atomic_json(output / "ledger.json", ledger)
                owner = psutil.Process(worker.pid)
                try:
                    while worker.poll() is None:
                        tree = [owner, *owner.children(recursive=True)]
                        rss = 0
                        live_pids = []
                        for process in tree:
                            try:
                                rss += process.memory_info().rss
                                live_pids.append(process.pid)
                            except psutil.NoSuchProcess:
                                continue
                        available = psutil.virtual_memory().available
                        sample = {"elapsed_seconds": time.perf_counter() - start, "tree_rss_bytes": rss,
                                  "available_ram_bytes": available, "owned_pids": live_pids}
                        if job["device"].startswith("cuda"):
                            observed = observe_diagnostics()
                            sample["gpu_observation"] = observed
                            sample["utilization_scope"] = "includes owned GPU activity; not an external-noise criterion"
                            gpu = observed.get("gpu")
                            if gpu and gpu["physical_free_mib"] * 1024**2 < GPU_FREE:
                                violation = "physical GPU free below2GiB during owned work"
                        samples.append(sample)
                        if rss > RSS_LIMIT:
                            violation = "owned recursive RSS above12GiB"
                        elif available < RAM_RESERVE:
                            violation = "available RAM below6GiB"
                        elif time.perf_counter() - start >= remaining:
                            violation = "cumulative2h compute budget reached"
                        if violation:
                            break
                        time.sleep(.8 if job["device"].startswith("cuda") else .2)
                except (psutil.Error, OSError, ValueError) as error:
                    violation = "resource observation failed: " + type(error).__name__
                finally:
                    code = stop_owned(owner, worker) if worker.poll() is None else worker.wait(timeout=30)
            elapsed = time.perf_counter() - start
            ledger["charged_seconds"] += elapsed
            ledger["compute_by_stage_seconds"][job["stage"]] = ledger["compute_by_stage_seconds"].get(job["stage"], 0.) + elapsed
            time.sleep(2.2)
            ledger["mandatory_drain_seconds"] += 2.2
            post_remaining = QUIET_SECONDS - ledger["extra_noise_wait_seconds"]
            post = wait_for_quiet(job["device"], AdmissionPolicy(timeout_seconds=min(15., post_remaining))) if post_remaining > 0 else {"status": "admission_failed", "elapsed_seconds": 0., "extra_wait_seconds": 0., "samples": []}
            # This proof is stamped only after Popen.wait has reaped the worker.
            post.update(worker_exited=True, worker_returncode=code, drain_seconds=2.2)
            ledger["admission_wait_seconds"] += post["elapsed_seconds"]
            ledger["extra_noise_wait_seconds"] += post["extra_wait_seconds"]
            resource = {"returncode": code, "violation": violation, "samples": samples,
                        "elapsed_seconds": elapsed, "admission": admission, "post_exit": post,
                        "maximum_recursive_rss_bytes": max((r["tree_rss_bytes"] for r in samples), default=0)}
            write_atomic_json(directory / "resources.json", resource)
            path = directory / "worker-result.json"
            if code == 0 and violation is None and path.exists():
                result = read_json(path)
                if post["status"] != "admitted":
                    result["status"] = "invalid_post_exit_observation"
            else:
                failed = directory / "failed-raw.json"
                result = read_json(failed) if failed.exists() else {"job": job, "records": []}
                result["status"] = "technical_failure"
            result["resources"] = artifact_record(directory / "resources.json")
    attempt["status"] = result["status"]
    write_atomic_json(directory / "result.json", result)
    write_atomic_json(output / "ledger.json", ledger)
    return result


def job_for(output, stage, device, batch, protocol_hash):
    session_id = f"{stage}-{device.replace(':', '')}-b{batch}"
    directory = Path(output) / "attempts" / session_id
    return {"session_id": session_id, "stage": stage, "device": device, "batch": batch,
            "directory": str(directory.resolve()), "output": str(Path(output).resolve()),
            "admission_path": str((directory / "admission.json").resolve()),
            "protocol_path": str((Path(output) / "protocol.json").resolve()), "protocol_sha256": protocol_hash}


def freeze(output, first, preflight_seconds=300.):
    output, first = Path(output).resolve(), Path(first).resolve()
    if output.exists():
        raise ProtocolError("Archive must be fresh; never overwrite an earlier attempt")
    if not 0 <= preflight_seconds < COMPUTE_SECONDS:
        raise ProtocolError("Finite preflight charge required")
    output.mkdir(parents=True)
    sources = source_manifest()
    root = Path(__file__).resolve().parents[2]
    for relative in sources["lf_source_hashes"]:
        destination = output / "frozen-source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / relative).read_bytes().replace(b"\r\n", b"\n"))
    write_atomic_json(output / "source-freeze.json", sources)
    prospective = {"schema": 3, "source": sources, "schedules": {s: schedule(s) for s in ("pilot", "main")},
        "precision": {"dtype": "float32", "threads": 1, "runtime": "torch.jit", "profiler": False,
                      "tf32": False, "cudnn_benchmark": False, "cudnn_deterministic": True},
        "admission": {**asdict(AdmissionPolicy()), "post_exit_timeout_seconds": 15.,
                      "physical_gpu_before_cuda_owner": True,
                      "active_gpu_utilization": "metadata includes owned work; never classified as external noise"},
        "gate": {"median_range": [.95, 1.05], "block_range": [.85, 1.15], "order_range": [.90, 1.10],
                 "minimum_in_range": {"pilot": 7, "main": 18}, "all_graphs_all_metrics": True,
                 "rationale": "pragmatic operational tolerances, not a formal null-significance test"},
        "resources": {"compute_seconds": COMPUTE_SECONDS, "extra_quiet_wait_seconds": QUIET_SECONDS,
                      "rss_bytes": RSS_LIMIT, "available_ram_bytes": RAM_RESERVE, "gpu_allocator_bytes": 14 * 1024**3,
                      "physical_gpu_free_bytes": GPU_FREE, "preflight_charged_seconds": preflight_seconds},
        "controls": "two independent TorchScript runtime loads of identical graph bytes and weights, both warmed",
        "warmup": {"minimum_calls": 50, "minimum_seconds": .25, "maximum_calls": 10000, "prepair_each_calls": 2},
        "timing": {"pilot_calls_per_metric": 10, "main_calls_per_metric": 20, "throughput_calls": 50,
                   "single_process_per_profile_stage": True, "new_training_updates": 0},
        "profiles": [{"device": d, "batch": b} for d, b in PROFILES],
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "external_gpu": observe_diagnostics(), "platform": platform.platform()}}
    write_atomic_json(output / "prospective.json", prospective)
    ledger = {"charged_seconds": preflight_seconds, "compute_by_stage_seconds": {"preflight": preflight_seconds},
              "admission_wait_seconds": 0., "extra_noise_wait_seconds": 0., "mandatory_drain_seconds": 0., "attempts": []}
    job = job_for(output, "prepare", "cpu", 64, stable_hash(prospective))
    job["first"] = str(first)
    result = monitored(output, job, ledger)
    if result["status"] != "succeeded":
        raise ProtocolError("Preparation failed; immutable failed archive retained")
    if source_manifest()["sha256"] != sources["sha256"]:
        raise ProtocolError("Sources changed during preparation; retain this attempt")
    prospective["prepared"] = result["prepared"]
    prospective["protocol_sha256"] = stable_hash(prospective)
    write_atomic_json(output / "protocol.json", prospective)
    return prospective


def verify(output):
    protocol = read_json(Path(output) / "protocol.json")
    expected = stable_hash({k: v for k, v in protocol.items() if k != "protocol_sha256"})
    if expected != protocol["protocol_sha256"] or source_manifest()["sha256"] != protocol["source"]["sha256"]:
        raise ProtocolError("Frozen protocol or scientific source changed")
    for record in protocol["prepared"]["graphs"].values():
        checked_file(record)
    checked_file(protocol["prepared"]["input"])
    for record in protocol["prepared"]["lineage"].values():
        checked_file(record)
    for checkpoint in protocol["prepared"]["recovered"].values():
        for key in ("architecture", "checkpoint", "artifact"):
            checked_file(checkpoint[key])
    return protocol


def run(output):
    output = Path(output)
    protocol = verify(output)
    if (output / "run.json").exists():
        raise ProtocolError("Completed run cannot be repeated")
    ledger, profiles = read_json(output / "ledger.json"), []
    gpu_name = protocol["environment"]["external_gpu"].get("gpu", {}).get("name")
    if not gpu_name:
        raise ProtocolError("Frozen physical GPU identity unavailable")
    for device, batch in PROFILES:
        entry = {"device": device, "batch": batch, "stages": {}}
        profiles.append(entry)
        for stage in ("pilot", "main"):
            verify(output)
            job = job_for(output, stage, device, batch, protocol["protocol_sha256"])
            job["expected_gpu_name"] = gpu_name
            result = monitored(output, job, ledger)
            complete = validate_records(protocol, job, result["records"]) if result.get("records") else False
            gate = control_gate(result.get("records", []), stage, device)
            record = artifact_record(Path(job["directory"]) / "result.json")
            entry["stages"][stage] = {"relative_path": str(Path(record["path"]).relative_to(output)),
                                      "sha256": record["sha256"], "gate": gate, "complete": complete}
            write_atomic_json(output / "run-progress.json", {"profiles": profiles})
            if not (result["status"] == "succeeded" and complete and gate["passed"]):
                entry["stop_reason"] = f"{stage} failed: retain entire profile, no retry/replacement"
                break
    result = {"schema": 3, "protocol_sha256": protocol["protocol_sha256"], "profiles": profiles,
              "ledger": artifact_record(output / "ledger.json"), "status": "completed_with_retained_gates",
              "statistical_scope": "descriptive same-process temporal blocks, no independent replication claim"}
    write_atomic_json(output / "run.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--first", type=Path)
    parser.add_argument("--stage", choices=("freeze", "run"))
    parser.add_argument("--preflight-seconds", type=float, default=300.)
    parser.add_argument("--worker-job", type=Path)
    args = parser.parse_args()
    if args.worker_job:
        job = read_json(args.worker_job)
        (prepare_worker if job["stage"] == "prepare" else timing_worker)(job)
    elif args.stage == "freeze":
        freeze(args.output, args.first, args.preflight_seconds)
    elif args.stage == "run":
        run(args.output)
    else:
        parser.error("Explicit stage or worker job required")


if __name__ == "__main__":
    main()
