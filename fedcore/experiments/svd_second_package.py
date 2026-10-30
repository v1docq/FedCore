"""Finite, inference-only second packet: recovered CE400 and Conv workloads.

The controller observes admission before spawning a CUDA-owning worker, records
all attempts, and invalidates complete planned blocks without selecting survivors.
No first-packet artifact or training checkpoint is modified by this module.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import torch
from torch import nn

from fedcore.tools.atomic_json import write_atomic_json
from fedcore.tools.export import export_model
from . import measurement, runner
from .protocol import ProtocolError, stable_hash, tensor_hash
from .svd_second_timing import timing_worker

MODULE = "fedcore.experiments.svd_second_package"
SEED = 42
RSS_LIMIT = 12 * 1024**3
OS_RESERVE = 6 * 1024**3
GPU_LIMIT = 14 * 1024**3
COMPUTE_SECONDS = 2 * 3600
GPU_FREE = 2 * 1024**3
COMPARISONS = ("conv2_svd", "conv2_weighted", "conv2_svd_linear16", "conv2_weighted_linear16", "historical_afm_linear8")
CONV_GROUPS = ("digits_conv2", "synthetic_64_32_r16", "synthetic_128_16_r32", "synthetic_256_8_r64")
REPRESENTATIONS = ("original_dense", "factor", "materialized")


@dataclass(frozen=True)
class TimingSession:
    session_id: str
    block_id: str
    kind: str
    subject: str
    device: str
    batch: int
    block: int
    side: str
    order: int


@dataclass(frozen=True)
class AdmissionPolicy:
    maximum_cpu_percent: float = 25.
    maximum_gpu_percent: float = 10.
    consecutive: int = 3
    interval_seconds: float = 1.
    timeout_seconds: float = 60.
    drain_seconds: float = 2.2

    def __post_init__(self):
        if (self.consecutive < 1 or self.interval_seconds <= 0 or self.timeout_seconds <= 0
                or self.drain_seconds < 0 or not all(math.isfinite(value) for value in
                (self.maximum_cpu_percent, self.maximum_gpu_percent, self.interval_seconds,
                 self.timeout_seconds, self.drain_seconds))):
            raise ProtocolError("Admission policy must be finite and bounded")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def timing_schedule(seed=SEED, comparison_blocks=10, control_blocks=5, conv_blocks=3):
    """Pairs stay adjacent; controls interleave; AB/BA and triple order balance."""
    if (comparison_blocks != 10 or control_blocks != 5 or conv_blocks != 3):
        raise ProtocolError("The approved finite matrix is 10/5/3 blocks")
    rng = random.Random(seed)
    schedule = []
    profiles = [(device, batch) for device in ("cpu", "cuda:0") for batch in (1, 64)]
    for profile_index, (device, batch) in enumerate(profiles):
        prefix = f"{device.replace(':', '')}-b{batch}"
        starts = {}
        for subject in COMPARISONS:
            starts[subject] = []
            for _ in range(5):
                first_order = rng.randrange(2)
                starts[subject].extend((first_order, 1 - first_order))
        control_orders = []
        for _ in range(2):
            first_order = rng.randrange(2)
            control_orders.extend((first_order, 1 - first_order))
        control_orders.append(profile_index % 2)
        rotations = {}
        for group in CONV_GROUPS:
            initial = list(REPRESENTATIONS)
            rng.shuffle(initial)
            rotations[group] = [initial[k:] + initial[:k] for k in range(3)]
        for block in range(10):
            units = []
            for subject in COMPARISONS:
                sides = ("baseline", "candidate") if starts[subject][block] == 0 else ("candidate", "baseline")
                units.append(("artifact", subject, block, sides))
            if block % 2 == 1:
                control_block = block // 2
                sides = ("baseline", "candidate") if control_orders[control_block] == 0 else ("candidate", "baseline")
                units.append(("control", "dense_dense", control_block, sides))
            if block < 3:
                units.extend(("conv", group, block, rotations[group][block]) for group in CONV_GROUPS)
            rng.shuffle(units)
            for kind, subject, unit_block, sides in units:
                block_id = f"{kind}-{subject}-{prefix}-k{unit_block}"
                for order, side in enumerate(sides):
                    schedule.append(TimingSession(f"{block_id}-{side}", block_id, kind, subject,
                                                  device, batch, unit_block, side, order))
    if len(schedule) != 584 or len({s.session_id for s in schedule}) != 584:
        raise ProtocolError("Frozen 584-session schedule invariant failed")
    return tuple(schedule)


def profile_schedule():
    return tuple(TimingSession(f"profile-{group}-{device.replace(':', '')}-b{batch}-{side}",
                 f"profile-{group}-{device.replace(':', '')}-b{batch}", "conv", group, device, batch, 0, side, order)
                 for device in ("cpu", "cuda:0") for batch in (1, 64)
                 for group in CONV_GROUPS for order, side in enumerate(REPRESENTATIONS))


def external_snapshot():
    """Controller only.  No torch.cuda call and no worker exists at admission."""
    import psutil
    snapshot = {"utc_epoch": time.time(), "observer_pid": os.getpid(),
        "available_ram_bytes": psutil.virtual_memory().available,
        "cpu_percent": psutil.cpu_percent(interval=.2), "gpu_scope": "global physical device, no attribution"}
    try:
        raw = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             check=True, timeout=5).stdout.strip()
        fields = raw.splitlines()[0].split(",")
        if fields[0].strip() != "0":
            raise ValueError("Approved GPU index0 missing")
        snapshot["gpu"] = {"index": 0, "name": fields[1].strip(), "utilization_percent": float(fields[2]),
                           "physical_used_mib": float(fields[3]), "physical_free_mib": float(fields[4])}
    except (OSError, ValueError, subprocess.SubprocessError, IndexError) as error:
        snapshot["gpu_observation_error"] = type(error).__name__
    return snapshot


def quiet_reasons(sample, device, policy=AdmissionPolicy()):
    reasons = []
    cpu = sample.get("cpu_percent")
    if cpu is None or not math.isfinite(cpu):
        reasons.append("CPU observation unavailable")
    elif cpu > policy.maximum_cpu_percent:
        reasons.append("external CPU exceeds 25%")
    available = sample.get("available_ram_bytes")
    if not isinstance(available, (float, int)) or not math.isfinite(available) or available < OS_RESERVE:
        reasons.append("available RAM below 6GiB")
    if str(device).startswith("cuda"):
        gpu = sample.get("gpu")
        if not gpu or not all(isinstance(gpu.get(k), (float, int)) and math.isfinite(gpu[k])
                              for k in ("utilization_percent", "physical_free_mib")):
            reasons.append("GPU observation unavailable")
        else:
            if gpu["utilization_percent"] > policy.maximum_gpu_percent:
                reasons.append("external GPU exceeds 10%")
            if gpu["physical_free_mib"] * 1024**2 < GPU_FREE:
                reasons.append("physical GPU free below 2GiB")
    return reasons


def wait_for_quiet(device, policy=AdmissionPolicy(), *, observer=external_snapshot,
                   monotonic=time.monotonic, sleep=time.sleep):
    start = monotonic()
    deadline = start + policy.timeout_seconds
    samples, streak = [], 0
    quiet_start = None
    while monotonic() < deadline:
        sample_start = monotonic()
        sample = observer()
        reasons = quiet_reasons(sample, device, policy)
        streak = 0 if reasons else streak + 1
        if reasons:
            quiet_start = None
        elif streak == 1:
            quiet_start = sample_start
        samples.append({"sample": sample, "reasons": reasons, "quiet_streak": streak,
                        "elapsed_seconds": monotonic() - start})
        if streak >= policy.consecutive and monotonic() <= deadline:
            elapsed = monotonic() - start
            extra = max(0., quiet_start - start)
            return {"status": "admitted", "elapsed_seconds": elapsed, "samples": samples,
                    "extra_wait_seconds": extra, "mandatory_observation_seconds": elapsed - extra}
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        pause = min(policy.interval_seconds, remaining)
        sleep(pause)
    elapsed = monotonic() - start
    return {"status": "admission_failed", "elapsed_seconds": elapsed, "samples": samples,
            "extra_wait_seconds": elapsed, "mandatory_observation_seconds": 0.,
            "reasons": ["bounded quiet-window deadline expired"]}


def source_manifest():
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "fedcore").rglob("*.py")) + list((root / "tests/refactoring").glob("test_svd_second*.py"))
    analyzer = root / "scripts/analyze_svd_second_package.py"
    if analyzer.exists():
        paths.append(analyzer)
    hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
              for path in sorted(paths)}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    return {"git_head": head, "git_lf_source_hashes": hashes, "sha256": stable_hash(hashes),
            "normalization": "CRLF to LF; all fedcore Python, second-packet tests and analyzer including untracked"}


def environment_manifest():
    """Allowlisted current environment, evaluated only in the freeze process."""
    import psutil
    environment = runner.environment_manifest()
    environment.update(logical_cpu_count=os.cpu_count(), physical_cpu_count=psutil.cpu_count(logical=False),
                       physical_ram_bytes=psutil.virtual_memory().total)
    if sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                environment["cpu_model"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            environment["cpu_model"] = None
    try:
        environment["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,pci.bus_id",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        environment["nvidia_smi_error"] = type(error).__name__
    return environment


def checked_file(record, *, root=None):
    path = Path(record["path"]).resolve()
    if root is not None and not path.is_relative_to(Path(root).resolve()):
        raise ProtocolError("Trusted archive escaped its owning package")
    if not path.is_file() or measurement.file_hash(path) != record["sha256"]:
        raise ProtocolError(f"Frozen archive identity differs: {path.name}")
    return path


def _artifact_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": measurement.file_hash(path)}


def recovered_checkpoint(first, name):
    recovery = read_json(Path(first) / "recovery.json")
    curves = [curve for curve in recovery["curves"] if curve["name"] == name and curve["objective"] == "ce"]
    if len(curves) != 1 or curves[0]["status"] != "succeeded" or curves[0]["training"]["training_steps"] != 400:
        raise ProtocolError("Expected exactly one completed trusted CE400 curve")
    rows = [row for row in curves[0]["checkpoints"] if row["update"] == 400]
    if len(rows) != 1:
        raise ProtocolError("Expected exactly one update400 checkpoint")
    row = rows[0]
    for key in ("architecture", "checkpoint", "artifact"):
        checked_file(row[key], root=Path(first) / "recovery" / f"{name}-ce" / "update-400")
    model = torch.load(row["architecture"]["path"], map_location="cpu", weights_only=False).eval()
    state = torch.load(row["checkpoint"]["path"], map_location="cpu", weights_only=True)
    if set(model.state_dict()) != set(state) or any(not torch.equal(model.state_dict()[key], state[key]) for key in state):
        raise ProtocolError("Trusted architecture and checkpoint tensors differ")
    if runner.model_state_hash(model) != row["model_hash"]:
        raise ProtocolError("Recovered model state differs")
    return model, row


def synthetic_conv(channels, size, rank, seed):
    """Representative kernel workload; constructed factors, no learned-quality claim."""
    from fedcore.models.network_impl.decomposed_layers import DecomposedConv2d
    from fedcore.algorithm.low_rank.reassembly.decomposed_recreation import to_standard_module
    generator = torch.Generator().manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        original = nn.Conv2d(channels, channels, 3, padding=1).eval()
        factor = DecomposedConv2d(original, decomposing_mode=None, compose_mode="two_layers")
    left = torch.randn(channels, rank, generator=generator) / math.sqrt(rank)
    right = torch.randn(rank, channels * 9, generator=generator) / math.sqrt(channels * 9)
    factor.set_U_S_Vh(left, torch.ones(rank), right)
    factor.compose_weight_for_inference()
    with torch.random.fork_rng(devices=[]):
        materialized = to_standard_module(factor).eval()
    inputs = torch.randn(64, channels, size, size, generator=generator)
    return original, factor.eval(), materialized, inputs


def prepare_sources(output, first, historical):
    """Recover identities and make new kernel artifacts; no CUDA and no training."""
    output, first, historical = Path(output).resolve(), Path(first).resolve(), Path(historical).resolve()
    if (output / "prepared.json").exists() or (output / "protocol.json").exists():
        raise ProtocolError("Prepared sources are immutable; do not recreate them")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    before = source_manifest()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    first_protocol = read_json(first / "protocol.json")
    data_record = first_protocol["input_files"]["data"]
    checked_file(data_record, root=first)
    data = torch.load(data_record["path"], map_location="cpu", weights_only=True)
    model_input = output / "sources" / "model-input.pt"
    model_input.parent.mkdir(parents=True, exist_ok=True)
    model_x = data["validation"]["x"][:64].clone()
    torch.save(model_x, model_input)
    models, checkpoints, sources = {}, {}, {}
    for name in (*COMPARISONS[:4], "dense"):
        models[name], checkpoints[name] = recovered_checkpoint(first, name)
        sources[name] = dict(checkpoints[name]["artifact"], model_hash=checkpoints[name]["model_hash"],
                             source_scope=f"existing {name} CE recovery update400; no additional updates")
    run_path = historical / "cnn-seed-42" / "run.json"
    run = read_json(run_path)
    afm = [row for row in run["records"] if row["case_id"] == "33b7f7fce9b5ae84"]
    if len(afm) != 1 or afm[0]["status"] != "succeeded":
        raise ProtocolError("Expected the historical AFM Linear8 record")
    sources["historical_dense"] = run["baseline"]["export"]["artifact"]
    sources["historical_afm_linear8"] = afm[0]["export"]["artifact"]
    for key in ("historical_dense", "historical_afm_linear8"):
        checked_file(sources[key], root=historical)
    from fedcore.algorithm.low_rank.reassembly.decomposed_recreation import to_standard_module
    original = copy.deepcopy(models["dense"].get_submodule("features.2")).eval()
    factor = copy.deepcopy(models["conv2_svd"].get_submodule("features.2")).eval()
    materialized = to_standard_module(factor).eval()
    handle = models["dense"].get_submodule("features.2").register_forward_pre_hook(
        lambda layer, args: captures.append(args[0].detach().clone()))
    captures = []
    with torch.inference_mode():
        models["dense"](data["validation"]["x"][:64])
    handle.remove()
    groups = {"digits_conv2": (original, factor, materialized, captures[0])}
    for index, (channels, size, rank) in enumerate(((64, 32, 16), (128, 16, 32), (256, 8, 64))):
        groups[CONV_GROUPS[index + 1]] = synthetic_conv(channels, size, rank, SEED + index)
    micro = {}
    with torch.inference_mode():
        for group, (dense, decomposed, composed, inputs) in groups.items():
            folder = output / "sources" / group
            folder.mkdir(parents=True, exist_ok=True)
            input_path = folder / "input.pt"
            torch.save(inputs, input_path)
            # A tiny parity pass avoids treating the synthetic 64-image pass as
            # a preparation benchmark.  Both batch shapes are checked by workers.
            expected, actual = decomposed(inputs[:1]), composed(inputs[:1])
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            representations = {}
            for side, layer in zip(REPRESENTATIONS, (dense, decomposed, composed)):
                artifact = export_model(layer, "torchscript", folder / side, inputs[:1])
                representations[side] = dict(_artifact_record(artifact), tensor_bytes=measurement.tensor_state_bytes(layer))
            micro[group] = {"input": dict(_artifact_record(input_path), tensor_sha256_by_batch={str(n): tensor_hash(inputs[:n]) for n in (1, 64)}), "representations": representations,
                "input_shape": list(inputs.shape), "rank": decomposed.U.shape[-1],
                "factor_weight_sha256": tensor_hash(decomposed._get_composed_weight()),
                "materialized_weight_sha256": tensor_hash(composed.weight),
                "factor_bias_sha256": tensor_hash(decomposed.bias), "materialized_bias_sha256": tensor_hash(composed.bias),
                "max_abs_parity_error": (actual - expected).abs().max().item(),
                "scope": ("trained Digits Conv2; dense and factor independently recovered; only factor/materialized same function"
                          if group == "digits_conv2" else "synthetic kernel workload, constructed factors; no trained quality claim")}
    comparisons = {name: {"baseline": "dense", "candidate": name, "scope": "recovered CE400 against recovered dense CE400"}
                   for name in COMPARISONS[:4]}
    comparisons[COMPARISONS[4]] = {"baseline": "historical_dense", "candidate": COMPARISONS[4],
                                  "scope": "historical AFM Linear8 against its original historical dense baseline"}
    comparisons["dense_dense"] = {"baseline": "dense", "candidate": "dense", "scope": "identity harness control"}
    after = source_manifest()
    prepared = {"schema": 2, "first_packet": str(first), "historical": str(historical),
        "source_before": before, "source_after": after, "sources": sources, "comparisons": comparisons,
        "model_input": dict(_artifact_record(model_input), tensor_sha256_by_batch={str(n): tensor_hash(model_x[:n]) for n in (1, 64)}), "conv_groups": micro,
        "recovered_checkpoints": checkpoints,
        "lineage_files": {"recovery": _artifact_record(first / "recovery.json"),
                          "first_protocol": _artifact_record(first / "protocol.json"),
                          "historical_run": _artifact_record(run_path), "historical_data": data_record},
        "source_unchanged_during_preparation": before["sha256"] == after["sha256"],
        "preparation_elapsed_seconds": time.perf_counter() - started, "new_training_updates": 0}
    write_atomic_json(output / "prepared.json", prepared)
    if not prepared["source_unchanged_during_preparation"]:
        raise ProtocolError("Source changed during preparation; retain this attempt")
    return prepared


def verify_sources(prepared):
    for record in prepared["lineage_files"].values():
        checked_file(record)
    checked_file(prepared["model_input"])
    for record in prepared["sources"].values():
        checked_file(record)
    for checkpoint in prepared["recovered_checkpoints"].values():
        for key in ("architecture", "checkpoint", "artifact"):
            checked_file(checkpoint[key])
    for group in prepared["conv_groups"].values():
        checked_file(group["input"])
        for record in group["representations"].values():
            checked_file(record)


def freeze_protocol(output, repeats=100, preflight_seconds=300.):
    output = Path(output)
    if (output / "protocol.json").exists():
        raise ProtocolError("Never replace the frozen protocol")
    if repeats not in (50, 100, 200) or not math.isfinite(preflight_seconds) or preflight_seconds < 0:
        raise ProtocolError("Pilot may select only predeclared repeat counts 50/100/200")
    prepared = read_json(output / "prepared.json")
    verify_sources(prepared)
    current_source = source_manifest()
    if (not prepared["source_unchanged_during_preparation"]
            or prepared["source_after"]["sha256"] != current_source["sha256"]):
        raise ProtocolError("Prepared sources have different source lineage; retain this preparation attempt")
    pilot_path = output / "pilot.json"
    pilot = read_json(pilot_path) if pilot_path.exists() else None
    if pilot is None or any(row["status"] != "succeeded" for row in pilot["sessions"]):
        raise ProtocolError("Freeze requires all eight retained feasibility-pilot sessions admitted and completed")
    projection = pilot["compute_estimates_by_repeats"][str(repeats)]
    if projection is None or projection > COMPUTE_SECONDS:
        raise ProtocolError("Selected repeat count does not fit the pilot's conservative2h compute estimate")
    if pilot and (pilot["source_before"]["sha256"] != current_source["sha256"]
                  or pilot["source_after"]["sha256"] != current_source["sha256"]):
        raise ProtocolError("Pilot source differs from final frozen source")
    protocol = {"schema": 2, "study": "FedCore second local SVD inference packet 2026-10-14", "seed": SEED,
        "source": current_source, "environment": environment_manifest(),
        "prepared": prepared, "prepared_sha256": measurement.file_hash(output / "prepared.json"),
        "timing_schedule": [asdict(session) for session in timing_schedule()],
        "profile_schedule": [asdict(session) for session in profile_schedule()],
        "comparisons": prepared["comparisons"],
        "timing": {"sessions": 584, "artifact_sessions": 400, "dense_control_sessions": 40, "conv_sessions": 144,
            "comparison_blocks": 10, "control_blocks": 5, "conv_blocks": 3, "warmup": 50, "individual_repeats": repeats,
            "minimum_warmup_seconds": .2, "maximum_warmup_calls": 100000,
            "throughput_blocks": 3, "calls_per_throughput_block": 50, "threads": 1, "dtype": "float32",
            "matmul_tf32": False, "cudnn_tf32": False, "cudnn_benchmark": False, "cudnn_deterministic": True,
            "runtime": "torch.jit", "profiler": False, "independent_training_replicates": 0,
            "passes": ["synchronized wall forward", "CPU-device-CPU full path", "CUDA event stream interval", "blocked throughput"],
            "cuda_event_scope": "includes possible host launch gaps, not pure kernel execution",
            "invalid_block_rule": "any invalid, absent or failed planned member invalidates complete pair/triple; no replacement",
            "unit": "one paired AB/BA timing block; repeated calls within session are not independent units",
            "analysis": {"ratio": "reference p50 / candidate p50", "point": "median block ratio plus full range",
                "median_interval": "largest k with 1 - 2*sum(comb(n,i),i=0..k-1)/2**n >= .95; sorted[k-1], sorted[n-k]",
                "minimum_interval_blocks": 6, "conv_interval": "n3 descriptive range only, no95%interval",
                "assumptions": "independent stationary timing blocks conditional on one machine; no training/quality confidence interval"}},
        "profiling": {"sessions": 48, "calls": 3, "warmup": 10, "threads": 1, "trace_limit_bytes": 32 * 1024**2,
                      "total_trace_limit_bytes": 48 * 32 * 1024**2, "separate_after_all_timing": True,
                      "timing_statistics_eligible": False, "purpose": "operator/launch/allocation diagnostics, no inference latency claim"},
        "admission": dict(asdict(AdmissionPolicy()), post_exit_timeout_seconds=15.,
                          total_extra_wait_budget_seconds=600., own_active_gpu_utilization="metadata only, never external noise",
                          mandatory_wait_scope="all quiet observations and drains separately recorded, excluded from extra-noise600s",
                          post_exit_drain="worker exited, then2.2s drain, then consecutive quiet observations"),
        "resources": {"compute_wall_seconds": COMPUTE_SECONDS, "tree_rss_bytes": RSS_LIMIT, "os_reserve_bytes": OS_RESERVE,
            "gpu_process_allocator_bytes": GPU_LIMIT, "gpu_physical_free_bytes": GPU_FREE, "preparation_threads": 8,
            "timing_threads": 1, "profile_threads": 1, "maximum_parallel_workers": 1,
            "preflight_charged_seconds": float(preflight_seconds), "admission_wait_charged_separately": True},
        "pilot": pilot, "failure": {"scientific_retries": 0, "retain_all_attempts": True,
                                     "source_changes_after_freeze": "stop; keep all prior results"},
        "training": {"new_student_updates": 0, "new_teacher_forwards": 0, "fit_allowed": False}}
    protocol["protocol_sha256"] = stable_hash(protocol)
    write_atomic_json(output / "protocol.json", protocol)
    pilot_compute = float(pilot.get("charged_seconds", 0)) if pilot else 0.
    write_atomic_json(output / "ledger.json", {"schema": 2, "protocol_sha256": protocol["protocol_sha256"],
        "attempts": [], "charged_seconds": prepared["preparation_elapsed_seconds"] + preflight_seconds + pilot_compute,
        "preparation_seconds": prepared["preparation_elapsed_seconds"], "preflight_charged_seconds": preflight_seconds,
        "pilot_charged_seconds": pilot_compute, "admission_wait_seconds": float(pilot.get("admission_wait_seconds", 0)) if pilot else 0.,
        "extra_noise_wait_seconds": float(pilot.get("extra_noise_wait_seconds", 0)) if pilot else 0.,
        "new_training_updates": 0})
    return protocol


def verify_protocol(output, *, verify_files=True):
    output = Path(output)
    protocol = read_json(output / "protocol.json")
    expected = protocol.pop("protocol_sha256")
    if stable_hash(protocol) != expected:
        raise ProtocolError("Frozen protocol changed")
    protocol["protocol_sha256"] = expected
    if source_manifest()["sha256"] != protocol["source"]["sha256"]:
        raise ProtocolError("Source changed after freeze")
    if measurement.file_hash(output / "prepared.json") != protocol["prepared_sha256"]:
        raise ProtocolError("Prepared manifest changed")
    if verify_files:
        verify_sources(protocol["prepared"])
    return protocol


def session_job(protocol, session, directory, *, profiler=False, pilot=False):
    row = asdict(session) if isinstance(session, TimingSession) else dict(session)
    prepared = protocol["prepared"]
    if row["kind"] == "conv":
        group = prepared["conv_groups"][row["subject"]]
        source, inputs = group["representations"][row["side"]], group["input"]
    else:
        source = prepared["sources"][prepared["comparisons"][row["subject"]][row["side"]]]
        inputs = prepared["model_input"]
    job = dict(row, directory=str(Path(directory).resolve()), protocol_sha256=protocol["protocol_sha256"],
        source_artifact=source["path"], source_artifact_sha256=source["sha256"],
        input_path=inputs["path"], input_file_sha256=inputs["sha256"],
        input_tensor_sha256=inputs.get("tensor_sha256_by_batch", {}).get(str(row["batch"])),
        repeats=20 if pilot else protocol["timing"]["individual_repeats"],
        warmup=10 if profiler else (20 if pilot else protocol["timing"]["warmup"]),
        minimum_warmup_seconds=.2, maximum_warmup_calls=100000,
        expected_gpu_name=protocol.get("environment", {}).get("cuda_devices", ["NVIDIA GeForce RTX 4080"])[0]
            if row["device"].startswith("cuda") else None,
        throughput_blocks=1 if pilot else protocol["timing"]["throughput_blocks"],
        throughput_calls=20 if pilot else protocol["timing"]["calls_per_throughput_block"],
        profiler=profiler, profile_calls=protocol.get("profiling", {}).get("calls", 3),
        trace_limit_bytes=32 * 1024**2, gpu_allocator_limit_bytes=GPU_LIMIT,
        admission_path=str((Path(directory) / "admission.json").resolve()))
    return job


def summarize_blocks(protocol, sessions):
    """Conserve every planned block; one bad member invalidates the whole block."""
    schedule = protocol["timing_schedule"]
    expected = {row["session_id"]: row for row in schedule}
    if len(expected) != len(schedule):
        raise ProtocolError("Duplicate planned session identity")
    actual = {}
    for session in sessions:
        job = session["job"]
        identity = job["session_id"]
        if identity in actual or identity not in expected:
            raise ProtocolError("Duplicate or unexpected result session identity")
        if any(job.get(key) != value for key, value in expected[identity].items()):
            raise ProtocolError("Result session differs from its planned block")
        frozen_job = session_job(protocol, expected[identity], job["directory"])
        for key in ("source_artifact_sha256", "input_file_sha256", "protocol_sha256"):
            if job.get(key) != frozen_job[key]:
                raise ProtocolError("Result source/input/protocol identity differs")
        if job.get("profiler") or session.get("profile", {}).get("profiler"):
            raise ProtocolError("Profiler observation cannot be a timing block")
        expected_tensor = frozen_job.get("input_tensor_sha256")
        if session.get("status") == "succeeded" and expected_tensor and session.get("input_sha256") != expected_tensor:
            raise ProtocolError("Observed input tensor identity differs")
        actual[identity] = session
    grouped = defaultdict(list)
    for row in schedule:
        grouped[row["block_id"]].append(row)
    blocks = []
    for block_id, members in grouped.items():
        reasons = []
        for member in members:
            result = actual.get(member["session_id"])
            if result is None or result.get("status") != "succeeded":
                reasons.append({"session_id": member["session_id"],
                                "status": result.get("status") if result else "missing",
                                "reasons": result.get("noise_reasons", []) if result else []})
        first = members[0]
        blocks.append({key: first[key] for key in ("block_id", "kind", "subject", "device", "batch", "block")}
                      | {"status": "invalid" if reasons else "succeeded", "invalid_reasons": reasons,
                         "session_ids": [member["session_id"] for member in members]})
    return blocks


def _terminate_owned(process):
    import psutil
    descendants = process.children(recursive=True)
    for child in reversed(descendants):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        process.terminate()
    except psutil.NoSuchProcess:
        pass
    _, alive = psutil.wait_procs(descendants, timeout=2)
    for item in alive:
        try:
            item.kill()
        except psutil.NoSuchProcess:
            pass
    # Popen must reap its own root, preserving the real OS exit status.
    try:
        if process.is_running():
            process.kill()
    except psutil.NoSuchProcess:
        pass


def monitored_session(output, protocol, job, ledger):
    """An external observer admits a worker; no concurrent workers are possible."""
    import psutil
    output = Path(output)
    directory = Path(job["directory"])
    if directory.exists() and any(directory.iterdir()):
        raise ProtocolError("Attempt directory already exists; scientific retry is forbidden")
    directory.mkdir(parents=True, exist_ok=True)
    write_atomic_json(directory / "job.json", job)
    attempt = {"session_id": job["session_id"], "directory": str(directory), "status": "admission_pending"}
    ledger["attempts"].append(attempt)
    write_atomic_json(output / "ledger.json", ledger)
    compute_remaining = COMPUTE_SECONDS - ledger["charged_seconds"]
    wait_remaining = protocol["admission"]["total_extra_wait_budget_seconds"] - ledger["extra_noise_wait_seconds"]
    if compute_remaining <= 0 or wait_remaining <= 0:
        result = {"schema": 2, "job": job, "status": "not_run_budget", "noise_reasons": ["cumulative finite budget reached"]}
        write_atomic_json(directory / "result.json", result)
        attempt.update(status=result["status"], elapsed_seconds=0.)
        write_atomic_json(output / "ledger.json", ledger)
        return result
    policy = AdmissionPolicy(timeout_seconds=min(60., wait_remaining))
    admission = wait_for_quiet(job["device"], policy)
    admission.update(session_id=job["session_id"], protocol_sha256=job["protocol_sha256"], before_worker_spawn=True)
    write_atomic_json(directory / "admission.json", admission)
    ledger["admission_wait_seconds"] += admission["elapsed_seconds"]
    ledger["extra_noise_wait_seconds"] += admission["extra_wait_seconds"]
    attempt.update(admission_status=admission["status"])
    if admission["status"] != "admitted":
        result = {"schema": 2, "job": job, "status": "admission_failed", "admission": admission,
                  "noise_reasons": admission["reasons"]}
        attempt.update(status=result["status"], elapsed_seconds=0.)
        write_atomic_json(directory / "result.json", result)
        write_atomic_json(output / "ledger.json", ledger)
        return result
    command = [sys.executable, "-m", MODULE, "--output", str(output), "--worker", "timing", "--job", str(directory / "job.json")]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONUTF8="1")
    samples, violation = [], None
    start = time.perf_counter()
    with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
        worker = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env)
        attempt.update(status="worker_started", worker_pid=worker.pid, command=command)
        write_atomic_json(output / "ledger.json", ledger)
        owned = psutil.Process(worker.pid)
        while worker.poll() is None:
            try:
                tree = [owned, *owned.children(recursive=True)]
                rss = sum(p.memory_info().rss for p in tree if p.is_running())
                available = psutil.virtual_memory().available
                stage_path = directory / "stage.json"
                stage = read_json(stage_path).get("stage", "import") if stage_path.exists() else "import"
                sample = {"elapsed_seconds": time.perf_counter() - start, "tree_sum_rss_bytes": rss,
                          "available_ram_bytes": available, "stage": stage, "owned_pids": [p.pid for p in tree]}
                # Global utilization while a worker owns CUDA cannot identify an
                # unrelated process.  It is never passed to quiet_reasons.
                if job["device"].startswith("cuda"):
                    observed = external_snapshot()
                    sample["global_gpu_metadata"] = observed.get("gpu")
                    sample["active_utilization_eligibility"] = "metadata only, includes owned CUDA activity"
                    gpu = observed.get("gpu")
                    if gpu and gpu["physical_free_mib"] * 1024**2 < GPU_FREE:
                        violation = "physical GPU headroom below 2GiB during owned worker"
                samples.append(sample)
                if rss > RSS_LIMIT:
                    violation = "owned recursive RSS above12GiB"
                elif available < OS_RESERVE:
                    violation = "available RAM below6GiB"
                elif time.perf_counter() - start >= compute_remaining:
                    violation = "cumulative2h measured computational budget reached"
                if violation:
                    _terminate_owned(owned)
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(.2)
        code = worker.wait(timeout=30)
    elapsed = time.perf_counter() - start
    ledger["charged_seconds"] += elapsed
    # The process has EXITED.  Wait out its NVIDIA sampling window before any
    # independent quietness assessment, rather than calling its own work noise.
    time.sleep(protocol["admission"]["drain_seconds"])
    ledger["mandatory_drain_seconds"] = ledger.get("mandatory_drain_seconds", 0.) + protocol["admission"]["drain_seconds"]
    remaining_wait = protocol["admission"]["total_extra_wait_budget_seconds"] - ledger["extra_noise_wait_seconds"]
    if remaining_wait > 0:
        post = wait_for_quiet(job["device"], AdmissionPolicy(timeout_seconds=min(15., remaining_wait)))
        ledger["admission_wait_seconds"] += post["elapsed_seconds"]
        ledger["extra_noise_wait_seconds"] += post["extra_wait_seconds"]
    else:
        post = {"status": "admission_failed", "elapsed_seconds": 0., "samples": [], "reasons": ["cumulative observation wait budget reached"]}
    post.update(worker_exited=True, drain_seconds=protocol["admission"]["drain_seconds"], worker_returncode=code)
    resource = {"schema": 2, "session_id": job["session_id"], "command": command, "worker_pid": worker.pid,
        "returncode": code, "violation": violation, "elapsed_seconds": elapsed, "samples": samples,
        "maximum_recursive_rss_bytes": max((row["tree_sum_rss_bytes"] for row in samples), default=0),
        "admission": admission, "post_exit_observation": post}
    write_atomic_json(directory / "resources.json", resource)
    worker_result = directory / "worker-result.json"
    if code != 0 or violation or not worker_result.exists():
        result = {"schema": 2, "job": job, "status": "technical_failure",
                  "noise_reasons": [violation or f"worker OS returncode{code}; inspect retained stderr/failed-raw"]}
    else:
        result = read_json(worker_result)
        result["status"] = "succeeded" if post["status"] == "admitted" else "invalid_noise"
        result["noise_reasons"] = [] if result["status"] == "succeeded" else post["reasons"]
    result.update(admission=admission, post_exit_observation=post, resources_path=str((directory / "resources.json").resolve()))
    write_atomic_json(directory / "result.json", result)
    attempt.update(status=result["status"], elapsed_seconds=elapsed, returncode=code, violation=violation,
                   resources_path=result["resources_path"])
    write_atomic_json(output / "ledger.json", ledger)
    return result


def run_stage(output, *, profiler=False):
    output = Path(output)
    protocol = verify_protocol(output)
    ledger = read_json(output / "ledger.json")
    if profiler and not (output / "timing.json").exists():
        raise ProtocolError("Profiler must run after the complete timing pass")
    name = "profiling" if profiler else "timing"
    if (output / f"{name}.json").exists():
        raise ProtocolError("Completed stage cannot be rerun")
    schedule = protocol["profile_schedule" if profiler else "timing_schedule"]
    results = []
    for session in schedule:
        verify_protocol(output, verify_files=False)
        directory = output / name / session["session_id"]
        job = session_job(protocol, session, directory, profiler=profiler)
        if (directory / "result.json").exists():
            prior = read_json(directory / "result.json")
            if prior["job"] != job:
                raise ProtocolError("Resumed session identity differs")
            results.append(prior)
            continue
        result = monitored_session(output, protocol, job, ledger)
        results.append(result)
        write_atomic_json(output / f"{name}-progress.json", {"completed": len(results), "planned": len(schedule),
            "valid": sum(row["status"] == "succeeded" for row in results), "charged_seconds": ledger["charged_seconds"],
            "admission_wait_seconds": ledger["admission_wait_seconds"]})
    summary = {"schema": 2, "protocol_sha256": protocol["protocol_sha256"], "sessions": results,
        "source_after": source_manifest(), "all_source_artifacts_unchanged": True,
        "new_training_updates": 0, "diagnostic_only": profiler}
    if not profiler:
        summary["blocks"] = summarize_blocks(protocol, results)
    else:
        trace_bytes = sum(row.get("trace", {}).get("bytes", 0) for row in results)
        if trace_bytes > protocol["profiling"]["total_trace_limit_bytes"]:
            raise ProtocolError("Aggregate traces exceed frozen cap")
        summary["trace_bytes"] = trace_bytes
    verify_protocol(output)
    write_atomic_json(output / f"{name}.json", summary)
    return summary


def run_pilot(output):
    """Eight retained pilot sessions to bound fresh-process cost before freeze."""
    output = Path(output)
    if (output / "protocol.json").exists() or (output / "pilot.json").exists():
        raise ProtocolError("One pilot only, before the scientific protocol freeze")
    prepared = read_json(output / "prepared.json")
    verify_sources(prepared)
    source_before = source_manifest()
    if source_before["sha256"] != prepared["source_after"]["sha256"]:
        raise ProtocolError("Pilot source differs from prepared source")
    protocol = {"prepared": prepared, "protocol_sha256": stable_hash({"prepared": prepared, "pilot": "8sessions20calls"}),
                "timing": {"warmup": 20, "individual_repeats": 20, "throughput_blocks": 1, "calls_per_throughput_block": 20},
                "admission": dict(asdict(AdmissionPolicy()), total_extra_wait_budget_seconds=600.)}
    pilot_sessions = [TimingSession(f"pilot-{group}-{device.replace(':', '')}-b{batch}", "pilot", "control" if group == "dense_dense" else "conv",
         group, device, batch, 0, "baseline" if group == "dense_dense" else "original_dense", 0)
         for device in ("cpu", "cuda:0") for batch in (1, 64)
         for group in ("dense_dense", "synthetic_256_8_r64")]
    ledger = {"schema": 2, "protocol_sha256": protocol["protocol_sha256"], "attempts": [],
              "charged_seconds": 0., "admission_wait_seconds": 0., "extra_noise_wait_seconds": 0.}
    results = []
    for session in pilot_sessions:
        job = session_job(protocol, session, output / "pilot" / session.session_id, pilot=True)
        results.append(monitored_session(output, protocol, job, ledger))
    durations = [row["elapsed_seconds"] for row in ledger["attempts"] if row.get("returncode") == 0]
    maximum = max(durations, default=float("inf"))
    estimates = {}
    complete = len(results) == 8 and all(row["status"] == "succeeded" for row in results)
    attempt_by_id = {row["session_id"]: row for row in ledger["attempts"]}
    for repeats in (50, 100, 200):
        predicted = 0.
        for row in results if complete else ():
            def mean(name):
                values = row[name]
                return sum(values) / len(values) / 1000 if values else 0.
            forward = mean("raw_synchronized_wall_ms")
            full_path = mean("raw_full_path_ms")
            event = mean("raw_cuda_event_ms")
            throughput_call = row["blocked_throughput"][0]["seconds"] / 20
            worker = attempt_by_id[row["job"]["session_id"]]["elapsed_seconds"]
            extra = (repeats - 20) * (forward + full_path + event) + 130 * throughput_call + 30 * forward
            sessions_per_profile = 110 if row["job"]["kind"] == "control" else 36
            predicted += sessions_per_profile * (worker + extra)
        # Separate trace workers have few calls, but reserve two additional
        # seconds each for profiler/trace work, then apply25% whole-series margin.
        profiler_seconds = 48 * (maximum + 2.)
        estimates[str(repeats)] = (1.25 * (predicted + profiler_seconds) + ledger["charged_seconds"]
            + prepared["preparation_elapsed_seconds"] + 300) if complete else None
    pilot = {"schema": 2, "sessions": results, "attempts": ledger["attempts"],
        "charged_seconds": ledger["charged_seconds"], "admission_wait_seconds": ledger["admission_wait_seconds"],
        "extra_noise_wait_seconds": ledger["extra_noise_wait_seconds"],
        "mandatory_drain_seconds": ledger.get("mandatory_drain_seconds", 0.),
        "maximum_worker_seconds": maximum if math.isfinite(maximum) else None,
        "compute_estimates_by_repeats": estimates,
        "estimate_scope": "pilot process cost plus extrapolated raw pass/call costs, all440 model/control and144 Conv workers,48 traces plus2s/trace;25% margin; heuristic, runtime2h hard guard remains",
        "source_before": source_before, "source_after": source_manifest(), "new_training_updates": 0,
        "decision_required_before_freeze": True}
    write_atomic_json(output / "pilot.json", pilot)
    return pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-packet", type=Path)
    parser.add_argument("--historical", type=Path)
    parser.add_argument("--stage", choices=("prepare", "pilot", "freeze", "timing", "profile"))
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--preflight-seconds", type=float, default=300.)
    parser.add_argument("--worker", choices=("timing",))
    parser.add_argument("--job", type=Path)
    args = parser.parse_args()
    if args.worker:
        timing_worker(read_json(args.job))
    elif args.stage == "prepare":
        if args.first_packet is None or args.historical is None:
            parser.error("prepare requires --first-packet and --historical")
        prepare_sources(args.output, args.first_packet, args.historical)
    elif args.stage == "pilot":
        run_pilot(args.output)
    elif args.stage == "freeze":
        freeze_protocol(args.output, args.repeats, args.preflight_seconds)
    elif args.stage in ("timing", "profile"):
        run_stage(args.output, profiler=args.stage == "profile")
    else:
        parser.error("choose a stage or worker")


if __name__ == "__main__":
    main()
