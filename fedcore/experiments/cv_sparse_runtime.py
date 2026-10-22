"""Effect boundaries for the named single-process A100 CV pilot.

The module itself uses the standard library only. Torch is imported after the
explicit execution and free-GPU admission boundary in cv_sparse_pilot.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .cv_sparse_protocol import digest, stage_deadline


class PilotStop(Exception):
    def __init__(self, stage, reason):
        self.stage, self.reason = stage, reason
        super().__init__(f"{stage}: {reason}")


class PilotFatal(BaseException):
    """Unknown numerical/callback failure bypasses GOLEM's retryable failures."""


def classify_cv_candidate_failure(failure):
    """Fail closed: only two exact, declared resource outcomes are candidate failures.

    Numerical/structural/validation errors in this prevalidated finite pilot are
    integrity failures. There is no separately admitted recoverable SVD boundary.
    Imports stay at the execution boundary so dry-run remains free of Torch.
    """
    from fedcore.algorithm.low_rank.grammar_execution import ResourceFailure
    if (type(failure) is ResourceFailure and type(failure.code) is str and type(failure.message) is str
            and failure.code in ("ResourceLimitExceeded", "EvaluationDeadline")):
        return "candidate_failure"
    return "fatal"


def sha256_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024**2), b""):
            hasher.update(part)
    return hasher.hexdigest()


def source_manifest(path, protocol):
    """Bind actual imported package files, without requiring a Git checkout."""
    import fedcore
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    commit = manifest.get("source_commit")
    def is_digest(value, length=64):
        return type(value) is str and len(value) == length and all(c in "0123456789abcdef" for c in value)
    if not is_digest(commit, 40) or not is_digest(manifest.get("wheel_sha256")):
        raise ValueError("Source manifest must name a full commit and wheel SHA-256")
    files = manifest.get("package_files_sha256")
    required = {"fedcore/experiments/" + name + ".py" for name in
        ("cv_sparse_pilot", "cv_sparse_protocol", "cv_sparse_data", "cv_sparse_runtime", "cv_sparse_supervision")}
    required |= {"fedcore/algorithm/low_rank/" + name + ".py" for name in
        ("cv_grammar", "cv_execution", "grammar_search", "low_rank_opt")}
    required |= {"fedcore/algorithm/pruning/sparse/cv_channels.py", "fedcore/algorithm/pruning/sparse/cv_checkpoint.py",
                 "fedcore/tools/registry/checkpoint_manager.py", "fedcore/interfaces/fedcore_optimizer.py"}
    if type(files) is not dict or not required.issubset(files):
        raise ValueError("Manifest must cover the complete current CV pilot and common optimizer implementation")
    root = Path(fedcore.__file__).resolve().parent.parent
    actual = {}
    for name, expected in sorted(files.items()):
        relative = Path(name)
        if (type(name) is not str or relative.is_absolute() or ".." in relative.parts
                or not is_digest(expected)):
            raise ValueError("Invalid package file identity")
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError(f"Missing packaged source: {name}")
        actual[name] = sha256_file(target)
        if actual[name] != expected:
            raise ValueError(f"Imported package source differs from the fixed wheel: {name}")
    if "protocol_sha256" in manifest and manifest["protocol_sha256"] != protocol.sha256:
        raise ValueError("Source manifest names another protocol")
    git = None
    if (root / ".git").exists():
        def command(*args):
            return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()
        git = {"commit": command("rev-parse", "HEAD"), "status": command("status", "--porcelain", "--untracked-files=normal")}
        if git["commit"] != commit or git["status"]:
            raise ValueError("Local checkout must be clean and match the fixed source commit")
    return {"supplied": manifest, "verified_package_files_sha256": actual, "package_root": str(root),
            "manifest_sha256": sha256_file(path), "protocol_sha256": protocol.sha256, "git": git}


def verify_loaded_sources(verified):
    root = Path(verified["package_root"])
    expected = verified["verified_package_files_sha256"]
    result = {}
    for name, module in tuple(sys.modules.items()):
        if name != "fedcore" and not name.startswith("fedcore."):
            continue
        filename = getattr(module, "__file__", None)
        if not filename or not str(filename).endswith(".py"):
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Mixed installed/source package import: {name}")
        relative = path.relative_to(root).as_posix()
        if relative not in expected or sha256_file(path) != expected[relative]:
            raise ValueError(f"An imported FedCore module is absent from the fixed manifest: {name}")
        result[name] = relative
    return result


def preparation(data_root, output_root, manifest_path, protocol):
    """No ML imports, dataset reads, CUDA calls or output directory creation."""
    if Path(output_root).exists():
        raise FileExistsError("Pilot output must be a new directory")
    verified = source_manifest(manifest_path, protocol)
    data = Path(data_root) / "cifar-10-batches-py"
    required = ["batches.meta", "test_batch", *(f"data_batch_{index}" for index in range(1, 6))]
    missing = [name for name in required if not (data / name).is_file()]
    versions = {}
    for package in ("fedcore", "torch", "torchvision", "fedot", "thegolem", "numpy", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"status": "prepared" if not missing else "missing_data", "execute": False,
            "protocol": protocol.payload(), "protocol_sha256": protocol.sha256,
            "source": verified, "missing_cifar_files": missing, "versions": versions}


def admit_free_gpu():
    """Run before importing Torch or creating a CUDA context; never retries."""
    command = ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
    processes = subprocess.run(command, capture_output=True, text=True, check=True, timeout=20)
    if processes.stdout.strip():
        raise ValueError("GPU admission requires zero existing compute processes")
    information = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=20)
    return {"compute_pids": [], "gpu_inventory": information.stdout.strip()}


def execution_deadline(started, protocol, external=None):
    deadline = started + protocol.max_slot_seconds
    if external is not None:
        value = float(external)
        if not math.isfinite(value) or value <= started:
            raise ValueError("External supervisor deadline must be finite and later than execution start")
        deadline = min(deadline, value)
    return deadline


def validate_selected_artifacts(output, artifacts, protocol):
    """Fresh filesystem admission before sealing selection or opening test.

    Pure metadata assertions precede reads; only confined existing checkpoint
    files with matching bytes/SHA and complete phase evidence can be selected.
    """
    def is_digest(value):
        return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    if type(artifacts) is not dict or not 2 <= len(artifacts) <= 7:
        raise ValueError("Select one original dense and at most six completed recovery artifacts")
    original, recovered, recovered_solutions, roles, tape_hashes, admitted = 0, 0, set(), set(), set(), {}
    allowed_roles = {"dense_recovery", "fixed_prune", "fixed_svd", "fixed_hybrid", "random_finalist", "evolution_finalist"}
    for key, artifact in artifacts.items():
        if type(artifact) is not dict or not is_digest(key) or artifact.get("artifact_id") != key:
            raise ValueError("Every selected artifact requires its exact canonical identity")
        phase, solution, checksum = artifact.get("phase"), artifact.get("solution_id"), artifact.get("sha256")
        if (not is_digest(solution) or not is_digest(checksum) or phase not in ("dense_original", "recovered_200_updates")
                or artifact.get("completed_within_deadline") is not True
                or type(artifact.get("file_bytes")) is not int or artifact["file_bytes"] <= 0
                or type(artifact.get("tensor_bytes")) is not int or not 0 < artifact["tensor_bytes"] <= 64 * 1024**2
                or key != digest({"solution_id": solution, "checkpoint_sha256": checksum, "phase": phase})):
            raise ValueError("Selected artifact is incomplete, unsupported or has a forged identity")
        if phase == "dense_original":
            original += 1
            proof, validation = artifact.get("training", {}), artifact.get("validation", {})
            if (proof.get("completed_epochs") != protocol.baseline_epochs
                    or proof.get("all_parameters_received_finite_gradients") is not True
                    or proof.get("all_parameter_tensors_changed") is not True
                    or type(validation.get("top1")) not in (int, float) or not math.isfinite(validation["top1"])
                    or validation["top1"] < protocol.baseline_min_top1):
                raise ValueError("Original dense artifact lacks complete baseline admission")
        else:
            recovered += 1
            if (type(artifact.get("recovery_updates")) is not int or artifact["recovery_updates"] != protocol.recovery_updates
                    or solution in recovered_solutions or not is_digest(artifact.get("recovery_tape_sha256"))
                    or type(artifact.get("roles")) is not list or not artifact["roles"]
                    or any(type(role) is not str or role not in allowed_roles or role in roles for role in artifact["roles"])
                    or len(artifact["roles"]) != len(set(artifact["roles"]))):
                raise ValueError("Recovery selection must be complete, deduplicated and use the frozen update tape")
            recovered_solutions.add(solution)
            tape_hashes.add(artifact["recovery_tape_sha256"])
            roles.update(artifact["roles"])
        relative = artifact.get("path")
        if type(relative) is not str or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Selected checkpoint paths must be confined relative paths")
        target = (output.root / relative).resolve()
        if not target.is_relative_to(output.root) or not target.is_file():
            raise ValueError("Selected checkpoint is missing or escapes the pilot output")
        if target.stat().st_size != artifact["file_bytes"] or sha256_file(target) != checksum:
            raise ValueError("Selected checkpoint bytes changed before test-set admission")
        admitted[key] = {"path": relative, "sha256": checksum, "file_bytes": artifact["file_bytes"], "phase": phase,
                         "solution_id": solution, "tensor_bytes": artifact["tensor_bytes"]}
    if original != 1 or not 1 <= recovered <= 6 or "dense_recovery" not in roles or len(tape_hashes) != 1:
        raise ValueError("Seal exactly one baseline, a complete dense recovery and one common recovery tape")
    return {"artifacts": admitted, "artifact_count": len(admitted), "roles": sorted(roles),
            "recovery_tape_sha256": next(iter(tape_hashes)), "admission_sha256": digest(admitted)}


class Output:
    def __init__(self, path):
        self.root = Path(path).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.ledger = self.root / "ledger.jsonl"
        self.heartbeat = self.root / "heartbeats.jsonl"

    def _target(self, relative):
        target = (self.root / relative).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError("Artifact must remain within the new pilot output directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def json(self, relative, value):
        target = self._target(relative)
        temporary = target.with_name(target.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        return str(target.relative_to(self.root))

    def bytes(self, relative, value):
        target = self._target(relative)
        temporary = target.with_name(target.name + ".tmp")
        with temporary.open("wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        return {"path": str(target.relative_to(self.root)), "sha256": sha256_file(target), "file_bytes": target.stat().st_size}

    def append(self, event, *, heartbeat=False):
        target = self.heartbeat if heartbeat else self.ledger
        with target.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


class Runtime:
    def __init__(self, protocol, output, started):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["OMP_NUM_THREADS"] = str(protocol.cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(protocol.cpu_threads)
        import torch
        self.torch, self.protocol, self.output = torch, protocol, output
        self.started, self.deadline = started, execution_deadline(started, protocol,
            os.environ.get("FEDCORE_PILOT_ABSOLUTE_DEADLINE"))
        self.stage, self.stage_started, self.stage_limit, self.last_heartbeat = "manifest", started, self.deadline, 0.0
        self.stage_limit = stage_deadline(started, self.deadline, "manifest")
        torch.set_num_threads(protocol.cpu_threads)
        torch.set_num_interop_threads(protocol.cpu_threads)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if time.perf_counter() >= self.stage_limit:
            raise PilotStop("manifest", "absolute_deadline_before_cuda_context")
        if torch.cuda.device_count() != 1:
            raise ValueError("The named pilot requires exactly one visible GPU")
        properties = torch.cuda.get_device_properties(0)
        if "A100" not in properties.name or not 38 * 1024**3 <= properties.total_memory <= 41 * 1024**3:
            raise ValueError("The named pilot requires one A100 40 GB")
        torch.manual_seed(protocol.model_seed)
        torch.cuda.manual_seed_all(protocol.model_seed)
        self.environment = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "device": properties.name, "total_memory": properties.total_memory,
            "cpu_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "tf32": False,
            "environment": {name: os.environ.get(name) for name in ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONHASHSEED")}}

    @contextmanager
    def phase(self, name):
        self.stage, self.stage_started = name, self.started if name == "manifest" else time.perf_counter()
        self.stage_limit = stage_deadline(self.stage_started, self.deadline, name)
        self.output.append({"event": "stage_started", "stage": name, "started_at": self.stage_started, "deadline": self.stage_limit})
        self.check()
        try:
            yield self.stage_limit
        finally:
            self.output.append({"event": "stage_ended", "stage": name, "ended_at": time.perf_counter(),
                                "seconds": time.perf_counter() - self.stage_started})

    def check(self, deadline=None, **progress):
        now = time.perf_counter()
        limit = min(self.deadline, self.stage_limit, deadline if deadline is not None else self.deadline)
        if now >= limit:
            raise PilotStop(self.stage, "absolute_deadline")
        allocated = self.torch.cuda.memory_allocated()
        reserved = self.torch.cuda.memory_reserved()
        if max(allocated, reserved) > self.protocol.max_vram_bytes:
            raise PilotStop(self.stage, "vram_limit")
        if now - self.last_heartbeat >= 5:
            self.output.append({"event": "heartbeat", "stage": self.stage, "timestamp": now,
                "elapsed_seconds": now - self.started, "deadline": limit, "allocated_bytes": allocated,
                "reserved_bytes": reserved, "peak_allocated_bytes": self.torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": self.torch.cuda.max_memory_reserved(), **progress}, heartbeat=True)
            self.last_heartbeat = now


class GPUData:
    def __init__(self, data, runtime):
        torch = runtime.torch
        self.runtime, self.data, self.torch = runtime, data, torch
        self.images = torch.from_numpy(data.images).permute(0, 3, 1, 2).contiguous().to("cuda")
        self.labels = torch.from_numpy(data.labels).to("cuda")
        self.mean = torch.tensor(data.mean, device="cuda").reshape(1, 3, 1, 1)
        self.std = torch.tensor(data.std, device="cuda").reshape(1, 3, 1, 1)
        runtime.check()

    def normalized(self, raw):
        return (raw.float().div(255) - self.mean) / self.std

    def training(self, indices, seed):
        torch = self.torch
        raw = self.images[torch.as_tensor(indices, dtype=torch.long, device="cuda")]
        generator = torch.Generator(device="cuda").manual_seed(seed)
        offsets = torch.randint(0, 9, (len(indices), 2), generator=generator, device="cuda")
        flips = torch.rand(len(indices), generator=generator, device="cuda") < .5
        padded = torch.nn.functional.pad(raw, (4, 4, 4, 4))
        rows = offsets[:, 0, None] + torch.arange(32, device="cuda")[None, :]
        columns = offsets[:, 1, None] + torch.arange(32, device="cuda")[None, :]
        positions = (rows[:, :, None] * 40 + columns[:, None, :]).reshape(len(indices), 1, 1024)
        cropped = padded.flatten(2).gather(2, positions.expand(-1, 3, -1)).reshape(-1, 3, 32, 32)
        cropped = torch.where(flips[:, None, None, None], cropped.flip(-1), cropped)
        return self.normalized(cropped), self.labels[torch.as_tensor(indices, dtype=torch.long, device="cuda")]

    def batches(self, indices, batch=128):
        for start in range(0, len(indices), batch):
            ids = indices[start:start + batch]
            rows = self.torch.as_tensor(ids, dtype=self.torch.long, device="cuda")
            yield ids, self.normalized(self.images[rows]), self.labels[rows]


def recovery_tape(indices, protocol):
    """One immutable common row/augmentation tape for every recovered artifact."""
    import numpy as np
    generator = np.random.default_rng(protocol.recovery_seed)
    ordered = generator.permutation(np.asarray(indices, dtype=np.int64))
    needed = protocol.recovery_updates * protocol.recovery_batch
    if needed > len(ordered):
        raise ValueError("This frozen recovery tape expects at most one training permutation")
    return tuple((tuple(int(value) for value in ordered[start:start + protocol.recovery_batch]),
                  int(generator.integers(0, 2**31))) for start in range(0, needed, protocol.recovery_batch))


def calibrate_bn(model, gpu, deadline):
    runtime, torch = gpu.runtime, gpu.torch
    model.eval()
    batch_norms = [layer for layer in model.modules() if type(layer) is torch.nn.BatchNorm2d]
    momenta = [layer.momentum for layer in batch_norms]
    try:
        for layer in batch_norms:
            layer.reset_running_stats()
            layer.momentum = None
            layer.train()
        with torch.no_grad():
            # This stage accesses image IDs only, never calibration labels.
            ids = gpu.data.roles.calibration
            for start in range(0, len(ids), 128):
                runtime.check(deadline, calibration_rows=start)
                rows = torch.as_tensor(ids[start:start + 128], dtype=torch.long, device="cuda")
                model(gpu.normalized(gpu.images[rows]))
                runtime.check(deadline, calibration_rows=min(start + 128, len(ids)))
    finally:
        for layer, momentum in zip(batch_norms, momenta):
            layer.momentum = momentum
        model.eval()


def evaluate(model, gpu, indices, *, deadline, label, test_data=None):
    """Persist per-row labels, predictions and CE for independent recounting."""
    import numpy as np
    runtime, torch = gpu.runtime, gpu.torch
    losses, predictions, targets, row_ids = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), 128):
            runtime.check(deadline, evaluation=label, rows=start)
            ids = indices[start:start + 128]
            if test_data is None:
                rows = torch.as_tensor(ids, dtype=torch.long, device="cuda")
                x, y = gpu.normalized(gpu.images[rows]), gpu.labels[rows]
            else:
                images, labels = test_data
                raw = torch.from_numpy(np.ascontiguousarray(images[np.asarray(ids)])).permute(0, 3, 1, 2).contiguous().to("cuda")
                x, y = gpu.normalized(raw), torch.from_numpy(np.asarray(labels[np.asarray(ids)], dtype=np.int64)).to("cuda")
            logits = model(x)
            per_row = torch.nn.functional.cross_entropy(logits, y, reduction="none")
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(per_row).all()):
                raise ValueError("Nonfinite logits or cross entropy")
            losses.extend(per_row.double().cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
            targets.extend(y.cpu().tolist())
            row_ids.extend(int(value) for value in ids)
            runtime.check(deadline, evaluation=label, rows=min(start + 128, len(indices)))
    path = runtime.output._target(f"predictions/{label}.npz")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, row_ids=np.asarray(row_ids), labels=np.asarray(targets),
                            predictions=np.asarray(predictions), cross_entropy=np.asarray(losses, dtype=np.float64))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    correct = sum(first == second for first, second in zip(predictions, targets))
    return {"rows": len(targets), "correct": correct, "top1": correct / len(targets),
            "cross_entropy": sum(losses) / len(losses), "prediction_file": str(path.relative_to(runtime.output.root)),
            "prediction_sha256": sha256_file(path)}


def checkpoint(model, gpu, *, label, solution_id, phase, deadline):
    """Persist an owned model, independently reconstruct it, and check probes."""
    from fedcore.tools.registry.checkpoint_manager import CheckpointManager
    from fedcore.algorithm.low_rank.candidate_storage import capture_inventory, inventory_cost, inventory_fingerprint
    runtime, torch = gpu.runtime, gpu.torch
    runtime.check(deadline, checkpoint=label)
    model.eval().requires_grad_(False)
    model._fedcore_method_evidence = {**getattr(model, "_fedcore_method_evidence", {}),
        "pilot_artifact": {"protocol_sha256": runtime.protocol.sha256, "solution_id": solution_id, "phase": phase}}
    ids = torch.tensor(gpu.data.roles.calibration[:2], dtype=torch.long, device="cuda")
    probe = gpu.normalized(gpu.images[ids])
    with torch.inference_mode():
        expected = model.to("cuda")(probe).cpu()
    model.cpu()
    manager = CheckpointManager(".", auto_cleanup=False)
    payload = manager.serialize_to_bytes(model)
    record = runtime.output.bytes(f"checkpoints/{label}.pt", payload)
    restored = manager.deserialize_from_bytes(payload, device="cpu")
    inventory = capture_inventory(model)
    if inventory_fingerprint(capture_inventory(restored)) != inventory_fingerprint(inventory):
        raise ValueError("Independent checkpoint restoration changed physical state")
    with torch.inference_mode():
        cpu_logits = restored(probe.cpu())
        torch.testing.assert_close(cpu_logits, expected, atol=2e-4, rtol=2e-4)
        reloaded = restored.to("cuda")(probe).cpu()
    if not torch.equal(reloaded, expected):
        raise ValueError("Independent checkpoint reload must preserve exact same-device GPU logits")
    restored.cpu()
    runtime.check(deadline, checkpoint=label)
    record.update({"solution_id": solution_id, "phase": phase, "tensor_bytes": inventory_cost(inventory, "bytes"),
        "inventory_fingerprint": inventory_fingerprint(inventory), "gpu_reload_exact": True,
        "cpu_gpu_probe_max_abs": float((cpu_logits - expected).abs().max()), "completed_within_deadline": True})
    record["artifact_id"] = digest({"solution_id": solution_id, "checkpoint_sha256": record["sha256"], "phase": phase})
    return record, restored


def train_baseline(gpu, deadline):
    import numpy as np
    from fedcore.algorithm.pruning.sparse.cv_channels import make_cifar_resnet18
    from fedcore.tools.registry.checkpoint_manager import CheckpointManager
    runtime, torch, protocol = gpu.runtime, gpu.torch, gpu.runtime.protocol
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(protocol.model_seed)
        model = make_cifar_resnet18()
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    model.to("cuda").train().requires_grad_(True)
    optimizer = torch.optim.SGD(model.parameters(), lr=protocol.baseline_lr, momentum=protocol.momentum,
        weight_decay=protocol.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=protocol.baseline_epochs)
    gradient_names, updates = set(), 0
    for epoch in range(protocol.baseline_epochs):
        ordered = np.random.default_rng(protocol.model_seed + epoch).permutation(np.asarray(gpu.data.roles.train))
        total_loss, rows = 0.0, 0
        for start in range(0, len(ordered), protocol.baseline_batch):
            runtime.check(deadline, epoch=epoch + 1, updates=updates)
            ids = ordered[start:start + protocol.baseline_batch].tolist()
            x, y = gpu.training(ids, protocol.model_seed + updates)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite baseline loss")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise ValueError(f"Baseline parameter has no finite training gradient: {name}")
                gradient_names.add(name)
            optimizer.step()
            updates += 1
            total_loss += float(loss) * len(ids)
            rows += len(ids)
            runtime.check(deadline, epoch=epoch + 1, updates=updates)
        scheduler.step()
        runtime.output.append({"event": "baseline_epoch", "epoch": epoch + 1, "updates": updates,
            "rows": rows, "cross_entropy": total_loss / rows, "next_lr": scheduler.get_last_lr()[0], "timestamp": time.perf_counter()})
        if (epoch + 1) % 20 == 0:
            payload = CheckpointManager(".", auto_cleanup=False).serialize_to_bytes(model)
            runtime.output.bytes(f"partial/baseline_epoch_{epoch + 1:03d}.pt", payload)
            runtime.check(deadline)
    model.eval()
    metrics = evaluate(model, gpu, gpu.data.roles.search_validation, deadline=deadline, label="baseline_validation")
    changed = {name: not torch.equal(value.detach().cpu(), initial[name]) for name, value in model.named_parameters()}
    proof = {"completed_epochs": protocol.baseline_epochs, "updates": updates,
             "all_parameters_received_finite_gradients": set(changed) == gradient_names,
             "all_parameter_tensors_changed": all(changed.values()), "parameter_tensor_count": len(changed),
             "changed_parameter_tensors": changed}
    runtime.output.json("baseline_training_admission.json", {"validation": metrics, "training": proof,
        "minimum_top1": protocol.baseline_min_top1, "accepted": metrics["top1"] >= protocol.baseline_min_top1
            and proof["all_parameters_received_finite_gradients"] and proof["all_parameter_tensors_changed"]})
    if (metrics["top1"] < protocol.baseline_min_top1 or not proof["all_parameters_received_finite_gradients"]
            or not proof["all_parameter_tensors_changed"]):
        raise PilotStop("baseline", "baseline_quality_or_training_admission")
    return model, metrics, proof


def recover(model, gpu, tape, *, deadline, role):
    runtime, torch, protocol = gpu.runtime, gpu.torch, gpu.runtime.protocol
    model.to("cuda").train().requires_grad_(True)
    optimizer = torch.optim.SGD(model.parameters(), lr=protocol.recovery_lr, momentum=protocol.momentum,
        weight_decay=protocol.weight_decay, nesterov=True)
    for update, (ids, seed) in enumerate(tape, 1):
        runtime.check(deadline, recovery=role, updates=update - 1)
        x, y = gpu.training(ids, seed)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite recovery loss")
        loss.backward()
        if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
            raise ValueError("Recovery has missing or nonfinite gradients")
        optimizer.step()
        runtime.check(deadline, recovery=role, updates=update)
    if len(tape) != protocol.recovery_updates:
        raise ValueError("Recovery tape does not contain the frozen update count")
    calibrate_bn(model, gpu, deadline)
    model._fedcore_method_evidence = {**getattr(model, "_fedcore_method_evidence", {}),
        "caller_owned_recovery": {"updates": len(tape), "tape_sha256": digest(tape),
            "optimizer": "fresh_SGD_momentum_0.9_nesterov_weight_decay_0.0005", "lr": protocol.recovery_lr,
            "grammar_recovery": "NoRecovery"}}
    return evaluate(model, gpu, gpu.data.roles.search_validation, deadline=deadline, label=f"recovery_{role}_validation")


def torchscript_artifact(artifact, gpu, deadline):
    """Trace an independent checkpoint reload and time the reloaded TS file."""
    from fedcore.tools.registry.checkpoint_manager import CheckpointManager
    runtime, torch = gpu.runtime, gpu.torch
    runtime.check(deadline, export=artifact["artifact_id"])
    existing_record = runtime.output.root / f"torchscript/{artifact['artifact_id']}.json"
    if existing_record.is_file():
        record = json.loads(existing_record.read_text(encoding="utf-8"))
        return independent_torchscript(record, gpu, deadline), record
    path = runtime.output.root / artifact["path"]
    if sha256_file(path) != artifact["sha256"]:
        raise ValueError("Frozen checkpoint changed before export")
    model = CheckpointManager(".", auto_cleanup=False).deserialize_from_bytes(path.read_bytes(), device="cpu").eval()
    probe = gpu.data.rows(gpu.data.roles.calibration[:2])
    with torch.inference_mode():
        expected_cpu = model(probe)
        traced = torch.jit.trace(model, probe, check_trace=True)
        torch.testing.assert_close(traced(probe), expected_cpu, atol=1e-5, rtol=1e-5)
    target = runtime.output._target(f"torchscript/{artifact['artifact_id']}.pt")
    temporary = target.with_name(target.name + ".tmp")
    torch.jit.save(traced, str(temporary))
    os.replace(temporary, target)
    restored = torch.jit.load(str(target), map_location="cpu").eval()
    with torch.inference_mode():
        torch.testing.assert_close(restored(probe), expected_cpu, atol=1e-5, rtol=1e-5)
        expected_gpu = model.to("cuda")(probe.to("cuda"))
        restored.to("cuda")
        actual_gpu = restored(probe.to("cuda"))
        torch.testing.assert_close(actual_gpu, expected_gpu, atol=1e-5, rtol=1e-5)
    del model, traced
    runtime.check(deadline, export=artifact["artifact_id"])
    record = {"path": str(target.relative_to(runtime.output.root)), "sha256": sha256_file(target),
        "file_bytes": target.stat().st_size, "artifact_id": artifact["artifact_id"], "format": "independently_reloaded_TorchScript"}
    runtime.output.json(f"torchscript/{artifact['artifact_id']}.json", record)
    return restored, record


def independent_torchscript(record, gpu, deadline):
    runtime = gpu.runtime
    runtime.check(deadline, export=record["artifact_id"])
    path = (runtime.output.root / record["path"]).resolve()
    if not path.is_relative_to(runtime.output.root) or sha256_file(path) != record["sha256"]:
        raise ValueError("Frozen TorchScript artifact changed before independent reload")
    result = gpu.torch.jit.load(str(path), map_location="cuda").eval()
    runtime.check(deadline)
    return result


def paired_latency(first, second, gpu, *, deadline, label, aa=False):
    """Fixed AB/BA pairs; CUDA-event and synchronized-wall times stay separate."""
    from .cv_sparse_protocol import aa_admission
    if aa and first is second:
        raise ValueError("A/A requires two independently loaded module objects")
    runtime, torch, protocol = gpu.runtime, gpu.torch, gpu.runtime.protocol
    result = {"label": label, "aa": aa, "batches": {}}
    with torch.inference_mode():
        for batch in (1, 128):
            x = gpu.data.rows(gpu.data.roles.calibration[:batch]).to("cuda")
            for model in (first, second):
                for _ in range(protocol.timing_warmup):
                    runtime.check(deadline, latency=label, batch=batch)
                    model(x)
            torch.cuda.synchronize()
            def measure(model):
                runtime.check(deadline, latency=label, batch=batch)
                torch.cuda.synchronize()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                wall = time.perf_counter()
                begin.record()
                for _ in range(protocol.timing_calls):
                    model(x)
                end.record()
                end.synchronize()
                synchronized_wall = (time.perf_counter() - wall) / protocol.timing_calls
                event = begin.elapsed_time(end) / 1000 / protocol.timing_calls
                runtime.check(deadline, latency=label, batch=batch)
                return {"cuda_event_seconds": event, "synchronized_wall_seconds": synchronized_wall}
            pairs = []
            for index in range(protocol.aa_pairs):
                order = "AB" if index % 2 == 0 else "BA"
                observations = {}
                for key in order:
                    observations[key] = measure(first if key == "A" else second)
                pairs.append({"pair": index, "order": order, **observations,
                    "cuda_event_ratio_B_over_A": observations["B"]["cuda_event_seconds"] / observations["A"]["cuda_event_seconds"],
                    "synchronized_wall_ratio_B_over_A": observations["B"]["synchronized_wall_seconds"] / observations["A"]["synchronized_wall_seconds"]})
                runtime.output.append({"event": "latency_pair", "label": label, "batch": batch, **pairs[-1]})
            record = {"pairs": pairs, "orders": [item["order"] for item in pairs]}
            if aa:
                record["cuda_event_admission"] = aa_admission([item["cuda_event_ratio_B_over_A"] for item in pairs], record["orders"])
                record["synchronized_wall_admission"] = aa_admission([item["synchronized_wall_ratio_B_over_A"] for item in pairs], record["orders"])
                record["accepted"] = record["cuda_event_admission"]["accepted"] and record["synchronized_wall_admission"]["accepted"]
            result["batches"][str(batch)] = record
    result["accepted"] = all(value.get("accepted", True) for value in result["batches"].values())
    runtime.output.json(f"latency/{label}.json", result)
    return result
