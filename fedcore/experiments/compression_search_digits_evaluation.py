"""Digits source sealing and an explicitly parameterized full evaluator.

The reused execution pipeline recalculates calibration statistics, copies M0,
checks storage, serializes/reloads and scores validation on every new attempt.
No test tensor is available to this module's source or evaluation context.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from fedcore.algorithm.low_rank.candidate_storage import capture_inventory, inventory_cost
from fedcore.algorithm.low_rank.compression_graph import RuntimeOptions
from fedcore.algorithm.low_rank.grammar_execution import prepare_grammar_space
from fedcore.algorithm.low_rank.plans import ValidationFailure
from .compression_search_evaluation import (
    EvaluationProfile, EvaluationSource, FullEvaluator, PackageFatal, append_event,
    file_sha256, isolated_rng, metrics, model_identity, predict)
from .compression_search_digits_protocol import (
    ATTEMPT_LIMIT, DENSE_TENSOR_BYTES, INITIALIZATION_SEED, MAX_PEAK_BYTES,
    MAX_WORKSPACE_BYTES, ROLE_SIZES, SPLIT_SEEDS, TENSOR_BYTE_BUDGET,
    TUNED_CATALOG, canonical_solution_id, expected_tensor_bytes)
from .protocol import TensorSplit, stable_hash, tensor_hash

DIGITS_PROFILE = EvaluationProfile(DENSE_TENSOR_BYTES, TENSOR_BYTE_BUDGET, 10)


class DigitsDeadline(BaseException):
    """Ordinary exhaustion, distinct from numerical/provenance corruption."""


def effective_deadline(job):
    return min(job[key] for key in ("package_deadline", "compute_admission_deadline", "admission_deadline")
               if job.get(key) is not None)


def check_deadline(job):
    if time.perf_counter() >= effective_deadline(job):
        raise DigitsDeadline("No new Digits computation after its absolute admission deadline")


def rng_signature():
    state = np.random.get_state()
    return {"python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        "numpy": hashlib.sha256(state[1].tobytes() + repr((state[0], *state[2:])).encode()).hexdigest(),
        "torch": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}


class DigitsLedger:
    """One serial append-only writer; incomplete effects never refund a charge."""
    def __init__(self, path, job):
        self.path, self.job = Path(path), job
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line] if self.path.exists() else []
        if any(row.get("attempt") != i for i, row in enumerate(rows, 1)):
            raise PackageFatal("Digits attempt ledger is not contiguous")
        self.used = len(rows)

    def charge(self, purpose, solution_id):
        check_deadline(self.job)
        if self.used >= ATTEMPT_LIMIT:
            raise DigitsDeadline("Global 10300-attempt allowance exhausted")
        self.used += 1
        append_event(self.path, {"event": "mlp_attempt", "attempt": self.used,
            "job": self.job["job"], "purpose": purpose, "solution_id": solution_id,
            "protocol_sha256": self.job["protocol_sha256"], "started_at": time.perf_counter()})
        return self.used


@contextmanager
def count_forward_batches(path, *, job_id, purpose, solution_id):
    """Observe actual complete-model forward calls, including deep copies.

    Global module hooks are scoped to one serial action. The exact five-child
    Sequential/BigMLP root is distinguished from two-child factor modules.
    Hooks are not stored on models and cannot contaminate checkpoints.
    """
    counts = {"started_batches": 0, "completed_batches": 0, "completed_samples": 0}
    def root(module):
        return isinstance(module, nn.Sequential) and len(module) == 5 and isinstance(module[4], nn.Linear) and module[4].out_features == 10
    def before(module, args):
        if root(module):
            counts["started_batches"] += 1
            append_event(Path(path).with_name("forward-calls.jsonl"), {"event": "forward_started",
                "job": job_id, "purpose": purpose, "solution_id": solution_id,
                "started_at": time.perf_counter(), "batch": counts["started_batches"]})
    def after(module, args, output):
        if root(module):
            counts["completed_batches"] += 1
            counts["completed_samples"] += len(output)
            append_event(Path(path).with_name("forward-calls.jsonl"), {"event": "forward_completed",
                "job": job_id, "purpose": purpose, "solution_id": solution_id,
                "completed_at": time.perf_counter(), "batch": counts["completed_batches"], "samples": len(output)})
    first = nn.modules.module.register_module_forward_pre_hook(before)
    second = nn.modules.module.register_module_forward_hook(after)
    started = time.perf_counter()
    try:
        yield counts
    finally:
        first.remove()
        second.remove()
        append_event(path, {"event": "forward_batches", "job": job_id, "purpose": purpose,
            "solution_id": solution_id, "started_at": started, "completed_at": time.perf_counter(), **counts})


class DigitsFullEvaluator(FullEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, profile=DIGITS_PROFILE, **kwargs)

    def __call__(self, workflow):
        identity = canonical_solution_id(workflow.genotype)
        if identity not in self.catalog_ids:
            raise PackageFatal("Candidate escaped the frozen Digits catalog")
        before_rng = rng_signature()
        forward_path = self.events_path.with_name("forward-events.jsonl")
        try:
            with count_forward_batches(forward_path, job_id=self.job_id,
                    purpose="full_validation", solution_id=identity):
                result = super().__call__(workflow)
            if self.rows[-1].get("status") == "finite" and self.rows[-1]["tensor_bytes"] != expected_tensor_bytes(workflow.genotype):
                raise PackageFatal("Digits independent integer storage formula disagrees")
            return result
        finally:
            if rng_signature() != before_rng:
                raise PackageFatal("Full Digits evaluator changed caller RNG state")


def fresh_model(job=None):
    """Normalize existing BigMLP with strict weight/output/storage agreement."""
    from .scenarios import BigMLP
    with isolated_rng():
        torch.manual_seed(INITIALIZATION_SEED)
        original = BigMLP(64, 10).eval()
        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10)).eval()
        model.load_state_dict(original.state_dict(), strict=True)
        probe = torch.arange(192, dtype=torch.float32).reshape(3, 64) / 192.
        observer = (count_forward_batches(Path(job["directory"]) / "forward-events.jsonl", job_id=job["job"],
            purpose="source_normalization", solution_id=f"M0-{job.get('split_seed', 'shared-initialization')}")
            if job is not None else nullcontext())
        with observer:
            if not torch.equal(predict(original, probe), predict(model, probe)):
                raise PackageFatal("Exact Sequential normalization changed BigMLP output")
    if inventory_cost(capture_inventory(model), "bytes") != DENSE_TENSOR_BYTES:
        raise PackageFatal("Digits dense inventory must equal 68904 tensor bytes")
    return model


def load_role(source_dir, role):
    """Open exactly one sealed role, with bytes and tensor-content checks."""
    if role not in ROLE_SIZES:
        raise PackageFatal("Unknown Digits role")
    source_dir = Path(source_dir)
    passport = json.loads((source_dir / "roles.json").read_text(encoding="utf-8"))
    record = passport["roles"][role]
    path = source_dir / record["file"]
    if file_sha256(path) != record["file_sha256"]:
        raise PackageFatal("Sealed Digits role file changed")
    value = torch.load(path, map_location="cpu", weights_only=True)
    split = TensorSplit(value["x"], value["y"], tuple(record["manifest"]["ids"]))
    if split.manifest() != record["manifest"] or len(split.ids) != ROLE_SIZES[role]:
        raise PackageFatal("Sealed Digits role content changed")
    return split


def load_source(job, ledger):
    seed = job["split_seed"]
    if seed not in SPLIT_SEEDS:
        raise PackageFatal("Unknown frozen Digits split seed")
    attempt = ledger.charge("source_baseline_validation", f"M0-{seed}")
    baseline_started = time.perf_counter()
    directory = Path(job["output"]) / "sources" / str(seed)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    roles_path = directory / "roles.json"
    if file_sha256(roles_path) != manifest["roles_sha256"]:
        raise PackageFatal("Digits role passport changed after training")
    path = directory / "baseline.pt"
    if file_sha256(path) != manifest["checkpoint_sha256"]:
        raise PackageFatal("Digits baseline checkpoint changed")
    calibration, validation = (load_role(directory, role) for role in ("calibration", "validation"))
    with isolated_rng():
        model = fresh_model(job)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    if model_identity(model) != manifest["normalized_identity"]:
        raise PackageFatal("Digits M0 identity changed")
    if inventory_cost(capture_inventory(model), "bytes") != DENSE_TENSOR_BYTES:
        raise PackageFatal("Digits M0 storage changed")
    with count_forward_batches(Path(job["directory"]) / "forward-events.jsonl", job_id=job["job"],
            purpose="source_baseline_validation", solution_id=f"M0-{seed}"):
        logits = predict(model, validation.x)
    measured = metrics(logits, validation.y)
    if measured != manifest["baseline_validation"]:
        raise PackageFatal("Digits baseline validation changed after strict reload")
    append_event(Path(job["directory"]) / "baseline-events.jsonl", {"event": "baseline_validation_completed",
        "job": job["job"], "model_seed": seed, "global_attempt": attempt,
        "started_at": baseline_started, "completed_at": time.perf_counter(), "metrics": measured})
    return EvaluationSource(seed, model, calibration, validation, manifest, measured, logits)


def prepare_space(source, catalog):
    options = RuntimeOptions(batch_size=64, max_workspace_bytes=MAX_WORKSPACE_BYTES,
        max_peak_bytes=MAX_PEAK_BYTES, tensor_byte_budget=TENSOR_BYTE_BUDGET)
    space = prepare_grammar_space(source.model, source.calibration.x,
        alternatives={path: catalog.alternatives for path in ("0", "2")},
        graph_version="issue99-local2-digits-frozen-M0-v1", options=options)
    if isinstance(space, ValidationFailure):
        raise PackageFatal(f"Frozen Digits catalog refused: {space}")
    return space


def source_evaluator(job, catalog):
    ledger = DigitsLedger(Path(job["output"]) / "mlp-attempts.jsonl", job)
    source = load_source(job, ledger)
    space = prepare_space(source, catalog)
    context = stable_hash({"profile": "digits-10class-68904-v1", "source": source.manifest["normalized_identity"],
        "calibration": source.calibration.manifest(), "validation": source.validation.manifest(),
        "source_sha256": job["source_sha256"], "protocol_sha256": job["protocol_sha256"],
        "metrics": "numpy-and-torch-stable-CE-v1;macroF1-v1", "quality_rule": "CE+.05,acc-.02",
        "rng_policy": "save/restore Python NumPy Torch; fresh per-run optimizer cache"})
    evaluator = DigitsFullEvaluator(source, space, ledger, Path(job["directory"]) / "events.jsonl",
        context_sha256=context, job_id=job["job"])
    evaluator.catalog_ids = frozenset(canonical_solution_id(value) for value in catalog.genotypes)
    return source, space, evaluator
