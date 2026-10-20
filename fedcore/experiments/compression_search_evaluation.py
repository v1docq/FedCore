"""Owned historical sources and a validation-only, fully charged evaluator.

The experimental evaluator does not train, select on test data or cache
statistics. A full attempt includes transformation, real storage inventory,
checkpoint construction/reload and validation of the reloaded operator.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from fedcore.algorithm.low_rank.candidate_storage import capture_inventory, inventory_cost, inventory_fingerprint
from fedcore.algorithm.low_rank.compression_grammar import compile_genotype
from fedcore.algorithm.low_rank.compression_graph import RuntimeOptions
from fedcore.algorithm.low_rank.grammar_execution import (
    CallbackFailure, NumericalFailure, ResourceFailure, StructuralInvalid,
    WorkflowSuccess, execute_workflow, prepare_grammar_space)
from fedcore.algorithm.low_rank.grammar_search import GrammarFitness
from fedcore.algorithm.low_rank.plans import ValidationFailure
from fedcore.tools.registry.checkpoint_manager import CheckpointManager
from .compression_search_protocol import (
    DENSE_TENSOR_BYTES, G3_TENSOR_BYTE_BUDGET, MAX_PEAK_BYTES,
    MAX_WORKSPACE_BYTES, canonical_solution_id)
from .protocol import TensorSplit, stable_hash, tensor_hash


class PackageFatal(BaseException):
    """Stop the package, including through GOLEM's ordinary exception wrapper."""


SOURCE_HASHES = {
    41: "ff29c69e6347c944362e2a61731f7c010cb668302260dee47944f302f2bc2718",
    42: "5ede6ba1f8658f2db9b047a21a2d97003dc1062200674b0cfa5b6a7b3d61ac40",
    43: "4ff016dabe4ac5d8a76fe292183046a9c9da9cdeb3234a0d59791b5a0c1762bc",
}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append_event(path, event):
    """One owned writer, flushed complete JSON lines; no system TEMP writes."""
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def model_identity(model):
    return stable_hash({"state": {name: tensor_hash(value) for name, value in model.state_dict().items()},
        "modules": [(name, type(layer).__name__, layer.training) for name, layer in model.named_modules()]})


@contextmanager
def isolated_rng():
    python, numpy, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(torch_state)


def metrics(logits, labels):
    """Stable CE and macro-F1 with a separate NumPy reference for CE."""
    if logits.ndim != 2 or labels.shape != (len(logits),) or not bool(torch.isfinite(logits).all()):
        raise PackageFatal("Invalid finite classification output")
    values, targets = logits.detach().double().numpy(), labels.detach().long().numpy()
    maximum = values.max(axis=1)
    ce = float((maximum + np.log(np.exp(values - maximum[:, None]).sum(axis=1))
        - values[np.arange(len(values)), targets]).mean())
    torch_ce = float(nn.functional.cross_entropy(logits.double(), labels.long()))
    if not math.isclose(ce, torch_ce, rel_tol=1e-12, abs_tol=1e-12):
        raise PackageFatal("Independent CE references disagree")
    predicted = values.argmax(axis=1)
    f1 = []
    for category in range(values.shape[1]):
        tp = int(((predicted == category) & (targets == category)).sum())
        fp = int(((predicted == category) & (targets != category)).sum())
        fn = int(((predicted != category) & (targets == category)).sum())
        denominator = 2 * tp + fp + fn
        f1.append(2 * tp / denominator if denominator else 0.)
    return {"cross_entropy": ce, "accuracy": float((predicted == targets).mean()),
        "macro_f1": float(np.mean(f1)), "samples": len(targets)}


@torch.no_grad()
def predict(model, features):
    return torch.cat([model(batch) for batch in features.split(64)])


def historical_splits(seed, repo, *, include_test=False):
    """Recreate exact archived IDs/preprocessing, never generate a new split."""
    from sklearn.datasets import load_breast_cancer
    archived = json.loads((Path(repo) / f"docs/svd_compression/results/local-2026-10-13/runs/mlp-seed-{seed}.json").read_text(encoding="utf-8"))
    data = load_breast_cancer()
    metadata, roles = archived["data"]["metadata"], archived["data"]["roles"]
    digest = lambda value: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    if digest(data.data) != metadata["data_sha256"] or digest(data.target) != metadata["target_sha256"]:
        raise PackageFatal("Historical raw dataset/label hash mismatch")
    ids = {role: tuple(value["ids"]) for role, value in roles.items()}
    flattened = sum((list(value) for value in ids.values()), [])
    if len(flattened) != 569 or len(set(flattened)) != 569 or set(ids) != {"train", "validation", "calibration", "test"}:
        raise PackageFatal("Historical roles overlap or do not cover the source")
    index = {role: np.array([int(value.split(":")[1]) for value in values]) for role, values in ids.items()}
    if set(flattened) != {f"tabular:{i}" for i in range(569)} or list(ids["train"]) != metadata["preprocessing_fit_ids"]:
        raise PackageFatal("Historical role identifiers or train-only preprocessing changed")
    from .scenarios import TrainPreprocessor
    preprocessor = TrainPreprocessor.fit(data.data[index["train"]])
    for name in ("mean", "median", "scale"):
        if not np.allclose(getattr(preprocessor, name), metadata["preprocessing"][name], rtol=1e-13, atol=1e-13):
            raise PackageFatal("Archived preprocessing differs from train-only fit")
    # Use archived values after validating their train-only origin.
    old = metadata["preprocessing"]
    preprocessor = TrainPreprocessor(*(np.array(old[name]) for name in ("median", "mean", "scale")))
    x = preprocessor.transform(data.data)
    splits = {role: TensorSplit(torch.from_numpy(x[rows]), torch.from_numpy(data.target[rows]), ids[role])
        for role, rows in index.items() if role != "test" or include_test}
    if {role: split.manifest() for role, split in splits.items()} != {role: roles[role] for role in splits}:
        raise PackageFatal("Archived role content or shape hashes differ")
    return splits, archived


@dataclass(frozen=True)
class EvaluationSource:
    seed: int
    model: nn.Module
    calibration: TensorSplit
    validation: TensorSplit
    manifest: dict
    baseline_metrics: dict
    baseline_logits: torch.Tensor


def load_source(seed, historical_root, repo):
    path = Path(historical_root) / f"mlp-seed-{seed}/baseline/state_dict.pt"
    if file_sha256(path) != SOURCE_HASHES[seed] or path.stat().st_size != 51_656:
        raise PackageFatal("Historical source checkpoint hash/size mismatch")
    splits, archived = historical_splits(seed, repo)
    from .scenarios import BigMLP
    with isolated_rng():
        original = BigMLP(30, 2).eval()
        model = nn.Sequential(nn.Linear(30, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2)).eval()
        state = torch.load(path, map_location="cpu", weights_only=True)
        original.load_state_dict(state, strict=True)
        model.load_state_dict(state, strict=True)
        if not torch.allclose(predict(original, splits["calibration"].x), predict(model, splits["calibration"].x), rtol=1e-6, atol=1e-7):
            raise PackageFatal("Exact Sequential normalization changed calibration output")
    if inventory_cost(capture_inventory(model), "bytes") != DENSE_TENSOR_BYTES:
        raise PackageFatal("Dense source storage mismatch")
    logits = predict(model, splits["validation"].x)
    measured = metrics(logits, splits["validation"].y)
    old = archived["baseline"]["quality"]["validation"]
    if not math.isclose(measured["cross_entropy"], old["cross_entropy"], rel_tol=1e-5, abs_tol=2e-6) or measured["accuracy"] != old["value"]:
        raise PackageFatal("Historical baseline validation metrics differ")
    manifest = {"seed": seed, "source_path": str(path), "checkpoint_sha256": SOURCE_HASHES[seed],
        "normalized_identity": model_identity(model), "normalized_constructor": "exact nn.Sequential 30-128-64-2 ReLU",
        "historical_record_sha256": file_sha256(Path(repo) / f"docs/svd_compression/results/local-2026-10-13/runs/mlp-seed-{seed}.json"),
        "historical_data": archived["data"], "baseline_validation": measured, "tensor_bytes": DENSE_TENSOR_BYTES}
    # The evaluator receives no train/test tensors or labels.
    return EvaluationSource(seed, model, splits["calibration"], splits["validation"], manifest, measured, logits)


def prepare_space(source, catalog):
    options = RuntimeOptions(batch_size=64, max_workspace_bytes=MAX_WORKSPACE_BYTES,
        max_peak_bytes=MAX_PEAK_BYTES, tensor_byte_budget=G3_TENSOR_BYTE_BUDGET)
    space = prepare_grammar_space(source.model, source.calibration.x,
        alternatives={path: catalog.alternatives for path in ("0", "2")},
        graph_version="issue99-local1-frozen-M0-v1", options=options)
    if isinstance(space, ValidationFailure):
        raise PackageFatal(f"Frozen catalog refused: {space}")
    return space


def roundtrip(model, features):
    manager = CheckpointManager(".", auto_cleanup=False)
    before = capture_inventory(model)
    payload = manager.serialize_to_bytes(model)
    restored = manager.deserialize_from_bytes(payload, device="cpu")
    after = capture_inventory(restored)
    if inventory_fingerprint(before) != inventory_fingerprint(after):
        raise PackageFatal("Checkpoint changed physical tensor inventory")
    graph = lambda value: [(name, type(layer).__name__, layer.training) for name, layer in value.named_modules()]
    if graph(model) != graph(restored) or not torch.equal(predict(model, features), predict(restored, features)):
        raise PackageFatal("Checkpoint changed module graph/modes/numerical output")
    return restored, payload, inventory_cost(after, "bytes")


class AttemptLedger:
    def __init__(self, path, job_id, protocol_sha256, *, admission_deadline=None):
        self.path, self.job_id, self.protocol_sha256 = Path(path), job_id, protocol_sha256
        self.admission_deadline = admission_deadline
        self.used = sum(1 for line in self.path.read_text(encoding="utf-8").splitlines() if line) if self.path.exists() else 0

    def charge(self, purpose, solution_id):
        if self.admission_deadline is not None and time.perf_counter() >= self.admission_deadline:
            raise PackageFatal("No new computation after the package's 115-minute deadline")
        if self.used >= 3300:
            raise PackageFatal("Global 3300-attempt allowance exhausted")
        self.used += 1
        append_event(self.path, {"event": "mlp_attempt", "attempt": self.used, "job": self.job_id,
            "purpose": purpose, "solution_id": solution_id, "protocol_sha256": self.protocol_sha256,
            "started_at": time.perf_counter()})
        return self.used


class FullEvaluator:
    def __init__(self, source, space, ledger, events_path, *, context_sha256, job_id, artifact_dir=None):
        self.source, self.space, self.ledger = source, space, ledger
        self.events_path, self.context_sha256, self.job_id = Path(events_path), context_sha256, job_id
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.rows = []
        self.last_model = None

    def __call__(self, workflow):
        source, started = self.source, time.perf_counter()
        identity = canonical_solution_id(workflow.genotype)
        attempt = self.ledger.charge("full_validation", identity)
        before = model_identity(source.model)
        stage = {}
        row = {"event": "evaluation", "job": self.job_id, "model_seed": source.seed,
            "solution_id": identity, "workflow_fingerprint": workflow.fingerprint,
            "evaluation_context_sha256": self.context_sha256, "global_attempt": attempt,
            "started_at": started}
        outcome = None
        try:
            with isolated_rng():
                phase = time.perf_counter()
                outcome = execute_workflow(source.model, source.calibration.x, workflow)
                stage["transform_seconds"] = time.perf_counter() - phase
                if isinstance(outcome, (CallbackFailure, StructuralInvalid)):
                    raise PackageFatal(f"Unknown callback/structural failure in frozen catalog: {outcome}")
                if not isinstance(outcome, WorkflowSuccess):
                    if not isinstance(outcome, (NumericalFailure, ResourceFailure)):
                        raise PackageFatal(f"Unknown execution outcome: {type(outcome).__name__}")
                    row.update(status=type(outcome).__name__, failure=str(outcome))
                    return outcome
                phase = time.perf_counter()
                restored, payload, storage_bytes = roundtrip(outcome.model, source.validation.x)
                stage["checkpoint_seconds"] = time.perf_counter() - phase
                expected = DENSE_TENSOR_BYTES
                for gene in workflow.genotype.genes:
                    layer = source.model.get_submodule(gene.aliases[0])
                    expected += 4 * (gene.choice.rank * (layer.in_features + layer.out_features) - layer.in_features * layer.out_features)
                if storage_bytes != expected or storage_bytes > G3_TENSOR_BYTE_BUDGET:
                    raise PackageFatal(f"Full inventory disagrees with independent byte formula: {storage_bytes}/{expected}")
                phase = time.perf_counter()
                logits = predict(restored, source.validation.x)
                quality = metrics(logits, source.validation.y)
                accepted = quality["cross_entropy"] <= source.baseline_metrics["cross_entropy"] + .05 and quality["accuracy"] >= source.baseline_metrics["accuracy"] - .02
                stage["validation_seconds"] = time.perf_counter() - phase
                row.update(status="finite", metrics=quality, quality=quality["cross_entropy"] / math.log(2),
                    cost=storage_bytes / DENSE_TENSOR_BYTES, tensor_bytes=storage_bytes,
                    expected_tensor_bytes=expected, quality_accepted=accepted, checkpoint_bytes=len(payload),
                    checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
                    logit_mse=float((logits.double() - source.baseline_logits.double()).square().mean()),
                    inventory_verified_after_reload=True)
                if self.artifact_dir is not None:
                    self.artifact_dir.mkdir(parents=True, exist_ok=True)
                    path = self.artifact_dir / f"{identity}.pt"
                    path.write_bytes(payload)
                    row["artifact_path"] = str(path)
                self.last_model = restored
                return GrammarFitness(row["quality"], row["cost"], accepted)
        except PackageFatal:
            row.update(status="package_fatal")
            raise
        except Exception as error:
            row.update(status="package_fatal", failure=f"{type(error).__name__}: {error}")
            raise PackageFatal(row["failure"]) from error
        finally:
            invariant_failure = None
            try:
                if model_identity(source.model) != before:
                    invariant_failure = "Source model changed during evaluation"
                source.calibration.verify_integrity()
                source.validation.verify_integrity()
            except Exception as error:
                invariant_failure = f"Source data integrity failed: {type(error).__name__}: {error}"
            if invariant_failure is not None:
                row.update(status="package_fatal", failure=invariant_failure)
            row.update(completed_at=time.perf_counter(), stages=stage)
            row["elapsed_seconds"] = row["completed_at"] - started
            self.rows.append(row)
            append_event(self.events_path, row)
            if invariant_failure is not None:
                raise PackageFatal(invariant_failure)


def compile_checked(genotype, space):
    workflow = compile_genotype(genotype, space)
    if isinstance(workflow, ValidationFailure):
        raise PackageFatal(f"Frozen executable solution failed compilation: {workflow}")
    return workflow
