"""Charged, validation-only evaluation for the third local issue-99 package.

The old package remains frozen. Every callback executes a complete transform,
physical inventory, checkpoint round trip and held-out validation. A store is
an explicit caller-owned capability, never an objective/model cache.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from fedcore.algorithm.low_rank.grammar_execution import (
    CallbackFailure, NumericalFailure, ResourceFailure, StructuralInvalid,
    WorkflowSuccess, execute_workflow)
from fedcore.algorithm.low_rank.grammar_search import GrammarFitness
from .compression_search_digits_evaluation import (
    DigitsDeadline, check_deadline, count_forward_batches, load_source,
    prepare_space, rng_signature)
from .compression_search_digits_protocol import (
    DENSE_TENSOR_BYTES, TENSOR_BYTE_BUDGET, TUNED_CATALOG,
    canonical_solution_id, expected_tensor_bytes)
from .compression_search_evaluation import (
    PackageFatal, append_event, isolated_rng, metrics,
    model_identity, predict, roundtrip)
from .compression_search_local2 import write_json
from .protocol import stable_hash, tensor_hash

MLP_ATTEMPT_LIMIT = 2700
STATISTICS_MAX_BYTES = 8 * 1024**2


def interval_union(intervals, *, start=None, end=None):
    """Length of the clipped union; nested intervals are never added twice."""
    clipped = []
    for first, last in intervals:
        first = max(first, start) if start is not None else first
        last = min(last, end) if end is not None else last
        if last > first:
            clipped.append((first, last))
    total, boundary = 0., None
    for first, last in sorted(clipped):
        if boundary is None or first > boundary:
            total += last-first
        else:
            total += max(0., last-boundary)
        boundary = max(last, boundary) if boundary is not None else last
    return total


class PhaseRecorder:
    """Absolute intervals, with explicit parent names and uncovered residuals."""
    def __init__(self):
        self.rows = []

    @contextmanager
    def measure(self, phase, *, parent=None, **metadata):
        first = time.perf_counter()
        try:
            yield
        finally:
            self.rows.append(dict(phase=phase, parent=parent, started_at=first,
                completed_at=time.perf_counter(), **metadata))

    def observe_transform(self, event):
        for row in reversed(self.rows):
            if (row.get("state") == "started" and row["phase"] == event["phase"]
                    and row.get("started_at") == event["started_at"]):
                row.update(event)
                return
        self.rows.append(dict(event, parent="fixed_profile" if "node" in event else "transform"))

    def partition(self, first, last, phases):
        children = [(row["started_at"], row["completed_at"]) for row in self.rows
                    if row["phase"] in phases and row.get("completed_at") is not None]
        covered = interval_union(children, start=first, end=last)
        return dict(total_seconds=last-first, covered_seconds=covered,
                    uncovered_seconds=max(0., last-first-covered))


class Local3Ledger:
    """One serial append-only global ledger; charge before every MLP effect."""
    def __init__(self, path, job):
        self.path, self.job = Path(path), job
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()] if self.path.exists() else []
        if any(row.get("attempt") != index for index, row in enumerate(rows, 1)):
            raise PackageFatal("Local3 MLP attempt ledger is not contiguous")
        if any(row.get("protocol_sha256") != job["protocol_sha256"] for row in rows):
            raise PackageFatal("Local3 MLP attempt ledger belongs to a different protocol")
        self.used, self.active = len(rows), None

    def charge(self, purpose, solution_id):
        check_deadline(self.job)
        if self.active is not None:
            raise PackageFatal("Previous charged MLP action was not completed")
        if self.used >= MLP_ATTEMPT_LIMIT:
            raise DigitsDeadline("Global 2700-action Local3 MLP allowance exhausted")
        now = time.perf_counter()
        self.used += 1
        row = dict(event="mlp_attempt", attempt=self.used, job=self.job["job"],
            purpose=purpose, solution_id=solution_id,
            protocol_sha256=self.job["protocol_sha256"], started_at=now)
        append_event(self.path, row)
        hard = min(value for value in (self.job.get("hard_deadline"),
            self.job.get("package_deadline")) if value is not None)
        if self.job["kind"] != "search_diagnostic":
            hard = min(hard, now+20.)
        self.active = dict(row, active=True, hard_deadline=hard)
        write_json(Path(self.job["directory"])/"active-request.json", self.active)
        return self.used

    def complete(self):
        if self.active is not None:
            write_json(Path(self.job["directory"])/"active-request.json",
                dict(self.active, active=False, completed_at=time.perf_counter()))
            self.active = None


def load_local3_source(job, ledger):
    """Adapt the old sealed loader to a direct sources root without copying."""
    root = Path(job["source_root"]).resolve()
    if root.name != "sources":
        raise PackageFatal("Local3 source_root must name the saved sources directory")
    expected = job.get("expected_source_files")
    if expected is not None:
        for name, digest in expected.items():
            path = (root/name).resolve()
            if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise PackageFatal("Local3 source differs from the frozen package file hashes")
    adapted = dict(job, output=str(root.parent))
    try:
        # This loader opens calibration/validation only, charges once before its
        # first forward and verifies role/checkpoint/tensor hashes and M0 CE.
        return load_source(adapted, ledger)
    finally:
        ledger.complete()


@torch.no_grad()
def dense_operator_snapshots(model, source):
    """Compare reconstructed affine operators, independent of SVD factor signs."""
    from fedcore.models.network_impl.decomposed_layers import IDecomposed
    result = {}
    for path in ("0", "2"):
        original, layer = source.model.get_submodule(path), model.get_submodule(path)
        weight = layer._get_composed_weight() if isinstance(layer, IDecomposed) else layer.weight
        bias = layer.bias if layer.bias is not None else torch.zeros(original.out_features, dtype=weight.dtype)
        result[path] = {"weight": weight.detach().clone(), "bias": bias.detach().clone()}
    return result


def assert_same_result(left, right, labels):
    """Strict numerical admission; no tolerance or quality-based skipping."""
    for name in ("outcome", "status", "tensor_bytes", "quality_accepted", "metrics"):
        if left.get(name) != right.get(name):
            raise PackageFatal(f"C0/C1 result differs: {name}")
    for side in (left, right):
        if side.get("status") == "finite" and metrics(side["logits"], labels) != side["metrics"]:
            raise PackageFatal("Stored validation logits do not reproduce recorded CE/accuracy")
    if left.get("status") != "finite":
        return
    if not torch.equal(left["logits"], right["logits"]):
        raise PackageFatal("C0/C1 validation logits differ at rtol=0, atol=0")
    if left["operators"].keys() != right["operators"].keys():
        raise PackageFatal("C0/C1 operator scopes differ")
    for path, operator in left["operators"].items():
        for name in ("weight", "bias"):
            if not torch.equal(operator[name], right["operators"][path][name]):
                raise PackageFatal("C0/C1 reconstructed dense operators differ at rtol=0, atol=0")


class Local3FullEvaluator:
    """Complete callback with narrow injectable effects and actual intervals."""
    def __init__(self, source, space, ledger, events_path, *, context_sha256,
                 job_id, store=None, capture_tensors=False, transform=execute_workflow,
                 checkpoint=roundtrip, prediction=predict):
        self.source, self.space, self.ledger = source, space, ledger
        self.events_path, self.context_sha256, self.job_id = Path(events_path), context_sha256, job_id
        self.store, self.capture_tensors = store, capture_tensors
        self.transform, self.checkpoint, self.prediction = transform, checkpoint, prediction
        self.catalog_ids = frozenset(canonical_solution_id(value) for value in TUNED_CATALOG.genotypes)
        self.rows, self.callback_intervals, self.tensor_results = [], [], []
        self.last_model = None
        self.stream_metadata = {}

    def __call__(self, workflow):
        first, row_count = time.perf_counter(), len(self.rows)
        try:
            return self._evaluate(workflow)
        finally:
            completed = time.perf_counter()
            if len(self.rows) > row_count:
                row = self.rows[-1]
                row.update(outer_started_at=first, outer_completed_at=completed)
                append_event(self.events_path, dict(event="outer_callback_completed",
                    job=self.job_id, global_attempt=row["global_attempt"],
                    outer_started_at=first, outer_completed_at=completed,
                    definition="after evaluation publication and active-request completion; before this diagnostic marker"))
            # This outer boundary includes source checks, all logging/counters
            # and active-request completion. It is what search subtracts.
            final = time.perf_counter()
            self.callback_intervals.append((first, final))
            if len(self.rows) > row_count:
                self.rows[-1]["outer_completed_at"] = final

    def _evaluate(self, workflow):
        source, started = self.source, time.perf_counter()
        identity = canonical_solution_id(workflow.genotype)
        if identity not in self.catalog_ids:
            raise PackageFatal("Candidate escaped the frozen Local3 Digits catalog")
        attempt = self.ledger.charge("full_validation", identity)
        timer, before_rng = PhaseRecorder(), rng_signature()
        row = dict(event="evaluation", job=self.job_id, split_seed=source.seed,
            solution_id=identity, workflow_fingerprint=workflow.fingerprint,
            evaluation_context_sha256=self.context_sha256, global_attempt=attempt,
            started_at=started, **self.stream_metadata)
        outcome, tensors, invariant_failure = None, None, None
        try:
            with timer.measure("source_identity", parent="evaluator"):
                before = model_identity(source.model)
            with isolated_rng(), count_forward_batches(self.events_path.with_name("forward-events.jsonl"),
                    job_id=self.job_id, purpose="full_validation", solution_id=identity):
                with timer.measure("transform", parent="evaluator"):
                    outcome = self.transform(source.model, source.calibration.x, workflow,
                        store=self.store, phase_callback=timer.observe_transform)
                row["outcome"] = type(outcome).__name__
                if isinstance(outcome, (CallbackFailure, StructuralInvalid)):
                    raise PackageFatal(f"Unknown failure in frozen Local3 catalog: {outcome}")
                if not isinstance(outcome, WorkflowSuccess):
                    if not isinstance(outcome, (NumericalFailure, ResourceFailure)):
                        raise PackageFatal(f"Unknown Local3 outcome: {type(outcome).__name__}")
                    row.update(status=type(outcome).__name__, failure=str(outcome))
                    return outcome
                with timer.measure("checkpoint", parent="evaluator"):
                    restored, payload, storage_bytes = self.checkpoint(outcome.model, source.validation.x)
                with timer.measure("inventory", parent="evaluator"):
                    expected = expected_tensor_bytes(workflow.genotype)
                    if storage_bytes != expected or storage_bytes > TENSOR_BYTE_BUDGET:
                        raise PackageFatal("Local3 independent storage formula disagrees")
                with timer.measure("validation", parent="evaluator"):
                    logits = self.prediction(restored, source.validation.x)
                    quality = metrics(logits, source.validation.y)
                    accepted = (quality["cross_entropy"] <= source.baseline_metrics["cross_entropy"]+.05
                        and quality["accuracy"] >= source.baseline_metrics["accuracy"]-.02)
                row.update(status="finite", metrics=quality, quality=quality["cross_entropy"]/math.log(10),
                    cost=storage_bytes/DENSE_TENSOR_BYTES, tensor_bytes=storage_bytes,
                    expected_tensor_bytes=expected, quality_accepted=accepted,
                    checkpoint_bytes=len(payload), checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
                    logits_sha256=tensor_hash(logits), inventory_verified_after_reload=True)
                if self.capture_tensors:
                    with timer.measure("comparison_capture", parent="evaluator"):
                        tensors = dict(outcome=type(outcome).__name__, status="finite",
                            solution_id=identity, tensor_bytes=storage_bytes, quality_accepted=accepted,
                            metrics=quality, logits=logits.detach().clone(),
                            operators=dense_operator_snapshots(restored, source))
                self.last_model = restored
                return GrammarFitness(row["quality"], row["cost"], accepted)
        except PackageFatal:
            row.update(status="package_fatal")
            raise
        except Exception as error:
            row.update(status="package_fatal", failure=f"{type(error).__name__}: {error}")
            raise PackageFatal(row["failure"]) from error
        finally:
            try:
                with timer.measure("integrity", parent="evaluator"):
                    if model_identity(source.model) != locals().get("before"):
                        raise ValueError("Source model changed during evaluation")
                    source.calibration.verify_integrity()
                    source.validation.verify_integrity()
                    if rng_signature() != before_rng:
                        raise ValueError("Full evaluator changed caller RNG state")
            except Exception as error:
                invariant_failure = f"Local3 source/RNG integrity failed: {type(error).__name__}: {error}"
                row.update(status="package_fatal", failure=invariant_failure)
            row.update(completed_at=time.perf_counter(), phases=timer.rows)
            row["elapsed_seconds"] = row["completed_at"]-started
            row["partition"] = timer.partition(started, row["completed_at"],
                {"source_identity", "transform", "checkpoint", "inventory", "validation", "comparison_capture", "integrity"})
            transform_phase = next((phase for phase in timer.rows if phase["phase"] == "transform"), None)
            if transform_phase:
                row["transform_partition"] = timer.partition(transform_phase["started_at"],
                    transform_phase["completed_at"], {"fixed_profile", "afm_whole_profile", "assembly_apply"})
            row["profile_partitions"] = [dict(scope=phase.get("scope"),
                **timer.partition(phase["started_at"], phase["completed_at"],
                    {"observe", "build_metric", "solve", "build_replacement", "apply_replacements"}))
                for phase in timer.rows if phase["phase"] == "fixed_profile"
                and phase.get("completed_at") is not None]
            publication_started = time.perf_counter()
            self.rows.append(row)
            self.tensor_results.append(tensors if tensors is not None else dict(
                outcome=row.get("outcome"), status=row["status"], solution_id=identity))
            append_event(self.events_path, row)
            self.ledger.complete()
            publication_completed = time.perf_counter()
            row["event_publication_seconds"] = publication_completed-publication_started
            timer.rows.append(dict(phase="event_publication", parent="evaluator",
                started_at=publication_started, completed_at=publication_completed))
            row["body_completed_at"] = row["completed_at"]
            row["completed_at"] = publication_completed
            row["elapsed_seconds"] = publication_completed-started
            row["partition"] = timer.partition(started, publication_completed,
                {"source_identity", "transform", "checkpoint", "inventory", "validation",
                 "comparison_capture", "integrity", "event_publication"})
            append_event(self.events_path, dict(event="evaluation_publication_completed",
                job=self.job_id, global_attempt=attempt, started_at=publication_started,
                completed_at=publication_completed, body_completed_at=row["body_completed_at"]))
            if invariant_failure is not None:
                raise PackageFatal(invariant_failure)


def source_evaluator(job, *, capture_tensors=False):
    timer = PhaseRecorder()
    ledger = Local3Ledger(Path(job["output"])/"mlp-attempts.jsonl", job)
    with timer.measure("source_load_and_validation", parent="worker"):
        source = load_local3_source(job, ledger)
    with timer.measure("space_preparation", parent="worker"):
        space = prepare_space(source, TUNED_CATALOG)
    context = stable_hash(dict(profile="local3-digits-10class-68904-v1",
        source=source.manifest["normalized_identity"], calibration=source.calibration.manifest(),
        validation=source.validation.manifest(), source_sha256=job["source_sha256"],
        protocol_sha256=job["protocol_sha256"], quality_rule="CE+.05,acc-.02",
        rng_policy="save/restore Python NumPy Torch; full StatisticsKey; fresh optimizer cache"))
    evaluator = Local3FullEvaluator(source, space, ledger, Path(job["directory"])/"events.jsonl",
        context_sha256=context, job_id=job["job"], capture_tensors=capture_tensors)
    return source, space, evaluator, timer
