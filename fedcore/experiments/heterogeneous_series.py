"""Explicit Evo-only whole-panel orchestration; importing never starts a run.

``run_series`` is called only by a separately authorized owned controller. It
does not download data, invoke the pilot, open the official test, discover a GPU,
connect to a host, or substitute CPU timings for hardware measurements. Native
execution requires one declared CUDA UUID and a caller-supplied latency callback.
The original controller start and mandatory GPU-seconds budget include every
phase and a cleanup reserve. An external OS whole-tree deadline is still required.

Effects can be replaced by counting CPU fakes. Such a run is labelled injected,
never A100 validation. Complete baseline/candidate trajectories, failures and
missing pairs survive in append-only journals. A write-once panel seal closes
ALL selections before any Vreport callback; reports cannot select another model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Callable

from fedcore.algorithm.heterogeneous.backend import BackendSpec
from fedcore.algorithm.heterogeneous.contracts import fingerprint, payload
from fedcore.algorithm.heterogeneous.cv import (
    BlockRecipe, HeterogeneousGene, HeterogeneousGenotype, compile_genotype,
    default_alternatives, genotype_from_payload, genotype_payload, prepare_space)
from fedcore.algorithm.heterogeneous.protocol import ExperimentProtocol
from fedcore.algorithm.heterogeneous.search import common_initial, run_search
from fedcore.algorithm.heterogeneous.selection import (
    IncompleteSearch, NoFeasibleFinalist, SelectedFinalist, select_finalist)
from fedcore.algorithm.low_rank.compression_graph import RuntimeOptions
from fedcore.algorithm.low_rank.grammar_execution import ResourceFailure
from fedcore.algorithm.low_rank.grammar_search import GrammarFitness, GrammarSearchConfig
from fedcore.algorithm.low_rank.plans import ValidationFailure
from .heterogeneous_runtime import CandidateEvaluator, RecoveryConfig, evaluate_loaded, recovery_tape


class SeriesError(RuntimeError):
    pass


class CandidateEvaluationError(SeriesError):
    def __init__(self, message, accounting):
        self.accounting = accounting
        super().__init__(message)


@dataclass(frozen=True)
class SeriesConfig:
    gpu_seconds: float
    search_seconds: float
    cleanup_seconds: float = 60.
    protocol: ExperimentProtocol = ExperimentProtocol()
    backend_spec: BackendSpec = BackendSpec(runtime="tensorrt")
    device: str = "cuda:0"
    max_peak_bytes: int = 8 * 1024**3
    storage_ratio: float = .5
    latency_protocol_id: str | None = None
    latency_pairs_per_session: int | None = None
    latency_aa_pairs_per_session: int | None = None

    def __post_init__(self):
        if (any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in
                (self.gpu_seconds, self.search_seconds, self.cleanup_seconds))
                or self.gpu_seconds <= self.cleanup_seconds or type(self.protocol) is not ExperimentProtocol
                or type(self.backend_spec) is not BackendSpec or self.device not in ("cuda", "cuda:0")
                or type(self.max_peak_bytes) is not int or not 0 < self.max_peak_bytes <= 12 * 1024**3
                or type(self.storage_ratio) not in (int, float) or not 0 < self.storage_ratio <= 1):
            raise ValueError("Explicit finite whole-panel/search budgets and one CUDA device are required")
        if self.latency_protocol_id is not None and (type(self.latency_protocol_id) is not str or not self.latency_protocol_id.strip()):
            raise ValueError("A supplied hardware protocol identity must be explicit and nonempty")
        if any(value is not None and (type(value) is not int or value < 1) for value in
               (self.latency_pairs_per_session, self.latency_aa_pairs_per_session)):
            raise ValueError("Hardware raw-pair counts must be positive integers when supplied")
        p = self.protocol
        if (len(p.model_seeds), len(p.search_seeds), p.fit_rows, p.calibration_rows, p.search_rows,
                p.report_rows, p.batch_size, p.epochs, p.attempts, p.population, p.recovery_updates,
                p.controls_per_model) != (3, 3, 40000, 2000, 4000, 4000, 128, 80, 64, 8, 200, 5):
            raise ValueError("This orchestrator implements the sealed 3-M0/27-search/15-control panel")


@dataclass(frozen=True)
class BaselineResult:
    model: object
    updates: int
    checkpoint_path: str


@dataclass(frozen=True)
class ArtifactRecord:
    manifest_path: str
    storage_bytes: int


@dataclass(frozen=True)
class CandidateRun:
    outcome: object
    artifact: ArtifactRecord | None
    sgd_attempted: int
    sgd_completed: int
    receipt_path: str | None = None


@dataclass(frozen=True)
class CandidateContext:
    source: object
    calibration: object
    tape: tuple
    data: object
    space: object
    output_root: str
    deadline: float
    config: SeriesConfig
    model_seed: int


def _new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload(value), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def _model_identity(model):
    from fedcore.algorithm.low_rank.cv_execution import _source
    return _source(model)[0]


def _data_identity(data):
    """Recompute actual immutable train corpus + role identities, not a cached label."""
    import numpy as np
    digest = hashlib.sha256()
    digest.update(memoryview(np.ascontiguousarray(data.images)).cast("B"))
    digest.update(np.asarray(data.labels).tobytes()); digest.update(data.roles.sha256.encode())
    return digest.hexdigest()


def _calibration_identity(calibration):
    from fedcore.algorithm.low_rank.cv_execution import _calibration
    return _calibration(calibration)


def train_baseline(data, seed, directory, *, protocol, device, deadline, clock, progress):
    """Real 80-epoch CIFAR ResNet18 SGD, explicit fit loader, drop_last=False.

    No validation-based checkpoint choice and no dataset construction/download.
    Native orchestration binds the physical CUDA UUID before calling this code.
    """
    import numpy as np
    import torch
    from fedcore.algorithm.pruning.sparse.cv_channels import CifarResNet18
    from fedcore.tools.registry.checkpoint_manager import CheckpointManager
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    target = torch.device(device)
    devices = [0] if target.type == "cuda" else []
    python_state, numpy_state = random.getstate(), np.random.get_state()
    updates = 0
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed); random.seed(seed); np.random.seed(seed % 2**32)
            model = CifarResNet18().to(target).train()
            optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=protocol.epochs)
            rng = random.Random(seed)
            for epoch in range(protocol.epochs):
                indices = list(data.roles.fit); rng.shuffle(indices)
                seen = 0
                for start in range(0, len(indices), protocol.batch_size):
                    if clock() >= deadline:
                        raise TimeoutError("Original series deadline expired during baseline")
                    ids = tuple(indices[start:start + protocol.batch_size])
                    x, y = data.training(ids, rng.randrange(2**32))
                    if (x.dtype != torch.float32 or y.dtype != torch.int64
                            or tuple(x.shape) != (len(ids), 3, 32, 32) or tuple(y.shape) != (len(ids),)
                            or not bool(torch.isfinite(x).all())):
                        raise ValueError("Explicit baseline loader must return finite CIFAR fit batches")
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(model(x.to(target)), y.to(target))
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("Nonfinite baseline loss")
                    progress("sgd_attempted", updates + 1)
                    loss.backward(); optimizer.step(); updates += 1
                    progress("sgd_completed", updates)
                    seen += len(ids)
                if seen != protocol.fit_rows:
                    raise ValueError("An epoch did not consume exactly all fit rows")
                scheduler.step()
                progress("epoch_completed", epoch + 1)
            if target.type == "cuda":
                torch.cuda.synchronize(target)
            if clock() >= deadline:
                raise TimeoutError("Baseline finished after original series deadline")
            model.eval().cpu()
            path = directory / "baseline.pt"
            path.write_bytes(CheckpointManager(None).serialize_to_bytes(model))
            _new_json(directory / "training.json", dict(seed=seed, epochs=protocol.epochs, updates=updates,
                batch_size=protocol.batch_size, drop_last=False, last_epoch_only=True, fit_rows=protocol.fit_rows,
                optimizer="SGD", learning_rate=.1, momentum=.9, weight_decay=5e-4, schedule="cosine_80_epochs"))
            return BaselineResult(model, updates, str(path))
    finally:
        random.setstate(python_state); np.random.set_state(numpy_state)


def _prepare(source, calibration, config):
    return prepare_space(source, calibration, alternatives=default_alternatives(), options=RuntimeOptions(
        max_peak_bytes=config.max_peak_bytes, max_workspace_bytes=min(256 * 1024**2, config.max_peak_bytes)))


def _export_baseline(source, calibration, space, destination, config, deadline):
    from fedcore.algorithm.heterogeneous.backend import DeploymentArtifact, export_onnx_qdq
    from fedcore.algorithm.heterogeneous.cv import execute_float_workflow
    from fedcore.algorithm.low_rank.cv_execution import CVWorkflowSuccess
    genotype = HeterogeneousGenotype(tuple(HeterogeneousGene(d.path, BlockRecipe()) for d in space.domains))
    workflow = compile_genotype(genotype, space)
    result = execute_float_workflow(source, calibration, workflow)
    if type(result) is not CVWorkflowSuccess:
        raise SeriesError("Dense S0 construction failed: " + str(result))
    if time.perf_counter() >= deadline:
        raise TimeoutError("No late baseline deployment")
    artifact = export_onnx_qdq(result.model, calibration, workflow, destination, spec=config.backend_spec,
                              recovery_evidence={"updates": 0, "role": "original_dense_S0"})
    if type(artifact) is not DeploymentArtifact:
        raise SeriesError("Dense S0 deployment failed: " + str(artifact))
    return ArtifactRecord(artifact.manifest_path, artifact.storage_bytes)


def _evaluate(context, workflow):
    evaluator = CandidateEvaluator(context.source, context.calibration, context.tape, context.data.training,
        lambda: context.data.search_batches(context.config.protocol.batch_size), context.output_root,
        context.deadline, context.config.backend_spec, context.data.roles.sha256 + ":Vsearch",
        validation_rows=context.config.protocol.search_rows,
        recovery_config=RecoveryConfig(updates=context.config.protocol.recovery_updates), device=context.config.device)
    try:
        outcome = evaluator(workflow)
    except Exception as error:
        raise CandidateEvaluationError("Unexpected candidate exception: " + str(error),
                                       _partial_accounting(context, workflow)) from error
    path = Path(context.output_root) / workflow.fingerprint / "receipt.json"
    if not path.exists():
        if type(outcome) is ResourceFailure and outcome.code == "CandidateDeadline":
            return CandidateRun(outcome, None, 0, 0)
        raise SeriesError("Candidate returned without its durable SGD/accounting receipt")
    state = json.loads(path.read_text(encoding="utf-8"))
    artifact = (ArtifactRecord(state["manifest_path"], state["storage_bytes"])
                if state["status"] == "succeeded" else None)
    return CandidateRun(outcome, artifact, state["sgd_attempted"], state["sgd_completed"], str(path))


def _partial_accounting(context, workflow):
    """Durable lower bounds distinguish completed, in-flight and unknown work."""
    directory = Path(context.output_root) / workflow.fingerprint
    receipt = directory / "receipt.json"
    try:
        if receipt.is_file():
            state = json.loads(receipt.read_text(encoding="utf-8"))
            if state.get("workflow_fingerprint") != workflow.fingerprint:
                return None
            attempted, completed = state["sgd_attempted"], state["sgd_completed"]
            source = str(receipt)
        else:
            progress = directory / "progress.jsonl"
            if not progress.is_file():
                return None
            attempted = completed = 0
            with progress.open("r", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        break  # A torn last append is not a completed step.
                    if event.get("event") == "sgd_attempted":
                        attempted = max(attempted, event["updates"])
                    elif event.get("event") == "sgd_completed":
                        completed = max(completed, event["updates"])
            source = str(progress)
        if (any(type(value) is not int or value < 0 for value in (attempted, completed))
                or completed > attempted or attempted > context.config.protocol.recovery_updates):
            return None
        return dict(sgd_attempted=attempted, sgd_completed=completed,
                    in_flight_unconfirmed=attempted - completed, source=source)
    except (ValueError, KeyError, OSError, TypeError):
        return None


def _artifact_identity(artifact):
    from fedcore.algorithm.heterogeneous.backend import deployment_size, load_deployment
    if type(artifact) is not ArtifactRecord or deployment_size(artifact.manifest_path) != artifact.storage_bytes:
        raise SeriesError("Stale or inconsistent deployment package")
    # Native loader checks actual model hash and exact Q/DQ coverage.
    load_deployment(artifact.manifest_path)
    return _sha(artifact.manifest_path)


def _report(data, artifact, *, config, deadline, seal_path):
    """The report role becomes readable only after a durable whole-panel seal."""
    import torch
    from fedcore.algorithm.heterogeneous.backend import load_deployment
    if not Path(seal_path).is_file():
        raise SeriesError("Vreport requires a durable whole-panel seal")
    def batches():
        for start in range(0, len(data.roles.report), config.protocol.batch_size):
            ids = list(data.roles.report[start:start + config.protocol.batch_size])
            yield data._tensor(data.images[ids]), torch.from_numpy(data.labels[ids].copy())
    return evaluate_loaded(load_deployment(artifact.manifest_path), batches(), deadline=deadline,
                           expected_rows=config.protocol.report_rows)


@dataclass(frozen=True)
class SeriesEffects:
    """Replace only at explicit boundaries; injected execution is labelled as such."""
    train_baseline: Callable = train_baseline
    prepare_space: Callable = _prepare
    export_baseline: Callable = _export_baseline
    evaluate_candidate: Callable = _evaluate
    search: Callable = run_search
    model_identity: Callable = _model_identity
    data_identity: Callable = _data_identity
    calibration_identity: Callable = _calibration_identity
    artifact_identity: Callable = _artifact_identity
    report: Callable = _report
    latency: Callable | None = None
    latency_boundary: Callable | None = None


def uniform_controls(space):
    """Five fixed controls: 3/4 internal width and the larger declared SVD rank."""
    result = {}
    for name in ("dense_recovery", "uniform_P", "uniform_SVD", "Q_only", "uniform_P_SVD_Q"):
        genes = []
        for domain in space.domains:
            options = domain.alternatives
            if name == "dense_recovery":
                choice = BlockRecipe()
            elif name == "Q_only":
                choice = BlockRecipe(ptq=True)
            elif name == "uniform_P":
                candidates = [c for c in options if c.keep_channels is not None and c.rank is None and not c.ptq]
                choice = max(candidates, key=lambda c: c.keep_channels)
            elif name == "uniform_SVD":
                candidates = [c for c in options if c.keep_channels is None and c.rank is not None and not c.ptq]
                choice = max(candidates, key=lambda c: c.rank)
            else:
                candidates = [c for c in options if c.keep_channels is not None and c.rank is not None and c.ptq]
                choice = max(candidates, key=lambda c: (c.keep_channels, c.rank))
            if choice not in options:
                raise SeriesError("Fixed uniform control is missing from the sealed domain")
            genes.append(HeterogeneousGene(domain.path, choice))
        result[name] = HeterogeneousGenotype(tuple(genes))
    return result


def _roles(data, protocol):
    groups = (data.roles.fit, data.roles.calibration, data.roles.search, data.roles.report)
    sizes = (protocol.fit_rows, protocol.calibration_rows, protocol.search_rows, protocol.report_rows)
    if (tuple(map(len, groups)) != sizes or any(type(g) is not tuple or tuple(sorted(set(g))) != g for g in groups)
            or set(i for g in groups for i in g) != set(range(50000)) or sum(map(len, groups)) != 50000):
        raise SeriesError("Data roles do not match the frozen 40000/2000/4000/4000 train-only protocol")


class _Journal:
    def __init__(self, root, clock):
        self.root, self.clock = root, clock
        self.baseline_attempted = self.baseline_completed = self.recovery_attempted = self.recovery_completed = 0
        self.charged = 0
        self.unknown_sgd = []
    def event(self, event, **details):
        with (self.root / "series.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload(dict(event=event, at=self.clock(), **details)), allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
    def counters(self):
        return dict(baseline_sgd_attempted=self.baseline_attempted, baseline_sgd_completed=self.baseline_completed,
                    recovery_sgd_attempted=self.recovery_attempted, recovery_sgd_completed=self.recovery_completed,
                    charged_search_attempts=self.charged)


def run_series(data, output_root, *, config, started_at, effects=None, hardware_latency=None,
               hardware_latency_boundary=None, clock=time.perf_counter):
    """Execute a NEW panel under the original controller budget; never a pilot.

    ``hardware_latency(artifact, *, batch_size, window, deadline, device_uuid,
    baseline, seal_path)`` measures the 45 final roles. The separate mandatory
    ``hardware_latency_boundary(S0, *, phase, batch_size, window, deadline,
    device_uuid, seal_path)`` checks pre/post A/A for each M0/batch/window.
    Admission occurs only after post A/A. Warmups, repeats and thresholds belong
    to the separately sealed ``latency_protocol_id``; this module invents none.
    No hardware callback means refusal before training. CPU fake callbacks are
    accepted only through explicit ``SeriesEffects`` and labelled injected.
    """
    if (type(config) is not SeriesConfig or type(started_at) not in (int, float) or not math.isfinite(started_at)
            or started_at < 0 or started_at > clock() or effects is not None and type(effects) is not SeriesEffects):
        raise ValueError("Explicit sealed series config, original controller start and typed effects required")
    deadline = started_at + config.gpu_seconds - config.cleanup_seconds
    if clock() >= deadline:
        raise TimeoutError("Original whole-series deadline exhausted before device binding")
    native = effects is None
    effects = effects or SeriesEffects(latency=hardware_latency, latency_boundary=hardware_latency_boundary)
    latency = effects.latency or hardware_latency
    boundary = effects.latency_boundary or hardware_latency_boundary
    if not callable(latency) or not callable(boundary):
        raise ValueError("Explicit hardware latency and pre/post boundary callbacks are mandatory; no CPU timing fallback")
    if native:
        from .heterogeneous_device import require_cuda_binding
        if config.backend_spec.runtime != "tensorrt":
            raise ValueError("The native A100 panel requires the declared TensorRT backend")
        if config.latency_protocol_id is None:
            raise ValueError("Freeze the post-pilot hardware protocol identity before native execution")
        if config.latency_pairs_per_session is None or config.latency_aa_pairs_per_session is None:
            raise ValueError("Freeze positive hardware pair and A/A pair counts before native execution")
        require_cuda_binding(config.backend_spec.device_uuid)  # Binds UUID before first CUDA initialization.
    _roles(data, config.protocol)
    root = Path(output_root).resolve(); root.mkdir(parents=True, exist_ok=False)
    journal = _Journal(root, clock)
    expected = config.protocol.budget()
    models, controls, searches, reports, timings, failures = {}, {}, {}, {}, {}, []
    calibrations = {}
    data_identity = effects.data_identity(data)
    status, seal_path, seal_hash = "failed", root / "panel.seal.json", None
    _new_json(root / "plan.json", dict(config=config, started_at=started_at, deadline=deadline,
        execution_mode="native" if native else "injected_effects_not_hardware_validation", data_identity=data_identity,
        expected_budget=expected, official_test_opened=False))

    def check(source=None, identity=None, *, verify_data=True):
        if clock() >= deadline:
            raise TimeoutError("Original whole-series deadline exhausted; no automatic retry")
        if verify_data and effects.data_identity(data) != data_identity:
            raise SeriesError("Stale dataset or role identity")
        if verify_data and any(effects.calibration_identity(tensor) != snapshot for tensor, snapshot in calibrations.values()):
            raise SeriesError("Stale calibration tensor identity")
        if source is not None and effects.model_identity(source) != identity:
            raise SeriesError("Frozen M0 changed during the panel")
        if (journal.baseline_attempted + journal.recovery_attempted > expected["total_sgd"]
                or journal.charged > expected["charged_attempts"]):
            raise SeriesError("Whole-panel SGD or charged-attempt budget exceeded")

    def candidate(context, workflow, records, key):
        check(context.source, models[context.model_seed]["identity"])
        if workflow.fingerprint in records:
            raise SeriesError("Cross-call candidate cache/recovery reuse is forbidden")
        journal.event("candidate_started", job=key, workflow=workflow.fingerprint)
        try:
            result = effects.evaluate_candidate(context, workflow)
        except Exception as error:
            accounting = (error.accounting if type(error) is CandidateEvaluationError else
                          _partial_accounting(context, workflow))
            if accounting is None:
                journal.unknown_sgd.append(dict(job=key, workflow=workflow.fingerprint,
                    reason="Unexpected exception without trustworthy durable step accounting"))
            else:
                journal.recovery_attempted += accounting["sgd_attempted"]
                journal.recovery_completed += accounting["sgd_completed"]
            journal.event("candidate_exception", job=key, workflow=workflow.fingerprint,
                          accounting=accounting, error=str(error), **journal.counters())
            raise SeriesError("Candidate failed unexpectedly; partial/unknown SGD recorded and panel stopped: " + str(error)) from error
        if (type(result) is not CandidateRun or any(type(v) is not int or v < 0 for v in
                (result.sgd_attempted, result.sgd_completed)) or result.sgd_completed > result.sgd_attempted
                or result.sgd_attempted > config.protocol.recovery_updates):
            accounting = _partial_accounting(context, workflow)
            if accounting is None:
                journal.unknown_sgd.append(dict(job=key, workflow=workflow.fingerprint,
                    reason="Malformed CandidateRun/counters without trustworthy durable accounting"))
            else:
                journal.recovery_attempted += accounting["sgd_attempted"]
                journal.recovery_completed += accounting["sgd_completed"]
            journal.event("candidate_malformed", job=key, workflow=workflow.fingerprint,
                          accounting=accounting, **journal.counters())
            raise SeriesError("Missing, invalid or excessive per-candidate SGD accounting")
        journal.recovery_attempted += result.sgd_attempted
        journal.recovery_completed += result.sgd_completed
        if type(result.outcome) is GrammarFitness:
            if (result.sgd_completed != config.protocol.recovery_updates or result.artifact is None
                    or result.outcome.cost != result.artifact.storage_bytes
                    or any(type(v) not in (float, int) or not math.isfinite(v) or v < 0
                           for v in (result.outcome.quality, result.outcome.cost))
                    or type(result.outcome.quality_accepted) is not bool):
                raise SeriesError("Finite fitness requires full fresh recovery and matching deployed artifact")
            artifact_hash = effects.artifact_identity(result.artifact)
        else:
            artifact_hash = None
        records[workflow.fingerprint] = dict(result=result, artifact_hash=artifact_hash)
        journal.event("candidate_finished", job=key, workflow=workflow.fingerprint,
                      outcome=type(result.outcome).__name__, **journal.counters())
        check(context.source, models[context.model_seed]["identity"])
        return result.outcome

    try:
        for model_seed in config.protocol.model_seeds:
            check()
            model_root = root / ("m0-" + str(model_seed)); model_root.mkdir()
            journal.event("baseline_started", model_seed=model_seed)
            local = {"sgd_attempted": 0, "sgd_completed": 0}
            baseline_updates = config.protocol.epochs * math.ceil(config.protocol.fit_rows / config.protocol.batch_size)
            def progress(event, count):
                if event in local:
                    if (type(count) is not int or count < local[event] or count > baseline_updates
                            or event == "sgd_completed" and count > local["sgd_attempted"]):
                        raise SeriesError("Baseline SGD counters regressed")
                    delta = count - local[event]; local[event] = count
                    attr = "baseline_attempted" if event == "sgd_attempted" else "baseline_completed"
                    setattr(journal, attr, getattr(journal, attr) + delta)
                journal.event("baseline_progress", model_seed=model_seed, phase_event=event, count=count,
                              **journal.counters()); check(verify_data=event == "epoch_completed")
            baseline = effects.train_baseline(data, model_seed, model_root / "training", protocol=config.protocol,
                device=config.device, deadline=deadline, clock=clock, progress=progress)
            if (type(baseline) is not BaselineResult or baseline.updates != baseline_updates
                    or tuple(local.values()) != (baseline_updates, baseline_updates)
                    or not Path(baseline.checkpoint_path).is_file()):
                raise SeriesError("Baseline did not complete its exact fresh 80-epoch trajectory")
            identity = effects.model_identity(baseline.model)
            if identity in {row["identity"] for row in models.values()}:
                raise SeriesError("Independent baseline seeds produced reused source identity")
            calibration = data.calibration_tensor()
            calibrations[model_seed] = (calibration, effects.calibration_identity(calibration))
            space = effects.prepare_space(baseline.model, calibration, config)
            if type(space) is ValidationFailure:
                raise SeriesError("Unsupported G1 space: " + str(space))
            tape = recovery_tape(data.roles.fit, model_seed=model_seed,
                config=RecoveryConfig(updates=config.protocol.recovery_updates))
            s0 = effects.export_baseline(baseline.model, calibration, space, model_root / "S0", config, deadline)
            s0_hash = effects.artifact_identity(s0)
            models[model_seed] = dict(identity=identity, checkpoint_path=baseline.checkpoint_path,
                checkpoint_sha256=_sha(baseline.checkpoint_path), baseline_updates=baseline_updates,
                S0=s0, S0_hash=s0_hash, tape_fingerprint=fingerprint(tape),
                calibration_fingerprint=calibrations[model_seed][1])
            check(baseline.model, identity)
            journal.event("baseline_completed", model_seed=model_seed, identity=identity, **journal.counters())
            # Controls live in separate directories and always repeat recovery;
            # no search evaluation is borrowed from a matching control.
            for name, genotype in uniform_controls(space).items():
                check(baseline.model, identity)
                key = str(model_seed) + ":control:" + name
                context = CandidateContext(baseline.model, calibration, tape, data, space,
                    str(model_root / "controls" / name), deadline, config, model_seed)
                records = {}
                outcome = candidate(context, compile_genotype(genotype, space), records, key)
                if type(outcome) is not GrammarFitness:
                    raise SeriesError("Required control failed: " + key)
                item = next(iter(records.values()))
                controls[key] = dict(status="complete", artifact=item["result"].artifact,
                    artifact_hash=item["artifact_hash"], genotype=genotype_payload(genotype))
            for _, search_seed, arm in (job for job in config.protocol.jobs() if job[0] == model_seed):
                check(baseline.model, identity)
                key = str(model_seed) + ":" + str(search_seed) + ":" + arm.value
                prefix = common_initial(space, config.protocol.population, search_seed)
                prefix_hash = fingerprint(prefix)
                job_root = model_root / "searches" / (str(search_seed) + "-" + arm.value)
                job_root.mkdir(parents=True, exist_ok=False)
                records, fatal = {}, []
                job_deadline = min(deadline, clock() + config.search_seconds)
                context = CandidateContext(baseline.model, calibration, tape, data, space,
                    str(job_root / "candidates"), job_deadline, config, model_seed)
                def assess(workflow):
                    try:
                        if fatal:
                            raise SeriesError("Candidate boundary already failed; further SGD is forbidden")
                        return candidate(context, workflow, records, key)
                    except SeriesError as error:
                        fatal.append(str(error)); raise
                def trace(event):
                    if event.get("event") == "evaluation_started":
                        journal.charged += 1
                        check(baseline.model, identity)
                    journal.event("search_trace", job=key, trace=event)
                search_config = GrammarSearchConfig(seed=search_seed, population_size=config.protocol.population,
                    generations=config.protocol.attempts, max_evaluations=config.protocol.attempts,
                    max_proposals=config.protocol.attempts * config.protocol.proposal_multiplier,
                    max_seconds=config.search_seconds)
                before_charged = journal.charged
                def context_guard():
                    if fatal:
                        from fedcore.algorithm.low_rank.grammar_execution import StructuralInvalid
                        from fedcore.algorithm.low_rank.plans import Violation
                        return StructuralInvalid((Violation("CandidateBoundaryFailed", "series", fatal[0]),))
                    if effects.calibration_identity(calibration) != calibrations[model_seed][1]:
                        from fedcore.algorithm.low_rank.grammar_execution import StructuralInvalid
                        from fedcore.algorithm.low_rank.plans import Violation
                        return StructuralInvalid((Violation("StaleCalibration", "series", "Calibration tensor changed"),))
                    return _source_guard(baseline.model, identity, effects)
                result = effects.search(space, search_config, assess, arm=arm, fitness_context=data_identity + ":Vsearch",
                    initial_genotypes=prefix, context_guard=context_guard,
                    absolute_deadline=job_deadline, trace_callback=trace)
                check(baseline.model, identity)
                if journal.charged - before_charged != result["charged_evaluations"]:
                    raise SeriesError("Search receipt and paid trace counters disagree")
                if fatal:
                    raise SeriesError("Invalid cache/accounting/source boundary: " + "; ".join(fatal))
                selection = select_finalist(result, baseline_bytes=s0.storage_bytes,
                    attempt_limit=config.protocol.attempts, storage_ratio=config.storage_ratio)
                row = dict(status="complete" if type(selection) is not IncompleteSearch else "incomplete",
                    selection_type=type(selection).__name__, selection=asdict(selection), prefix_fingerprint=prefix_hash,
                    tape_fingerprint=fingerprint(tape), charged=result["charged_evaluations"])
                if type(selection) is SelectedFinalist:
                    genotype = genotype_from_payload(selection.genotype, space)
                    workflow = compile_genotype(genotype, space)
                    matching = [point for point in result["finite_archive"]
                        if point.get("canonical_attempt_key", point.get("workflow_fingerprint")) == selection.identity]
                    if len(matching) != 1 or matching[0].get("workflow_fingerprint") != workflow.fingerprint:
                        raise SeriesError("Canonical selection and executable workflow identity disagree")
                    item = records.get(workflow.fingerprint)
                    if item is None or item["result"].artifact is None:
                        raise SeriesError("Selected finalist artifact/receipt is missing; do not rebuild or rerank")
                    row.update(artifact=item["result"].artifact, artifact_hash=item["artifact_hash"])
                searches[key] = row
                _new_json(job_root / "search.json", result)
                check(baseline.model, identity)
        check()  # Includes all separate calibration snapshots before the seal.
        # Every panel position is closed BEFORE reading report images or timing.
        selected = {key: row for key, row in searches.items() if row["selection_type"] == "SelectedFinalist"}
        originals = {str(seed) + ":original_S0": dict(status="complete", artifact=row["S0"], artifact_hash=row["S0_hash"])
                     for seed, row in models.items()}
        final_roles = {**originals, **controls, **selected}
        seal = dict(schema="heterogeneous-panel-seal-v1", config=config, data_identity=data_identity,
            models=models, controls=controls, searches=searches, counters=journal.counters(), official_test_opened=False)
        seal["final_artifacts"] = final_roles
        if len(controls) != 15 or len(searches) != 27:
            raise SeriesError("Panel closure is incomplete")
        _new_json(seal_path, seal); seal_hash = _sha(seal_path)
        journal.event("panel_sealed", seal_sha256=seal_hash)
        def verify_sealed(*rows):
            if _sha(seal_path) != seal_hash or any(
                    effects.artifact_identity(row["artifact"]) != row["artifact_hash"] for row in rows):
                raise SeriesError("Panel seal or actual deployment changed during a callback")
        for key, row in final_roles.items():
            check()
            if _sha(seal_path) != seal_hash or effects.artifact_identity(row["artifact"]) != row["artifact_hash"]:
                raise SeriesError("Panel seal or selected deployment changed before Vreport")
            quality = effects.report(data, row["artifact"], config=config, deadline=deadline, seal_path=str(seal_path))
            verify_sealed(row)
            check()
            if (type(quality) is not dict or quality.get("rows") != config.protocol.report_rows
                    or any(type(quality.get(k)) not in (int, float) or not math.isfinite(quality[k])
                           for k in ("cross_entropy", "accuracy"))):
                raise SeriesError("Incomplete or nonfinite sealed Vreport result")
            reports[key] = quality
            _new_json(root / ("report-" + key.replace(":", "-") + ".json"), quality)
        # A/A surrounds each M0/batch/window. Measurements remain inadmissible
        # until the matching post boundary succeeds; failed windows never retry.
        for window in (0, 1):
            for model_seed, model_row in models.items():
                for batch_size in (1, 128):
                    group = str(model_seed) + ":" + str(window) + ":" + str(batch_size)
                    def aa(phase):
                        check()
                        if (_sha(seal_path) != seal_hash or effects.artifact_identity(model_row["S0"]) != model_row["S0_hash"]):
                            raise SeriesError("Sealed baseline changed before A/A timing")
                        evidence = boundary(model_row["S0"], phase=phase, batch_size=batch_size,
                            window=window, deadline=deadline, device_uuid=config.backend_spec.device_uuid, seal_path=str(seal_path))
                        verify_sealed(dict(artifact=model_row["S0"], artifact_hash=model_row["S0_hash"]))
                        _validate_boundary(evidence, native, config, batch_size, window, phase)
                        timings["aa:" + group + ":" + phase] = evidence
                        journal.event("latency_boundary", group=group, phase=phase, evidence=evidence)
                        check()
                        if not evidence["aa_passed"]:
                            journal.event("latency_window_inadmissible", group=group, phase=phase)
                            raise SeriesError("Hardware A/A failed; window inadmissible, no retry")
                    aa("pre")
                    group_keys = []
                    for key, row in final_roles.items():
                        if int(key.split(":", 1)[0]) != model_seed:
                            continue
                        check()
                        if (_sha(seal_path) != seal_hash or effects.artifact_identity(row["artifact"]) != row["artifact_hash"]):
                            raise SeriesError("Sealed deployment changed before hardware timing")
                        measurement = latency(row["artifact"], batch_size=batch_size, window=window, deadline=deadline,
                            device_uuid=config.backend_spec.device_uuid, baseline=model_row["S0"], seal_path=str(seal_path))
                        verify_sealed(row, dict(artifact=model_row["S0"], artifact_hash=model_row["S0_hash"]))
                        _validate_timing(measurement, native, config, batch_size, window)
                        check()
                        timing_key = key + ":" + str(window) + ":" + str(batch_size)
                        timings[timing_key] = dict(admitted=False, group=group, evidence=measurement)
                        group_keys.append(timing_key)
                        journal.event("latency_measured_pending_post_aa", key=timing_key, evidence=measurement)
                    aa("post")
                    for timing_key in group_keys:
                        timings[timing_key]["admitted"] = True
                    journal.event("latency_window_admitted", group=group, measurements=len(group_keys))
        verify_sealed(*final_roles.values())
        check()
        status = "complete" if all(row["status"] == "complete" for row in searches.values()) else "incomplete_panel"
    except (Exception,) as error:
        failures.append(dict(type=type(error).__name__, message=str(error)))
        journal.event("series_failed", failure=failures[-1], **journal.counters())
    finally:
        for model_seed in config.protocol.model_seeds:
            for name in ("dense_recovery", "uniform_P", "uniform_SVD", "Q_only", "uniform_P_SVD_Q"):
                controls.setdefault(str(model_seed) + ":control:" + name, dict(status="not_completed"))
        for model_seed, seed, arm in config.protocol.jobs():
            key = str(model_seed) + ":" + str(seed) + ":" + arm.value
            searches.setdefault(key, dict(status="not_completed", selection_type="IncompleteSearch", charged=None))
        receipt = dict(schema="heterogeneous-series-receipt-v1", status=status,
            execution_mode="native" if native else "injected_effects_not_hardware_validation",
            started_at=started_at, ended_at=clock(), deadline=deadline, gpu_seconds=config.gpu_seconds,
            cleanup_seconds=config.cleanup_seconds, expected_budget=expected, counters=journal.counters(),
            models=models, controls=controls, searches=searches, reports=reports, latency=timings, failures=failures,
            unknown_sgd_trajectories=journal.unknown_sgd,
            panel_seal=str(seal_path) if seal_path.exists() else None, panel_seal_sha256=seal_hash,
            official_test_opened=False, automatic_retry=False)
        _new_json(root / "series.receipt.json", receipt)
    return receipt


def _validate_timing(measurement, native, config, batch_size, window):
    if type(measurement) is not dict or not measurement:
        raise SeriesError("Hardware latency callback must return explicit inspectable evidence")
    if native and (measurement.get("hardware_timing") is not True or measurement.get("backend") != "tensorrt"
            or measurement.get("device_uuid") != config.backend_spec.device_uuid
            or measurement.get("batch_size") != batch_size or measurement.get("window") != window
            or measurement.get("protocol_id") != config.latency_protocol_id
            or measurement.get("timing_source") not in ("cuda_events", "synchronized_cuda_wall")):
        raise SeriesError("Missing explicit synchronized GPU/A-A timing evidence; CPU latency is forbidden")
    if native:
        _validate_raw_pairs(measurement, config.latency_pairs_per_session)


def _validate_boundary(evidence, native, config, batch_size, window, phase):
    if type(evidence) is not dict or type(evidence.get("aa_passed")) is not bool:
        raise SeriesError("Explicit pre/post A/A outcome is required")
    if native and (evidence.get("hardware_timing") is not True or evidence.get("backend") != "tensorrt"
            or evidence.get("device_uuid") != config.backend_spec.device_uuid
            or evidence.get("batch_size") != batch_size or evidence.get("window") != window
            or evidence.get("phase") != phase or evidence.get("protocol_id") != config.latency_protocol_id):
        raise SeriesError("Missing exact hardware/UUID/phase/window/batch and raw A/A boundary evidence")
    if native:
        _validate_raw_pairs(evidence, config.latency_aa_pairs_per_session)


def _validate_raw_pairs(evidence, expected):
    pairs = evidence.get("raw_pairs")
    if (type(expected) is not int or expected < 1 or evidence.get("units") != "ms"
            or type(pairs) is not list or len(pairs) != expected
            or any(type(pair) is not dict or set(pair) != {"baseline_ms", "candidate_ms"}
                   or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                          for value in pair.values()) for pair in pairs)):
        raise SeriesError("Exact nonempty positive finite raw timing pairs in ms are required")


def _source_guard(model, identity, effects):
    from fedcore.algorithm.low_rank.grammar_execution import StructuralInvalid
    from fedcore.algorithm.low_rank.plans import Violation
    return (None if effects.model_identity(model) == identity else
            StructuralInvalid((Violation("StaleSource", "series", "Frozen baseline changed"),)))
