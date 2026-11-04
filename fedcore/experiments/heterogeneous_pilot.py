"""One explicitly authorized A100 cost pilot; imports perform no numerical work.

The pilot has no search, finalist selection, Vreport or official-test path. Its
original controller window covers admission, one fresh M0 and fixed recipes.
An external guard owns launch and the OS deadline. Native execution checks that
boundary live after the guard's start gate; a JSON claim alone is not evidence.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Callable


FIXED_RECIPES = (
    "dense_recovery", "uniform_P", "uniform_SVD", "Q_only",
    "uniform_P_SVD", "uniform_P_Q", "uniform_SVD_Q", "uniform_P_SVD_Q",
    "uniform_P_half", "uniform_SVD_low", "uniform_P_SVD_Q_half", "mixed_eight_families",
)


@dataclass(frozen=True)
class PilotProtocol:
    fit_rows: int = field(default=40000, init=False)
    calibration_rows: int = field(default=2000, init=False)
    search_rows: int = field(default=4000, init=False)
    report_rows: int = field(default=4000, init=False)
    batch_size: int = field(default=128, init=False)
    epochs: int = field(default=80, init=False)
    recovery_updates: int = field(default=200, init=False)


@dataclass(frozen=True)
class PilotConfig:
    gpu_uuid: str
    source_revision: str
    gpu_seconds: float = 7200.
    cleanup_seconds: float = 60.
    model_seed: int = 20261124
    split_seed: int = 20261102
    recipes: tuple[str, ...] = FIXED_RECIPES
    max_peak_bytes: int = 8 * 1024**3
    device: str = field(default="cuda:0", init=False)
    protocol: PilotProtocol = field(default=PilotProtocol(), init=False)

    def __post_init__(self):
        if (type(self.gpu_uuid) is not str or re.fullmatch(
                r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", self.gpu_uuid) is None
                or type(self.source_revision) is not str or re.fullmatch(r"[a-f0-9]{40}", self.source_revision) is None
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                       for v in (self.gpu_seconds, self.cleanup_seconds))
                or not self.cleanup_seconds < self.gpu_seconds <= 7200
                or any(type(v) is not int or not 0 <= v < 2**32 for v in (self.model_seed, self.split_seed))
                or type(self.recipes) is not tuple or not 1 <= len(self.recipes) <= 12
                or any(type(name) is not str or name not in FIXED_RECIPES for name in self.recipes)
                or self.recipes != tuple(name for name in FIXED_RECIPES if name in self.recipes)
                or type(self.max_peak_bytes) is not int or not 0 < self.max_peak_bytes <= 12 * 1024**3):
            raise ValueError("A fixed ordered recipe subset, one GPU, revision and <=7200s window are required")

    @property
    def backend_spec(self):
        from fedcore.algorithm.heterogeneous.backend import BackendSpec
        return BackendSpec(runtime="tensorrt", device_uuid=self.gpu_uuid)

    def budget(self):
        return dict(models=1, recipes=len(self.recipes), baseline_sgd=25040,
                    recovery_sgd=200 * len(self.recipes), total_sgd=25040 + 200 * len(self.recipes),
                    max_seconds=self.gpu_seconds, searches=0, automatic_retries=0)


class PilotError(RuntimeError):
    pass


def pilot_config_from_payload(value):
    """Decode exact JSON fields; budgets cannot leak in from a series config."""
    if type(value) is not dict:
        raise ValueError("Pilot config must be a JSON object")
    allowed = {name for name, item in PilotConfig.__dataclass_fields__.items() if item.init}
    extras = set(value) - allowed - {"device", "protocol"}
    if (extras or "device" in value and value["device"] != "cuda:0"
            or "protocol" in value and value["protocol"] != asdict(PilotProtocol())):
        raise ValueError("Unknown pilot fields or modified fixed training protocol")
    fields = {name: item for name, item in value.items() if name in allowed}
    if "recipes" in fields:
        if type(fields["recipes"]) is not list:
            raise ValueError("JSON recipes must be an explicit array")
        fields["recipes"] = tuple(fields["recipes"])
    return PilotConfig(**fields)


def wait_for_start_gate(path, *, timeout=10., clock=time.monotonic, sleep=time.sleep):
    """Bounded file barrier only; live OS validation follows its release."""
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 10:
        raise ValueError("The external guard gate wait must be positive and <=10 seconds")
    gate, end = Path(path), clock() + timeout
    while True:
        if gate.is_symlink():
            raise PilotError("Start gate must be an owned regular file")
        if gate.is_file():
            if clock() >= end:
                raise TimeoutError("Start gate arrived after its bounded wait")
            return
        left = end - clock()
        if left <= 0:
            raise TimeoutError("External guard did not release the start gate; no experiment started")
        sleep(min(.05, left))


def _json_value(value):
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    if type(value) is dict:
        return {key: _json_value(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [_json_value(item) for item in value]
    return value


def _write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(_json_value(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_genotypes(space, names=FIXED_RECIPES):
    """Deterministic domain-relative plans; no RNG, proposals or search calls.

    P uses the larger declared width; SVD uses the larger declared rank.
    ``half``/``low`` select the smaller declared width/rank. The mixed recipe
    assigns the eight G1 families once each in canonical block order.
    """
    from fedcore.algorithm.heterogeneous.cv import BlockRecipe, HeterogeneousGene, HeterogeneousGenotype
    if (type(names) is not tuple or not names or names != tuple(n for n in FIXED_RECIPES if n in names)):
        raise ValueError("Fixed distinct recipes in canonical order are required")
    result = {}
    for name in names:
        genes = []
        for index, domain in enumerate(sorted(space.domains, key=lambda d: d.path)):
            family = name
            if name == "mixed_eight_families":
                if len(space.domains) != 8:
                    raise PilotError("The mixed recipe requires all eight CIFAR ResNet18 blocks")
                family = FIXED_RECIPES[index]
            if family in ("dense_recovery", "Q_only"):
                choice = BlockRecipe(ptq=family == "Q_only")
            else:
                pruning = family in ("uniform_P", "uniform_P_half", "uniform_P_Q", "uniform_P_SVD",
                                   "uniform_P_SVD_Q", "uniform_P_SVD_Q_half")
                svd = family in ("uniform_SVD", "uniform_SVD_low", "uniform_SVD_Q", "uniform_P_SVD",
                               "uniform_P_SVD_Q", "uniform_P_SVD_Q_half")
                ptq = family in ("uniform_P_Q", "uniform_SVD_Q", "uniform_P_SVD_Q", "uniform_P_SVD_Q_half")
                choices = [r for r in domain.alternatives if (r.keep_channels is not None) == pruning
                           and (r.rank is not None) == svd and r.ptq == ptq]
                if not choices:
                    raise PilotError("Fixed recipe is absent from domain: " + name + ":" + domain.path)
                key = lambda r: (r.keep_channels or 0, r.rank or 0)
                choice = (min if family.endswith("half") or family.endswith("low") else max)(choices, key=key)
            if choice not in domain.alternatives:
                raise PilotError("Fixed recipe is outside the prepared G1 domain")
            genes.append(HeterogeneousGene(domain.path, choice))
        result[name] = HeterogeneousGenotype(tuple(genes))
    return result


def _native_identity(name, value):
    from . import heterogeneous_series as series
    return getattr(series, "_" + name + "_identity")(value)


def _native_train(*args, **kwargs):
    from .heterogeneous_series import train_baseline
    original = kwargs["progress"]
    def progress(event, count):
        if event == "sgd_completed":
            import torch
            torch.cuda.synchronize(0)
        original(event, count)
    return train_baseline(*args, **{**kwargs, "progress": progress})


def _native_prepare(*args):
    from .heterogeneous_series import _prepare
    return _prepare(*args)


def _native_export(*args):
    from .heterogeneous_series import _export_baseline
    return _export_baseline(*args)


def _native_evaluate(context, workflow):
    from .heterogeneous_runtime import CandidateEvaluator, RecoveryConfig
    from .heterogeneous_series import ArtifactRecord, CandidateEvaluationError, CandidateRun, _partial_accounting
    evaluator = CandidateEvaluator(context.source, context.calibration, context.tape, context.data.training,
        lambda: context.data.search_batches(128), context.output_root, context.deadline, context.config.backend_spec,
        context.data.roles.sha256 + ":Vsearch", validation_rows=4000,
        recovery_config=RecoveryConfig(), device=context.config.device, synchronize_updates=True)
    try:
        outcome = evaluator(workflow)
    except Exception as error:
        raise CandidateEvaluationError(str(error), _partial_accounting(context, workflow)) from error
    path = Path(context.output_root) / workflow.fingerprint / "receipt.json"
    if not path.is_file():
        raise CandidateEvaluationError("Candidate has no durable receipt", _partial_accounting(context, workflow))
    state = json.loads(path.read_text(encoding="utf-8"))
    artifact = (ArtifactRecord(state["manifest_path"], state["storage_bytes"])
                if state["status"] == "succeeded" else None)
    return CandidateRun(outcome, artifact, state["sgd_attempted"], state["sgd_completed"], str(path))


def _native_calibration(data, calibration):
    import torch
    if (not isinstance(calibration, torch.Tensor) or tuple(calibration.shape) != (2000, 3, 32, 32)
            or calibration.dtype != torch.float32 or calibration.device.type != "cpu"
            or not bool(torch.isfinite(calibration).all())):
        raise PilotError("All 2000 finite calibration rows must be supplied")
    expected = data.calibration_tensor()
    if not torch.equal(calibration, expected):
        raise PilotError("Calibration does not match the sealed training-corpus role")
    return _native_identity("calibration", calibration)


def _native_admission(directory, config, deadline, clock):
    """Exercise the complete P→SVD→PTQ path on synthetic data with zero SGD."""
    if clock() >= deadline:
        raise TimeoutError("No synthetic admission may start after the original deadline")
    import torch
    from fedcore.algorithm.pruning.sparse.cv_channels import CifarResNet18
    from fedcore.algorithm.heterogeneous.backend import DeploymentArtifact, export_onnx_qdq
    from fedcore.algorithm.heterogeneous.cv import compile_genotype, execute_float_workflow
    from fedcore.algorithm.low_rank.cv_execution import CVWorkflowSuccess
    from .heterogeneous_device import require_cuda_binding
    binding = require_cuda_binding(config.gpu_uuid)
    name = torch.cuda.get_device_name(0)
    if "A100" not in name or torch.cuda.get_device_capability(0) != (8, 0):
        raise PilotError("This pilot admission requires the declared A100")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.model_seed)
        source = CifarResNet18().eval()
        calibration = torch.randn(2000, 3, 32, 32)
    space = _native_prepare(source, calibration, config)
    rows = {}
    for recipe, genotype in fixed_genotypes(space, ("dense_recovery", "uniform_P_SVD_Q")).items():
        if clock() >= deadline:
            raise TimeoutError("Admission exceeded original pilot deadline")
        workflow = compile_genotype(genotype, space)
        built = execute_float_workflow(source, calibration, workflow)
        if type(built) is not CVWorkflowSuccess:
            raise PilotError("Synthetic admission construction failed: " + str(built))
        artifact = export_onnx_qdq(built.model, calibration, workflow, directory / recipe,
            spec=config.backend_spec, recovery_evidence=dict(updates=0, role="synthetic_admission_not_M0"))
        if type(artifact) is not DeploymentArtifact:
            raise PilotError("Synthetic TensorRT admission failed: " + str(artifact))
        runtime = artifact.load()
        output = runtime(calibration[:2])
        import numpy as np
        if output.shape != (2, 10) or not np.isfinite(output).all():
            raise PilotError("Synthetic independently loaded TensorRT output is invalid")
        rows[recipe] = dict(manifest_path=artifact.manifest_path, storage_bytes=artifact.storage_bytes,
                            manifest_sha256=_sha(artifact.manifest_path), updates=0)
        del runtime
    torch.cuda.synchronize(0)
    if clock() >= deadline:
        raise TimeoutError("No late synthetic admission")
    versions = {package: importlib.metadata.version(package) for package in
                ("torch", "onnx", "onnxruntime")}
    import tensorrt
    versions["tensorrt"] = tensorrt.__version__
    return dict(kind="native_A100_synthetic_export_reload", device_name=name, cuda_binding=binding,
                synthetic=True, sgd_attempted=0, sgd_completed=0, artifacts=rows, versions=versions)


@dataclass(frozen=True)
class PilotEffects:
    """Explicit replacements are always labelled nonhardware, even if native-like."""
    admission: Callable = _native_admission
    train_baseline: Callable = _native_train
    prepare_space: Callable = _native_prepare
    export_baseline: Callable = _native_export
    evaluate_candidate: Callable = _native_evaluate
    model_identity: Callable = lambda value: _native_identity("model", value)
    data_identity: Callable = lambda value: _native_identity("data", value)
    calibration_identity: Callable = lambda value: _native_identity("calibration", value)
    calibration: Callable = _native_calibration
    artifact_identity: Callable = lambda value: _native_identity("artifact", value)


class _Journal:
    def __init__(self, root, clock):
        self.root, self.clock = root, clock
        self.counters = dict(baseline_sgd_attempted=0, baseline_sgd_completed=0,
                             recovery_sgd_attempted=0, recovery_sgd_completed=0)
        self.stage_seconds = {}
    def event(self, event, **details):
        with (self.root / "pilot.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_json_value(dict(event=event, at=self.clock(), **details)), allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
    @contextmanager
    def stage(self, name):
        began = self.clock()
        self.event("stage_started", stage=name)
        try:
            yield
        finally:
            elapsed = self.clock() - began
            self.stage_seconds[name] = elapsed
            self.event("stage_finished", stage=name, seconds=elapsed)


def run_pilot(data, output_root, *, config, started_at, boundary_check, effects=None, clock=time.monotonic):
    """No retries/resume. The external guard must release its start gate first.

    ``boundary_check(config=..., started_at=..., deadline=...)`` must verify the
    LIVE current worker cgroup and invocation against the launcher's receipt.
    Its returned provenance is journalled. Native execution also verifies the
    exact checked-out revision before any synthetic or training computation.
    """
    if (type(config) is not PilotConfig or type(started_at) not in (int, float)
            or not math.isfinite(started_at) or not 0 <= started_at <= clock()
            or not callable(boundary_check) or effects is not None and type(effects) is not PilotEffects):
        raise ValueError("Frozen pilot config, original start, live boundary verifier and typed effects are required")
    native = effects is None
    effects = effects or PilotEffects()
    deadline = started_at + config.gpu_seconds - config.cleanup_seconds
    root = Path(output_root).resolve(); root.mkdir(parents=True, exist_ok=False)
    journal = _Journal(root, clock)
    status, experiment_started, admission = "admission_failed", False, None
    baseline_record, candidates, failure, unknown_sgd = None, {}, None, []
    data_identity = calibration_identity = source_identity = None
    source = calibration = None
    _write_new(root / "plan.json", dict(schema="heterogeneous-pilot-plan-v1", config=config,
        started_at=started_at, deadline=deadline, budget=config.budget(),
        execution_mode="native" if native else "injected_effects_not_hardware_validation",
        official_test_opened=False, Vreport_opened=False, searches=0,
        module_sha256=_sha(__file__), python=sys.version))

    def check(*, identities=False):
        if clock() >= deadline:
            raise TimeoutError("Original pilot deadline exhausted; no automatic retry")
        if identities:
            if effects.data_identity(data) != data_identity:
                raise PilotError("Frozen train corpus or role identity changed")
            if calibration is not None and effects.calibration_identity(calibration) != calibration_identity:
                raise PilotError("Frozen calibration changed")
            if source is not None and effects.model_identity(source) != source_identity:
                raise PilotError("Frozen M0 changed")
        c = journal.counters
        if c["baseline_sgd_attempted"] + c["recovery_sgd_attempted"] > config.budget()["total_sgd"]:
            raise PilotError("Pilot SGD budget exceeded")

    try:
        with journal.stage("live_boundary"):
            check()
            boundary = boundary_check(config=config, started_at=started_at, deadline=deadline)
            if type(boundary) is not dict or not boundary:
                raise PilotError("A concrete live external deadline boundary receipt is mandatory")
            journal.event("external_boundary_verified", provenance=boundary)
            if native:
                actual = subprocess.run(("git", "rev-parse", "HEAD"), cwd=Path(__file__).resolve().parents[2],
                    check=True, capture_output=True, text=True, timeout=3).stdout.strip()
                if actual != config.source_revision:
                    raise PilotError("Checked-out revision differs from authorized pilot source")
            check()
        with journal.stage("synthetic_admission"):
            # Freeze exact recipes using the published finite catalog before
            # admission initializes CUDA or training can affect any choice.
            from types import SimpleNamespace
            from fedcore.algorithm.heterogeneous.cv import default_alternatives, genotype_payload
            catalog = SimpleNamespace(domains=tuple(SimpleNamespace(path=path, alternatives=choices)
                for path, choices in default_alternatives().items()))
            sealed_genotypes = fixed_genotypes(catalog, config.recipes)
            _write_new(root / "sealed-genotypes.json", {name: genotype_payload(genotype)
                       for name, genotype in sealed_genotypes.items()})
            check()
            admission = effects.admission(root / "admission", config, deadline, clock)
            if (type(admission) is not dict or admission.get("sgd_attempted") != 0
                    or admission.get("sgd_completed") != 0):
                raise PilotError("Admission must use zero SGD")
            check()
        from .heterogeneous_series import (BaselineResult, CandidateContext, CandidateEvaluationError,
            CandidateRun, _partial_accounting, _roles)
        from .heterogeneous_runtime import RecoveryConfig, recovery_tape
        from fedcore.algorithm.heterogeneous.contracts import fingerprint
        from fedcore.algorithm.heterogeneous.cv import compile_genotype, genotype_payload
        from fedcore.algorithm.low_rank.grammar_search import GrammarFitness
        from fedcore.algorithm.low_rank.plans import ValidationFailure
        _roles(data, config.protocol)
        if data.roles.seed != config.split_seed:
            raise PilotError("Data split seed differs from the frozen pilot configuration")
        data_identity = effects.data_identity(data)
        calibration = data.calibration_tensor()
        calibration_identity = effects.calibration(data, calibration)
        check(identities=True)
        status, experiment_started = "incomplete", True
        journal.event("experiment_started", model_seed=config.model_seed, data_identity=data_identity,
                      calibration_identity=calibration_identity, **journal.counters)
        def progress(event, count):
            if event in ("sgd_attempted", "sgd_completed"):
                key = "baseline_" + event
                current = journal.counters[key]
                if (type(count) is not int or not current <= count <= 25040
                        or event == "sgd_completed" and count > journal.counters["baseline_sgd_attempted"]):
                    raise PilotError("Baseline SGD accounting regressed or exceeded its bound")
                journal.counters[key] = count
            journal.event("baseline_progress", phase_event=event, count=count, **journal.counters)
            check(identities=event == "epoch_completed")
        with journal.stage("baseline_training"):
            baseline = effects.train_baseline(data, config.model_seed, root / "M0", protocol=config.protocol,
                device=config.device, deadline=deadline, clock=clock, progress=progress)
            if (type(baseline) is not BaselineResult or baseline.updates != 25040
                    or journal.counters["baseline_sgd_attempted"] != 25040
                    or journal.counters["baseline_sgd_completed"] != 25040
                    or not Path(baseline.checkpoint_path).is_file()):
                raise PilotError("The fresh M0 did not complete exactly 80 epochs / 25040 SGD")
            source = baseline.model; source_identity = effects.model_identity(source)
            baseline_record = dict(source_identity=source_identity, checkpoint_path=baseline.checkpoint_path,
                                   checkpoint_sha256=_sha(baseline.checkpoint_path), updates=25040)
        with journal.stage("prepare_space_and_fixed_plan"):
            check(identities=True)
            space = effects.prepare_space(source, calibration, config)
            if type(space) is ValidationFailure:
                raise PilotError("Pilot G1 preparation failed: " + str(space))
            genotypes = fixed_genotypes(space, config.recipes)
            if genotypes != sealed_genotypes:
                raise PilotError("Prepared G1 domain changed a recipe frozen before CUDA admission")
            workflows = {name: compile_genotype(genotype, space) for name, genotype in genotypes.items()}
            if any(type(w) is ValidationFailure for w in workflows.values()):
                raise PilotError("A fixed recipe did not compile")
            if len({w.fingerprint for w in workflows.values()}) != len(workflows):
                raise PilotError("Fixed pilot recipes must have distinct executable plans")
            _write_new(root / "fixed-recipes.json", {name: dict(genotype=genotype_payload(genotypes[name]),
                workflow_fingerprint=workflow.fingerprint) for name, workflow in workflows.items()})
            tape = recovery_tape(data.roles.fit, model_seed=config.model_seed, config=RecoveryConfig())
            baseline_record["recovery_tape_fingerprint"] = fingerprint(tape)
        with journal.stage("baseline_deployment"):
            check(identities=True)
            s0 = effects.export_baseline(source, calibration, space, root / "S0", config, deadline)
            baseline_record.update(S0=asdict(s0), S0_manifest_sha256=effects.artifact_identity(s0))
            check(identities=True)
        for name, workflow in workflows.items():
            check(identities=True)
            context = CandidateContext(source, calibration, tape, data, space, str(root / "recipes" / name),
                                       deadline, config, config.model_seed)
            with journal.stage("recipe:" + name):
                journal.event("recipe_started", recipe=name, workflow=workflow.fingerprint)
                try:
                    result = effects.evaluate_candidate(context, workflow)
                except Exception as error:
                    accounting = (error.accounting if type(error) is CandidateEvaluationError else
                                  _partial_accounting(context, workflow))
                    if accounting is None:
                        unknown_sgd.append(dict(recipe=name, reason="No trustworthy durable candidate accounting"))
                    else:
                        for kind in ("attempted", "completed"):
                            journal.counters["recovery_sgd_" + kind] += accounting["sgd_" + kind]
                    journal.event("recipe_exception", recipe=name, accounting=accounting, error=str(error),
                                  **journal.counters)
                    raise
                if (type(result) is not CandidateRun or any(type(v) is not int or not 0 <= v <= 200
                        for v in (result.sgd_attempted, result.sgd_completed))
                        or result.sgd_completed > result.sgd_attempted):
                    accounting = _partial_accounting(context, workflow)
                    if accounting is None:
                        unknown_sgd.append(dict(recipe=name, reason="Malformed candidate accounting"))
                    else:
                        for kind in ("attempted", "completed"):
                            journal.counters["recovery_sgd_" + kind] += accounting["sgd_" + kind]
                    raise PilotError("Invalid candidate SGD accounting")
                for kind in ("attempted", "completed"):
                    journal.counters["recovery_sgd_" + kind] += getattr(result, "sgd_" + kind)
                row = dict(workflow=workflow.fingerprint, outcome=type(result.outcome).__name__,
                    sgd_attempted=result.sgd_attempted, sgd_completed=result.sgd_completed,
                    receipt_path=result.receipt_path)
                candidates[name] = row
                journal.event("recipe_finished", recipe=name, **row, **journal.counters)
                if (type(result.outcome) is not GrammarFitness or result.artifact is None
                        or result.sgd_attempted != 200 or result.sgd_completed != 200
                        or any(type(v) not in (float, int) or not math.isfinite(v) or v < 0
                               for v in (result.outcome.quality, result.outcome.cost))
                        or result.outcome.cost != result.artifact.storage_bytes):
                    raise PilotError("Fixed recipe failed; no fallback or automatic retry: " + name)
                row.update(cross_entropy_Vsearch=result.outcome.quality, artifact=asdict(result.artifact),
                           manifest_sha256=effects.artifact_identity(result.artifact))
                if native:
                    receipt = json.loads(Path(result.receipt_path).read_text(encoding="utf-8"))
                    manifest = json.loads(Path(result.artifact.manifest_path).read_text(encoding="utf-8"))
                    if (receipt["validation_rows"] != 4000 or receipt["quality"]["rows"] != 4000
                            or receipt["validation_identity"] != data.roles.sha256 + ":Vsearch"
                            or manifest["calibration_fingerprint"] != calibration_identity
                            or manifest["backend"] != "tensorrt"):
                        raise PilotError("Candidate does not prove full calibration and loaded TensorRT Vsearch quality")
                    row.update(quality=receipt["quality"], phase_seconds=receipt["phase_seconds"])
                check(identities=True)
        check(identities=True)
        status = "complete"
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error))
        journal.event("pilot_failed", experiment_started=experiment_started, failure=failure, **journal.counters)
    receipt = dict(schema="heterogeneous-pilot-receipt-v1", status=status, experiment_started=experiment_started,
        execution_mode="native" if native else "injected_effects_not_hardware_validation",
        config=config, budget=config.budget(), started_at=started_at, deadline=deadline, ended_at=clock(),
        admission=admission, baseline=baseline_record, candidates=candidates, failure=failure,
        counters=journal.counters, unknown_sgd=unknown_sgd, phase_seconds=journal.stage_seconds,
        recovery_in_flight_unconfirmed=journal.counters["recovery_sgd_attempted"] - journal.counters["recovery_sgd_completed"],
        baseline_in_flight_unconfirmed=journal.counters["baseline_sgd_attempted"] - journal.counters["baseline_sgd_completed"],
        official_test_opened=False, Vreport_opened=False, searches=0, automatic_retries=0,
        interpretation="cost pilot only; no 27-search panel, inference latency claim or ranking validation")
    _write_new(root / "receipt.json", receipt)
    journal.event("pilot_finished", status=status, **journal.counters)
    return _json_value(receipt)


def verify_worker_boundary(receipt_path, *, config, started_at, deadline):
    """Read-only live check of the exact current systemd invocation.

    Receipt is ``dataclasses.asdict(DeadlineReceipt)`` from the owned launcher.
    The guard must first validate it and then release its separate start gate.
    This verifier neither starts services nor stops processes.
    """
    from .heterogeneous_deadline import (BoundedLocalTransport, DeadlineBudget, DeadlineRequest,
        _properties, compile_launch, parse_systemd_duration)
    if sys.platform != "linux":
        raise PilotError("Native pilot requires a direct Linux systemd worker")
    with Path(receipt_path).open("r", encoding="utf-8") as stream:
        raw = stream.read(65537)
    if len(raw.encode()) > 65536:
        raise PilotError("External deadline receipt exceeds the bounded profile")
    receipt = json.loads(raw)
    plan, request = receipt["plan"], receipt["plan"]["request"]
    if (type(plan.get("start_microseconds")) is not int or plan["start_microseconds"] <= 0
            or any(type(plan.get(key)) is not int or plan[key] <= 0
                   for key in ("runtime_microseconds", "stop_microseconds"))):
        raise PilotError("External OS durations must be positive finite integer microseconds")
    # Recompile the pure launch plan, including transport/start/stop reserves.
    # A changed JSON runtime cannot enlarge a live unit's allowed 2h window.
    typed_request = DeadlineRequest(**{**request, "argv": tuple(request["argv"])})
    typed_budget = DeadlineBudget(started_at, config.gpu_seconds, cleanup_seconds=config.cleanup_seconds,
                                  start_timeout_seconds=plan["start_microseconds"] / 1_000_000)
    compiled = compile_launch(typed_request, typed_budget, now=plan["compiled_at"])
    if _json_value(compiled) != plan:
        raise PilotError("External OS launch plan differs from its original bounded controller window")
    token, invocation, group = request["token"], receipt["invocation_id"], receipt["control_group"]
    if (re.fullmatch(r"[a-f0-9]{32}", token) is None or re.fullmatch(r"[a-f0-9]{32}", invocation) is None
            or request["launch_kind"] != "direct_guard_worker" or request["gpu_uuid"] != config.gpu_uuid
            or plan["deadline_at"] != started_at + config.gpu_seconds
            or not deadline < plan["deadline_at"]
            or os.environ.get("INVOCATION_ID") != invocation
            or os.environ.get("CUDA_VISIBLE_DEVICES") != config.gpu_uuid):
        raise PilotError("Launcher receipt, original window and worker environment do not agree")
    current = Path("/proc/self/cgroup").read_text(encoding="ascii").strip()
    if current != "0::" + group or not group.endswith("/fedcore-heterogeneous-" + token + ".service"):
        raise PilotError("Current worker is outside the exact owned service cgroup")
    fields = ("InvocationID", "Description", "ControlGroup", "ActiveState", "KillMode", "SendSIGKILL",
              "RuntimeMaxUSec", "TimeoutStartUSec", "TimeoutStopUSec", "Delegate", "ProtectControlGroups", "Restart",
              "FinalKillSignal", "TimeoutStopFailureMode", "NoNewPrivileges", "Type", "WorkingDirectory")
    unit = "fedcore-heterogeneous-" + token + ".service"
    values = _properties(BoundedLocalTransport()(("systemctl", "--user", "show") + tuple(
        "--property=" + key for key in fields) + ("--", unit), timeout=3.))
    expected = dict(InvocationID=invocation, Description="fedcore-owned-deadline:" + token,
        ControlGroup=group, ActiveState="active", KillMode="control-group", SendSIGKILL="yes",
        Delegate="no", ProtectControlGroups="yes", Restart="no", TimeoutStopFailureMode="kill", NoNewPrivileges="yes",
        Type="exec", WorkingDirectory=request["working_directory"])
    if any(values.get(key) != value for key, value in expected.items()) or values.get("FinalKillSignal") not in ("9", "SIGKILL"):
        raise PilotError("Live systemd invocation does not satisfy the owned whole-tree kill contract")
    for name, key in (("RuntimeMaxUSec", "runtime_microseconds"), ("TimeoutStartUSec", "start_microseconds"),
                      ("TimeoutStopUSec", "stop_microseconds")):
        actual = parse_systemd_duration(values[name])
        if type(plan[key]) is not int or not 0 < actual <= plan[key]:
            raise PilotError("Live OS deadline exceeds the validated launcher limit")
    return dict(kind="live_systemd_worker_boundary", invocation_id=invocation, control_group=group,
                receipt_sha256=_sha(receipt_path), worker_pid=os.getpid(), unit=unit,
                deadline_at=plan["deadline_at"], properties=values)
