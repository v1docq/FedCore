"""Count the entire panel through explicit fakes; no training/CUDA/ORT calls."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fedcore.algorithm.heterogeneous.contracts import fingerprint
from fedcore.algorithm.heterogeneous.cv import (
    HeterogeneousDomain, HeterogeneousSpace, compile_genotype, default_alternatives, genotype_payload)
from fedcore.algorithm.heterogeneous.search import common_initial
from fedcore.algorithm.low_rank.compression_graph import CalibrationRef, ModelVersion
from fedcore.algorithm.low_rank.cv_grammar import CVBlockDomain, CVGrammarSpace
from fedcore.algorithm.low_rank.grammar_search import GrammarFitness
from fedcore.experiments.heterogeneous_data import DataRoles
from fedcore.experiments.heterogeneous_series import (
    ArtifactRecord, BaselineResult, CandidateContext, CandidateEvaluationError, CandidateRun,
    SeriesConfig, SeriesEffects, SeriesError, _evaluate, _validate_boundary, _validate_timing, run_series)


class Data:
    def __init__(self):
        self.roles = DataRoles(tuple(range(40000)), tuple(range(40000, 42000)),
                               tuple(range(42000, 46000)), tuple(range(46000, 50000)), 1)
        self.identity = "sealed-data"
    def calibration_tensor(self):
        return SimpleNamespace(version=0)
    def training(self, indices, seed):
        raise AssertionError("Counting fakes must never request real training batches")


class Effects:
    def __init__(self):
        self.trained, self.candidates, self.reported, self.timed, self.aa, self.prefixes, self.tapes = [], [], [], [], [], {}, {}
        self.mutate_data = False
        self.search_override = None

    def model_identity(self, model):
        return hashlib.sha256(str(model.seed).encode()).hexdigest()

    def train(self, data, seed, directory, *, protocol, device, deadline, clock, progress):
        assert protocol.epochs == 80 and protocol.fit_rows == 40000 and protocol.batch_size == 128
        assert device == "cuda:0"  # An injected fake never touches this device.
        directory.mkdir()
        checkpoint = directory / "baseline.pt"
        checkpoint.write_text(str(seed), encoding="ascii")
        updates = 80 * 313
        progress("sgd_attempted", updates); progress("sgd_completed", updates)
        self.trained.append(seed)
        return BaselineResult(SimpleNamespace(seed=seed), updates, str(checkpoint))

    def prepare(self, source, calibration, config):
        domains = tuple(HeterogeneousDomain(path, choices) for path, choices in default_alternatives().items()
                        if path in ("layer1.0", "layer1.1"))
        structural = CVGrammarSpace(ModelVersion("fake", self.model_identity(source)),
            CalibrationRef("b" * 64, (2, 3, 32, 32), "float32"),
            tuple(CVBlockDomain(domain.path, tuple(dict.fromkeys(choice.structural for choice in domain.alternatives)))
                  for domain in domains), "c" * 64, 32 * 1024**2)
        return HeterogeneousSpace(structural, domains)

    def artifact(self, directory, value, size):
        directory = Path(directory); directory.mkdir(parents=True)
        manifest = directory / "manifest.json"
        manifest.write_text(json.dumps(dict(value=value, size=size)), encoding="utf-8")
        return ArtifactRecord(str(manifest), size)

    def export(self, source, calibration, space, destination, config, deadline):
        return self.artifact(destination, "S0-" + str(source.seed), 1000)

    def artifact_identity(self, artifact):
        raw = Path(artifact.manifest_path).read_bytes()
        assert json.loads(raw)["size"] == artifact.storage_bytes
        return hashlib.sha256(raw).hexdigest()

    def evaluate(self, context, workflow):
        self.tapes.setdefault(context.model_seed, set()).add(fingerprint(context.tape))
        self.candidates.append((context.model_seed, context.output_root, workflow.fingerprint))
        artifact = self.artifact(Path(context.output_root) / workflow.fingerprint / "artifact", workflow.fingerprint, 100)
        if self.mutate_data:
            context.data.identity = "stale-data"
        quality = .5 + int(workflow.fingerprint[:4], 16) / 1e6
        return CandidateRun(GrammarFitness(quality, 100.), artifact, 200, 200)

    def search(self, space, config, evaluate, **options):
        if self.search_override:
            return self.search_override(space, config, evaluate, **options)
        assert (config.generations, config.population_size, config.max_evaluations, config.max_proposals) == (64, 8, 64, 640)
        prefix = options["initial_genotypes"]
        key = (space.structural_space.source.fingerprint, config.seed)
        self.prefixes.setdefault(key, set()).add(fingerprint(prefix))
        choices = list(prefix)
        known = {fingerprint(g) for g in choices}
        # Complete the counted fake trajectory with distinct valid plans.
        for genotype in common_initial(space, 64 + config.population_size, config.seed + 10000):
            if fingerprint(genotype) not in known:
                choices.append(genotype); known.add(fingerprint(genotype))
            if len(choices) == 64:
                break
        rows = []
        for index, genotype in enumerate(choices):
            workflow = compile_genotype(genotype, space)
            options["trace_callback"](dict(event="evaluation_started", attempt_id=index + 1))
            outcome = evaluate(workflow)
            rows.append(dict(canonical_attempt_key="canonical-" + workflow.fingerprint,
                workflow_fingerprint=workflow.fingerprint, quality=outcome.quality,
                cost=outcome.cost, quality_accepted=True, genotype=genotype_payload(genotype)))
        return dict(status="succeeded", charged_evaluations=64, stop_reason="evaluation_budget",
                    finite_archive=[min(rows, key=lambda r: (r["quality"], r["canonical_attempt_key"]))])

    def report(self, data, artifact, *, config, deadline, seal_path):
        seal = json.loads(Path(seal_path).read_text(encoding="utf-8"))
        assert len(seal["searches"]) == 27 and len(seal["controls"]) == 15 and len(seal["models"]) == 3
        assert len(seal["final_artifacts"]) == 45
        self.reported.append(artifact.manifest_path)
        return dict(cross_entropy=.6, accuracy=.8, rows=4000)

    def latency(self, artifact, **options):
        assert Path(options["seal_path"]).exists()
        self.timed.append((artifact.manifest_path, options["batch_size"], options["window"]))
        return {"counting_fake_not_hardware": True}

    def boundary(self, artifact, **options):
        assert Path(options["seal_path"]).exists()
        self.aa.append((artifact.manifest_path, options["batch_size"], options["window"], options["phase"]))
        return {"counting_fake_not_hardware": True, "aa_passed": True}

    def effects(self):
        return SeriesEffects(train_baseline=self.train, prepare_space=self.prepare, export_baseline=self.export,
            evaluate_candidate=self.evaluate, search=self.search, model_identity=self.model_identity,
            data_identity=lambda data: data.identity, calibration_identity=lambda value: str(value.version),
            artifact_identity=self.artifact_identity,
            report=self.report, latency=self.latency, latency_boundary=self.boundary)


def config():
    return SeriesConfig(gpu_seconds=100000, search_seconds=1000)


def test_full_panel_counts_three_baselines_fifteen_fresh_controls_and_twentyseven_evo_jobs(tmp_path):
    fake = Effects()
    receipt = run_series(Data(), tmp_path / "panel", config=config(), started_at=100., effects=fake.effects(), clock=lambda: 100.)
    assert receipt["status"] == "complete", receipt["failures"]
    assert fake.trained == list(config().protocol.model_seeds)
    assert len(fake.candidates) == 15 + 27 * 64
    assert receipt["counters"] == dict(baseline_sgd_attempted=75120, baseline_sgd_completed=75120,
        recovery_sgd_attempted=348600, recovery_sgd_completed=348600, charged_search_attempts=1728)
    assert len(receipt["searches"]) == 27 and len(receipt["controls"]) == 15
    assert len(fake.reported) == 45 and len(fake.timed) == 180 and len(fake.aa) == 24
    assert all(len(prefixes) == 1 for prefixes in fake.prefixes.values())
    assert len(fake.prefixes) == 9 and all(len(tapes) == 1 for tapes in fake.tapes.values())
    assert len({directory for _, directory, _ in fake.candidates}) == 42
    assert receipt["official_test_opened"] is False
    assert receipt["execution_mode"] == "injected_effects_not_hardware_validation"
    assert Path(receipt["panel_seal"]).is_file()
    assert sum(key.startswith("aa:") for key in receipt["latency"]) == 24
    assert all(row["admitted"] for key, row in receipt["latency"].items() if not key.startswith("aa:"))
    # A second call cannot replace an earlier panel, seal, checkpoint or report.
    with pytest.raises(FileExistsError):
        run_series(Data(), tmp_path / "panel", config=config(), started_at=100., effects=fake.effects(), clock=lambda: 100.)


def test_stale_data_stops_before_report_and_preserves_counts_and_missing_pairs(tmp_path):
    fake = Effects(); fake.mutate_data = True
    receipt = run_series(Data(), tmp_path / "panel", config=config(), started_at=100., effects=fake.effects(), clock=lambda: 100.)
    assert receipt["status"] == "failed"
    assert "Stale dataset" in receipt["failures"][0]["message"]
    assert len(receipt["searches"]) == 27 and len(receipt["controls"]) == 15
    assert not fake.reported and not fake.timed and receipt["panel_seal"] is None
    assert receipt["counters"]["recovery_sgd_completed"] == 200


def test_duplicate_evaluator_call_cannot_reuse_a_parent_or_control_recovery(tmp_path):
    fake = Effects()
    def duplicate(space, config, evaluate, **options):
        workflow = compile_genotype(options["initial_genotypes"][0], space)
        for index in range(2):
            options["trace_callback"](dict(event="evaluation_started", attempt_id=index + 1))
            evaluate(workflow)
    fake.search_override = duplicate
    receipt = run_series(Data(), tmp_path / "panel", config=config(), started_at=100., effects=fake.effects(), clock=lambda: 100.)
    assert receipt["status"] == "failed" and "cache/recovery reuse" in receipt["failures"][0]["message"]
    assert receipt["counters"]["charged_search_attempts"] == 2
    assert receipt["counters"]["recovery_sgd_completed"] == 6 * 200
    assert not fake.reported and receipt["panel_seal"] is None


def test_missing_executable_finalist_is_not_rebuilt_or_replaced_from_archive(tmp_path):
    fake = Effects()
    original = fake.search
    def wrong_identity(space, config, evaluate, **options):
        override = fake.search_override; fake.search_override = None
        try:
            result = original(space, config, evaluate, **options)
        finally:
            fake.search_override = override
        result["finite_archive"][0]["workflow_fingerprint"] = "missing-executable"
        return result
    fake.search_override = wrong_identity
    receipt = run_series(Data(), tmp_path / "panel", config=config(), started_at=100., effects=fake.effects(), clock=lambda: 100.)
    assert receipt["status"] == "failed" and "workflow identity" in receipt["failures"][0]["message"]
    assert not fake.reported and not fake.timed
    assert len(fake.trained) == 1


def test_original_deadline_and_explicit_latency_are_required_before_any_training(tmp_path):
    fake = Effects()
    with pytest.raises(ValueError, match="latency"):
        run_series(Data(), tmp_path / "none", config=config(), started_at=100.,
                   effects=replace(fake.effects(), latency=None), clock=lambda: 100.)
    with pytest.raises(TimeoutError, match="before device binding"):
        run_series(Data(), tmp_path / "expired", config=SeriesConfig(gpu_seconds=61, search_seconds=10),
                   started_at=100., effects=fake.effects(), clock=lambda: 102.)
    assert not fake.trained and not fake.reported


def test_unexpected_candidate_exception_preserves_partial_sgd_and_stops_the_panel(tmp_path):
    fake = Effects()
    def broken(context, workflow):
        directory = Path(context.output_root) / workflow.fingerprint
        directory.mkdir(parents=True)
        (directory / "progress.jsonl").write_text(
            json.dumps(dict(event="sgd_attempted", updates=2)) + "\n" +
            json.dumps(dict(event="sgd_completed", updates=1)) + "\n", encoding="utf-8")
        raise ArithmeticError("injected failure after partial SGD")
    effects = replace(fake.effects(), evaluate_candidate=broken)
    receipt = run_series(Data(), tmp_path / "partial", config=config(), started_at=100., effects=effects, clock=lambda: 100.)
    assert receipt["status"] == "failed" and not fake.reported
    assert receipt["counters"]["recovery_sgd_attempted"] == 2
    assert receipt["counters"]["recovery_sgd_completed"] == 1
    assert not receipt["unknown_sgd_trajectories"]
    assert "panel stopped" in receipt["failures"][0]["message"]


def test_default_evaluator_attaches_durable_partial_receipt_to_unexpected_exception(monkeypatch, tmp_path):
    from fedcore.experiments import heterogeneous_series as series
    fake, data = Effects(), Data()
    source = SimpleNamespace(seed=1)
    space = fake.prepare(source, "calibration", config())
    workflow = compile_genotype(common_initial(space, 1, 1)[0], space)
    context = CandidateContext(source, "calibration", (), data, space, str(tmp_path), 1000., config(), 1)
    class Broken:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, workflow):
            directory = tmp_path / workflow.fingerprint
            directory.mkdir()
            (directory / "receipt.json").write_text(json.dumps(dict(workflow_fingerprint=workflow.fingerprint,
                sgd_attempted=3, sgd_completed=2)), encoding="utf-8")
            raise ArithmeticError("late callback failure")
    monkeypatch.setattr(series, "CandidateEvaluator", Broken)
    with pytest.raises(CandidateEvaluationError) as captured:
        _evaluate(context, workflow)
    assert captured.value.accounting["sgd_attempted"] == 3
    assert captured.value.accounting["sgd_completed"] == 2
    assert captured.value.accounting["in_flight_unconfirmed"] == 1


def test_failed_post_aa_rejects_the_whole_window_and_never_retries(tmp_path):
    fake = Effects()
    # One finite candidate plus 63 explicit failed compilation attempts is enough
    # to exercise the sealed orchestration, without a second large artifact set.
    def sparse_search(space, config, evaluate, **options):
        workflow = compile_genotype(options["initial_genotypes"][0], space)
        for index in range(64):
            options["trace_callback"](dict(event="evaluation_started", attempt_id=index + 1))
            if index:
                options["trace_callback"](dict(event="evaluated", outcome="StructuralInvalid", attempt_id=index + 1))
            else:
                outcome = evaluate(workflow)
        return dict(status="succeeded", charged_evaluations=64, finite_archive=[dict(
            canonical_attempt_key="canonical-" + workflow.fingerprint, workflow_fingerprint=workflow.fingerprint,
            quality=outcome.quality, cost=outcome.cost, quality_accepted=True, genotype=genotype_payload(workflow.genotype))])
    def boundary(artifact, **options):
        evidence = fake.boundary(artifact, **options)
        evidence["aa_passed"] = options["phase"] == "pre"
        return evidence
    effects = replace(fake.effects(), search=sparse_search, latency_boundary=boundary)
    receipt = run_series(Data(), tmp_path / "bad-aa", config=config(), started_at=100., effects=effects, clock=lambda: 100.)
    assert receipt["status"] == "failed" and "A/A failed" in receipt["failures"][0]["message"]
    assert len(fake.reported) == 45 and len(fake.aa) == 2 and len(fake.timed) == 15
    measurements = [row for key, row in receipt["latency"].items() if not key.startswith("aa:")]
    assert len(measurements) == 15 and all(row["admitted"] is False for row in measurements)


def test_separate_calibration_mutation_is_rejected_even_when_dataset_is_unchanged(tmp_path):
    fake = Effects()
    def mutate(context, workflow):
        result = fake.evaluate(context, workflow)
        context.calibration.version += 1
        return result
    receipt = run_series(Data(), tmp_path / "calibration", config=config(), started_at=100.,
        effects=replace(fake.effects(), evaluate_candidate=mutate), clock=lambda: 100.)
    assert receipt["status"] == "failed" and "Stale calibration" in receipt["failures"][0]["message"]
    assert receipt["counters"]["recovery_sgd_completed"] == 200 and not fake.reported


@pytest.mark.parametrize("durable", [False, True])
def test_malformed_candidate_result_uses_durable_counts_or_explicit_unknown_marker(tmp_path, durable):
    fake = Effects()
    def malformed(context, workflow):
        directory = Path(context.output_root) / workflow.fingerprint
        directory.mkdir(parents=True)
        (directory / "receipt.json").write_text(json.dumps(dict(workflow_fingerprint=workflow.fingerprint,
            sgd_attempted=3, sgd_completed=2)) if durable else "{corrupt", encoding="utf-8")
        return CandidateRun(None, None, -1, -1)
    receipt = run_series(Data(), tmp_path / "malformed", config=config(), started_at=100.,
        effects=replace(fake.effects(), evaluate_candidate=malformed), clock=lambda: 100.)
    assert receipt["status"] == "failed" and not fake.reported
    assert bool(receipt["unknown_sgd_trajectories"]) == (not durable)
    assert receipt["counters"]["recovery_sgd_attempted"] == (3 if durable else 0)
    assert receipt["counters"]["recovery_sgd_completed"] == (2 if durable else 0)


def test_last_post_boundary_cannot_modify_artifact_and_still_complete(tmp_path):
    fake = Effects()
    def sparse_search(space, config, evaluate, **options):
        workflow = compile_genotype(options["initial_genotypes"][0], space)
        for index in range(64):
            options["trace_callback"](dict(event="evaluation_started", attempt_id=index + 1))
            if not index:
                outcome = evaluate(workflow)
        return dict(status="succeeded", charged_evaluations=64, finite_archive=[dict(
            canonical_attempt_key="canonical-" + workflow.fingerprint, workflow_fingerprint=workflow.fingerprint,
            quality=outcome.quality, cost=outcome.cost, quality_accepted=True, genotype=genotype_payload(workflow.genotype))])
    def mutate_last(artifact, **options):
        result = fake.boundary(artifact, **options)
        if len(fake.aa) == 24:
            path = Path(artifact.manifest_path)
            value = json.loads(path.read_text(encoding="utf-8")); value["changed"] = True
            path.write_text(json.dumps(value), encoding="utf-8")
        return result
    receipt = run_series(Data(), tmp_path / "last-callback", config=config(), started_at=100.,
        effects=replace(fake.effects(), search=sparse_search, latency_boundary=mutate_last), clock=lambda: 100.)
    assert len(fake.aa) == 24 and receipt["status"] == "failed"
    assert "deployment changed during a callback" in receipt["failures"][0]["message"]


@pytest.mark.parametrize("pairs", [[], [{"baseline_ms": 1., "candidate_ms": float("nan")},
                                       {"baseline_ms": 1., "candidate_ms": 2.}],
    [{"baseline_ms": 1., "candidate_ms": 2.}]])
def test_native_timing_and_aa_reject_empty_nan_or_wrong_count_raw_pairs(pairs):
    cfg = replace(config(), latency_protocol_id="sealed-pilot", latency_pairs_per_session=2,
                  latency_aa_pairs_per_session=2)
    evidence = dict(hardware_timing=True, backend="tensorrt", device_uuid=cfg.backend_spec.device_uuid,
        batch_size=1, window=0, protocol_id=cfg.latency_protocol_id, timing_source="cuda_events",
        phase="pre", aa_passed=True, units="ms", raw_pairs=pairs)
    with pytest.raises(SeriesError, match="raw timing pairs"):
        _validate_timing(evidence, True, cfg, 1, 0)
    with pytest.raises(SeriesError, match="raw timing pairs"):
        _validate_boundary(evidence, True, cfg, 1, 0, "pre")


def test_expired_native_window_refuses_before_cuda_binding(monkeypatch, tmp_path):
    from fedcore.experiments import heterogeneous_device
    calls = []
    monkeypatch.setattr(heterogeneous_device, "require_cuda_binding", lambda uuid: calls.append(uuid))
    with pytest.raises(TimeoutError, match="before device binding"):
        run_series(Data(), tmp_path / "native-expired", config=SeriesConfig(gpu_seconds=61, search_seconds=10),
                   started_at=100., clock=lambda: 102.)
    assert calls == [] and not (tmp_path / "native-expired").exists()
