"""Pilot budget/order accounting with CPU fakes; never hardware evidence."""
from contextlib import nullcontext
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from fedcore.algorithm.heterogeneous.cv import HeterogeneousDomain, HeterogeneousSpace, default_alternatives
from fedcore.algorithm.low_rank.compression_graph import CalibrationRef, ModelVersion
from fedcore.algorithm.low_rank.cv_grammar import CVBlockDomain, CVGrammarSpace
from fedcore.algorithm.low_rank.grammar_execution import ResourceFailure
from fedcore.algorithm.low_rank.grammar_search import GrammarFitness
from fedcore.experiments.heterogeneous_data import DataRoles
from fedcore.experiments.heterogeneous_pilot import (
    FIXED_RECIPES, PilotConfig, PilotEffects, fixed_genotypes, pilot_config_from_payload, run_pilot)
from fedcore.experiments.heterogeneous_runtime import RecoveryConfig, recover_owned, recovery_tape
from fedcore.experiments.heterogeneous_series import ArtifactRecord, BaselineResult, CandidateRun


GPU = "GPU-12345678-1234-1234-1234-123456789abc"


def config(**kwargs):
    return PilotConfig(GPU, "a" * 40, **kwargs)


class Data:
    def __init__(self):
        self.roles = DataRoles(tuple(range(40000)), tuple(range(40000, 42000)),
                               tuple(range(42000, 46000)), tuple(range(46000, 50000)), 20261102)
        self.identity = "sealed-train-and-roles"
    def calibration_tensor(self):
        return SimpleNamespace(version=0)
    def training(self, *args):
        raise AssertionError("Counting fake must not load training images")
    def search_batches(self, *args):
        raise AssertionError("Counting fake must not load evaluation images")


class Fake:
    def __init__(self):
        self.events, self.tapes, self.directories = [], [], []
    def admission(self, directory, cfg, deadline, clock):
        assert (directory.parent / "sealed-genotypes.json").is_file()
        self.events.append("admission")
        return dict(synthetic=True, sgd_attempted=0, sgd_completed=0)
    def train(self, data, seed, directory, *, protocol, device, deadline, clock, progress):
        self.events.append("train")
        assert (protocol.epochs, protocol.fit_rows, protocol.batch_size) == (80, 40000, 128)
        directory.mkdir()
        path = directory / "baseline.pt"; path.write_text("fake-owned-baseline")
        progress("sgd_attempted", 25040); progress("sgd_completed", 25040)
        return BaselineResult(SimpleNamespace(seed=seed), 25040, str(path))
    def prepare(self, source, calibration, cfg):
        domains = tuple(HeterogeneousDomain(path, alternatives) for path, alternatives in default_alternatives().items())
        structural = CVGrammarSpace(ModelVersion("fake", self.model_identity(source)),
            CalibrationRef("b" * 64, (2000, 3, 32, 32), "float32"),
            tuple(CVBlockDomain(d.path, tuple(dict.fromkeys(r.structural for r in d.alternatives))) for d in domains),
            "c" * 64, 32 * 1024**2)
        return HeterogeneousSpace(structural, domains)
    def artifact(self, directory, value, size=100):
        directory = Path(directory); directory.mkdir(parents=True)
        manifest = directory / "manifest.json"
        manifest.write_text(json.dumps(dict(identity=value, storage_bytes=size)))
        return ArtifactRecord(str(manifest), size)
    def export(self, source, calibration, space, directory, cfg, deadline):
        self.events.append("S0")
        return self.artifact(directory, "S0", 1000)
    def model_identity(self, source):
        return hashlib.sha256(str(source.seed).encode()).hexdigest()
    def evaluate(self, context, workflow):
        self.events.append("candidate")
        self.tapes.append(context.tape); self.directories.append(context.output_root)
        artifact = self.artifact(Path(context.output_root) / workflow.fingerprint, workflow.fingerprint)
        return CandidateRun(GrammarFitness(.7, 100.), artifact, 200, 200)
    def effects(self):
        return PilotEffects(admission=self.admission, train_baseline=self.train, prepare_space=self.prepare,
            export_baseline=self.export, evaluate_candidate=self.evaluate, model_identity=self.model_identity,
            data_identity=lambda data: data.identity, calibration_identity=lambda tensor: str(tensor.version),
            calibration=lambda data, tensor: str(tensor.version), artifact_identity=lambda a: str(a.storage_bytes))


def run(tmp_path, fake, **kwargs):
    return run_pilot(Data(), tmp_path / "pilot", config=config(), started_at=100.,
        boundary_check=lambda **kw: {"injected_boundary_not_hardware": True},
        effects=kwargs.pop("effects", fake.effects()), clock=lambda: 100., **kwargs)


def test_one_fresh_m0_twelve_fixed_recipes_and_27440_sgd_without_search_or_report(tmp_path):
    fake = Fake(); receipt = run(tmp_path, fake)
    assert receipt["status"] == "complete", receipt["failure"]
    assert fake.events == ["admission", "train", "S0"] + ["candidate"] * 12
    assert tuple(receipt["candidates"]) == FIXED_RECIPES
    assert receipt["counters"] == dict(baseline_sgd_attempted=25040, baseline_sgd_completed=25040,
                                        recovery_sgd_attempted=2400, recovery_sgd_completed=2400)
    assert len(set(fake.directories)) == 12 and all(tape is fake.tapes[0] for tape in fake.tapes)
    assert receipt["experiment_started"] and receipt["searches"] == 0 and receipt["automatic_retries"] == 0
    assert receipt["Vreport_opened"] is False and receipt["official_test_opened"] is False
    assert receipt["execution_mode"] == "injected_effects_not_hardware_validation"
    assert (tmp_path / "pilot" / "receipt.json").is_file()
    with pytest.raises(FileExistsError):
        run(tmp_path, fake)


def test_synthetic_admission_failure_is_distinct_from_experiment_start(tmp_path):
    fake = Fake()
    def rejected(*args):
        raise RuntimeError("synthetic INT8 inspector refused")
    receipt = run(tmp_path, fake, effects=replace(fake.effects(), admission=rejected))
    assert receipt["status"] == "admission_failed" and not receipt["experiment_started"]
    assert set(receipt["counters"].values()) == {0} and fake.events == []
    assert not receipt["candidates"] and "inspector" in receipt["failure"]["message"]


def test_external_deadline_rejection_precedes_synthetic_admission_and_training(tmp_path):
    fake = Fake()
    def rejected(**kwargs):
        raise RuntimeError("worker cgroup is foreign")
    receipt = run_pilot(Data(), tmp_path / "pilot", config=config(), started_at=100., boundary_check=rejected,
        effects=fake.effects(), clock=lambda: 100.)
    assert receipt["status"] == "admission_failed" and not receipt["experiment_started"] and not fake.events


def test_first_recipe_failure_stops_without_retry_and_retains_spent_updates(tmp_path):
    fake = Fake()
    def failure(context, workflow):
        fake.events.append("failed_candidate")
        return CandidateRun(ResourceFailure("Rejected", "injected"), None, 2, 1)
    receipt = run(tmp_path, fake, effects=replace(fake.effects(), evaluate_candidate=failure))
    assert receipt["status"] == "incomplete" and receipt["experiment_started"]
    assert fake.events == ["admission", "train", "S0", "failed_candidate"]
    assert receipt["counters"]["recovery_sgd_attempted"] == 2
    assert receipt["counters"]["recovery_sgd_completed"] == 1 and receipt["recovery_in_flight_unconfirmed"] == 1


def test_unexpected_candidate_error_reads_durable_partial_sgd(tmp_path):
    fake = Fake()
    def broken(context, workflow):
        directory = Path(context.output_root) / workflow.fingerprint; directory.mkdir(parents=True)
        (directory / "progress.jsonl").write_text(json.dumps(dict(event="sgd_attempted", updates=3)) + "\n" +
            json.dumps(dict(event="sgd_completed", updates=2)) + "\n" + '{"event":')
        raise ArithmeticError("interrupted")
    receipt = run(tmp_path, fake, effects=replace(fake.effects(), evaluate_candidate=broken))
    assert receipt["status"] == "incomplete"
    assert receipt["counters"]["recovery_sgd_attempted"] == 3
    assert receipt["counters"]["recovery_sgd_completed"] == 2 and receipt["unknown_sgd"] == []


def test_missing_durable_candidate_accounting_is_explicitly_unknown(tmp_path):
    fake = Fake()
    def broken(*args):
        raise RuntimeError("lost receipt")
    receipt = run(tmp_path, fake, effects=replace(fake.effects(), evaluate_candidate=broken))
    assert receipt["status"] == "incomplete" and receipt["unknown_sgd"][0]["recipe"] == "dense_recovery"
    assert receipt["failure"]["message"] == "lost receipt"


def test_exact_data_identity_is_rechecked_after_each_candidate(tmp_path):
    fake = Fake()
    def mutated(context, workflow):
        result = fake.evaluate(context, workflow)
        context.data.identity = "changed"
        return result
    receipt = run(tmp_path, fake, effects=replace(fake.effects(), evaluate_candidate=mutated))
    assert receipt["status"] == "incomplete" and len(receipt["candidates"]) == 1
    assert receipt["counters"]["recovery_sgd_completed"] == 200
    assert "identity changed" in receipt["failure"]["message"]


def test_full_config_roundtrip_and_no_series_overrides():
    cfg = config(recipes=FIXED_RECIPES[:3])
    value = json.loads(json.dumps(asdict(cfg)))
    assert pilot_config_from_payload(value) == cfg
    with pytest.raises(ValueError):
        pilot_config_from_payload({**value, "protocol": {**value["protocol"], "epochs": 81}})
    with pytest.raises(ValueError):
        pilot_config_from_payload({**value, "search_seconds": 100})
    with pytest.raises(ValueError):
        config(recipes=("Q_only", "dense_recovery"))
    with pytest.raises(ValueError):
        config(gpu_seconds=7201)


@pytest.mark.parametrize("sync_fails", [False, True])
def test_native_recovery_records_completion_only_after_cuda_synchronization(monkeypatch, sync_fails):
    """Fake the CUDA API completely; a submitted step may remain in flight."""
    events = []
    class Tensor:
        def __init__(self, shape, dtype): self.shape, self.dtype = shape, dtype
        def to(self, target): return self
    class Loss:
        def backward(self): events.append("backward")
    class Optimizer:
        def __init__(self, *args, **kwargs): pass
        def zero_grad(self, **kwargs): pass
        def step(self): events.append("step")
    class Model:
        def to(self, target): return self
        def train(self): return self
        def eval(self): return self
        def cpu(self): return self
        def parameters(self): return ()
        def __call__(self, value): return value
    def synchronize(target):
        events.append("synchronize")
        if sync_fails: raise RuntimeError("submitted CUDA step failed")
    fake_torch = SimpleNamespace(Tensor=Tensor, float32="float32", int64="int64",
        device=lambda name: SimpleNamespace(type="cuda", index=0),
        cuda=SimpleNamespace(synchronize=synchronize), manual_seed=lambda seed: None,
        random=SimpleNamespace(fork_rng=lambda **kw: nullcontext()),
        optim=SimpleNamespace(SGD=Optimizer),
        nn=SimpleNamespace(functional=SimpleNamespace(cross_entropy=lambda *args: Loss())),
        isfinite=lambda value: SimpleNamespace(all=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    cfg = RecoveryConfig(updates=1, batch_size=2)
    def execute():
        return recover_owned(Model(), recovery_tape((0, 1), model_seed=1, config=cfg),
            lambda *args: (Tensor((2, 3), "float32"), Tensor((2,), "int64")), config=cfg,
            device="cuda:0", deadline=10., clock=lambda: 0., synchronize_updates=True,
            progress=lambda event, count: events.append(event))
    if sync_fails:
        with pytest.raises(RuntimeError, match="submitted CUDA"):
            execute()
        assert events == ["sgd_attempted", "backward", "step", "synchronize"]
    else:
        assert execute()["updates"] == 1
        assert events == ["sgd_attempted", "backward", "step", "synchronize", "sgd_completed", "synchronize"]
