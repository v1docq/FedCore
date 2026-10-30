"""Real finite CPU measurement and independent KL/storage references."""
from copy import deepcopy
from dataclasses import replace
import json
import statistics

import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.allocation import exact_unique_cost
from fedcore.experiments import measurement, runner
from fedcore.experiments.protocol import ExperimentBundle, ExperimentProtocol, TensorSplit
from fedcore.experiments.svd_profile_measurement import (
    ProfileMeasurementError, RankProfileInfeasible, collect_aafm_kl_candidates,
    flar_latency_profile_key, measure_flar_rank_candidates, _validation_kl,
)
from fedcore.experiments.svd_rank_policies import select_isolated_ranks


class BufferedClassifier(nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.first = nn.Linear(4, 4, bias=False, dtype=dtype)
        self.head = nn.Linear(4, 3, dtype=dtype)
        self.register_buffer("offset", torch.linspace(-.2, .3, 4, dtype=dtype))
        generator = torch.Generator().manual_seed(851)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=dtype))

    def forward(self, x):
        # The test role contains this sentinel. Any test inference fails.
        if not torch.jit.is_tracing() and bool((x > 50).any()):
            raise AssertionError("test role must never be evaluated")
        return self.head(torch.tanh(self.first(x) + self.offset))


@pytest.fixture(autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


def problem():
    model = BufferedClassifier()
    generator = torch.Generator().manual_seed(24)
    splits = []
    for index, role in enumerate(("train", "validation", "calibration", "test")):
        x = (torch.randn((7, 4), generator=generator, dtype=torch.float64) + index/3
             if role != "test" else torch.full((7, 4), 99., dtype=torch.float64))
        y = torch.arange(7) % 3
        splits.append(TensorSplit(x, y, tuple(f"{role}-{i}" for i in range(7))))
    bundle = ExperimentBundle(*splits, "classification", model)
    protocol = ExperimentProtocol(batch_size=3, baseline_epochs=1, finetune_epochs=0,
                                  measurement_repeats=3, warmup=1, threads=1)
    runner.train_model(model, bundle.train, bundle.task, epochs=1, batch_size=3, learning_rate=.005)
    model.train()  # Helpers must preserve the caller's state and mode.
    return model, bundle, protocol


def flar(model, bundle, protocol, directory, **options):
    return measure_flar_rank_candidates(model, bundle, protocol, target_path="first", rank_grid=(1, 2),
        output_dir=directory, baseline_id="trained-model", checkpoint_id="checkpoint-sha256",
        source_version="source-snapshot-sha256", maximum_parameters=35, **options)


def test_real_flar_artifacts_raw_repeats_and_actual_whole_model_budget(tmp_path):
    model, bundle, protocol = problem()
    state_hash = runner.model_state_hash(model)
    result = flar(model, bundle, protocol, tmp_path, predicted_latency_ms={1: 123., 2: .000001})
    assert runner.model_state_hash(model) == state_hash and model.training
    assert result.profile.key.runtime == "torch.jit"
    assert result.profile.key.runtime_version == str(torch.__version__)
    assert result.profile.key.dtype == "float64" and result.profile.key.batch == 3
    assert result.evidence["fixed_stored_elements"] == 19  # head(15) + buffer(4)
    assert result.evidence["selected_whole_model_stored_elements"] <= 35
    for item, record in zip(result.profile.measurements, result.evidence["evaluations"]):
        measured = record["measurement"]
        assert len(item.measured_samples_ms) == len(measured["raw_end_to_end_ms"]) == 3
        assert all(value > 0 for value in item.measured_samples_ms)
        assert item.latency_ms == statistics.median(item.measured_samples_ms)
        assert item.latency_ms != item.predicted_latency_ms
        assert item.artifact_id == measured["artifact"]["sha256"]
        assert measurement.file_hash(measured["artifact"]["path"]) == item.artifact_id
        loaded = measurement.predict_loaded_artifact(measured["artifact"], bundle.validation.x[:3])
        assert loaded.shape == (3, 3)
        assert record["whole_model_stored_elements"] == 19 + 8*item.rank
        assert record["whole_model_tensor_bytes"] == 8*record["whole_model_stored_elements"]
    assert result.selection.manifest["latency_evidence"] == "final_artifact_measurement"
    assert result.evidence["test_evaluated"] is False
    json.dumps(result.to_dict(), allow_nan=False)


def test_real_flar_latency_limit_refutes_optimistic_predictions(tmp_path):
    model, bundle, protocol = problem()
    with pytest.raises(RankProfileInfeasible, match="latency/quality"):
        flar(model, bundle, protocol, tmp_path, predicted_latency_ms={1: 1e-30, 2: 1e-30},
             maximum_latency_ms=1e-20)
    assert len(list(tmp_path.rglob("*.pt"))) == 2


def test_flar_full_budget_and_stale_domain_refuse_before_execution(tmp_path, monkeypatch):
    model, bundle, protocol = problem()
    calls = []
    monkeypatch.setattr(runner, "apply_candidate", lambda *args: calls.append(args))
    key = flar_latency_profile_key(model, "first", bundle.validation.x[:3],
                                   source_version="source-snapshot-sha256")
    for field, value in (("source_version", "different-source"), ("runtime_version", "different-runtime"),
                         ("device", "cuda:0"), ("batch", 2), ("dtype", "float32"), ("graph_version", "stale")):
        with pytest.raises(ProfileMeasurementError, match="Stale"):
            flar(model, bundle, protocol, tmp_path, expected_key=replace(key, **{field: value}))
    with pytest.raises(RankProfileInfeasible) as error:
        measure_flar_rank_candidates(model, bundle, protocol, target_path="first", rank_grid=(1, 2),
            output_dir=tmp_path, baseline_id="trained", checkpoint_id="checkpoint", source_version="source",
            maximum_parameters=26)
    assert error.value.failure.minimum_cost == 27 and not calls


def test_latency_key_changes_for_hardware_threads_graph_and_full_input_shape(monkeypatch):
    model, bundle, _ = problem()
    x = bundle.validation.x[:3]
    key = flar_latency_profile_key(model, "first", x, source_version="source")
    assert flar_latency_profile_key(model, "first", x[:, None], source_version="source") != key
    assert flar_latency_profile_key(model, "first", x, source_version="source", threads=2) != key
    monkeypatch.setattr("fedcore.experiments.svd_profile_measurement.platform.processor", lambda: "another CPU")
    assert flar_latency_profile_key(model, "first", x, source_version="source") != key
    with torch.no_grad():
        model.first.weight[0, 0].add_(.1)
    assert flar_latency_profile_key(model, "first", x, source_version="source") != key


@pytest.mark.parametrize("unit", ["parameters", "bytes"])
def test_aafm_isolated_real_outputs_kl_reference_fixed_buffers_and_joint_cost(unit):
    model, bundle, protocol = problem()
    baseline_hash = runner.model_state_hash(model)
    result = collect_aafm_kl_candidates(model, bundle, protocol, target_paths=("first", "head"),
        rank_grid={"first": (1, 2, 4), "head": (1, 3)}, baseline_id="trained", checkpoint_id="checkpoint", unit=unit)
    assert runner.model_state_hash(model) == baseline_hash and model.training
    assert result.evidence["fixed_cost"] == 4*(8 if unit == "bytes" else 1)
    assert len(result.evaluations) == 5
    teacher = deepcopy(model).eval()
    with torch.inference_mode():
        log_p = teacher(bundle.validation.x).double().log_softmax(-1)
        for item, record in zip(result.evaluations, result.evidence["evaluations"]):
            candidate = runner.CandidateSpec.from_dict(record["candidate"])
            student, _ = runner.apply_candidate(model, bundle, protocol, candidate)
            log_q = student(bundle.validation.x).double().log_softmax(-1)
            expected = (log_p.exp()*(log_p-log_q)).sum(-1).mean().clamp_min(0).item()
            assert item.score == pytest.approx(expected, abs=1e-14)
            assert item.data_role == "validation" and item.statistics_role == "calibration"
            assert item.statistics_id and item.graph_version == baseline_hash
            assert record["kl_observations"] == 7
            assert record["whole_model_tensor_bytes"] == measurement.tensor_state_bytes(student)
            if item.rank == min(student.get_submodule(item.path).in_features, student.get_submodule(item.path).out_features):
                assert item.score < 1e-13
    # First lacks an original bias, but AFM's affine correction stores four.
    assert result.operators[0].bias_elements == 4
    budget = (4 + (8*2+4) + (7*1+3))*(8 if unit == "bytes" else 1)
    plan = select_isolated_ranks("afm", result.operators, result.rank_grid, result.evaluations,
                                 budget, fixed_storages=result.fixed_storages, unit=unit)
    assert plan.evidence["joint_evaluation_required"] and not plan.evidence["joint_quality_observed"]
    assert plan.allocation.total_cost <= budget
    # Verify the selected joint artifact, not an additive quality claim.
    joint, _ = runner.apply_candidate(model, bundle, protocol, plan.candidate)
    actual = sum(p.numel() for p in joint.parameters()) + sum(b.numel() for b in joint.buffers())
    assert plan.allocation.total_cost == actual*(8 if unit == "bytes" else 1)
    assert exact_unique_cost(result.fixed_storages, unit) == result.evidence["fixed_cost"]
    json.dumps(result.to_dict(), allow_nan=False)


def test_aafm_collects_from_same_baseline_and_separate_calibration(monkeypatch):
    model, bundle, protocol = problem()
    original = runner.apply_candidate
    hashes, roles, epochs = [], [], []
    def capture(source, data, config, candidate):
        hashes.append(runner.model_state_hash(source))
        roles.append(data.calibration.ids)
        epochs.append(candidate.parameters["finetune_epochs"])
        return original(source, data, config, candidate)
    monkeypatch.setattr(runner, "apply_candidate", capture)
    result = collect_aafm_kl_candidates(model, bundle, protocol, target_paths=("first",),
        rank_grid={"first": iter((1, 2))}, baseline_id="trained", checkpoint_id="checkpoint")
    assert len(set(hashes)) == 1 and epochs == [0, 0]
    assert roles == [bundle.calibration.ids]*2
    assert len({item.statistics_id for item in result.evaluations}) == 1
    assert set(result.evidence["calibration"]["ids"]).isdisjoint(result.evidence["validation"]["ids"])


def test_causal_kl_shift_padding_and_fp64_reference():
    teacher, student = nn.Linear(3, 4, dtype=torch.float64), nn.Linear(3, 4, dtype=torch.float64)
    generator = torch.Generator().manual_seed(12)
    x = torch.randn((3, 5, 3), generator=generator, dtype=torch.float64)
    labels = torch.tensor([[0, 1, 2, -100, -100], [1, 2, 3, 0, -100], [3, 1, 2, 1, 0]])
    split = TensorSplit(x, labels, ("v1", "v2", "v3"))
    score, count = _validation_kl(teacher, student, split, "language_model", 2)
    mask = labels[:, 1:] != -100
    lp = teacher(x)[:, :-1][mask].log_softmax(-1)
    lq = student(x)[:, :-1][mask].log_softmax(-1)
    expected = (lp.exp()*(lp-lq)).sum(-1).mean().item()
    assert count == int(mask.sum()) == 9
    assert score == pytest.approx(expected, abs=1e-14)


def test_aafm_rejects_regression_tied_targets_and_partial_storage_views():
    model, bundle, protocol = problem()
    regression = replace(bundle, task="regression")
    with pytest.raises(ProfileMeasurementError, match="logits"):
        collect_aafm_kl_candidates(model, regression, protocol, target_paths=("first",),
            rank_grid={"first": (1,)}, baseline_id="trained", checkpoint_id="checkpoint")
    tied = deepcopy(model)
    tied.alias = nn.Linear(4, 4, bias=False, dtype=torch.float64)
    tied.alias.weight = tied.first.weight
    with pytest.raises(ProfileMeasurementError, match="share tensor storage"):
        collect_aafm_kl_candidates(tied, bundle, protocol, target_paths=("first",),
            rank_grid={"first": (1,)}, baseline_id="trained", checkpoint_id="checkpoint")
    view_model = deepcopy(model)
    view_model.register_buffer("partial", torch.arange(8, dtype=torch.float64)[1:5])
    with pytest.raises(ProfileMeasurementError, match="full dense"):
        collect_aafm_kl_candidates(view_model, bundle, protocol, target_paths=("first",),
            rank_grid={"first": (1,)}, baseline_id="trained", checkpoint_id="checkpoint")


def test_flar_failed_export_cannot_fall_back_to_latency_prediction(tmp_path, monkeypatch):
    model, bundle, protocol = problem()
    monkeypatch.setattr(measurement, "measure_artifact", lambda *args, **kwargs: {"status": "failed", "reason": "broken export"})
    with pytest.raises(ProfileMeasurementError, match="artifact measurement failed"):
        flar(model, bundle, protocol, tmp_path, predicted_latency_ms={1: .1, 2: .2})
