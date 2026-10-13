"""Independent cost/data/matching oracles for the frozen local comparison."""
from copy import deepcopy

import numpy as np
import pytest
import sys
import time
import torch
from torch import nn

from fedcore.algorithm.low_rank.svd_tools import decompose_module
from fedcore.experiments import runner, scenarios
from fedcore.experiments.protocol import ExperimentProtocol, ProtocolError
from fedcore.experiments.svd_comparison import (
    build_bundle, candidate_spec, deployed_ranks, direct_error, exact_storage,
    expected_factor_storage, frozen_cases, isolated_errors, _reference_observations,
    _export_record,
)


@pytest.fixture(autouse=True)
def limited_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("architecture,dense,costs", [
    ("mlp", 49416, {"common_linear": [22792, 28936, 35080]}),
    ("cnn", 274920, {"common_linear": [47592, 82408, 152040],
                     "joint_channel_conv_linear": [40460, 77936, 150228]}),
])
def test_complete_storage_matches_independent_hand_count_and_all_methods(architecture, dense, costs):
    bundle = build_bundle(architecture, 41)
    assert exact_storage(bundle.original_model)["tensor_bytes"] == dense
    cases = frozen_cases(architecture)
    assert len(cases) == (12 if architecture == "mlp" else 18)
    assert len({case["case_id"] for case in cases}) == len(cases)
    for case in cases:
        expected = costs[case["cohort"]][case["level"]]
        budget = expected_factor_storage(bundle.original_model, case["ranks"])
        assert budget["whole_model_tensor_byte_budget"] == expected
    # The equality is a structural identity, not two independent experiments.
    for level in range(3):
        matched = [case for case in cases if case["cohort"] == "common_linear" and case["level"] == level]
        assert {case["method"] for case in matched} == {"svd", "weighted_svd", "asvd", "afm"}
        assert len({expected_factor_storage(bundle.original_model, case["ranks"])["whole_model_tensor_byte_budget"]
                    for case in matched}) == 1


def test_full_storage_includes_shared_storage_once_buffers_bias_and_untouched_weights():
    model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    model.register_buffer("buffer", torch.ones(5, dtype=torch.float64))
    model.register_parameter("shared", model[0].weight)
    # Dense parameters: 12+3+6+2 float32; five float64 buffer values.
    assert exact_storage(model)["tensor_bytes"] == 23 * 4 + 5 * 8
    simple = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    simple.register_buffer("buffer", torch.ones(5, dtype=torch.float64))
    assert expected_factor_storage(simple, {"0": 1})["whole_model_tensor_byte_budget"] == (7+3+6+2)*4+40


def test_train_preprocessing_and_all_four_roles_are_separate():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    bundle = build_bundle("mlp", 41)
    roles = (bundle.train, bundle.validation, bundle.calibration, bundle.test)
    assert sum(len(split.ids) for split in roles) == len(data.target)
    for i, split in enumerate(roles):
        for other in roles[i+1:]:
            assert not set(split.ids) & set(other.ids)
    train_ids = [int(item.split(":")[-1]) for item in bundle.train.ids]
    assert np.allclose(bundle.metadata["preprocessing"]["mean"], data.data[train_ids].mean(0))
    assert set(bundle.metadata["preprocessing_fit_ids"]) == set(bundle.train.ids)
    # Alter every held-out feature radically; fitting parameters stay identical.
    altered = data.data.copy()
    heldout = sorted(set(range(len(altered))) - set(train_ids))
    altered[heldout] += 1000
    modified = scenarios.build_tabular(41, features=altered, labels=data.target)
    assert modified.metadata["preprocessing"] == bundle.metadata["preprocessing"]
    torch.testing.assert_close(modified.train.x, bundle.train.x, rtol=0, atol=0)


def test_svd_explicit_targets_preserve_every_unselected_tensor_and_source():
    bundle = build_bundle("mlp", 41)
    source = bundle.original_model
    before = deepcopy(source.state_dict())
    case = next(case for case in frozen_cases("mlp") if case["method"] == "svd")
    model, evidence = runner.apply_candidate(source, bundle, ExperimentProtocol(finetune_epochs=0), candidate_spec(case))
    assert deployed_ranks(model, {"2": 8})["2"]["numerical_rank"] == 8
    assert exact_storage(model)["tensor_bytes"] == 22792
    assert evidence["ranks"][0]["path"] == "2"
    assert evidence["ranks"][0]["strategy"] == "explicit_rank"
    for name, value in before.items():
        torch.testing.assert_close(source.state_dict()[name], value, rtol=0, atol=0)
        if not name.startswith("2."):
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["svd", "weighted_svd", "asvd", "afm"])
def test_all_four_existing_public_paths_realize_same_actual_rank_and_budget(method):
    bundle = build_bundle("mlp", 41)
    baseline_hash = runner.model_state_hash(bundle.original_model)
    case = next(case for case in frozen_cases("mlp") if case["method"] == method and case["level"] == 0)
    model, evidence = runner.apply_candidate(bundle.original_model, bundle, ExperimentProtocol(finetune_epochs=0),
                                             candidate_spec(case))
    assert exact_storage(model)["tensor_bytes"] == 22792
    assert deployed_ranks(model, {"2": 8})["2"]["numerical_rank"] == 8
    assert runner.model_state_hash(bundle.original_model) == baseline_hash
    assert evidence["training_steps"] == 0
    assert torch.isfinite(model(bundle.validation.x)).all()
    if method == "weighted_svd":
        numerics = evidence["weighted_transform"]["layers"][0]["numerics"][0]
        assert numerics["moment_rank"] < 128  # only57 separate calibration rows
        assert numerics["ridge"] == 0 and numerics["rcond"] is None
        assert numerics["effective_cutoff"] > 0 and numerics["discarded_metric_mass"] >= 0
        assert numerics["condition_number"] >= 1


def test_exact_multi_operator_ranks_and_legacy_default_are_preserved():
    bundle = build_bundle("cnn", 42)
    case = next(case for case in frozen_cases("cnn") if case["cohort"] == "joint_channel_conv_linear" and case["method"] == "svd")
    model, _ = runner.apply_candidate(bundle.original_model, bundle, ExperimentProtocol(finetune_epochs=0), candidate_spec(case))
    assert {path: item["stored_rank"] for path, item in deployed_ranks(model, case["ranks"]).items()} == case["ranks"]
    assert exact_storage(model)["tensor_bytes"] == 40460
    small = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2)).eval()
    x = torch.randn(5, 4)
    legacy = decompose_module(deepcopy(small), compose_mode="two_layers")
    assert legacy[0].U is not None and legacy[1].U is not None
    torch.testing.assert_close(legacy(x), small(x), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("targets", [[], ["missing"], ["1"], ["0", "0"], "0"])
def test_invalid_selection_refuses_without_mutating_source(targets):
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    before = deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="target_paths"):
        decompose_module(model, target_paths=targets)
    for key, value in before.items():
        assert torch.equal(model.state_dict()[key], value)


@pytest.mark.parametrize("ranks", [{"0": 1, "2": 1}, {"0": 0}, {"0": True}, {"0": 99}])
def test_invalid_rank_map_cannot_silently_fall_back(ranks):
    bundle = build_bundle("mlp", 41)
    with pytest.raises(ProtocolError):
        runner.apply_candidate(bundle.original_model, bundle, ExperimentProtocol(finetune_epochs=0),
            runner.CandidateSpec("svd", {"target_paths": ["0"], "ranks": ranks}))


def test_direct_common_output_error_has_independent_reference_and_no_hidden_refit():
    expected = torch.tensor([[1., 2.], [3., 4.]])
    actual = expected + torch.tensor([[1., -1.], [2., -2.]])
    result = direct_error(expected, actual)
    assert result["mse"] == 2.5 and result["relative_mse"] == pytest.approx(1/3)
    bundle = build_bundle("mlp", 41)
    baseline = bundle.original_model.eval()
    observations = _reference_observations(baseline, bundle.validation, ("2",))
    altered = deepcopy(baseline)
    with torch.no_grad():
        altered[2].bias.add_(2.)
    isolated = isolated_errors(altered, observations)
    assert isolated["2"]["mse"] == pytest.approx(4., rel=1e-6)
    assert not baseline[2]._forward_hooks


def test_checkpoint_export_reloaded_validation_and_test_artifact_and_metadata(tmp_path):
    bundle = build_bundle("mlp", 41)
    case = next(case for case in frozen_cases("mlp") if case["method"] == "afm" and case["level"] == 0)
    model, _ = runner.apply_candidate(bundle.original_model, bundle, ExperimentProtocol(finetune_epochs=0), candidate_spec(case))
    record = {"storage": exact_storage(model)}
    _export_record(model.eval(), bundle, tmp_path, record)
    assert record["checkpoint"]["reload_correct"]
    assert record["export_reload"]["all_validation_test_parity"]
    assert record["storage"]["artifact_file_bytes"] > record["storage"]["tensor_bytes"]
    assert record["artifact_sidecar"]["file_bytes"] > 0
    assert record["export_reload"]["roles"]["test"]["quality"]["samples"] == len(bundle.test.ids)


def test_actual_recursive_python_child_memory_cap_terminates_interpreter_tree(tmp_path, monkeypatch):
    from fedcore.experiments import svd_comparison as study
    monkeypatch.setattr(study, "RSS_LIMIT", 80 * 1024**2)
    monkeypatch.setattr(study, "OS_RESERVE", 0)
    allocation = "import time; payload=bytearray(128*1024**2); time.sleep(20)"
    parent = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c',%r]); p.wait()" % allocation
    result = study.monitored_process([sys.executable, "-c", parent], tmp_path, time.monotonic() + 30)
    assert result["violation"] and "RSS cap" in result["violation"]
    assert result["returncode"] != 0
    assert result["sampled_maximum_tree_sum_rss_bytes"] > 80 * 1024**2
    assert any(len(sample["owned_processes"]) >= 2 for sample in result["samples"])
    assert result["sampled_maximum_tree_sum_rss_bytes"] >= result["sampled_maximum_individual_rss_bytes"]


def test_prior_instrumentation_attempt_wall_is_charged_in_frozen_new_protocol(tmp_path, monkeypatch):
    import json
    from fedcore.experiments import svd_comparison as study
    prior = tmp_path / "prior"
    prior.mkdir()
    (prior / "execution.json").write_text(json.dumps({"architecture_workers": [
        {"resources": {"elapsed_seconds": 101.}}]}), encoding="utf-8")
    (prior / "timing-execution.json").write_text(json.dumps({"workers": [
        {"resources": {"elapsed_seconds": 202.}}]}), encoding="utf-8")
    monkeypatch.setattr(study.runner, "environment_manifest", lambda: {})
    monkeypatch.setattr(study, "source_identity", lambda: {})
    plan = study.predeclare_plan(tmp_path / "new", prior_runs=[prior])
    assert plan["resources"]["prior_charged_worker_wall_seconds"] == 303
    assert plan["resources"]["remaining_at_release_seconds"] == study.COMPUTE_SECONDS - 303
    assert plan["training"]["recovery_updates"] == 0 and plan["seeds"] == study.SEEDS
