"""Real public runner lifecycle, leakage rejection and safe replay checks."""
from dataclasses import replace
import json

import pytest
import torch
from torch import nn

from fedcore.experiments import (CandidateSpec, ExperimentBundle, ExperimentProtocol,
                                 ExperimentRunner, ProtocolError, TensorSplit,
                                 load_run, measure_artifact, split_indices)
from fedcore.experiments.runner import apply_candidate, model_state_hash, task_loss


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model_factory():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        return nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))


def bundle(*, test_flip=False):
    splits = {}
    for role in ("train", "validation", "calibration", "test"):
        y = torch.arange(12) % 2
        x = nn.functional.one_hot(y, 2).float()
        splits[role] = TensorSplit(x, 1 - y if test_flip and role == "test" else y,
                                  tuple(f"{role}-{index}" for index in range(len(x))))
    return ExperimentBundle(**splits, task="classification", original_model=model_factory(),
                            metadata={"preprocessing_fit_ids": list(splits["train"].ids),
                                      "scenario": "independent_onehot_fixture", "api_key": "must_be_redacted"})


def protocol(**overrides):
    return ExperimentProtocol(baseline_epochs=1, finetune_epochs=1, batch_size=12,
                              learning_rate=.1, measurement_repeats=2, warmup=1,
                              quality_tolerance=1.0, **overrides)


def test_independent_units_partition_determinism_and_conservation():
    ids = tuple(f"sample-{index}" for index in range(20))
    units = tuple(f"client-{index // 2}" for index in range(20))
    left = split_indices(ids, seed=9, unit_ids=units)
    assert left == split_indices(ids, seed=9, unit_ids=units)
    assert sorted(index for part in left.values() for index in part) == list(range(20))
    used = [set(units[index] for index in part) for part in left.values()]
    assert all(not a & b for index, a in enumerate(used) for b in used[index + 1:])


def test_role_leakage_duplicate_clients_and_inclusive_windows_rejected():
    source = bundle()
    with pytest.raises(ProtocolError, match="Sample IDs overlap"):
        replace(source, test=replace(source.test, ids=source.train.ids))
    splits = {role: replace(getattr(source, role), unit_ids=tuple("same-client" for _ in getattr(source, role).ids))
              for role in ("train", "validation", "calibration", "test")}
    with pytest.raises(ProtocolError, match="Independent units overlap"):
        replace(source, **splits)
    splits = {role: replace(getattr(source, role), intervals=tuple((index * 100, index * 100 + 5) for index in range(12)))
              for role in ("train", "validation", "calibration", "test")}
    with pytest.raises(ProtocolError, match="Raw time windows overlap"):
        replace(source, **splits)
    with pytest.raises(ProtocolError, match="Preprocessing fit IDs"):
        replace(source, metadata={"preprocessing_fit_ids": list(source.test.ids)})


def test_ids_and_parameters_are_frozen_and_tensor_mutation_is_detected():
    source = bundle()
    candidate = CandidateSpec("svd", {"rank_ratio": .5, "nested": {"value": 1}})
    with pytest.raises(TypeError):
        candidate.parameters["nested"]["value"] = 2
    source.train.x[0, 0] += 1
    with pytest.raises(ProtocolError, match="modified in place"):
        source.manifest()


def test_reordered_labels_reduce_quality_and_causal_loss_shifts_tokens():
    from fedcore.experiments.runner import quality_metrics
    source = bundle()
    identity = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        identity.weight.copy_(torch.eye(2))
    quality = quality_metrics(identity, source.validation, "classification")
    flipped = replace(source.validation, y=1 - source.validation.y)
    assert quality["value"] == quality["macro_f1"] == 1
    assert quality_metrics(identity, flipped, "classification")["value"] == 0
    logits = torch.tensor([[[10., -10.], [-10., 10.], [10., -10.]]])
    labels = torch.tensor([[1, 0, 1]])
    assert float(task_loss(logits, labels, "language_model")) < 1e-5
    with pytest.raises(ProtocolError, match="nonpadding"):
        task_loss(logits, torch.full_like(labels, -100), "language_model")


def test_public_runner_trains_clones_records_failure_and_safe_replay(tmp_path):
    source = bundle()
    before = model_state_hash(source.original_model)
    candidates = (CandidateSpec("train", {"epochs": 2}), CandidateSpec("svd", {"rank": 1, "finetune_epochs": 0}),
                  CandidateSpec("does_not_exist"))
    manifest = ExperimentRunner(source, protocol(), tmp_path / "first").run(candidates)
    assert manifest["status"] == "succeeded"
    assert model_state_hash(source.original_model) == before
    assert manifest["baseline_training"]["state_sha256"] != before
    assert len({record["source_model_sha256"] for record in manifest["candidates"]}) == 1
    assert manifest["candidates"][-1]["status"] == "unsupported"
    assert manifest["candidates"][-1]["wall_seconds"] > 0
    assert manifest["test_gate"]["selection_role"] == "validation"
    assert manifest["data"]["metadata"]["api_key"] == "[redacted]"
    record = manifest["candidates"][0]
    predictions = torch.load(record["validation_predictions"]["path"], weights_only=True)
    assert predictions["ids"] == list(source.validation.ids)
    torch.testing.assert_close(predictions["targets"], source.validation.y)
    restored = load_run(tmp_path / "first")
    assert restored["selection"] == manifest["selection"]
    replay = ExperimentRunner.from_manifest(tmp_path / "first", model_factory, tmp_path / "replay")
    replayed = replay.run(candidates)
    assert replayed["baseline_training"]["state_sha256"] == manifest["baseline_training"]["state_sha256"]
    assert [row.get("validation") for row in replayed["candidates"]] == [row.get("validation") for row in manifest["candidates"]]
    (tmp_path / "first" / "data.pt").write_bytes(b"modified")
    with pytest.raises(ProtocolError, match="hash mismatch"):
        load_run(tmp_path / "first")


def test_changing_test_does_not_change_selected_configuration(tmp_path):
    candidates = (CandidateSpec("train", {"epochs": 5}), CandidateSpec("pruning", {"amount": .25, "finetune_epochs": 1}))
    first = ExperimentRunner(bundle(), protocol(), tmp_path / "one").run(candidates)
    second = ExperimentRunner(bundle(test_flip=True), protocol(), tmp_path / "two").run(candidates)
    assert first["selection"]["archive_ids"] == second["selection"]["archive_ids"]
    assert [row.get("validation") for row in first["candidates"]] == [row.get("validation") for row in second["candidates"]]
    assert first["candidates"][0]["test"]["value"] != second["candidates"][0]["test"]["value"]


def test_cache_invalidated_by_parameters_and_data(tmp_path):
    cache = tmp_path / "shared-cache"
    candidates = (CandidateSpec("train", {"epochs": 1}),)
    first = ExperimentRunner(bundle(), protocol(), tmp_path / "one", cache_dir=cache, use_cache=True).run(candidates)
    second = ExperimentRunner(bundle(), protocol(), tmp_path / "two", cache_dir=cache, use_cache=True).run(candidates)
    assert all(row["cache_hit"] for row in second["candidates"])
    third = ExperimentRunner(bundle(), protocol(), tmp_path / "three", cache_dir=cache, use_cache=True).run((CandidateSpec("train", {"epochs": 2}),))
    assert third["candidates"][-1]["cache_key"] != first["candidates"][-1]["cache_key"]
    source = bundle()
    changed = replace(source, validation=replace(source.validation, x=source.validation.x + .01))
    fourth = ExperimentRunner(changed, protocol(), tmp_path / "four", cache_dir=cache, use_cache=True).run(candidates)
    assert not any(row["cache_hit"] for row in fourth["candidates"])


def test_real_artifact_units_missing_device_and_onnx(tmp_path):
    source = bundle()
    measured = measure_artifact(source.original_model, source.validation.x[:4], tmp_path / "тест" / "model",
                               repeats=3, warmup=1)
    assert measured["status"] == "succeeded"
    assert measured["profile"]["units"]["latency_p50_ms"] == "ms/batch"
    assert len(measured["raw_inference_ms"]) == 3
    assert measured["metrics"]["file_bytes"] > 0
    assert measured["metrics"]["cuda_peak_allocated_bytes"] is None
    unsupported = measure_artifact(source.original_model, source.validation.x[:4], tmp_path / "invalid", format="onnx", device="cuda")
    assert unsupported["status"] == "unsupported"
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    onnx = measure_artifact(source.original_model, source.validation.x[:4], tmp_path / "onnx-model", format="onnx", repeats=2, warmup=0)
    assert onnx["status"] == "succeeded", onnx
    assert onnx["artifact"]["format"] == "onnx"


def test_real_ptq_qat_and_structural_pruning_leave_source_unchanged():
    source = bundle()
    configuration = protocol()
    before = model_state_hash(source.original_model)
    for candidate in (CandidateSpec("ptq", {"mode": "static"}), CandidateSpec("qat", {"epochs": 1}),
                      CandidateSpec("structural_pruning", {"pruning_ratio": .4, "finetune_epochs": 0})):
        model, evidence = apply_candidate(source.original_model, source, configuration, candidate)
        assert model_state_hash(source.original_model) == before
        assert model_state_hash(model) != before
        if candidate.method == "qat":
            assert evidence["training_steps"] == 1
        if candidate.method == "structural_pruning":
            assert evidence["parameters_after"] < evidence["parameters_before"]


def test_protocol_parse_roundtrip_and_unknown_field():
    configuration = protocol()
    assert ExperimentProtocol.from_dict(configuration.to_dict()) == configuration
    specification = CandidateSpec("chain", chain=(CandidateSpec("pruning", {"amount": .2}), CandidateSpec("svd", {"rank": 1})))
    assert CandidateSpec.from_dict(specification.to_dict()).candidate_id == specification.candidate_id
    with pytest.raises(ProtocolError, match="Unknown"):
        ExperimentProtocol.from_dict({"unexpected": 2})


def test_real_runner_fedcore_public_path(tmp_path):
    candidate = CandidateSpec("fedcore", {"operation": "training", "operation_parameters": {"epochs": 1}})
    result = ExperimentRunner(bundle(), protocol(), tmp_path / "fedcore").run((candidate,))
    record = result["candidates"][-1]
    assert record["status"] == "succeeded", record
    assert record["operation"]["implementation"] == "ConfigFactory + FedCore.fit_no_evo"


def test_interruption_retains_explicit_status_and_closed_test_gate(tmp_path, monkeypatch):
    import fedcore.experiments.runner as runtime
    original = runtime.apply_candidate
    def interrupt(model, source, configuration, candidate):
        if candidate.method == "train":
            raise KeyboardInterrupt()
        return original(model, source, configuration, candidate)
    monkeypatch.setattr(runtime, "apply_candidate", interrupt)
    runner = ExperimentRunner(bundle(), protocol(), tmp_path / "interrupted")
    with pytest.raises(KeyboardInterrupt):
        runner.run((CandidateSpec("train"),))
    saved = json.loads((tmp_path / "interrupted" / "manifest.json").read_text(encoding="utf-8"))
    assert saved["status"] == saved["candidates"][-1]["status"] == "interrupted"
    assert saved["candidates"][-1]["wall_seconds"] > 0
    assert saved["test_gate"]["status"] == "closed"
