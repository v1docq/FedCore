"""Budget effects, profile isolation, sealed roles and validation-only fitness."""
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import torch

from fedcore.experiments.compression_search_evaluation import (
    EvaluationProfile, EvaluationSource, FullEvaluator, PackageFatal,
    compile_checked, metrics, model_identity, predict)
from fedcore.experiments.compression_search_digits_evaluation import (
    DIGITS_PROFILE, DigitsDeadline, DigitsFullEvaluator, DigitsLedger,
    fresh_model, prepare_space, rng_signature, source_evaluator)
from fedcore.experiments.compression_search_digits_protocol import (
    DENSE_TENSOR_BYTES, INITIAL_GENOTYPES, TUNED_CATALOG, canonical_solution_id, expected_tensor_bytes)
from fedcore.experiments.protocol import TensorSplit


def job(tmp_path, **changes):
    now = time.perf_counter()
    return {"job": "synthetic-test", "kind": "search", "split_seed": 19901,
        "output": str(tmp_path), "directory": str(tmp_path), "protocol_sha256": "p", "source_sha256": "s",
        "package_deadline": now+100, "compute_admission_deadline": now+100,
        "admission_deadline": now+100, **changes}


def test_old_profile_defaults_remain_unchanged_and_digits_is_explicit():
    assert EvaluationProfile() == EvaluationProfile(49_416, 46_945, 2)
    assert DIGITS_PROFILE == EvaluationProfile(68_904, 65_458, 10)
    with pytest.raises(ValueError):
        EvaluationProfile(68904, 65458, 1)


def test_deadline_prevents_charge_and_strongest_absolute_deadline_wins(tmp_path):
    ledger = DigitsLedger(tmp_path / "ledger.jsonl", job(tmp_path, compute_admission_deadline=0))
    with pytest.raises(DigitsDeadline):
        ledger.charge("full_validation", "x")
    assert not ledger.path.exists()
    allowed = DigitsLedger(tmp_path / "ledger.jsonl", job(tmp_path))
    assert allowed.charge("full_validation", "x") == 1
    # A fresh worker resumes the append-only count, with no refund of a failure.
    resumed = DigitsLedger(allowed.path, job(tmp_path))
    assert resumed.charge("full_validation", "y") == 2
    assert [json.loads(line)["attempt"] for line in allowed.path.read_text().splitlines()] == [1, 2]


def test_exhausted_or_corrupt_ledger_does_not_start_another_action(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(''.join(json.dumps({"attempt": i})+'\n' for i in range(1, 10301)))
    ledger = DigitsLedger(path, job(tmp_path))
    with pytest.raises(DigitsDeadline):
        ledger.charge("full_validation", "x")
    assert len(path.read_text().splitlines()) == 10300
    path.write_text('{"attempt": 2}\n')
    with pytest.raises(PackageFatal, match="contiguous"):
        DigitsLedger(path, job(tmp_path))


def synthetic_source():
    model = fresh_model()
    features = torch.arange(8*64, dtype=torch.float32).reshape(8, 64) / 512
    labels = predict(model, features).argmax(1)
    calibration = TensorSplit(features[:4], labels[:4], tuple(f"cal:{i}" for i in range(4)))
    validation = TensorSplit(features[4:], labels[4:], tuple(f"val:{i}" for i in range(4)))
    logits = predict(model, validation.x)
    return EvaluationSource(19901, model, calibration, validation, {}, metrics(logits, validation.y), logits)


def test_full_candidate_pipeline_uses_ten_class_normalization_and_strict_inventory(tmp_path):
    source = synthetic_source()
    before, rng_before = model_identity(source.model), rng_signature()
    space = prepare_space(source, TUNED_CATALOG)
    evaluator = DigitsFullEvaluator(source, space, DigitsLedger(tmp_path / "ledger.jsonl", job(tmp_path)),
        tmp_path / "events.jsonl", context_sha256="validation-calibration-only", job_id="synthetic-test")
    evaluator.catalog_ids = frozenset(canonical_solution_id(value) for value in TUNED_CATALOG.genotypes)
    genotype = INITIAL_GENOTYPES[0]
    result = evaluator(compile_checked(genotype, space))
    row = evaluator.rows[0]
    assert row["status"] == "finite"
    assert result.quality == row["metrics"]["cross_entropy"] / math.log(10)
    assert result.cost == row["tensor_bytes"] / 68_904
    assert row["tensor_bytes"] == expected_tensor_bytes(genotype)
    assert row["inventory_verified_after_reload"] is True
    assert model_identity(source.model) == before and rng_signature() == rng_before
    # Full-evaluation context owns exactly calibration and validation roles.
    assert not hasattr(source, "test") and not hasattr(source, "train")
    forwards = [json.loads(line) for line in (tmp_path / "forward-calls.jsonl").read_text().splitlines()]
    assert sum(row["event"] == "forward_started" for row in forwards) == sum(row["event"] == "forward_completed" for row in forwards) > 0


def test_failed_execution_remains_charged_before_the_execution_effect(tmp_path, monkeypatch):
    import fedcore.experiments.compression_search_evaluation as shared
    from fedcore.algorithm.low_rank.grammar_execution import NumericalFailure
    source = synthetic_source()
    space = prepare_space(source, TUNED_CATALOG)
    path = tmp_path / "ledger.jsonl"
    def failed(*args):
        assert len(path.read_text().splitlines()) == 1
        return NumericalFailure("SyntheticFailure", "deliberate failure")
    monkeypatch.setattr(shared, "execute_workflow", failed)
    evaluator = DigitsFullEvaluator(source, space, DigitsLedger(path, job(tmp_path)), tmp_path / "events.jsonl",
        context_sha256="c", job_id="synthetic-test")
    evaluator.catalog_ids = frozenset(canonical_solution_id(value) for value in TUNED_CATALOG.genotypes)
    assert isinstance(evaluator(compile_checked(INITIAL_GENOTYPES[0], space)), NumericalFailure)
    assert len(path.read_text().splitlines()) == 1
    assert evaluator.rows[0]["status"] == "NumericalFailure"
    assert evaluator.rows[0]["completed_at"] >= evaluator.rows[0]["started_at"]
