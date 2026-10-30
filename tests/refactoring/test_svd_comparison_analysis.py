"""Independent arithmetic laws for the published comparison analysis."""
import pytest
import json

from scripts.analyze_svd_comparison import analytical_macs, describe, paired_delta, analyze


def test_analytical_macs_counts_spatial_applications_and_untouched_operators():
    assert analytical_macs("mlp") == 12160
    assert analytical_macs("mlp", {"2": 8}) == 3840 + 8 * 192 + 128
    assert analytical_macs("cnn") == 64 * (16 * 9 + 16 * 144) + 64 * 1024 + 10 * 64
    assert analytical_macs("cnn", {"classifier.0": 8}) == 166016
    assert analytical_macs("cnn", {"features.0": 1, "features.2": 4, "classifier.0": 8}) == 51904


def test_equal_ranks_have_identical_declared_arithmetic_independent_of_algorithm():
    for rank in (8, 16, 32):
        expected = 156672 + 1088 * rank + 640
        assert analytical_macs("cnn", {"classifier.0": rank}) == expected
    with pytest.raises(ValueError):
        analytical_macs("cnn", {"features.0": 10})
    with pytest.raises(ValueError):
        analytical_macs("cnn", {"missing": 1})


def test_pairing_requires_same_architecture_and_seed():
    baseline = {"architecture": "cnn", "seed": 41, "accuracy": .9}
    candidate = {"architecture": "cnn", "seed": 41, "accuracy": .85}
    assert paired_delta(candidate, baseline) == pytest.approx(-.05)
    for field, replacement in (("seed", 42), ("architecture", "mlp")):
        with pytest.raises(ValueError):
            paired_delta(dict(candidate, **{field: replacement}), baseline)


def test_descriptive_spread_retains_missingness_and_does_not_label_confidence():
    assert describe([]) == {"n": 0, "mean": None, "sample_sd": None, "min": None, "max": None}
    assert describe([2])["sample_sd"] is None
    summary = describe([-1, 0, 1])
    assert summary == {"n": 3, "mean": 0, "sample_sd": 1, "min": -1, "max": 1}
    with pytest.raises(ValueError):
        describe([float("nan")])


def test_worker_failure_rows_remain_inspectable_without_quality_or_baseline_hash(tmp_path):
    source = tmp_path / "input"
    run_dir = source / "mlp-seed-41"
    run_dir.mkdir(parents=True)
    (source / "protocol.json").write_text(json.dumps({"study": "failure fixture", "timing": {"repeats": 200}}), encoding="utf-8")
    run = {"architecture": "mlp", "seed": 41, "baseline_hash": None, "gate": {"passed": False},
           "baseline": {"case_id": "baseline", "method": "baseline", "cohort": "baseline", "status": "worker_failed"},
           "records": [{"case_id": "not-completed", "method": "svd", "cohort": "common_linear", "level": 0,
                        "status": "not_completed_worker_failure", "reason": "resource guard stopped worker"}]}
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    result = analyze(source, tmp_path / "output")
    assert result["counts"] == {"baselines": 1, "candidates": 1, "succeeded_candidates": 0, "timing_sessions": 0}
    assert result["paired_summaries"][0]["failed_seeds"] == [41]
    assert result["paired_summaries"][0]["accuracy_delta_pp"]["mean"] is None
    assert result["rows"][1]["reason"] == "resource guard stopped worker"
