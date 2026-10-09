"""Independent reference points, measured GOLEM search and report arithmetic."""
from dataclasses import replace
import copy
import json

import pytest
import torch

from fedcore.experiments import CandidateSpec, ExperimentRunner, SearchConfig, hypervolume_2d
from fedcore.experiments.protocol import ProtocolError
from fedcore.experiments.reporting import audit_historical_percent, relative_change, table_rows
from fedcore.experiments.search import archive_summary, compare_search_runs
from tests.petra.test_protocol_runner import bundle, protocol


def record(identifier, loss, cost, *, status="succeeded", method="svd"):
    return {"candidate_id": identifier, "status": status,
            "configuration": {"method": method, "parameters": {}, "chain": []},
            "validation": {"metric": "accuracy", "value": 1 - loss, "loss": loss, "direction": "maximize"},
            "measurement": {"status": status, "artifact": {"sha256": "a"},
                            "profile": {"device": "cpu", "repeats": 3},
                            "raw_inference_ms": [1., 2., 3.],
                            "metrics": {"file_bytes": cost, "latency_p50_ms": 2., "throughput": 3.}},
            "wall_seconds": 1., "cache_hit": False}


def test_known_archive_hypervolume_and_contributions():
    # Union: [1,4]x[3,4] plus [2,4]x[2,3] plus [3,4]x[1,2] = 6.
    points = [(1., 3.), (2., 2.), (3., 1.), (3., 3.)]
    assert hypervolume_2d(points, (4., 4.)) == 6
    assert hypervolume_2d(list(reversed(points)) + [points[0]], (4., 4.)) == 6
    assert hypervolume_2d([(5., 1.)], (4., 4.)) == 0
    baseline = record("baseline", 3., 3., method="baseline")
    records = [baseline, record("a", 1., 3.), record("b", 2., 2.), record("c", 3., 1.),
               record("unsupported", -100., 0., status="unsupported")]
    settings = replace(protocol(), quality_scale=1., cost_scale=1., hypervolume_reference=(4., 4.))
    summary = archive_summary(records, baseline, settings)
    assert set(summary["archive_ids"]) == {"a", "b", "c"}
    assert summary["hypervolume"] == 6
    assert summary["candidate_hypervolume_contributions"] == {"a": 1., "b": 1., "c": 1.}
    with pytest.raises(ProtocolError, match="finite"):
        hypervolume_2d([(float("nan"), 1)], (4, 4))


@pytest.mark.parametrize("base,new,expected", [(42.655, 40.852, -4.22693705310045), (2, 2.8, 40), (175, 322, 84)])
def test_historical_percent_arithmetic_against_independent_reference(base, new, expected):
    assert relative_change(base, new)["percent"] == pytest.approx(expected)
    assert relative_change(base, new)["percent"] == pytest.approx(100 * (new - base) / base)


def test_missing_nonfinite_zero_sign_and_duplicate_configuration_statuses():
    assert relative_change(0, 1)["status"] == "undefined"
    assert relative_change(0, 0)["percent"] == 0
    assert relative_change(None, 1)["status"] == "unavailable"
    assert relative_change(1, float("inf"))["status"] == "invalid"
    audited = audit_historical_percent(2, 2.8, -40)
    assert audited["sign_mismatch"]
    assert "rounded" in audited["source"]
    records = [record("same", .1, 20, method="baseline"), record("same", .2, 10)]
    with pytest.raises(ProtocolError, match="Duplicate"):
        table_rows({"candidates": records})


@pytest.mark.parametrize("method", ["exhaustive", "random", "evolution", "bayesian"])
def test_small_actual_search_budget_trace_and_generated_reports(tmp_path, method):
    if method == "bayesian":
        pytest.importorskip("optuna")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        settings = protocol()
        space = (CandidateSpec("train", {"epochs": 2}), CandidateSpec("svd", {"rank_ratio": .5, "finetune_epochs": 0}),
                 CandidateSpec("unsupported_operation"), CandidateSpec("pruning", {"amount": .2, "finetune_epochs": 0}))
        manifest = ExperimentRunner(bundle(), settings, tmp_path / method).search(
            space, SearchConfig(method=method, max_evaluations=4, max_seconds=40, population_size=2, tpe_startup_trials=1))
        search = manifest["search"]
        assert search["status"] == "succeeded", search
        assert search["charged_evaluations"] <= 4
        assert search["baseline_evaluations"] == 1
        assert all(record["wall_seconds"] > 0 for record in manifest["candidates"])
        assert any(item["event"] == "evaluated" for item in search["trace"])
        if method == "evolution":
            assert "GOLEM" in search["engine"]
            assert any(item["event"] == "golem_population" for item in search["trace"])
            assert not any(item.get("type") == "crossover" for item in search["trace"])
            assert all(item["crossover_types"] == ["CrossoverTypesEnum.none"] for item in search["trace"] if item["event"] == "golem_population")
        if method == "bayesian":
            assert any(item["event"] == "optuna_proposal" and not item["startup_phase"] for item in search["trace"])
        rows = table_rows(manifest)
        saved = json.loads((tmp_path / method / "table.json").read_text(encoding="utf-8"))
        assert saved == rows
        assert (tmp_path / method / "archive.png").is_file()
        assert all(row["test_value"] is None for row in rows if row["status"] != "succeeded")
    finally:
        torch.set_num_threads(previous)


def comparison_manifest(method, seed, hv, *, engine_status="succeeded", feasible=True):
    baseline = record("baseline", .2, 100, method="baseline")
    candidate = record("candidate", .1, 50, status="succeeded" if feasible else "unsupported")
    return {"status": "succeeded", "data": {"task": "classification", "metadata": {"scenario": "test", "data_sha256": "shared-raw"},
              "roles": {"train": {"x_sha256": f"partition-{seed}"}}},
            "environment": {"source_sha256": "code"}, "baseline_training": {"state_sha256": f"trained-{seed}"},
            "protocol": {"seed": seed, "repeat_seeds": [seed], "device": "cpu"},
            "candidates": [baseline, candidate],
            "search": {"status": engine_status, "configuration": {"method": method, "seed": seed, "max_evaluations": 4, "max_seconds": 60}},
            "selection": {"hypervolume": hv, "feasible_ids": ["baseline", "candidate"] if feasible else ["baseline"]}}


def test_paired_seeds_bootstrap_grouping_failures_and_baseline_exclusion():
    manifests = [comparison_manifest(method, seed, float(seed + (method == "evolution")), feasible=seed != 2)
                 for method in ("random", "evolution") for seed in range(3)]
    summary = compare_search_runs(manifests, bootstrap_samples=50)
    comparison = summary["comparisons"][0]
    assert comparison["paired_seeds"] == [0, 1, 2]
    assert comparison["mean_difference"] == 1
    assert comparison["feasible_solution_probability"] == pytest.approx(2 / 3)
    assert comparison["bootstrap_95_percentile_interval"] == [1., 1.]
    with pytest.raises(ProtocolError, match="Duplicate"):
        compare_search_runs(manifests + [manifests[0]])
    changed = copy.deepcopy(manifests)
    changed[-1]["data"]["roles"]["train"]["x_sha256"] = "different-data"
    with pytest.raises(ProtocolError, match="share all data"):
        compare_search_runs(changed)
