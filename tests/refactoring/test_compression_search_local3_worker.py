"""Synthetic schedule, timer, ledger and full-callback boundary checks."""
import json
import time

import pytest
import torch
from torch import nn
from hypothesis import given, strategies as st

from fedcore.experiments.compression_search_local3_evaluation import (
    Local3FullEvaluator, Local3Ledger, PhaseRecorder, interval_union, load_local3_source)
from fedcore.experiments.compression_search_local3_worker import (
    admission_genotypes, statistics_candidates, statistics_schedule, verify_metadata_tapes)


@given(st.lists(st.tuples(st.integers(-10, 30), st.integers(-10, 30)), max_size=30))
def test_interval_union_matches_independent_integer_coverage(intervals):
    ordered = [(min(first, last), max(first, last)) for first, last in intervals]
    reference = sum(any(first <= point and last >= point+1 for first, last in ordered) for point in range(20))
    assert interval_union(ordered, start=0, end=20) == reference


def test_nested_and_failed_phases_retain_the_parent_residual():
    recorder = PhaseRecorder()
    recorder.observe_transform(dict(phase="observe", started_at=1., completed_at=None, state="started", node="n"))
    recorder.observe_transform(dict(phase="observe", started_at=1., completed_at=3., state="completed", node="n"))
    recorder.observe_transform(dict(phase="solve", started_at=3., completed_at=None, state="started", node="s"))
    recorder.rows.append(dict(phase="fixed_profile", started_at=.5, completed_at=4.))
    assert len(recorder.rows) == 3
    assert recorder.partition(0., 5., {"observe", "solve", "fixed_profile"}) == dict(
        total_seconds=5., covered_seconds=3.5, uncovered_seconds=1.5)
    assert recorder.partition(.5, 4., {"observe", "solve"})["uncovered_seconds"] == 1.5


def test_s_schedule_preserves_all_488_calls_and_balanced_orders():
    schedule = statistics_schedule()
    assert len(schedule) == 22 and sum(row["count"] for row in schedule) == 488
    assert [(row["arm"], row["count"]) for row in schedule[:2]] == [("C0", 4), ("C1", 4)]
    assert [row["group"] for row in schedule[2:6]] == ["control"]*4
    assert [row["group"] for row in schedule[-4:]] == ["control"]*4
    paired = [row for row in schedule if row["group"] == "paired"]
    assert [(paired[i]["arm"], paired[i+1]["arm"]) for i in range(0, 12, 2)] == [("C0", "C1"), ("C1", "C0")]*3
    assert len({row["stream_id"] for row in schedule}) == 22
    from fedcore.experiments.compression_search_digits_protocol import INITIAL_GENOTYPES, canonical_solution_id
    tape = statistics_candidates()
    assert len(tape) == len({canonical_solution_id(value) for value in tape}) == 24
    assert tape[:4] == INITIAL_GENOTYPES
    assert [(type(value.genes[0].choice).__name__, value.genes[0].choice.rank) for value in admission_genotypes()] == [
        ("WeightedChoice", 4), ("WeightedChoice", 32), ("ASVDChoice", 8),
        ("ASVDChoice", 32), ("AffinePCAChoice", 4), ("WeightedChoice", 8)]


def test_worker_tapes_match_the_pre_source_protocol_and_reject_drift():
    from fedcore.experiments.compression_search_evaluation import PackageFatal
    from fedcore.experiments.compression_search_local3_protocol import metadata_tapes
    expected = metadata_tapes()
    verify_metadata_tapes(expected)
    expected["statistics_ids"] = list(reversed(expected["statistics_ids"]))
    with pytest.raises(PackageFatal):
        verify_metadata_tapes(expected)


def job(tmp_path):
    return dict(job="synthetic", kind="statistics_admission", directory=str(tmp_path),
        protocol_sha256="p", admission_deadline=time.perf_counter()+30,
        hard_deadline=time.perf_counter()+50, package_deadline=time.perf_counter()+60,
        compute_admission_deadline=time.perf_counter()+60)


def test_ledger_is_global_bounded_and_never_refunds_failed_effects(tmp_path):
    from fedcore.experiments.compression_search_digits_evaluation import DigitsDeadline
    path, spec = tmp_path/"attempts.jsonl", job(tmp_path)
    path.write_text("\n".join(json.dumps(dict(attempt=i, protocol_sha256="p")) for i in range(1, 2700))+"\n", encoding="utf-8")
    ledger = Local3Ledger(path, spec)
    assert ledger.charge("full_validation", "candidate") == 2700
    active = json.loads((tmp_path/"active-request.json").read_text())
    assert active["active"] and active["hard_deadline"] <= active["started_at"]+20
    ledger.complete()
    assert json.loads((tmp_path/"active-request.json").read_text())["active"] is False
    next_job = Local3Ledger(path, dict(spec, job="next"))
    with pytest.raises(DigitsDeadline):
        next_job.charge("full_validation", "candidate")
    assert len(path.read_text().splitlines()) == 2700


def test_source_adapter_uses_saved_root_and_completes_one_action_without_copy(tmp_path, monkeypatch):
    import fedcore.experiments.compression_search_local3_evaluation as module
    root = tmp_path/"sources"
    root.mkdir()
    spec = dict(job(tmp_path), source_root=str(root), split_seed=19901, output="new-output")
    ledger = Local3Ledger(tmp_path/"attempts.jsonl", spec)
    def load(adapted, received):
        assert adapted["output"] == str(tmp_path) and received is ledger
        received.charge("source_baseline_validation", "M0-19901")
        return "source"
    monkeypatch.setattr(module, "load_source", load)
    assert load_local3_source(spec, ledger) == "source"
    assert ledger.used == 1 and ledger.active is None


def test_numerical_comparison_rejects_operator_and_class_changes_even_with_same_quality():
    from copy import deepcopy
    from fedcore.experiments.compression_search_evaluation import PackageFatal, metrics
    from fedcore.experiments.compression_search_local3_evaluation import assert_same_result
    labels, logits = torch.zeros(3, dtype=torch.long), torch.zeros(3, 10)
    original = dict(outcome="WorkflowSuccess", status="finite", tensor_bytes=100,
        quality_accepted=True, metrics=metrics(logits, labels), logits=logits,
        operators={"0": dict(weight=torch.eye(2), bias=torch.zeros(2))})
    assert_same_result(original, deepcopy(original), labels)
    changed = deepcopy(original)
    changed["operators"]["0"]["weight"][0, 0] += 1e-7
    with pytest.raises(PackageFatal, match="operators differ"):
        assert_same_result(original, changed, labels)
    changed = deepcopy(original)
    changed["outcome"] = "ResourceFailure"
    with pytest.raises(PackageFatal, match="outcome"):
        assert_same_result(original, changed, labels)


def test_dense_capture_reads_weights_without_bias_cancellation_or_forwards(monkeypatch):
    from types import SimpleNamespace
    from fedcore.models.network_impl.decomposed_layers import DecomposedLinear
    from fedcore.experiments.compression_search_local3_evaluation import dense_operator_snapshots
    original = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
    compressed = DecomposedLinear(original[0], decomposing_mode=False, compose_mode="two_layers")
    compressed.set_U_S_Vh(torch.eye(2), torch.ones(2), torch.eye(2)*.001)
    compressed.compose_weight_for_inference()
    with torch.no_grad():
        compressed.bias.fill_(1e8)
    def forbidden(*args):
        raise AssertionError("Operator capture must not run forward")
    monkeypatch.setattr(compressed, "forward", forbidden)
    result = dense_operator_snapshots(nn.Sequential(compressed, nn.ReLU(), original[2]),
        SimpleNamespace(model=original))
    assert torch.equal(result["0"]["weight"], torch.eye(2)*.001)
    assert torch.equal(result["0"]["bias"], torch.full((2,), 1e8))


def test_worker_interprets_every_stream_instead_of_reusing_equal_objectives(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import fedcore.experiments.compression_search_evaluation as evaluation
    import fedcore.experiments.compression_search_local3_worker as worker
    from fedcore.experiments.compression_search_digits_protocol import canonical_solution_id
    from fedcore.experiments.protocol import TensorSplit
    split = TensorSplit(torch.zeros(1, 1), torch.zeros(1, dtype=torch.long), ("synthetic:0",))
    logits = torch.zeros(1, 10)
    quality = evaluation.metrics(logits, split.y)
    calls = []
    class Evaluator:
        def __init__(self):
            self.rows, self.tensor_results = [], []
            self.last_model, self.store = None, None
        def __call__(self, genotype):
            calls.append((self.stream_metadata["stream_id"], canonical_solution_id(genotype)))
            self.rows.append(dict(status="finite"))
            self.tensor_results.append(dict(solution_id=canonical_solution_id(genotype),
                outcome="WorkflowSuccess", status="finite", tensor_bytes=100,
                quality_accepted=True, metrics=quality, logits=logits,
                operators={"0": dict(weight=torch.ones(1, 1), bias=torch.zeros(1))}))
    monkeypatch.setattr(evaluation, "compile_checked", lambda genotype, space: genotype)
    monkeypatch.setattr(worker, "_persist_tensors", lambda *args: dict(synthetic=True))
    evaluator = Evaluator()
    result = worker._statistics_job(dict(job(tmp_path), kind="statistics_streams"),
        SimpleNamespace(seed=19901, validation=split), object(), evaluator, PhaseRecorder())
    assert result["candidate_calls"] == len(calls) == 488
    assert len(result["streams"]) == 22
    tape = [canonical_solution_id(value) for value in statistics_candidates()]
    for stream in result["streams"]:
        assert [identity for stream_id, identity in calls if stream_id == stream["stream_id"]] == tape[:stream["count"]]
        assert stream["store_initial_bytes"] == 0 and stream["store_closed"]


@pytest.mark.parametrize("kind", ["statistics_admission", "statistics_streams"])
def test_fixed_stream_stops_on_nonfinite_result_instead_of_claiming_equivalence(tmp_path, monkeypatch, kind):
    from types import SimpleNamespace
    import fedcore.experiments.compression_search_evaluation as evaluation
    import fedcore.experiments.compression_search_local3_worker as worker
    from fedcore.experiments.protocol import TensorSplit
    split = TensorSplit(torch.zeros(1, 1), torch.zeros(1, dtype=torch.long), ("synthetic:0",))
    class Evaluator:
        def __init__(self):
            self.rows, self.tensor_results = [], []
            self.last_model, self.store = None, None
        def __call__(self, workflow):
            self.rows.append(dict(status="ResourceFailure"))
            self.tensor_results.append(dict(status="ResourceFailure", outcome="ResourceFailure"))
    monkeypatch.setattr(evaluation, "compile_checked", lambda genotype, space: genotype)
    evaluator = Evaluator()
    with pytest.raises(evaluation.PackageFatal, match="finite model"):
        worker._statistics_job(dict(job(tmp_path), kind=kind),
            SimpleNamespace(seed=19901, validation=split), object(), evaluator, PhaseRecorder())
    assert len(evaluator.rows) == 1
    streams = json.loads((tmp_path/"statistics-streams.json").read_text())
    assert streams[0]["status"] == "partial" and streams[0]["candidate_calls"] == 1


def test_full_evaluator_injection_charges_before_real_pipeline_boundaries(tmp_path):
    from fedcore.algorithm.low_rank.compression_grammar import compile_genotype
    from fedcore.algorithm.low_rank.grammar_execution import WorkflowSuccess
    from fedcore.experiments.compression_search_digits_evaluation import prepare_space
    from fedcore.experiments.compression_search_digits_protocol import INITIAL_GENOTYPES, TUNED_CATALOG, expected_tensor_bytes
    from fedcore.experiments.compression_search_evaluation import EvaluationSource, metrics
    from fedcore.experiments.protocol import TensorSplit
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10)).eval()
    split = TensorSplit(torch.zeros(5, 64), torch.zeros(5, dtype=torch.long), tuple(f"s{i}" for i in range(5)))
    logits = torch.zeros(5, 10)
    source = EvaluationSource(19901, model, split, split, {}, metrics(logits, split.y), logits)
    space = prepare_space(source, TUNED_CATALOG)
    workflow = compile_genotype(INITIAL_GENOTYPES[0], space)
    calls, restored = [], object()
    class Ledger:
        def charge(self, *args):
            calls.append("charge")
            return 1
        def complete(self):
            calls.append("complete")
    def transform(original, calibration, checked, *, store, phase_callback):
        assert calls == ["charge"] and original is model and checked is workflow
        calls.append("transform")
        return WorkflowSuccess(model, workflow, {})
    def checkpoint(current, features):
        assert calls[-1] == "transform"
        calls.append("checkpoint")
        return restored, b"checkpoint", expected_tensor_bytes(workflow.genotype)
    def prediction(current, features):
        assert current is restored and calls[-1] == "checkpoint"
        calls.append("validation")
        return logits
    evaluator = Local3FullEvaluator(source, space, Ledger(), tmp_path/"events.jsonl",
        context_sha256="synthetic", job_id="synthetic", transform=transform,
        checkpoint=checkpoint, prediction=prediction)
    result = evaluator(workflow)
    assert result.quality_accepted
    assert calls == ["charge", "transform", "checkpoint", "validation", "complete"]
    row = evaluator.rows[0]
    assert row["partition"]["total_seconds"] == pytest.approx(
        row["partition"]["covered_seconds"]+row["partition"]["uncovered_seconds"])
    assert evaluator.callback_intervals[0][1]-evaluator.callback_intervals[0][0] >= row["elapsed_seconds"]
    assert row["outer_completed_at"] == evaluator.callback_intervals[0][1]
