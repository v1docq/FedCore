"""Independent geometry and artifact-analysis references; no model runtime."""
from fractions import Fraction
from itertools import combinations, product
import json
import math

import pytest

from fedcore.experiments.compression_search_analysis import (
    build_analysis, finite_evaluations, hypervolume_2d, normalized_hypervolume,
    pareto_points, select_at_caps, virtual_stop_diagnostic)


def rectangle_union_reference(points, reference=(Fraction(2), Fraction(21, 20))):
    """Independent cell decomposition of the union, using exact fractions."""
    points = [(Fraction(x), Fraction(y)) for x, y in points
              if Fraction(x) < reference[0] and Fraction(y) < reference[1]]
    xs = sorted({reference[0]} | {point[0] for point in points})
    ys = sorted({reference[1]} | {point[1] for point in points})
    area = Fraction(0)
    for left, right in zip(xs, xs[1:]):
        for bottom, top in zip(ys, ys[1:]):
            center = ((left + right) / 2, (bottom + top) / 2)
            if any(x <= center[0] and y <= center[1] for x, y in points):
                area += (right - left) * (top - bottom)
    return area


def test_hypervolume_matches_exact_rectangle_union_for_all_small_subsets():
    points = [(Fraction(1, 4), Fraction(4, 5)), (Fraction(1, 2), Fraction(1, 2)),
        (Fraction(1), Fraction(1, 4)), (Fraction(3, 2), Fraction(1, 8)),
        (Fraction(3), Fraction(1, 10)), (Fraction(1, 8), Fraction(2))]
    for length in range(len(points) + 1):
        for subset in combinations(points, length):
            floats = [(float(x), float(y)) for x, y in subset]
            expected = float(rectangle_union_reference(subset))
            assert hypervolume_2d(floats) == pytest.approx(expected, rel=1e-14, abs=1e-14)
            assert hypervolume_2d(floats[::-1] + floats) == pytest.approx(expected, rel=1e-14, abs=1e-14)
            assert normalized_hypervolume(floats) == pytest.approx(expected / 2.1)


def test_pareto_ties_dominance_and_fixed_reference_clipping():
    points = [(1, .5), (1, .75), (1.5, .25), (1, .5), (2, .1), (.25, 1.05)]
    assert pareto_points(points) == ((.25, 1.05), (1., .5), (1.5, .25), (2., .1))
    assert hypervolume_2d(points) == pytest.approx(.675)
    assert hypervolume_2d([(3, .01), (.1, 2), (2, .5), (.5, 1.05)]) == 0
    assert hypervolume_2d([(0, 0)]) == 2.1
    assert normalized_hypervolume([(0, 0)]) == 1
    assert hypervolume_2d([]) == 0


@pytest.mark.parametrize("point", ((math.nan, .5), (.5, math.inf), (True, .5), (.5,)))
def test_nonfinite_or_malformed_fitness_is_never_given_synthetic_area(point):
    with pytest.raises(ValueError):
        hypervolume_2d([point])


def evaluation(identity, quality, tensor_bytes, completed_at, *, accepted=True, attempt=1):
    return {"event": "evaluation", "status": "finite", "solution_id": identity,
        "workflow_fingerprint": "workflow-" + identity, "quality": quality,
        "cost": tensor_bytes / 49_416, "tensor_bytes": tensor_bytes,
        "quality_accepted": accepted, "global_attempt": attempt,
        "started_at": completed_at - .1, "completed_at": completed_at,
        "metrics": {"cross_entropy": quality * math.log(2), "accuracy": .9, "macro_f1": .9}}


def test_deadline_filter_and_selection_preserve_real_bytes_and_tie_breaking():
    rows = [evaluation("z", .5, 12_488, 101), evaluation("a", .5, 12_488, 102),
        evaluation("rejected", .1, 12_488, 103, accepted=False),
        evaluation("late", .01, 12_488, 131), evaluation("larger", .4, 39_112, 104)]
    finite = finite_evaluations(rows, deadline=130)
    assert {row["solution_id"] for row in finite} == {"z", "a", "rejected", "larger"}
    selected = select_at_caps(finite)
    assert [row["selected"]["solution_id"] for row in selected] == ["a", "a", "larger"]
    assert select_at_caps(finite, caps=(100,))[0]["selected"] is None


def test_virtual_stopping_uses_distinct_finite_evaluations_and_retains_full_archive():
    rows = [evaluation(str(index), 1, 22_792, 100 + index, attempt=index) for index in range(1, 21)]
    rows[-1] = evaluation("20", .2, 12_488, 120, attempt=20)
    diagnostic = virtual_stop_diagnostic(rows, parent_started_at=100)
    assert diagnostic["status"] == "triggered" and diagnostic["stop_after_finite"] == 18
    assert diagnostic["later_finite_evaluations"] == 2
    assert diagnostic["stopped_after_parent_seconds"] == 18
    initial_hv = float(rectangle_union_reference([(Fraction(1), Fraction(22_792, 49_416))]))
    final_hv = float(rectangle_union_reference([(Fraction(1, 5), Fraction(12_488, 49_416))]))
    assert diagnostic["prefix_hypervolume"] == pytest.approx(initial_hv)
    assert diagnostic["full_hypervolume"] == pytest.approx(final_hv)
    assert diagnostic["missed_hypervolume"] == pytest.approx(final_hv - initial_hv)
    assert diagnostic["actual_time_saved"] is None
    assert virtual_stop_diagnostic(rows[:17])["status"] == "not_triggered"
    assert virtual_stop_diagnostic(rows, applicable=False, reason="12 attempts")["status"] == "not_applicable"
    repeated = [evaluation(str(index % 12), 1, 22_792, 100 + index) for index in range(48)]
    assert virtual_stop_diagnostic(repeated)["finite_unique_evaluations"] == 12
    assert virtual_stop_diagnostic(repeated)["status"] == "not_triggered"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def store_job(output, job, result=None, *, evaluations=(), trace=(), composition=(), requests=()):
    directory = output / "jobs" / job["job"]
    write_json(directory / "job.json", job)
    if result is not None:
        write_json(directory / "result.json", {"job": job["job"], "kind": job["kind"], "status": "succeeded", **result})
    for filename, rows in (("events.jsonl", evaluations), ("search-trace.jsonl", trace),
                           ("composition-events.jsonl", composition), ("cache-requests.jsonl", requests)):
        if rows:
            (directory / filename).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_analysis_reconstructs_deadline_archive_and_exact_byte_pairs(tmp_path):
    jobs = [{"job": "oracle-41", "kind": "oracle", "model_seed": 41, "worker_started_at": 1},
        *({"job": f"search-41-time-11-{mode}", "kind": "search", "model_seed": 41,
            "search_seed": 11, "mode": mode, "budget": "time",
            **({"worker_started_at": 100} if mode in "RT" else {})} for mode in "RTE")]
    write_json(tmp_path / "protocol.json", {"protocol_sha256": "frozen", "jobs": jobs})
    write_json(tmp_path / "resource-ledger.json", {"status": "stopped", "jobs": [], "elapsed_seconds": 150})
    oracle = [evaluation(f"oracle-{index}", .2, 12_488, 2 + index, attempt=index) for index in range(80)]
    store_job(tmp_path, jobs[0], {"model_seed": 41, "evaluations": oracle}, evaluations=oracle)
    r = [evaluation("r", 1., 39_112, 120), evaluation("late", .01, 12_488, 131, attempt=2)]
    t = [evaluation("t", .8, 39_112, 125)]
    for job, rows in ((jobs[1], r), (jobs[2], t)):
        store_job(tmp_path, job, {"model_seed": 41, "mode": job["mode"], "search_seed": 11,
            "budget": "time", "deadline": 130, "evaluations": rows,
            "search": {"created_proposals": len(rows), "charged_evaluations": len(rows), "stop_reason": "time_budget"}},
            evaluations=rows, trace=[{"event": "proposal", "proposal_id": index,
                "canonical_solution_id": row["solution_id"]} for index, row in enumerate(rows, 1)])
    analysis = build_analysis(tmp_path)
    assert analysis["status"] == "partial" and analysis["protocol_sha256"] == "frozen"
    by_mode = {row["mode"]: row for row in analysis["search_runs"]}
    assert by_mode["E"]["status"] == "unexecuted"
    assert by_mode["R"]["finite_completed"] == 2 and by_mode["R"]["finite_within_deadline"] == 1
    assert by_mode["R"]["finite_after_deadline"] == 1
    assert by_mode["R"]["observed_proposals"] == 2
    assert by_mode["R"]["observed_proposals_basis"] == "completed_evaluation_rows"
    assert by_mode["R"]["hypervolume"] == pytest.approx((2 - 1) * (1.05 - 39_112 / 49_416))
    assert by_mode["R"]["oracle_complete"] is True and by_mode["R"]["gap_to_oracle"] > 0
    assert by_mode["R"]["curves"][-1]["included_in_deadline_archive"] is False
    assert by_mode["R"]["curves"][-1]["elapsed_parent_seconds"] == 31
    frontier = next(row for row in analysis["frontiers"] if row["job"] == jobs[1]["job"])
    assert [row["solution_id"] for row in frontier["all"]] == ["r"]
    assert [row["solution_id"] for row in frontier["all_including_late"]] == ["late"]
    common = next(row for row in analysis["common_byte_pairs"] if row["tensor_bytes"] == 39_112)
    assert common["R"]["solution_id"] == "r" and common["T"]["solution_id"] == "t"
    assert common["delta_CE_T_minus_R"] == pytest.approx(-.2 * math.log(2))
    assert jobs[3]["job"] in analysis["missing_jobs"]
    assert len(analysis["oracle_runs"][0]["curves"]) == 80  # JSONL/result duplicates are not counted twice.


def test_partial_worker_events_remain_visible_without_result_or_oracle(tmp_path):
    job = {"job": "interrupted", "kind": "search", "model_seed": 41, "search_seed": 11,
           "mode": "E", "budget": "attempts", "worker_started_at": 100}
    write_json(tmp_path / "protocol.json", {"jobs": [job]})
    store_job(tmp_path, job, evaluations=[evaluation("finished", .5, 12_488, 110)])
    analysis = build_analysis(tmp_path)
    row, = analysis["search_runs"]
    assert row["status"] == "partial" and row["gap_to_oracle"] is None
    assert row["finite_within_deadline"] == 1 and not row["oracle_complete"]
    assert row["virtual_stop"]["status"] == "not_applicable"
    assert analysis["missing_jobs"] == ["interrupted"]


def test_launched_worker_without_result_or_events_is_partial(tmp_path):
    job = {"job": "launched", "kind": "search", "model_seed": 41, "search_seed": 11,
           "mode": "T", "budget": "time", "worker_started_at": 100}
    write_json(tmp_path / "protocol.json", {"jobs": [job]})
    store_job(tmp_path, job)
    attempts = [{"event": "mlp_attempt", "job": "launched", "purpose": "full_validation",
                 "solution_id": "unfinished", "started_at": 110}]
    (tmp_path / "mlp-attempts.jsonl").write_text(json.dumps(attempts[0]) + "\n", encoding="utf-8")
    analysis = build_analysis(tmp_path)
    row, = analysis["search_runs"]
    assert row["status"] == "partial" and row["full_evaluations_completed"] == 0
    assert row["observed_proposals"] == 1
    assert row["observed_proposals_basis"] == "recorded_full_validation_starts"
    assert row["full_evaluations_started"] == 1
    assert analysis["missing_jobs"] == ["launched"]


def test_empty_package_is_explicitly_partial_and_json_safe(tmp_path):
    analysis = build_analysis(tmp_path)
    assert analysis["status"] == "partial" and analysis["search_runs"] == []
    assert analysis["admissibility"]["status"] == "unexecuted"
    assert analysis["early_late_ratio_summary"]["count"] == 0
    json.dumps(analysis, allow_nan=False)


def test_early_late_pair_checks_outputs_and_saves_all_raw_timing_counters(tmp_path):
    ids = ["word-" + "".join(word) for length in (1, 2, 3) for word in product("PLQDR", repeat=length)]
    ids += ["rank-zero", "rank-too-large", "duplicate-mask", "out-of-range-mask",
        "compute-dtype", "unknown-scope", "overlapping-scopes", "onnx"]
    jobs = [{"job": "pair-" + branch, "kind": "early_late", "stream_seed": 9911, "branch": branch} for branch in "AB"]
    write_json(tmp_path / "protocol.json", {"jobs": jobs})
    for job in jobs:
        rows = [{"case_id": identity, "accepted": index < 72,
            "output_identity": "out-" + identity if index < 72 else None,
            "tensor_bytes": 200 if index < 72 else None,
            "elapsed_seconds": .02 if job["branch"] == "A" else .03,
            "check_seconds": .001, "execution_seconds": .01,
            "decompositions_started": int(index < 72 or job["branch"] == "B"),
            "completed_steps": 1 if index < 72 else 0,
            "stage_timings": {"numerical_seconds": .01}}
            for index, identity in enumerate(ids)]
        store_job(tmp_path, job, {"rows": rows, "stream_seconds": 4 if job["branch"] == "A" else 6})
    pair, = build_analysis(tmp_path)["early_late_pairs"]
    assert pair["valid"] is True and pair["early_over_late"] == pytest.approx(2 / 3)
    assert pair["A_decompositions_started"] == 72 and pair["B_decompositions_started"] == 163
    assert len(pair["measurements"]) == 163 and pair["subsets"]["accepted"]["count"] == 72
    assert pair["measurements"][0]["A"]["stage_timings"] == {"numerical_seconds": .01}
    path = tmp_path / "jobs" / "pair-B" / "result.json"
    bad = json.loads(path.read_text())
    bad["rows"][0]["output_identity"] = "changed"
    write_json(path, bad)
    analysis = build_analysis(tmp_path)
    assert analysis["early_late_pairs"][0]["valid"] is False
    assert analysis["status"] == "inconsistent"


def test_cache_pair_validates_48_status_quality_byte_records_and_useful_hits(tmp_path):
    jobs = [{"job": "cache-" + branch, "kind": "cache", "model_seed": 41,
             "stream_seed": 9921, "branch": branch} for branch in "AB"]
    write_json(tmp_path / "protocol.json", {"jobs": jobs})
    for job in jobs:
        requests = [{"index": index, "solution_id": str(index % 12), "status": "finite",
            "quality": .5, "tensor_bytes": 12_488, "cache_hit": job["branch"] == "A" and index >= 12}
            for index in range(48)]
        count = 12 if job["branch"] == "A" else 48
        rows = [evaluation(str(index % 12), .5, 12_488, 100 + index, attempt=index) for index in range(count)]
        store_job(tmp_path, job, {"model_seed": 41, "stream_seed": 9921, "branch": job["branch"],
            "requests": requests, "evaluations": rows, "stream_seconds": 1 if job["branch"] == "A" else 4})
    pair, = build_analysis(tmp_path)["cache_pairs"]
    assert pair["valid"] is True
    assert pair["A_full_evaluations"] == 12 and pair["B_full_evaluations"] == 48
    assert pair["A_cache_hits"] == 36 and pair["B_cache_hits"] == 0
    assert pair["enabled_over_disabled"] == .25
    path = tmp_path / "jobs" / "cache-B" / "result.json"
    bad = json.loads(path.read_text())
    bad["requests"][30]["quality"] = .6
    write_json(path, bad)
    analysis = build_analysis(tmp_path)
    assert analysis["cache_pairs"][0]["valid"] is False
    assert {"index": 30, "fields": ["quality"]} in analysis["cache_pairs"][0]["mismatches"]


def test_order_pairs_keep_matched_dense_storage_and_metric_sign(tmp_path):
    job = {"job": "order", "kind": "order", "model_seed": 41}
    write_json(tmp_path / "protocol.json", {"jobs": [job]})
    rows = [{"pair": "mask-.25-8", "word": word, "model_seed": 41, "tensor_bytes": 49_416,
        "metrics": {"cross_entropy": ce, "accuracy": .9, "macro_f1": .8},
        "singular_values": [1, .5], "zero_fraction": .25} for word, ce in (("PLD", .1), ("LDP", .2))]
    store_job(tmp_path, job, {"rows": rows})
    pair, = build_analysis(tmp_path)["order_pairs"]
    assert pair["equal_dense_bytes"] is True
    assert pair["metrics_second_minus_first"]["cross_entropy"] == pytest.approx(.1)
