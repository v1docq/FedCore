"""CPU witness tests; all numerical examples use only 8x8 synthetic state."""
import json
from dataclasses import replace
from pathlib import Path
import struct

import pytest

from fedcore.experiments.compression_search_matrix import (
    ARM_ORDERS, SIZES, SVDTelemetry, invoke_arm, matrix_request,
    matrix_source, request_tape, stream_schedule)
from fedcore.experiments.compression_search_local2 import freeze_selections, search_jobs, write_json


def test_frozen_schedule_counts_and_balanced_orders():
    requests = svds = 0
    for n in SIZES:
        schedule = stream_schedule(n)
        assert len(schedule) == 107
        assert len({row["stream_id"] for row in schedule}) == 107
        for row in schedule:
            tape = request_tape(row["invalid_count"], row["tape_seed"])
            assert sorted(i for i, _ in tape) == list(range(8))
            assert sum(word == "LL" for _, word in tape) == row["invalid_count"]
            requests += 8
            svds += sum(word == "LD" or row["arm"] == "B" for _, word in tape)
    assert requests+35 == 4315
    assert svds+25 == 3545
    for arm in "ABC":
        assert [sum(order[i] == arm for order in ARM_ORDERS) for i in range(3)] == [2, 2, 2]
    assert len(search_jobs()) == 54
    assert matrix_request(8, "LD").max_peak_bytes == 512*1024**2


@pytest.mark.parametrize("arm", ["A", "B", "C"])
@pytest.mark.parametrize("word", ["LD", "LL"])
def test_reached_transition_has_exact_svd_law_and_preserves_source(arm, word, tmp_path):
    import torch
    from fedcore.experiments.compression_search_evaluation import model_identity
    source = matrix_source(8)
    identity = model_identity(source)
    rng = torch.get_rng_state().clone()
    with SVDTelemetry(tmp_path) as telemetry:
        # The unit test shares a process with thousands of unrelated imported
        # models. Test transition laws independently of that accumulated RSS;
        # real fresh experiment workers retain the frozen 512 MiB request cap.
        request = replace(matrix_request(8, word), max_peak_bytes=8*1024**3)
        actual, row = invoke_arm(source, request, arm)
        assert row["accepted"] == (word == "LD"), str(row.get("early_rejection") or row.get("rejection"))
        assert telemetry.started == int(word == "LD" or arm == "B")
        assert telemetry.completed == telemetry.started
        assert model_identity(source) == identity
        assert torch.equal(torch.get_rng_state(), rng)
        if word == "LD":
            assert actual.storage_bytes == 576
            assert torch.equal(actual.model[0].bias, source[0].bias)
        else:
            errors = row.get("early_rejection") or [row["rejection"]]
            assert all(error["code"] == "UnsupportedTransition" for error in errors), str(errors)
    header = struct.unpack_from("<QQdd", (tmp_path/"svd-telemetry.bin").read_bytes())
    assert header[0] == header[1] == int(word == "LD" or arm == "B")


def test_validation_selection_is_frozen_on_time_before_test(tmp_path):
    job = dict(kind="search", split_seed=19901, job="search-x", admission_deadline=10.)
    directory = tmp_path/"jobs"/"search-x"
    write_json(directory/"job.json", job)
    def candidate(identity, quality, at, byte_count=9512):
        return dict(solution_id=identity, quality=quality, tensor_bytes=byte_count,
            completed_at=at, status="finite", quality_accepted=True,
            metrics={"cross_entropy":quality*2.302585092994046})
    write_json(directory/"result.json", dict(status="succeeded",deadline=10.,evaluations=[
        candidate("early", .2, 9.), candidate("late", .01, 11.), candidate("large", .1, 8.,60000)]))
    digest = freeze_selections(tmp_path)
    chosen = json.loads((tmp_path/"selected.json").read_text(encoding="utf-8"))
    assert len(digest) == 64
    assert {row["solution_id"] for row in chosen["selected"]} == {"early"}
    with pytest.raises(RuntimeError, match="overwrite"):
        freeze_selections(tmp_path)
