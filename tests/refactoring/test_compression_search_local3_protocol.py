"""Independent serial budget, resource coverage and session partition laws."""
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from fedcore.experiments.compression_search_local3_protocol import resource_coverage, schedule
from fedcore.experiments.compression_search_local3_matrix import session_schedule, summarize_sessions


def test_schedule_keeps_all_conditions_and_nonrepeatable_admissions():
    rows = schedule()
    assert len(rows) == 30 and len({row["job"] for row in rows}) == 30
    assert [row["kind"] for row in rows[:3]] == ["statistics_admission"]*3
    matrices = [row for row in rows if row["kind"] == "matrix"]
    assert [(row["session_id"], row["n"]) for row in matrices] == [
        ("M32", 32), ("M1-512", 512), ("M1-1024", 1024), ("M2-1024", 1024), ("M2-512", 512)]
    searches = [row for row in rows if row["kind"] == "search_diagnostic"]
    assert len(searches) == 18
    assert sum(row["mode"] == "T" for row in searches) == 9
    assert sum(row["mode"] == "E" for row in searches) == 9
    assert sum(searches[index]["mode"] == "T" for index in range(0, 18, 2)) == 5
    assert 18*64+18 + 3*(6*2*24+4*2*24+2*4+1) + 3*(6*2+1) == 2676
    assert 5*(4*6*3*8+4*4*2*8+3*8)+3*7 == 4301
    assert 5*(6*sum(24-2*k for k in (0, 2, 4, 6))+4*4*2*8+16)+15 == 3535


def test_resource_coverage_accounts_for_both_edges_and_internal_gaps():
    samples = [dict(timestamp=t, rss_bytes=100, monitor_error=None) for t in (1., 2., 3.)]
    assert resource_coverage(samples, 0., 4.)["complete"]
    assert not resource_coverage(samples, -2., 4.)["complete"]
    assert not resource_coverage(samples, 0., 6.)["complete"]
    assert not resource_coverage([samples[0], dict(samples[2], timestamp=4.)], 0., 5.)["complete"]
    assert not resource_coverage([dict(samples[0], rss_bytes=None)], 0., 2.)["complete"]
    assert not resource_coverage([], 0., 1.)["complete"]


def _streams(session):
    return [dict(row, primary_seconds=(.6 if row["arm"] == "A" else .65 if row["arm"] == "C" else 1.),
                 request_count=8) for row in session_schedule(1024, session)]


def _resources(*sessions):
    return {name: dict(coverage={"complete": True}, stop=None, descendants_closed=True) for name in sessions}


def test_sessions_never_borrow_a_missing_control_and_retain_negative_result():
    first, second = _streams("one"), _streams("two")
    second = [row for row in second if not(row["group"] == "control" and row["pair"] == 3 and row["invalid_count"] == 6)]
    planned = [dict(session_id=name, n=1024) for name in ("one", "two")]
    result = summarize_sessions(first+second, _resources("one", "two"), planned)
    full = [row for row in result if row["session_id"] == "one"]
    assert len(full) == 8 and all(row["observed_win"] for row in full)
    incomplete = [row for row in result if row["session_id"] == "two" and row["invalid_count"] == 6]
    assert len(incomplete) == 2 and all(not row["eligible"] for row in incomplete)
    resource = _resources("one")
    resource["one"]["coverage"]["complete"] = False
    assert all(not row["eligible"] for row in summarize_sessions(first, resource, planned))
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_sessions(first+[first[-1]], _resources("one"), planned)
    with pytest.raises(ValueError, match="Unplanned"):
        summarize_sessions(first, {})


def test_parent_and_schedule_import_without_loading_ml():
    code = ("import sys; import fedcore.experiments.compression_search_local3 as parent; "
            "data=parent.metadata_tapes(); assert len(data['catalog_ids']) == 440; "
            "assert len(set(data['statistics_ids'])) == 24; "
            "assert not ({'torch','numpy','sklearn','golem','fedot'} & set(sys.modules))")
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


def test_unexpected_watchdog_exception_closes_worker_before_escape(tmp_path):
    from fedcore.experiments.compression_search_local3 import _worker_lifetime
    process = SimpleNamespace(pid=123, poll=lambda: None)
    with patch("fedcore.experiments.compression_search_local3._kill_owned", return_value=[]) as kill:
        with patch("fedcore.experiments.compression_search_local3._survivors", return_value=([], [])):
            with pytest.raises(OSError, match="publication"):
                with _worker_lifetime(tmp_path, {123: 1.}) as lifetime:
                    lifetime["process"] = process
                    raise OSError("resource publication failed")
    kill.assert_called_once()
    assert (tmp_path/"emergency-cleanup.json").exists()


def test_missing_session_and_partial_control_remain_visible():
    result = summarize_sessions([], {})
    assert len(result) == 40 and all(not row["eligible"] and row["pairs"] == 0 for row in result)
    rows = _streams("M1-1024")
    index = next(i for i, row in enumerate(rows) if row["group"] == "control" and row["invalid_count"] == 6)
    rows[index] = dict(rows[index], request_count=7)
    cells = [row for row in summarize_sessions(rows, _resources("M1-1024"))
             if row["session_id"] == "M1-1024" and row["invalid_count"] == 6]
    assert len(cells) == 2 and all(not row["bb_pass"] and not row["eligible"] for row in cells)
