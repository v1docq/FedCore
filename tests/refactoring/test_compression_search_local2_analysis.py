"""Pure-file regressions for censored and failed second-package evidence."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT / "docs/svd_compression/issue99-local2-report-source/export.py"


def _load_export():
    spec = importlib.util.spec_from_file_location("local2_export_pure_regression", EXPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class TestLocal2AnalysisFailures(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.export = _load_export()
        self.charges = []
        _write(self.run / "protocol.json", dict(git_revision="pure-fixture", utc="fixture"))
        _write(self.run / "resource-ledger.json", dict(status="partial", jobs=[]))

    def job(self, name, mode="T", completed=1, charged=None, state="succeeded",
            kind="search", fatal_last=False, last_at=None):
        directory = self.run / "jobs" / name
        job = dict(job=name, kind=kind, split_seed=19901, admission_deadline=10., worker_started_at=0.)
        if kind == "search":
            job.update(mode=mode, budget="attempts", search_seed=11)
        _write(directory / "job.json", job)
        _write(directory / "resources.json", dict(status=state, resources={}))
        _write(directory / "result.json", dict(status="succeeded" if state == "succeeded" else "failed",
            deadline=10., complete_catalog=kind == "oracle" and completed == 440))
        rows = []
        for number in range(1, completed + 1):
            rows.append(dict(status="package_fatal" if fatal_last and number == completed else "finite",
                solution_id=f"candidate-{number}", completed_at=number / 100., global_attempt=number,
                quality=.4, cost=42280 / 68904, tensor_bytes=42280, quality_accepted=True,
                metrics=dict(cross_entropy=.4 * math.log(10), accuracy=.95)))
        if rows and last_at is not None:
            rows[-1]["completed_at"] = last_at
        (directory / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        for _ in range(completed if charged is None else charged):
            self.charges.append(dict(attempt=len(self.charges) + 1, job=name, purpose="full_validation"))
        (self.run / "mlp-attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.charges), encoding="utf-8")
        return directory

    def test_orphan_charge_is_counted_and_cache_recovered_from_actual_trace(self):
        directory = self.job("stopped-e", mode="E", completed=1, charged=2, state="partial")
        trace = [dict(event="proposal", proposal_id=1), dict(event="proposal", proposal_id=2),
                 dict(event="cache_observation", cache_hit=False, repeated_failure=False),
                 dict(event="cache_observation", cache_hit=True, repeated_failure=False),
                 dict(event="cache_observation", cache_hit=False, repeated_failure=True)]
        (directory / "search-trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in trace), encoding="utf-8")
        row = self.export.build_analysis(self.run)["search_runs"][0]
        self.assertEqual((row["attempts"], row["completed_attempts"], row["incomplete_attempts"]), (2, 1, 1))
        self.assertEqual((row["created_proposals"], row["cache_hits"], row["trace_events"]), (2, 1, 5))
        self.assertFalse(row["main_complete"])

    def test_fatal_last_oracle_attempt_never_creates_a_full_reference(self):
        self.job("search-t", completed=24)
        self.job("oracle", completed=440, kind="oracle", state="failed", fatal_last=True)
        analysis = self.export.build_analysis(self.run)
        oracle = analysis["oracles"][0]
        self.assertEqual((oracle["attempts"], oracle["completed_attempts"], oracle["finite"]), (440, 440, 439))
        self.assertFalse(oracle["complete"])
        self.assertIsNone(oracle["normalized_hypervolume"])
        self.assertIsNone(analysis["search_runs"][0]["oracle_gap"])
        self.assertGreater(oracle["partial_archive_hypervolume"], 0)

    def test_incomplete_common_byte_pair_is_visible_without_a_primary_delta(self):
        self.job("ready", mode="R", completed=15)
        self.job("tuned", mode="T", completed=1, charged=2, state="partial")
        analysis = self.export.build_analysis(self.run)
        row = next(row for row in analysis["common_bytes"] if row["model_seed"] == 19901
                   and row["budget"] == "attempts" and row["search_seed"] == 11 and row["tensor_bytes"] == 42280)
        self.assertFalse(row["main_complete"])
        self.assertEqual((row["R_attempts"], row["T_attempts"]), (15, 1))
        self.assertIsNotNone(row["CE_R"])
        self.assertIsNotNone(row["CE_T"])
        self.assertIsNone(row["delta_CE_T_minus_R"])

    def test_late_completion_cannot_improve_the_deadline_archive(self):
        self.job("tuned", mode="T", completed=24, last_at=11.)
        analysis = self.export.build_analysis(self.run)
        row = analysis["search_runs"][0]
        self.assertEqual((row["attempts"], row["completed_attempts"], row["on_time"], row["late"]), (24, 24, 23, 1))
        self.assertFalse(row["main_complete"])
        self.assertEqual(max(point["elapsed_parent_seconds"] for point in analysis["curves"]), .23)


if __name__ == "__main__":
    unittest.main()
