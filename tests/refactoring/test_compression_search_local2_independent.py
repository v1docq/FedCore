"""Independent, standard-library checks for the second issue-99 protocol.

Only selected pure declarations are loaded from source.  Importing this module
never imports Torch, NumPy, sklearn, GOLEM or an experimental worker.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import os
from pathlib import Path
import statistics
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def _load_pure(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError(f"Missing requested declarations: {names}")
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class TestFrozenLocal2Arithmetic(unittest.TestCase):
    def test_tensor_byte_formula_counts_biases_and_protected_classifier(self):
        # Each source layer includes its own bias; only the two weights shrink.
        shapes = ((64, 128), (128, 64), (64, 10))
        dense = 4 * sum(left * right + right for left, right in shapes)
        self.assertEqual(dense, 68_904)
        for rank in (4, 8, 16, 24, 32):
            for scope in (0, 1):
                expected = dense
                left, right = shapes[scope]
                expected += 4 * (rank * (left + right) - left * right)
                self.assertEqual(expected, 36_136 + 768 * rank)
                self.assertLessEqual(expected, 65_458)
            for other in (4, 8, 16, 24, 32):
                expected = 4 * ((64 + 128) * rank + 128
                                + (128 + 64) * other + 64 + 64 * 10 + 10)
                self.assertEqual(expected, 3_368 + 768 * (rank + other))
                self.assertLessEqual(expected, 65_458)

    def test_catalog_nesting_and_action_budgets(self):
        ready = {(method, 8, alpha) for method, alpha in (
            ("weighted", None), ("asvd", .5), ("afm", None))}
        tuned = {(method, rank, alpha) for rank in (4, 8, 16, 24, 32)
                 for method, alpha in (("weighted", None), ("asvd", .25),
                                       ("asvd", .5), ("afm", None))}
        self.assertLess(ready, tuned)
        self.assertEqual(2 * len(ready) + len(ready) ** 2, 15)
        self.assertEqual(2 * len(tuned) + len(tuned) ** 2, 440)
        parts = (9 * (15 + 24 + 24), 9 * (15 + 440 + 440), 3 * 440,
                 (54 + 3) * 3, 3 * (3 + 12), 3 + 3 + 54 + 3 + 3, 3)
        self.assertEqual(parts, (567, 8055, 1320, 171, 45, 66, 3))
        self.assertEqual(sum(parts), 10_227)
        self.assertEqual(10_300 - sum(parts), 73)
        self.assertEqual(3 * 40 * ((1078 + 63) // 64), 2040)

    def test_matrix_chain_requests_and_real_svd_calls_are_distinct(self):
        main_requests = main_svd = 0
        for _size in (8, 32, 128, 512, 1024):
            for invalid_count in (0, 2, 4, 6):
                for _block in range(6):
                    for arm in ("A", "B", "C"):
                        for request_id in range(8):
                            main_requests += 1
                            main_svd += arm == "B" or request_id >= invalid_count
        self.assertEqual((main_requests, main_svd), (2880, 2160))
        self.assertEqual(main_requests + 5 * 4 * 4 * 2 * 8 + 5 * 3 * 8 + 5 * 7, 4315)
        self.assertEqual(main_svd + 5 * 4 * 4 * 2 * 8 + 5 * 16 + 5 * 5, 3545)


class TestRealSearchBudgetWithoutML(unittest.TestCase):
    def setUp(self):
        values = _load_pure("fedcore/algorithm/low_rank/grammar_search.py",
                            ("_BudgetStop", "_SearchBudget"), {"dataclass": dataclass})
        self.budget_type, self.stop_type = values["_SearchBudget"], values["_BudgetStop"]

    def test_crossover_reservation_is_atomic_and_cannot_spend_one_remaining_slot(self):
        budget = self.budget_type(24, 5, 100.)
        self.assertEqual(budget.reserve_proposals(2, 0.), (1, 2))
        self.assertEqual(budget.reserve_proposals(2, 0.), (3, 4))
        with self.assertRaises(self.stop_type):
            budget.reserve_proposals(2, 0.)
        self.assertEqual(budget.proposals, 4)
        self.assertEqual(budget.attempts, 0)
        self.assertEqual(budget.stopped_by, "proposal_budget")

    def test_reserved_children_can_be_evaluated_after_reaching_proposal_cap(self):
        budget = self.budget_type(24, 2, 100.)
        budget.reserve_proposals(2, 0.)
        self.assertEqual(budget.charge_evaluation(0.), 1)
        self.assertEqual(budget.charge_evaluation(0.), 2)
        self.assertEqual(budget.reason(0.), "proposal_budget")

    def test_attempt_and_parent_deadline_guards_run_before_new_work(self):
        budget = self.budget_type(1, 20, 10.)
        self.assertEqual(budget.charge_evaluation(9.), 1)
        with self.assertRaises(self.stop_type):
            budget.charge_evaluation(9.)
        self.assertEqual(budget.attempts, 1)
        expired = self.budget_type(24, 20, 10.)
        with self.assertRaises(self.stop_type):
            expired.charge_evaluation(10.)
        self.assertEqual(expired.attempts, 0)


class TestHistoricalEvaluationDefaults(unittest.TestCase):
    def test_explicit_new_profile_preserves_historical_default_normalization(self):
        values = _load_pure("fedcore/experiments/compression_search_evaluation.py",
                            ("EvaluationProfile",), {"dataclass": dataclass,
                            "DENSE_TENSOR_BYTES": 49_416, "G3_TENSOR_BYTE_BUDGET": 46_945})
        profile_type = values["EvaluationProfile"]
        old, new = profile_type(), profile_type(68_904, 65_458, 10)
        self.assertEqual((old.dense_tensor_bytes, old.tensor_byte_budget, old.class_count),
                         (49_416, 46_945, 2))
        self.assertEqual((new.dense_tensor_bytes, new.tensor_byte_budget, new.class_count),
                         (68_904, 65_458, 10))
        for args in ((0, 65_458, 10), (68_904, 0, 10), (68_904, 65_458, 1),
                     (True, 65_458, 10), (68_904, 65_458, 10.)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                profile_type(*args)


class TestMatrixClaimGatesWithoutML(unittest.TestCase):
    orders = ("ABC", "BCA", "CAB", "ACB", "CBA", "BAC")

    def setUp(self):
        values = _load_pure("fedcore/experiments/compression_search_matrix.py",
                            ("summarize_matrix",), {"statistics": statistics,
                            "SIZES": (8,), "INVALID_COUNTS": (4,), "ARM_ORDERS": self.orders})
        self.summarize = values["summarize_matrix"]

    def complete_streams(self, baseline=.1):
        streams = []
        for block in range(6):
            for arm in "ABC":
                streams.append(dict(n=8, invalid_count=4, group="timed", block=block,
                                    arm=arm, request_count=8,
                                    primary_seconds=baseline * (1 if arm == "B" else .8)))
        for pair in range(4):
            for label in ("B1", "B2"):
                streams.append(dict(n=8, invalid_count=4, group="control", pair=pair,
                                    label=label, request_count=8, primary_seconds=baseline))
        return streams

    def test_complete_stable_observations_can_pass_all_frozen_gates(self):
        rows = self.summarize(self.complete_streams())
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["eligible"] and row["observed_win"] for row in rows))

    def test_incomplete_abc_block_cannot_supply_a_complete_ab_pair(self):
        streams = self.complete_streams()
        streams = [row for row in streams if not (
            row["group"] == "timed" and row["block"] == 5 and row["arm"] == "C")]
        rows = self.summarize(streams)
        self.assertTrue(all(not row["eligible"] and not row["observed_win"] for row in rows))

    def test_minimum_duration_and_bb_controls_are_mandatory(self):
        self.assertTrue(all(not row["eligible"] for row in self.summarize(self.complete_streams(.049))))
        self.assertTrue(all(row["eligible"] for row in self.summarize(self.complete_streams(.05))))
        streams = self.complete_streams()
        for row in streams:
            if row["group"] == "control" and row["label"] == "B1":
                row["primary_seconds"] = .094
        self.assertTrue(all(not row["bb_pass"] and not row["observed_win"]
                            for row in self.summarize(streams)))

    def test_order_imbalance_remains_diagnostic_even_with_fast_median(self):
        streams = self.complete_streams()
        for row in streams:
            if row["group"] == "timed" and row["arm"] == "A":
                row["primary_seconds"] = .07 if self.orders[row["block"]].index("A") < self.orders[row["block"]].index("B") else .09
        a = next(row for row in self.summarize(streams) if row["arm"] == "A")
        self.assertFalse(a["order_pass"])
        self.assertFalse(a["observed_win"])


class TestJsonPublicationWithoutWorkers(unittest.TestCase):
    def clock(self):
        state = SimpleNamespace(now=0., sleeps=[])
        def sleep(seconds):
            state.sleeps.append(seconds)
            state.now += seconds
        return SimpleNamespace(monotonic=lambda: state.now, sleep=sleep), state

    def writer(self, replace, clock, dumps=json.dumps):
        values = _load_pure("fedcore/experiments/compression_search_local2.py",
            ("write_json",), dict(Path=Path, json=SimpleNamespace(dumps=dumps),
                                  os=SimpleNamespace(replace=replace), time=clock))
        return values["write_json"]

    def test_transient_permission_failure_publishes_identical_bytes_without_rewriting(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            target = Path(temporary)/"active-request.json"
            target.write_text('{"active": false}\n', encoding="utf-8")
            pending = target.with_name(target.name+".pending")
            payload = dict(active=True, global_attempt=18, job="matrix-8", word="LD")
            attempts, serializations, writes = [], [], []
            clock, state = self.clock()
            def dumps(value, **kwargs):
                serialized = json.dumps(value, **kwargs)
                serializations.append(serialized)
                return serialized
            def replace(source, destination):
                self.assertEqual((Path(source), Path(destination)), (pending, target))
                attempts.append(pending.read_bytes())
                self.assertEqual(json.loads(attempts[-1]), payload)
                self.assertEqual(attempts[-1], attempts[0])
                self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"active": False})
                if len(attempts) < 3:
                    raise PermissionError("simulated Windows sharing denial")
                os.replace(source, destination)
            original_write = Path.write_text
            def write_text(path, *args, **kwargs):
                writes.append(path)
                return original_write(path, *args, **kwargs)
            with patch.object(Path, "write_text", write_text):
                self.writer(replace, clock, dumps)(target, payload)
            self.assertEqual(len(attempts), 3)
            self.assertEqual(len(serializations), 1)
            self.assertEqual(writes, [pending])
            self.assertEqual(state.sleeps, [.01, .01])
            self.assertEqual(target.read_bytes(), attempts[0])
            self.assertFalse(pending.exists())

    def test_persistent_permission_failure_is_bounded_and_preserves_original_exception(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            target = Path(temporary)/"active-request.json"
            target.write_text('{"active": false}\n', encoding="utf-8")
            first = PermissionError("original sharing denial")
            clock, state = self.clock()
            attempts = []
            def replace(source, destination):
                attempts.append(state.now)
                if len(attempts) == 1:
                    raise first
                raise PermissionError("later sharing denial")
            with self.assertRaises(PermissionError) as caught:
                self.writer(replace, clock)(target, {"active": True})
            self.assertIs(caught.exception, first)
            self.assertGreater(len(attempts), 1)
            self.assertLessEqual(len(attempts), 51)
            self.assertTrue(all(at < .5 for at in attempts))
            self.assertTrue(all(0 < delay <= .01 for delay in state.sleeps))
            self.assertAlmostEqual(state.now, .5)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"active": False})
            self.assertEqual(json.loads(target.with_name(target.name+".pending").read_text(encoding="utf-8")),
                             {"active": True})

    def test_other_os_errors_propagate_without_retry_or_sleep(self):
        errors = (OSError("I/O failure"), FileNotFoundError("pending missing"),
                  IsADirectoryError("target is a directory"))
        for error in errors:
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory(dir=ROOT) as temporary:
                target = Path(temporary)/"active-request.json"
                target.write_text('{"active": false}\n', encoding="utf-8")
                attempts = []
                def replace(source, destination):
                    attempts.append((source, destination))
                    raise error
                clock = SimpleNamespace(
                    monotonic=lambda: self.fail("Non-permission failure must not enter the retry clock"),
                    sleep=lambda seconds: self.fail("Non-permission failure must not sleep"))
                with self.assertRaises(type(error)) as caught:
                    self.writer(replace, clock)(target, {"active": True})
                self.assertIs(caught.exception, error)
                self.assertEqual(len(attempts), 1)
                self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"active": False})


class TestJsonReadingWithoutWorkers(unittest.TestCase):
    def clock(self):
        state = SimpleNamespace(now=0., sleeps=[])
        def sleep(seconds):
            state.sleeps.append(seconds)
            state.now += seconds
        return SimpleNamespace(monotonic=lambda: state.now, sleep=sleep), state

    def reader(self, read_text, clock):
        values = _load_pure("fedcore/experiments/compression_search_local2.py",
            ("read_json",), dict(Path=lambda path: SimpleNamespace(read_text=read_text),
                                json=json, time=clock))
        return values["read_json"]

    def test_transient_permission_failure_returns_complete_json(self):
        attempts = []
        clock, state = self.clock()
        payload = {"active": True, "hard_deadline": 123., "details": {"word": "LD"}}
        def read_text(**kwargs):
            attempts.append(kwargs)
            if len(attempts) < 3:
                raise PermissionError("simulated Windows read denial")
            return json.dumps(payload)
        self.assertEqual(self.reader(read_text, clock)("active-request.json"), payload)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(state.sleeps, [.01, .01])

    def test_persistent_permission_failure_is_bounded_and_preserves_original_exception(self):
        attempts = []
        clock, state = self.clock()
        first = PermissionError("original read denial")
        def read_text(**kwargs):
            attempts.append(state.now)
            if len(attempts) == 1:
                raise first
            raise PermissionError("later read denial")
        with self.assertRaises(PermissionError) as caught:
            self.reader(read_text, clock)("active-request.json")
        self.assertIs(caught.exception, first)
        self.assertGreater(len(attempts), 1)
        self.assertLessEqual(len(attempts), 51)
        self.assertTrue(all(at < .5 for at in attempts))
        self.assertTrue(all(0 < delay <= .01 for delay in state.sleeps))
        self.assertAlmostEqual(state.now, .5)

    def test_corrupt_json_is_not_retried(self):
        attempts = []
        def read_text(**kwargs):
            attempts.append(kwargs)
            return '{"active":'
        clock = SimpleNamespace(
            monotonic=lambda: self.fail("Corrupt JSON must not enter the retry clock"),
            sleep=lambda seconds: self.fail("Corrupt JSON must not sleep"))
        with self.assertRaises(json.JSONDecodeError):
            self.reader(read_text, clock)("active-request.json")
        self.assertEqual(len(attempts), 1)

    def test_other_read_os_errors_propagate_without_retry(self):
        for error in (OSError("I/O failure"), FileNotFoundError("missing JSON")):
            with self.subTest(error=type(error).__name__):
                attempts = []
                def read_text(**kwargs):
                    attempts.append(kwargs)
                    raise error
                clock = SimpleNamespace(
                    monotonic=lambda: self.fail("Other read errors must not enter the retry clock"),
                    sleep=lambda seconds: self.fail("Other read errors must not sleep"))
                with self.assertRaises(type(error)) as caught:
                    self.reader(read_text, clock)("result.json")
                self.assertIs(caught.exception, error)
                self.assertEqual(len(attempts), 1)


class TestParentWatchdogWithoutWorkers(unittest.TestCase):
    def test_status_monitor_failure_kills_owned_processes_and_saves_resources(self):
        for failure in ("permission", "decode", "other-os"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(dir=ROOT) as temporary:
                output = Path(temporary)
                (output/"jobs").mkdir()
                state = SimpleNamespace(now=0., sleeps=[])
                def sleep(seconds):
                    state.sleeps.append(seconds)
                    state.now += seconds
                clock = SimpleNamespace(perf_counter=lambda: 100.+state.now,
                                        monotonic=lambda: state.now, sleep=sleep)
                process = SimpleNamespace(pid=123, returncode=None, poll=lambda: process.returncode,
                                          wait=lambda timeout: process.returncode)
                killed = []
                def kill_worker():
                    killed.append("worker")
                    process.returncode = -9
                child = SimpleNamespace(is_running=lambda: True,
                    memory_info=lambda: SimpleNamespace(rss=64), kill=lambda: killed.append("child"))
                owned = SimpleNamespace(children=lambda recursive: [child], is_running=lambda: True,
                    memory_info=lambda: SimpleNamespace(rss=128), kill=kill_worker)
                fake_psutil = SimpleNamespace(Process=lambda pid: owned,
                    virtual_memory=lambda: SimpleNamespace(available=8*1024**3),
                    NoSuchProcess=type("NoSuchProcess", (Exception,), {}),
                    AccessDenied=type("AccessDenied", (Exception,), {}))
                def spawn(arguments, **kwargs):
                    directory = Path(arguments[-1]).parent
                    (output/"matrix-attempts.jsonl").write_text('{"global_attempt":1}\n', encoding="utf-8")
                    telemetry = bytearray(64)
                    struct.pack_into("<QQdd", telemetry, 0, 1, 0, 100., 0.)
                    (directory/"svd-telemetry.bin").write_bytes(telemetry)
                    text = '{"active":' if failure == "decode" else '{"active":true,"hard_deadline":120}'
                    (directory/"active-request.json").write_text(text, encoding="utf-8")
                    return process
                values = _load_pure("fedcore/experiments/compression_search_local2.py",
                    ("write_json", "read_json", "count_lines", "_telemetry", "run_job"),
                    dict(Path=Path, json=json, os=os, struct=struct, time=clock,
                        source_hashes=lambda: {}, REPO=output, MODULE="mock.worker", sys=sys,
                        subprocess=SimpleNamespace(Popen=spawn, CREATE_NO_WINDOW=0)))
                original_read = Path.read_text
                status_reads = []
                def read_text(path, *args, **kwargs):
                    if path.name == "active-request.json":
                        status_reads.append(state.now)
                        if failure == "permission":
                            raise PermissionError("status sharing denial")
                        if failure == "other-os":
                            raise OSError("status I/O failure")
                    return original_read(path, *args, **kwargs)
                protocol = dict(source_files={}, protocol_sha256="p", source_sha256="s")
                with patch.dict(sys.modules, {"psutil": fake_psutil}), patch.object(Path, "read_text", read_text):
                    result = values["run_job"](output, dict(job="matrix-8", kind="matrix", n=8,
                        hard_seconds=20), protocol, 0., 999.)
                self.assertEqual((result["status"], result["stop"], result["returncode"]),
                                 ("failed", "status_monitor_io_failed", -9))
                self.assertEqual(killed, ["child", "worker"])
                self.assertIn({"permission": "PermissionError", "decode": "JSONDecodeError",
                               "other-os": "OSError"}[failure], result["error"])
                self.assertEqual((result["svd_telemetry"]["started"], result["svd_telemetry"]["completed"]), (1, 0))
                self.assertEqual(len((output/"matrix-attempts.jsonl").read_text().splitlines()), 1)
                saved = json.loads((output/"jobs"/"matrix-8"/"resources.json").read_text(encoding="utf-8"))
                self.assertEqual(saved, result)
                if failure == "permission":
                    self.assertGreater(len(status_reads), 1)
                    self.assertAlmostEqual(state.now, .5)
                else:
                    self.assertEqual(len(status_reads), 1)
                    self.assertEqual(state.sleeps, [])

    def test_killed_native_call_keeps_the_charge_and_its_started_telemetry(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            output = Path(temporary)
            (output / "jobs").mkdir()
            ticks = iter((100., 100., 102., 102.))
            clock = SimpleNamespace(perf_counter=lambda: next(ticks, 102.),
                                    sleep=lambda seconds: self.fail("Expired worker must be killed immediately"))
            process = SimpleNamespace(pid=123, returncode=None, poll=lambda: process.returncode,
                                      wait=lambda timeout: process.returncode)
            def killed():
                process.returncode = -9
            owned = SimpleNamespace(children=lambda recursive: [], is_running=lambda: True,
                                    memory_info=lambda: SimpleNamespace(rss=128), kill=killed)
            fake_psutil = SimpleNamespace(Process=lambda pid: owned,
                virtual_memory=lambda: SimpleNamespace(available=8 * 1024**3),
                NoSuchProcess=type("NoSuchProcess", (Exception,), {}),
                AccessDenied=type("AccessDenied", (Exception,), {}))
            def spawn(arguments, **kwargs):
                # Emulate one admitted native call that is still running. No
                # operating-system process and no numerical runtime is started.
                job_path = Path(arguments[-1])
                directory = job_path.parent
                (output / "matrix-attempts.jsonl").write_text('{"global_attempt":1}\n', encoding="utf-8")
                telemetry = bytearray(64)
                struct.pack_into("<QQdd", telemetry, 0, 1, 0, 100.5, 0.)
                (directory / "svd-telemetry.bin").write_bytes(telemetry)
                return process
            values = _load_pure("fedcore/experiments/compression_search_local2.py",
                ("write_json", "read_json", "count_lines", "_telemetry", "run_job"),
                dict(Path=Path, json=json, os=os, struct=struct, time=clock,
                    source_hashes=lambda: {}, REPO=output, MODULE="mock.worker", sys=sys,
                    subprocess=SimpleNamespace(Popen=spawn, CREATE_NO_WINDOW=0)))
            protocol = dict(source_files={}, protocol_sha256="p", source_sha256="s")
            with patch.dict(sys.modules, {"psutil": fake_psutil}):
                result = values["run_job"](output, dict(job="matrix-8", kind="matrix", n=8,
                    hard_seconds=1), protocol, 0., 999.)
            self.assertEqual((result["status"], result["stop"], result["returncode"]),
                             ("partial", "hard_deadline", -9))
            self.assertEqual((result["svd_telemetry"]["started"], result["svd_telemetry"]["completed"]), (1, 0))
            self.assertEqual(len((output / "matrix-attempts.jsonl").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
