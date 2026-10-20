"""Frozen CPU matrix witness for issue 99, second local package.

Schedules and analysis import only the standard library. ML imports occur in
the admitted child process. The three timing arms share the same interpreter.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import mmap
from pathlib import Path
import random
import statistics
import struct
import time

SIZES = (8, 32, 128, 512, 1024)
INVALID_COUNTS = (0, 2, 4, 6)
ARM_ORDERS = ("ABC", "BCA", "CAB", "ACB", "CBA", "BAC")
SIZE_SECONDS = dict(zip(SIZES, (60, 60, 120, 360, 900)))
MATRIX_ATTEMPT_LIMIT = 4350


def request_tape(invalid_count, seed):
    if invalid_count not in INVALID_COUNTS:
        raise ValueError("Use a frozen invalid count")
    order = list(range(8))
    random.Random(seed).shuffle(order)
    return tuple((index, "LL" if index < invalid_count else "LD") for index in order)


def stream_schedule(n):
    if n not in SIZES:
        raise ValueError("Use a frozen matrix size")
    rows = [dict(group="warmup", arm=arm, invalid_count=4, block=-1,
                 pair=None, label=None, tape_seed=9930) for arm in "ABC"]
    for pair in (0, 1):
        for count in INVALID_COUNTS:
            labels = ("B1", "B2") if pair == 0 else ("B2", "B1")
            rows.extend(dict(group="control", arm="B", invalid_count=count,
                             block=-1, pair=pair, label=label, tape_seed=9940+pair)
                        for label in labels)
    for block, arms in enumerate(ARM_ORDERS):
        shift = block % 4
        counts = INVALID_COUNTS[shift:] + INVALID_COUNTS[:shift]
        for count in counts:
            rows.extend(dict(group="timed", arm=arm, invalid_count=count,
                             block=block, pair=None, label=None, tape_seed=9931+block)
                        for arm in arms)
    for pair in (2, 3):
        for count in INVALID_COUNTS:
            labels = ("B2", "B1") if pair == 2 else ("B1", "B2")
            rows.extend(dict(group="control", arm="B", invalid_count=count,
                             block=-1, pair=pair, label=label, tape_seed=9940+pair)
                        for label in labels)
    return tuple(dict(row, n=n, stream_id=f"n{n}-stream-{i:03d}") for i, row in enumerate(rows))


def _append(path, value):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False)+"\n")
        stream.flush()


def _write(path, value):
    from .compression_search_local2 import write_json
    write_json(path, value)


class MatrixDeadline(Exception):
    """Expected partial execution at a frozen admission boundary."""


class MatrixLedger:
    def __init__(self, job):
        self.job = job
        self.path = Path(job["output"]) / "matrix-attempts.jsonl"
        self.count = len(self.path.read_text(encoding="utf-8").splitlines()) if self.path.exists() else 0

    def charge(self, **metadata):
        now = time.perf_counter()
        deadline = min(self.job["compute_admission_deadline"],
                       self.job.get("request_admission_deadline", self.job["admission_deadline"]))
        if now >= deadline or self.count >= MATRIX_ATTEMPT_LIMIT:
            raise MatrixDeadline("Matrix request admission budget exhausted")
        self.count += 1
        row = dict(metadata, global_attempt=self.count, job=self.job["job"], started_at=now)
        _append(self.path, row)
        _write(Path(self.job["directory"])/"active-request.json",
               dict(row, hard_deadline=min(now+20, self.job["admission_deadline"]), active=True))
        return row


class SVDTelemetry:
    """Memory mapped starts survive a killed LAPACK call, without timed disk flush.

    The file contains <started:uint64, completed:uint64, last-start:f64,
    last-completion:f64>. Only the child writes. A parent reads after termination.
    The same Python instrumentation is inside all arms; no cost is subtracted.
    """
    def __init__(self, directory):
        self.path = Path(directory)/"svd-telemetry.bin"
        self.stream = self.path.open("w+b")
        self.stream.write(bytes(64))
        self.stream.flush()
        self.mapping = mmap.mmap(self.stream.fileno(), 64, access=mmap.ACCESS_WRITE)
        self.started = self.completed = 0
        self.events = []

    def __enter__(self):
        import torch
        self.original = torch.linalg.svd
        def measured(*args, **kwargs):
            self.started += 1
            start = time.perf_counter()
            struct.pack_into("<Q", self.mapping, 0, self.started)
            struct.pack_into("<d", self.mapping, 16, start)
            native_start = time.perf_counter()
            try:
                return self.original(*args, **kwargs)
            finally:
                finish = time.perf_counter()
                self.completed += 1
                struct.pack_into("<Q", self.mapping, 8, self.completed)
                struct.pack_into("<d", self.mapping, 24, finish)
                self.events.append(dict(index=self.started, started_at=start,
                                        completed_at=finish, native_call_seconds=finish-native_start))
        torch.linalg.svd = measured
        return self

    def __exit__(self, *exc):
        import torch
        torch.linalg.svd = self.original
        self.mapping.flush()
        self.mapping.close()
        self.stream.close()


def _rng_identity():
    import pickle
    import numpy as np
    import torch
    value = pickle.dumps((random.getstate(), np.random.get_state()), protocol=4)
    return hashlib.sha256(value+torch.get_rng_state().numpy().tobytes()).hexdigest()


def matrix_source(n):
    import torch
    from torch import nn
    # Constructors also consume RNG. Keep the whole construction isolated.
    with torch.random.fork_rng(devices=[]):
        model = nn.Sequential(nn.Linear(n, n, bias=True, dtype=torch.float64)).eval()
        generator = torch.Generator().manual_seed(9900+n)
        with torch.no_grad():
            model[0].weight.copy_(torch.randn(n, n, generator=generator, dtype=torch.float64)/(n**.5))
            model[0].bias.zero_()
    return model


def matrix_request(n, word):
    from fedcore.algorithm.low_rank.compression_composition import (
        CompositionRecipe, CompositionRequest, LowRank, MaterializeDense)
    if word not in ("LD", "LL"):
        raise ValueError("Only the frozen common-prefix chains are allowed")
    last = MaterializeDense() if word == "LD" else LowRank(n//4)
    return CompositionRequest((CompositionRecipe("0", (LowRank(n//4), last)),),
                              max_peak_bytes=512*1024**2)


def invoke_arm(model, request, arm):
    from fedcore.algorithm.low_rank.compression_composition import prepare_composition
    from fedcore.algorithm.low_rank.plans import ValidationFailure
    from .compression_composition_reference import execute_reference, inspect_contract
    check_start = time.perf_counter()
    checked = prepare_composition(model, request) if arm == "A" else (
        inspect_contract(request, model) if arm == "C" else None)
    check_seconds = time.perf_counter()-check_start if arm != "B" else 0.
    rejected = isinstance(checked, ValidationFailure) if arm == "A" else (
        not checked.accepted if arm == "C" else False)
    if rejected:
        if arm == "A":
            errors = [asdict(item) for item in checked.violations]
        else:
            errors = [asdict(checked.rejection)]
        return None, dict(accepted=False, check_seconds=check_seconds,
                          execution_seconds=0., early_rejection=errors,
                          decompositions_started=0, completed_steps=0)
    actual = execute_reference(model, request)
    return actual, dict(accepted=actual.accepted, check_seconds=check_seconds,
                        execution_seconds=actual.timings.total_seconds,
                        stage_timings=asdict(actual.timings),
                        decompositions_started=actual.decompositions_started,
                        completed_steps=actual.completed_steps,
                        rejection=asdict(actual.rejection) if actual.rejection else None,
                        tensor_bytes=actual.storage_bytes,
                        final_states=[(path, asdict(state)) for path, state in actual.final_states])


def _check_result(actual, row, word, n, expected, model, identity, rng):
    import torch
    from .compression_search_evaluation import model_identity
    if model_identity(model) != identity or _rng_identity() != rng:
        raise RuntimeError("Matrix request mutated source or RNG")
    if row["accepted"] != (word == "LD"):
        raise RuntimeError("Matrix admission/execution disagrees with frozen contract")
    if word == "LL":
        errors = row.get("early_rejection") or [row["rejection"]]
        if not errors or any(error["code"] != "UnsupportedTransition" for error in errors):
            raise RuntimeError("LL did not produce its expected structural rejection")
        if not all("steps[1]" in error["path"] or error.get("step_index") == 1 for error in errors):
            raise RuntimeError("LL rejection did not identify its second step")
        return
    if row["tensor_bytes"] != 8*(n*n+n):
        raise RuntimeError("LD final storage did not match dense bytes")
    if expected is not None:
        torch.testing.assert_close(actual.model[0].weight, expected["weight"], rtol=2e-12, atol=2e-12)
        torch.testing.assert_close(actual.model[0].bias, expected["bias"], rtol=0, atol=0)
        probe = torch.arange(3*n, dtype=torch.float64).reshape(3, n)/n
        torch.testing.assert_close(actual.model(probe), expected["output"], rtol=2e-12, atol=2e-12)


def _request_call(job, ledger, telemetry, model, request, metadata, expected=None):
    from .compression_search_evaluation import model_identity
    identity, rng = model_identity(model), _rng_identity()
    admission = ledger.charge(**metadata)
    first_svd = telemetry.started
    event_offset = len(telemetry.events)
    start = time.perf_counter()
    actual, row = invoke_arm(model, request, metadata["arm"])
    end = time.perf_counter()
    row.update(metadata, global_attempt=admission["global_attempt"],
               started_at=start, completed_at=end, call_seconds=end-start,
               observed_svd_starts=telemetry.started-first_svd,
               native_svd_events=telemetry.events[event_offset:])
    _check_result(actual, row, metadata["word"], metadata["n"], expected, model, identity, rng)
    expected_svd = int(metadata["word"] == "LD" or metadata["arm"] == "B")
    if row["observed_svd_starts"] != expected_svd or row["decompositions_started"] != expected_svd:
        raise RuntimeError("Actual SVD calls disagree with reference and early-rejection law")
    row["output_identity"] = model_identity(actual.model) if actual and actual.accepted else None
    _append(Path(job["directory"])/"matrix-events.jsonl", row)
    _write(Path(job["directory"])/"active-request.json", dict(admission, active=False))
    return actual, row


def run_matrix_admission(job):
    import torch
    from fedcore.algorithm.low_rank.compression_composition import (
        CompositionSuccess, execute_composition, prepare_composition)
    from .compression_composition_reference import assert_reference_parity
    from .compression_search_evaluation import model_identity
    directory = Path(job["output"])/"matrix-sources"
    directory.mkdir(exist_ok=True)
    ledger, rows, sources = MatrixLedger(job), [], []
    with SVDTelemetry(job["directory"]) as telemetry:
        for n in SIZES:
            model = matrix_source(n)
            identity, rng = model_identity(model), _rng_identity()
            source_path = directory/f"source-{n}.pt"
            torch.save(model.state_dict(), source_path)
            reference = None
            for word in ("LD", "LL"):
                request = matrix_request(n, word)
                for arm in "ABC":
                    actual, row = _request_call(job, ledger, telemetry, model, request,
                        dict(n=n, word=word, arm=arm, group="admission", stream_id=f"gate-{n}", request_index=len(rows)))
                    rows.append(row)
                    if word == "LD":
                        if reference is None:
                            reference = actual
                        else:
                            assert_reference_parity(reference, actual.model, request, rtol=2e-12, atol=2e-12)
            admission = ledger.charge(n=n, word="LD", arm="production", group="admission")
            start_svd, start = telemetry.started, time.perf_counter()
            checked = prepare_composition(model, matrix_request(n, "LD"))
            production = execute_composition(model, checked)
            if not isinstance(production, CompositionSuccess):
                raise RuntimeError(f"Production LD failed: {production}")
            assert_reference_parity(reference, production.model, matrix_request(n, "LD"), rtol=2e-12, atol=2e-12)
            if telemetry.started-start_svd != 1 or model_identity(model) != identity or _rng_identity() != rng:
                raise RuntimeError("Production gate changed source/RNG or SVD accounting")
            probe = torch.arange(3*n, dtype=torch.float64).reshape(3, n)/n
            expected_path = directory/f"expected-{n}.pt"
            torch.save(dict(weight=reference.model[0].weight.detach(), bias=reference.model[0].bias.detach(),
                            output=reference.model(probe).detach()), expected_path)
            row = dict(n=n, word="LD", arm="production", group="admission", accepted=True,
                       global_attempt=admission["global_attempt"], observed_svd_starts=1,
                       checkpoint_bytes=len(production.checkpoint_bytes),
                       checkpoint_sha256=hashlib.sha256(production.checkpoint_bytes).hexdigest(),
                       tensor_bytes=production.storage_bytes, call_seconds=time.perf_counter()-start)
            _append(Path(job["directory"])/"matrix-events.jsonl", row)
            _write(Path(job["directory"])/"active-request.json", dict(admission, active=False))
            rows.append(row)
            sources.append(dict(n=n, identity=identity, seed=9900+n,
                source_file=source_path.name, source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                expected_file=expected_path.name, expected_sha256=hashlib.sha256(expected_path.read_bytes()).hexdigest()))
    _write(directory/"manifest.json", sources)
    return dict(rows=rows, sources=sources, matrix_attempts=len(rows), observed_svd_starts=telemetry.started)


def run_matrix_job(job):
    import torch
    from .compression_search_evaluation import model_identity
    n = job["n"]
    folder = Path(job["output"])/"matrix-sources"
    sources = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    source = next(item for item in sources if item["n"] == n)
    for name in ("source", "expected"):
        if hashlib.sha256((folder/source[name+"_file"]).read_bytes()).hexdigest() != source[name+"_sha256"]:
            raise RuntimeError("Frozen matrix source changed")
    model = matrix_source(n)
    model.load_state_dict(torch.load(folder/source["source_file"], weights_only=True), strict=True)
    if model_identity(model) != source["identity"]:
        raise RuntimeError("Matrix source identity changed on reload")
    expected = torch.load(folder/source["expected_file"], weights_only=True)
    ledger, streams = MatrixLedger(job), []
    partial = None
    requests = {word: matrix_request(n, word) for word in ("LD", "LL")}
    with SVDTelemetry(job["directory"]) as telemetry:
        try:
            for metadata in stream_schedule(n):
                started, rows = time.perf_counter(), []
                for index, (request_id, word) in enumerate(request_tape(metadata["invalid_count"], metadata["tape_seed"])):
                    _, row = _request_call(job, ledger, telemetry, model, requests[word],
                        dict(metadata, request_index=index, tape_id=request_id, word=word), expected)
                    rows.append(row)
                stream = dict(metadata, primary_seconds=sum(row["call_seconds"] for row in rows),
                    stream_seconds=time.perf_counter()-started, started_at=started, completed_at=time.perf_counter(),
                    request_count=len(rows), observed_svd_starts=sum(row["observed_svd_starts"] for row in rows))
                _append(Path(job["directory"])/"matrix-streams.jsonl", stream)
                streams.append(stream)
        except MatrixDeadline as error:
            partial = str(error)
    return dict(n=n, streams=streams, partial=partial, source=source,
                observed_svd_starts=telemetry.started, svd_completed=telemetry.completed,
                scheduled_streams=len(stream_schedule(n)), completed_streams=len(streams))


def summarize_matrix(streams):
    """Exact engineering gates; missing blocks never become zero-time samples."""
    result = []
    for n in SIZES:
        for invalid in INVALID_COUNTS:
            cell = [row for row in streams if row["n"] == n and row["invalid_count"] == invalid]
            controls = {(row["pair"], row["label"]): row for row in cell if row["group"] == "control"}
            control_ratios = [controls[(pair,"B1")]["primary_seconds"]/controls[(pair,"B2")]["primary_seconds"]
                              for pair in range(4) if (pair,"B1") in controls and (pair,"B2") in controls]
            bb_pass = len(control_ratios) == 4 and .95 <= statistics.median(control_ratios) <= 1.05 and all(.85 <= v <= 1.15 for v in control_ratios)
            timed = {(row["block"], row["arm"]): row for row in cell if row["group"] == "timed" and row.get("request_count", 8) == 8}
            complete_blocks = {block for block in range(6) if all((block, arm) in timed for arm in "ABC")}
            for arm in "AC":
                paired = [(block, timed[(block,arm)], timed[(block,"B")]) for block in range(6)
                          if block in complete_blocks]
                ratios = [a["primary_seconds"]/b["primary_seconds"] for _,a,b in paired]
                before = [a["primary_seconds"]/b["primary_seconds"] for block,a,b in paired if ARM_ORDERS[block].index(arm) < ARM_ORDERS[block].index("B")]
                after = [a["primary_seconds"]/b["primary_seconds"] for block,a,b in paired if ARM_ORDERS[block].index(arm) > ARM_ORDERS[block].index("B")]
                order_ratio = statistics.median(before)/statistics.median(after) if len(before) == len(after) == 3 else None
                median_b = statistics.median([b["primary_seconds"] for _,_,b in paired]) if paired else None
                median_ratio = statistics.median(ratios) if ratios else None
                order_pass = order_ratio is not None and .9 <= order_ratio <= 1.1
                duration_pass = median_b is not None and median_b >= .05
                eligible = len(paired) == 6 and bb_pass and order_pass and duration_pass
                win = eligible and median_ratio <= .9 and sum(v < 1 for v in ratios) >= 5
                result.append(dict(n=n, invalid_count=invalid, invalid_fraction=invalid/8, arm=arm,
                    pairs=len(paired), ratios=ratios, median_ratio=median_ratio,
                    min_ratio=min(ratios) if ratios else None, max_ratio=max(ratios) if ratios else None,
                    median_absolute_seconds=statistics.median([a["primary_seconds"]-b["primary_seconds"] for _,a,b in paired]) if paired else None,
                    bb_ratios=control_ratios, bb_pass=bb_pass, order_ratio=order_ratio,
                    order_pass=order_pass, median_b_seconds=median_b, duration_pass=duration_pass,
                    eligible=eligible, observed_win=win))
    return result
