"""New complete matrix sessions; old local-2 measurements are never amended."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from .compression_search_matrix import (
    MatrixDeadline, MatrixLedger, SVDTelemetry, _append, _request_call,
    _rng_identity, _write, matrix_request, matrix_source, request_tape,
    stream_schedule, summarize_matrix)


def session_schedule(n, session_id):
    return tuple(dict(row, session_id=session_id,
                      stream_id=session_id+":"+row["stream_id"]) for row in stream_schedule(n))


def run_admission(job):
    import torch
    from fedcore.algorithm.low_rank.compression_composition import (
        CompositionSuccess, execute_composition, prepare_composition)
    from .compression_composition_reference import assert_reference_parity
    from .compression_search_evaluation import model_identity
    historical = Path(job["matrix_source_root"])
    prior = json.loads((historical/"manifest.json").read_text(encoding="utf-8"))
    destination = Path(job["output"])/"matrix-sources"
    destination.mkdir(exist_ok=False)
    ledger, rows, sources = MatrixLedger(job), [], []
    with SVDTelemetry(job["directory"]) as telemetry:
        for n in job["sizes"]:
            record = next(row for row in prior if row["n"] == n)
            raw = (historical/record["source_file"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != record["source_sha256"]:
                raise RuntimeError("Historical matrix source hash changed")
            model = matrix_source(n)
            model.load_state_dict(torch.load(historical/record["source_file"], weights_only=True), strict=True)
            identity, rng = model_identity(model), _rng_identity()
            if identity != record["identity"]:
                raise RuntimeError("Historical matrix tensor identity changed")
            source_path = destination/record["source_file"]
            source_path.write_bytes(raw)
            reference = None
            for word in ("LD", "LL"):
                request = matrix_request(n, word)
                for arm in "ABC":
                    actual, row = _request_call(job, ledger, telemetry, model, request,
                        dict(n=n, word=word, arm=arm, group="admission",
                             session_id=job["session_id"], stream_id=f"gate-{n}", request_index=len(rows)))
                    rows.append(row)
                    if word == "LD":
                        if reference is None:
                            reference = actual
                        else:
                            assert_reference_parity(reference, actual.model, request, rtol=2e-12, atol=2e-12)
            charged = ledger.charge(n=n, word="LD", arm="production", group="admission",
                                   session_id=job["session_id"])
            first, start = telemetry.started, time.perf_counter()
            checked = prepare_composition(model, matrix_request(n, "LD"))
            actual = execute_composition(model, checked)
            end = time.perf_counter()
            if not isinstance(actual, CompositionSuccess):
                raise RuntimeError("Production matrix admission failed")
            assert_reference_parity(reference, actual.model, matrix_request(n, "LD"), rtol=2e-12, atol=2e-12)
            if telemetry.started-first != 1 or model_identity(model) != identity or _rng_identity() != rng:
                raise RuntimeError("Production admission changed source/RNG or SVD accounting")
            probe = torch.arange(3*n, dtype=torch.float64).reshape(3, n)/n
            expected_path = destination/f"expected-{n}.pt"
            torch.save(dict(weight=reference.model[0].weight.detach(), bias=reference.model[0].bias.detach(),
                            output=reference.model(probe).detach()), expected_path)
            row = dict(n=n, word="LD", arm="production", group="admission", session_id=job["session_id"],
                       accepted=True, global_attempt=charged["global_attempt"], observed_svd_starts=1,
                       started_at=start, completed_at=end, call_seconds=end-start,
                       tensor_bytes=actual.storage_bytes)
            _append(Path(job["directory"])/"matrix-events.jsonl", row)
            _write(Path(job["directory"])/"active-request.json", dict(charged, active=False))
            rows.append(row)
            sources.append(dict(record, expected_file=expected_path.name,
                expected_sha256=hashlib.sha256(expected_path.read_bytes()).hexdigest(),
                historical_source=str(historical/record["source_file"]),
                historical_manifest_sha256=hashlib.sha256((historical/"manifest.json").read_bytes()).hexdigest()))
    _write(destination/"manifest.json", sources)
    return dict(rows=rows, sources=sources, matrix_attempts=len(rows),
                observed_svd_starts=telemetry.started, svd_completed=telemetry.completed)


def run_session(job):
    import torch
    from .compression_search_evaluation import model_identity
    n, folder = job["n"], Path(job["output"])/"matrix-sources"
    source = next(row for row in json.loads((folder/"manifest.json").read_text(encoding="utf-8")) if row["n"] == n)
    for name in ("source", "expected"):
        if hashlib.sha256((folder/source[name+"_file"]).read_bytes()).hexdigest() != source[name+"_sha256"]:
            raise RuntimeError("Frozen matrix artifact changed")
    model = matrix_source(n)
    model.load_state_dict(torch.load(folder/source["source_file"], weights_only=True), strict=True)
    if model_identity(model) != source["identity"]:
        raise RuntimeError("Matrix identity changed")
    expected = torch.load(folder/source["expected_file"], weights_only=True)
    ledger, streams, partial = MatrixLedger(job), [], None
    requests = {word: matrix_request(n, word) for word in ("LD", "LL")}
    with SVDTelemetry(job["directory"]) as telemetry:
        try:
            for metadata in session_schedule(n, job["session_id"]):
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
    return dict(n=n, session_id=job["session_id"], streams=streams, partial=partial, source=source,
                observed_svd_starts=telemetry.started, svd_completed=telemetry.completed,
                scheduled_streams=107, completed_streams=len(streams))


def summarize_sessions(streams, resources, planned=None):
    """Keep each session's controls, order and resource coverage independent."""
    result = []
    from .compression_search_local3_protocol import schedule
    jobs = schedule() if planned is None else planned
    sessions = {row["session_id"]: row["n"] for row in jobs if row.get("kind", "matrix") == "matrix"}
    for row in streams:
        if row["session_id"] not in sessions:
            raise ValueError("Unplanned matrix session")
        if sessions[row["session_id"]] != row["n"]:
            raise ValueError("Session size differs from its frozen schedule")
    for session, n in sessions.items():
        selected = [row for row in streams if row["session_id"] == session]
        sizes = {row["n"] for row in selected}
        if len(sizes) > 1:
            raise ValueError("A session must own exactly one size")
        keys = [(row["n"], row["invalid_count"], row["group"], row.get("block"),
                 row.get("pair"), row.get("label"), row["arm"]) for row in selected]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate stream in a matrix session")
        resource = resources.get(session, {})
        monitored = bool(resource.get("coverage", {}).get("complete")) and resource.get("stop") in (
            None, "hard_deadline", "request_hard_deadline") and resource.get("descendants_closed") is True
        full = [row for row in selected if row.get("request_count") == 8]
        for row in summarize_matrix(full):
            if row["n"] != n:
                continue
            eligible = row["eligible"] and monitored
            result.append(dict(row, session_id=session, resource_eligible=monitored,
                               timing_eligible=row["eligible"], eligible=eligible,
                               observed_win=row["observed_win"] and monitored))
    return result
