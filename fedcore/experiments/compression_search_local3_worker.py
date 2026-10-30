"""One admitted numerical worker for E or S in issue-99 local package three."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time

MODULE_ENTERED_AT = time.perf_counter()
STREAM_RNG_SEED = 990011


def statistics_schedule():
    """Two warmups, four C0/C0 controls and six paired C0/C1 repetitions."""
    rows = [dict(group="warmup", pair=None, arm=arm, label=arm, count=4) for arm in ("C0", "C1")]
    def controls(pair):
        labels = ("C0a", "C0b") if pair in (0, 3) else ("C0b", "C0a")
        return [dict(group="control", pair=pair, arm="C0", label=label, count=24) for label in labels]
    rows.extend(controls(0))
    rows.extend(controls(1))
    for pair in range(6):
        arms = ("C0", "C1") if pair % 2 == 0 else ("C1", "C0")
        rows.extend(dict(group="paired", pair=pair, arm=arm, label=arm, count=24) for arm in arms)
    rows.extend(controls(2))
    rows.extend(controls(3))
    return tuple(dict(row, stream_id=f"stream-{index:02d}", position=index)
                 for index, row in enumerate(rows))


def admission_genotypes():
    from fedcore.algorithm.low_rank.compression_grammar import (
        ASVDChoice, AffinePCAChoice, CompressionGenotype, ScopeGene, WeightedChoice)
    from fedcore.algorithm.low_rank.method_specs import ASVD
    pairs = [(WeightedChoice(4), WeightedChoice(4)),
        (WeightedChoice(32), WeightedChoice(32)),
        (ASVDChoice(8, ASVD(alpha=.25)), ASVDChoice(8, ASVD(alpha=.25))),
        (ASVDChoice(32, ASVD(alpha=.5)), ASVDChoice(32, ASVD(alpha=.5))),
        (AffinePCAChoice(4), AffinePCAChoice(4)),
        (WeightedChoice(8), AffinePCAChoice(32))]
    return tuple(CompressionGenotype((ScopeGene(("0",), left), ScopeGene(("2",), right)))
                 for left, right in pairs)


def statistics_candidates():
    from .compression_search_digits_protocol import TUNED_CATALOG, random_search_stream
    return random_search_stream(TUNED_CATALOG, 11)[:24]


def verify_metadata_tapes(expected):
    """Refuse a metadata mismatch before loading M0 or measuring any quality."""
    from .compression_search_digits_protocol import INITIAL_GENOTYPES, TUNED_CATALOG, canonical_solution_id
    from .compression_search_evaluation import PackageFatal
    from .protocol import stable_hash
    ids = [canonical_solution_id(value) for value in statistics_candidates()]
    actual = dict(catalog_ids=[canonical_solution_id(value) for value in TUNED_CATALOG.genotypes],
        initial_ids=[canonical_solution_id(value) for value in INITIAL_GENOTYPES],
        statistics_ids=ids, statistics_sha256=stable_hash(ids),
        admission_ids=[canonical_solution_id(value) for value in admission_genotypes()],
        rng_seed=STREAM_RNG_SEED)
    if actual != expected:
        raise PackageFatal("Numerical worker metadata differs from the pre-source frozen tapes")


def _rng_state():
    import numpy as np
    import torch
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def _frozen_rng():
    import numpy as np
    import torch
    from .compression_search_evaluation import isolated_rng
    with isolated_rng():
        random.seed(STREAM_RNG_SEED)
        np.random.seed(STREAM_RNG_SEED)
        torch.manual_seed(STREAM_RNG_SEED)
        return _rng_state()


def _persist_tensors(directory, stream_id, tensors):
    """Raw little-endian tensors plus JSON descriptors allow a stdlib audit."""
    import torch
    from .compression_search_local2 import write_json
    from .protocol import tensor_hash
    path = Path(directory)/f"{stream_id}-tensors.bin"
    records = []
    with path.open("wb") as output:
        def save(tensor):
            if tensor.dtype != torch.float32:
                raise ValueError("Local3 saved candidate tensors must be float32")
            data = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
            record = dict(file=path.name, offset=output.tell(), nbytes=len(data),
                shape=list(tensor.shape), dtype="float32", byte_order="little",
                raw_sha256=hashlib.sha256(data).hexdigest(), tensor_sha256=tensor_hash(tensor))
            output.write(data)
            return record
        for row in tensors:
            record = {name: value for name, value in row.items() if name not in ("logits", "operators")}
            if row.get("status") == "finite":
                record["logits"] = save(row["logits"])
                record["operators"] = {scope: {name: save(tensor) for name, tensor in operator.items()}
                    for scope, operator in row["operators"].items()}
            records.append(record)
    index_path = Path(directory)/f"{stream_id}-tensors.json"
    write_json(index_path, dict(schema="local3-f32-tensors-v1", records=records,
        payload_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return dict(index_file=index_path.name, payload_file=path.name,
        payload_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _statistics_job(job, source, space, evaluator, timer):
    from dataclasses import asdict
    from fedcore.algorithm.low_rank.artifacts import StatisticsStore
    from .compression_search_digits_evaluation import check_deadline, rng_signature
    from .compression_search_digits_protocol import canonical_solution_id
    from .compression_search_evaluation import PackageFatal, append_event, compile_checked
    from .compression_search_local2 import write_json
    from .compression_search_local3_evaluation import STATISTICS_MAX_BYTES, assert_same_result
    from .protocol import stable_hash
    admission = job["kind"] == "statistics_admission"
    tape = admission_genotypes() if admission else statistics_candidates()
    ids = [canonical_solution_id(value) for value in tape]
    tape_hash = stable_hash(ids)
    schedule = (tuple(dict(group="admission", pair=0, arm=arm, label=arm, count=6,
        stream_id=f"admission-{arm}", position=index) for index, arm in enumerate(("C0", "C1")))
        if admission else statistics_schedule())
    write_json(Path(job["directory"])/"statistics-tape.json", dict(solution_ids=ids,
        sha256=tape_hash, rng_seed=STREAM_RNG_SEED, schedule=schedule))
    write_json(Path(job["directory"])/"validation-labels.json", dict(
        labels=source.validation.y.tolist(), manifest=source.validation.manifest()))
    frozen, caller_rng = _frozen_rng(), _rng_state()
    references, streams = {}, []
    write_json(Path(job["directory"])/"statistics-streams.json", streams)
    try:
        for spec in schedule:
            check_deadline(job)
            _restore_rng(frozen)
            before_rng = rng_signature()
            start_index, tensor_index = len(evaluator.rows), len(evaluator.tensor_results)
            stream_started = time.perf_counter()
            store = StatisticsStore(STATISTICS_MAX_BYTES) if spec["arm"] == "C1" else None
            row = dict(spec, job=job["job"], split_seed=source.seed,
                stream_id=f"{job['job']}-{spec['stream_id']}", started_at=stream_started,
                tape_sha256=tape_hash, rng=before_rng, store_initial_bytes=store.used_bytes if store else 0,
                store_initial_entries=len(store.refs) if store else 0,
                store_capacity_bytes=STATISTICS_MAX_BYTES if store else None,
                store_ownership="caller_per_stream" if store else "executor_per_profile")
            append_event(Path(job["directory"])/"stream-events.jsonl", dict(event="stream_started", **row))
            evaluator.store, evaluator.stream_metadata = store, {name: row[name] for name in (
                "stream_id", "group", "pair", "arm", "position")}
            try:
                for genotype in tape[:spec["count"]]:
                    check_deadline(job)
                    with timer.measure("candidate_compilation", parent="statistics_stream", stream_id=row["stream_id"]):
                        workflow = compile_checked(genotype, space)
                    evaluator(workflow)
                    if evaluator.rows[-1].get("status") != "finite":
                        raise PackageFatal("A fixed S/admission candidate did not produce a finite model")
                row.update(store_final_bytes=store.used_bytes if store else 0,
                    store_final_entries=len(store.refs) if store else 0,
                    statistic_refs=[asdict(ref) for ref in store.refs] if store else [])
                if rng_signature() != before_rng:
                    raise PackageFatal("Statistics stream changed the frozen caller RNG state")
                row["status"] = "succeeded"
            finally:
                evaluator.last_model = None
                row.setdefault("status", "partial")
                row.update(store_final_bytes=store.used_bytes if store else 0,
                    store_final_entries=len(store.refs) if store else 0,
                    statistic_refs=[asdict(ref) for ref in store.refs] if store else [])
                if store is not None:
                    store.close()
                evaluator.store = None
                row.update(completed_at=time.perf_counter(),
                    candidate_calls=len(evaluator.rows)-start_index, store_closed=True)
                row["elapsed_seconds"] = row["completed_at"]-stream_started
                streams.append(row)
                append_event(Path(job["directory"])/"stream-events.jsonl", dict(event="stream_completed", **row))
                write_json(Path(job["directory"])/"statistics-streams.json", streams)
            saved = evaluator.tensor_results[tensor_index:]
            with timer.measure("stream_file_publication", parent="worker", stream_id=row["stream_id"]):
                row["tensor_artifacts"] = _persist_tensors(job["directory"], row["stream_id"], saved)
            with timer.measure("stream_numerical_comparison", parent="worker", stream_id=row["stream_id"]):
                for identity, result in zip(ids, saved):
                    if identity in references:
                        assert_same_result(references[identity], result, source.validation.y)
                    else:
                        references[identity] = result
            row["exact_comparison_passed"] = True
            write_json(Path(job["directory"])/"statistics-streams.json", streams)
            # Keep only the canonical reference tensors across streams.
            del evaluator.tensor_results[tensor_index:]
    finally:
        _restore_rng(caller_rng)
    calls = sum(row["candidate_calls"] for row in streams)
    expected = 12 if admission else 488
    if calls != expected:
        raise PackageFatal(f"Statistics schedule count differs: {calls}/{expected}")
    return dict(streams=streams, tape_sha256=tape_hash, tape_solution_ids=ids,
        candidate_calls=calls, expected_candidate_calls=expected,
        exact_comparison_passed=True, stop_reason="completed_fixed_schedule")


def _search_job(job, space, evaluator, timer):
    from fedcore.algorithm.low_rank.grammar_search import GrammarSearchConfig, run_grammar_search
    from .compression_search_digits_evaluation import check_deadline, effective_deadline
    from .compression_search_digits_protocol import INITIAL_GENOTYPES, TUNED_CATALOG, canonical_solution_id, random_search_stream
    from .compression_search_evaluation import PackageFatal, append_event, compile_checked
    from .compression_search_local3_evaluation import interval_union
    trace_path = Path(job["directory"])/"search-trace.jsonl"
    deadline = effective_deadline(job)
    search, reason = None, "catalog_exhausted"
    first = time.perf_counter()
    # The outer boundary precedes run_grammar_search's GOLEM imports.
    with timer.measure("search_external", parent="worker"):
        if job["mode"] == "E":
            config = GrammarSearchConfig(seed=job["search_seed"], population_size=4, generations=64,
                mutation_only=False, max_evaluations=64, max_seconds=60., max_proposals=512)
            search = run_grammar_search(space, config, evaluator,
                fitness_context=evaluator.context_sha256, initial_genotypes=INITIAL_GENOTYPES,
                absolute_deadline=deadline, trace_callback=lambda event: append_event(trace_path, event),
                phase_callback=lambda event: timer.rows.append(dict(event, parent="search_external")))
            if search["status"] not in ("succeeded", "no_valid_fitness"):
                raise PackageFatal(f"Local3 optimizer failed: {search.get('reason')}")
            reason = search["stop_reason"]
        elif job["mode"] == "T":
            proposals = 0
            for genotype in random_search_stream(TUNED_CATALOG, job["search_seed"]):
                if len(evaluator.rows) >= 64:
                    reason = "evaluation_budget"
                    break
                if proposals >= 512:
                    reason = "proposal_budget"
                    break
                check_deadline(job)
                proposals += 1
                proposal_started = time.perf_counter()
                with timer.measure("candidate_compilation", parent="search_external"):
                    workflow = compile_checked(genotype, space)
                append_event(trace_path, dict(event="proposal", proposal_id=proposals,
                    canonical_solution_id=canonical_solution_id(genotype), started_at=proposal_started,
                    ended_at=time.perf_counter(), operator="initial" if proposals <= 4 else "uniform_remaining"))
                evaluator(workflow)
            search = dict(engine="frozen uniform remainder after common prefix", status="succeeded",
                charged_evaluations=len(evaluator.rows), created_proposals=proposals, cache_hits=0,
                component_reads=0, stop_reason=reason)
        else:
            raise PackageFatal("Local3 diagnostic search requires T or E")
    last = time.perf_counter()
    seconds = interval_union(evaluator.callback_intervals, start=first, end=last)
    clean = {key: value for key, value in search.items() if key not in ("best_genotypes", "trace", "finite_archive")}
    clean["stop_reason"] = reason
    return dict(search=clean, stop_reason=reason, completed_evaluation_budget=len(evaluator.rows) == 64,
        reached_64=len(evaluator.rows) == 64, complete_budget=len(evaluator.rows) == 64,
        partial=len(evaluator.rows) != 64, search_partition=dict(started_at=first, completed_at=last,
            total_seconds=last-first, evaluator_union_seconds=seconds,
            uncovered_seconds=max(0., last-first-seconds),
            uncovered_label="search shell, imports, proposals, objective reads and optimizer; unseparated"))


def run_job(job):
    """No training, oracle or test mode exists in this worker."""
    entered = time.perf_counter()
    from .compression_search_digits_evaluation import DigitsDeadline
    from .compression_search_evaluation import PackageFatal
    from .compression_search_local2 import write_json
    from .compression_search_local3_evaluation import source_evaluator
    from .compression_search_matrix import SVDTelemetry
    if job["kind"] not in ("search_diagnostic", "statistics_admission", "statistics_streams"):
        raise PackageFatal("Unknown Local3 MLP worker kind")
    verify_metadata_tapes(job.get("expected_tapes"))
    hard = job.get("hard_deadline", job["package_deadline"])
    admission = min(job["admission_deadline"], job["worker_started_at"]+60.) if job["kind"] == "search_diagnostic" else min(job["admission_deadline"], hard-20.)
    bounded = dict(job, admission_deadline=admission)
    with SVDTelemetry(job["directory"]) as telemetry:
        preparation_started = time.perf_counter()
        source, space, evaluator, timer = source_evaluator(bounded,
            capture_tensors=job["kind"] != "search_diagnostic")
        timer.rows.insert(0, dict(phase="launch_and_import", parent="worker",
            started_at=job["worker_started_at"], completed_at=preparation_started,
            worker_module_entered_at=MODULE_ENTERED_AT, worker_run_job_entered_at=entered))
        write_json(Path(job["directory"])/"source-manifest.json", source.manifest)
        try:
            result = (_search_job(bounded, space, evaluator, timer) if job["kind"] == "search_diagnostic"
                else _statistics_job(bounded, source, space, evaluator, timer))
        except DigitsDeadline as error:
            result = dict(partial=True, stop_reason="time_or_attempt_budget", reason=str(error))
            streams_path = Path(job["directory"])/"statistics-streams.json"
            if streams_path.exists():
                result["streams"] = json.loads(streams_path.read_text(encoding="utf-8"))
            if job["kind"] == "search_diagnostic":
                external = next(row for row in reversed(timer.rows) if row["phase"] == "search_external")
                from .compression_search_local3_evaluation import interval_union
                total = external["completed_at"]-external["started_at"]
                covered = interval_union(evaluator.callback_intervals,
                    start=external["started_at"], end=external["completed_at"])
                result.update(reached_64=len(evaluator.rows) == 64,
                    complete_budget=len(evaluator.rows) == 64,
                    completed_evaluation_budget=len(evaluator.rows) == 64,
                    search_partition=dict(started_at=external["started_at"], completed_at=external["completed_at"],
                        total_seconds=total, evaluator_union_seconds=covered,
                        uncovered_seconds=max(0., total-covered),
                        uncovered_label="unseparated search shell/imports/proposals/objective/optimizer"))
        result.update(split_seed=source.seed, model_seed=source.seed, mode=job.get("mode"),
            search_seed=job.get("search_seed"), deadline=admission,
            evaluations=evaluator.rows, callback_intervals=evaluator.callback_intervals,
            phases=timer.rows, source_manifest_sha256=hashlib.sha256(
                (Path(job["source_root"])/str(source.seed)/"manifest.json").read_bytes()).hexdigest(),
            evaluation_context_sha256=evaluator.context_sha256,
            observed_svd_starts=telemetry.started, observed_svd_completions=telemetry.completed,
            svd_events=telemetry.events)
        completed = time.perf_counter()
        worker_phases = {"launch_and_import", "source_load_and_validation", "space_preparation",
            "search_external", "stream_file_publication", "stream_numerical_comparison"}
        worker_intervals = [(row["started_at"], row["completed_at"]) for row in timer.rows
            if row["phase"] in worker_phases]
        worker_intervals.extend((row["started_at"], row["completed_at"])
            for row in result.get("streams", []))
        from .compression_search_local3_evaluation import interval_union
        covered = interval_union(worker_intervals, start=job["worker_started_at"], end=completed)
        result["worker_partition"] = dict(started_at=job["worker_started_at"], completed_at=completed,
            total_seconds=completed-job["worker_started_at"], covered_seconds=covered,
            uncovered_seconds=max(0., completed-job["worker_started_at"]-covered))
    return result
