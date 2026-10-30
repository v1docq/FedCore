"""Frozen 3-M0 A100 series. No --execute: integrity checks only, no ML/CUDA/output.

Numerical work lives in sequential fresh child processes. The controller never
imports Torch or retains a CUDA context. An external owned-tree guard supplies
the earlier absolute monotonic deadline and supervises resources/foreign GPUs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from .cv_independent_protocol import (GlobalSeal, IndependentProtocol, MODELS,
    SEARCH_SEEDS, WINDOW_SEEDS, SearchOutcome, Strategy, digest, eligible_finalists,
    primary_aggregate, search_jobs, stage_deadline, aa_session_admitted, finalize_timing_admissions, is_digest)


MODULE = 'fedcore.experiments.cv_independent_series'
OWN_MODULES = ('cv_independent_protocol', 'cv_independent_data',
               'cv_independent_runtime', 'cv_independent_series', 'cv_independent_analysis')
OFFICIAL_MD5 = {
    'data_batch_1': 'c99cafc152244af753f735de768cd75f',
    'data_batch_2': 'd4bba439e000b95fd0a9bffe97cbabec',
    'data_batch_3': '54ebc095f3ab1f0389bbae665268c751',
    'data_batch_4': '634d18415352ddfa80567beed471001a',
    'data_batch_5': '482c414d41f54cd18b22e5b47cb7c3cb',
    'test_batch': '40351d587109b95175f43aff81a1287e',
    'batches.meta': '5ff9c542aee3614f3951f8cda6e48888'}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def preparation(data_root, output_root, manifest_path, protocol=IndependentProtocol()):
    from .cv_sparse_runtime import source_manifest
    if Path(output_root).exists():
        raise FileExistsError('Preserve earlier outputs; choose a new output directory')
    verified = source_manifest(manifest_path, protocol)
    expected = {f'fedcore/experiments/{name}.py' for name in OWN_MODULES}
    expected.add('fedcore/interfaces/fedcore_dispatcher.py')
    if not expected.issubset(verified['verified_package_files_sha256']):
        raise ValueError('Source manifest must cover the complete independent-M0 implementation')
    data_files, missing = {}, []
    for name, expected_md5 in OFFICIAL_MD5.items():
        path = Path(data_root) / 'cifar-10-batches-py' / name
        if not path.is_file():
            missing.append(name)
            continue
        sha, md5 = hashlib.sha256(), hashlib.md5()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024**2), b''):
                sha.update(block)
                md5.update(block)
        if md5.hexdigest() != expected_md5:
            raise ValueError(f'Official CIFAR10 integrity mismatch: {name}')
        data_files[name] = {'sha256': sha.hexdigest(), 'bytes': path.stat().st_size}
    versions = {}
    for name in ('fedcore', 'torch', 'torchvision', 'fedot', 'thegolem', 'numpy', 'scipy', 'psutil'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'status': 'missing_data' if missing else 'prepared', 'execute': False,
            'protocol': protocol.payload(), 'protocol_sha256': protocol.sha256,
            'source': verified, 'source_sha256': verified['manifest_sha256'],
            'data_files': data_files, 'missing_data': missing, 'versions': versions}


def _deadline(started, external=None):
    values = [started + IndependentProtocol().worker_seconds]
    for value in (external, os.environ.get('FEDCORE_PILOT_ABSOLUTE_DEADLINE')):
        if value is not None:
            parsed = float(value)
            if not math.isfinite(parsed) or parsed <= started:
                raise ValueError('Absolute deadline must be finite, monotonic and in the future')
            values.append(parsed)
    return min(values)


def _confined(root, relative):
    path = (Path(root).resolve() / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError('An existing file confined to this series is required')
    return path


def _artifact(root, record):
    from .cv_sparse_runtime import sha256_file
    path = _confined(root, record['path'])
    if (sha256_file(path) != record['sha256'] or path.stat().st_size != record['file_bytes']
            or type(record['tensor_bytes']) is not int or record['tensor_bytes'] <= 0
            or record.get('completed_within_deadline') is not True):
        raise ValueError('A selected independent checkpoint failed fresh file admission')
    return path


def _rebase(artifact, worker_relative):
    return {**artifact, 'path': str(Path(worker_relative) / artifact['path'])}


def search_context(job, *, source_tensor_bytes, source_m0_fingerprint, data_sha256,
                   tape_sha256, source_sha256, deadline, process_started_at):
    payload = {'schema': 'independent_search_context_v1', 'job_id': job.name,
        'model': job.model, 'seed': job.seed, 'strategy': job.strategy.value,
        'source_tensor_bytes': source_tensor_bytes, 'byte_ceiling': math.floor(.8*source_tensor_bytes),
        'source_m0_fingerprint': source_m0_fingerprint, 'data_sha256': data_sha256,
        'tape_sha256': tape_sha256, 'source_sha256': source_sha256,
        'protocol_sha256': IndependentProtocol().sha256, 'deadline': deadline,
        'process_started_at': process_started_at}
    record = {'context': payload, 'context_sha256': digest(payload)}
    validate_search_context(record)
    return record


def validate_search_context(record):
    context = record['context']
    jobs = {job.name: job for job in search_jobs()}
    job = jobs.get(context.get('job_id'))
    source_bytes = context.get('source_tensor_bytes')
    if (record.get('context_sha256') != digest(context)
            or context.get('schema') != 'independent_search_context_v1' or job is None
            or context.get('model') != job.model or context.get('seed') != job.seed
            or context.get('strategy') != job.strategy.value
            or context.get('protocol_sha256') != IndependentProtocol().sha256
            or type(source_bytes) is not int or source_bytes <= 1
            or type(context.get('byte_ceiling')) is not int
            or context['byte_ceiling'] != math.floor(.8*source_bytes)
            or any(not is_digest(context.get(key)) for key in
                   ('source_m0_fingerprint', 'data_sha256', 'tape_sha256', 'source_sha256'))
            or any(type(context.get(key)) not in (float, int) or not math.isfinite(context[key])
                   for key in ('deadline', 'process_started_at'))
            or context['deadline'] <= context['process_started_at']):
        raise ValueError('Invalid durable independent search context')
    return context


def search_cost_accounting(events, *, process_started_at, process_completed_at):
    """Partition elapsed wall time without adding nested optimizer phases."""
    if (any(type(x) not in (float, int) or not math.isfinite(x)
            for x in (process_started_at, process_completed_at))
            or process_completed_at < process_started_at):
        raise ValueError('Finite ordered process timestamps required')
    first, last = process_started_at, process_completed_at
    wall = last-first
    starts = [e['timestamp'] for e in events if e.get('event') == 'search_engine_started']
    ends = [e['timestamp'] for e in events if e.get('event') == 'search_engine_ended']
    if not starts:
        return {'process_wall_seconds': wall, 'engine_wall_seconds': 0., 'callback_sum_seconds': 0.,
            'callback_union_seconds': 0., 'engine_import_setup_seconds': 0., 'engine_residual_seconds': 0.,
            'process_outside_engine_seconds': wall, 'engine_incomplete': True,
            'callback_intervals': 0, 'open_callbacks': 0, 'nested_phases_are_not_additive': True}
    engine_start = min(last, max(first, min(starts)))
    engine_end = min(last, max(engine_start, max(ends))) if ends else last
    opened, intervals = {}, []
    for event in events:
        if event.get('event') == 'search_callback_started':
            opened[event['callback_id']] = event['timestamp']
        elif event.get('event') == 'search_phase' and event.get('phase') == 'numerical_callback':
            opened.pop(event['callback_id'], None)
            intervals.append((event['started_at'], event['completed_at']))
    intervals.extend((timestamp, engine_end) for timestamp in opened.values())
    clipped = sorted((max(engine_start, a), min(engine_end, b)) for a, b in intervals
                     if min(engine_end, b) > max(engine_start, a))
    ready_times = [e['timestamp'] for e in events if e.get('event') == 'search_engine_ready']
    ready = min(ready_times) if ready_times else engine_end
    if clipped:
        ready = min(ready, clipped[0][0])
    ready = min(engine_end, max(engine_start, ready))
    setup, summed, union = ready-engine_start, sum(b-a for a, b in clipped), 0.
    union_end = engine_start
    for a, b in clipped:
        union += max(0., b-max(a, union_end))
        union_end = max(union_end, b)
    engine_wall = engine_end-engine_start
    return {'process_wall_seconds': wall, 'engine_wall_seconds': engine_wall,
        'callback_sum_seconds': summed, 'callback_union_seconds': union,
        'engine_import_setup_seconds': setup, 'engine_residual_seconds': max(0., engine_wall-setup-union),
        'process_outside_engine_seconds': wall-engine_wall, 'engine_incomplete': not bool(ends),
        'callback_intervals': len(clipped), 'open_callbacks': len(opened),
        'nested_phases_are_not_additive': True}


def reconcile_worker_output(target, kind, *, force_partial=False, process_returncode=None,
                            observed_process_started_at=None, observed_process_completed_at=None):
    """Retain durable charged/SGD/evaluation prefixes even after a hard kill.

    A final torn append is explicitly marked. Other malformed journal rows are
    integrity failures. Counts describe confirmed records, not unobserved work.
    """
    target = Path(target)
    result_file = target/'result.json'
    result = (_read(result_file) if result_file.is_file() else
              {'status': 'partial', 'reason': 'worker_deadline_or_missing_result'})
    context = None
    if kind == 'search' and (target/'search_context.json').is_file():
        context = validate_search_context(_read(target/'search_context.json'))
        result['search_context_sha256'] = digest(context)
        for key in ('job_id', 'source_tensor_bytes', 'byte_ceiling', 'source_m0_fingerprint',
                    'data_sha256', 'tape_sha256', 'source_sha256', 'protocol_sha256', 'deadline'):
            if key in result and result[key] != context[key]:
                raise ValueError('Terminal search metadata differs from its durable context')
            result[key] = context[key]
    if force_partial:
        result.update(status='partial', complete_q32=False, controller_late=True)
    if process_returncode is not None and process_returncode != 0 and result['status'] == 'completed':
        result.update(status='failed', complete_q32=False, reason='nonzero_worker_exit_after_completed_result')
    for filename, field in (('quality_partial.json', 'quality'), ('timing_partial.json', None)):
        path = target/filename
        if path.is_file():
            partial = _read(path)
            if field is not None:
                result.setdefault(field, partial)
            else:
                for key, value in partial.items():
                    result.setdefault(key, value)
    if kind == 'timing':
        finalize_timing_admissions(result, completed=result['status'] == 'completed'
                                   and result.get('window_completed') is True)
    attempts_file, ledger = target/'attempts.json', target/'ledger.jsonl'
    persisted_attempts = _read(attempts_file) if attempts_file.is_file() else []
    by_solution = {}

    def retain_record(row):
        key = row['solution_id']
        existing = by_solution.get(key)
        if existing is None:
            by_solution[key] = dict(row)
            return
        # Later durable summaries own metadata. No source may undo an earlier
        # integrity/deadline exclusion, even when its own cached flag is True.
        merged = {**row, **existing}
        for flag in ('completed_within_deadline', 'quality_accepted'):
            values = [record[flag] for record in (existing, row) if flag in record]
            if values:
                merged[flag] = all(value is True for value in values)
        if any(record.get('status') == 'late' for record in (existing, row)):
            merged['status'] = 'late'
        if any(record.get('status') == 'fatal' for record in (existing, row)):
            merged['status'] = 'fatal'
        by_solution[key] = merged

    for row in [*result.get('attempts', []), *persisted_attempts]:
        retain_record(row)
    charged, completed_updates, evaluation_rows, common_ordinals = {}, {}, {}, {}
    baseline_updates = 0
    events = []
    if ledger.is_file():
        raw = ledger.read_bytes()
        lines = raw.splitlines()
        for index, line in enumerate(lines):
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if index == len(lines)-1 and not raw.endswith(b'\n'):
                    result['torn_last_ledger_append'] = True
                    continue
                raise ValueError('Malformed durable worker journal')
            event_type = event.get('event')
            events.append(event)
            if event_type == 'candidate_charged':
                charged[event['solution_id']] = event
            elif event_type == 'candidate_ended':
                retain_record({k: v for k, v in event.items() if k != 'event'})
            elif event_type == 'evolution_trace':
                trace = event['trace']
                if trace.get('event') == 'evaluation_started':
                    key = trace['canonical_solution_id']
                    common_ordinals[key] = trace['attempt_id']
                    charged.setdefault(key, {'solution_id': key, 'attempt_id': trace['attempt_id'],
                        'genotype': trace['genotype'], 'started_at': trace['started_at'],
                        'charged_attempt': True, 'source': 'common_pre_callback_compilation'})
                elif trace.get('event') == 'evaluated':
                    key = trace['canonical_solution_id']
                    if key not in by_solution:
                        by_solution[key] = {'solution_id': key, 'attempt_id': trace['attempt_id'],
                            'genotype': trace['genotype'], 'charged_attempt': True,
                            'status': 'failed', 'failure': trace['failure'], 'recovery_updates': 0,
                            'quality_accepted': False, 'started_at': trace['started_at'],
                            'ended_at': trace['ended_at'],
                            'completed_within_deadline': trace['completed_within_deadline']}
                    record = by_solution[key]
                    record['completed_within_deadline'] = (
                        record.get('completed_within_deadline') is True
                        and trace['completed_within_deadline'] is True)
                    failure = trace.get('failure_details') or {}
                    deadline_failure = failure.get('code') == 'EvaluationDeadline'
                    if not record['completed_within_deadline'] or deadline_failure:
                        record['completed_within_deadline'] = False
                        if record.get('status') == 'succeeded':
                            record['status'] = 'late'
            elif event_type == 'recovery_update':
                completed_updates[event['role']] = max(completed_updates.get(event['role'], 0), event['update'])
            elif event_type == 'baseline_update':
                baseline_updates = max(baseline_updates, event['update'])
            elif event_type in ('evaluation_progress', 'evaluation_batch'):
                evaluation_rows[event['label']] = max(evaluation_rows.get(event['label'], 0), event['completed_rows'])
    for key, event in charged.items():
        by_solution.setdefault(key, {**event, 'status': 'incomplete', 'quality_accepted': False,
                                    'recovery_updates': completed_updates.get(key, 0),
                                    'completed_within_deadline': False})
    for key, record in by_solution.items():
        if key in common_ordinals:
            record['attempt_id'] = common_ordinals[key]
        record['confirmed_sgd_updates'] = completed_updates.get(key, record.get('recovery_updates', 0))
    result['confirmed_baseline_updates'] = baseline_updates
    result['confirmed_recovery_updates'] = sum(completed_updates.values())
    result['confirmed_evaluation_rows'] = evaluation_rows
    if kind == 'search':
        result['charged'] = max(result.get('charged', 0), len(charged))
        result['attempts'] = sorted(by_solution.values(), key=lambda row: row['attempt_id'])
        if result['status'] != 'completed':
            result['complete_q32'] = False
        if type(result.get('byte_ceiling')) is int:
            eligible = [row for row in eligible_finalists(result['attempts'], byte_ceiling=result['byte_ceiling'])
                        if 'checkpoint' in row]
            result['finalist'] = eligible[0]['checkpoint'] if eligible else None
            result['finalist_admission'] = ('pending_controller_fresh_file_check' if eligible else 'absent')
        elif result.get('finalist'):
            row = by_solution.get(result['finalist']['solution_id'])
            if row is None or row.get('status') != 'succeeded' or row.get('completed_within_deadline') is not True:
                result['finalist'] = None
        process_start = (observed_process_started_at if observed_process_started_at is not None
                         else result.get('started_at', (context or {}).get('process_started_at')))
        process_end = (observed_process_completed_at if observed_process_completed_at is not None
                       else result.get('ended_at'))
        if process_start is not None and process_end is not None:
            result['cost_accounting'] = search_cost_accounting(events,
                process_started_at=process_start, process_completed_at=process_end)
    return result


def _seal_from_payload(value):
    return GlobalSeal(value['protocol_sha256'], value['data_sha256'], value['source_sha256'],
        tuple(SearchOutcome(**row) for row in value['outcomes']),
        tuple(tuple(row) for row in value['roles']), tuple(tuple(row) for row in value['artifacts']))


def _sealed_inputs(root, protocol):
    from .cv_sparse_runtime import sha256_file
    frozen = _read(Path(root) / 'global_seal.json')
    seal = _seal_from_payload(frozen['seal'])
    if seal.sha256 != frozen['seal_sha256'] or seal.protocol_sha256 != protocol.sha256:
        raise ValueError('The global series seal changed')
    artifacts = frozen['artifacts']
    if set(artifacts) != {key for key, _ in seal.artifacts}:
        raise ValueError('Global seal artifact set changed')
    for key, sha in seal.artifacts:
        if artifacts[key]['sha256'] != sha:
            raise ValueError('Global seal checkpoint hash changed')
        _artifact(root, artifacts[key])
    if sha256_file(Path(root) / 'source_manifest.json') != frozen['source_record_sha256']:
        raise ValueError('Series source record changed after sealing')
    return seal, artifacts


def _load_teacher(root, model_name, torch):
    from fedcore.tools.registry.checkpoint_manager import CheckpointManager
    record = _read(Path(root) / 'models' / f'{model_name}.json')
    path = _artifact(root, record)
    model = CheckpointManager('.', auto_cleanup=False).deserialize_from_bytes(path.read_bytes(), device='cpu')
    return model.eval().requires_grad_(False), record


def _worker(args, started):
    """Internal child entry point. No automatic retry or recovery of a job."""
    from .cv_sparse_runtime import (Output, PilotFatal, PilotStop, admit_free_gpu,
        checkpoint, verify_loaded_sources)
    from .cv_independent_runtime import IndependentEvaluator, IndependentGPUData, Runtime, train_baseline
    from .cv_independent_data import IndependentCifarData
    from .cv_sparse_pilot import solution_id, cv_menu, fixed_prefix
    protocol = IndependentProtocol()
    prepared = preparation(args.data_root, args.output_root, args.source_manifest, protocol)
    if prepared['status'] != 'prepared':
        raise ValueError('Prepared official CIFAR10 files required')
    output = Output(args.output_root)
    result = {'worker': args.worker, 'job_id': args.job_id, 'status': 'started', 'started_at': started}
    runtime = None
    try:
        limit = _deadline(started, args.absolute_deadline)
        result['deadline'] = limit
        result['gpu_admission'] = admit_free_gpu()
        runtime = Runtime(protocol, output, started, limit)
        runtime.source_identity = prepared['source_sha256']
        data = IndependentCifarData(args.data_root, protocol)
        if any(data.source_hashes.get(name) != row for name, row in prepared['data_files'].items()):
            raise ValueError('Data source changed after integrity preparation')
        runtime.check()
        output.json('environment.json', runtime.environment)
        output.json('data_roles.json', data.roles.payload())
        if args.worker != 'manifest':
            frozen_data = _read(Path(args.series_root) / 'data_manifest.json')
            if data.identity != frozen_data['identity']:
                raise ValueError('Fresh worker roles or dataset differ from series manifest')
        if args.worker == 'manifest':
            gpu = IndependentGPUData(data, runtime)
            result.update(identity=data.identity, roles_sha256=data.roles.sha256,
                          source_files=data.source_hashes, transforms={'mean': data.mean, 'std': data.std})
            output.json('job_order.json', protocol.payload()['search_jobs'])
            output.json('timing_windows.json', {'seeds': WINDOW_SEEDS})
            del gpu
        elif args.worker == 'baseline':
            seeds = next(m for m in MODELS if m.name == args.job_id)
            gpu = IndependentGPUData(data, runtime)
            runtime.stage = 'baseline'
            model, validation, proof = train_baseline(gpu, seeds, limit)
            key = solution_id(fixed_prefix(cv_menu())[0])
            artifact, restored = checkpoint(model, gpu, label=f'{seeds.name}_M0', solution_id=key,
                                            phase='dense_original_epoch80', deadline=limit)
            artifact.update(validation=validation, training=proof, model=seeds.name)
            result['baseline'] = artifact
            del restored, model, gpu
        elif args.worker in ('controls', 'search'):
            model_name = args.job_id if args.worker == 'controls' else args.job_id.split('_')[0]
            seeds = next(m for m in MODELS if m.name == model_name)
            teacher, baseline = _load_teacher(args.series_root, model_name, runtime.torch)
            evaluator = IndependentEvaluator(runtime, data, teacher, seeds)
            result.update(byte_ceiling=evaluator.byte_ceiling,
                          source_tensor_bytes=evaluator.space.source_storage_bytes,
                          source_m0_fingerprint=evaluator.teacher_fingerprint,
                          tape_sha256=evaluator.tape_sha256)
            runtime.stage = args.worker
            if args.worker == 'controls':
                for role, genotype in zip(('dense_recovery', 'fixed_prune', 'fixed_svd', 'fixed_hybrid'), evaluator.prefix):
                    runtime.check(limit, role=role)
                    evaluator.evaluate(genotype, role, limit)
                if any(r['status'] != 'succeeded' for r in evaluator.records):
                    raise PilotStop('controls', 'incomplete_fixed_control')
                result['controls'] = {role: record['checkpoint'] for role, record in
                    zip(('dense_recovery', 'fixed_prune', 'fixed_svd', 'fixed_hybrid'), evaluator.records)}
                if result['controls']['dense_recovery']['validation']['cross_entropy'] <= 0:
                    raise ValueError('Positive dense-recovery Vsearch CE required for fixed normalization')
            else:
                job = next(j for j in search_jobs() if j.name == args.job_id)
                context = search_context(job,
                    source_tensor_bytes=evaluator.space.source_storage_bytes,
                    source_m0_fingerprint=evaluator.teacher_fingerprint, data_sha256=data.identity,
                    tape_sha256=evaluator.tape_sha256, source_sha256=prepared['source_sha256'],
                    deadline=limit, process_started_at=started)
                if context['context']['byte_ceiling'] != evaluator.byte_ceiling:
                    raise ValueError('Public CV space differs from the frozen durable byte ceiling')
                output.json('search_context.json', context)
                output.append({'event': 'search_started', 'job': job.name, 'cache_entries': 0,
                               'deadline': limit, 'timestamp': time.perf_counter()})
                try:
                    search = (evaluator.random_search(job.seed, limit) if job.strategy is Strategy.RANDOM
                              else evaluator.evolution(job.seed, limit))
                except PilotStop as error:
                    if error.reason != 'absolute_deadline':
                        raise
                    search = {'reason': 'time_budget', 'charged_evaluations': len(evaluator.records)}
                # Common charging includes structural attempts that failed before
                # invoking our numerical callback. Preserve that distinction.
                charged = search['charged_evaluations']
                finalists = eligible_finalists(evaluator.records, byte_ceiling=evaluator.byte_ceiling)
                result.update(search=search, charged=charged,
                    successful=sum(r['status'] == 'succeeded' and r['completed_within_deadline'] for r in evaluator.records),
                    finalist=finalists[0]['checkpoint'] if finalists else None,
                    complete_q32=charged == 32 and time.perf_counter() <= limit)
            result['attempts'] = evaluator.records
            del evaluator, teacher
        elif args.worker == 'quality':
            result.update(_quality_worker(args.series_root, runtime, data, limit))
        elif args.worker == 'timing':
            result.update(_timing_worker(args.series_root, runtime, data, limit, int(args.job_id)))
        else:
            raise ValueError('Unsupported internal worker')
        runtime.check(limit)
        output.json('imported_sources.json', verify_loaded_sources(prepared['source']))
        result['status'] = 'completed'
    except PilotStop as error:
        result.update(status='partial' if error.reason == 'absolute_deadline' else 'failed',
                      reason=error.reason, stage=error.stage)
    except (Exception, PilotFatal) as error:
        result.update(status='failed', failure_type=type(error).__name__, failure=str(error),
                      traceback=traceback.format_exc())
    finally:
        result['ended_at'] = time.perf_counter()
        result['wall_seconds'] = result['ended_at']-started
        if runtime is not None:
            result['peak_allocated_bytes'] = runtime.torch.cuda.max_memory_allocated()
            result['peak_reserved_bytes'] = runtime.torch.cuda.max_memory_reserved()
        # A caught failure must expose the same durable prefixes as a killed
        # process. Never leave a prior provisional AB marked admitted.
        output.json('result.json', result)
        result = reconcile_worker_output(output.root, args.worker)
        output.json('result.json', result)
        output.append({'event': 'worker_stopped', 'status': result['status'], 'timestamp': result['ended_at']})
    return result


def _quality_worker(root, runtime, data, deadline):
    import numpy as np
    from fedcore.tools.registry.checkpoint_manager import CheckpointManager
    from .cv_independent_runtime import IndependentGPUData, evaluate
    from .cv_sparse_runtime import torchscript_artifact, independent_torchscript
    seal, artifacts = _sealed_inputs(root, runtime.protocol)
    gpu = IndependentGPUData(data, runtime)
    report_ids = data.open_report(seal)
    quality = {}
    # All eager Vreport results are retained even if an export is ineligible.
    for key, original in artifacts.items():
        runtime.check(deadline, artifact=key)
        path = _artifact(root, original)
        model = CheckpointManager('.', auto_cleanup=False).deserialize_from_bytes(path.read_bytes(), device='cuda').eval()
        eager = evaluate(model, gpu, report_ids, deadline=deadline, label=f'{key}_Vreport_eager')
        model.cpu()
        # The checked exporter resolves files against its current worker output.
        # A confined byte copy gives it exactly the checkpoint sealed by parent.
        local = runtime.output.bytes(f'inputs/{key}.pt', path.read_bytes())
        if local['sha256'] != original['sha256']:
            raise ValueError('Sealed checkpoint changed while copying for export')
        artifact = {**original, **local}
        row = {'eager': eager, 'export_eligible': False}
        quality[key] = row
        runtime.output.json('quality_partial.json', quality)
        try:
            ts, record = torchscript_artifact(artifact, gpu, deadline)
            del ts
            ts = independent_torchscript(record, gpu, deadline)
            ts_metrics = evaluate(ts, gpu, report_ids, deadline=deadline, label=f'{key}_Vreport_ts')
            eager_rows = np.load(runtime.output.root / eager['prediction_file'])
            ts_rows = np.load(runtime.output.root / ts_metrics['prediction_file'])
            exact = all(np.array_equal(eager_rows[name], ts_rows[name]) for name in ('row_ids', 'labels', 'predictions'))
            losses_equal = np.allclose(eager_rows['cross_entropy'], ts_rows['cross_entropy'], atol=1e-5, rtol=1e-5)
            row.update(torchscript=record, ts=ts_metrics, exact_ids_labels_classes=bool(exact),
                       losses_within_tolerance=bool(losses_equal), export_eligible=bool(exact and losses_equal))
            del ts
        except (runtime.torch.jit.frontend.NotSupportedError,
                runtime.torch.jit.frontend.UnsupportedNodeError) as error:
            # Only declared unsupported TorchScript syntax is an export outcome.
            # RuntimeError, AssertionError, CUDA OOM and integrity errors are fatal.
            row.update(export_failure_type=type(error).__name__, export_failure=str(error))
        runtime.output.json('quality_partial.json', quality)
        del model
        runtime.check(deadline)
    # One test opening, after the durable global seal and all Vreport exports.
    test = data.open_test(seal)
    from .cv_sparse_runtime import independent_torchscript
    for key, row in quality.items():
        if not row['export_eligible']:
            continue
        ts = independent_torchscript(row['torchscript'], gpu, deadline)
        row['test'] = evaluate(ts, gpu, tuple(range(10000)), deadline=deadline,
                               label=f'{key}_official_repeated_test_ts', test_data=test)
        del ts
        runtime.output.json('quality_partial.json', quality)
    return {'quality': quality, 'seal_sha256': seal.sha256,
            'Vreport_opened_after_global_seal': True, 'official_test_open_count': 1}


def _timing_worker(root, runtime, data, deadline, window):
    import random
    from .cv_independent_runtime import IndependentGPUData
    from .cv_sparse_runtime import paired_latency, sha256_file
    if window not in (0, 1, 2):
        raise ValueError('Exactly three predeclared timing windows exist')
    seal, artifacts = _sealed_inputs(root, runtime.protocol)
    quality = _read(Path(root) / 'quality.json')['quality']
    gpu = IndependentGPUData(data, runtime)
    role_artifact = dict(seal.roles)
    timings = {}
    result = {'window': window, 'seed': WINDOW_SEEDS[window], 'models': timings, 'window_completed': False}

    def load(key):
        row = quality[key]
        if not row['export_eligible']:
            raise ValueError('Timing requires the very same admitted TS used in repeated test')
        record = row['torchscript']
        path = _confined(root, record['path'])
        if sha256_file(path) != record['sha256']:
            raise ValueError('The reported/tested TorchScript bytes changed before timing')
        runtime.check(deadline)
        return runtime.torch.jit.load(str(path), map_location='cuda').eval()

    def aa(key, label):
        first, second = load(key), load(key)
        try:
            return paired_latency(first, second, gpu, deadline=deadline, label=label, aa=True)
        finally:
            del first, second

    # Frozen permutations are independent of A/A outcomes; no extra windows.
    order = _read(Path(root) / 'timing_order.json')[str(window)]
    for model in MODELS:
        dense = role_artifact[f'{model.name}_dense_recovery']
        model_result = {'candidates': {}, 'status': 'started'}
        timings[model.name] = model_result
        if not quality[dense]['export_eligible']:
            model_result.update(status='ineligible_dense_export')
            continue
        model_result['initial_dense_AA'] = aa(dense, f'{model.name}_initial_dense_AA')
        candidates = [key for key in order[model.name] if quality[key]['export_eligible']]
        for key in candidates:
            own = aa(key, f'{model.name}_{key}_own_AA')
            model_result['candidates'][key] = {'own_AA': own, 'comparison_admitted': False}
            runtime.output.json('timing_partial.json', result)
        model_result['dense_pre_AA'] = aa(dense, f'{model.name}_dense_pre_AA')
        pre_ok = aa_session_admitted(model_result['initial_dense_AA']) and aa_session_admitted(model_result['dense_pre_AA'])
        for key in candidates:
            row = model_result['candidates'][key]
            if pre_ok and aa_session_admitted(row['own_AA']):
                first, second = load(dense), load(key)
                try:
                    row['AB'] = paired_latency(first, second, gpu, deadline=deadline, label=f'{model.name}_{key}_AB')
                finally:
                    del first, second
            runtime.output.json('timing_partial.json', result)
        model_result['dense_post_AA'] = aa(dense, f'{model.name}_dense_post_AA')
        model_result['status'] = 'completed'
        runtime.output.json('timing_partial.json', result)
    runtime.check(deadline)
    finalize_timing_admissions(result, completed=True)
    runtime.output.json('timing_partial.json', result)
    return result


class ControllerStop(Exception):
    pass


def run_series(data_root, output_root, manifest, *, started=None, absolute_deadline=None):
    from .cv_sparse_runtime import Output, sha256_file
    protocol = IndependentProtocol()
    started = time.perf_counter() if started is None else started
    absolute = _deadline(started, absolute_deadline)
    prepared = preparation(data_root, output_root, manifest, protocol)
    if prepared['status'] != 'prepared':
        raise ValueError('Prepared official CIFAR10 files required')
    output = Output(output_root)
    root = output.root
    report = {'schema': protocol.schema, 'status': 'started', 'started_at': started,
              'absolute_deadline': absolute, 'protocol_sha256': protocol.sha256, 'workers': [],
              'test_scope': 'historically used official test, descriptive repetition',
              'independent_trainings': 3, 'search_repetitions_per_model': 3}
    output.json('protocol.json', protocol.payload())
    output.json('source_manifest.json', prepared['source'])
    output.json('job_order.json', protocol.payload()['search_jobs'])
    outcome_map = {j.name: SearchOutcome(j.name, 'not_started', 0, None, False) for j in search_jobs()}
    models, controls, selected, roles, searches = {}, {}, {}, [], {}
    active_stage = {}

    def check(limit):
        if time.perf_counter() >= min(limit, absolute):
            raise ControllerStop('absolute_or_stage_deadline')

    def begin_stage(name, *, anchor=None):
        now = time.perf_counter()
        if active_stage:
            output.append({'event': 'stage_ended', 'stage': active_stage['name'],
                'timestamp': now, 'seconds': now-active_stage['started']})
        began = now if anchor is None else anchor
        limit = stage_deadline(began, absolute, name)
        active_stage.update(name=name, started=began)
        output.append({'event': 'stage_started', 'stage': name, 'timestamp': began, 'deadline': limit})
        check(limit)
        return limit

    def child(kind, job_id, relative, deadline):
        check(deadline)
        target = root / relative
        command = [sys.executable, '-m', MODULE, '--execute', '--data-root', str(Path(data_root).resolve()),
                   '--output-root', str(target), '--source-manifest', str(Path(manifest).resolve()),
                   '--absolute-deadline', str(min(deadline, absolute)), '--worker', kind,
                   '--job-id', job_id, '--series-root', str(root)]
        output.append({'event': 'child_started', 'worker': kind, 'job': job_id, 'command': command,
                       'deadline': deadline, 'timestamp': time.perf_counter()})
        console = output._target(f'console/{relative.replace("/", "_")}.log')
        process_started_at = time.perf_counter()
        with console.open('wb') as stream:
            proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            try:
                code = proc.wait(timeout=max(.01, min(deadline, absolute)-time.perf_counter()))
            except subprocess.TimeoutExpired:
                # Only this child is owned; no external GPU process is touched.
                proc.kill()
                proc.wait()
                code = proc.returncode
        process_completed_at = time.perf_counter()
        result = reconcile_worker_output(target, kind, force_partial=time.perf_counter() > min(deadline, absolute),
            process_returncode=code, observed_process_started_at=process_started_at,
            observed_process_completed_at=process_completed_at)
        result['process_returncode'] = code
        result['worker_relative'] = relative
        report['workers'].append({'kind': kind, 'job': job_id, 'relative': relative,
                                  'status': result['status'], 'process_returncode': code})
        output.append({'event': 'child_ended', **report['workers'][-1], 'timestamp': time.perf_counter()})
        output.json('report.json', report)
        return result

    try:
        if shutil.disk_usage(root).free < protocol.minimum_disk_bytes:
            raise ControllerStop('less_than_100_GiB_free_disk')
        stage = begin_stage('manifest', anchor=started)
        result = child('manifest', 'manifest', 'workers/manifest', stage)
        if result['status'] != 'completed':
            raise ControllerStop('manifest_worker_' + result['status'])
        output.json('data_manifest.json', result)
        stage = begin_stage('baseline')
        fingerprints = set()
        for model in MODELS:
            limit = min(stage, time.perf_counter()+protocol.baseline_seconds)
            relative = f'workers/baseline_{model.name}'
            result = child('baseline', model.name, relative, limit)
            if result['status'] != 'completed':
                raise ControllerStop(f'{model.name}_baseline_' + result['status'])
            artifact = _rebase(result['baseline'], relative)
            _artifact(root, artifact)
            fingerprint = artifact['training']['initial_fingerprint']
            if fingerprint in fingerprints:
                raise ControllerStop('independent_initialization_fingerprints_collided')
            fingerprints.add(fingerprint)
            models[model.name] = artifact
            selected[artifact['artifact_id']] = artifact
            roles.append((f'{model.name}_dense_original', artifact['artifact_id']))
            output.json(f'models/{model.name}.json', artifact)
        stage = begin_stage('controls')
        for model in MODELS:
            relative = f'workers/controls_{model.name}'
            result = child('controls', model.name, relative, stage)
            if result['status'] != 'completed':
                raise ControllerStop(f'{model.name}_controls_' + result['status'])
            controls[model.name] = {}
            for role, raw in result['controls'].items():
                artifact = _rebase(raw, relative)
                _artifact(root, artifact)
                selected[artifact['artifact_id']] = artifact
                roles.append((f'{model.name}_{role}', artifact['artifact_id']))
                controls[model.name][role] = artifact
        output.json('controls.json', controls)
        stage = begin_stage('search')
        feasibility_failed = False
        for index, job in enumerate(search_jobs()):
            relative = f'workers/search_{job.name}'
            result = child('search', job.name, relative, min(stage, time.perf_counter()+720))
            raw = result.get('finalist')
            artifact = _rebase(raw, relative) if raw is not None else None
            if artifact is not None:
                _artifact(root, artifact)
                result['finalist_admission'] = 'controller_fresh_file_verified'
                selected[artifact['artifact_id']] = artifact
                roles.append((f'{job.name}_finalist', artifact['artifact_id']))
            full = result.get('complete_q32') is True and result['status'] == 'completed'
            outcome_map[job.name] = SearchOutcome(job.name, 'completed' if full else
                ('failed' if result['status'] == 'failed' else 'partial'), result.get('charged', 0),
                artifact['artifact_id'] if artifact else None, full)
            searches[job.name] = result
            output.json('search_outcomes.json', [asdict(outcome_map[j.name]) for j in search_jobs()])
            if result['status'] == 'failed':
                raise ControllerStop('unknown_or_integrity_search_failure')
            # BOTH first two prescribed searches are executed once. Their shared
            # failure stops the later panel; no retry or parameter adjustment.
            if index <= 1 and not full:
                feasibility_failed = True
            if index == 1 and feasibility_failed:
                report['feasibility_gate'] = 'failed_first_pair_Q32'
                break
        output.json('searches.json', searches)
        outcomes = tuple(outcome_map[j.name] for j in search_jobs())
        report['completeness'] = primary_aggregate(outcomes)
        stage = begin_stage('final')
        check(stage)
        for artifact in selected.values():
            _artifact(root, artifact)
        seal = GlobalSeal(protocol.sha256, _read(root/'data_manifest.json')['identity'],
                          prepared['source_sha256'], outcomes, tuple(sorted(roles)),
                          tuple(sorted((key, row['sha256']) for key, row in selected.items())))
        output.json('global_seal.json', {'seal': asdict(seal), 'seal_sha256': seal.sha256,
            'artifacts': selected, 'frozen_at': time.perf_counter(),
            'source_record_sha256': sha256_file(root/'source_manifest.json')})
        # Persist all three permutations BEFORE the first timing measurement.
        timing_order = {}
        import random
        for window, seed in enumerate(WINDOW_SEEDS):
            rng, per_model = random.Random(seed), {}
            for model in MODELS:
                keys = [key for role, key in roles if role.startswith(model.name+'_')
                        and not role.endswith('dense_original') and not role.endswith('dense_recovery')]
                keys = sorted(set(keys))
                if len(keys) > 9:
                    raise ValueError('At most nine compact timing candidates per M0')
                rng.shuffle(keys)
                per_model[model.name] = keys
            timing_order[str(window)] = per_model
        output.json('timing_order.json', timing_order)
        quality = child('quality', 'quality', 'workers/quality', stage)
        if quality['status'] != 'completed':
            raise ControllerStop('quality_' + quality['status'])
        for row in quality['quality'].values():
            if 'torchscript' in row:
                row['torchscript'] = {**row['torchscript'],
                    'path': str(Path('workers/quality')/row['torchscript']['path'])}
            for name in ('eager', 'ts', 'test'):
                if name in row:
                    row[name]['prediction_file'] = str(Path('workers/quality')/row[name]['prediction_file'])
        output.json('quality.json', quality)
        stage = begin_stage('timing')
        windows = []
        for window in range(3):
            check(stage)
            result = child('timing', str(window), f'workers/timing_{window}',
                           min(stage, time.perf_counter()+960))
            # A/A failure is a completed diagnostic window. A timeout remains
            # partial; the next predeclared window is allowed only within cap.
            if result['status'] != 'completed':
                result['window_completed'] = False
                for row in result.get('models', {}).values():
                    for candidate in row.get('candidates', {}).values():
                        candidate['comparison_admitted'] = False
            windows.append(result)
            output.json('timing_windows.json', windows)
            if result['status'] == 'failed':
                raise ControllerStop('timing_integrity_failure')
        stage = begin_stage('close')
        check(stage)
        from .cv_independent_analysis import write_analysis
        report['analysis'] = write_analysis(root, seal, selected, quality['quality'], searches, controls, windows)
        report['status'] = ('completed' if report['completeness']['panel_complete']
                            else 'partial_search_panel')
        check(stage)
    except ControllerStop as error:
        report.update(status='partial', reason=str(error))
    except Exception as error:
        report.update(status='failed', failure_type=type(error).__name__, failure=str(error),
                      traceback=traceback.format_exc())
    finally:
        if active_stage:
            now = time.perf_counter()
            output.append({'event': 'stage_ended', 'stage': active_stage['name'],
                'timestamp': now, 'seconds': now-active_stage['started']})
        report['outcomes'] = [asdict(outcome_map[j.name]) for j in search_jobs()]
        report['ended_at'] = time.perf_counter()
        report['charged_slot_seconds'] = report['ended_at']-started
        report['within_deadline'] = report['ended_at'] <= absolute
        output.json('report.json', report)
        output.append({'event': 'series_stopped', 'status': report['status'], 'timestamp': report['ended_at']})
    return report


def main(argv=None, *, execute_started=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--absolute-deadline', type=float)
    parser.add_argument('--worker', choices=('manifest', 'baseline', 'controls', 'search', 'quality', 'timing'),
                        help=argparse.SUPPRESS)
    parser.add_argument('--job-id', help=argparse.SUPPRESS)
    parser.add_argument('--series-root', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    started = execute_started if execute_started is not None else time.perf_counter()
    if not args.execute:
        result = preparation(args.data_root, args.output_root, args.source_manifest)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0 if result['status'] == 'prepared' else 2
    if args.worker:
        if not args.series_root or not args.job_id or args.absolute_deadline is None:
            parser.error('Internal worker requires series root, job ID and inherited absolute deadline')
        result = _worker(args, started)
    else:
        result = run_series(args.data_root, args.output_root, args.source_manifest,
                            started=started, absolute_deadline=args.absolute_deadline)
    print(json.dumps({'status': result['status'], 'output': str(Path(args.output_root).resolve())}, allow_nan=False))
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main(execute_started=time.perf_counter()))
