"""One explicitly authorized completion of the stopped 2026-10-22 CV pilot.

The default command is an integrity-only preflight. No baseline training,
candidate evaluation, search or automatic retry is available in this entrypoint.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

from .cv_sparse_protocol import PilotProtocol, digest, eligible_finalists
from .cv_sparse_runtime import (Output, PilotFatal, PilotStop, Runtime, admit_free_gpu,
    checkpoint, evaluate, independent_torchscript, paired_latency, preparation,
    recover, recovery_tape, sha256_file, torchscript_artifact, verify_loaded_sources)

MAX_SECONDS = 1800
PRIOR_SECONDS = 3339.3260714150965
PARENT_INVENTORY_SHA = '5ec5d0dd4fe1d403ffa929e1b0904fde881562368040b2ec876a5e396d408a15'
PARENT_REPORT_SHA = '10ea02afb504252e925bedb9556712ba3b2d9964659e39c93691c3fc5a4af05d'
BASELINE_SHA = '531d02916a6a4ad7341f95a1f7e075c91c7b5350bc7009a09d5b93077c4d5510'
FINALISTS = {'random': '211ec99046465bfcf6bb3046f6798d2cba3a4cac6b2c09fae408074e9d010c40',
              'evolution': 'ee8d9a0d94d06a62cdd86e3837749454b79225b79f71cb0a653a16a972a7bd94'}


@dataclass(frozen=True)
class CompletionPlan:
    """Ordered deduplicated roles; only the pre-existing selected solutions."""
    selections: tuple[tuple[str, tuple[str, ...]], ...]
    finalists: tuple[tuple[str, str], ...]


def confined_file(root, relative):
    if type(relative) is not str or not relative or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('Parent artifacts require confined relative paths')
    target = (root/relative).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError('Parent artifact is missing or escapes its root')
    return target


def completion_plan(report):
    """Reuse the original eligibility/order, never rank by future test results."""
    if (report.get('status') != 'failed' or report.get('failure_type') != 'TypeError'
            or report.get('failure') != 'Object of type CVGenotype is not JSON serializable'
            or report.get('source_tensor_bytes') != 44734408 or report.get('byte_ceiling') != 35787526
            or report.get('protocol_sha256') != PilotProtocol().sha256
            or any(report.get(name) for name in ('recovered', 'official_test', 'finalist_solution_ids'))):
        raise ValueError('Only the pinned post-search/pre-recovery interruption can be completed')
    records = report.get('all_short_attempts', {})
    if (set(records) != {'random', 'evolution', 'fixed_prune', 'fixed_svd', 'fixed_hybrid'}
            or {name: len(rows) for name, rows in records.items()} != {
                'random':12, 'evolution':12, 'fixed_prune':1, 'fixed_svd':1, 'fixed_hybrid':1}):
        raise ValueError('All 27 original short evaluations are required')
    by_id = {}
    for rows in records.values():
        for row in rows:
            if (row.get('status') != 'succeeded' or row.get('completed_within_deadline') is not True
                    or row.get('quality_accepted') is not True or digest(row.get('genotype')) != row.get('solution_id')
                    or not row['started_at'] <= row['ended_at'] <= row['deadline']):
                raise ValueError('Original candidate metadata or original monotonic deadline is invalid')
            if row.get('run') == 'evolution' and not row['ended_at'] <= row['common_search_completed_at'] <= row['deadline']:
                raise ValueError('Evolution completion must meet its original authoritative deadline')
            by_id[row['solution_id']] = row
    winners = {run: eligible_finalists(records[run], byte_ceiling=report['byte_ceiling'])[0]['solution_id']
               for run in ('random', 'evolution')}
    if winners != FINALISTS:
        raise ValueError('The pinned source no longer selects the approved R/E pair')
    dense = report['baseline']['solution_id']
    selected = {dense: ['dense_recovery']}
    for name in ('fixed_prune', 'fixed_svd', 'fixed_hybrid'):
        selected.setdefault(records[name][0]['solution_id'], []).append(name)
    for name, identity in winners.items():
        selected.setdefault(identity, []).append(name+'_finalist')
    if len(selected) != 5 or selected[winners['random']] != ['fixed_svd', 'random_finalist']:
        raise ValueError('Exactly five approved recovery solutions must remain after role deduplication')
    return CompletionPlan(tuple((key, tuple(roles)) for key,roles in selected.items()), tuple(winners.items()))


def inspect_parent(parent_root, inventory_path):
    """Check the independently published inventory, including every old binary."""
    root, inventory_path = Path(parent_root).resolve(), Path(inventory_path).resolve()
    if sha256_file(inventory_path) != PARENT_INVENTORY_SHA:
        raise ValueError('Parent inventory differs from the reviewed published evidence')
    inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
    verified = {}
    for name, expected in sorted(inventory['files'].items()):
        target = confined_file(root, name)
        if target.stat().st_size != expected['bytes'] or sha256_file(target) != expected['sha256']:
            raise ValueError(f'Parent artifact integrity failure: {name}')
        verified[name] = expected
    report = json.loads(confined_file(root, 'report.json').read_text(encoding='utf-8'))
    if (sha256_file(root/'report.json') != PARENT_REPORT_SHA
            or report['baseline']['sha256'] != BASELINE_SHA
            or sha256_file(confined_file(root, report['baseline']['path'])) != BASELINE_SHA):
        raise ValueError('Pinned report or full M0 has changed')
    source = json.loads(confined_file(root, 'source_manifest.json').read_text())['supplied']
    if (source['source_commit'] != '3bc0825a9ef1bcd0ac8eb0181b6d5314b57ebb8d'
            or source['wheel_sha256'] != '26fcf56c64c8b68dd6aa00879b16dc926193ed7f4159a01fc3b8e42e069fb913'
            or source['protocol_sha256'] != PilotProtocol().sha256):
        raise ValueError('Parent wheel, code or protocol lineage changed')
    plan = completion_plan(report)
    return {'parent_root': str(root), 'inventory_sha256': PARENT_INVENTORY_SHA,
            'report_sha256': PARENT_REPORT_SHA, 'verified_files': verified,
            'plan': {'selections': plan.selections, 'finalists': plan.finalists}, 'report': report}


def continuation_preparation(data_root, output_root, manifest, parent_root, inventory):
    root, output = Path(parent_root).resolve(), Path(output_root).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Continuation and immutable parent require disjoint output trees')
    prepared = preparation(data_root, output_root, manifest, PilotProtocol())
    parent = inspect_parent(root, inventory)
    if 'fedcore/experiments/cv_sparse_continuation.py' not in prepared['source']['verified_package_files_sha256']:
        raise ValueError('The new installed wheel must include the completion entrypoint')
    return {**prepared, 'parent': parent, 'continuation_seconds': MAX_SECONDS,
            'prior_charged_seconds': PRIOR_SECONDS, 'automatic_retry': False}


@contextmanager
def completion_phase(runtime, name, seconds):
    """Explicit narrower per-stage windows within the external 30-minute slot."""
    runtime.stage, runtime.stage_started = name, time.perf_counter()
    runtime.stage_limit = min(runtime.deadline, runtime.stage_started+seconds)
    runtime.output.append({'event':'stage_started', 'stage':name, 'started_at':runtime.stage_started,
                           'deadline':runtime.stage_limit, 'continuation_stage_seconds':seconds})
    runtime.check()
    try:
        yield runtime.stage_limit
    finally:
        runtime.output.append({'event':'stage_ended', 'stage':name, 'ended_at':time.perf_counter(),
                               'seconds':time.perf_counter()-runtime.stage_started})


def verify_baseline_predictions(previous_path, current_path):
    import numpy as np
    with np.load(previous_path, allow_pickle=False) as old, np.load(current_path, allow_pickle=False) as new:
        if set(old.files) != set(new.files):
            raise ValueError('Baseline prediction schema changed')
        for key in old.files:
            if old[key].dtype.kind == 'f':
                np.testing.assert_allclose(new[key], old[key], rtol=1e-6, atol=1e-7)
            else:
                np.testing.assert_array_equal(new[key], old[key])


def frozen_dependencies(text):
    """Only the explicitly replaced FedCore wheel may differ in pip freeze."""
    return tuple(sorted(line.strip() for line in text.splitlines()
                        if line.strip() and not line.lstrip().startswith('#')
                        and not re.match(r'^fedcore(?:==| @ )', line.strip(), re.IGNORECASE)))


def run_completion(data_root, output_root, manifest, parent_root, inventory, *, started=None):
    started = time.perf_counter() if started is None else started
    protocol = PilotProtocol()
    prepared = continuation_preparation(data_root, output_root, manifest, parent_root, inventory)
    if prepared['status'] != 'prepared':
        raise ValueError('Prepared official CIFAR files are required')
    output = Output(output_root)
    report = {'schema':'fedcore_sparse_cv_completion_v1', 'status':'started', 'started_at':started,
              'protocol_sha256':protocol.sha256, 'continuation_seconds':MAX_SECONDS,
              'prior_charged_seconds':PRIOR_SECONDS, 'automatic_retry':False,
              'baseline_training_performed':False, 'search_performed':False,
              'parent_report_sha256':PARENT_REPORT_SHA, 'parent_inventory_sha256':PARENT_INVENTORY_SHA}
    output.json('protocol.json', protocol.payload())
    output.json('source_manifest.json', prepared['source'])
    output.json('parent_admission.json', prepared['parent'])
    runtime, pilot = None, None
    old = prepared['parent']['report']
    parent = Path(parent_root).resolve()
    try:
        report['gpu_admission'] = admit_free_gpu()
        runtime = Runtime(protocol, output, started)
        runtime.deadline = min(runtime.deadline, started+MAX_SECONDS-10)
        report['absolute_deadline'] = runtime.deadline
        with completion_phase(runtime, 'manifest', 300) as deadline:
            from .cv_sparse_data import CifarData
            from .cv_sparse_pilot import Pilot, solution_id
            from fedcore.tools.registry.checkpoint_manager import CheckpointManager
            from fedcore.algorithm.low_rank.cv_grammar import CVGenotype, cv_genotype_from_payload
            data = CifarData(data_root, protocol, download=False)
            if (data.roles.payload() != json.loads((parent/'data_roles.json').read_text())
                    or data.source_hashes != old['data']['source_files']):
                raise ValueError('Dataset bytes or roles changed since the original search')
            previous_environment = json.loads((parent/'environment.json').read_text())
            if runtime.environment != previous_environment:
                raise ValueError('The scientific runtime environment differs from the original pilot')
            pilot = Pilot(runtime, data)
            report['environment'] = runtime.environment
            report['data'] = {'roles_sha256': data.roles.sha256, 'source_files':data.source_hashes}
            output.json('environment.json', runtime.environment)
            output.json('data_roles.json', data.roles.payload())
            freeze = subprocess.run([sys.executable,'-m','pip','freeze'],
                capture_output=True,text=True,check=True,timeout=30).stdout
            if frozen_dependencies(freeze) != frozen_dependencies((parent/'environment_freeze.txt').read_text()):
                raise ValueError('Scientific dependency versions differ from the original pilot')
            output.bytes('environment_freeze.txt', freeze.encode())
            baseline = dict(old['baseline'])
            copied = output.bytes(baseline['path'], confined_file(parent,baseline['path']).read_bytes())
            if copied['sha256'] != BASELINE_SHA:
                raise ValueError('Full M0 copy changed bytes')
            evidence = baseline['validation']
            copied_predictions = output.bytes(evidence['prediction_file'],
                confined_file(parent, evidence['prediction_file']).read_bytes())
            if copied_predictions['sha256'] != evidence['prediction_sha256']:
                raise ValueError('Original full M0 prediction evidence changed during copy')
            model = CheckpointManager('.', auto_cleanup=False).deserialize_from_bytes(
                (output.root/baseline['path']).read_bytes(), device='cpu').eval()
            metrics = evaluate(model.to('cuda'), pilot.gpu, data.roles.search_validation,
                               deadline=deadline, label='continuation_baseline_validation')
            verify_baseline_predictions(parent/old['baseline']['validation']['prediction_file'],
                                        output.root/metrics['prediction_file'])
            if metrics['top1'] != baseline['validation']['top1']:
                raise ValueError('Full M0 prediction quality changed')
            model.cpu()
            pilot.bind_teacher(model)
            if pilot.space.source_storage_bytes != old['source_tensor_bytes'] or pilot.byte_ceiling != old['byte_ceiling']:
                raise ValueError('Full M0 physical storage changed')
            pilot.genotypes[solution_id(pilot.prefix[0])] = pilot.prefix[0]
            for records in old['all_short_attempts'].values():
                for record in records:
                    genotype = cv_genotype_from_payload(record['genotype'], pilot.space)
                    if type(genotype) is not CVGenotype or solution_id(genotype) != record['solution_id']:
                        raise ValueError('Original genotype is no longer valid in the original model domain')
                    pilot.genotypes[record['solution_id']] = genotype
            report.update(baseline=baseline, baseline_recheck=metrics, source_tensor_bytes=pilot.space.source_storage_bytes,
                          byte_ceiling=pilot.byte_ceiling, inherited_short_attempts=old['all_short_attempts'])
            output.json('baseline.json', baseline)
            output.json('imported_sources.json', verify_loaded_sources(prepared['source']))
            baseline_ts, ts_record = torchscript_artifact(baseline, pilot.gpu, deadline)
            independent = independent_torchscript(ts_record, pilot.gpu, deadline)
            initial_aa = paired_latency(baseline_ts, independent, pilot.gpu, deadline=deadline,
                                        label='initial_dense_AA', aa=True)
            del baseline_ts, independent
            report['initial_aa'] = initial_aa
        plan = completion_plan(old)
        report['finalist_solution_ids'] = dict(plan.finalists)
        output.json('recovery_selection.json', {'roles':dict(plan.selections), 'finalists':dict(plan.finalists),
                                               'source_report_sha256':PARENT_REPORT_SHA})
        tape = recovery_tape(data.roles.train, protocol)
        output.json('recovery_tape.json', {'sha256':digest(tape), 'updates':[
            {'row_ids':list(ids), 'augmentation_seed':seed} for ids,seed in tape]})
        recovered = {}
        report['recovered'] = recovered
        with completion_phase(runtime, 'recovery', 600) as deadline:
            from fedcore.algorithm.low_rank.cv_execution import CVWorkflowSuccess
            for key,roles in plan.selections:
                runtime.check(deadline, recovery_solution=key)
                result = pilot.build(pilot.genotypes[key])
                if type(result) is not CVWorkflowSuccess:
                    raise PilotFatal(f'Selected recovery graph failed: {type(result).__name__}')
                metrics = recover(result.model, pilot.gpu, tape, deadline=deadline, role=key)
                artifact, restored = checkpoint(result.model, pilot.gpu, label=f'recovered_{key}',
                    solution_id=key, phase='recovered_200_updates', deadline=deadline)
                artifact.update(roles=list(roles), validation=metrics, recovery_updates=protocol.recovery_updates,
                                recovery_tape_sha256=digest(tape))
                recovered[key] = artifact
                output.json(f'recovery/{key}.json', artifact)
                del result, restored
        with completion_phase(runtime, 'final', 600) as deadline:
            from .cv_sparse_pilot import finalize_pilot
            finalize_pilot(pilot, baseline, recovered, initial_aa, report, deadline)
        with completion_phase(runtime, 'close', 300):
            from fedcore.algorithm.low_rank.candidate_storage import capture_inventory, inventory_fingerprint
            if inventory_fingerprint(capture_inventory(pilot.teacher)) != pilot.teacher_fingerprint:
                raise ValueError('Frozen M0 changed')
            if inspect_parent(parent, inventory)['verified_files'] != prepared['parent']['verified_files']:
                raise ValueError('Parent evidence changed during completion')
            output.json('imported_sources.json', verify_loaded_sources(prepared['source']))
            runtime.check()
            report['status'] = 'completed'
    except PilotStop as error:
        report.update(status='pilot_incomplete_'+error.stage, stop_reason=error.reason)
    except (Exception, PilotFatal) as error:
        report.update(status='failed', failure_type=type(error).__name__, failure=str(error), traceback=traceback.format_exc())
    finally:
        report['ended_at'] = time.perf_counter()
        report['wall_seconds'] = report['ended_at']-started
        report['charged_slot_seconds'] = report['wall_seconds']
        report['total_charged_seconds'] = PRIOR_SECONDS+report['wall_seconds']
        if runtime is not None:
            report.update(peak_allocated_bytes=runtime.torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=runtime.torch.cuda.max_memory_reserved(),
                          within_slot_deadline=report['ended_at']<=runtime.deadline)
        output.json('report.json', report)
        output.append({'event':'completion_stopped', 'status':report['status'], 'ended_at':report['ended_at']})
    return report


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','output-root','source-manifest','parent-root','parent-inventory'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    values = (args.data_root,args.output_root,args.source_manifest,args.parent_root,args.parent_inventory)
    if not args.execute:
        result = continuation_preparation(*values)
        print(json.dumps({'status':result['status'],'execute':False,'parent_files':len(result['parent']['verified_files']),
                          'recovery_models':len(result['parent']['plan']['selections']), 'continuation_seconds':MAX_SECONDS}))
        return 0 if result['status']=='prepared' else 2
    result = run_completion(*values, started=started)
    print(json.dumps({'status':result['status'],'output':args.output_root}))
    return 0 if result['status']=='completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
