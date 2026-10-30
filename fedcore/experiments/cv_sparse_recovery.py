"""One fixed recovery/extra-BN matrix on immutable completion evidence.

Without --execute this command uses the standard library for integrity preflight
only. The 30 quality variants do not open the official test, search, or timing.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import cv_sparse_recovery_protocol as recovery
from .cv_sparse_continuation import completion_phase, confined_file, frozen_dependencies
from .cv_sparse_diagnostics import recover_without_calibration
from .cv_sparse_protocol import PilotProtocol, digest
from .cv_sparse_runtime import (Output, PilotFatal, PilotStop, Runtime, admit_free_gpu,
    calibrate_bn, checkpoint, evaluate, preparation, recovery_tape, sha256_file,
    verify_loaded_sources)


@dataclass(frozen=True)
class QualityVariant:
    family: str
    solution_id: str
    tensor_bytes: int
    state: str
    lr: float | None
    updates: int
    extra_bn: bool
    label: str
    diagnostic_label: str | None


def quality_plan():
    """Five graph choices, three states and two BN modes, in fixed order."""
    rows = []
    for family, identity, storage in recovery.FAMILIES:
        for lr in (None, *recovery.LEARNING_RATES):
            state = 'no_SGD' if lr is None else f'SGD_lr_{lr:g}'
            for bn in (False, True):
                suffix = 'post_BN' if bn else 'pre_BN'
                label = f'{family}_{state}_{suffix}'
                old_label = None
                if family == 'M0':
                    old_label = ('M0_post_BN' if bn else 'M0_untouched') if lr is None else f'{state}_{suffix}'
                rows.append(QualityVariant(family, identity, storage, state, lr,
                                           0 if lr is None else 200, bn, label, old_label))
    return tuple(rows)


def _verify_file(root, files, name):
    target = confined_file(root, name)
    expected = files.get(name)
    if (type(expected) is not dict or type(expected.get('bytes')) is not int
            or target.stat().st_size != expected['bytes']
            or sha256_file(target) != expected.get('sha256')):
        raise ValueError(f'Parent artifact integrity failure: {name}')
    return dict(expected)


def _parent_inventory(root, inventory_path, inventory_sha, report_sha):
    root, inventory_path = Path(root).resolve(), Path(inventory_path).resolve()
    if sha256_file(inventory_path) != inventory_sha:
        raise ValueError('Parent inventory differs from the pinned evidence')
    files = json.loads(inventory_path.read_text(encoding='utf-8')).get('files')
    if type(files) is not dict:
        raise ValueError('Parent inventory requires a file map')
    verified = {'report.json': _verify_file(root, files, 'report.json')}
    if verified['report.json']['sha256'] != report_sha:
        raise ValueError('Parent report differs from the pinned evidence')
    report = json.loads(confined_file(root, 'report.json').read_text(encoding='utf-8'))
    if (report.get('status') != 'completed' or report.get('within_slot_deadline') is not True
            or report.get('protocol_sha256') != PilotProtocol().sha256
            or report.get('baseline_training_performed') is not False
            or report.get('search_performed') is not False):
        raise ValueError('Only completed, timely parent evidence is admitted')
    for name in ('environment.json', 'environment_freeze.txt', 'data_roles.json'):
        verified[name] = _verify_file(root, files, name)
    return root, files, report, verified


def selected_payloads(report):
    """Recover the original graph choices without invoking a search or ranking."""
    expected_roles = {'M0': ['dense_recovery'], 'P75': ['fixed_prune'],
                      'R': ['fixed_svd', 'random_finalist'],
                      'E': ['evolution_finalist'], 'hybrid': ['fixed_hybrid']}
    selected = {identity for _, identity, _ in recovery.FAMILIES}
    if set(report.get('recovered', {})) != selected:
        raise ValueError('The five original deduplicated recovery graph choices changed')
    records = report.get('inherited_short_attempts', {})
    if set(records) != {'random', 'evolution', 'fixed_prune', 'fixed_svd', 'fixed_hybrid'}:
        raise ValueError('The original genotype payloads are required')
    payloads = {}
    for rows in records.values():
        for row in rows:
            identity = row.get('solution_id')
            if identity in selected:
                payload = row.get('genotype')
                if digest(payload) != identity or (identity in payloads and payloads[identity] != payload):
                    raise ValueError('Original genotype payload differs from its pinned graph choice')
                payloads[identity] = payload
    if set(payloads) != selected:
        raise ValueError('Every pinned graph choice requires its original genotype payload')
    for family, identity, storage in recovery.FAMILIES:
        record = report['recovered'][identity]
        if (record.get('solution_id') != identity or record.get('roles') != expected_roles[family]
                or record.get('tensor_bytes') != storage or record.get('recovery_updates') != 200):
            raise ValueError('Original recovery roles, graph identity or physical bytes changed')
    fixed = {'fixed_prune': 'P75', 'fixed_svd': 'R', 'fixed_hybrid': 'hybrid'}
    identities = {family: identity for family, identity, _ in recovery.FAMILIES}
    for role, family in fixed.items():
        if len(records[role]) != 1 or records[role][0].get('solution_id') != identities[family]:
            raise ValueError('The original fixed graph choices changed')
    return payloads


def inspect_completion(completion_root, completion_inventory):
    root, files, report, verified = _parent_inventory(completion_root, completion_inventory,
        recovery.COMPLETION_INVENTORY_SHA, recovery.COMPLETION_REPORT_SHA)
    baseline = report['baseline']
    _, identity, storage = recovery.FAMILIES[0]
    if (baseline.get('sha256') != recovery.BASELINE_SHA or baseline.get('solution_id') != identity
            or baseline.get('tensor_bytes') != storage or baseline.get('phase') != 'dense_original'):
        raise ValueError('The original full M0 identity changed')
    payloads = selected_payloads(report)
    for name in (baseline['path'], baseline['validation']['prediction_file']):
        verified[name] = _verify_file(root, files, name)
    if (verified[baseline['path']]['sha256'] != recovery.BASELINE_SHA
            or verified[baseline['validation']['prediction_file']]['sha256'] != baseline['validation']['prediction_sha256']
            or baseline['validation'].get('rows') != 5000):
        raise ValueError('Original full M0 bytes or predictions changed')
    return {'root': str(root), 'report_sha256': recovery.COMPLETION_REPORT_SHA,
            'inventory_sha256': recovery.COMPLETION_INVENTORY_SHA,
            'verified_files': verified, 'report': report, 'genotype_payloads': payloads}


def inspect_diagnostic(diagnostic_root, diagnostic_inventory):
    from . import cv_sparse_diagnostic_protocol as previous
    root, files, report, verified = _parent_inventory(diagnostic_root, diagnostic_inventory,
        recovery.DIAGNOSTIC_INVENTORY_SHA, recovery.DIAGNOSTIC_REPORT_SHA)
    if (report.get('diagnostic_sha256') != previous.sha256()
            or report.get('completion_report_sha256') != recovery.COMPLETION_REPORT_SHA
            or report.get('completion_inventory_sha256') != recovery.COMPLETION_INVENTORY_SHA
            or report.get('official_test_access') is not False):
        raise ValueError('Diagnostic lineage or scientific protocol changed')
    dense = [item for item in quality_plan() if item.family == 'M0']
    if set(report.get('quality', {})) != {item.diagnostic_label for item in dense}:
        raise ValueError('The diagnostic must contain exactly the six dense conditions')
    predictions = {}
    for variant in dense:
        item = report['quality'][variant.diagnostic_label]
        if (item.get('solution_id') != variant.solution_id or item.get('tensor_bytes') != variant.tensor_bytes
                or type(item.get('SGD_updates')) is not int or item['SGD_updates'] != variant.updates
                or item.get('lr') != variant.lr or item.get('additional_BN_calibration') is not variant.extra_bn
                or item.get('validation', {}).get('rows') != 5000):
            raise ValueError('A diagnostic dense condition changed')
        evidence = item['validation']
        name = evidence['prediction_file']
        verified[name] = _verify_file(root, files, name)
        if verified[name]['sha256'] != evidence['prediction_sha256']:
            raise ValueError('Diagnostic prediction metadata differs from its inventory')
        predictions[variant.diagnostic_label] = name
    return {'root': str(root), 'report_sha256': recovery.DIAGNOSTIC_REPORT_SHA,
            'inventory_sha256': recovery.DIAGNOSTIC_INVENTORY_SHA,
            'verified_files': verified, 'report': report, 'predictions': predictions}


def recovery_preparation(data_root, output_root, manifest, completion_root,
                         completion_inventory, diagnostic_root, diagnostic_inventory):
    output = Path(output_root).resolve()
    parents = (Path(completion_root).resolve(), Path(diagnostic_root).resolve())
    for parent in parents:
        if output.is_relative_to(parent) or parent.is_relative_to(output):
            raise ValueError('Recovery output and immutable parents require disjoint trees')
    for inventory in (completion_inventory, diagnostic_inventory):
        if Path(inventory).resolve().is_relative_to(output):
            raise ValueError('Parent inventory cannot lie in the recovery output tree')
    prepared = preparation(data_root, output_root, manifest, PilotProtocol())
    required = {'fedcore/experiments/'+name+'.py' for name in (
        'cv_sparse_continuation', 'cv_sparse_diagnostics', 'cv_sparse_diagnostic_protocol',
        'cv_sparse_recovery', 'cv_sparse_recovery_protocol')}
    if not required.issubset(prepared['source']['verified_package_files_sha256']):
        raise ValueError('Source manifest must include every recovery execution module')
    if prepared['source']['supplied'].get('recovery_protocol_sha256') != recovery.sha256():
        raise ValueError('Source manifest must bind the fixed recovery protocol')
    completion = inspect_completion(parents[0], completion_inventory)
    diagnostic = inspect_diagnostic(parents[1], diagnostic_inventory)
    if any(completion['report'][key] != diagnostic['report'][key] for key in ('environment', 'data')):
        raise ValueError('The two immutable parents disagree about data or environment')
    return {**prepared, 'completion': completion, 'diagnostic': diagnostic,
            'recovery': recovery.payload(), 'recovery_sha256': recovery.sha256(), 'automatic_retry': False}


def decode_genotypes(payloads, pilot):
    from fedcore.algorithm.low_rank.cv_grammar import CVGenotype, cv_genotype_from_payload
    from .cv_sparse_pilot import solution_id
    result = {}
    for family, identity, _ in recovery.FAMILIES:
        genotype = cv_genotype_from_payload(payloads[identity], pilot.space)
        if type(genotype) is not CVGenotype or solution_id(genotype) != identity:
            raise ValueError('The original genotype is invalid in the original M0 domain')
        result[family] = genotype
    return result


def tensor_evidence(named_tensors):
    """Hash exact CPU tensor bytes, shape and dtype without retaining copies."""
    import torch
    result = {}
    for name, value in named_tensors:
        tensor = value.detach().cpu().contiguous()
        result[name] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                        'sha256': hashlib.sha256(tensor.reshape(-1).view(dtype=torch.uint8).numpy().tobytes()).hexdigest()}
    return result


def model_evidence(model):
    return {'parameters': tensor_evidence(model.named_parameters()),
            'buffers': tensor_evidence(model.named_buffers())}


def load_checkpoint(artifact, gpu, deadline):
    from fedcore.tools.registry.checkpoint_manager import CheckpointManager
    gpu.runtime.check(deadline, checkpoint='recovery_reload')
    path = confined_file(gpu.runtime.output.root, artifact['path'])
    if sha256_file(path) != artifact['sha256'] or path.stat().st_size != artifact['file_bytes']:
        raise ValueError('Recovery snapshot changed before independent reload')
    model = CheckpointManager('.', auto_cleanup=False).deserialize_from_bytes(path.read_bytes(), device='cpu').eval()
    gpu.runtime.check(deadline)
    return model


def verify_dense_predictions(previous_path, current_path):
    """Fail closed on every row, dtype and exact CE value; no fitted tolerance."""
    import numpy as np
    expected = {'row_ids', 'labels', 'predictions', 'cross_entropy'}
    with np.load(previous_path, allow_pickle=False) as old, np.load(current_path, allow_pickle=False) as new:
        if set(old.files) != expected or set(new.files) != expected:
            raise ValueError('Dense replay prediction schema changed')
        for name in sorted(expected):
            left, right = old[name], new[name]
            dtype = np.dtype('float64' if name == 'cross_entropy' else 'int64')
            if (left.shape != (5000,) or left.dtype != dtype or left.dtype != right.dtype
                    or not np.array_equal(left, right)
                    or (left.dtype.kind == 'f' and not np.isfinite(left).all())):
                raise ValueError(f'Dense replay mismatch: {name}')
    return {'rows': 5000, 'fields': sorted(expected), 'exact': True,
            'previous_sha256': sha256_file(previous_path), 'current_sha256': sha256_file(current_path)}


def build_start(pilot, genotype, family, gpu, deadline):
    from fedcore.algorithm.low_rank.cv_execution import CVWorkflowSuccess
    from fedcore.algorithm.low_rank.candidate_storage import capture_inventory, inventory_fingerprint
    identity, storage = next((key, size) for role, key, size in recovery.FAMILIES if role == family)
    before = model_evidence(pilot.teacher)
    gpu.runtime.check(deadline, build_family=family)
    result = pilot.build(genotype)
    if type(result) is not CVWorkflowSuccess or result.evidence.get('storage_bytes') != storage:
        raise ValueError('The pinned public CV graph build or physical cost changed')
    if inventory_fingerprint(capture_inventory(pilot.teacher)) != pilot.teacher_fingerprint:
        raise ValueError('Public build changed the original frozen M0')
    after = model_evidence(result.model)
    counters = [name for name in before['buffers'] if name.endswith('num_batches_tracked')]
    if not counters or any(after['buffers'].get(name) != before['buffers'][name] for name in counters):
        raise ValueError('Public build performed hidden training or BN calibration')
    artifact, restored = checkpoint(result.model, gpu, label=f'{family}_uncalibrated_start',
        solution_id=identity, phase='recovery_uncalibrated_start', deadline=deadline)
    if artifact['tensor_bytes'] != storage or model_evidence(restored) != after:
        raise ValueError('The common initial snapshot changed during persistence')
    artifact.update(family=family, no_extra_BN_before_training=True,
                    source_M0_sha256=recovery.BASELINE_SHA, workflow_fingerprint=result.workflow.fingerprint,
                    initial_state=after, initial_state_sha256=digest(after))
    gpu.runtime.output.json(f'starts/{family}.json', artifact)
    return artifact


def run_bn_pair(source, variants, gpu, tape, *, deadline, training, results, bn_pairs,
                diagnostic=None, baseline_predictions=None):
    """Both branches reload the same saved pre-BN state independently."""
    if len(variants) != 2 or [item.extra_bn for item in variants] != [False, True]:
        raise ValueError('An extra-BN pair requires its two ordered conditions')
    output = gpu.runtime.output
    baseline_state = None
    before_item = None
    for variant in variants:
        model = load_checkpoint(source, gpu, deadline)
        state = model_evidence(model)
        if baseline_state is None:
            baseline_state = state
        elif state != baseline_state:
            raise ValueError('Extra-BN branches did not reload the identical pre-BN snapshot')
        if variant.extra_bn:
            calibrate_bn(model.to('cuda'), gpu, deadline)
            model.cpu()
        after = model_evidence(model)
        if after['parameters'] != baseline_state['parameters']:
            raise ValueError('Extra-BN calibration changed learnable parameters')
        if not variant.extra_bn and after != baseline_state:
            raise ValueError('The no-extra-BN branch changed its saved state')
        artifact = dict(source)
        if variant.extra_bn:
            artifact, restored = checkpoint(model, gpu, label=variant.label,
                solution_id=variant.solution_id, phase='recovery_'+variant.label, deadline=deadline)
            del restored
        if artifact['tensor_bytes'] != variant.tensor_bytes or artifact['solution_id'] != variant.solution_id:
            raise ValueError('The quality variant changed its pinned graph identity or cost')
        metrics = evaluate(model.to('cuda'), gpu, gpu.data.roles.search_validation,
                           deadline=deadline, label=variant.label)
        model.cpu()
        if metrics['rows'] != 5000 or model_evidence(model) != after:
            raise ValueError('Quality evaluation changed state or validation workload')
        item = {**artifact, 'family':variant.family, 'state':variant.state,
                'validation':metrics, 'SGD_updates':variant.updates, 'lr':variant.lr,
                'additional_BN_calibration':variant.extra_bn, 'training':training,
                'shared_pre_BN_checkpoint_sha256':source['sha256'],
                'initial_checkpoint_sha256':source['initial_checkpoint_sha256'],
                'recovery_tape_sha256':digest(tape),
                'source_M0_sha256':recovery.BASELINE_SHA,
                'parameter_fingerprint':digest(after['parameters']),
                'buffer_fingerprint':digest(after['buffers'])}
        results[variant.label] = item
        output.json(f'quality/{variant.label}.json', item)
        if diagnostic is not None:
            parent = Path(diagnostic['root'])
            previous = diagnostic['predictions'][variant.diagnostic_label]
            item['dense_replay'] = verify_dense_predictions(confined_file(parent, previous),
                confined_file(output.root, metrics['prediction_file']))
            if baseline_predictions is not None and variant.lr is None and not variant.extra_bn:
                item['original_M0_replay'] = verify_dense_predictions(baseline_predictions,
                    confined_file(output.root, metrics['prediction_file']))
            output.json(f'quality/{variant.label}.json', item)
        if variant.extra_bn:
            changed = {name: after['buffers'][name] != baseline_state['buffers'][name]
                       for name in baseline_state['buffers']}
            if any(value for name, value in changed.items() if not name.endswith(
                    ('running_mean', 'running_var', 'num_batches_tracked'))):
                raise ValueError('Extra-BN changed a buffer outside BN running statistics')
            proof = {'family':variant.family, 'state':variant.state,
                'shared_pre_BN_checkpoint_sha256':source['sha256'], 'independent_reloads':True,
                'parameters_exactly_unchanged':True, 'parameter_evidence':after['parameters'],
                'buffers_before':baseline_state['buffers'], 'buffers_after':after['buffers'],
                'buffers_changed':changed, 'calibration_rows':1000,
                'reset_running_stats':True, 'momentum_during_calibration':None,
                'no_BN_variant':before_item['validation']['prediction_file'],
                'extra_BN_variant':metrics['prediction_file']}
            bn_pairs[f'{variant.family}_{variant.state}'] = proof
            output.json(f'BN_pairs/{variant.family}_{variant.state}.json', proof)
        else:
            before_item = item
        del model
        if sha256_file(confined_file(output.root, source['path'])) != source['sha256']:
            raise ValueError('The saved pre-BN snapshot changed during a BN pair')


def run_quality(pilot, genotypes, tape, *, deadline, diagnostic, baseline_predictions,
                results, starts, trajectories, bn_pairs):
    gpu, plan = pilot.gpu, quality_plan()
    if len(gpu.data.roles.search_validation) != 5000 or len(gpu.data.roles.calibration) != 1000:
        raise ValueError('Recovery requires the original 5000 validation and 1000 calibration rows')
    if len(tape) != 200 or any(len(ids) != 128 for ids, _ in tape):
        raise ValueError('Recovery requires the original 200-by-128 training tape')
    for family, _, _ in recovery.FAMILIES:
        # M0's six comparisons complete before even building a compressed graph.
        if family != 'M0' and not all(results.get(item.label, {}).get('dense_replay', {}).get('exact') is True
                                     for item in plan if item.family == 'M0'):
            raise ValueError('All six exact dense replays must pass before compressed families')
        initial = build_start(pilot, genotypes[family], family, gpu, deadline)
        initial['initial_checkpoint_sha256'] = initial['sha256']
        starts[family] = initial
        for lr in (None, *recovery.LEARNING_RATES):
            variants = tuple(item for item in plan if item.family == family and item.lr == lr)
            source = dict(initial)
            training = {'updates':0, 'lr':None, 'additional_BN_calibration':False,
                        'training_BN_updates':False}
            if lr is not None:
                model = load_checkpoint(initial, gpu, deadline)
                before = model_evidence(model)
                if before != initial['initial_state']:
                    raise ValueError('A recovery trajectory did not start from the common initial weights')
                role = f'{family}_{variants[0].state}'
                training = recover_without_calibration(model, gpu, tape, deadline=deadline, role=role, lr=lr)
                after = model_evidence(model)
                changed = {name: value != before['parameters'][name] for name, value in after['parameters'].items()}
                training.update(initial_checkpoint_sha256=initial['sha256'],
                    initial_parameter_fingerprint=digest(before['parameters']),
                    final_parameter_fingerprint=digest(after['parameters']),
                    changed_parameter_tensors=changed, all_parameter_tensors_changed=all(changed.values()),
                    all_parameters_received_finite_gradients=True)
                trajectories[role] = training
                gpu.runtime.output.json(f'training/{role}.json', training)
                source, restored = checkpoint(model, gpu, label=variants[0].label,
                    solution_id=variants[0].solution_id, phase='recovery_'+variants[0].label, deadline=deadline)
                source['initial_checkpoint_sha256'] = initial['sha256']
                if source['tensor_bytes'] != variants[0].tensor_bytes:
                    raise ValueError('Recovery changed the pinned physical model bytes')
                del model, restored
            run_bn_pair(source, variants, gpu, tape, deadline=deadline, training=training,
                results=results, bn_pairs=bn_pairs,
                diagnostic=diagnostic if family == 'M0' else None,
                baseline_predictions=baseline_predictions if family == 'M0' else None)
        gpu.runtime.check(deadline, completed_family=family)
    if (len(results) != 30 or len(trajectories) != 10 or len(bn_pairs) != 15
            or sum(row['updates'] for row in trajectories.values()) != 2000
            or sum(row['validation']['rows'] for row in results.values()) != 150000):
        raise ValueError('The approved recovery matrix workload changed')


def count_training_updates(ledger):
    """Return the verified event prefix even if the ledger ends malformed."""
    expected = {f'{family}_SGD_lr_{lr:g}':lr
                for family, _, _ in recovery.FAMILIES for lr in recovery.LEARNING_RATES}
    observed = {}
    if not Path(ledger).exists():
        return observed, None
    try:
        with Path(ledger).open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                event = json.loads(line)
                if type(event) is not dict:
                    raise ValueError(f'Non-object event at ledger line {line_number}')
                if event.get('event') != 'diagnostic_SGD_update':
                    continue
                role, update = event.get('trajectory'), event.get('updates')
                if (type(role) is not str or role not in expected or type(update) is not int
                        or update != observed.get(role, 0)+1 or update > 200
                        or type(event.get('lr')) is not float or event['lr'] != expected[role]):
                    raise ValueError(f'Invalid, duplicated or out-of-budget SGD event at ledger line {line_number}')
                observed[role] = update
    except (OSError, ValueError, UnicodeError) as error:
        return observed, f'{type(error).__name__}: {error}'
    return observed, None


def run_recovery(data_root, output_root, manifest, completion_root, completion_inventory,
                 diagnostic_root, diagnostic_inventory, *, started=None):
    started = time.perf_counter() if started is None else started
    protocol = PilotProtocol()
    prepared = recovery_preparation(data_root, output_root, manifest, completion_root,
        completion_inventory, diagnostic_root, diagnostic_inventory)
    if prepared['status'] != 'prepared':
        raise ValueError('Prepared official CIFAR files are required')
    output = Output(output_root)
    report = {'schema':recovery.payload()['schema'], 'status':'started', 'started_at':started,
        'protocol_sha256':protocol.sha256, 'recovery_sha256':recovery.sha256(),
        'automatic_retry':False, 'baseline_training_performed':False, 'search_performed':False,
        'official_test_access':False, 'latency_sessions':0, 'quality':{}, 'starts':{},
        'trajectories':{}, 'BN_pairs':{}, 'completion_report_sha256':recovery.COMPLETION_REPORT_SHA,
        'completion_inventory_sha256':recovery.COMPLETION_INVENTORY_SHA,
        'diagnostic_report_sha256':recovery.DIAGNOSTIC_REPORT_SHA,
        'diagnostic_inventory_sha256':recovery.DIAGNOSTIC_INVENTORY_SHA}
    output.json('protocol.json', protocol.payload())
    output.json('recovery_protocol.json', recovery.payload())
    output.json('source_manifest.json', prepared['source'])
    output.json('completion_admission.json', prepared['completion'])
    output.json('diagnostic_admission.json', prepared['diagnostic'])
    runtime, pilot = None, None
    old = prepared['completion']['report']
    try:
        report['gpu_admission'] = admit_free_gpu()
        runtime = Runtime(protocol, output, started)
        runtime.deadline = min(runtime.deadline, started+recovery.WORKER_SECONDS)
        report['absolute_deadline'] = runtime.deadline
        with completion_phase(runtime, 'manifest', dict(recovery.STAGES)['manifest']) as deadline:
            from .cv_sparse_data import CifarData
            from .cv_sparse_pilot import Pilot
            data = CifarData(data_root, protocol, download=False)
            for parent in (prepared['completion'], prepared['diagnostic']):
                root = Path(parent['root'])
                if (data.roles.payload() != json.loads(confined_file(root,'data_roles.json').read_text())
                        or data.source_hashes != parent['report']['data']['source_files']
                        or data.roles.sha256 != parent['report']['data']['roles_sha256']):
                    raise ValueError('Dataset bytes or roles changed since the immutable parents')
                environment = json.loads(confined_file(root,'environment.json').read_text())
                if runtime.environment != environment or runtime.environment != parent['report']['environment']:
                    raise ValueError('Recovery scientific environment differs from a parent')
            freeze = subprocess.run([sys.executable,'-m','pip','freeze'], capture_output=True,
                                    text=True, check=True, timeout=30).stdout
            for parent in (prepared['completion'], prepared['diagnostic']):
                previous = confined_file(Path(parent['root']),'environment_freeze.txt').read_text()
                if frozen_dependencies(freeze) != frozen_dependencies(previous):
                    raise ValueError('Scientific dependency versions differ from a parent')
            output.bytes('environment_freeze.txt', freeze.encode())
            output.json('environment.json', runtime.environment)
            output.json('data_roles.json', data.roles.payload())
            report['environment'] = runtime.environment
            report['data'] = {'source_files':data.source_hashes, 'roles_sha256':data.roles.sha256}
            baseline = dict(old['baseline'])
            parent = Path(prepared['completion']['root'])
            copied = output.bytes(baseline['path'], confined_file(parent,baseline['path']).read_bytes())
            if copied['sha256'] != recovery.BASELINE_SHA:
                raise ValueError('The original M0 copy changed')
            copied_predictions = output.bytes('inherited/original_M0_predictions.npz',
                confined_file(parent,baseline['validation']['prediction_file']).read_bytes())
            if copied_predictions['sha256'] != baseline['validation']['prediction_sha256']:
                raise ValueError('The original M0 prediction copy changed')
            pilot = Pilot(runtime, data)
            teacher = load_checkpoint(baseline, pilot.gpu, deadline)
            pilot.bind_teacher(teacher)
            report['teacher_fingerprint_before'] = pilot.teacher_fingerprint
            if pilot.space.source_storage_bytes != recovery.FAMILIES[0][2]:
                raise ValueError('Original full M0 physical bytes changed')
            genotypes = decode_genotypes(prepared['completion']['genotype_payloads'], pilot)
            tape = recovery_tape(data.roles.train, protocol)
            tape_sha = digest(tape)
            if (protocol.recovery_seed != 990032 or protocol.recovery_updates != 200
                    or protocol.recovery_batch != 128
                    or any(item.get('recovery_tape_sha256') != tape_sha for item in old['recovered'].values())
                    or prepared['diagnostic']['report'].get('recovery_tape_sha256') != tape_sha):
                raise ValueError('The common recovery tape or its fixed budget changed')
            report['recovery_tape_sha256'] = tape_sha
            output.json('recovery_tape.json', {'sha256':tape_sha, 'updates':[
                {'row_ids':list(ids), 'augmentation_seed':seed} for ids,seed in tape]})
            output.json('imported_sources.json', verify_loaded_sources(prepared['source']))
        with completion_phase(runtime, 'quality', dict(recovery.STAGES)['quality']) as deadline:
            run_quality(pilot, genotypes, tape, deadline=deadline,
                diagnostic=prepared['diagnostic'], baseline_predictions=output.root/copied_predictions['path'],
                results=report['quality'], starts=report['starts'], trajectories=report['trajectories'],
                bn_pairs=report['BN_pairs'])
        with completion_phase(runtime, 'close', dict(recovery.STAGES)['close']):
            from fedcore.algorithm.low_rank.candidate_storage import capture_inventory, inventory_fingerprint
            report['teacher_fingerprint_after'] = inventory_fingerprint(capture_inventory(pilot.teacher))
            if report['teacher_fingerprint_after'] != pilot.teacher_fingerprint:
                raise ValueError('The frozen original M0 changed')
            for name, inspected in (
                ('completion', inspect_completion(completion_root,completion_inventory)),
                ('diagnostic', inspect_diagnostic(diagnostic_root,diagnostic_inventory))):
                if inspected['verified_files'] != prepared[name]['verified_files']:
                    raise ValueError('Immutable parent evidence changed during recovery')
            report['parents_read_only_verified'] = True
            for item in (*report['starts'].values(), *report['quality'].values()):
                if sha256_file(confined_file(output.root,item['path'])) != item['sha256']:
                    raise ValueError('A persisted recovery snapshot changed')
            for item in report['quality'].values():
                evidence = item['validation']
                if sha256_file(confined_file(output.root,evidence['prediction_file'])) != evidence['prediction_sha256']:
                    raise ValueError('Persisted per-row quality evidence changed')
            output.json('imported_sources.json', verify_loaded_sources(prepared['source']))
            runtime.check()
            report['status'] = 'completed'
    except PilotStop as error:
        report.update(status='recovery_incomplete_'+error.stage, stop_reason=error.reason)
    except (Exception, PilotFatal) as error:
        report.update(status='failed', failure_type=type(error).__name__, failure=str(error), traceback=traceback.format_exc())
    finally:
        observed, count_failure = count_training_updates(output.ledger)
        report['observed_trajectory_updates'] = observed
        report['completed_trajectory_count'] = len(report['trajectories'])
        report['completed_trajectory_updates'] = sum(row['updates'] for row in report['trajectories'].values())
        report['total_SGD_updates'] = sum(observed.values())
        expected = {f'{family}_SGD_lr_{lr:g}'
                    for family, _, _ in recovery.FAMILIES for lr in recovery.LEARNING_RATES}
        if report['status'] == 'completed' and (
                set(observed) != expected or set(report['trajectories']) != expected
                or any(update != 200 for update in observed.values())
                or report['completed_trajectory_updates'] != report['total_SGD_updates']
                or report['total_SGD_updates'] != 2000):
            count_failure = count_failure or 'Completed matrix requires exactly ten distinct 200-update trajectories'
        if count_failure is not None:
            report.update(status='failed', ledger_count_failure=count_failure)
        report['total_validation_rows'] = sum(row['validation']['rows'] for row in report['quality'].values())
        report['ended_at'] = time.perf_counter()
        report['wall_seconds'] = report['ended_at']-started
        report['charged_slot_seconds'] = report['wall_seconds']
        report['total_charged_seconds'] = recovery.PRIOR_SECONDS+report['wall_seconds']
        if runtime is not None:
            report.update(peak_allocated_bytes=runtime.torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=runtime.torch.cuda.max_memory_reserved(),
                          within_slot_deadline=report['ended_at']<=runtime.deadline)
        output.json('report.json', report)
        output.append({'event':'recovery_stopped', 'status':report['status'], 'ended_at':report['ended_at']})
    return report


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    names = ('data-root','output-root','source-manifest','completion-root','completion-inventory',
             'diagnostic-root','diagnostic-inventory')
    for name in names:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    values = tuple(getattr(args, name.replace('-','_')) for name in names)
    if not args.execute:
        result = recovery_preparation(*values)
        print(json.dumps({'status':result['status'], 'execute':False,
            'completion_files':len(result['completion']['verified_files']),
            'diagnostic_files':len(result['diagnostic']['verified_files']),
            'recovery_sha256':recovery.sha256(), 'max_seconds':recovery.MAX_SECONDS}))
        return 0 if result['status'] == 'prepared' else 2
    result = run_recovery(*values, started=started)
    print(json.dumps({'status':result['status'], 'output':args.output_root}))
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
