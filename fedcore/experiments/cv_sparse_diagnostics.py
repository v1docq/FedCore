"""One bounded diagnostic series on immutable completion-1 artifacts.

The default command verifies inputs only. Torch, CIFAR readers, CUDA and output
creation remain behind the explicit execution boundary. No search or test access
is part of this entrypoint.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import cv_sparse_diagnostic_protocol as diagnostic
from .cv_sparse_continuation import (completion_phase, confined_file,
    frozen_dependencies, verify_baseline_predictions)
from .cv_sparse_protocol import PilotProtocol, aa_admission, digest
from .cv_sparse_runtime import (GPUData, Output, PilotFatal, PilotStop, Runtime,
    admit_free_gpu, calibrate_bn, checkpoint, evaluate, paired_latency,
    preparation, recovery_tape, sha256_file, verify_loaded_sources)


@dataclass(frozen=True)
class AASession:
    role: str
    artifact_id: str
    condition: str
    label: str


def timing_plan():
    """Nine fixed sessions, independent of measurements and input dict order."""
    sessions = []
    for index, (role, identity) in enumerate(diagnostic.TIMING_ARTIFACTS):
        order = ('canonical', 'legacy') if index % 2 == 0 else ('legacy', 'canonical')
        for condition in order:
            sessions.append(AASession(role, identity, condition, f'{role}_{condition}_AA'))
    role, identity = diagnostic.TIMING_ARTIFACTS[0]
    sessions.append(AASession(role, identity, 'canonical', 'M0_final_canonical_AA'))
    return tuple(sessions)


def _verify_file(root, files, name):
    target = confined_file(root, name)
    expected = files.get(name)
    if (type(expected) is not dict or type(expected.get('bytes')) is not int
            or target.stat().st_size != expected['bytes']
            or sha256_file(target) != expected.get('sha256')):
        raise ValueError(f'Completion artifact integrity failure: {name}')
    return dict(expected)


def inspect_completion(completion_root, completion_inventory):
    """Verify the pinned inventory and only files this diagnostic actually uses.

    Exported launcher/control files in the inventory may live outside the worker
    artifact root. They are neither consumed nor demanded inside that root.
    """
    root = Path(completion_root).resolve()
    inventory_path = Path(completion_inventory).resolve()
    if sha256_file(inventory_path) != diagnostic.COMPLETION_INVENTORY_SHA:
        raise ValueError('Completion inventory differs from the pinned evidence')
    inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
    files = inventory.get('files')
    if type(files) is not dict:
        raise ValueError('Completion inventory requires a file map')
    verified = {'report.json': _verify_file(root, files, 'report.json')}
    if verified['report.json']['sha256'] != diagnostic.COMPLETION_REPORT_SHA:
        raise ValueError('Completion report differs from the pinned evidence')
    report = json.loads(confined_file(root, 'report.json').read_text(encoding='utf-8'))
    if (report.get('status') != 'completed' or report.get('within_slot_deadline') is not True
            or report.get('protocol_sha256') != PilotProtocol().sha256
            or report.get('baseline_training_performed') is not False
            or report.get('search_performed') is not False):
        raise ValueError('Only the completed, timely completion-1 evidence is admitted')
    baseline = report['baseline']
    if (baseline.get('sha256') != diagnostic.BASELINE_SHA
            or baseline.get('artifact_id') != diagnostic.TIMING_ARTIFACTS[0][1]
            or baseline.get('phase') != 'dense_original'):
        raise ValueError('The original full M0 identity changed')
    required = {'environment.json', 'environment_freeze.txt', 'data_roles.json',
                baseline['path'], baseline['validation']['prediction_file']}
    timing = {}
    roles = {'hybrid': 'fixed_hybrid', 'R': 'random_finalist', 'E': 'evolution_finalist'}
    for role, identity in diagnostic.TIMING_ARTIFACTS:
        if role != 'M0':
            artifacts = [item for item in report['recovered'].values() if roles[role] in item.get('roles', ())]
            if len(artifacts) != 1 or artifacts[0].get('artifact_id') != identity:
                raise ValueError('Completion role does not identify the pinned timing artifact')
        record = report['latency'][identity]['torchscript']
        if (record.get('artifact_id') != identity
                or record.get('format') != 'independently_reloaded_TorchScript'):
            raise ValueError('Timing requires the original persisted TorchScript bytes')
        required.add(record['path'])
        timing[role] = dict(record)
    for name in sorted(required):
        verified[name] = _verify_file(root, files, name)
    if verified[baseline['path']]['sha256'] != diagnostic.BASELINE_SHA:
        raise ValueError('Original full M0 bytes changed')
    evidence = baseline['validation']
    if verified[evidence['prediction_file']]['sha256'] != evidence['prediction_sha256']:
        raise ValueError('Original full M0 predictions changed')
    for record in timing.values():
        item = verified[record['path']]
        if item['sha256'] != record.get('sha256') or item['bytes'] != record.get('file_bytes'):
            raise ValueError('Timing TorchScript metadata differs from the pinned file inventory')
    return {'completion_root': str(root), 'report_sha256': diagnostic.COMPLETION_REPORT_SHA,
            'inventory_sha256': diagnostic.COMPLETION_INVENTORY_SHA,
            'verified_files': verified, 'report': report, 'timing_artifacts': timing}


def diagnostic_preparation(data_root, output_root, manifest, completion_root, completion_inventory):
    root, output = Path(completion_root).resolve(), Path(output_root).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Diagnostic output and immutable completion require disjoint trees')
    prepared = preparation(data_root, output_root, manifest, PilotProtocol())
    required = {'fedcore/experiments/'+name+'.py' for name in (
        'cv_sparse_continuation', 'cv_sparse_diagnostics', 'cv_sparse_diagnostic_protocol')}
    if not required.issubset(prepared['source']['verified_package_files_sha256']):
        raise ValueError('Source manifest must include every diagnostic execution module')
    if prepared['source']['supplied'].get('diagnostic_protocol_sha256') != diagnostic.sha256():
        raise ValueError('Source manifest must bind the fixed diagnostic protocol')
    completion = inspect_completion(root, completion_inventory)
    return {**prepared, 'completion': completion, 'diagnostic': diagnostic.payload(),
            'diagnostic_sha256': diagnostic.sha256(), 'automatic_retry': False}


def load_baseline(artifact, gpu, deadline):
    from fedcore.tools.registry.checkpoint_manager import CheckpointManager
    gpu.runtime.check(deadline, checkpoint='original_M0_reload')
    path = confined_file(gpu.runtime.output.root, artifact['path'])
    if sha256_file(path) != diagnostic.BASELINE_SHA:
        raise ValueError('Diagnostic full M0 copy changed')
    model = CheckpointManager('.', auto_cleanup=False).deserialize_from_bytes(path.read_bytes(), device='cpu')
    gpu.runtime.check(deadline)
    return model.eval()


def recover_without_calibration(model, gpu, tape, *, deadline, role, lr):
    """One fresh trajectory; training BN behaves normally, no extra BN reset."""
    runtime, torch, protocol = gpu.runtime, gpu.torch, gpu.runtime.protocol
    if type(lr) is not float or lr not in diagnostic.LEARNING_RATES or len(tape) != protocol.recovery_updates:
        raise ValueError('Only the two fixed 200-update diagnostic trajectories are allowed')
    model.to('cuda').train().requires_grad_(True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=protocol.momentum,
                                weight_decay=protocol.weight_decay, nesterov=True)
    tape_sha = digest(tape)
    for update, (ids, seed) in enumerate(tape, 1):
        runtime.check(deadline, recovery=role, updates=update-1)
        x, y = gpu.training(ids, seed)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite diagnostic recovery loss')
        loss.backward()
        if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
            raise ValueError('Diagnostic recovery has missing or nonfinite gradients')
        optimizer.step()
        runtime.output.append({'event':'diagnostic_SGD_update', 'trajectory':role, 'lr':lr,
                              'updates':update, 'loss':float(loss), 'tape_sha256':tape_sha})
        runtime.check(deadline, recovery=role, updates=update)
    evidence = {'updates':len(tape), 'lr':lr, 'tape_sha256':tape_sha,
                'optimizer':'fresh_SGD_momentum_0.9_nesterov_weight_decay_0.0005',
                'training_BN_updates':True, 'additional_BN_calibration':False}
    model._fedcore_method_evidence = {**getattr(model, '_fedcore_method_evidence', {}),
                                      'diagnostic_recovery': evidence}
    return evidence


def run_quality(baseline, gpu, tape, *, deadline, previous_predictions, results):
    """Six validation passes; each BN pair shares weights via an owned reload."""
    runtime = gpu.runtime
    validation = gpu.data.roles.search_validation
    if len(validation) != 5000:
        raise ValueError('Diagnostics require the original 5000 validation rows')

    def assess(model, label, updates, lr, bn, artifact=None):
        if artifact is None:
            artifact, restored = checkpoint(model, gpu, label=label, solution_id=baseline['solution_id'],
                phase='diagnostic_'+label, deadline=deadline)
        else:
            restored = model
        metrics = evaluate(restored.to('cuda'), gpu, validation, deadline=deadline, label=label)
        restored.cpu()
        if metrics['rows'] != 5000:
            raise ValueError('Diagnostic validation row count changed')
        item = {**artifact, 'validation':metrics, 'SGD_updates':updates, 'lr':lr,
                'additional_BN_calibration':bn, 'recovery_tape_sha256':digest(tape)}
        results[label] = item
        runtime.output.json(f'quality/{label}.json', item)
        return item, restored

    model = load_baseline(baseline, gpu, deadline)
    original, _ = assess(model, 'M0_untouched', 0, None, False, dict(baseline))
    verify_baseline_predictions(previous_predictions, runtime.output.root/original['validation']['prediction_file'])
    if original['validation']['top1'] != baseline['validation']['top1']:
        raise ValueError('Original full M0 validation quality changed')
    del model
    model = load_baseline(baseline, gpu, deadline)
    calibrate_bn(model.to('cuda'), gpu, deadline)
    assess(model, 'M0_post_BN', 0, None, True)
    del model
    for lr in diagnostic.LEARNING_RATES:
        trajectory = f'SGD_lr_{lr:g}'
        model = load_baseline(baseline, gpu, deadline)
        proof = recover_without_calibration(model, gpu, tape, deadline=deadline, role=trajectory, lr=lr)
        before, independent = assess(model, trajectory+'_pre_BN', 200, lr, False)
        before['training'] = proof
        runtime.output.json(f'quality/{trajectory}_pre_BN.json', before)
        # checkpoint() returned an independently deserialized object. The saved
        # pre-BN bytes and the caller's original model cannot alias its buffers.
        calibrate_bn(independent.to('cuda'), gpu, deadline)
        after, _ = assess(independent, trajectory+'_post_BN', 200, lr, True)
        after['training'] = proof
        after['shared_200_update_checkpoint_sha256'] = before['sha256']
        runtime.output.json(f'quality/{trajectory}_post_BN.json', after)
        if sha256_file(confined_file(runtime.output.root, before['path'])) != before['sha256']:
            raise ValueError('Pre-BN checkpoint changed during additional calibration')
        del model, independent
    if len(results) != 6 or sum(item['SGD_updates'] for key,item in results.items() if key.endswith('_pre_BN')) != 400:
        raise ValueError('The six-variant diagnostic workload changed')


def aa_accepted(result):
    """Gate both complete batches and both clocks, using the original AA rule."""
    if (type(result) is not dict or result.get('aa') is not True
            or result.get('accepted') is not True or set(result.get('batches', {})) != {'1','128'}):
        return False
    for record in result['batches'].values():
        if record.get('accepted') is not True:
            return False
        pairs = record.get('pairs', ())
        orders = [row.get('order') for row in pairs]
        if orders != record.get('orders'):
            return False
        for clock in ('cuda_event', 'synchronized_wall'):
            if not aa_admission([row.get(clock+'_ratio_B_over_A') for row in pairs], orders)['accepted']:
                return False
            if record.get(clock+'_admission', {}).get('accepted') is not True:
                return False
    return True


def load_timing_model(root, record, gpu, deadline, *, cpu_then_cuda=False):
    gpu.runtime.check(deadline, timing_load=record['artifact_id'])
    path = confined_file(Path(root).resolve(), record['path'])
    if sha256_file(path) != record['sha256'] or path.stat().st_size != record['file_bytes']:
        raise ValueError('Immutable TorchScript bytes changed before reload')
    model = gpu.torch.jit.load(str(path), map_location='cpu' if cpu_then_cuda else 'cuda').eval()
    if cpu_then_cuda:
        model.to('cuda')
    gpu.runtime.check(deadline)
    return model


def verify_timing_probe(first, second, gpu, deadline):
    """Same saved module and images; the existing TS export tolerance applies."""
    if first is second:
        raise ValueError('Timing probes require independent module objects')
    gpu.runtime.check(deadline)
    probe = gpu.data.rows(gpu.data.roles.calibration[:2]).to('cuda')
    with gpu.torch.inference_mode():
        left, right = first(probe), second(probe)
        if not bool(gpu.torch.isfinite(left).all()) or not bool(gpu.torch.isfinite(right).all()):
            raise ValueError('Independent same-TorchScript GPU logits are nonfinite')
        gpu.torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)
        maximum = float((left-right).abs().max())
    gpu.runtime.check(deadline)
    return {'rows':2, 'same_probe':True, 'atol':1e-5, 'rtol':1e-5,
            'same_device_logits_accepted':True, 'max_abs_difference':maximum}


def run_timing(completion_root, artifacts, gpu, *, deadline, results):
    """All nine AA sessions precede the at most three gated canonical AB pairs."""
    aa, ab = results.setdefault('AA', {}), results.setdefault('AB', {})
    for session in timing_plan():
        record = artifacts[session.role]
        first = load_timing_model(completion_root, record, gpu, deadline, cpu_then_cuda=session.condition=='legacy')
        second = load_timing_model(completion_root, record, gpu, deadline)
        probe = verify_timing_probe(first, second, gpu, deadline)
        result = paired_latency(first, second, gpu, deadline=deadline, label=session.label, aa=True)
        result.update(role=session.role, artifact_id=session.artifact_id, condition=session.condition,
                      torchscript=record, probe=probe)
        aa[session.label] = result
        gpu.runtime.output.json(f'latency/{session.label}.json', result)
        del first, second
    m0_admitted = aa_accepted(aa['M0_canonical_AA']) and aa_accepted(aa['M0_final_canonical_AA'])
    results['M0_canonical_initial_and_final_accepted'] = m0_admitted
    for role, identity in diagnostic.TIMING_ARTIFACTS[1:]:
        admitted = m0_admitted and aa_accepted(aa[f'{role}_canonical_AA'])
        if not admitted:
            ab[role] = {'comparison_admitted':False, 'reason':'canonical_AA_gate'}
            continue
        # Independent fresh direct-CUDA loads are used again for AB. Extra
        # un-timed same-artifact references prove each reload's logits, never
        # equality between the compressed model and M0.
        first = load_timing_model(completion_root, artifacts['M0'], gpu, deadline)
        reference = load_timing_model(completion_root, artifacts['M0'], gpu, deadline)
        first_probe = verify_timing_probe(first, reference, gpu, deadline)
        del reference
        second = load_timing_model(completion_root, artifacts[role], gpu, deadline)
        reference = load_timing_model(completion_root, artifacts[role], gpu, deadline)
        second_probe = verify_timing_probe(second, reference, gpu, deadline)
        del reference
        result = paired_latency(first, second, gpu, deadline=deadline, label=f'M0_vs_{role}_canonical_AB')
        result.update(comparison_admitted=True, first_torchscript=artifacts['M0'],
                      second_torchscript=artifacts[role], first_probe=first_probe, second_probe=second_probe,
                      role=role, artifact_id=identity, condition='canonical')
        ab[role] = result
        gpu.runtime.output.json(f'latency/M0_vs_{role}_canonical_AB.json', result)
        del first, second


def run_diagnostics(data_root, output_root, manifest, completion_root, completion_inventory, *, started=None):
    started = time.perf_counter() if started is None else started
    protocol = PilotProtocol()
    prepared = diagnostic_preparation(data_root, output_root, manifest, completion_root, completion_inventory)
    if prepared['status'] != 'prepared':
        raise ValueError('Prepared official CIFAR files are required')
    output = Output(output_root)
    report = {'schema':diagnostic.payload()['schema'], 'status':'started', 'started_at':started,
              'protocol_sha256':protocol.sha256, 'diagnostic_sha256':diagnostic.sha256(),
              'automatic_retry':False, 'baseline_training_performed':False, 'search_performed':False,
              'official_test_access':False, 'quality':{}, 'timing':{},
              'completion_report_sha256':diagnostic.COMPLETION_REPORT_SHA,
              'completion_inventory_sha256':diagnostic.COMPLETION_INVENTORY_SHA}
    output.json('protocol.json', protocol.payload())
    output.json('diagnostic_protocol.json', diagnostic.payload())
    output.json('source_manifest.json', prepared['source'])
    output.json('completion_admission.json', prepared['completion'])
    runtime = None
    parent = Path(completion_root).resolve()
    old = prepared['completion']['report']
    try:
        report['gpu_admission'] = admit_free_gpu()
        runtime = Runtime(protocol, output, started)
        runtime.deadline = min(runtime.deadline, started+diagnostic.WORKER_SECONDS)
        report['absolute_deadline'] = runtime.deadline
        with completion_phase(runtime, 'manifest', dict(diagnostic.STAGES)['manifest']) as deadline:
            from .cv_sparse_data import CifarData
            data = CifarData(data_root, protocol, download=False)
            if (data.roles.payload() != json.loads(confined_file(parent,'data_roles.json').read_text())
                    or data.source_hashes != old['data']['source_files']
                    or data.roles.sha256 != old['data']['roles_sha256']):
                raise ValueError('Dataset bytes or roles changed since completion-1')
            previous_environment = json.loads(confined_file(parent,'environment.json').read_text())
            if runtime.environment != previous_environment or runtime.environment != old['environment']:
                raise ValueError('Diagnostic scientific runtime environment differs from completion-1')
            freeze = subprocess.run([sys.executable,'-m','pip','freeze'], capture_output=True,text=True,check=True,timeout=30).stdout
            if frozen_dependencies(freeze) != frozen_dependencies(confined_file(parent,'environment_freeze.txt').read_text()):
                raise ValueError('Scientific dependency versions differ from completion-1')
            output.bytes('environment_freeze.txt', freeze.encode())
            output.json('environment.json', runtime.environment)
            output.json('data_roles.json', data.roles.payload())
            report['environment'] = runtime.environment
            report['data'] = {'source_files':data.source_hashes, 'roles_sha256':data.roles.sha256}
            baseline = dict(old['baseline'])
            copied = output.bytes(baseline['path'], confined_file(parent,baseline['path']).read_bytes())
            if copied['sha256'] != diagnostic.BASELINE_SHA:
                raise ValueError('Original full M0 copy changed')
            predictions = baseline['validation']
            previous_predictions = output.bytes(predictions['prediction_file'],
                confined_file(parent,predictions['prediction_file']).read_bytes())
            if previous_predictions['sha256'] != predictions['prediction_sha256']:
                raise ValueError('Original baseline prediction copy changed')
            gpu = GPUData(data, runtime)
            tape = recovery_tape(data.roles.train, protocol)
            if any(item.get('recovery_tape_sha256') != digest(tape) for item in old['recovered'].values()):
                raise ValueError('Diagnostic update tape differs from the original recovery tape')
            output.json('recovery_tape.json', {'sha256':digest(tape), 'updates':[
                {'row_ids':list(ids), 'augmentation_seed':seed} for ids,seed in tape]})
            report['recovery_tape_sha256'] = digest(tape)
            output.json('imported_sources.json', verify_loaded_sources(prepared['source']))
        with completion_phase(runtime, 'quality', dict(diagnostic.STAGES)['quality']) as deadline:
            run_quality(baseline, gpu, tape, deadline=deadline,
                previous_predictions=output.root/previous_predictions['path'], results=report['quality'])
            report['total_SGD_updates'] = 400
            report['total_validation_rows'] = 30000
        with completion_phase(runtime, 'timing', dict(diagnostic.STAGES)['timing']) as deadline:
            run_timing(parent, prepared['completion']['timing_artifacts'], gpu,
                       deadline=deadline, results=report['timing'])
        with completion_phase(runtime, 'close', dict(diagnostic.STAGES)['close']):
            if inspect_completion(parent, completion_inventory)['verified_files'] != prepared['completion']['verified_files']:
                raise ValueError('Immutable completion evidence changed during diagnostics')
            for artifact in report['quality'].values():
                if sha256_file(confined_file(output.root, artifact['path'])) != artifact['sha256']:
                    raise ValueError('Diagnostic checkpoint changed after validation')
            output.json('imported_sources.json', verify_loaded_sources(prepared['source']))
            runtime.check()
            report['status'] = 'completed'
    except PilotStop as error:
        report.update(status='diagnostic_incomplete_'+error.stage, stop_reason=error.reason)
    except (Exception, PilotFatal) as error:
        report.update(status='failed', failure_type=type(error).__name__, failure=str(error), traceback=traceback.format_exc())
    finally:
        report['ended_at'] = time.perf_counter()
        report['wall_seconds'] = report['ended_at']-started
        report['charged_slot_seconds'] = report['wall_seconds']
        report['total_charged_seconds'] = diagnostic.PRIOR_SECONDS+report['wall_seconds']
        if runtime is not None:
            report.update(peak_allocated_bytes=runtime.torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=runtime.torch.cuda.max_memory_reserved(),
                          within_slot_deadline=report['ended_at']<=runtime.deadline)
        output.json('report.json', report)
        output.append({'event':'diagnostic_stopped', 'status':report['status'], 'ended_at':report['ended_at']})
    return report


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','output-root','source-manifest','completion-root','completion-inventory'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    values = (args.data_root,args.output_root,args.source_manifest,args.completion_root,args.completion_inventory)
    if not args.execute:
        result = diagnostic_preparation(*values)
        print(json.dumps({'status':result['status'], 'execute':False,
                          'completion_files':len(result['completion']['verified_files']),
                          'diagnostic_sha256':diagnostic.sha256(), 'max_seconds':diagnostic.MAX_SECONDS}))
        return 0 if result['status']=='prepared' else 2
    result = run_diagnostics(*values, started=started)
    print(json.dumps({'status':result['status'], 'output':args.output_root}))
    return 0 if result['status']=='completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
