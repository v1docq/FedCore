"""Evaluate five pinned saved SGD=.001 states; never train, calibrate or search.

The default CLI is a standard-library integrity preflight. Execution retains
the latest evaluated batch prefix and every raw timing pair, including failed gates.
The official test has been used previously: these are descriptive measurements
of frozen validation-selected states, not an untouched final test.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import cv_sparse_evaluation_protocol as evaluation
from . import cv_sparse_recovery_protocol as recovery
from .cv_sparse_continuation import completion_phase, confined_file, frozen_dependencies
from .cv_sparse_diagnostics import load_timing_model, verify_timing_probe
from .cv_sparse_protocol import PilotProtocol, aa_admission, digest
from .cv_sparse_recovery import load_checkpoint, model_evidence, verify_dense_predictions
from .cv_sparse_runtime import (GPUData, Output, PilotFatal, PilotStop, Runtime,
    admit_free_gpu, paired_latency, preparation, sha256_file, source_manifest,
    torchscript_artifact as export_torchscript_artifact, verify_loaded_sources)


FAMILIES = tuple(row[0] for row in evaluation.ARTIFACTS)
PRE_AA = (('M0', 'M0_initial_AA'), ('P75', 'P75_AA'), ('R', 'R_AA'),
          ('E', 'E_AA'), ('hybrid', 'hybrid_AA'), ('M0', 'M0_pre_AB_AA'))
POST_AA = 'M0_post_AB_AA'
FIELDS = {'row_ids', 'labels', 'predictions', 'cross_entropy'}


def _guard_paths(data_root, output_root, manifest, recovery_root, recovery_inventory):
    output, parent, data = (Path(value).resolve() for value in (output_root, recovery_root, data_root))
    for immutable in (parent, data):
        if output.is_relative_to(immutable) or immutable.is_relative_to(output):
            raise ValueError('Evaluation output and immutable input trees require disjoint paths')
    for path in (manifest, recovery_inventory):
        if Path(path).resolve().is_relative_to(output):
            raise ValueError('Immutable manifest and inventory cannot lie in the output tree')
    return parent, output


def _verify_file(root, files, name):
    target = confined_file(root, name)
    record = files.get(name)
    if (type(record) is not dict or type(record.get('bytes')) is not int or record['bytes'] <= 0
            or target.stat().st_size != record['bytes'] or sha256_file(target) != record.get('sha256')):
        raise ValueError(f'Recovery artifact integrity failure: {name}')
    return dict(record)


def _selected_sources(report):
    """Admit exactly the five predeclared states without ranking any metric."""
    records = report.get('quality')
    if type(records) is not dict:
        raise ValueError('Recovery report requires the original quality matrix')
    selected = {}
    graph_ids = {family: identity for family, identity, _ in recovery.FAMILIES}
    for family, identity, checkpoint_sha, storage in evaluation.ARTIFACTS:
        label = f'{family}_SGD_lr_0.001_pre_BN'
        item = records.get(label, {})
        training, validation = item.get('training', {}), item.get('validation', {})
        if (item.get('family') != family or item.get('artifact_id') != identity
                or item.get('solution_id') != graph_ids[family] or item.get('sha256') != checkpoint_sha
                or type(item.get('tensor_bytes')) is not int or item['tensor_bytes'] != storage
                or item.get('phase') != 'recovery_'+label or item.get('state') != 'SGD_lr_0.001'
                or type(item.get('lr')) is not float or item['lr'] != .001
                or type(item.get('SGD_updates')) is not int or item['SGD_updates'] != 200
                or item.get('additional_BN_calibration') is not False
                or item.get('completed_within_deadline') is not True or item.get('gpu_reload_exact') is not True
                or item.get('shared_pre_BN_checkpoint_sha256') != checkpoint_sha
                or item.get('source_M0_sha256') != recovery.BASELINE_SHA
                or item.get('recovery_tape_sha256') != report.get('recovery_tape_sha256')
                or training.get('tape_sha256') != report.get('recovery_tape_sha256')
                or type(training.get('updates')) is not int or training['updates'] != 200
                or type(training.get('lr')) is not float or training['lr'] != .001
                or training.get('additional_BN_calibration') is not False
                or training.get('training_BN_updates') is not True
                or validation.get('rows') != 5000 or type(item.get('file_bytes')) is not int
                or item['file_bytes'] <= 0
                or digest({'solution_id':item.get('solution_id'), 'checkpoint_sha256':checkpoint_sha,
                           'phase':item.get('phase')}) != identity):
            raise ValueError(f'Pinned saved recovery state, role or physical cost changed: {family}')
        selected[family] = dict(item)
    if len({item['path'] for item in selected.values()}) != 5 or len({
            item['validation']['prediction_file'] for item in selected.values()}) != 5:
        raise ValueError('Five distinct source checkpoints and validation references are required')
    return selected


def inspect_recovery(recovery_root, recovery_inventory):
    """Verify only 14 consumed files; exported controllers need not be in root."""
    root, inventory = Path(recovery_root).resolve(), Path(recovery_inventory).resolve()
    if sha256_file(inventory) != evaluation.PARENT_INVENTORY_SHA:
        raise ValueError('Recovery inventory differs from the pinned evidence')
    files = json.loads(inventory.read_text(encoding='utf-8')).get('files')
    if type(files) is not dict:
        raise ValueError('Recovery inventory requires a file map')
    verified = {'report.json':_verify_file(root, files, 'report.json')}
    if verified['report.json']['sha256'] != evaluation.PARENT_REPORT_SHA:
        raise ValueError('Recovery report differs from the pinned evidence')
    report = json.loads(confined_file(root, 'report.json').read_text(encoding='utf-8'))
    if (report.get('status') != 'completed' or report.get('within_slot_deadline') is not True
            or report.get('protocol_sha256') != PilotProtocol().sha256
            or report.get('recovery_sha256') != recovery.sha256()
            or report.get('baseline_training_performed') is not False or report.get('search_performed') is not False
            or report.get('official_test_access') is not False or report.get('automatic_retry') is not False
            or report.get('latency_sessions') != 0 or report.get('total_SGD_updates') != 2000
            or report.get('total_validation_rows') != 150000 or report.get('parents_read_only_verified') is not True
            or report.get('completion_report_sha256') != recovery.COMPLETION_REPORT_SHA
            or report.get('completion_inventory_sha256') != recovery.COMPLETION_INVENTORY_SHA
            or report.get('diagnostic_report_sha256') != recovery.DIAGNOSTIC_REPORT_SHA
            or report.get('diagnostic_inventory_sha256') != recovery.DIAGNOSTIC_INVENTORY_SHA
            or type(report.get('total_charged_seconds')) not in (int,float)
            or type(report.get('charged_slot_seconds')) not in (int,float)
            or not 0 < report['charged_slot_seconds'] <= recovery.MAX_SECONDS
            or report['total_charged_seconds'] != recovery.PRIOR_SECONDS+report['charged_slot_seconds']
            or report['total_charged_seconds'] > evaluation.PRIOR_SECONDS):
        raise ValueError('Only completed, timely evidence from the original recovery protocol is admitted')
    for name in ('environment.json', 'environment_freeze.txt', 'data_roles.json'):
        verified[name] = _verify_file(root, files, name)
    if (json.loads(confined_file(root,'environment.json').read_text(encoding='utf-8')) != report.get('environment')
            or digest(json.loads(confined_file(root,'data_roles.json').read_text(encoding='utf-8')))
                != report.get('data',{}).get('roles_sha256')):
        raise ValueError('Recovery environment or data-role metadata disagrees with its report')
    sources = _selected_sources(report)
    for item in sources.values():
        for name in (item['path'], item['validation']['prediction_file']):
            verified[name] = _verify_file(root, files, name)
        if (verified[item['path']]['sha256'] != item['sha256']
                or verified[item['path']]['bytes'] != item['file_bytes']
                or verified[item['validation']['prediction_file']]['sha256']
                    != item['validation'].get('prediction_sha256')):
            raise ValueError('Saved model or validation reference metadata differs from its inventory')
    if len(verified) != 14:
        raise ValueError('Exactly fourteen distinct consumed recovery files are required')
    return {'root':str(root), 'report_sha256':evaluation.PARENT_REPORT_SHA,
            'inventory_sha256':evaluation.PARENT_INVENTORY_SHA, 'verified_files':verified,
            'report':report, 'sources':sources}


def evaluation_preparation(data_root, output_root, manifest, recovery_root, recovery_inventory):
    parent, _ = _guard_paths(data_root, output_root, manifest, recovery_root, recovery_inventory)
    prepared = preparation(data_root, output_root, manifest, PilotProtocol())
    required = {'fedcore/experiments/'+name+'.py' for name in (
        'cv_sparse_continuation', 'cv_sparse_diagnostics', 'cv_sparse_diagnostic_protocol',
        'cv_sparse_recovery', 'cv_sparse_recovery_protocol', 'cv_sparse_evaluation',
        'cv_sparse_evaluation_protocol')}
    if not required.issubset(prepared['source']['verified_package_files_sha256']):
        raise ValueError('Source manifest must include every saved-evaluation execution module')
    if prepared['source']['supplied'].get('evaluation_protocol_sha256') != evaluation.sha256():
        raise ValueError('Source manifest must bind the saved-evaluation protocol')
    return {**prepared, 'recovery':inspect_recovery(parent,recovery_inventory),
            'evaluation':evaluation.payload(), 'evaluation_sha256':evaluation.sha256(), 'automatic_retry':False}


def verify_ts_predictions(eager_path, torchscript_path):
    """Exact row order, labels and classes; fixed CE tolerance from the protocol."""
    import numpy as np
    maximum = 0.
    with np.load(eager_path,allow_pickle=False) as left, np.load(torchscript_path,allow_pickle=False) as right:
        if set(left.files) != FIELDS or set(right.files) != FIELDS:
            raise ValueError('TorchScript validation prediction schema changed')
        for name in sorted(FIELDS):
            a, b = left[name], right[name]
            dtype = np.dtype('float64' if name == 'cross_entropy' else 'int64')
            if a.shape != (5000,) or b.shape != (5000,) or a.dtype != dtype or b.dtype != dtype:
                raise ValueError(f'TorchScript validation shape or dtype changed: {name}')
            if name == 'cross_entropy':
                if (not np.isfinite(a).all() or not np.isfinite(b).all()
                        or not np.allclose(b,a,atol=evaluation.TS_CE_ATOL,rtol=evaluation.TS_CE_RTOL)):
                    raise ValueError('TorchScript validation CE exceeds the fixed tolerance')
                maximum = float(np.max(np.abs(a-b)))
            elif not np.array_equal(a,b):
                raise ValueError(f'TorchScript validation mismatch: {name}')
    return {'rows':5000, 'row_ids_labels_predictions_exact':True, 'CE_accepted':True,
            'CE_atol':evaluation.TS_CE_ATOL, 'CE_rtol':evaluation.TS_CE_RTOL,
            'CE_max_abs_difference':maximum, 'eager_sha256':sha256_file(eager_path),
            'torchscript_sha256':sha256_file(torchscript_path)}


def _prediction_prefix(output, label, row_ids, targets, predictions, losses):
    import numpy as np
    buffer = io.BytesIO()
    np.savez_compressed(buffer, row_ids=np.asarray(row_ids,dtype=np.int64),
        labels=np.asarray(targets,dtype=np.int64), predictions=np.asarray(predictions,dtype=np.int64),
        cross_entropy=np.asarray(losses,dtype=np.float64))
    return output.bytes(f'predictions/{label}.npz',buffer.getvalue())


def evaluate_saved(model, gpu, indices, *, deadline, label, test_data=None):
    """Same ordered 128-row oracle as evaluate, with durable batch prefixes.

    Lists retain at most 10000 rows. Rewriting the small compressed prefix per
    batch costs bounded host I/O, but preserves partial evidence on termination.
    There is no device transfer or filesystem write inside the timing loop.
    """
    import numpy as np
    runtime, torch = gpu.runtime, gpu.torch
    row_ids, targets, predictions, losses = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(indices),128):
            runtime.check(deadline,evaluation=label,rows=start)
            ids = indices[start:start+128]
            if test_data is None:
                rows = torch.as_tensor(ids,dtype=torch.long,device='cuda')
                x, y = gpu.normalized(gpu.images[rows]), gpu.labels[rows]
            else:
                images, labels = test_data
                raw = torch.from_numpy(np.ascontiguousarray(images[np.asarray(ids)])).permute(0,3,1,2).contiguous().to('cuda')
                x = gpu.normalized(raw)
                y = torch.from_numpy(np.asarray(labels[np.asarray(ids)],dtype=np.int64)).to('cuda')
            logits = model(x)
            per_row = torch.nn.functional.cross_entropy(logits,y,reduction='none')
            losses.extend(per_row.double().cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
            targets.extend(y.cpu().tolist())
            row_ids.extend(int(value) for value in ids)
            saved = _prediction_prefix(runtime.output,label,row_ids,targets,predictions,losses)
            runtime.output.append({'event':'saved_evaluation_batch', 'label':label,
                'batch_rows':len(ids), 'rows':len(row_ids), 'prediction_file':saved['path'],
                'prediction_sha256':saved['sha256']})
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(per_row).all()):
                raise ValueError('Nonfinite logits or cross entropy in saved evaluation')
            runtime.check(deadline,evaluation=label,rows=len(row_ids))
    if not row_ids:
        raise ValueError('A saved evaluation requires nonempty fixed rows')
    correct = sum(first == second for first,second in zip(predictions,targets))
    return {'rows':len(row_ids), 'correct':correct, 'top1':correct/len(row_ids),
            'cross_entropy':sum(losses)/len(row_ids), 'prediction_file':saved['path'],
            'prediction_sha256':saved['sha256']}


def copy_sources(parent, output):
    sources = {}
    root = Path(parent['root'])
    for family in FAMILIES:
        item = dict(parent['sources'][family])
        source = output.bytes(f'inherited/{family}.pt',confined_file(root,item['path']).read_bytes())
        reference = output.bytes(f'inherited/{family}_validation.npz',
            confined_file(root,item['validation']['prediction_file']).read_bytes())
        if (source['sha256'] != item['sha256'] or source['file_bytes'] != item['file_bytes']
                or reference['sha256'] != item['validation']['prediction_sha256']):
            raise ValueError('A copied saved source or prediction reference changed')
        item.update(path=source['path'], parent_checkpoint_path=item['path'],
                    parent_prediction_path=item['validation']['prediction_file'], validation_reference=reference)
        sources[family] = item
        output.json(f'inherited/{family}.json',item)
    return sources


def _verify_owned(output, record):
    path = confined_file(output.root,record['path'])
    if sha256_file(path) != record['sha256'] or path.stat().st_size != record['file_bytes']:
        raise ValueError('A persisted evaluation artifact changed')
    return path


def _model_identity(model, artifact):
    from fedcore.algorithm.low_rank.candidate_storage import capture_inventory, inventory_cost, inventory_fingerprint
    inventory = capture_inventory(model)
    state = model_evidence(model)
    metadata = getattr(model,'_fedcore_method_evidence',{}).get('pilot_artifact',{})
    if (inventory_cost(inventory,'bytes') != artifact['tensor_bytes']
            or inventory_fingerprint(inventory) != artifact['inventory_fingerprint']
            or digest(state['parameters']) != artifact['parameter_fingerprint']
            or digest(state['buffers']) != artifact['buffer_fingerprint']
            or metadata.get('protocol_sha256') != PilotProtocol().sha256
            or metadata.get('solution_id') != artifact['solution_id'] or metadata.get('phase') != artifact['phase']):
        raise ValueError('Reloaded source metadata, physical tensors or exact saved state changed')
    return state


def run_validation(sources, gpu, *, deadline, results):
    output = gpu.runtime.output
    validation = gpu.data.roles.search_validation
    if len(validation) != 5000 or tuple(sources) != FAMILIES:
        raise ValueError('Exactly five frozen sources and the original validation role are required')
    for family in FAMILIES:
        source = sources[family]
        item = results.setdefault(family,{'source':source})
        model = load_checkpoint(source,gpu,deadline)
        before = _model_identity(model,source)
        eager = evaluate_saved(model.to('cuda'),gpu,validation,deadline=deadline,label=family+'_eager_validation')
        model.cpu()
        if model_evidence(model) != before:
            raise ValueError('Eager validation changed the frozen saved state')
        item['eager_validation'] = eager
        item['eager_replay'] = verify_dense_predictions(_verify_owned(output,source['validation_reference']),
            confined_file(output.root,eager['prediction_file']))
        del model
        legacy_export_object, record = export_torchscript_artifact(source,gpu,deadline)
        del legacy_export_object  # The export helper returns a CPU-to-CUDA object; never time it.
        if (record.get('artifact_id') != source['artifact_id']
                or record.get('format') != 'independently_reloaded_TorchScript'):
            raise ValueError('Exported TorchScript identity or persistence format changed')
        record.update(source_checkpoint_sha256=source['sha256'],source_artifact_id=source['artifact_id'])
        _verify_owned(output,record)
        item['torchscript'] = record
        output.json(f"torchscript/{source['artifact_id']}.json",record)
        model = load_timing_model(output.root,record,gpu,deadline)
        item['TS_validation'] = evaluate_saved(model,gpu,validation,deadline=deadline,label=family+'_TS_validation')
        item['TS_equivalence'] = verify_ts_predictions(confined_file(output.root,eager['prediction_file']),
            confined_file(output.root,item['TS_validation']['prediction_file']))
        del model
        output.json(f'quality/{family}.json',item)
        gpu.runtime.check(deadline,completed_validation_family=family)


def seal_selection(sources, quality, gpu, deadline):
    """Own narrow admission; the older pilot admits different recovery phases."""
    from .cv_sparse_data import FrozenSelection
    output = gpu.runtime.output
    if tuple(sources) != FAMILIES or set(quality) != set(FAMILIES):
        raise ValueError('All five saved states must pass validation before opening test')
    sealed = {}
    for family, identity, checkpoint_sha, storage in evaluation.ARTIFACTS:
        source, item = sources[family], quality[family]
        ts = item.get('torchscript',{})
        if (item.get('source') != source or source['artifact_id'] != identity or source['sha256'] != checkpoint_sha
                or source['tensor_bytes'] != storage or item.get('eager_validation',{}).get('rows') != 5000
                or item.get('TS_validation',{}).get('rows') != 5000
                or item.get('eager_replay',{}).get('exact') is not True
                or item.get('TS_equivalence',{}).get('row_ids_labels_predictions_exact') is not True
                or item.get('TS_equivalence',{}).get('CE_accepted') is not True
                or ts.get('artifact_id') != identity or ts.get('source_artifact_id') != identity
                or ts.get('source_checkpoint_sha256') != checkpoint_sha
                or ts.get('format') != 'independently_reloaded_TorchScript'):
            raise ValueError('Incomplete validation or changed source/TorchScript lineage before test')
        _verify_owned(output,source)
        _verify_owned(output,source['validation_reference'])
        _verify_owned(output,ts)
        for name in ('eager_validation','TS_validation'):
            evidence = item[name]
            if sha256_file(confined_file(output.root,evidence['prediction_file'])) != evidence['prediction_sha256']:
                raise ValueError('Validation evidence changed before the durable selection seal')
        # Re-read all fields immediately before the seal; metadata cannot substitute for NPZ evidence.
        verify_dense_predictions(_verify_owned(output,source['validation_reference']),
            confined_file(output.root,item['eager_validation']['prediction_file']))
        verify_ts_predictions(confined_file(output.root,item['eager_validation']['prediction_file']),
            confined_file(output.root,item['TS_validation']['prediction_file']))
        sealed[family] = {'source':source,'torchscript':ts,
            'eager_validation':item['eager_validation'],'TS_validation':item['TS_validation']}
    selection = FrozenSelection(PilotProtocol().sha256,
        tuple((family,sources[family]['artifact_id']) for family in FAMILIES),
        tuple((sources[family]['artifact_id'],sources[family]['sha256']) for family in FAMILIES))
    gpu.runtime.check(deadline)
    seal = {**asdict(selection), 'evaluation_sha256':evaluation.sha256(), 'artifacts':sealed,
            'frozen_at':time.perf_counter(), 'official_test_previous_use_disclosed':True,
            'quality_scope':evaluation.payload()['quality_scope']}
    path = output.json('frozen_selection.json',seal)
    return selection, {'path':path,'sha256':sha256_file(output.root/path),
                       'file_bytes':(output.root/path).stat().st_size}


def run_official_test(sources, quality, gpu, *, deadline, report):
    selection, seal = seal_selection(sources,quality,gpu,deadline)
    report['frozen_selection'] = seal
    gpu.runtime.check(deadline)
    test_data = gpu.data.open_test(selection)
    report.update(official_test_access=True,test_opened_after_frozen_selection=True)
    if len(test_data[0]) != 10000 or len(test_data[1]) != 10000:
        raise ValueError('The complete official 10000-image test is required')
    results = report.setdefault('official_test',{})
    for family in FAMILIES:
        _verify_owned(gpu.runtime.output,seal)
        record = quality[family]['torchscript']
        model = load_timing_model(gpu.runtime.output.root,record,gpu,deadline)
        evidence = evaluate_saved(model,gpu,tuple(range(10000)),deadline=deadline,
            label=family+'_TS_official_test',test_data=test_data)
        results[family] = {**evidence,'torchscript_sha256':record['sha256'],
                           'source_checkpoint_sha256':sources[family]['sha256']}
        gpu.runtime.output.json(f'official_test/{family}.json',results[family])
        del model
    reference = results['M0']
    report['descriptive_quality'] = {family:{
        'top1_drop_from_matched_M0':reference['top1']-item['top1'],
        'CE_difference_from_matched_M0':item['cross_entropy']-reference['cross_entropy'],
        'selection_performed':False} for family,item in results.items()}


def _raw_ratios(record, clock):
    pairs = record.get('pairs',())
    if (len(pairs) != 20 or [row.get('pair') for row in pairs] != list(range(20))
            or [row.get('order') for row in pairs] != ['AB','BA']*10
            or record.get('orders') != ['AB','BA']*10):
        raise ValueError('Incomplete or changed raw timing pair protocol')
    ratios = []
    for pair in pairs:
        observations = [pair.get(side,{}).get(clock+'_seconds') for side in ('A','B')]
        if any(type(value) not in (int,float) or not math.isfinite(value) or value <= 0
               for value in observations):
            raise ValueError('Raw timing observations must be finite positive seconds')
        ratios.append(observations[1]/observations[0])
    return ratios


def aa_accepted(result):
    """Recompute both clock gates from raw A/B seconds, never cached ratios."""
    if type(result) is not dict or result.get('aa') is not True or set(result.get('batches',{})) != {'1','128'}:
        return False
    try:
        return all(aa_admission(_raw_ratios(record,clock),['AB','BA']*10)['accepted']
                   for record in result['batches'].values() for clock in ('cuda_event','synchronized_wall'))
    except (ValueError,TypeError,KeyError,ZeroDivisionError,OverflowError):
        return False


def _full_pairs(result):
    if type(result) is not dict or set(result.get('batches',{})) != {'1','128'}:
        return False
    try:
        for record in result['batches'].values():
            for clock in ('cuda_event','synchronized_wall'):
                _raw_ratios(record,clock)
    except (ValueError,TypeError,KeyError,ZeroDivisionError,OverflowError):
        return False
    return True


def run_timing(artifacts, gpu, *, deadline, results):
    """Six AA, up to four AB, then one AA; no retry after any failed gate."""
    if tuple(artifacts) != FAMILIES:
        raise ValueError('Timing requires exactly the five validation-sealed TorchScripts')
    output = gpu.runtime.output
    aa, ab = results.setdefault('AA',{}), results.setdefault('AB',{})
    for family in FAMILIES[1:]:
        ab[family] = {'admitted_before_AB':False,'comparison_admitted':False,
                      'status':'not_reached','reason':'pre_AB_controls_not_completed'}

    def same_artifact(family,label):
        record = artifacts[family]
        first = load_timing_model(output.root,record,gpu,deadline)
        second = load_timing_model(output.root,record,gpu,deadline)
        probe = verify_timing_probe(first,second,gpu,deadline)
        result = paired_latency(first,second,gpu,deadline=deadline,label=label,aa=True)
        result.update(family=family,torchscript=record,probe=probe,load_policy='independent_direct_CUDA',
                      raw_AA_admitted=aa_accepted(result))
        aa[label] = result
        output.json(f'latency/{label}.json',result)
        del first,second

    post_accepted = False
    try:
        for family,label in PRE_AA:
            same_artifact(family,label)
        m0 = aa_accepted(aa['M0_initial_AA']) and aa_accepted(aa['M0_pre_AB_AA'])
        results['M0_pre_AB_controls_accepted'] = m0
        for family in FAMILIES[1:]:
            admitted = m0 and aa_accepted(aa[family+'_AA'])
            ab[family].update(admitted_before_AB=admitted,status='skipped' if not admitted else 'started',
                              reason='pre_AB_AA_gate' if not admitted else 'post_AB_AA_pending')
            output.json(f'latency/M0_vs_{family}_AB.json',ab[family])
            if not admitted:
                continue
            first = load_timing_model(output.root,artifacts['M0'],gpu,deadline)
            reference = load_timing_model(output.root,artifacts['M0'],gpu,deadline)
            first_probe = verify_timing_probe(first,reference,gpu,deadline)
            del reference
            second = load_timing_model(output.root,artifacts[family],gpu,deadline)
            reference = load_timing_model(output.root,artifacts[family],gpu,deadline)
            second_probe = verify_timing_probe(second,reference,gpu,deadline)
            del reference
            result = paired_latency(first,second,gpu,deadline=deadline,label=f'M0_vs_{family}_AB')
            result.update(admitted_before_AB=True,comparison_admitted=False,status='measured',
                reason='post_AB_AA_pending',family=family,first_torchscript=artifacts['M0'],
                second_torchscript=artifacts[family],first_probe=first_probe,second_probe=second_probe,
                load_policy='independent_direct_CUDA')
            ab[family] = result
            output.json(f'latency/M0_vs_{family}_AB.json',result)
            del first,second
        same_artifact('M0',POST_AA)
        post_accepted = aa_accepted(aa[POST_AA])
    finally:
        results['M0_post_AB_control_accepted'] = post_accepted
        for family,item in ab.items():
            item['comparison_admitted'] = bool(item['admitted_before_AB'] and post_accepted and _full_pairs(item))
            if item['admitted_before_AB']:
                item['reason'] = ('all_AA_controls_accepted' if item['comparison_admitted']
                                  else 'post_AB_M0_AA_failed_or_incomplete' if not post_accepted
                                  else 'incomplete_AB_pairs')
            output.json(f'latency/M0_vs_{family}_AB.json',item)


def count_workload(ledger):
    """Count verified batch/pair prefixes even if the last ledger line is bad."""
    limits = {family+suffix:rows for family in FAMILIES for suffix,rows in (
        ('_eager_validation',5000),('_TS_validation',5000),('_TS_official_test',10000))}
    timing_labels = {label for _,label in PRE_AA} | {POST_AA} | {f'M0_vs_{family}_AB' for family in FAMILIES[1:]}
    rows, pairs, error = {}, {}, None
    if not Path(ledger).exists():
        return {'evaluation_rows':rows,'timing_pairs':pairs,'training_updates':0},None
    try:
        with Path(ledger).open(encoding='utf-8') as stream:
            for number,line in enumerate(stream,1):
                event = json.loads(line)
                if type(event) is not dict:
                    raise ValueError(f'Non-object event at ledger line {number}')
                kind = event.get('event')
                if kind in ('diagnostic_SGD_update','recovery_update','baseline_epoch'):
                    raise ValueError('Training is forbidden in saved evaluation')
                if kind == 'saved_evaluation_batch':
                    label, end, size = event.get('label'),event.get('rows'),event.get('batch_rows')
                    if (label not in limits or type(end) is not int or type(size) is not int
                            or size != min(128,limits[label]-rows.get(label,0))
                            or not 0 < end <= limits[label] or end != rows.get(label,0)+size):
                        raise ValueError(f'Duplicated, invalid or over-budget evaluation batch at line {number}')
                    rows[label] = end
                if kind == 'latency_pair':
                    label,batch,index = event.get('label'),event.get('batch'),event.get('pair')
                    key = f'{label}/batch_{batch}'
                    expected = pairs.get(key,0)
                    if (label not in timing_labels or type(batch) is not int or batch not in (1,128)
                            or type(index) is not int or index != expected or index >= 20
                            or event.get('order') != ('AB' if index%2 == 0 else 'BA')):
                        raise ValueError(f'Duplicated, invalid or over-budget timing pair at line {number}')
                    pairs[key] = expected+1
    except (OSError,ValueError,TypeError,UnicodeError) as failure:
        error = f'{type(failure).__name__}: {failure}'
    return {'evaluation_rows':rows,'timing_pairs':pairs,'training_updates':0},error


def _close_artifacts(output, sources, report):
    for source in sources.values():
        _verify_owned(output,source)
        _verify_owned(output,source['validation_reference'])
    _verify_owned(output,report['frozen_selection'])
    for family in FAMILIES:
        item = report['quality'][family]
        _verify_owned(output,item['torchscript'])
        for evidence in (item['eager_validation'],item['TS_validation'],report['official_test'][family]):
            if sha256_file(confined_file(output.root,evidence['prediction_file'])) != evidence['prediction_sha256']:
                raise ValueError('Persisted per-row evaluation evidence changed before close')
        if report['official_test'][family]['torchscript_sha256'] != item['torchscript']['sha256']:
            raise ValueError('Official test and timing must use the same persisted TorchScript bytes')
    if (set(report['timing'].get('AA',{})) != {label for _,label in PRE_AA}|{POST_AA}
            or any(not _full_pairs(row) for row in report['timing']['AA'].values())
            or set(report['timing'].get('AB',{})) != set(FAMILIES[1:])
            or any(not _full_pairs(row) if row.get('admitted_before_AB') else row.get('status') != 'skipped'
                   for row in report['timing']['AB'].values())):
        raise ValueError('Completion requires all seven full AA and four measured or explicitly skipped AB entries')


def run_evaluation(data_root, output_root, manifest, recovery_root, recovery_inventory, *, started=None):
    started = time.perf_counter() if started is None else started
    # Do not create a failure report in an existing or overlapping immutable tree.
    _guard_paths(data_root,output_root,manifest,recovery_root,recovery_inventory)
    if Path(output_root).exists():
        raise FileExistsError('Evaluation output must be a new directory')
    protocol, runtime, output, prepared = PilotProtocol(), None, None, None
    sources = {}
    report = {'schema':evaluation.payload()['schema'],'status':'started','started_at':started,
        'protocol_sha256':protocol.sha256,'evaluation_sha256':evaluation.sha256(),
        'recovery_report_sha256':evaluation.PARENT_REPORT_SHA,'recovery_inventory_sha256':evaluation.PARENT_INVENTORY_SHA,
        'automatic_retry':False,'baseline_training_performed':False,'search_performed':False,
        'training_updates':0,'total_SGD_updates':0,'additional_BN_calibration':False,
        'official_test_access':False,'official_test_previous_use_disclosed':True,
        'quality_scope':evaluation.payload()['quality_scope'],'quality':{},'official_test':{},'timing':{}}
    try:
        prepared = evaluation_preparation(data_root,output_root,manifest,recovery_root,recovery_inventory)
        if prepared['status'] != 'prepared':
            raise ValueError('Prepared official CIFAR files are required')
        output = Output(output_root)
        output.json('protocol.json',protocol.payload())
        output.json('evaluation_protocol.json',evaluation.payload())
        output.json('source_manifest.json',prepared['source'])
        output.json('recovery_admission.json',prepared['recovery'])
        report['gpu_admission'] = admit_free_gpu()
        runtime = Runtime(protocol,output,started)
        runtime.deadline = min(runtime.deadline,started+evaluation.WORKER_SECONDS)
        report['absolute_deadline'] = runtime.deadline
        parent = prepared['recovery']
        with completion_phase(runtime,'manifest',dict(evaluation.STAGES)['manifest']) as deadline:
            from .cv_sparse_data import CifarData
            data = CifarData(data_root,protocol,download=False)
            root, old = Path(parent['root']),parent['report']
            if (data.roles.payload() != json.loads(confined_file(root,'data_roles.json').read_text(encoding='utf-8'))
                    or data.source_hashes != old['data']['source_files'] or data.roles.sha256 != old['data']['roles_sha256']):
                raise ValueError('Dataset bytes or roles changed since the immutable recovery parent')
            if runtime.environment != old['environment']:
                raise ValueError('Scientific runtime environment differs from the recovery parent')
            freeze = subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True,timeout=30).stdout
            if frozen_dependencies(freeze) != frozen_dependencies(confined_file(root,'environment_freeze.txt').read_text(encoding='utf-8')):
                raise ValueError('Scientific dependency versions differ from the recovery parent')
            output.bytes('environment_freeze.txt',freeze.encode())
            output.json('environment.json',runtime.environment)
            output.json('data_roles.json',data.roles.payload())
            report.update(environment=runtime.environment,data={'source_files':data.source_hashes,'roles_sha256':data.roles.sha256})
            sources = copy_sources(parent,output)
            report['sources'] = sources
            gpu = GPUData(data,runtime)
            output.json('imported_sources.json',verify_loaded_sources(prepared['source']))
        with completion_phase(runtime,'validation',dict(evaluation.STAGES)['validation']) as deadline:
            run_validation(sources,gpu,deadline=deadline,results=report['quality'])
        with completion_phase(runtime,'test',dict(evaluation.STAGES)['test']) as deadline:
            run_official_test(sources,report['quality'],gpu,deadline=deadline,report=report)
        with completion_phase(runtime,'timing',dict(evaluation.STAGES)['timing']) as deadline:
            run_timing({family:report['quality'][family]['torchscript'] for family in FAMILIES},gpu,
                       deadline=deadline,results=report['timing'])
        with completion_phase(runtime,'close',dict(evaluation.STAGES)['close']):
            if inspect_recovery(recovery_root,recovery_inventory)['verified_files'] != parent['verified_files']:
                raise ValueError('Immutable recovery parent changed during evaluation')
            report['parents_read_only_verified'] = True
            current = source_manifest(manifest,protocol)
            if current != prepared['source']:
                raise ValueError('Evaluation source manifest or package sources changed during execution')
            _close_artifacts(output,sources,report)
            output.json('imported_sources.json',verify_loaded_sources(prepared['source']))
            runtime.check()
            report['status'] = 'completed'
    except PilotStop as error:
        report.update(status='evaluation_incomplete_'+error.stage,stop_reason=error.reason)
    except (Exception,PilotFatal) as error:
        report.update(status='failed',failure_type=type(error).__name__,failure=str(error),traceback=traceback.format_exc())
    finally:
        if output is None:
            output = Output(output_root)
        counts, failure = count_workload(output.ledger)
        rows = counts['evaluation_rows']
        report.update(observed_workload=counts,
            total_eager_validation_rows=sum(value for key,value in rows.items() if key.endswith('_eager_validation')),
            total_TS_validation_rows=sum(value for key,value in rows.items() if key.endswith('_TS_validation')),
            total_official_test_rows=sum(value for key,value in rows.items() if key.endswith('_TS_official_test')),
            total_latency_pairs=sum(counts['timing_pairs'].values()))
        report['total_validation_rows'] = report['total_eager_validation_rows']+report['total_TS_validation_rows']
        report['total_evaluated_rows'] = sum(rows.values())
        if report['status'] == 'completed' and (report['total_eager_validation_rows'] != 25000
                or report['total_TS_validation_rows'] != 25000 or report['total_official_test_rows'] != 50000
                or report['total_latency_pairs'] != 280+40*sum(
                    row['admitted_before_AB'] for row in report['timing']['AB'].values())):
            failure = failure or 'Completed saved evaluation requires every fixed row and admitted timing pair'
        if report['total_evaluated_rows'] > 100000 or report['total_latency_pairs'] > 440:
            failure = failure or 'Saved evaluation exceeded its fixed workload budget'
        if failure is not None:
            report.update(status='failed',ledger_count_failure=failure)
        report['ended_at'] = time.perf_counter()
        report['wall_seconds'] = report['ended_at']-started
        report['charged_slot_seconds'] = report['wall_seconds']
        report['total_charged_seconds'] = evaluation.PRIOR_SECONDS+report['wall_seconds']
        limit = min(started+evaluation.WORKER_SECONDS,runtime.deadline if runtime is not None else started+evaluation.WORKER_SECONDS)
        report['within_slot_deadline'] = report['ended_at'] <= limit
        if report['status'] == 'completed' and (not report['within_slot_deadline'] or report['total_charged_seconds'] > 14400):
            report.update(status='evaluation_incomplete_close',stop_reason='absolute_deadline')
        if runtime is not None:
            try:
                report.update(peak_allocated_bytes=runtime.torch.cuda.max_memory_allocated(),
                              peak_reserved_bytes=runtime.torch.cuda.max_memory_reserved())
            except Exception as error:
                report.update(status='failed',final_memory_accounting_failure=str(error))
        if report['status'] != 'completed':
            for family,item in report['timing'].get('AB',{}).items():
                item.update(comparison_admitted=False,terminal_report_admitted=False,
                            terminal_invalidation_reason=report['status'])
                output.json(f'latency/M0_vs_{family}_AB.json',item)
        output.json('report.json',report)
        output.append({'event':'saved_evaluation_stopped','status':report['status'],'ended_at':report['ended_at']})
    return report


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    names = ('data-root','output-root','source-manifest','recovery-root','recovery-inventory')
    for name in names:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args(argv)
    values = tuple(getattr(args,name.replace('-','_')) for name in names)
    if not args.execute:
        result = evaluation_preparation(*values)
        print(json.dumps({'status':result['status'],'execute':False,
            'recovery_files':len(result['recovery']['verified_files']),
            'evaluation_sha256':evaluation.sha256(),'max_seconds':evaluation.MAX_SECONDS}))
        return 0 if result['status'] == 'prepared' else 2
    result = run_evaluation(*values,started=started)
    print(json.dumps({'status':result['status'],'output':args.output_root}))
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
