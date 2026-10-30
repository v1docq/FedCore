"""CPU integrity, lineage and orchestration contracts; no GPU execution."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from fedcore.experiments import cv_sparse_evaluation as module


RESULTS = Path(__file__).resolve().parents[2]/'docs/svd_compression/results'


def predictions(rows=5000):
    return {'row_ids':np.arange(rows,dtype=np.int64),
            'labels':np.arange(rows,dtype=np.int64)%10,
            'predictions':np.arange(rows,dtype=np.int64)%10,
            'cross_entropy':np.ones(rows,dtype=np.float64)}


def write_npz(path, values):
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**values)


def miniature_parent(tmp_path,monkeypatch):
    root = tmp_path/'parent'; root.mkdir()
    report = json.loads((RESULTS/'recovery-bn-matrix-2026-10-26/raw/report.json').read_text(encoding='utf-8'))
    environment = {'scientific':'unchanged'}
    roles = {'train':[5001,5002],'calibration':list(range(1000)),
             'search_validation':list(range(5000)),'seed':990031}
    report.update(environment=environment,data={'source_files':{'training':'fixed'},'roles_sha256':module.digest(roles)})
    (root/'environment.json').write_text(json.dumps(environment))
    (root/'environment_freeze.txt').write_text('numpy==1.26.4\nfedcore==old\n')
    (root/'data_roles.json').write_text(json.dumps(roles))
    pinned = []
    for family,_,_,storage in module.evaluation.ARTIFACTS:
        item = report['quality'][family+'_SGD_lr_0.001_pre_BN']
        path = root/item['path']; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(('immutable-checkpoint-'+family).encode())
        item.update(sha256=module.sha256_file(path),file_bytes=path.stat().st_size)
        item['shared_pre_BN_checkpoint_sha256'] = item['sha256']
        item['artifact_id'] = module.digest({'solution_id':item['solution_id'],
            'checkpoint_sha256':item['sha256'],'phase':item['phase']})
        write_npz(root/item['validation']['prediction_file'],predictions())
        item['validation']['prediction_sha256'] = module.sha256_file(root/item['validation']['prediction_file'])
        pinned.append((family,item['artifact_id'],item['sha256'],storage))
    monkeypatch.setattr(module.evaluation,'ARTIFACTS',tuple(pinned))
    inventory = tmp_path/'inventory.json'
    def seal():
        (root/'report.json').write_text(json.dumps(report))
        files = {path.relative_to(root).as_posix():{'bytes':path.stat().st_size,'sha256':module.sha256_file(path)}
                 for path in root.rglob('*') if path.is_file()}
        files['external-supervisor-control.json'] = {'bytes':123,'sha256':'0'*64}
        inventory.write_text(json.dumps({'files':files}))
        monkeypatch.setattr(module.evaluation,'PARENT_REPORT_SHA',module.sha256_file(root/'report.json'))
        monkeypatch.setattr(module.evaluation,'PARENT_INVENTORY_SHA',module.sha256_file(inventory))
    seal()
    return root,inventory,report,roles,seal


def source_payload():
    files = {'fedcore/experiments/'+name+'.py':'a'*64 for name in (
        'cv_sparse_continuation','cv_sparse_diagnostics','cv_sparse_diagnostic_protocol',
        'cv_sparse_recovery','cv_sparse_recovery_protocol','cv_sparse_evaluation','cv_sparse_evaluation_protocol')}
    return {'verified_package_files_sha256':files,'supplied':{'evaluation_protocol_sha256':module.evaluation.sha256()}}


def test_fixed_five_source_states_and_scalar_workload():
    assert module.FAMILIES == ('M0','P75','R','E','hybrid')
    assert [family for family,_,_,_ in module.evaluation.ARTIFACTS] == list(module.FAMILIES)
    assert module.evaluation.STAGES == (('manifest',240),('validation',480),('test',240),('timing',600),('close',240))
    protocol = module.evaluation.payload()
    assert protocol['training_updates'] == 0 and protocol['additional_BN_calibration'] is False
    assert protocol['eager_validation_rows']+protocol['TS_validation_rows'] == 50000
    assert protocol['TS_official_test_rows'] == 50000
    assert protocol['AA_sessions']*40+protocol['AB_sessions_max']*40 == protocol['latency_pairs_max'] == 440
    assert protocol['official_test_previous_use_disclosed'] is True
    assert protocol['automatic_retry'] is False


def test_parent_admission_reads_only_fourteen_consumed_files(tmp_path,monkeypatch):
    root,inventory,report,_,_ = miniature_parent(tmp_path,monkeypatch)
    before = {path:path.read_bytes() for path in root.rglob('*') if path.is_file()}
    inspected = module.inspect_recovery(root,inventory)
    assert len(inspected['verified_files']) == 14
    assert tuple(inspected['sources']) == module.FAMILIES
    assert 'external-supervisor-control.json' not in inspected['verified_files']
    assert {path:path.read_bytes() for path in before} == before
    # Test metrics and dict order cannot promote another state.
    report['quality'] = dict(reversed(list(report['quality'].items())))
    for name,item in report['quality'].items():
        if not name.endswith('SGD_lr_0.001_pre_BN'):
            item['validation']['top1'] = 1.
    assert tuple(module._selected_sources(report)) == module.FAMILIES


@pytest.mark.parametrize('mutation',[
    'inventory','report_bytes','checkpoint_bytes','prediction_bytes','status','deadline','protocol',
    'recovery_protocol','test_access','new_search','BN','lr','updates','state','family','solution','artifact',
    'cost','phase','training_BN','training_updates','training_tape','prediction_sha','checkpoint_size','escape',
    'environment','data_roles','parent_total_budget','parent_slot_budget'])
def test_parent_integrity_and_metadata_fail_closed(tmp_path,monkeypatch,mutation):
    root,inventory,report,_,seal = miniature_parent(tmp_path,monkeypatch)
    item = report['quality']['R_SGD_lr_0.001_pre_BN']
    if mutation == 'status': report['status'] = 'failed'
    if mutation == 'deadline': report['within_slot_deadline'] = False
    if mutation == 'protocol': report['protocol_sha256'] = '0'*64
    if mutation == 'recovery_protocol': report['recovery_sha256'] = '0'*64
    if mutation == 'test_access': report['official_test_access'] = True
    if mutation == 'new_search': report['search_performed'] = True
    if mutation == 'BN': item['additional_BN_calibration'] = True
    if mutation == 'lr': item['lr'] = .01
    if mutation == 'updates': item['SGD_updates'] = 199
    if mutation == 'state': item['state'] = 'no_SGD'
    if mutation == 'family': item['family'] = 'E'
    if mutation == 'solution': item['solution_id'] = '0'*64
    if mutation == 'artifact': item['artifact_id'] = '0'*64
    if mutation == 'cost': item['tensor_bytes'] += 4
    if mutation == 'phase': item['phase'] = 'dense_original'
    if mutation == 'training_BN': item['training']['additional_BN_calibration'] = True
    if mutation == 'training_updates': item['training']['updates'] = 201
    if mutation == 'training_tape': item['training']['tape_sha256'] = '0'*64
    if mutation == 'prediction_sha': item['validation']['prediction_sha256'] = '0'*64
    if mutation == 'checkpoint_size': item['file_bytes'] += 1
    if mutation == 'escape': item['path'] = '../outside.pt'
    if mutation == 'environment': report['environment']['scientific'] = 'different'
    if mutation == 'data_roles': report['data']['roles_sha256'] = '0'*64
    if mutation == 'parent_total_budget': report['total_charged_seconds'] = 14401.
    if mutation == 'parent_slot_budget': report['charged_slot_seconds'] = 1801.
    if mutation not in ('inventory','report_bytes','checkpoint_bytes','prediction_bytes'): seal()
    if mutation == 'inventory': inventory.write_text('{}')
    if mutation == 'report_bytes': (root/'report.json').write_text('{}')
    if mutation == 'checkpoint_bytes': (root/item['path']).write_bytes(b'tampered')
    if mutation == 'prediction_bytes': (root/item['validation']['prediction_file']).write_bytes(b'tampered')
    with pytest.raises(ValueError): module.inspect_recovery(root,inventory)


def test_preflight_binds_new_modules_protocol_and_disjoint_input_trees(tmp_path,monkeypatch):
    root,inventory,_,_,_ = miniature_parent(tmp_path,monkeypatch)
    source = source_payload()
    monkeypatch.setattr(module,'preparation',lambda *args:{'status':'prepared','source':source})
    values = (tmp_path/'data',tmp_path/'out',tmp_path/'manifest',root,inventory)
    assert module.evaluation_preparation(*values)['status'] == 'prepared'
    assert not values[1].exists()
    source['supplied']['evaluation_protocol_sha256'] = '0'*64
    with pytest.raises(ValueError,match='protocol'): module.evaluation_preparation(*values)
    source['supplied']['evaluation_protocol_sha256'] = module.evaluation.sha256()
    del source['verified_package_files_sha256']['fedcore/experiments/cv_sparse_evaluation.py']
    with pytest.raises(ValueError,match='module'): module.evaluation_preparation(*values)
    for output in (root/'inside',tmp_path,values[0]/'out'):
        with pytest.raises(ValueError,match='disjoint'):
            module.evaluation_preparation(values[0],output,*values[2:])
    with pytest.raises(ValueError,match='manifest'):
        module.evaluation_preparation(values[0],values[1],values[1]/'manifest',root,inventory)
    with pytest.raises(ValueError,match='inventory'):
        module.evaluation_preparation(values[0],values[1],values[2],root,values[1]/'inventory')


def test_default_cli_is_standard_library_only_and_never_creates_output(tmp_path):
    code = '''import sys
from pathlib import Path
from fedcore.experiments.cv_sparse_evaluation import main
names=('data-root','output-root','source-manifest','recovery-root','recovery-inventory')
args=[part for name,value in zip(names,sys.argv[1:]) for part in ('--'+name,value)]
try: main(args)
except (FileNotFoundError,ValueError): pass
assert not any(name=='torch' or name.startswith(('torch.','torchvision','fedcore.algorithm','numpy')) for name in sys.modules)
assert not Path(sys.argv[2]).exists()
'''
    subprocess.run([sys.executable,'-c',code,*[str(tmp_path/name) for name in (
        'data','out','manifest','parent','inventory')]],check=True)


@pytest.mark.parametrize('field',['row_ids','labels','predictions','cross_entropy','dtype','schema','rows','nonfinite'])
def test_eager_replay_requires_all_four_arrays_exact(tmp_path,field):
    left,right = tmp_path/'eager_old.npz',tmp_path/'eager_new.npz'
    values = predictions(); write_npz(left,values); write_npz(right,values)
    assert module.verify_dense_predictions(left,right)['exact'] is True
    if field in values: values[field][4999] += 1e-12 if field == 'cross_entropy' else 1
    if field == 'dtype': values['row_ids'] = values['row_ids'].astype(np.int32)
    if field == 'schema': values['logits'] = values['predictions'].copy()
    if field == 'rows': values = {key:value[:-1] for key,value in values.items()}
    if field == 'nonfinite': values['cross_entropy'][15] = np.inf
    write_npz(right,values)
    with pytest.raises(ValueError,match='Dense replay'): module.verify_dense_predictions(left,right)


@pytest.mark.parametrize('mutation',['row_ids','labels','predictions','dtype','schema','rows','nonfinite','CE_outside'])
def test_torchscript_equivalence_is_exact_except_fixed_CE_tolerance(tmp_path,mutation):
    left,right = tmp_path/'eager.npz',tmp_path/'ts.npz'
    values = predictions(); write_npz(left,values)
    if mutation in ('row_ids','labels','predictions'): values[mutation][4567] += 1
    if mutation == 'dtype': values['labels'] = values['labels'].astype(np.int32)
    if mutation == 'schema': values.pop('labels')
    if mutation == 'rows': values = {key:value[:-1] for key,value in values.items()}
    if mutation == 'nonfinite': values['cross_entropy'][0] = np.nan
    if mutation == 'CE_outside': values['cross_entropy'][0] += .000021
    write_npz(right,values)
    with pytest.raises(ValueError,match='TorchScript'): module.verify_ts_predictions(left,right)


def test_torchscript_tiny_CE_boundary_uses_frozen_absolute_tolerance(tmp_path):
    left,right = tmp_path/'eager.npz',tmp_path/'ts.npz'
    values = predictions(); values['cross_entropy'].fill(0.)
    write_npz(left,values)
    values['cross_entropy'][4999] = module.evaluation.TS_CE_ATOL
    write_npz(right,values)
    result = module.verify_ts_predictions(left,right)
    assert result['CE_accepted'] is True and result['CE_atol'] == result['CE_rtol'] == 1e-5
    values['cross_entropy'][4999] = np.nextafter(module.evaluation.TS_CE_ATOL,np.inf)
    write_npz(right,values)
    with pytest.raises(ValueError,match='CE exceeds'): module.verify_ts_predictions(left,right)


def raw_latency(label,aa=True,ratio=1.):
    pairs = [{'pair':index,'order':'AB' if index%2 == 0 else 'BA',
        'A':{'cuda_event_seconds':.001,'synchronized_wall_seconds':.002},
        'B':{'cuda_event_seconds':.001*ratio,'synchronized_wall_seconds':.002*ratio},
        'cuda_event_ratio_B_over_A':ratio,'synchronized_wall_ratio_B_over_A':ratio} for index in range(20)]
    return {'label':label,'aa':aa,'accepted':True,'batches':{
        str(batch):{'pairs':copy.deepcopy(pairs),'orders':['AB','BA']*10,
            'accepted':True,'cuda_event_admission':{'accepted':True},
            'synchronized_wall_admission':{'accepted':True}} for batch in (1,128)}}


@pytest.mark.parametrize('mutation',['raw_event','raw_wall','orders','pair_index','missing_batch','nonfinite','same_ratio_lie'])
def test_AA_gate_recomputes_every_clock_from_raw_seconds(mutation):
    result = raw_latency('AA')
    assert module.aa_accepted(result) is True
    if mutation == 'raw_event': result['batches']['1']['pairs'][0]['B']['cuda_event_seconds'] = .01
    if mutation == 'raw_wall':
        for pair in result['batches']['128']['pairs']: pair['B']['synchronized_wall_seconds'] *= 1.2
    if mutation == 'orders': result['batches']['1']['pairs'][0]['order'] = 'BA'
    if mutation == 'pair_index': result['batches']['128']['pairs'][19]['pair'] = 18
    if mutation == 'missing_batch': del result['batches']['128']
    if mutation == 'nonfinite': result['batches']['1']['pairs'][0]['A']['cuda_event_seconds'] = float('nan')
    if mutation == 'same_ratio_lie':
        for pair in result['batches']['1']['pairs']:
            pair['B']['cuda_event_seconds'] *= 1.1
            pair['cuda_event_ratio_B_over_A'] = 1.
    # A single event outlier is permitted by the original 18/20 AA rule. Three are not.
    if mutation == 'raw_event':
        for pair in result['batches']['1']['pairs'][:3]: pair['B']['cuda_event_seconds'] = .01
    assert module.aa_accepted(result) is False


def timing_stack(tmp_path,monkeypatch,*,failed=(),raise_post=False):
    output = module.Output(tmp_path/'timing')
    events,loads = [],[]
    artifacts = {family:{'artifact_id':family,'path':family+'.pt','sha256':family,'file_bytes':1}
                 for family in module.FAMILIES}
    gpu = SimpleNamespace(runtime=SimpleNamespace(output=output))
    def load(root,record,gpu,deadline,**kwargs):
        assert kwargs.get('cpu_then_cuda',False) is False
        model = SimpleNamespace(family=record['artifact_id'],instance=len(loads))
        loads.append(model)
        return model
    def probe(first,second,*args):
        assert first is not second and first.family == second.family
        return {'same_device_logits_accepted':True}
    def latency(first,second,gpu,*,deadline,label,aa=False):
        events.append(label)
        if not aa:
            for path in (output.root/'latency').glob('M0_vs_*_AB.json'):
                assert json.loads(path.read_text())['comparison_admitted'] is False
        if label == module.POST_AA and raise_post:
            raise module.PilotStop('timing','absolute_deadline')
        result = raw_latency(label,aa,1.2 if label in failed else 1.)
        for batch,record in result['batches'].items():
            for pair in record['pairs']:
                output.append({'event':'latency_pair','label':label,'batch':int(batch),**pair})
        return result
    monkeypatch.setattr(module,'load_timing_model',load)
    monkeypatch.setattr(module,'verify_timing_probe',probe)
    monkeypatch.setattr(module,'paired_latency',latency)
    return artifacts,gpu,events,loads


@pytest.mark.parametrize('failed', [(),('P75_AA',),('M0_initial_AA',),('M0_pre_AB_AA',),('M0_post_AB_AA',)])
def test_seven_AA_order_four_AB_gates_and_post_control(tmp_path,monkeypatch,failed):
    artifacts,gpu,events,loads = timing_stack(tmp_path,monkeypatch,failed=failed)
    result = {}
    module.run_timing(artifacts,gpu,deadline=10.,results=result)
    assert events[:6] == [label for _,label in module.PRE_AA]
    assert events[-1] == module.POST_AA and len(result['AA']) == 7 and len(result['AB']) == 4
    expected = [] if any(name.startswith('M0_') and name != module.POST_AA for name in failed) else [
        family for family in module.FAMILIES[1:] if family+'_AA' not in failed]
    assert events[6:-1] == [f'M0_vs_{family}_AB' for family in expected]
    assert len(events) == len(set(events))  # Failed gates are never retried.
    for family,item in result['AB'].items():
        assert item['admitted_before_AB'] is (family in expected)
        assert item['comparison_admitted'] is (family in expected and module.POST_AA not in failed)
        if family in expected:
            assert module._full_pairs(item) and item['first_probe']['same_device_logits_accepted']
            assert item['second_probe']['same_device_logits_accepted']
    counts,error = module.count_workload(gpu.runtime.output.ledger)
    assert error is None and sum(counts['timing_pairs'].values()) == (7+len(expected))*40
    assert len(loads) == 14+4*len(expected)


def test_post_AA_timeout_retains_raw_AB_but_invalidates_every_comparison(tmp_path,monkeypatch):
    artifacts,gpu,events,_ = timing_stack(tmp_path,monkeypatch,raise_post=True)
    result = {}
    with pytest.raises(module.PilotStop): module.run_timing(artifacts,gpu,deadline=10.,results=result)
    assert events[-1] == module.POST_AA
    for family,item in result['AB'].items():
        assert item['admitted_before_AB'] is True and item['comparison_admitted'] is False
        saved = json.loads((gpu.runtime.output.root/f'latency/M0_vs_{family}_AB.json').read_text())
        assert module._full_pairs(saved) and saved['comparison_admitted'] is False


def emit_rows(output,label,rows):
    for start in range(0,rows,128):
        size = min(128,rows-start)
        output.append({'event':'saved_evaluation_batch','label':label,'batch_rows':size,'rows':start+size})


def test_partial_batch_and_pair_accounting_preserves_valid_prefix(tmp_path):
    output = module.Output(tmp_path/'counts')
    emit_rows(output,'M0_eager_validation',256)
    output.append({'event':'latency_pair','label':'M0_initial_AA','batch':1,'pair':0,'order':'AB'})
    with output.ledger.open('a') as stream: stream.write('{broken')
    counts,error = module.count_workload(output.ledger)
    assert counts == {'evaluation_rows':{'M0_eager_validation':256},
                      'timing_pairs':{'M0_initial_AA/batch_1':1},'training_updates':0}
    assert error is not None


@pytest.mark.parametrize('event',[
    {'event':'saved_evaluation_batch','label':'M0_eager_validation','batch_rows':128,'rows':128},
    {'event':'latency_pair','label':'M0_initial_AA','batch':1,'pair':1,'order':'BA'},
    {'event':'diagnostic_SGD_update','updates':1},
])
def test_accounting_rejects_retries_missing_pairs_and_any_training(tmp_path,event):
    output = module.Output(tmp_path/'counts')
    emit_rows(output,'M0_eager_validation',128)
    output.append(event)
    counts,error = module.count_workload(output.ledger)
    assert counts['evaluation_rows']['M0_eager_validation'] == 128 and error is not None


def test_real_CPU_toy_trace_saved_reload_predictions_and_partial_prefix(tmp_path,monkeypatch):
    torch = pytest.importorskip('torch')
    model = torch.nn.Sequential(torch.nn.Linear(2,10),torch.nn.ReLU()).eval()
    x = torch.arange(10000,dtype=torch.float32).reshape(5000,2)/10000
    labels = torch.arange(5000,dtype=torch.int64)%10
    output = module.Output(tmp_path/'toy')
    path = output._target('torchscript/toy.pt')
    traced = torch.jit.trace(model,x[:2],check_trace=True)
    torch.jit.save(traced,str(path))
    direct_loads = []
    def cpu_load(path,*,map_location):
        direct_loads.append(map_location)
        return torch.jit.load(path,map_location='cpu')
    class CpuTorch:
        jit = SimpleNamespace(load=cpu_load)
        def __getattr__(self,name): return getattr(torch,name)
        def as_tensor(self,*args,**kwargs):
            kwargs.pop('device',None)
            return torch.as_tensor(*args,**kwargs)
    runtime = SimpleNamespace(output=output,check=lambda *args,**kwargs:None)
    gpu = SimpleNamespace(runtime=runtime,torch=CpuTorch(),images=x,labels=labels,normalized=lambda value:value)
    record = {'path':'torchscript/toy.pt','sha256':module.sha256_file(path),'file_bytes':path.stat().st_size,'artifact_id':'toy'}
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        eager = module.evaluate_saved(model,gpu,tuple(range(5000)),deadline=10.,label='M0_eager_validation')
        first = module.load_timing_model(output.root,record,gpu,10.)
        second = module.load_timing_model(output.root,record,gpu,10.)
        assert first is not second and direct_loads == ['cuda','cuda']
        script = module.evaluate_saved(first,gpu,tuple(range(5000)),deadline=10.,label='M0_TS_validation')
        assert module.verify_ts_predictions(output.root/eager['prediction_file'],output.root/script['prediction_file'])['CE_accepted']
        def stop_after_two_batches(deadline=None,**progress):
            if progress.get('rows') == 256: raise module.PilotStop('validation','absolute_deadline')
        runtime.check = stop_after_two_batches
        with pytest.raises(module.PilotStop):
            module.evaluate_saved(second,gpu,tuple(range(5000)),deadline=10.,label='P75_TS_validation')
        with np.load(output.root/'predictions/P75_TS_validation.npz',allow_pickle=False) as saved:
            assert set(saved.files) == module.FIELDS and len(saved['row_ids']) == 256
        counts,error = module.count_workload(output.ledger)
        assert error is None and counts['evaluation_rows']['P75_TS_validation'] == 256
        path.write_bytes(b'changed')
        runtime.check = lambda *args,**kwargs:None
        with pytest.raises(ValueError,match='TorchScript bytes changed'):
            module.load_timing_model(output.root,record,gpu,10.)
    finally:
        torch.set_num_threads(old_threads)


def validated_fixture(parent,output):
    sources = module.copy_sources(parent,output)
    quality = {}
    for family in module.FAMILIES:
        source = sources[family]
        reference = output.root/source['validation_reference']['path']
        ts = output.bytes('torchscript/'+family+'.pt',('saved-TS-'+family).encode())
        ts.update(artifact_id=source['artifact_id'],source_artifact_id=source['artifact_id'],
            source_checkpoint_sha256=source['sha256'],format='independently_reloaded_TorchScript')
        metrics = {}
        for key,suffix in (('eager_validation','_eager_validation'),('TS_validation','_TS_validation')):
            copied = output.bytes('predictions/'+family+suffix+'.npz',reference.read_bytes())
            metrics[key] = {'rows':5000,'prediction_file':copied['path'],'prediction_sha256':copied['sha256']}
            emit_rows(output,family+suffix,5000)
        quality[family] = {'source':source,**metrics,'torchscript':ts,'eager_replay':{'exact':True},
            'TS_equivalence':{'row_ids_labels_predictions_exact':True,'CE_accepted':True}}
    return sources,quality


@pytest.mark.parametrize('failure',[None,'eager','TS'])
def test_validation_uses_five_independent_sources_and_fresh_TS_not_export_object(tmp_path,monkeypatch,failure):
    root,inventory,_,_,_ = miniature_parent(tmp_path,monkeypatch)
    output = module.Output(tmp_path/'validation')
    sources = module.copy_sources(module.inspect_recovery(root,inventory),output)
    events = []
    runtime = SimpleNamespace(output=output,check=lambda *args,**kwargs:None)
    gpu = SimpleNamespace(runtime=runtime,data=SimpleNamespace(roles=SimpleNamespace(search_validation=tuple(range(5000)))))
    class Model:
        def __init__(self,family,kind): self.family,self.kind = family,kind
        def to(self,*args): return self
        def cpu(self): return self
    def eager_reload(source,*args):
        events.append(('eager_reload',source['family']))
        return Model(source['family'],'eager')
    def export(source,*args):
        events.append(('export',source['family']))
        saved = output.bytes('torchscript/'+source['family']+'.pt',b'saved-TS')
        saved.update(artifact_id=source['artifact_id'],format='independently_reloaded_TorchScript')
        return Model(source['family'],'legacy_export'),saved
    def direct_reload(root,record,*args):
        family = next(family for family in module.FAMILIES if sources[family]['artifact_id'] == record['artifact_id'])
        events.append(('direct_reload',family))
        return Model(family,'fresh_direct_TS')
    def evaluate(model,gpu,indices,*,deadline,label):
        expected = 'eager' if label.endswith('_eager_validation') else 'fresh_direct_TS'
        assert model.kind == expected and len(indices) == 5000
        events.append(('evaluate_'+expected,model.family))
        values = predictions()
        if model.family == 'P75' and failure == ('eager' if expected == 'eager' else 'TS'):
            values['predictions'][12] += 1
        saved = module._prediction_prefix(output,label,values['row_ids'],values['labels'],
            values['predictions'],values['cross_entropy'])
        return {'rows':5000,'prediction_file':saved['path'],'prediction_sha256':saved['sha256']}
    monkeypatch.setattr(module,'load_checkpoint',eager_reload)
    monkeypatch.setattr(module,'_model_identity',lambda *args:{'immutable':'state'})
    monkeypatch.setattr(module,'model_evidence',lambda *args:{'immutable':'state'})
    monkeypatch.setattr(module,'export_torchscript_artifact',export)
    monkeypatch.setattr(module,'load_timing_model',direct_reload)
    monkeypatch.setattr(module,'evaluate_saved',evaluate)
    result = {}
    if failure is None:
        module.run_validation(sources,gpu,deadline=10.,results=result)
        assert tuple(result) == module.FAMILIES
        assert events == [event for family in module.FAMILIES for event in (
            ('eager_reload',family),('evaluate_eager',family),('export',family),
            ('direct_reload',family),('evaluate_fresh_direct_TS',family))]
    else:
        with pytest.raises(ValueError): module.run_validation(sources,gpu,deadline=10.,results=result)
        assert {family for _,family in events} == {'M0','P75'}
        assert not (output.root/'frozen_selection.json').exists()
        if failure == 'eager': assert ('export','P75') not in events


@pytest.mark.parametrize('mutation',['missing','CE','eager','TS_SHA','TS_lineage','source_SHA','prediction_bytes'])
def test_seal_rejects_any_missing_validation_or_changed_lineage(tmp_path,monkeypatch,mutation):
    root,inventory,_,_,_ = miniature_parent(tmp_path,monkeypatch)
    output = module.Output(tmp_path/'sealed')
    sources,quality = validated_fixture(module.inspect_recovery(root,inventory),output)
    if mutation == 'missing': del quality['hybrid']
    if mutation == 'CE': quality['R']['TS_equivalence']['CE_accepted'] = False
    if mutation == 'eager': quality['R']['eager_replay']['exact'] = False
    if mutation == 'TS_SHA': quality['R']['torchscript']['sha256'] = '0'*64
    if mutation == 'TS_lineage': quality['R']['torchscript']['source_checkpoint_sha256'] = '0'*64
    if mutation == 'source_SHA': sources['R']['sha256'] = '0'*64
    if mutation == 'prediction_bytes': (output.root/quality['R']['eager_validation']['prediction_file']).write_bytes(b'changed')
    gpu = SimpleNamespace(runtime=SimpleNamespace(output=output,check=lambda *args,**kwargs:None))
    with pytest.raises(ValueError): module.seal_selection(sources,quality,gpu,10.)
    assert not (output.root/'frozen_selection.json').exists()


def test_all_five_TS_are_sealed_before_one_test_open_and_test_reloads(tmp_path,monkeypatch):
    root,inventory,_,_,_ = miniature_parent(tmp_path,monkeypatch)
    output = module.Output(tmp_path/'sealed')
    sources,quality = validated_fixture(module.inspect_recovery(root,inventory),output)
    events = []
    def open_test(selection):
        seal = json.loads((output.root/'frozen_selection.json').read_text())
        assert selection.protocol_sha256 == module.PilotProtocol().sha256
        assert seal['evaluation_sha256'] == module.evaluation.sha256()
        assert tuple(seal['artifacts']) == tuple(sorted(module.FAMILIES))  # Atomic JSON sorts map keys.
        assert set(seal['artifacts']) == set(module.FAMILIES)
        events.append('open_test')
        return np.empty((10000,1)),np.zeros(10000,dtype=np.int64)
    gpu = SimpleNamespace(runtime=SimpleNamespace(output=output,check=lambda *args,**kwargs:None),
                          data=SimpleNamespace(open_test=open_test))
    def reload(root,record,gpu,deadline):
        events.append(('reload',record['sha256']))
        return object()
    def evaluate(model,gpu,indices,*,deadline,label,test_data):
        assert len(indices) == 10000 and events[-1][0] == 'reload'
        family = label.removesuffix('_TS_official_test')
        saved = module._prediction_prefix(output,label,**dict(zip(
            ('row_ids','targets','predictions','losses'),predictions(10000).values())))
        emit_rows(output,label,10000)
        return {'rows':10000,'top1':1.,'cross_entropy':1.,'prediction_file':saved['path'],'prediction_sha256':saved['sha256']}
    monkeypatch.setattr(module,'load_timing_model',reload)
    monkeypatch.setattr(module,'evaluate_saved',evaluate)
    report = {}
    module.run_official_test(sources,quality,gpu,deadline=10.,report=report)
    assert events == ['open_test']+[('reload',quality[family]['torchscript']['sha256']) for family in module.FAMILIES]
    assert len(report['official_test']) == 5 and report['test_opened_after_frozen_selection'] is True
    assert all(row['selection_performed'] is False for row in report['descriptive_quality'].values())


def test_unknown_execution_error_still_writes_partial_report_without_gpu(tmp_path,monkeypatch):
    def fail(*args): raise RuntimeError('unknown input failure')
    monkeypatch.setattr(module,'evaluation_preparation',fail)
    output = tmp_path/'partial'
    result = module.run_evaluation(tmp_path/'data',output,tmp_path/'manifest',tmp_path/'parent',tmp_path/'inventory')
    assert result['status'] == 'failed' and result['failure_type'] == 'RuntimeError'
    assert result['total_evaluated_rows'] == result['total_latency_pairs'] == result['training_updates'] == 0
    assert json.loads((output/'report.json').read_text())['failure'] == 'unknown input failure'


def test_terminal_close_failure_invalidates_previously_admitted_AB(tmp_path,monkeypatch):
    root,inventory,_,_,_ = miniature_parent(tmp_path,monkeypatch)
    parent = module.inspect_recovery(root,inventory)
    prepared = {'status':'prepared','source':source_payload(),'recovery':parent}
    monkeypatch.setattr(module,'evaluation_preparation',lambda *args:prepared)
    monkeypatch.setattr(module,'admit_free_gpu',lambda:{'compute_pids':[]})
    monkeypatch.setattr(module,'verify_loaded_sources',lambda *args:{})
    fake_cuda = SimpleNamespace(max_memory_allocated=lambda:0,max_memory_reserved=lambda:0)
    class FakeRuntime:
        def __init__(self,protocol,output,started):
            self.protocol,self.output,self.deadline = protocol,output,started+1790
            self.environment = parent['report']['environment']
            self.torch = SimpleNamespace(cuda=fake_cuda)
        def check(self,*args,**kwargs): pass
    monkeypatch.setattr(module,'Runtime',FakeRuntime)
    roles = json.loads((root/'data_roles.json').read_text())
    fake_roles = SimpleNamespace(payload=lambda:roles,sha256=parent['report']['data']['roles_sha256'])
    data = SimpleNamespace(roles=fake_roles,source_hashes=parent['report']['data']['source_files'])
    from fedcore.experiments import cv_sparse_data
    monkeypatch.setattr(cv_sparse_data,'CifarData',lambda *args,**kwargs:data)
    monkeypatch.setattr(module,'GPUData',lambda data,runtime:SimpleNamespace(data=data,runtime=runtime))
    monkeypatch.setattr(module.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(stdout='numpy==1.26.4\nfedcore==new\n'))
    monkeypatch.setattr(module,'run_official_test',lambda *args,**kwargs:None)
    def timing(artifacts,gpu,*,results,**kwargs):
        results['AB'] = {'R':{**raw_latency('M0_vs_R_AB',False),
            'admitted_before_AB':True,'comparison_admitted':True}}
    # The wrapper still needs the five frozen TS keys to reach the test's timing stub.
    def validation(sources,gpu,*,results,**kwargs):
        emit_rows(gpu.runtime.output,'M0_eager_validation',128)
        results.update({family:{'torchscript':{'sha256':family}} for family in module.FAMILIES})
    monkeypatch.setattr(module,'run_validation',validation)
    monkeypatch.setattr(module,'run_timing',timing)
    monkeypatch.setattr(module,'source_manifest',lambda *args:(_ for _ in ()).throw(ValueError('source changed at close')))
    result = module.run_evaluation(tmp_path/'data',tmp_path/'terminal',tmp_path/'manifest',root,inventory)
    assert result['status'] == 'failed' and result['total_eager_validation_rows'] == 128
    assert result['timing']['AB']['R']['comparison_admitted'] is False
    assert result['timing']['AB']['R']['terminal_invalidation_reason'] == 'failed'
    saved = json.loads((tmp_path/'terminal/latency/M0_vs_R_AB.json').read_text())
    assert module._full_pairs(saved) and saved['comparison_admitted'] is False
