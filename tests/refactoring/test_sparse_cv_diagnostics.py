"""CPU contracts for the one bounded diagnostic series; no official test reads."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given, strategies as st

from fedcore.experiments import cv_sparse_diagnostics as module


def miniature_completion(tmp_path, monkeypatch):
    root = tmp_path/'completion'; root.mkdir()
    def write(name, payload):
        target = root/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return {'path':name, 'sha256':module.sha256_file(target), 'file_bytes':len(payload)}
    baseline = write('checkpoints/M0.pt', b'original-full-model')
    baseline.update(artifact_id=module.diagnostic.TIMING_ARTIFACTS[0][1], phase='dense_original', solution_id='dense')
    predictions = write('predictions/original.npz', b'original-predictions')
    baseline['validation'] = {'rows':5000, 'top1':.95, 'prediction_file':predictions['path'],
                              'prediction_sha256':predictions['sha256']}
    environment = {'scientific':'unchanged'}
    role_payload = {'train':[1,2], 'calibration':[3,4], 'search_validation':list(range(5000)), 'seed':990031}
    write('environment.json', json.dumps(environment).encode())
    write('environment_freeze.txt', b'numpy==1.26.4\nfedcore==old\n')
    write('data_roles.json', json.dumps(role_payload).encode())
    latency, recovered = {}, {}
    role_names = {'hybrid':'fixed_hybrid', 'R':'random_finalist', 'E':'evolution_finalist'}
    for role, identity in module.diagnostic.TIMING_ARTIFACTS:
        record = write(f'torchscript/{role}.pt', ('saved-TS-'+role).encode())
        record.update(artifact_id=identity, format='independently_reloaded_TorchScript')
        latency[identity] = {'torchscript':record}
        if role != 'M0':
            recovered[role] = {'artifact_id':identity, 'roles':[role_names[role]], 'recovery_tape_sha256':'common-tape'}
    report = {'status':'completed', 'within_slot_deadline':True,
              'protocol_sha256':module.PilotProtocol().sha256,
              'baseline_training_performed':False, 'search_performed':False,
              'baseline':baseline, 'latency':latency, 'recovered':recovered,
              'data':{'source_files':{'train':'fixed'}, 'roles_sha256':'roles'}, 'environment':environment}
    inventory = tmp_path/'published-inventory.json'
    def seal():
        write('report.json', json.dumps(report).encode())
        files = {path.relative_to(root).as_posix():{'bytes':path.stat().st_size, 'sha256':module.sha256_file(path)}
                 for path in root.rglob('*') if path.is_file()}
        # Published controls were exported out of the worker output tree.
        files['launcher-exported-elsewhere.py'] = {'bytes':10, 'sha256':'0'*64}
        inventory.write_text(json.dumps({'files':files}), encoding='utf-8')
        monkeypatch.setattr(module.diagnostic,'COMPLETION_REPORT_SHA',module.sha256_file(root/'report.json'))
        monkeypatch.setattr(module.diagnostic,'COMPLETION_INVENTORY_SHA',module.sha256_file(inventory))
        monkeypatch.setattr(module.diagnostic,'BASELINE_SHA',baseline['sha256'])
    seal()
    return root, inventory, report, seal


def test_integrity_admits_only_consumed_files_and_preserves_input_bytes(tmp_path, monkeypatch):
    root, inventory, _, _ = miniature_completion(tmp_path, monkeypatch)
    before = {path:path.read_bytes() for path in root.rglob('*') if path.is_file()}
    first = module.inspect_completion(root,inventory)
    assert first == module.inspect_completion(root,inventory)
    assert len(first['verified_files']) == 10
    assert 'launcher-exported-elsewhere.py' not in first['verified_files']
    assert {path:path.read_bytes() for path in before} == before
    (root/'torchscript/R.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='integrity'):
        module.inspect_completion(root,inventory)


@pytest.mark.parametrize('mutation', ['report_hash','inventory_hash','baseline_sha','TS_metadata','role','escape'])
def test_changed_completion_identity_or_confined_paths_fail_closed(tmp_path, monkeypatch, mutation):
    root, inventory, report, seal = miniature_completion(tmp_path, monkeypatch)
    identity = module.diagnostic.TIMING_ARTIFACTS[1][1]
    if mutation == 'baseline_sha': report['baseline']['sha256'] = '0'*64
    if mutation == 'TS_metadata': report['latency'][identity]['torchscript']['sha256'] = '0'*64
    if mutation == 'role': report['recovered']['hybrid']['roles'] = ['fixed_prune']
    if mutation == 'escape': report['latency'][identity]['torchscript']['path'] = '../outside.pt'
    if mutation not in ('report_hash','inventory_hash'): seal()
    if mutation == 'report_hash': (root/'report.json').write_bytes(b'{}')
    if mutation == 'inventory_hash': inventory.write_bytes(b'{}')
    with pytest.raises(ValueError): module.inspect_completion(root,inventory)


def test_preparation_requires_diagnostic_sources_and_fixed_protocol_without_outputs(tmp_path, monkeypatch):
    root, inventory, _, _ = miniature_completion(tmp_path, monkeypatch)
    files = {'fedcore/experiments/'+name+'.py':'a'*64 for name in (
        'cv_sparse_continuation','cv_sparse_diagnostics','cv_sparse_diagnostic_protocol')}
    source = {'verified_package_files_sha256':files,
              'supplied':{'diagnostic_protocol_sha256':module.diagnostic.sha256()}}
    monkeypatch.setattr(module,'preparation',lambda *args:{'status':'prepared','source':source})
    result = module.diagnostic_preparation(tmp_path/'data',tmp_path/'out',tmp_path/'source',root,inventory)
    assert result['status']=='prepared' and not (tmp_path/'out').exists()
    source['supplied']['diagnostic_protocol_sha256'] = '0'*64
    with pytest.raises(ValueError, match='protocol'):
        module.diagnostic_preparation(tmp_path/'data',tmp_path/'out',tmp_path/'source',root,inventory)
    source['supplied']['diagnostic_protocol_sha256'] = module.diagnostic.sha256()
    files.pop('fedcore/experiments/cv_sparse_diagnostics.py')
    with pytest.raises(ValueError, match='module'):
        module.diagnostic_preparation(tmp_path/'data',tmp_path/'out',tmp_path/'source',root,inventory)
    with pytest.raises(ValueError, match='disjoint'):
        module.diagnostic_preparation(tmp_path/'data',root/'nested',tmp_path/'source',root,inventory)


def test_default_command_never_imports_torch_or_creates_output(tmp_path):
    code = '''import sys
from fedcore.experiments.cv_sparse_diagnostics import main
try: main(['--data-root',sys.argv[1], '--output-root',sys.argv[2], '--source-manifest',sys.argv[3],
           '--completion-root',sys.argv[4], '--completion-inventory',sys.argv[5]])
except (FileNotFoundError,ValueError): pass
assert not any(name=='torch' or name.startswith(('torch.','torchvision','fedcore.algorithm')) for name in sys.modules)
from pathlib import Path
assert not Path(sys.argv[2]).exists()
'''
    subprocess.run([sys.executable,'-c',code,*[str(tmp_path/name) for name in ('data','out','source','completion','inventory')]],check=True)


def accepted_aa():
    pairs = [{'order':'AB' if index%2==0 else 'BA', 'cuda_event_ratio_B_over_A':1.,
              'synchronized_wall_ratio_B_over_A':1.} for index in range(20)]
    batch = {'pairs':pairs, 'orders':[row['order'] for row in pairs], 'accepted':True,
             'cuda_event_admission':{'accepted':True}, 'synchronized_wall_admission':{'accepted':True}}
    return {'aa':True, 'accepted':True, 'batches':{'1':copy.deepcopy(batch),'128':copy.deepcopy(batch)}}


@given(st.sampled_from(('1','128')), st.sampled_from(('cuda_event','synchronized_wall')))
def test_gate_requires_each_complete_batch_and_clock_even_if_flags_claim_success(batch, clock):
    result = accepted_aa()
    assert module.aa_accepted(result)
    for pair in result['batches'][batch]['pairs']:
        pair[clock+'_ratio_B_over_A'] = 1.2
    assert not module.aa_accepted(result)


@pytest.mark.parametrize('mutation', ['missing_batch','nineteen_pairs','wrong_order','false_clock','not_AA'])
def test_incomplete_or_failed_AA_never_opens_AB(mutation):
    result = accepted_aa()
    if mutation == 'missing_batch': result['batches'].pop('128')
    if mutation == 'nineteen_pairs': result['batches']['1']['pairs'].pop()
    if mutation == 'wrong_order': result['batches']['1']['pairs'][0]['order']='BA'
    if mutation == 'false_clock': result['batches']['128']['synchronized_wall_admission']['accepted']=False
    if mutation == 'not_AA': result['aa']=False
    assert not module.aa_accepted(result)


@pytest.mark.parametrize('failed', [None,'M0_canonical_AA','M0_final_canonical_AA','hybrid_canonical_AA','R_legacy_AA'])
def test_timing_runs_nine_fixed_AA_once_then_only_admitted_fresh_canonical_AB(tmp_path, monkeypatch, failed):
    loads, measurements = [], []
    artifacts = {role:{'artifact_id':identity,'path':role+'.pt','sha256':role,'file_bytes':1}
                 for role,identity in module.diagnostic.TIMING_ARTIFACTS}
    class Model:
        pass
    def load(root,record,gpu,deadline,**kwargs):
        model=Model(); model.role=record['path']; model.legacy=kwargs.get('cpu_then_cuda',False)
        loads.append(model)
        return model
    def latency(first,second,gpu,*,deadline,label,aa=False):
        assert first is not second
        if aa:
            assert first.role==second.role and second.legacy is False
            assert first.legacy==('_legacy_' in label)
            result=accepted_aa()
            if label==failed: result['accepted']=False
        else:
            assert len([row for row in measurements if row[1]])==9
            assert not first.legacy and not second.legacy and first.role=='M0.pt'
            result={'aa':False,'batches':{}}
        measurements.append((label,aa,first,second))
        return result
    monkeypatch.setattr(module,'load_timing_model',load)
    monkeypatch.setattr(module,'verify_timing_probe',lambda *args:{'accepted':True})
    monkeypatch.setattr(module,'paired_latency',latency)
    gpu=SimpleNamespace(runtime=SimpleNamespace(output=module.Output(tmp_path/'out')))
    results={}
    module.run_timing(tmp_path,artifacts,gpu,deadline=100.,results=results)
    assert [row[0] for row in measurements[:9]]==[
        'M0_canonical_AA','M0_legacy_AA','hybrid_legacy_AA','hybrid_canonical_AA',
        'R_canonical_AA','R_legacy_AA','E_legacy_AA','E_canonical_AA','M0_final_canonical_AA']
    assert len({id(model) for model in loads})==len(loads)
    if failed in ('M0_canonical_AA','M0_final_canonical_AA'):
        expected=[]
    elif failed=='hybrid_canonical_AA': expected=['R','E']
    else: expected=['hybrid','R','E']
    assert [row[0] for row in measurements[9:]]==[f'M0_vs_{role}_canonical_AB' for role in expected]
    assert len(results['AA'])==9
    if failed: assert results['AA'][failed]['accepted'] is False
    # 18 AA modules and four new modules per admitted AB, including untimed probes.
    assert len(loads)==18+4*len(expected)


@pytest.mark.parametrize('cpu_then_cuda', [False,True])
def test_timing_load_policy_and_same_bytes_are_real_helper_contracts(tmp_path,cpu_then_cuda):
    path=tmp_path/'frozen.pt'; path.write_bytes(b'same-TS-bytes')
    calls=[]
    class Model:
        def eval(self): return self
        def to(self,device): calls.append(('to',device)); return self
    model=Model()
    def load(filename,*,map_location):
        assert Path(filename).read_bytes()==b'same-TS-bytes'
        calls.append(('load',map_location)); return model
    gpu=SimpleNamespace(torch=SimpleNamespace(jit=SimpleNamespace(load=load)),
                        runtime=SimpleNamespace(check=lambda *args,**kwargs:None))
    record={'path':path.name,'sha256':module.sha256_file(path),'file_bytes':path.stat().st_size,'artifact_id':'id'}
    assert module.load_timing_model(tmp_path,record,gpu,100.,cpu_then_cuda=cpu_then_cuda) is model
    assert calls==([('load','cpu'),('to','cuda')] if cpu_then_cuda else [('load','cuda')])
    path.write_bytes(b'changed')
    with pytest.raises(ValueError): module.load_timing_model(tmp_path,record,gpu,100.)


def test_probe_accepts_export_tolerance_and_rejects_layout_drift():
    torch=pytest.importorskip('torch')
    class Probe:
        def to(self,device): assert device=='cuda'; return torch.tensor([1.,2.])
    gpu=SimpleNamespace(torch=torch,runtime=SimpleNamespace(check=lambda *args:None),
        data=SimpleNamespace(rows=lambda ids:Probe(),roles=SimpleNamespace(calibration=(1,2))))
    first=lambda x:x
    close=lambda x:x+1e-6
    result=module.verify_timing_probe(first,close,gpu,100.)
    assert result['max_abs_difference']>0 and result['atol']==result['rtol']==1e-5
    with pytest.raises(AssertionError): module.verify_timing_probe(first,lambda x:x+.1,gpu,100.)
    with pytest.raises(ValueError): module.verify_timing_probe(first,first,gpu,100.)


@pytest.mark.parametrize('lr', [.01,.001])
def test_real_SGD_helper_uses_all_200_common_updates_and_training_mode(lr,monkeypatch):
    torch=pytest.importorskip('torch')
    events,seen=[],[]
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight=torch.nn.Parameter(torch.tensor([.3,-.2])); self.training_calls=0
            self.bn=torch.nn.BatchNorm1d(2)
        def to(self,device): assert device=='cuda'; return self
        def forward(self,x):
            assert self.training
            self.training_calls+=1
            return self.bn(x[:,None]*self.weight[None,:])
    model=Model(); before=model.weight.detach().clone()
    tape=tuple(((index,index+1),990032+index) for index in range(200))
    def training(ids,seed):
        seen.append((ids,seed)); return torch.tensor([.2,1.]),torch.tensor([0,1])
    gpu=SimpleNamespace(torch=torch,training=training,runtime=SimpleNamespace(protocol=module.PilotProtocol(),
        output=SimpleNamespace(append=events.append),check=lambda *args,**kwargs:None))
    optimizers=[]; original_optimizer=torch.optim.SGD
    def optimizer(parameters,**kwargs):
        optimizers.append(kwargs)
        return original_optimizer(parameters,**kwargs)
    monkeypatch.setattr(torch.optim,'SGD',optimizer)
    original_threads=torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        proof=module.recover_without_calibration(model,gpu,tape,deadline=100.,role='trajectory',lr=lr)
    finally:
        torch.set_num_threads(original_threads)
    assert seen==list(tape) and model.training_calls==200
    assert not torch.equal(model.weight.detach(),before)
    assert model.bn.training and int(model.bn.num_batches_tracked)==200
    assert not torch.equal(model.bn.running_mean,torch.zeros(2))
    assert optimizers==[{'lr':lr,'momentum':.9,'weight_decay':.0005,'nesterov':True}]
    assert [event['updates'] for event in events]==list(range(1,201))
    assert all(event['lr']==lr and event['tape_sha256']==module.digest(tape) for event in events)
    assert proof['updates']==200 and proof['training_BN_updates'] is True
    assert proof['additional_BN_calibration'] is False and model.training


def test_quality_pair_owns_pre_BN_checkpoint_and_starts_each_trajectory_from_M0(tmp_path,monkeypatch):
    output=module.Output(tmp_path/'out'); events=[]
    class Model:
        def __init__(self): self.weights=0; self.bn=0
        def to(self,device): return self
        def cpu(self): return self
    baseline_bytes=b'fullM0'; baseline=output.bytes('checkpoints/M0.pt',baseline_bytes)
    baseline.update(solution_id='dense',validation={'rows':5000,'top1':.95})
    original=output.bytes('original.npz',b'predictions')
    def load(*args): events.append(('fresh_M0',)); return Model()
    def recover(model,gpu,tape,*,lr,**kwargs):
        assert model.weights==model.bn==0
        events.append(('SGD',lr,id(tape))); model.weights=200; model.bn=200
        return {'updates':200,'lr':lr}
    def save(model,gpu,*,label,**kwargs):
        record=output.bytes('checkpoints/'+label+'.pt',json.dumps([model.weights,model.bn]).encode())
        events.append(('saved',label,model.weights,model.bn))
        return record,copy.deepcopy(model)
    def calibrate(model,*args): model.bn=-1
    def evaluate(model,gpu,indices,*,label,**kwargs):
        events.append(('validation',label,model.weights,model.bn))
        record=output.bytes('predictions/'+label+'.npz',b'predictions')
        return {'rows':len(indices),'top1':.95,'prediction_file':record['path']}
    def verify(old,new): assert Path(old).read_bytes()==Path(new).read_bytes()==b'predictions'
    monkeypatch.setattr(module,'load_baseline',load)
    monkeypatch.setattr(module,'recover_without_calibration',recover)
    monkeypatch.setattr(module,'checkpoint',save)
    monkeypatch.setattr(module,'calibrate_bn',calibrate)
    monkeypatch.setattr(module,'evaluate',evaluate)
    monkeypatch.setattr(module,'verify_baseline_predictions',verify)
    gpu=SimpleNamespace(runtime=SimpleNamespace(output=output),data=SimpleNamespace(roles=SimpleNamespace(search_validation=tuple(range(5000)))))
    tape=(((1,2),3),); results={}
    module.run_quality(baseline,gpu,tape,deadline=100.,previous_predictions=output.root/original['path'],results=results)
    assert len(results)==6 and sum(event[0]=='fresh_M0' for event in events)==4
    assert [(event[2],event[3]) for event in events if event[0]=='validation']==[
        (0,0),(0,-1),(200,200),(200,-1),(200,200),(200,-1)]
    assert [(event[1],event[2]) for event in events if event[0]=='SGD']==[(.01,id(tape)),(.001,id(tape))]
    for lr in (.01,.001):
        pre=results[f'SGD_lr_{lr:g}_pre_BN']; post=results[f'SGD_lr_{lr:g}_post_BN']
        assert (output.root/pre['path']).read_bytes()==b'[200, 200]'
        assert (output.root/post['path']).read_bytes()==b'[200, -1]'
        assert post['shared_200_update_checkpoint_sha256']==pre['sha256']
    assert (output.root/baseline['path']).read_bytes()==baseline_bytes


@pytest.mark.parametrize('mismatch', [None,'environment','dependencies','roles'])
def test_public_run_admits_sources_then_quality_then_timing_and_never_test(tmp_path,monkeypatch,mismatch):
    root,inventory,old,seal=miniature_completion(tmp_path,monkeypatch)
    roles=json.loads((root/'data_roles.json').read_text())
    tape=(((1,2),3),)
    for artifact in old['recovered'].values(): artifact['recovery_tape_sha256']=module.digest(tape)
    seal(); completion=module.inspect_completion(root,inventory)
    monkeypatch.setattr(module,'diagnostic_preparation',lambda *args:{'status':'prepared','source':{},'completion':completion})
    monkeypatch.setattr(module,'inspect_completion',lambda *args:completion)
    monkeypatch.setattr(module,'admit_free_gpu',lambda:{'compute_pids':[]})
    monkeypatch.setattr(module,'verify_loaded_sources',lambda *args:{})
    freeze=(root/'environment_freeze.txt').read_text()
    monkeypatch.setattr(module.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(stdout=freeze+'scipy==changed\n' if mismatch=='dependencies' else freeze))
    class Runtime:
        def __init__(self,protocol,output,started):
            self.protocol,self.output,self.deadline=protocol,output,started+1800
            self.environment={} if mismatch=='environment' else old['environment']
            self.torch=SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda:0,max_memory_reserved=lambda:0))
        def check(self,*args,**kwargs): pass
    class Data:
        def __init__(self,*args,**kwargs):
            assert kwargs=={'download':False}
            self.roles=SimpleNamespace(train=(1,2),sha256='roles',payload=lambda:{} if mismatch=='roles' else roles)
            self.source_hashes=old['data']['source_files']
        def open_test(self,*args): pytest.fail('Official test must stay sealed')
    monkeypatch.setitem(sys.modules,'fedcore.experiments.cv_sparse_data',SimpleNamespace(CifarData=Data))
    monkeypatch.setattr(module,'Runtime',Runtime)
    monkeypatch.setattr(module,'GPUData',lambda data,runtime:SimpleNamespace(data=data,runtime=runtime))
    monkeypatch.setattr(module,'recovery_tape',lambda *args:tape)
    events=[]
    def quality(baseline,gpu,shared_tape,**kwargs):
        assert shared_tape is tape and gpu.runtime.stage=='quality'
        assert Path(kwargs['previous_predictions']).read_bytes()==b'original-predictions'
        kwargs['results']['untouched']={**baseline}
        events.append('quality')
    def timing(parent,artifacts,gpu,**kwargs):
        assert gpu.runtime.stage=='timing' and Path(parent)==root
        assert artifacts==completion['timing_artifacts']
        events.append('timing')
    monkeypatch.setattr(module,'run_quality',quality)
    monkeypatch.setattr(module,'run_timing',timing)
    result=module.run_diagnostics(tmp_path/'data',tmp_path/'out',tmp_path/'source',root,inventory)
    if mismatch:
        assert result['status']=='failed' and events==[]
    else:
        assert result['status']=='completed',result.get('traceback')
        assert events==['quality','timing']
        assert result['total_SGD_updates']==400 and result['total_validation_rows']==30000
    assert result['official_test_access'] is result['search_performed'] is result['baseline_training_performed'] is False


def test_deadline_stop_keeps_partial_raw_evidence_and_does_not_retry(tmp_path,monkeypatch):
    gpu=SimpleNamespace(runtime=SimpleNamespace(output=module.Output(tmp_path/'out')))
    count=[]
    monkeypatch.setattr(module,'load_timing_model',lambda *args,**kwargs:object())
    monkeypatch.setattr(module,'verify_timing_probe',lambda *args:{})
    def stopped(*args,**kwargs):
        count.append(kwargs['label'])
        gpu.runtime.output.append({'event':'latency_pair','label':kwargs['label'],'pair':0,'A':{'cuda_event_seconds':.1}})
        raise module.PilotStop('timing','absolute_deadline')
    monkeypatch.setattr(module,'paired_latency',stopped)
    artifacts={role:{} for role,_ in module.diagnostic.TIMING_ARTIFACTS}
    with pytest.raises(module.PilotStop): module.run_timing(tmp_path,artifacts,gpu,deadline=100.,results={})
    assert count==['M0_canonical_AA']
    assert 'latency_pair' in gpu.runtime.output.ledger.read_text()
