"""CPU contracts for the fixed recovery/extra-BN matrix; no training launch."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from fedcore.experiments import cv_sparse_recovery as module


RESULTS = Path(__file__).resolve().parents[2]/'docs/svd_compression/results'


def miniature_parents(tmp_path, monkeypatch):
    completion = tmp_path/'completion'; completion.mkdir()
    diagnostic = tmp_path/'diagnostic'; diagnostic.mkdir()
    old = json.loads((RESULTS/'a100-completion-2026-10-26/raw/report.json').read_text())
    previous = json.loads((RESULTS/'bn-ts-diagnostic-2026-10-26/raw/report.json').read_text())
    environment = {'scientific':'fixed'}
    roles = {'train':[1,2], 'calibration':list(range(1000)), 'search_validation':list(range(5000))}
    data = {'source_files':{'training':'unchanged'}, 'roles_sha256':'fixed-roles'}
    def write(root, name, payload):
        target = root/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return {'path':name, 'sha256':module.sha256_file(target), 'file_bytes':len(payload)}
    for root, report in ((completion,old),(diagnostic,previous)):
        report.update(environment=environment, data=data)
        write(root,'environment.json',json.dumps(environment).encode())
        write(root,'environment_freeze.txt',b'numpy==1.26.4\nfedcore==old\n')
        write(root,'data_roles.json',json.dumps(roles).encode())
    baseline = old['baseline']
    record = write(completion,baseline['path'],b'original-M0-immutable')
    baseline.update(sha256=record['sha256'],file_bytes=record['file_bytes'])
    record = write(completion,baseline['validation']['prediction_file'],b'baseline-predictions')
    baseline['validation']['prediction_sha256'] = record['sha256']
    for name, item in previous['quality'].items():
        record = write(diagnostic,item['validation']['prediction_file'],('old-'+name).encode())
        item['validation']['prediction_sha256'] = record['sha256']
    inventories = (tmp_path/'completion-inventory.json',tmp_path/'diagnostic-inventory.json')
    def seal():
        for index, (root, report, prefix) in enumerate(((completion,old,'COMPLETION'),(diagnostic,previous,'DIAGNOSTIC'))):
            if index:
                report.update(completion_report_sha256=module.recovery.COMPLETION_REPORT_SHA,
                              completion_inventory_sha256=module.recovery.COMPLETION_INVENTORY_SHA)
            write(root,'report.json',json.dumps(report).encode())
            files = {path.relative_to(root).as_posix():{'bytes':path.stat().st_size, 'sha256':module.sha256_file(path)}
                     for path in root.rglob('*') if path.is_file()}
            files['external-worker-launcher.log'] = {'bytes':10,'sha256':'0'*64}
            inventories[index].write_text(json.dumps({'files':files}))
            monkeypatch.setattr(module.recovery,prefix+'_REPORT_SHA',module.sha256_file(root/'report.json'))
            monkeypatch.setattr(module.recovery,prefix+'_INVENTORY_SHA',module.sha256_file(inventories[index]))
        monkeypatch.setattr(module.recovery,'BASELINE_SHA',baseline['sha256'])
    seal()
    return completion, diagnostic, inventories, old, previous, seal


def test_plan_has_thirty_conditions_and_exact_scalar_workload():
    plan = module.quality_plan()
    assert plan == module.quality_plan() and len(plan) == len({row.label for row in plan}) == 30
    assert [row.family for row in plan[::6]] == ['M0','P75','R','E','hybrid']
    assert sum(row.updates for row in plan if not row.extra_bn) == 2000
    assert sum(row.updates for row in plan) == 4000  # Paired evaluations are not trajectories.
    assert sum(row.lr is not None and not row.extra_bn for row in plan) == 10
    assert sum(row.extra_bn for row in plan) == 15 and len(plan)*5000 == 150000
    assert module.recovery.payload()['total_updates'] == 2000
    assert module.recovery.STAGES == (('manifest',300),('quality',1200),('close',300))
    assert module.recovery.WORKER_SECONDS == 1790 and module.recovery.MAX_SECONDS == 1800
    for family in ('M0','P75','R','E','hybrid'):
        rows = [row for row in plan if row.family == family]
        assert [(row.lr,row.extra_bn) for row in rows] == [
            (None,False),(None,True),(.01,False),(.01,True),(.001,False),(.001,True)]


def test_actual_original_payloads_decode_to_typed_pinned_graphs():
    from fedcore.algorithm.low_rank.cv_grammar import CVGrammarSpace, CVBlockDomain, CVGenotype, compile_cv_genotype
    from fedcore.algorithm.low_rank.compression_graph import ModelVersion, CalibrationRef
    from fedcore.experiments.cv_sparse_pilot import cv_menu, solution_id
    old = json.loads((RESULTS/'a100-completion-2026-10-26/raw/report.json').read_text())
    payloads = module.selected_payloads(old)
    space = CVGrammarSpace(ModelVersion('test','a'*64),CalibrationRef('b'*64,(1000,3,32,32),'float32'),
        tuple(CVBlockDomain(path,options) for path,options in cv_menu().items()),'c'*64,44734408)
    genotypes = module.decode_genotypes(payloads,SimpleNamespace(space=space))
    for family, identity, storage in module.recovery.FAMILIES:
        assert type(genotypes[family]) is CVGenotype and solution_id(genotypes[family]) == identity
        assert compile_cv_genotype(genotypes[family],space).expected_storage_bytes == storage
    changed = copy.deepcopy(payloads)
    changed[module.recovery.FAMILIES[1][1]]['genes'][0]['choice']['keep_channels'] = 32
    with pytest.raises(ValueError,match='genotype'):
        module.decode_genotypes(changed,SimpleNamespace(space=space))
    old['recovered'][module.recovery.FAMILIES[2][1]]['roles'] = ['random_finalist']
    with pytest.raises(ValueError,match='roles'):
        module.selected_payloads(old)


def test_parents_verify_only_consumed_files_and_are_read_only(tmp_path,monkeypatch):
    completion, diagnostic, inventories, _, _, _ = miniature_parents(tmp_path,monkeypatch)
    before = {path:path.read_bytes() for root in (completion,diagnostic) for path in root.rglob('*') if path.is_file()}
    admitted = module.inspect_completion(completion,inventories[0])
    previous = module.inspect_diagnostic(diagnostic,inventories[1])
    assert len(admitted['verified_files']) == 6 and len(previous['verified_files']) == 10
    assert not any('torchscript' in name or 'external-worker' in name for name in admitted['verified_files'])
    assert {path:path.read_bytes() for path in before} == before
    (diagnostic/next(iter(previous['predictions'].values()))).write_bytes(b'tampered')
    with pytest.raises(ValueError,match='integrity'):
        module.inspect_diagnostic(diagnostic,inventories[1])


@pytest.mark.parametrize('mutation',['inventory','report','baseline','genotype','R_role','escape','dense_state','dense_sha'])
def test_parent_tamper_or_changed_matrix_fails_closed(tmp_path,monkeypatch,mutation):
    completion, diagnostic, inventories, old, previous, seal = miniature_parents(tmp_path,monkeypatch)
    if mutation == 'baseline': old['baseline']['sha256'] = '0'*64
    if mutation == 'genotype': old['inherited_short_attempts']['fixed_prune'][0]['genotype']['genes'].pop()
    if mutation == 'R_role': old['recovered'][module.recovery.FAMILIES[2][1]]['roles'] = ['fixed_svd']
    if mutation == 'escape': old['baseline']['path'] = '../outside.pt'
    if mutation == 'dense_state': previous['quality']['SGD_lr_0.01_pre_BN']['SGD_updates'] = 199
    if mutation == 'dense_sha': previous['quality']['M0_post_BN']['validation']['prediction_sha256'] = '0'*64
    if mutation not in ('inventory','report'): seal()
    if mutation == 'inventory': inventories[0].write_bytes(b'{}')
    if mutation == 'report': (completion/'report.json').write_bytes(b'{}')
    with pytest.raises(ValueError):
        module.inspect_completion(completion,inventories[0])
        module.inspect_diagnostic(diagnostic,inventories[1])


def test_preflight_pins_new_modules_protocol_and_disjoint_output(tmp_path,monkeypatch):
    completion, diagnostic, inventories, _, _, _ = miniature_parents(tmp_path,monkeypatch)
    files = {'fedcore/experiments/'+name+'.py':'a'*64 for name in (
        'cv_sparse_continuation','cv_sparse_diagnostics','cv_sparse_diagnostic_protocol',
        'cv_sparse_recovery','cv_sparse_recovery_protocol')}
    source = {'verified_package_files_sha256':files,'supplied':{'recovery_protocol_sha256':module.recovery.sha256()}}
    monkeypatch.setattr(module,'preparation',lambda *args:{'status':'prepared','source':source})
    values = (tmp_path/'data',tmp_path/'out',tmp_path/'manifest',completion,inventories[0],diagnostic,inventories[1])
    assert module.recovery_preparation(*values)['status'] == 'prepared'
    assert not (tmp_path/'out').exists()
    source['supplied']['recovery_protocol_sha256'] = '0'*64
    with pytest.raises(ValueError,match='protocol'): module.recovery_preparation(*values)
    source['supplied']['recovery_protocol_sha256'] = module.recovery.sha256()
    files.pop('fedcore/experiments/cv_sparse_recovery.py')
    with pytest.raises(ValueError,match='module'): module.recovery_preparation(*values)
    for out in (completion/'inside',diagnostic/'inside',tmp_path):
        with pytest.raises(ValueError,match='disjoint'):
            module.recovery_preparation(values[0],out,*values[2:])


def test_default_cli_never_imports_torch_or_creates_outputs(tmp_path):
    code = '''import sys
from fedcore.experiments.cv_sparse_recovery import main
names=('data-root','output-root','source-manifest','completion-root','completion-inventory','diagnostic-root','diagnostic-inventory')
args=[part for name,value in zip(names,sys.argv[1:]) for part in ('--'+name,value)]
try: main(args)
except (FileNotFoundError,ValueError): pass
assert not any(name=='torch' or name.startswith(('torch.','torchvision','fedcore.algorithm')) for name in sys.modules)
from pathlib import Path
assert not Path(sys.argv[2]).exists()
'''
    subprocess.run([sys.executable,'-c',code,*[str(tmp_path/name) for name in (
        'data','out','manifest','completion','completion-inventory','diagnostic','diagnostic-inventory')]],check=True)


@pytest.mark.parametrize('field',['row_ids','labels','predictions','cross_entropy','dtype','schema','rows','nonfinite'])
def test_dense_replay_checks_all_rows_exactly_and_does_not_fit_tolerance(tmp_path,field):
    values = {'row_ids':np.arange(5000,dtype=np.int64),'labels':np.zeros(5000,dtype=np.int64),
              'predictions':np.zeros(5000,dtype=np.int64),'cross_entropy':np.ones(5000,dtype=np.float64)}
    first, second = tmp_path/'old.npz',tmp_path/'new.npz'
    np.savez(first,**values); np.savez(second,**values)
    assert module.verify_dense_predictions(first,second)['exact'] is True
    changed = {name:array.copy() for name,array in values.items()}
    if field in values:
        changed[field][3456] += 1e-12 if field == 'cross_entropy' else 1
    if field == 'dtype': changed['row_ids'] = changed['row_ids'].astype(np.int32)
    if field == 'schema': changed['logits'] = changed['predictions'].copy()
    if field == 'rows': changed = {name:array[:-1] for name,array in changed.items()}
    if field == 'nonfinite': changed['cross_entropy'][0] = np.nan
    np.savez(second,**changed)
    with pytest.raises(ValueError,match='Dense replay'): module.verify_dense_predictions(first,second)


def tiny_model(torch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([.3,-.2]))
            self.bn = torch.nn.BatchNorm2d(2)
        def to(self,*args,**kwargs):
            if args == ('cuda',): return self
            return super().to(*args,**kwargs)
        def forward(self,x):
            return self.bn(x*self.weight.reshape(1,2,1,1)).mean((2,3))
    return Model()


def checkpoint_stack(output,monkeypatch,torch):
    snapshots, loads = {}, []
    def save(model,gpu,*,label,solution_id,phase,deadline):
        evidence = module.model_evidence(model)
        artifact = output.bytes('checkpoints/'+label+'.pt',json.dumps(evidence,sort_keys=True).encode())
        snapshots[artifact['path']] = copy.deepcopy(model)
        family = label.split('_')[0]
        size = next(size for role,_,size in module.recovery.FAMILIES if role == family)
        artifact.update(solution_id=solution_id,tensor_bytes=size,phase=phase)
        return artifact,copy.deepcopy(model)
    def load(artifact,gpu,deadline):
        assert module.sha256_file(output.root/artifact['path']) == artifact['sha256']
        model = copy.deepcopy(snapshots[artifact['path']]); loads.append(model)
        return model
    monkeypatch.setattr(module,'checkpoint',save)
    monkeypatch.setattr(module,'load_checkpoint',load)
    return save, snapshots, loads


@pytest.mark.parametrize('lr',[.01,.001])
def test_real_200_step_CPU_training_and_extra_BN_isolate_exact_weights(tmp_path,monkeypatch,lr):
    torch = pytest.importorskip('torch')
    model = tiny_model(torch)
    output = module.Output(tmp_path/'out'); events = []
    save, snapshots, loads = checkpoint_stack(output,monkeypatch,torch)
    actual_as_tensor = torch.as_tensor
    class CpuTorch:
        def __getattr__(self,name): return getattr(torch,name)
        def as_tensor(self,data,**kwargs):
            kwargs.pop('device',None)
            return actual_as_tensor(data,**kwargs)
    tape = tuple(((0,1),990032+step) for step in range(200))
    rows = torch.stack((torch.ones(2,2,2)*.2,torch.ones(2,2,2)),0)
    runtime = SimpleNamespace(protocol=module.PilotProtocol(),output=output,check=lambda *args,**kwargs:None)
    gpu = SimpleNamespace(torch=CpuTorch(),runtime=runtime,images=rows.repeat(500,1,1,1),normalized=lambda x:x,
        training=lambda ids,seed:(rows,torch.tensor([0,1])),
        data=SimpleNamespace(roles=SimpleNamespace(calibration=tuple(range(1000)),search_validation=tuple(range(5000)))))
    initial = module.model_evidence(model)
    actual_optimizer = torch.optim.SGD
    optimizer_calls = []
    def optimizer(parameters,**kwargs):
        optimizer_calls.append(kwargs)
        return actual_optimizer(parameters,**kwargs)
    monkeypatch.setattr(torch.optim,'SGD',optimizer)
    thread_count = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        proof = module.recover_without_calibration(model,gpu,tape,deadline=100.,role='M0_SGD',lr=lr)
        assert int(model.bn.num_batches_tracked) == 200
        trained = module.model_evidence(model)
        source, _ = save(model,gpu,label='M0_trained',solution_id=module.recovery.FAMILIES[0][1],phase='trained',deadline=100.)
        source['initial_checkpoint_sha256'] = 'same-start'
        before_bytes = (output.root/source['path']).read_bytes()
        def evaluate(model,gpu,indices,*,label,**kwargs):
            events.append((label,module.model_evidence(model)))
            artifact = output.bytes('predictions/'+label+'.npz',b'raw-rows')
            return {'rows':len(indices),'prediction_file':artifact['path'],'prediction_sha256':artifact['sha256']}
        monkeypatch.setattr(module,'evaluate',evaluate)
        variants = tuple(item for item in module.quality_plan() if item.family == 'M0' and item.lr == lr)
        results, pairs = {}, {}
        module.run_bn_pair(source,variants,gpu,tape,deadline=100.,training=proof,results=results,bn_pairs=pairs)
    finally:
        torch.set_num_threads(thread_count)
    assert optimizer_calls == [{'lr':lr,'momentum':.9,'nesterov':True,'weight_decay':.0005}]
    assert proof['updates'] == 200 and proof['training_BN_updates'] is True
    assert initial['parameters'] != trained['parameters']
    assert len(loads) == 2 and loads[0] is not loads[1]
    assert events[0][1] == trained
    assert events[1][1]['parameters'] == trained['parameters']
    assert events[1][1]['buffers'] != trained['buffers']
    assert int(loads[0].bn.num_batches_tracked) == 200 and int(loads[1].bn.num_batches_tracked) == 8
    assert (output.root/source['path']).read_bytes() == before_bytes
    assert module.model_evidence(snapshots[source['path']]) == trained
    assert next(iter(pairs.values()))['parameters_exactly_unchanged'] is True
    ledger = [json.loads(line) for line in output.ledger.read_text().splitlines()]
    assert [event['updates'] for event in ledger] == list(range(1,201))
    assert all(event['tape_sha256'] == module.digest(tape) for event in ledger)


@pytest.mark.parametrize('replay_failure',[None,0,5])
def test_quality_uses_one_common_start_per_family_and_dense_gate_before_compressed_build(tmp_path,monkeypatch,replay_failure):
    torch = pytest.importorskip('torch')
    output = module.Output(tmp_path/'out')
    save, snapshots, loads = checkpoint_stack(output,monkeypatch,torch)
    events, checked = [], []
    gpu = SimpleNamespace(runtime=SimpleNamespace(output=output,check=lambda *args,**kwargs:None),
        data=SimpleNamespace(roles=SimpleNamespace(calibration=tuple(range(1000)),search_validation=tuple(range(5000)))))
    pilot = SimpleNamespace(gpu=gpu)
    def build(pilot,genotype,family,gpu,deadline):
        events.append(('build',family))
        model = tiny_model(torch)
        artifact,_ = save(model,gpu,label=family+'_start',solution_id=next(key for role,key,_ in module.recovery.FAMILIES if role == family),phase='start',deadline=deadline)
        artifact['initial_state'] = module.model_evidence(model)
        return artifact
    def recover(model,gpu,tape,*,role,lr,**kwargs):
        assert int(model.bn.num_batches_tracked) == 0
        assert torch.equal(model.weight.detach(),torch.tensor([.3,-.2]))
        events.append(('SGD',role,lr))
        with torch.no_grad(): model.weight.add_(lr); model.bn.num_batches_tracked.add_(200)
        return {'updates':200,'lr':lr}
    def calibrate(model,*args):
        with torch.no_grad(): model.bn.num_batches_tracked.fill_(8); model.bn.running_mean.add_(.25)
    def evaluate(model,gpu,indices,*,label,**kwargs):
        events.append(('quality',label))
        prediction = output.bytes('predictions/'+label+'.npz',b'predictions')
        return {'rows':len(indices),'prediction_file':prediction['path'],'prediction_sha256':prediction['sha256']}
    def verify(old,new):
        if 'original' not in str(old):
            checked.append(Path(old).name)
            if len(checked)-1 == replay_failure: raise ValueError('Dense replay mismatch: cross_entropy')
        return {'exact':True}
    monkeypatch.setattr(module,'build_start',build)
    monkeypatch.setattr(module,'recover_without_calibration',recover)
    monkeypatch.setattr(module,'calibrate_bn',calibrate)
    monkeypatch.setattr(module,'evaluate',evaluate)
    monkeypatch.setattr(module,'verify_dense_predictions',verify)
    parent = tmp_path/'parent'; parent.mkdir()
    prediction_names = {}
    for item in module.quality_plan()[:6]:
        name = item.diagnostic_label+'.npz'; (parent/name).write_bytes(b'old')
        prediction_names[item.diagnostic_label] = name
    results,starts,trajectories,pairs = {},{},{},{}
    def run():
        module.run_quality(pilot,{name:object() for name,_,_ in module.recovery.FAMILIES},
            tuple((tuple(range(128)),index) for index in range(200)),deadline=100.,
            diagnostic={'root':str(parent),'predictions':prediction_names},baseline_predictions=parent/'original.npz',
            results=results,starts=starts,trajectories=trajectories,bn_pairs=pairs)
    if replay_failure is not None:
        with pytest.raises(ValueError,match='Dense replay'): run()
        assert [event for event in events if event[0] == 'build'] == [('build','M0')]
        assert len(checked) == replay_failure+1
        return
    run()
    assert len(results) == 30 and len(starts) == 5 and len(trajectories) == 10 and len(pairs) == 15
    assert sum(row['updates'] for row in trajectories.values()) == 2000
    assert [event[1] for event in events if event[0] == 'build'] == ['M0','P75','R','E','hybrid']
    assert len([event for event in events[:events.index(('build','P75'))] if event[0] == 'quality']) == 6
    assert len(checked) == 6
    for family,_,_ in module.recovery.FAMILIES:
        rows = [row for key,row in trajectories.items() if key.startswith(family+'_')]
        assert len(rows) == 2
        assert {row['initial_checkpoint_sha256'] for row in rows} == {starts[family]['sha256']}
        assert len({row['initial_parameter_fingerprint'] for row in rows}) == 1
    assert len(loads) == 40  # Two fresh training reloads and six BN branch reloads per family.


@pytest.mark.parametrize('changed',['parameters','outside_buffer'])
def test_extra_BN_parameter_or_unrelated_buffer_mutation_fails_closed(tmp_path,monkeypatch,changed):
    torch = pytest.importorskip('torch')
    model = tiny_model(torch); model.register_buffer('outside',torch.tensor([1.]))
    output = module.Output(tmp_path/'out'); save,_,_ = checkpoint_stack(output,monkeypatch,torch)
    source,_ = save(model,None,label='M0_start',solution_id=module.recovery.FAMILIES[0][1],phase='start',deadline=100.)
    source['initial_checkpoint_sha256'] = source['sha256']
    def calibrate(model,*args):
        with torch.no_grad():
            if changed == 'parameters': model.weight.add_(1.)
            else: model.outside.add_(1.)
    def evaluate(model,gpu,indices,*,label,**kwargs):
        artifact = output.bytes('predictions/'+label+'.npz',b'predictions')
        return {'rows':5000,'prediction_file':artifact['path'],'prediction_sha256':artifact['sha256']}
    monkeypatch.setattr(module,'calibrate_bn',calibrate); monkeypatch.setattr(module,'evaluate',evaluate)
    gpu = SimpleNamespace(runtime=SimpleNamespace(output=output),
        data=SimpleNamespace(roles=SimpleNamespace(search_validation=tuple(range(5000)))))
    with pytest.raises(ValueError,match='parameters|outside BN'):
        module.run_bn_pair(source,module.quality_plan()[:2],gpu,(),deadline=100.,training={},results={},bn_pairs={})


@pytest.mark.parametrize('hidden_BN',[False,True])
def test_public_build_start_rejects_hidden_BN_before_any_training(tmp_path,monkeypatch,hidden_BN):
    torch = pytest.importorskip('torch')
    from fedcore.algorithm.low_rank.cv_execution import CVWorkflowSuccess
    import fedcore.algorithm.low_rank.candidate_storage as storage
    teacher, built = tiny_model(torch),tiny_model(torch)
    if hidden_BN: built.bn.num_batches_tracked.add_(8)
    output = module.Output(tmp_path/'out')
    checkpoint_stack(output,monkeypatch,torch)
    monkeypatch.setattr(storage,'capture_inventory',lambda model:())
    monkeypatch.setattr(storage,'inventory_fingerprint',lambda inventory:'teacher')
    genotype, calls = object(),[]
    def build(value):
        assert value is genotype
        calls.append('public_build')
        return CVWorkflowSuccess(built,SimpleNamespace(fingerprint='workflow'),b'',(),{'storage_bytes':44734408})
    pilot = SimpleNamespace(teacher=teacher,teacher_fingerprint='teacher',build=build)
    gpu = SimpleNamespace(runtime=SimpleNamespace(output=output,check=lambda *args,**kwargs:None))
    if hidden_BN:
        with pytest.raises(ValueError,match='hidden'): module.build_start(pilot,genotype,'M0',gpu,100.)
    else:
        artifact = module.build_start(pilot,genotype,'M0',gpu,100.)
        assert artifact['no_extra_BN_before_training'] is True and artifact['workflow_fingerprint'] == 'workflow'
    assert calls == ['public_build']


@pytest.mark.parametrize('fault',['malformed','duplicate','over_budget','unknown_role','bool_update','wrong_lr'])
def test_ledger_counter_preserves_verified_prefix_and_rejects_invalid_events(tmp_path,fault):
    ledger = tmp_path/'ledger.jsonl'
    role = 'M0_SGD_lr_0.01'
    events = [{'event':'diagnostic_SGD_update','trajectory':role,'updates':index,'lr':.01} for index in range(1,4)]
    extra = dict(events[-1],updates=4)
    if fault == 'duplicate': extra['updates'] = 3
    if fault == 'over_budget': extra['updates'] = 201
    if fault == 'unknown_role': extra['trajectory'] = 'sixth_family'
    if fault == 'bool_update': extra['updates'] = True
    if fault == 'wrong_lr': extra['lr'] = .1
    text = ''.join(json.dumps(event)+'\n' for event in events)
    ledger.write_text(text+('{broken\n' if fault == 'malformed' else json.dumps(extra)+'\n'))
    observed, failure = module.count_training_updates(ledger)
    assert observed == {role:3} and failure


@pytest.mark.parametrize('stop',[False,True,'malformed','budget'])
def test_public_run_wires_fixed_matrix_and_keeps_partial_update_evidence(tmp_path,monkeypatch,stop):
    completion, diagnostic, inventories, old, previous, _ = miniature_parents(tmp_path,monkeypatch)
    admitted = module.inspect_completion(completion,inventories[0])
    replay = module.inspect_diagnostic(diagnostic,inventories[1])
    prepared = {'status':'prepared','source':{},'completion':admitted,'diagnostic':replay}
    monkeypatch.setattr(module,'recovery_preparation',lambda *args:prepared)
    monkeypatch.setattr(module,'inspect_completion',lambda *args:admitted)
    monkeypatch.setattr(module,'inspect_diagnostic',lambda *args:replay)
    monkeypatch.setattr(module,'admit_free_gpu',lambda:{'compute_pids':[]})
    monkeypatch.setattr(module,'verify_loaded_sources',lambda *args:{})
    monkeypatch.setattr(module.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(stdout='numpy==1.26.4\nfedcore==new\n'))
    class Runtime:
        def __init__(self,protocol,output,started):
            self.protocol,self.output,self.deadline = protocol,output,started+1800
            self.environment = old['environment']
            self.torch = SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda:0,max_memory_reserved=lambda:0))
        def check(self,*args,**kwargs): pass
    class Data:
        def __init__(self,*args,**kwargs):
            assert kwargs == {'download':False}
            roles = json.loads((completion/'data_roles.json').read_text())
            self.roles = SimpleNamespace(train=(1,2),sha256='fixed-roles',payload=lambda:roles)
            self.source_hashes = old['data']['source_files']
        def open_test(self): pytest.fail('Official test must stay sealed')
    class Pilot:
        def __init__(self,runtime,data):
            self.runtime,self.gpu = runtime,SimpleNamespace(runtime=runtime)
            self.space = SimpleNamespace(source_storage_bytes=44734408)
            self.teacher_fingerprint = 'teacher'
        def bind_teacher(self,model): self.teacher = model
        def short(self,*args): pytest.fail('Short evaluations calibrate BN and must never run')
        def random_search(self,*args): pytest.fail('New search is forbidden')
        def evolution(self,*args): pytest.fail('New search is forbidden')
    monkeypatch.setitem(sys.modules,'fedcore.experiments.cv_sparse_data',SimpleNamespace(CifarData=Data))
    monkeypatch.setitem(sys.modules,'fedcore.experiments.cv_sparse_pilot',SimpleNamespace(Pilot=Pilot))
    import fedcore.algorithm.low_rank.candidate_storage as storage
    monkeypatch.setattr(storage,'capture_inventory',lambda model:())
    monkeypatch.setattr(storage,'inventory_fingerprint',lambda inventory:'teacher')
    monkeypatch.setattr(module,'Runtime',Runtime)
    monkeypatch.setattr(module,'load_checkpoint',lambda *args:object())
    monkeypatch.setattr(module,'decode_genotypes',lambda *args:{'fixed':'genotypes'})
    tape = tuple((tuple(range(128)),index) for index in range(200))
    tape_sha = module.digest(tape)
    for item in admitted['report']['recovered'].values(): item['recovery_tape_sha256'] = tape_sha
    replay['report']['recovery_tape_sha256'] = tape_sha
    monkeypatch.setattr(module,'recovery_tape',lambda *args:tape)
    calls = []
    def quality(pilot,genotypes,shared_tape,**kwargs):
        assert shared_tape is tape and genotypes == {'fixed':'genotypes'}
        assert pilot.runtime.stage == 'quality' and kwargs['diagnostic'] is replay
        calls.append('quality')
        output = pilot.runtime.output
        prediction = output.bytes('predictions/partial.npz',b'partial')
        record = {**old['baseline'],'validation':{'rows':5000,'prediction_file':prediction['path'],
                                               'prediction_sha256':prediction['sha256']}}
        partial = stop is True or stop == 'malformed'
        count = 7 if partial else 200
        roles = [(f'{family}_SGD_lr_{lr:g}',lr) for family,_,_ in module.recovery.FAMILIES for lr in module.recovery.LEARNING_RATES]
        for trajectory, (role,lr) in enumerate(roles[:1] if partial else roles):
            updates = 199 if stop == 'budget' and trajectory == 9 else count
            for update in range(1,updates+1):
                output.append({'event':'diagnostic_SGD_update','trajectory':role,'updates':update,'lr':lr})
            if not partial: kwargs['trajectories'][role] = {'updates':200}
        for index in range(1 if partial else 30): kwargs['results'][str(index)] = record
        if stop == 'malformed':
            with output.ledger.open('a') as stream: stream.write('{broken\n')
        if partial: raise module.PilotStop('quality','absolute_deadline')
    monkeypatch.setattr(module,'run_quality',quality)
    report = module.run_recovery(tmp_path/'data',tmp_path/'out',tmp_path/'manifest',
        completion,inventories[0],diagnostic,inventories[1])
    partial = stop is True or stop == 'malformed'
    expected_status = 'failed' if stop in ('malformed','budget') else 'recovery_incomplete_quality' if stop is True else 'completed'
    assert calls == ['quality'] and report['total_SGD_updates'] == (7 if partial else 1999 if stop == 'budget' else 2000)
    assert report['status'] == expected_status,report.get('traceback')
    assert report['total_validation_rows'] == (5000 if partial else 150000)
    assert report['completed_trajectory_count'] == (0 if partial else 10)
    if stop in ('malformed','budget'): assert report['ledger_count_failure']
    assert report['official_test_access'] is report['search_performed'] is report['baseline_training_performed'] is False
    assert report['latency_sessions'] == 0 and report['automatic_retry'] is False
    assert (tmp_path/'out/report.json').is_file()
