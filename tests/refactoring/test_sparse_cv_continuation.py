"""Resume only a pinned finished search; never silently rerun training/search."""
import ast
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from fedcore.experiments import cv_sparse_continuation as module


RAW = Path(__file__).resolve().parents[2]/'docs/svd_compression/results/a100-pilot-2-2026-10-22/raw'


def parent_report():
    return json.loads((RAW/'report.json').read_text(encoding='utf-8'))


def test_actual_source_selects_exactly_five_and_deduplicates_random_svd():
    value = parent_report(); before = copy.deepcopy(value)
    plan = module.completion_plan(value)
    assert len(plan.selections) == 5 and dict(plan.finalists) == module.FINALISTS
    assert dict(plan.selections)[module.FINALISTS['random']] == ('fixed_svd', 'random_finalist')
    assert sum(len(roles) for _,roles in plan.selections) == 6
    assert value == before
    assert json.loads(json.dumps(asdict(plan)))['finalists'][0][0] == 'random'


@settings(max_examples=12, deadline=None)
@given(st.permutations(tuple(range(12))))
def test_winner_selection_does_not_depend_on_stored_record_order(order):
    value = parent_report(); expected = module.completion_plan(value)
    for run in ('random', 'evolution'):
        rows = value['all_short_attempts'][run]
        value['all_short_attempts'][run] = [rows[i] for i in order]
    assert module.completion_plan(value) == expected


@pytest.mark.parametrize('mutation', ['status', 'missing', 'late', 'fake_genotype', 'wrong_ce', 'recovery', 'common_late'])
def test_source_change_is_rejected_instead_of_requesting_more_search(mutation):
    value = parent_report(); rows = value['all_short_attempts']['evolution']
    if mutation == 'status': value['status'] = 'completed'
    if mutation == 'missing': rows.pop()
    if mutation == 'late': rows[0]['completed_within_deadline'] = False
    if mutation == 'fake_genotype': rows[0]['genotype']['genes'][0]['choice']['rank'] = 17
    if mutation == 'wrong_ce': rows[0]['cross_entropy'] = -100
    if mutation == 'recovery': value['recovered'] = {'already':'run'}
    if mutation == 'common_late': rows[0]['common_search_completed_at'] = rows[0]['deadline']+1
    with pytest.raises(ValueError): module.completion_plan(value)


def miniature_parent(tmp_path, monkeypatch):
    root = tmp_path/'parent'; root.mkdir()
    (root/'checkpoints').mkdir()
    (root/'checkpoints/dense_original.pt').write_bytes(b'unchanged-model-bytes')
    model_sha = module.sha256_file(root/'checkpoints/dense_original.pt')
    value = parent_report(); value['baseline']['sha256'] = model_sha
    (root/'report.json').write_text(json.dumps(value), encoding='utf-8')
    (root/'source_manifest.json').write_bytes((RAW/'source_manifest.json').read_bytes())
    files = {p.relative_to(root).as_posix():{'bytes':p.stat().st_size,'sha256':module.sha256_file(p)}
             for p in root.rglob('*') if p.is_file()}
    inventory = tmp_path/'inventory.json'
    inventory.write_text(json.dumps({'files':files}), encoding='utf-8')
    monkeypatch.setattr(module, 'PARENT_INVENTORY_SHA', module.sha256_file(inventory))
    monkeypatch.setattr(module, 'PARENT_REPORT_SHA', module.sha256_file(root/'report.json'))
    monkeypatch.setattr(module, 'BASELINE_SHA', model_sha)
    return root, inventory


def test_parent_integrity_reads_are_repeatable_and_do_not_modify_old_files(tmp_path, monkeypatch):
    root, inventory = miniature_parent(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert module.inspect_parent(root, inventory) == module.inspect_parent(root, inventory)
    assert {p:p.read_bytes() for p in before} == before
    (root/'checkpoints/dense_original.pt').write_bytes(b'changed-model-bytes!!')
    with pytest.raises(ValueError, match='integrity'): module.inspect_parent(root, inventory)


@pytest.mark.parametrize('name', ['../escape', '/absolute', '', 'missing', 'folder/../../escape'])
def test_parent_paths_fail_closed(tmp_path, name):
    with pytest.raises(ValueError): module.confined_file(tmp_path.resolve(), name)


def test_preflight_never_opens_cuda_creates_output_or_imports_ml(tmp_path):
    code = '''import sys
from fedcore.experiments.cv_sparse_continuation import main
try: main(['--data-root',sys.argv[1], '--output-root',sys.argv[2], '--source-manifest',sys.argv[3],
           '--parent-root',sys.argv[4], '--parent-inventory',sys.argv[5]])
except (FileNotFoundError,ValueError): pass
assert not any(name=='torch' or name.startswith(('torch.','torchvision','fedcore.algorithm')) for name in sys.modules)
from pathlib import Path
assert not Path(sys.argv[2]).exists()
'''
    subprocess.run([sys.executable,'-c',code,*[str(tmp_path/name) for name in ('data','out','source','parent','inventory')]],check=True)


def test_prediction_verification_rejects_wrong_labels_predictions_and_loss(tmp_path):
    old = tmp_path/'old.npz'; new = tmp_path/'new.npz'
    values = {'row_ids':np.array([1,4]), 'targets':np.array([2,3]), 'predictions':np.array([2,3]), 'losses':np.array([.1,.2])}
    np.savez(old, **values); np.savez(new, **values)
    module.verify_baseline_predictions(old,new)
    for key in values:
        changed = {k:v.copy() for k,v in values.items()}; changed[key][0] += 1
        np.savez(new, **changed)
        with pytest.raises(AssertionError): module.verify_baseline_predictions(old,new)


def test_scientific_freeze_may_only_change_the_declared_fedcore_wheel():
    old = 'numpy==1.26.4\nscipy==1.15.3\nfedcore==0.0.5.4\n'
    replacement = 'scipy==1.15.3\nfedcore @ file:///new/fedcore.whl\nnumpy==1.26.4\n'
    assert module.frozen_dependencies(old) == module.frozen_dependencies(replacement)
    assert module.frozen_dependencies(old) != module.frozen_dependencies(replacement.replace('1.26.4','2.0.0'))


def test_completion_stage_deadline_is_narrower_than_frozen_pilot_windows():
    owner = SimpleNamespace(deadline=100., output=SimpleNamespace(append=lambda row:None), check=lambda:None)
    with module.completion_phase(owner, 'recovery', 600) as deadline:
        assert deadline == 100. and owner.stage == 'recovery'
    assert module.MAX_SECONDS == 1800 and module.PRIOR_SECONDS == 3339.3260714150965


def test_completion_worker_has_no_baseline_training_or_search_calls():
    tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
    calls = [node.func for node in ast.walk(tree) if isinstance(node,ast.Call)]
    names = [getattr(node,'id',getattr(node,'attr','')) for node in calls]
    assert not {'train_baseline','random_search','evolution','short','run_cv_grammar_search'} & set(names)
    assert {'recover','finalize_pilot','verify_baseline_predictions'} <= set(names)


@pytest.mark.parametrize('changed_environment', [False, True])
def test_completion_entrypoint_copies_evidence_and_wires_five_fresh_recoveries(tmp_path, monkeypatch, changed_environment):
    from fedcore.experiments import cv_sparse_pilot as pilot, cv_sparse_data as data_module
    from fedcore.algorithm.low_rank.cv_grammar import CVGrammarSpace, CVBlockDomain
    from fedcore.algorithm.low_rank.compression_graph import ModelVersion, CalibrationRef
    from fedcore.algorithm.low_rank.cv_execution import CVWorkflowSuccess
    from fedcore.algorithm.low_rank import candidate_storage
    from fedcore.tools.registry.checkpoint_manager import CheckpointManager
    root, inventory = miniature_parent(tmp_path, monkeypatch)
    old = json.loads((root/'report.json').read_text())
    roles = json.loads((RAW/'data_roles.json').read_text())
    data = SimpleNamespace(roles=SimpleNamespace(train=tuple(roles['train']), search_validation=tuple(roles['search_validation']),
        sha256=old['data']['roles_sha256'], payload=lambda:roles), source_hashes=old['data']['source_files'])
    environment = json.loads((RAW/'environment.json').read_text())
    (root/'environment.json').write_text(json.dumps(environment))
    freeze = (RAW/'environment_freeze.txt').read_text()
    (root/'environment_freeze.txt').write_text(freeze)
    (root/'data_roles.json').write_text(json.dumps(roles))
    prediction = root/old['baseline']['validation']['prediction_file']; prediction.parent.mkdir()
    prediction.write_bytes((RAW/old['baseline']['validation']['prediction_file']).read_bytes())
    original = {'verified_files':{}, 'report':old}
    monkeypatch.setattr(module,'continuation_preparation',lambda *args:{'status':'prepared','source':{},'parent':original})
    monkeypatch.setattr(module,'inspect_parent',lambda *args:original)
    monkeypatch.setattr(module,'admit_free_gpu',lambda:{})
    monkeypatch.setattr(module,'verify_loaded_sources',lambda *args:{})
    monkeypatch.setattr(module.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(stdout=freeze.replace('numpy==1.26.4','numpy==2.0.0') if changed_environment else freeze))
    monkeypatch.setattr(data_module,'CifarData',lambda *args,**kwargs:data)
    events=[]
    class Model:
        def eval(self): return self
        def to(self,*args): return self
        def cpu(self): return self
    class Runtime:
        def __init__(self,protocol,output,started):
            self.protocol,self.output,self.deadline=protocol,output,started+1800
            self.environment=environment
            self.torch=SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda:0,max_memory_reserved=lambda:0))
        def check(self,*args,**kwargs): pass
    space=CVGrammarSpace(ModelVersion('test','a'*64),CalibrationRef('b'*64,(1000,3,32,32),'float32'),
        tuple(CVBlockDomain(path,options) for path,options in pilot.cv_menu().items()),'c'*64,44734408)
    class Owner:
        def __init__(self,runtime,data):
            self.runtime,self.output,self.protocol,self.data=runtime,runtime.output,runtime.protocol,data
            self.prefix=pilot.fixed_prefix(pilot.cv_menu()); self.genotypes={}; self.gpu=object()
            self.space=space; self.byte_ceiling=35787526; self.teacher=Model(); self.teacher_fingerprint='teacher'
        def bind_teacher(self,model): self.teacher=model
        def build(self,genotype):
            events.append('build')
            return CVWorkflowSuccess(Model(),None,b'',(),{})
    monkeypatch.setattr(module,'Runtime',Runtime); monkeypatch.setattr(pilot,'Pilot',Owner)
    monkeypatch.setattr(CheckpointManager,'deserialize_from_bytes',lambda *args,**kwargs:Model())
    def evaluate(model,gpu,indices,**kwargs):
        events.append('recheck')
        target=tmp_path/'out/predictions/recheck.npz';target.write_bytes(prediction.read_bytes())
        return {**old['baseline']['validation'],'prediction_file':'predictions/recheck.npz'}
    monkeypatch.setattr(module,'evaluate',evaluate)
    monkeypatch.setattr(module,'torchscript_artifact',lambda *args:(object(),{}))
    monkeypatch.setattr(module,'independent_torchscript',lambda *args:object())
    monkeypatch.setattr(module,'paired_latency',lambda *args,**kwargs:{'accepted':False})
    def recover(model,gpu,tape,**kwargs):
        assert len(tape)==200 and all(len(ids)==128 and set(ids)<=set(data.roles.train) for ids,seed in tape)
        events.append('recover');return {}
    monkeypatch.setattr(module,'recover',recover)
    monkeypatch.setattr(module,'checkpoint',lambda model,gpu,**kwargs:({'solution_id':kwargs['solution_id'],
        'phase':kwargs['phase'],'artifact_id':kwargs['solution_id']},Model()))
    def final(owner,baseline,recovered,aa,report,deadline):
        assert aa['accepted'] is False and len(recovered)==5
        assert (owner.output.root/baseline['path']).read_bytes()==b'unchanged-model-bytes'
        assert (owner.output.root/baseline['validation']['prediction_file']).read_bytes()==prediction.read_bytes()
        assert all(a['recovery_updates']==200 for a in recovered.values())
        assert len({a['recovery_tape_sha256'] for a in recovered.values()})==1
        events.append('final')
    monkeypatch.setattr(pilot,'finalize_pilot',final)
    monkeypatch.setattr(candidate_storage,'capture_inventory',lambda *args:())
    monkeypatch.setattr(candidate_storage,'inventory_fingerprint',lambda *args:'teacher')
    result=module.run_completion(tmp_path/'data',tmp_path/'out',tmp_path/'manifest',root,inventory)
    if changed_environment:
        assert result['status']=='failed' and 'dependency versions' in result['failure'] and not events
    else:
        assert result['status']=='completed',result.get('traceback')
        assert events==['recheck']+['build','recover']*5+['final']
        assert result['baseline_training_performed'] is result['search_performed'] is False
