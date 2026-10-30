"""Budget and process boundary of evaluation-only frozen artifacts."""
import json
import os

import pytest

from fedcore.experiments import cv_sparse_evaluation_protocol as protocol


def test_saved_evaluation_has_five_frozen_states_and_no_training_budget():
    plan=protocol.payload()
    assert [row[0] for row in protocol.ARTIFACTS]==['M0','P75','R','E','hybrid']
    assert len({row[1] for row in protocol.ARTIFACTS})==5
    assert plan['training_updates']==0 and plan['new_search'] is plan['additional_BN_calibration'] is False
    assert plan['eager_validation_rows']==plan['TS_validation_rows']==25000
    assert plan['TS_official_test_rows']==50000
    assert plan['AA_sessions']==7 and plan['AB_sessions_max']==4
    assert plan['latency_pairs_max']==(7+4)*2*20==440
    assert protocol.WORKER_SECONDS+10==protocol.MAX_SECONDS==1800
    assert protocol.PRIOR_SECONDS==3827.6313528195024
    assert protocol.PRIOR_SECONDS+1800<=14400
    assert plan['official_test_previous_use_disclosed'] is True
    assert plan['automatic_retry'] is False


def test_saved_evaluation_cli_forwards_both_parent_arguments(monkeypatch):
    from fedcore.experiments import cv_sparse_supervision as module
    calls=[]
    monkeypatch.setattr(module,'supervise',lambda *a,**kw:calls.append((a,kw)) or 0)
    assert module.main(['--execute','--data-root','data','--output-root','out','--source-manifest','source',
        '--evaluation-parent','recovery','--evaluation-inventory','inventory'])==0
    assert calls==[(('data','out','source'),{'continuation_parent':None,'parent_inventory':None,
        'diagnostic_completion':None,'completion_inventory':None,
        'evaluation_parent':'recovery','evaluation_inventory':'inventory'})]


@pytest.mark.skipif(os.name!='posix',reason='Linux process isolation boundary')
@pytest.mark.parametrize('extra',[
    {'evaluation_parent':'parent'}, {'evaluation_inventory':'inventory'},
    {'evaluation_parent':'parent','evaluation_inventory':'inventory','continuation_parent':'other','parent_inventory':'other_inventory'},
    {'evaluation_parent':'parent','evaluation_inventory':'inventory','diagnostic_completion':'other','completion_inventory':'other_inventory'},
    {'evaluation_parent':'parent','evaluation_inventory':'inventory','recovery_completion':'other','completion_inventory':'other_inventory','recovery_diagnostic':'diag','diagnostic_inventory':'diag_inventory'},
])
def test_partial_or_mixed_evaluation_profile_never_creates_lock_or_child(tmp_path,monkeypatch,extra):
    from fedcore.experiments import cv_sparse_supervision as module
    def forbidden(*a,**kw):pytest.fail('Invalid evaluation profile started a worker')
    monkeypatch.setattr(module.subprocess,'Popen',forbidden)
    with pytest.raises(ValueError):module.supervise(tmp_path/'data',tmp_path/'out',tmp_path/'manifest',**extra)
    assert not (tmp_path/'.fedcore-sparse-cv.lock').exists() and not (tmp_path/'out').exists()


@pytest.mark.skipif(os.name!='posix',reason='Linux process isolation boundary')
def test_saved_evaluation_launches_one_worker_with_1790_seconds(tmp_path,monkeypatch):
    import psutil
    from fedcore.experiments import cv_sparse_supervision as module
    commands=[]
    class Finished:
        pid=999999999
        returncode=0
        def poll(self):return 0
    def launch(command,**kw):commands.append(command);return Finished()
    def vanished(pid):raise psutil.NoSuchProcess(pid)
    monkeypatch.setattr(module,'gpu_processes',lambda:[])
    monkeypatch.setattr(module.subprocess,'Popen',launch)
    monkeypatch.setattr(module,'terminate_owned_group',lambda *a:None)
    monkeypatch.setattr(psutil,'Process',vanished)
    manifest=tmp_path/'source';manifest.write_text('{}')
    assert module.supervise(tmp_path/'data',tmp_path/'out',manifest,
        evaluation_parent=tmp_path/'recovery',evaluation_inventory=tmp_path/'inventory')==0
    state=json.loads((tmp_path/'out-supervision.json').read_text())
    assert len(commands)==1 and commands[0][2]=='fedcore.experiments.cv_sparse_evaluation'
    assert commands[0][commands[0].index('--recovery-root')+1]==str((tmp_path/'recovery').resolve())
    assert commands[0][commands[0].index('--recovery-inventory')+1]==str((tmp_path/'inventory').resolve())
    assert state['max_slot_seconds']==1800 and state['worker_limit_seconds']==1790
    assert state['evaluation_protocol_sha256']==protocol.sha256()
