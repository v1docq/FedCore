"""Fixed workload and process boundary for the approved recovery matrix."""
import json
import os

import pytest

from fedcore.experiments import cv_sparse_recovery_protocol as protocol


def test_recovery_matrix_has_exact_deduplicated_fixed_workload():
    plan=protocol.payload()
    assert [row[0] for row in protocol.FAMILIES]==['M0','P75','R','E','hybrid']
    assert len({row[1] for row in protocol.FAMILIES})==5
    assert plan['quality_variants']==5*3*2==30
    assert plan['total_validation_rows']==30*5000==150000
    assert plan['SGD_trajectories']==5*2==10
    assert plan['total_updates']==10*200==2000
    assert plan['prior_seconds']==3648.862081296742
    assert plan['prior_seconds']+plan['max_seconds']<=plan['total_authorized_seconds']
    assert protocol.WORKER_SECONDS+10==protocol.MAX_SECONDS==1800
    assert plan['official_test_access'] is plan['new_search'] is plan['automatic_retry'] is False
    assert plan['latency_sessions']==0


def test_recovery_cli_forwards_all_parent_arguments_without_changing_old_modes(monkeypatch):
    from fedcore.experiments import cv_sparse_supervision as module
    calls=[]
    monkeypatch.setattr(module,'supervise',lambda *a,**kw:calls.append((a,kw)) or 0)
    assert module.main(['--execute','--data-root','data','--output-root','out','--source-manifest','source',
        '--recovery-completion','completed','--completion-inventory','completed_inventory',
        '--recovery-diagnostic','diagnostic','--diagnostic-inventory','diagnostic_inventory'])==0
    assert calls==[(('data','out','source'),{'continuation_parent':None,'parent_inventory':None,
        'diagnostic_completion':None,'completion_inventory':'completed_inventory',
        'recovery_completion':'completed','recovery_diagnostic':'diagnostic',
        'diagnostic_inventory':'diagnostic_inventory'})]


@pytest.mark.skipif(os.name!='posix',reason='Linux worker isolation boundary')
@pytest.mark.parametrize('extra',[
    {'recovery_completion':'parent'},
    {'recovery_diagnostic':'diagnostic'},
    {'diagnostic_inventory':'inventory'},
    {'recovery_completion':'parent','completion_inventory':'inventory','recovery_diagnostic':'diag'},
    {'recovery_completion':'parent','completion_inventory':'inventory','diagnostic_inventory':'diag_inventory'},
    {'recovery_completion':'parent','completion_inventory':'inventory','recovery_diagnostic':'diag','diagnostic_inventory':'diag_inventory','diagnostic_completion':'other'},
    {'recovery_completion':'parent','completion_inventory':'inventory','recovery_diagnostic':'diag','diagnostic_inventory':'diag_inventory','continuation_parent':'older','parent_inventory':'older_inventory'},
])
def test_invalid_recovery_modes_cannot_create_a_lock_or_child(tmp_path,monkeypatch,extra):
    from fedcore.experiments import cv_sparse_supervision as module
    def forbidden(*a,**kw):pytest.fail('Invalid recovery profile launched a child')
    monkeypatch.setattr(module.subprocess,'Popen',forbidden)
    with pytest.raises(ValueError):module.supervise(tmp_path/'data',tmp_path/'out',tmp_path/'manifest',**extra)
    assert not (tmp_path/'.fedcore-sparse-cv.lock').exists()
    assert not (tmp_path/'out').exists()


@pytest.mark.skipif(os.name!='posix',reason='Linux worker isolation boundary')
def test_recovery_profile_launches_exactly_one_pinned_worker_with_1790_second_limit(tmp_path,monkeypatch):
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
    manifest=tmp_path/'manifest';manifest.write_text('{}')
    assert module.supervise(tmp_path/'data',tmp_path/'out',manifest,
        recovery_completion=tmp_path/'completion',completion_inventory=tmp_path/'ci',
        recovery_diagnostic=tmp_path/'diagnostic',diagnostic_inventory=tmp_path/'di')==0
    state=json.loads((tmp_path/'out-supervision.json').read_text())
    assert len(commands)==1
    assert commands[0][2]=='fedcore.experiments.cv_sparse_recovery'
    for flag,path in (('--completion-root','completion'),('--completion-inventory','ci'),
                      ('--diagnostic-root','diagnostic'),('--diagnostic-inventory','di')):
        assert commands[0][commands[0].index(flag)+1]==str((tmp_path/path).resolve())
    assert state['max_slot_seconds']==1800 and state['worker_limit_seconds']==1790
    assert state['recovery_protocol_sha256']==protocol.sha256()
