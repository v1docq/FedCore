"""The process terminal state has priority over disappearing telemetry."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from fedcore.experiments.cv_sparse_supervision import (
    MonitorAction, ProcessObservation, monitor_action,
)


def decide(**values):
    return monitor_action(ProcessObservation(**values), max_seconds=100., max_rss_bytes=1000)


def test_successful_exit_is_not_reclassified_as_unknown_rss_or_foreign_gpu():
    assert decide(returncode=0, elapsed=5., rss_bytes=None, gpu_process_count=None,
                  telemetry_failures=3) is MonitorAction.FINISHED
    assert decide(returncode=1, elapsed=5., rss_bytes=None, gpu_process_count=2) is MonitorAction.FINISHED


def test_only_running_owned_worker_can_be_stopped_by_resource_observations():
    assert decide(returncode=None, elapsed=5., rss_bytes=500, gpu_process_count=1) is MonitorAction.WAIT
    assert decide(returncode=None, elapsed=100., rss_bytes=500, gpu_process_count=1) is MonitorAction.DEADLINE
    assert decide(returncode=None, elapsed=5., rss_bytes=1001, gpu_process_count=1) is MonitorAction.MEMORY
    assert decide(returncode=None, elapsed=5., rss_bytes=500, gpu_process_count=2) is MonitorAction.CONTENTION
    assert decide(returncode=None, elapsed=5., rss_bytes=None, gpu_process_count=None,
                  telemetry_failures=3) is MonitorAction.TELEMETRY


@pytest.mark.skipif(os.name != 'posix', reason='The real supervisor uses Linux process groups')
def test_fast_child_exit_before_psutil_attach_retains_success(monkeypatch, tmp_path):
    import psutil
    import fedcore.experiments.cv_sparse_supervision as module

    class FinishedChild:
        pid = 999999999
        returncode = 0

        def poll(self):
            return 0

    def vanished(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(module, 'gpu_processes', lambda: [])
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **kw: FinishedChild())
    monkeypatch.setattr(psutil, 'Process', vanished)
    manifest = tmp_path / 'source.json'
    manifest.write_text('{}', encoding='utf-8')
    assert module.supervise(tmp_path / 'data', tmp_path / 'run', manifest) == 0
    state = json.loads((tmp_path / 'run-supervision.json').read_text(encoding='utf-8'))
    assert state['status'] == 'finished' and state['reason'] == 'child_exit'
    assert state['returncode'] == 0 and state['rss_bytes'] is None


@pytest.mark.skipif(os.name != 'posix', reason='Supervisor process groups require Linux')
def test_diagnostic_worker_dispatch_and_limit_are_bound_before_one_Popen(monkeypatch, tmp_path):
    import psutil
    import fedcore.experiments.cv_sparse_supervision as module
    commands=[]
    class FinishedChild:
        pid=999999999
        returncode=0
        def poll(self):return 0
    def launch(command, **kwargs):commands.append(command);return FinishedChild()
    def vanished(pid):raise psutil.NoSuchProcess(pid)
    monkeypatch.setattr(module,'gpu_processes',lambda:[])
    monkeypatch.setattr(module.subprocess,'Popen',launch)
    monkeypatch.setattr(module,'terminate_owned_group',lambda *args:None)
    monkeypatch.setattr(psutil,'Process',vanished)
    manifest=tmp_path/'source.json';manifest.write_text('{}')
    assert module.supervise(tmp_path/'data',tmp_path/'run',manifest,
        diagnostic_completion=tmp_path/'completed',completion_inventory=tmp_path/'inventory')==0
    state=json.loads((tmp_path/'run-supervision.json').read_text())
    assert len(commands)==1 and commands[0][2]=='fedcore.experiments.cv_sparse_diagnostics'
    assert commands[0][commands[0].index('--completion-root')+1]==str((tmp_path/'completed').resolve())
    assert commands[0][commands[0].index('--completion-inventory')+1]==str((tmp_path/'inventory').resolve())
    assert state['max_slot_seconds']==1800 and state['worker_limit_seconds']==1790
    assert len(state['diagnostic_protocol_sha256'])==64


@pytest.mark.skipif(os.name != 'posix', reason='Supervisor process groups require Linux')
@pytest.mark.parametrize('extra',[
    {'diagnostic_completion':'completed'},
    {'completion_inventory':'inventory'},
    {'diagnostic_completion':'completed','completion_inventory':'inventory','continuation_parent':'parent','parent_inventory':'old_inventory'},
])
def test_invalid_diagnostic_supervision_never_creates_a_lock_or_worker(tmp_path,monkeypatch,extra):
    import fedcore.experiments.cv_sparse_supervision as module
    def forbidden(*args,**kwargs):pytest.fail('Invalid profile launched a numerical worker')
    monkeypatch.setattr(module.subprocess,'Popen',forbidden)
    with pytest.raises(ValueError):module.supervise(tmp_path/'data',tmp_path/'output',tmp_path/'source',**extra)
    assert not (tmp_path/'.fedcore-sparse-cv.lock').exists() and not (tmp_path/'output').exists()


def test_diagnostic_CLI_dispatch_preserves_all_arguments(monkeypatch):
    import fedcore.experiments.cv_sparse_supervision as module
    calls=[]
    monkeypatch.setattr(module,'supervise',lambda *a,**kw:calls.append((a,kw)) or 0)
    assert module.main(['--execute','--data-root','data','--output-root','out','--source-manifest','source',
                        '--diagnostic-completion','completed','--completion-inventory','inventory'])==0
    assert calls==[(('data','out','source'),{'continuation_parent':None,'parent_inventory':None,
                     'diagnostic_completion':'completed','completion_inventory':'inventory'})]


def test_deadline_callback_after_success_does_not_turn_it_into_a_timeout(monkeypatch):
    import fedcore.experiments.cv_sparse_supervision as module
    completed = type('Completed', (), {'poll': lambda self: 0})()
    active = type('Active', (), {'poll': lambda self: None})()
    cleanup = []
    monkeypatch.setattr(module, 'terminate_owned_group', cleanup.append)
    flag = threading.Event()
    module.expire_owned_worker(completed, 71, flag)
    assert not flag.is_set() and cleanup == [71]
    module.expire_owned_worker(active, 72, flag)
    assert flag.is_set() and cleanup == [71, 72]


@pytest.mark.skipif(os.name != 'posix', reason='Actual descendant cleanup requires Linux process groups')
def test_group_cleanup_terminates_a_descendant_after_its_leader_has_exited():
    from fedcore.experiments.cv_sparse_supervision import terminate_owned_group
    code = ('import subprocess,sys; '
            'child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]); '
            'print(child.pid,flush=True)')
    leader = subprocess.Popen([sys.executable, '-c', code], start_new_session=True,
                              stdout=subprocess.PIPE, text=True)
    try:
        descendant = int(leader.stdout.readline().strip())
        assert leader.wait(timeout=5) == 0
        terminate_owned_group(leader.pid)
        for _ in range(100):
            path = Path('/proc') / str(descendant) / 'stat'
            if not path.exists():
                break
            try:
                if path.read_text().rsplit(') ', 1)[1].split()[0] == 'Z':
                    break
            except FileNotFoundError:
                break
            time.sleep(.01)
        else:
            pytest.fail('Owned descendant survived group cleanup after leader exit')
    finally:
        terminate_owned_group(leader.pid)
        leader.stdout.close()
