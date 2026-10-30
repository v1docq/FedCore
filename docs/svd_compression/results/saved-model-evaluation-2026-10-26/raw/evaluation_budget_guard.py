"""Enforce the authorized remaining budget without altering the frozen package."""
import argparse
import ctypes
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import psutil

# Adopt only descendants of this invocation if an intermediate supervisor exits.
# Detached sessions alone do not provide a whole-tree lifetime boundary.
libc=ctypes.CDLL(None,use_errno=True)
if libc.prctl(36,1,0,0,0)!=0:  # PR_SET_CHILD_SUBREAPER
    raise OSError(ctypes.get_errno(),'Unable to become a child subreaper')

parser=argparse.ArgumentParser()
parser.add_argument('packet',type=Path)
args=parser.parse_args()
packet=json.loads(args.packet.read_text())
root=args.packet.parent
output=Path(packet['output_root'])
guard_path=output.parent/(output.name+'-budget-guard.json')
inner_state=output.parent/(output.name+'-supervision.json')
assert not guard_path.exists() and not output.exists()
began=packet['launch_monotonic']
assert 0<=time.perf_counter()-began<60,'Stale launch packet; do not start'
limit=packet['repeat_slot_seconds']
assert limit==1800 and packet['prior_charged_seconds']==3827.6313528195024 and packet['prior_charged_seconds']+limit<=14400
reserve=10
child=None
worker_identity=None
owned_groups={}
cleanup_lock=threading.Lock()
deadline_fired=threading.Event()
state={'schema':'fedcore-repeat-budget-guard-v1','status':'starting',
       'repeat_slot_seconds':limit,'termination_reserve_seconds':reserve,
       'prior_charged_seconds':packet['prior_charged_seconds'],
       'total_authorized_seconds':14400,'automatic_retry':False,
       'subreaper_enabled':True,'utc_started':datetime.datetime.now(datetime.timezone.utc).isoformat()}

def atomic():
    temporary=guard_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,guard_path)

def capture_worker():
    global worker_identity
    # Membership in this subreaper's descendant tree proves ownership even if
    # the inner supervisor died before publishing its worker PID.
    descendants=psutil.Process().children(recursive=True)
    for process in descendants:
        try:
            if os.getpgid(process.pid)==process.pid:
                owned_groups[process.pid]=process.create_time()
        except (ProcessLookupError,psutil.NoSuchProcess):
            pass
    if worker_identity is not None or child is None:return
    try:
        pid=json.loads(inner_state.read_text()).get('owned_pid')
        if pid is None:return
        process=psutil.Process(pid)
        if process.create_time()!=owned_groups.get(pid) or os.getpgid(pid)!=pid:
            raise RuntimeError('Worker ownership mismatch')
        worker_identity=(pid,process.create_time())
        state['owned_worker_pid']=pid
    except (FileNotFoundError,json.JSONDecodeError,psutil.NoSuchProcess):
        pass

def stop_owned():
    with cleanup_lock:
        def kill_groups(exclude_supervisor=False):
            capture_worker()
            for pid,birth in list(owned_groups.items()):
                if exclude_supervisor and child is not None and pid==child.pid:continue
                try:
                    process=psutil.Process(pid)
                    # Ownership was already established. Reparenting cannot
                    # revoke it; PID birth and the private PGID must still match.
                    if process.create_time()==birth and os.getpgid(pid)==pid:
                        os.killpg(pid,signal.SIGKILL)
                except (ProcessLookupError,psutil.NoSuchProcess):
                    pass
        kill_groups(exclude_supervisor=True)
        if child is not None and child.poll() is None:
            try:child.wait(timeout=2)
            except subprocess.TimeoutExpired:pass
        kill_groups()
        if child is not None:
            try:child.wait(timeout=1)
            except subprocess.TimeoutExpired:pass
        # Catch descendants adopted after an abrupt supervisor exit.
        kill_groups()
        for pid in list(owned_groups):
            if child is not None and pid==child.pid:continue
            try:os.waitpid(pid,os.WNOHANG)
            except ChildProcessError:pass

def timer_stop():
    if child is not None and child.poll() is None:
        deadline_fired.set()
        stop_owned()

atomic()
timer=None
try:
    with (output.parent/(output.name+'-budget-guard-console.log')).open('x') as console:
        child=subprocess.Popen(packet['command'],cwd=root,stdin=subprocess.DEVNULL,
            stdout=console,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(status='running',supervisor_pid=child.pid)
        timer=threading.Timer(max(0.,began+limit-reserve-time.perf_counter()),timer_stop)
        timer.daemon=True
        timer.start()
        while child.poll() is None:
            capture_worker()
            state.update(elapsed_seconds=time.perf_counter()-began)
            atomic()
            time.sleep(.5)
        state.update(status='stopped' if deadline_fired.is_set() else 'finished',
                     reason='remaining_budget_deadline' if deadline_fired.is_set() else 'supervisor_exit',
                     returncode=child.returncode)
finally:
    if timer is not None:timer.cancel()
    stop_owned()
    elapsed=time.perf_counter()-began
    state.update(elapsed_seconds=elapsed,total_charged_seconds=packet['prior_charged_seconds']+elapsed,
                 within_approved_total=packet['prior_charged_seconds']+elapsed<=14400)
    atomic()
print(json.dumps(state),flush=True)
