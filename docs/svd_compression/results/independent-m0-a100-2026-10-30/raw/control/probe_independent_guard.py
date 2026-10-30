"""CPU-only lifetime cases for the exact external guard implementation."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from independent_budget_guard import OwnedTree, classify_gpu_rows
import psutil

assert ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)==0
for mode in ('controller_alive','controller_exited'):
    script="import subprocess,sys,time,os;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True);print(p.pid,flush=True);"
    script+='time.sleep(30)' if mode=='controller_alive' else 'os._exit(0)'
    child=subprocess.Popen([sys.executable,'-c',script],stdout=subprocess.PIPE,text=True,start_new_session=True)
    pid=int(child.stdout.readline())
    if mode=='controller_exited':child.wait(timeout=3)
    owned=OwnedTree();owned.capture();assert owned.owns(pid)
    assert classify_gpu_rows([{'pid':pid,'used_bytes':1}],owned) is None
    assert classify_gpu_rows([{'pid':os.getpid(),'used_bytes':1}],owned)=='foreign_gpu_process'
    began=time.perf_counter();owned.stop(child)
    for _ in range(60):
        try:
            process=psutil.Process(pid)
            if process.status()==psutil.STATUS_ZOMBIE:break
        except psutil.NoSuchProcess:break
        time.sleep(.05)
    else:raise AssertionError('Owned detached worker survived')
    assert classify_gpu_rows([{'pid':pid,'used_bytes':1}],owned) is None
    print(json.dumps({'case':mode,'detached_descendant_stopped':True,'elapsed_seconds':time.perf_counter()-began}))
print(json.dumps({'cpu_only':True,'numerical_experiments':0,'guard_lifetime_cases':2,'telemetry_race_cases':6}))
