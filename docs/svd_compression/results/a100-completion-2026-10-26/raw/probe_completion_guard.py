"""Exercise the guard's actual cleanup functions on owned sleeping CPU children."""
import ast
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import psutil

assert ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)==0
tree=ast.parse(Path(__file__).with_name('completion_budget_guard.py').read_text())
functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('capture_worker','stop_owned')]
assert len(functions)==2
for mode in ('supervisor_alive','supervisor_exited_before_capture'):
    with tempfile.TemporaryDirectory(prefix='fedcore-guard-probe-') as folder:
        inner_state=Path(folder)/'state.json'
        script="import subprocess,sys,time,os; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True); print(p.pid,flush=True); "
        script += 'time.sleep(30)' if mode=='supervisor_alive' else 'os._exit(0)'
        child=subprocess.Popen([sys.executable,'-c',script],stdout=subprocess.PIPE,text=True,start_new_session=True)
        pid=int(child.stdout.readline().strip())
        if mode!='supervisor_alive':child.wait(timeout=3)
        inner_state.write_text(json.dumps({'owned_pid':pid}))
        ns=dict(globals(),child=child,inner_state=inner_state,worker_identity=None,
                owned_groups={},cleanup_lock=threading.Lock(),state={})
        exec(compile(ast.Module(body=functions,type_ignores=[]),'<actual guard functions>','exec'),ns)
        began=time.perf_counter()
        ns['stop_owned']()
        for _ in range(40):
            try:
                process=psutil.Process(pid)
                if process.status()==psutil.STATUS_ZOMBIE:break
            except psutil.NoSuchProcess:break
            time.sleep(.05)
        else:raise AssertionError('Owned detached worker survived')
        try:os.waitpid(pid,os.WNOHANG)
        except ChildProcessError:pass
        assert child.poll() is not None
        print(json.dumps({'case':mode,'owned_worker_stopped':True,'elapsed_seconds':time.perf_counter()-began}),flush=True)
print(json.dumps({'cpu_only':True,'numerical_experiments':0,'guard_cases_passed':2}))
