"""Stop only this series after an observed multi-process GPU violation."""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import psutil

root=Path('/root/fedcore_independent_cv_20261030')
receipt=json.loads((root/'launch-receipt.json').read_text())
state=json.loads((root/'results/series-1-budget-guard.json').read_text())
guard=psutil.Process(receipt['guard_pid'])
assert str(root/'independent_budget_guard.py') in guard.cmdline()
assert state['status']=='running'
observed=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
    '--format=csv,noheader,nounits'],text=True,timeout=5).strip().splitlines()
assert len(observed)>1,'The manually observed violation is no longer present; do not improvise a stop'
descendants=guard.children(recursive=True)
record={'schema':'fedcore-independent-cv-manual-stop-v1','reason':'second_gpu_process_and_unresolved_host_pid_namespace',
    'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu_rows':observed,
    'owned_descendants':[{'pid':p.pid,'birth':p.create_time(),'cmdline':p.cmdline()} for p in descendants]}
path=root/'logs/manual-stop.json'
with path.open('x') as stream:json.dump(record,stream,indent=2);stream.flush();os.fsync(stream.fileno())
pipeline=psutil.Process(state['controller_pid'])
assert pipeline.parent().pid==guard.pid
assert str(root/'run_independent_pipeline.py') in pipeline.cmdline()
pipeline.send_signal(signal.SIGTERM)
limit=time.monotonic()+10
while time.monotonic()<limit:
    final=json.loads((root/'results/series-1-budget-guard.json').read_text())
    if final['status'] in ('finished','stopped') and not final.get('remaining_owned_pids',[1]):break
    time.sleep(.2)
else:raise RuntimeError('Owned series cleanup not confirmed within ten seconds')
record.update(final_guard=final,gpu_rows_after=subprocess.check_output(['nvidia-smi',
    '--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=5).strip().splitlines())
(root/'logs/manual-stop-receipt.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'reason':record['reason'],'elapsed_seconds':final['elapsed_seconds'],
    'remaining_owned_pids':final['remaining_owned_pids'],'gpu_rows_after':record['gpu_rows_after']}),flush=True)
