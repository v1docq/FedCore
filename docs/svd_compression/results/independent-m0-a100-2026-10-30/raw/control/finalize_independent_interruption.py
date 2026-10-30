"""Read-only numerical evidence closure after the one-shot run was interrupted."""
import datetime
import json
from pathlib import Path
import subprocess
import psutil

root=Path('/root/fedcore_independent_cv_20261030')
manual=json.loads((root/'logs/manual-stop.json').read_text())
guard=json.loads((root/'results/series-1-budget-guard.json').read_text())
assert guard['status']=='finished' and guard['returncode']==-15
observations=[]
for item in manual['owned_descendants']:
    try:
        process=psutil.Process(item['pid'])
        same=process.create_time()==item['birth'];status=process.status()
        assert not same or status==psutil.STATUS_ZOMBIE,'An owned process remains alive'
        observations.append({'pid':item['pid'],'same_birth':same,'status':status})
    except psutil.NoSuchProcess:observations.append({'pid':item['pid'],'status':'absent'})
supervision=[json.loads(line) for line in (root/'logs/series-1-supervision.jsonl').read_text().splitlines()]
multi=[row for row in supervision if row.get('gpu') is not None and len(row['gpu'])>1]
counts={}
for path in (root/'results/series-1/workers').glob('*/ledger.jsonl'):
    events=[json.loads(line) for line in path.read_text().splitlines()]
    counts[path.parent.name]={name:sum(row.get('event')==name for row in events) for name in
       ('baseline_update','baseline_epoch','recovery_update','candidate_charged','evaluation_batch','latency_pair')}
receipt={'schema':'fedcore-independent-cv-interruption-closure-v1',
    'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':manual['reason'],
    'new_numerical_launches':0,'automatic_retry':False,'guard_terminal':guard,
    'first_multiple_gpu_seconds':multi[0]['elapsed_seconds'] if multi else None,
    'last_multiple_gpu_seconds':multi[-1]['elapsed_seconds'] if multi else None,
    'multiple_gpu_observations':len(multi),'owned_processes_now':observations,
    'confirmed_counts':counts,'remaining_live_owned_processes':0,
    'gpu_rows_after':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits'],text=True,timeout=5).strip().splitlines()}
with (root/'logs/interruption-closure.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
print(json.dumps(receipt),flush=True)
