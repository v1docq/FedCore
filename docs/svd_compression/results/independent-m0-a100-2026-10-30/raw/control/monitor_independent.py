"""Read-only bounded snapshots; no ML imports or numerical work."""
import datetime
import json
from pathlib import Path
import subprocess

root=Path('/root/fedcore_independent_cv_20261030');output=root/'results/series-1'
def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None
def tail_events(path):
    if not path.is_file():return []
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-20000))
        lines=stream.read().splitlines()
    events=[]
    for line in lines:
        try:events.append(json.loads(line))
        except (json.JSONDecodeError,UnicodeDecodeError):pass
    return events

master=tail_events(output/'ledger.jsonl');stage=next((r for r in reversed(master) if r.get('event')=='stage_started'),None)
workers=[]
for folder in sorted((output/'workers').glob('*')):
    terminal=read_json(folder/'result.json');events=tail_events(folder/'ledger.jsonl')
    workers.append({'worker':folder.name,'status':terminal.get('status') if terminal else 'running_or_partial',
        'last_event':events[-1] if events else None,
        'latest_progress':next((r for r in reversed(events) if r.get('event') in
             ('baseline_epoch','baseline_update','recovery_update','candidate_ended','latency_pair','evaluation_batch')),None)})
guard=read_json(root/'results/series-1-budget-guard.json')
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'guard':guard,'stage':stage,'last_master':master[-3:],'workers':workers,
    'pipeline':read_json(root/'logs/pipeline-receipt.json'),
    'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5).strip()},allow_nan=False))
