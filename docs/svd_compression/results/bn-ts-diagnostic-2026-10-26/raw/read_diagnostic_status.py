import datetime
import json
from pathlib import Path
import subprocess
import time
root=Path('/root/fedcore_bn_ts_diagnostic_20261026')
output=root/'results/diagnostic-1'
def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None
def tail(path,limit=2200):
    try:
        with path.open('rb') as stream:
            start=max(0,path.stat().st_size-limit);stream.seek(start);text=stream.read().decode(errors='replace')
        return text.split('\n',1)[-1] if start else text
    except FileNotFoundError:return ''
receipt=read(root/'launch-receipt.json');report=read(output/'report.json')
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'elapsed_seconds':None if receipt is None else time.perf_counter()-receipt['launch_monotonic'],
    'outer':read(root/'results/diagnostic-1-budget-guard.json'),
    'inner':read(root/'results/diagnostic-1-supervision.json'),
    'heartbeat':tail(output/'heartbeats.jsonl',1000),'recent_events':tail(output/'ledger.jsonl',2000),
    'quality_variants':len(list((output/'quality').glob('*.json'))),'AA_files':len(list((output/'latency').glob('*_AA.json'))),
    'report':None if report is None else {k:report.get(k) for k in ('status','failure','stop_reason','traceback','wall_seconds','total_SGD_updates','total_validation_rows','official_test_access')},
    'console':tail(root/'results/diagnostic-1-console.log',1500),
    'launch_console':tail(root/'logs/launch-console.log',700)},ensure_ascii=False),flush=True)
