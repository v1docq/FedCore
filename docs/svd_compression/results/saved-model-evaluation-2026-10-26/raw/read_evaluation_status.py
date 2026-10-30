"""Read compact status only; never import ML packages or resume work."""
import datetime
import json
from pathlib import Path
root=Path('/root/fedcore_saved_eval_20261026');out=root/'results/evaluation-1'
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
for label,path in (('outer',root/'results/evaluation-1-budget-guard.json'),('inner',root/'results/evaluation-1-supervision.json'),('report',out/'report.json')):
    if path.is_file():
        data=json.loads(path.read_text())
        if label=='report':data={key:value for key,value in data.items() if key in ('status','failure','failure_type','stop_reason','traceback','total_SGD_updates','total_validation_rows','total_official_test_rows','total_latency_pairs','wall_seconds')}
        result[label]=data
result['quality_variants']=len(list((out/'quality').glob('*.json')))
result['test_results']=len(list((out/'official_test').glob('*.json')))
result['latency_files']=len(list((out/'latency').glob('*.json')))
if (out/'ledger.jsonl').is_file():
    events=[json.loads(line) for line in (out/'ledger.jsonl').read_text().splitlines()]
    result['recent_events']=events[-3:]
if (root/'results/evaluation-1-console.log').is_file():result['console_tail']=(root/'results/evaluation-1-console.log').read_text()[-2500:]
print(json.dumps(result))
