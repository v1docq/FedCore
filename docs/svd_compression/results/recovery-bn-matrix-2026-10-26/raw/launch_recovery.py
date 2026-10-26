"""Launch exactly one approved five-family recovery/BN window after CPU checks."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/root/fedcore_recovery_bn_20261026')
completion=Path('/root/fedcore_sparse_cv_completion_20261026')
diagnostic=Path('/root/fedcore_bn_ts_diagnostic_20261026')
original=Path('/root/fedcore_sparse_cv_20261021')
output=root/'results/recovery-1'
packet_path=root/'launch-packet.json';receipt_path=root/'launch-receipt.json'
assert not packet_path.exists() and not receipt_path.exists() and not output.exists()
assert not (output.parent/'.fedcore-sparse-cv.lock').exists()
manifest=json.loads((root/'source-manifest.json').read_text())
for name,checksum in manifest['package_files_sha256'].items():
    assert hashlib.sha256((root/'.venv/lib/python3.10/site-packages'/name).read_bytes()).hexdigest()==checksum,name
assert hashlib.sha256((root/'fedcore-0.0.5.4-py3-none-any.whl').read_bytes()).hexdigest()==manifest['wheel_sha256']
prepared=json.loads((root/'logs/installed-preflight.json').read_text())
assert prepared['status']=='prepared' and prepared['execute'] is False
assert prepared['recovery_sha256']==manifest['recovery_protocol_sha256']
environment=json.loads((original/'launch-packet-pilot-2.json').read_text())['environment']
environment['PYTHONPATH']=''
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10).strip(),'GPU occupied'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader,nounits'],text=True,timeout=10).strip()=='GPU-c31a29dc-f57d-9ce4-0303-849ada1f7cb8'
prior=json.loads((diagnostic/'results/diagnostic-1-budget-guard.json').read_text())['total_charged_seconds']
assert prior==3648.862081296742 and prior+1800<=14400
command=[str(root/'.venv/bin/python'),'-m','fedcore.experiments.cv_sparse_supervision','--execute',
    '--data-root',str(original/'data'),'--output-root',str(output),'--source-manifest',str(root/'source-manifest.json'),
    '--recovery-completion',str(completion/'results/completion-1'),'--completion-inventory',str(root/'completion-file-inventory.json'),
    '--recovery-diagnostic',str(diagnostic/'results/diagnostic-1'),'--diagnostic-inventory',str(root/'diagnostic-file-inventory.json')]
packet={'schema':'fedcore-recovery-bn-launch-v1','output_root':str(output),'command':command,
    'environment':environment,'repeat_slot_seconds':1800,'prior_charged_seconds':prior,
    'total_authorized_seconds':14400,'automatic_retry':False,'one_recovery_authorized':True,
    'launch_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'launch_monotonic':time.perf_counter(),
    'source_manifest_sha256':hashlib.sha256((root/'source-manifest.json').read_bytes()).hexdigest(),
    'budget_guard_sha256':hashlib.sha256((root/'recovery_budget_guard.py').read_bytes()).hexdigest(),
    'source_commit':manifest['source_commit'],'wheel_sha256':manifest['wheel_sha256'],
    'protocol_sha256':manifest['protocol_sha256'],'recovery_protocol_sha256':manifest['recovery_protocol_sha256']}
with packet_path.open('x') as stream:
    stream.write(json.dumps(packet,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
record={**packet,'packet_sha256':hashlib.sha256(packet_path.read_bytes()).hexdigest(),'status':'launch_intent'}
with receipt_path.open('x') as stream:
    stream.write(json.dumps(record,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
with (root/'logs/launch-console.log').open('x') as console:
    process=subprocess.Popen([str(root/'.venv/bin/python'),str(root/'recovery_budget_guard.py'),str(packet_path)],
        cwd=root,env=dict(os.environ,**environment),stdin=subprocess.DEVNULL,stdout=console,stderr=subprocess.STDOUT,start_new_session=True)
record.update(status='launched',budget_guard_pid=process.pid)
temporary=receipt_path.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n');os.replace(temporary,receipt_path)
print(json.dumps({key:record[key] for key in ('status','source_commit','launch_utc','budget_guard_pid','repeat_slot_seconds','prior_charged_seconds')}),flush=True)
