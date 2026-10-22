"""Launch exactly the one human-authorized repeat into a fresh output directory."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/root/fedcore_sparse_cv_20261021')
old_receipt=json.loads((root/'launch-receipt-2026-10-22.json').read_text())
old_state=json.loads((root/'results/pilot-1-supervision.json').read_text())
spent=old_state['elapsed_seconds']
assert spent==1379.477359779179 and old_state['reason']=='external_gpu_contention'
assert old_receipt['automatic_retry'] is False
packet_path=root/'launch-packet.json'
packet_sha=hashlib.sha256(packet_path.read_bytes()).hexdigest()
assert packet_sha=='2c4034abefd3a4301bda6191af8086018e71440b2d588546383f655bda6e98da'
old_packet=json.loads(packet_path.read_text())
source_path=root/'source-manifest.json'
assert hashlib.sha256(source_path.read_bytes()).hexdigest()==old_packet['source_manifest_sha256']
source=json.loads(source_path.read_text())
assert source['source_commit']=='3bc0825a9ef1bcd0ac8eb0181b6d5314b57ebb8d'
assert source['protocol_sha256']==old_receipt['protocol_sha256']
assert hashlib.sha256((root/'fedcore-0.0.5.4-py3-none-any.whl').read_bytes()).hexdigest()==source['wheel_sha256']
for name,checksum in source['package_files_sha256'].items():
    assert hashlib.sha256((root/'.venv/lib/python3.10/site-packages'/name).read_bytes()).hexdigest()==checksum,name
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                   capture_output=True,text=True,timeout=10,check=True)
assert not gpu.stdout.strip(),'GPU occupied; no repeat launched'
for name in ('pilot-2','pilot-2-supervision.json','pilot-2-supervision.jsonl','pilot-2-console.log',
             'pilot-2-budget-guard.json','.fedcore-sparse-cv.lock'):
    assert not (root/'results'/name).exists(),name
receipt_path=root/'launch-receipt-pilot-2-2026-10-22.json'
new_packet_path=root/'launch-packet-pilot-2.json'
assert not receipt_path.exists() and not new_packet_path.exists()
command=list(old_packet['command'])
command[command.index('--output-root')+1]=str(root/'results/pilot-2')
packet={**old_packet,'command':command,'output_root':str(root/'results/pilot-2'),
        'repeat_slot_seconds':13020,'prior_charged_seconds':spent,
        'automatic_retry':False,'repeat_authorized':True,'parent_packet_sha256':packet_sha,
        'launch_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'launch_monotonic':time.perf_counter()}
assert packet['repeat_slot_seconds']+spent<=14400
with new_packet_path.open('x') as stream:
    stream.write(json.dumps(packet,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
guard=root/'retry_budget_guard.py'
record={'status':'launch_intent','one_repeat_authorized':True,'automatic_retry':False,
    'source_commit':source['source_commit'],'wheel_sha256':source['wheel_sha256'],
    'protocol_sha256':source['protocol_sha256'],
    'parent_packet_sha256':packet_sha,'packet_sha256':hashlib.sha256(new_packet_path.read_bytes()).hexdigest(),
    'budget_guard_sha256':hashlib.sha256(guard.read_bytes()).hexdigest(),
    'prior_charged_seconds':spent,'repeat_slot_seconds':13020,'total_authorized_seconds':14400,
    'command':command,'environment_overrides':packet['environment'],
    'launch_utc':packet['launch_utc'],'launch_monotonic':packet['launch_monotonic']}
with receipt_path.open('x') as stream:
    stream.write(json.dumps(record,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
environment=dict(os.environ,**packet['environment'])
with (root/'logs/pilot-2-launch-console.log').open('x') as console:
    process=subprocess.Popen([str(root/'.venv/bin/python'),str(guard),str(new_packet_path)],
        cwd=root,env=environment,stdin=subprocess.DEVNULL,stdout=console,stderr=subprocess.STDOUT,start_new_session=True)
record.update(status='launched',budget_guard_pid=process.pid)
temporary=receipt_path.with_suffix('.tmp')
temporary.write_text(json.dumps(record,indent=2)+'\n')
os.replace(temporary,receipt_path)
print(json.dumps(record),flush=True)
