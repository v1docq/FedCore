import datetime,hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/root/fedcore_sparse_cv_20261021')
packet_path=root/'launch-packet.json'
assert hashlib.sha256(packet_path.read_bytes()).hexdigest()=='2c4034abefd3a4301bda6191af8086018e71440b2d588546383f655bda6e98da'
packet=json.loads(packet_path.read_text())
manifest_path=root/'source-manifest.json'
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==packet['source_manifest_sha256']
manifest=json.loads(manifest_path.read_text())
assert manifest['source_commit']==packet['source_commit']=='3bc0825a9ef1bcd0ac8eb0181b6d5314b57ebb8d'
assert hashlib.sha256((root/'fedcore-0.0.5.4-py3-none-any.whl').read_bytes()).hexdigest()==manifest['wheel_sha256']
package=root/'.venv/lib/python3.10/site-packages'
for name,checksum in manifest['package_files_sha256'].items():
    assert hashlib.sha256((package/name).read_bytes()).hexdigest()==checksum,name
result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20,check=True)
assert not result.stdout.strip(),'GPU was occupied again before launch'
for name in ('pilot-1','pilot-1-supervision.json','pilot-1-supervision.jsonl','pilot-1-console.log','.fedcore-sparse-cv.lock'):
    assert not (root/'results'/name).exists(),name
for name in ('batches.meta','test_batch',*[f'data_batch_{index}' for index in range(1,6)]):
    assert (root/'data/cifar-10-batches-py'/name).is_file(),name
assert json.loads((root/'prepared-admission.json').read_text())['status']=='prepared'
receipt_path=root/'launch-receipt-2026-10-22.json'
environment=dict(os.environ,**packet['environment'])
record={'status':'launch_intent','source_commit':manifest['source_commit'],'protocol_sha256':manifest['protocol_sha256'],
        'wheel_sha256':manifest['wheel_sha256'],'packet_sha256':'2c4034abefd3a4301bda6191af8086018e71440b2d588546383f655bda6e98da',
        'command':packet['command'],'environment_overrides':packet['environment'],
        'max_slot_seconds':packet['max_slot_seconds'],'automatic_retry':False,
        'launch_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'launch_monotonic':time.perf_counter()}
with receipt_path.open('x') as receipt:
    receipt.write(json.dumps(record,indent=2)+'\n');receipt.flush();os.fsync(receipt.fileno())
with (root/'logs/pilot-1-launch-console.log').open('x') as console:
    process=subprocess.Popen(packet['command'],cwd=root,env=environment,stdin=subprocess.DEVNULL,
        stdout=console,stderr=subprocess.STDOUT,start_new_session=True)
record.update(status='launched',supervisor_pid=process.pid)
temporary=receipt_path.with_suffix('.tmp')
with temporary.open('w') as receipt:
    receipt.write(json.dumps(record,indent=2)+'\n');receipt.flush();os.fsync(receipt.fileno())
os.replace(temporary,receipt_path)
print(json.dumps(record),flush=True)
