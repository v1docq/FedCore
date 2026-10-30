"""One authorized launch. Preparation and CPU checks do not start this clock."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

root=Path('/root/fedcore_independent_cv_20261030');original=Path('/root/fedcore_sparse_cv_20261021')
output=root/'results/series-1';packet_path=root/'launch-packet.json';receipt_path=root/'launch-receipt.json'
assert not packet_path.exists() and not receipt_path.exists() and not output.exists()
assert shutil.disk_usage(root).free>=100*1024**3
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
manifest=json.loads((root/'source-manifest.json').read_text())
for name,checksum in manifest['package_files_sha256'].items():
    assert sha(root/'.venv/lib/python3.10/site-packages'/name)==checksum,name
assert sha(root/'fedcore-0.0.5.4-py3-none-any.whl')==manifest['wheel_sha256']
prepared=json.loads((root/'logs/installed-preflight.json').read_text())
assert prepared['status']=='prepared' and prepared['execute'] is False and prepared['protocol_sha256']==manifest['protocol_sha256']
checks=json.loads((root/'logs/installed-checks-receipt.json').read_text())
assert checks['installed_cpu_checks_passed'] and checks['guard_probes_passed'] and checks['dependencies_match']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader,nounits'],text=True,timeout=10).strip()=='GPU-c31a29dc-f57d-9ce4-0303-849ada1f7cb8'
environment=json.loads((original/'launch-packet-pilot-2.json').read_text())['environment']
environment.update(PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
began=time.perf_counter();deadline=began+28790
environment['FEDCORE_PILOT_ABSOLUTE_DEADLINE']=str(deadline)
series_command=[str(root/'.venv/bin/python'),'-m','fedcore.experiments.cv_independent_series','--execute',
    '--data-root',str(original/'data'),'--output-root',str(output),
    '--source-manifest',str(root/'source-manifest.json'),'--absolute-deadline',str(deadline)]
command=[str(root/'.venv/bin/python'),str(root/'run_independent_pipeline.py'),'--packet',str(packet_path)]
external_files=('independent_budget_guard.py','launch_independent.py','run_independent_pipeline.py','verify_and_export_independent.py')
packet={'schema':'fedcore-independent-cv-launch-v1','one_launch_authorized':True,'automatic_retry':False,
    'max_seconds':28800,'worker_seconds':28790,'prior_seconds':0,'output_root':str(output),
    'command':command,'series_command':series_command,'environment':environment,'launch_monotonic':began,
    'launch_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'guard_sha256':sha(root/'independent_budget_guard.py'),'external_files_sha256':{n:sha(root/n) for n in external_files},
    'source_manifest_sha256':sha(root/'source-manifest.json'),'source_commit':manifest['source_commit'],
    'wheel_sha256':manifest['wheel_sha256'],'protocol_sha256':manifest['protocol_sha256']}
with packet_path.open('x') as stream:json.dump(packet,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
record={'status':'launch_intent','packet_sha256':sha(packet_path),'launch_utc':packet['launch_utc']}
with receipt_path.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
with (root/'logs/guard-launch-console.log').open('x') as log:
    process=subprocess.Popen([str(root/'.venv/bin/python'),str(root/'independent_budget_guard.py'),str(packet_path)],
        env=dict(os.environ,**environment),cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record.update(status='launched',guard_pid=process.pid)
temporary=receipt_path.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n');os.replace(temporary,receipt_path)
print(json.dumps(record),flush=True)
