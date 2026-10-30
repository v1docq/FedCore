"""Read-only post-run audit of the frozen recovery parent and older packages."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
root=Path('/root/fedcore_saved_eval_20261026')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()
admission=json.loads((root/'results/evaluation-1/recovery_admission.json').read_text())
parent=Path(admission['root'])
for name,row in admission['verified_files'].items():
    path=parent/name
    assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],name
counts={}
for label,path in (('original',Path('/root/fedcore_sparse_cv_20261021')),
                   ('completion',Path('/root/fedcore_sparse_cv_completion_20261026')),
                   ('diagnostic',Path('/root/fedcore_bn_ts_diagnostic_20261026')),
                   ('recovery',Path('/root/fedcore_recovery_bn_20261026'))):
    manifest=json.loads((path/'source-manifest.json').read_text())
    for name,checksum in manifest['package_files_sha256'].items():
        assert sha(path/'.venv/lib/python3.10/site-packages'/name)==checksum,(label,name)
    assert sha(path/'fedcore-0.0.5.4-py3-none-any.whl')==manifest['wheel_sha256']
    counts[label]=len(manifest['package_files_sha256'])
snapshot=json.loads(Path('/root/fedcore_sparse_cv_completion_20261026/logs/parent-snapshot.json').read_text())
for row in snapshot['copied']:
    path=Path(row['source']);assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],row['source']
result={'schema':'fedcore-saved-evaluation-preservation-v1',
    'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'consumed_parent_files_verified':len(admission['verified_files']),
    'old_snapshot_files_verified':len(snapshot['copied']),
    'old_package_files_verified':counts,'old_wheels_unchanged':True,
    'gpu_compute_processes_after':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip()}
(root/'logs/postrun-preservation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
