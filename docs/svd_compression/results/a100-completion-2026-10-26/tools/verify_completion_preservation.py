import datetime
import hashlib
import json
from pathlib import Path
import subprocess
root=Path('/root/fedcore_sparse_cv_completion_20261026')
old=Path('/root/fedcore_sparse_cv_20261021')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()
snapshot=json.loads((root/'logs/parent-snapshot.json').read_text())
for row in snapshot['copied']:
    path=Path(row['source']);assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],row['source']
manifest=json.loads((old/'source-manifest.json').read_text())
for name,digest in manifest['package_files_sha256'].items():
    assert sha(old/'.venv/lib/python3.10/site-packages'/name)==digest,name
assert sha(old/'fedcore-0.0.5.4-py3-none-any.whl')==manifest['wheel_sha256']
result={'original_parent_files_verified':len(snapshot['copied']),
    'original_packaged_sources_verified':len(manifest['package_files_sha256']),
    'original_wheel_unchanged':True,'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'gpu_compute_processes_after':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip()}
(root/'logs/postrun-preservation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
