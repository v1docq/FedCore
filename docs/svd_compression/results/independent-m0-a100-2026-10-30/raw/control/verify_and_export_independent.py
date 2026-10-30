"""Post-run preservation and bounded export; never loads a model or a dataset."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile
import zipfile

root=Path('/root/fedcore_independent_cv_20261030');output=root/'results/series-1'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1024**2),b''):h.update(part)
    return h.hexdigest()

prior=json.loads((root/'logs/prior-preservation.json').read_text())['files']
for name,row in prior.items():
    path=Path(name);assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],name
manifest=json.loads((root/'source-manifest.json').read_text())
for name,checksum in manifest['package_files_sha256'].items():
    assert sha(root/'.venv/lib/python3.10/site-packages'/name)==checksum,name
assert sha(root/'fedcore-0.0.5.4-py3-none-any.whl')==manifest['wheel_sha256']
preservation={'schema':'fedcore-independent-cv-preservation-v1','old_files_verified':len(prior),
    'new_package_files_verified':len(manifest['package_files_sha256']),'new_wheel_verified':True,
    'gpu_processes_after':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits'],text=True,timeout=5).strip()}
(root/'logs/postrun-preservation.json').write_text(json.dumps(preservation,indent=2)+'\n')
export=root/'export-series-1';export.mkdir(exist_ok=False)
sources={p.relative_to(output).as_posix():p for p in output.rglob('*') if p.is_file()}
for p in output.rglob('*'):assert not p.is_symlink(),'Do not export symlinked artifacts'
for name in ('source-manifest.json','launch-packet.json','launch-receipt.json','independent_budget_guard.py',
             'launch_independent.py','run_independent_pipeline.py','verify_and_export_independent.py',
             'prepare_independent_environment.py','probe_independent_guard.py','probe_independent_guard_main.py'):
    sources['control/'+name]=root/name
for name in ('prior-preservation.json','postrun-preservation.json','install.log','pip-check.log',
             'installed-cpu-tests.log','installed-preflight.json','installed-checks-receipt.json',
             'guard-probe.log','guard-main-probe.log'):
    sources['control/'+name]=root/'logs'/name
files={}
for name,path in sorted(sources.items()):
    row={'bytes':path.stat().st_size,'sha256':sha(path)}
    if path.suffix=='.pt':
        with zipfile.ZipFile(path) as archive:
            tensors=[x for x in archive.infolist() if re.match(r'^[^/]+/data/\d+$',x.filename)]
            row.update(torch_zip_storage_bytes=sum(x.file_size for x in tensors),torch_zip_storage_count=len(tensors))
    files[name]=row
inventory=export/'independent-file-inventory.json'
inventory.write_text(json.dumps({'schema':'fedcore-independent-cv-inventory-v1','files':files,
    'export_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'models':'retained on A100; hashed independently without executing Torch'},indent=2,sort_keys=True)+'\n')
archive_path=export/'evidence.tar.gz';copied=0
with tarfile.open(archive_path,'w:gz') as archive:
    for name,path in sorted(sources.items()):
        if path.suffix=='.pt':continue
        archive.add(path,arcname=name,recursive=False);copied+=1
    archive.add(inventory,arcname=inventory.name,recursive=False)
report=json.loads((output/'report.json').read_text()) if (output/'report.json').exists() else {}
receipt={'archive':str(archive_path),'bytes':archive_path.stat().st_size,'sha256':sha(archive_path),
         'copied_files':copied+1,'retained_binary_files':sum(p.suffix=='.pt' for p in sources.values()),
         'status':report.get('status','missing_series_report')}
(export/'export-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
