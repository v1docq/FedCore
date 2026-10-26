"""Export stopped completion evidence; retain executable models on A100."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import tarfile
import zipfile

root=Path('/root/fedcore_recovery_bn_20261026')
output=root/'results/recovery-1'
guard=json.loads((root/'results/recovery-1-budget-guard.json').read_text())
inner=json.loads((root/'results/recovery-1-supervision.json').read_text())
assert guard['status'] in ('finished','stopped') and inner['status'] in ('finished','stopped')
remaining=[]
for path in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        command=path.read_bytes()
        if b'fedcore.experiments.cv_sparse_' in command and str(output).encode() in command:
            remaining.append(int(path.parent.name))
    except OSError:pass
assert not remaining
export=root/'export-recovery-1';export.mkdir()
sources={p.relative_to(output).as_posix():p for p in output.rglob('*') if p.is_file()}
for folder in (root,root/'results',root/'logs'):
    for p in folder.iterdir():
        if p.is_file() and p.suffix in ('.json','.jsonl','.log','.py','.txt'):
            name=p.name
            if name in sources:name='control/'+name
            sources[name]=p
files={}
for name,p in sorted(sources.items()):
    digest=hashlib.sha256()
    with p.open('rb') as stream:
        for part in iter(lambda:stream.read(1024**2),b''):digest.update(part)
    row={'bytes':p.stat().st_size,'sha256':digest.hexdigest(),'mtime_ns':p.stat().st_mtime_ns}
    if p.suffix=='.pt':
        with zipfile.ZipFile(p) as archive:
            stored=[i for i in archive.infolist() if re.match(r'^[^/]+/data/\d+$',i.filename)]
            row.update(torch_zip_storage_bytes=sum(i.file_size for i in stored),torch_zip_storage_count=len(stored))
    files[name]=row
inventory={'schema':'fedcore-cv-completion-inventory-v1','files':files,'original_sources_modified':False,
           'export_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'binary_models':'retained on A100, hashes and physical tensor storage sizes recorded'}
path=export/'recovery-file-inventory.json';path.write_text(json.dumps(inventory,indent=2,sort_keys=True)+'\n')
archive_path=export/'evidence.tar.gz'
copied=0
with tarfile.open(archive_path,'w:gz') as archive:
    for name,p in sorted(sources.items()):
        if p.suffix in ('.json','.jsonl','.log','.py','.txt','.csv','.npz') or p.name.endswith('.json.tmp'):
            archive.add(p,arcname=name,recursive=False);copied+=1
    archive.add(path,arcname=path.name,recursive=False)
receipt={'archive':str(archive_path),'bytes':archive_path.stat().st_size,
         'sha256':hashlib.sha256(archive_path.read_bytes()).hexdigest(),'copied_files':copied+1,
         'status':json.loads((output/'report.json').read_text())['status'],
         'retained_binary_files':sum(p.suffix=='.pt' for p in sources.values())}
(export/'export-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
