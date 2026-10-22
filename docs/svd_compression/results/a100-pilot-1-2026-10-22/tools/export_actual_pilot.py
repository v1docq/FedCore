"""Post-run evidence export; never alter the original pilot directory."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import tarfile
import zipfile

root=Path('/root/fedcore_sparse_cv_20261021')
output=root/'results/pilot-1'
state=json.loads((root/'results/pilot-1-supervision.json').read_text())
assert state['status'] in ('finished','stopped'),'Do not export an active numerical run'
report_path=output/'report.json'
report=json.loads(report_path.read_text()) if report_path.is_file() else {
    'status':'supervisor_interrupted_without_worker_report','reason':state.get('reason')}
export=root/'export-pilot-1-2026-10-22'
export.mkdir(exist_ok=False)
sources={path.relative_to(output).as_posix():path for path in output.rglob('*') if path.is_file()}
for path in (root/'results/pilot-1-supervision.json',root/'results/pilot-1-supervision.jsonl',
             root/'results/pilot-1-console.log',root/'logs/pilot-1-launch-console.log',
             root/'launch-receipt-2026-10-22.json',root/'launch-packet.json',
             root/'source-manifest.json',root/'prepared-admission.json',root/'data-prepared.json'):
    if path.is_file(): sources[path.name]=path
interruption=root/'interruption-observation-2026-10-22.json'
if interruption.is_file():sources[interruption.name]=interruption
files={}
for name,path in sorted(sources.items()):
    hasher=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):hasher.update(chunk)
    record={'bytes':path.stat().st_size,'sha256':hasher.hexdigest()}
    if path.suffix=='.pt':
        try:
            with zipfile.ZipFile(path) as archive:
                stored=[entry for entry in archive.infolist() if re.match(r'^[^/]+/data/\d+$',entry.filename)]
                record['torch_zip_storage_bytes']=sum(entry.file_size for entry in stored)
                record['torch_zip_storage_count']=len(stored)
        except zipfile.BadZipFile: record['zip_error']='invalid_or_incomplete_zip'
    files[name]=record
remaining=[]
for status in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        command=status.read_bytes()
        if b'fedcore.experiments.cv_sparse_' in command and b'--execute' in command:
            remaining.append(int(status.parent.name))
    except OSError: pass
assert not remaining,'The owned numerical/supervision process has not fully exited'
inventory={'exported_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'source_commit':'3bc0825a9ef1bcd0ac8eb0181b6d5314b57ebb8d','pilot_status':report['status'],
           'remaining_owned_pids':remaining,'files':files,'worker_report_present':report_path.is_file(),
           'binary_models':'retained on A100; all file hashes, sizes and Torch ZIP storage sizes recorded'}
inventory_path=export/'remote-file-inventory.json'
inventory_path.write_text(json.dumps(inventory,indent=2,sort_keys=True)+'\n')
suffixes={'.json','.jsonl','.npz','.log','.txt','.csv'}
archive_path=export/'evidence.tar.gz'
copied=[]
with tarfile.open(archive_path,'w:gz') as archive:
    for name,path in sorted(sources.items()):
        if path.suffix in suffixes:
            archive.add(path,arcname=name,recursive=False)
            copied.append(name)
    archive.add(inventory_path,arcname=inventory_path.name,recursive=False)
receipt={'archive':str(archive_path),'bytes':archive_path.stat().st_size,
         'sha256':hashlib.sha256(archive_path.read_bytes()).hexdigest(),
         'copied_files':len(copied)+1,'retained_binary_files':sum(Path(name).suffix=='.pt' for name in sources),
         'pilot_status':report['status'],'original_files_modified':False}
(export/'export-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
