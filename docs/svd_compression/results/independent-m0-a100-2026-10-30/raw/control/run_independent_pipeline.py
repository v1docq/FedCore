"""Series and evidence export share the same external eight-hour lifetime."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

parser=argparse.ArgumentParser();parser.add_argument('--packet',type=Path,required=True);args=parser.parse_args()
root=args.packet.parent;packet=json.loads(args.packet.read_text());began=packet['launch_monotonic']
deadline=began+packet['worker_seconds']
for name,checksum in packet['external_files_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==checksum,name
series=subprocess.run(packet['series_command'],cwd=root)
close_started=time.perf_counter()
ledger=Path(packet['output_root'])/'ledger.jsonl'
if ledger.exists():
    for line in ledger.read_text().splitlines():
        try:event=json.loads(line)
        except json.JSONDecodeError:break
        if event.get('event')=='stage_started' and event.get('stage')=='close':
            close_started=event['timestamp']
remaining=min(close_started+14*60,deadline)-time.perf_counter()
receipt={'schema':'fedcore-independent-cv-pipeline-v1','series_returncode':series.returncode,
    'export_attempted':False,'automatic_retry':False,'close_started_monotonic':close_started}
if remaining>1:
    receipt['export_attempted']=True
    with (root/'logs/postrun-export.log').open('x') as stream:
        try:
            post=subprocess.run([sys.executable,str(root/'verify_and_export_independent.py')],
                cwd=root,stdout=stream,stderr=subprocess.STDOUT,timeout=remaining)
            receipt['export_returncode']=post.returncode
        except subprocess.TimeoutExpired:receipt['export_returncode']=124
receipt.update(total_pipeline_seconds=time.perf_counter()-began,within_worker_deadline=time.perf_counter()<=deadline)
(root/'logs/pipeline-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
raise SystemExit(0 if series.returncode==0 and receipt.get('export_returncode')==0 else 1)
