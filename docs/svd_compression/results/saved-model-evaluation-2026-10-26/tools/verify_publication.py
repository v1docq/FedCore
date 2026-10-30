"""Verify immutable evidence, deterministic replay and optional Git index bytes."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote,urlsplit

p=argparse.ArgumentParser();p.add_argument('--analysis-replay',type=Path,required=True);p.add_argument('--report-replay',type=Path,required=True)
p.add_argument('--archive',type=Path);p.add_argument('--require-index',action='store_true');a=p.parse_args()
packet=Path(__file__).resolve().parents[1];docs=packet.parents[1];repo=docs.parents[1]
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
manifest=json.loads((packet/'manifest.json').read_text());files=manifest['files']
for name,row in files.items():
    path=docs/name;assert path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],name
for path in packet.rglob('*'):
    if path.is_file() and path!=packet/'manifest.json':assert path.relative_to(docs).as_posix() in files,str(path)
for name in ('analysis.json','checks.json'):
    assert (packet/'analysis'/name).read_bytes()==(a.analysis_replay/name).read_bytes(),name
for name in ('SAVED_MODEL_EVALUATION.md','saved-model-evaluation-report.html'):
    assert (docs/name).read_bytes()==(a.report_replay/name).read_bytes(),name
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=set();self.remote_resources=[]
    def handle_starttag(self,tag,attrs):
        values=dict(attrs)
        if 'id' in values:self.ids.add(values['id'])
        if tag=='a' and 'href' in values:self.links.append(values['href'])
        if tag in ('script','img','link') and any(urlsplit(values.get(k,'')).scheme for k in ('src','href')):self.remote_resources.append(values)
html=(docs/'saved-model-evaluation-report.html').read_text(encoding='utf-8');parser=Links();parser.feed(html);local=0
assert not parser.remote_resources
for value in parser.links:
    url=urlsplit(value)
    if url.scheme:continue
    local+=1
    if url.path:assert (docs/unquote(url.path)).is_file(),value
    else:assert url.fragment in parser.ids,value
for value in re.findall(r'\]\(([^)]+)\)',(docs/'SAVED_MODEL_EVALUATION.md').read_text(encoding='utf-8')):
    url=urlsplit(value)
    if not url.scheme:assert (docs/unquote(url.path)).exists(),value
embedded=json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>',html,re.S).group(1))
assert embedded['summary']==json.loads((packet/'analysis/analysis.json').read_text())
assert len(embedded['pairs'])==320 and len(embedded['summary']['AA'])==28
old=0
for name in ('a100-pilot-1-2026-10-22','a100-pilot-2-2026-10-22','a100-completion-2026-10-26',
             'bn-ts-diagnostic-preparation-2026-10-26','bn-ts-diagnostic-2026-10-26','recovery-bn-matrix-2026-10-26'):
    path=docs/'results'/name/'manifest.json';data=json.loads(path.read_text());values=data.get('files',data);base=docs if 'files' in data else path.parent
    for relative,row in values.items():
        target=base/relative;assert target.is_file() and sha(target)==row['sha256'],(name,relative)
        old+=1
archive_verified=None
if a.archive:
    receipt=json.loads((packet/'validation/saved-evaluation-export-receipt.json').read_text())
    assert sha(a.archive)==receipt['sha256'] and a.archive.stat().st_size==receipt['bytes'];archive_verified=True
index=0
if a.require_index:
    tracked=set(subprocess.check_output(['git','ls-files','--cached','-z'],cwd=repo).decode().split('\0'))
    for name,row in files.items():
        relative='docs/svd_compression/'+name;assert relative in tracked,relative
        content=subprocess.check_output(['git','show',':'+relative],cwd=repo)
        assert hashlib.sha256(content).hexdigest()==row['sha256'],relative
        index+=1
print(json.dumps({'hashed_files':len(files),'index_hashed_files':index,'old_hashed_files_unchanged':old,
    'html_links':len(parser.links),'local_targets_verified':local,'analysis_byte_replay':True,
    'report_byte_replay':True,'embedded_summary_exact':True,'archive_SHA_verified':archive_verified}))
