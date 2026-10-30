"""Copy the exported parent namespace without altering any original file."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

root=Path('/root/fedcore_sparse_cv_completion_20261026')
old=Path('/root/fedcore_sparse_cv_20261021')
snapshot=root/'parent-snapshot'
assert not snapshot.exists()
inventory=json.loads((root/'remote-file-inventory.json').read_text())
assert shutil.disk_usage(root).free>sum(v['bytes'] for v in inventory['files'].values())+1024**3
from fedcore.experiments.cv_sparse_runtime import sha256_file
from fedcore.experiments.cv_sparse_continuation import inspect_parent,frozen_dependencies
snapshot.mkdir()
copied=[]
for name,expected in inventory['files'].items():
    choices=[old/'results/pilot-2'/name,old/name,old/'results'/name,old/'logs'/name]
    matches=[p for p in choices if p.is_file() and p.stat().st_size==expected['bytes'] and sha256_file(p)==expected['sha256']]
    assert matches,name
    source=matches[0];target=(snapshot/name).resolve()
    assert target.is_relative_to(snapshot)
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
    assert sha256_file(target)==expected['sha256']
    copied.append({'source':str(source),'target':name,'sha256':expected['sha256'],'bytes':expected['bytes']})
admission=inspect_parent(snapshot,root/'remote-file-inventory.json')
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
freeze=subprocess.check_output([str(root/'.venv/bin/python'),'-m','pip','freeze'],env=environment,text=True)
assert frozen_dependencies(freeze)==frozen_dependencies((snapshot/'environment_freeze.txt').read_text())
(root/'logs/parent-snapshot.json').write_text(json.dumps({'copied':copied,'original_files_modified':False,
    'full_parent_verified':True,'dependencies_match':True,'gpu_started':False},indent=2)+'\n')
command=[str(root/'.venv/bin/python'),'-m','fedcore.experiments.cv_sparse_continuation',
    '--data-root',str(old/'data'),'--output-root',str(root/'results/preflight-not-executed'),
    '--source-manifest',str(root/'source-manifest.json'),'--parent-root',str(snapshot),
    '--parent-inventory',str(root/'remote-file-inventory.json')]
with (root/'logs/complete-parent-preflight.json').open('x') as log:
    result=subprocess.run(command,env=environment,cwd=root,stdout=log,stderr=subprocess.STDOUT)
assert result.returncode==0,(root/'logs/complete-parent-preflight.json').read_text()[-3000:]
assert not (root/'results/preflight-not-executed').exists()
print(json.dumps({'copied_files':len(copied),'copied_bytes':sum(r['bytes'] for r in copied),
    'preflight':json.loads((root/'logs/complete-parent-preflight.json').read_text()),'dependencies_match':True}),flush=True)
