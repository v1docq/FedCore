"""Isolated wheel overlay with read-only frozen dependencies and no CUDA."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root=Path('/root/fedcore_independent_cv_20261030');old=Path('/root/fedcore_sparse_cv_20261021');venv=root/'.venv'
assert not venv.exists()
subprocess.run([str(old/'.venv/bin/python'),'-m','venv',str(venv)],check=True)
(venv/'lib/python3.10/site-packages/frozen_dependencies.pth').write_text(str(old/'.venv/lib/python3.10/site-packages')+'\n')
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='',PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',
    OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',
    HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
wheel=root/'fedcore-0.0.5.4-py3-none-any.whl';manifest=json.loads((root/'source-manifest.json').read_text())
assert hashlib.sha256(wheel.read_bytes()).hexdigest()==manifest['wheel_sha256']
with (root/'logs/install.log').open('x') as log:
    subprocess.run([str(venv/'bin/python'),'-m','pip','install','--no-deps','--ignore-installed',str(wheel)],
        env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
with (root/'logs/pip-check.log').open('x') as log:
    subprocess.run([str(venv/'bin/python'),'-m','pip','check'],env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
print(json.dumps({'new_environment':str(venv),'gpu_started':False,'old_environment_modified':False}))
