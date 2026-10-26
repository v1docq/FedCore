import json
import os
from pathlib import Path
import subprocess
import tarfile
root=Path('/root/fedcore_sparse_cv_completion_20261026')
checks=root/'installed-checks';checks.mkdir()
with tarfile.open(root/'completion-checks.tar.gz') as archive:
    for member in archive.getmembers():
        assert member.isfile() and not member.issym() and not member.islnk()
        assert (checks/member.name).resolve().is_relative_to(checks)
    archive.extractall(checks)
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='',PYTHONUTF8='1',
    OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',
    HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
python=str(root/'.venv/bin/python')
with (root/'logs/installed-cpu-tests.log').open('x') as log:
    result=subprocess.run([python,'-m','pytest','-q','tests/refactoring'],env=environment,cwd=checks,stdout=log,stderr=subprocess.STDOUT)
print((root/'logs/installed-cpu-tests.log').read_text()[-3000:],flush=True)
assert result.returncode==0
command=[python,'-m','fedcore.experiments.cv_sparse_continuation',
    '--data-root','/root/fedcore_sparse_cv_20261021/data','--output-root',str(root/'results/preflight-not-executed'),
    '--source-manifest',str(root/'source-manifest.json'),'--parent-root','/root/fedcore_sparse_cv_20261021/results/pilot-2',
    '--parent-inventory',str(root/'remote-file-inventory.json')]
with (root/'logs/installed-preflight.json').open('x') as log:
    result=subprocess.run(command,env=environment,cwd=root,stdout=log,stderr=subprocess.STDOUT)
print((root/'logs/installed-preflight.json').read_text()[-3000:],flush=True)
assert result.returncode==0 and not (root/'results/preflight-not-executed').exists()
print(json.dumps({'installed_checks_passed':True,'gpu_started':False}),flush=True)
