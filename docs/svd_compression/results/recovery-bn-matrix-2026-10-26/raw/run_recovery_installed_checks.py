"""Installed Linux CPU checks, then integrity-only preflight; CUDA stays hidden."""
import json
import os
from pathlib import Path
import subprocess
import tarfile
root=Path('/root/fedcore_recovery_bn_20261026')
original=Path('/root/fedcore_sparse_cv_20261021')
completion=Path('/root/fedcore_sparse_cv_completion_20261026')
diagnostic=Path('/root/fedcore_bn_ts_diagnostic_20261026')
checks=root/'installed-checks';checks.mkdir()
with tarfile.open(root/'recovery-checks.tar.gz') as archive:
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
print((root/'logs/installed-cpu-tests.log').read_text()[-2400:],flush=True)
assert result.returncode==0
from fedcore.experiments.cv_sparse_continuation import frozen_dependencies
freeze=subprocess.check_output([python,'-m','pip','freeze'],env=environment,text=True)
assert frozen_dependencies(freeze)==frozen_dependencies((completion/'results/completion-1/environment_freeze.txt').read_text())
command=[python,'-m','fedcore.experiments.cv_sparse_recovery','--data-root',str(original/'data'),
    '--output-root',str(root/'results/preflight-not-executed'),'--source-manifest',str(root/'source-manifest.json'),
    '--completion-root',str(completion/'results/completion-1'),'--completion-inventory',str(root/'completion-file-inventory.json'),
    '--diagnostic-root',str(diagnostic/'results/diagnostic-1'),'--diagnostic-inventory',str(root/'diagnostic-file-inventory.json')]
with (root/'logs/installed-preflight.json').open('x') as log:
    result=subprocess.run(command,env=environment,cwd=root,stdout=log,stderr=subprocess.STDOUT)
print((root/'logs/installed-preflight.json').read_text()[-2400:],flush=True)
assert result.returncode==0 and not (root/'results/preflight-not-executed').exists()
print(json.dumps({'installed_checks_passed':True,'dependencies_match':True,'gpu_started':False}),flush=True)
