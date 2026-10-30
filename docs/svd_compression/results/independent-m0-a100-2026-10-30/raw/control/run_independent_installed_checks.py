"""Sequential installed-wheel checks; CUDA disabled throughout preparation."""
import json
import os
from pathlib import Path
import subprocess
import tarfile

root=Path('/root/fedcore_independent_cv_20261030');original=Path('/root/fedcore_sparse_cv_20261021')
checks=root/'installed-checks';checks.mkdir(exist_ok=False)
with tarfile.open(root/'independent-checks.tar.gz') as archive:
    for member in archive.getmembers():
        assert member.isfile() and not member.issym() and not member.islnk()
        assert (checks/member.name).resolve().is_relative_to(checks)
    archive.extractall(checks)
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='',PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',
    OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',
    HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
python=str(root/'.venv/bin/python');names=json.loads((root/'independent-installed-test-list.json').read_text())
with (root/'logs/installed-cpu-tests.log').open('x') as log:
    result=subprocess.run([python,'-m','pytest','-q',*names],cwd=checks,env=environment,stdout=log,stderr=subprocess.STDOUT)
print((root/'logs/installed-cpu-tests.log').read_text()[-2200:],flush=True);assert result.returncode==0
freeze=subprocess.check_output([python,'-m','pip','freeze'],env=environment,text=True)
def normalized(value):
    return tuple(sorted(line.strip() for line in value.splitlines() if line.strip()
        and not line.lower().startswith(('fedcore==','fedcore @ '))))
assert normalized(freeze)==normalized(Path('/root/fedcore_saved_eval_20261026/results/evaluation-1/environment_freeze.txt').read_text())
command=[python,'-m','fedcore.experiments.cv_independent_series','--data-root',str(original/'data'),
    '--output-root',str(root/'results/preflight-not-executed'),'--source-manifest',str(root/'source-manifest.json')]
with (root/'logs/installed-preflight.json').open('x') as log:
    result=subprocess.run(command,cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT)
assert result.returncode==0 and not (root/'results/preflight-not-executed').exists()
prepared=json.loads((root/'logs/installed-preflight.json').read_text());assert prepared['status']=='prepared' and prepared['execute'] is False
for name in ('guard-probe.log','guard-main-probe.log'):
    assert 'cpu_only' in (root/'logs'/name).read_text(),name
receipt={'installed_cpu_checks_passed':True,'test_modules':len(names),'dependencies_match':True,
    'guard_probes_passed':True,'gpu_started':False,'protocol_sha256':prepared['protocol_sha256'],
    'source_commit':prepared['source']['supplied']['source_commit']}
(root/'logs/installed-checks-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
