"""CPU-only checks after interruption, using immutable dependency packages."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

root=Path('/root/fedcore_independent_cv_20261030');target=root/'poststop-guard-checks'
target.mkdir(exist_ok=False);started=time.monotonic()
manifest=json.loads((root/'poststop-guard-checks-manifest.json').read_text())
with tarfile.open(root/'poststop-guard-checks.tar.gz') as archive:
    for member in archive.getmembers():
        assert member.isfile() and not member.issym() and not member.islnk()
        assert (target/member.name).resolve().is_relative_to(target)
    archive.extractall(target)
for name,checksum in manifest.items():assert hashlib.sha256((target/name).read_bytes()).hexdigest()==checksum
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(target),
    OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
python=str(root/'.venv/bin/python')
command=[python,'-c',"import pytest,sys; status=pytest.main(['-q','tests/refactoring/test_cv_independent_supervision.py','tests/refactoring/test_cv_independent_guard.py']); assert 'torch' not in sys.modules; print('torch_imported=False; GPU numerical launches=0'); raise SystemExit(status)"]
with (root/'logs/poststop-guard-cpu-tests.log').open('x') as stream:
    result=subprocess.run(command,cwd=target,env=environment,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
print((root/'logs/poststop-guard-cpu-tests.log').read_text(),flush=True)
assert result.returncode==0
receipt={'schema':'fedcore-poststop-guard-cpu-checks-v1','tests_passed':True,'torch_imported':False,
    'numerical_gpu_launches':0,'seconds':time.monotonic()-started,'source_files_sha256':manifest,
    'scope':'Only CPU fakes of guard v2; no host observer, CUDA, real GPU telemetry, models or experiment retry.'}
(root/'logs/poststop-guard-cpu-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
