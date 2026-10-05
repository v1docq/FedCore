"""Opt-in integration against the caller's actual modern FEDOT environment."""
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest
import torch
from torch import nn
from fedcore.external_runtime.client import save_dataset
from fedcore.external_runtime.models import save_model_bundle


def test_modern_fedot_manifest_real_registry_and_isolated_worker(tmp_path):
    modern = os.environ.get("FEDCORE_MODERN_FEDOT_ROOT")
    caller_python = os.environ.get("FEDCORE_MODERN_FEDOT_PYTHON", sys.executable)
    if not modern:
        pytest.skip("Set FEDCORE_MODERN_FEDOT_ROOT for the actual IndustrialTS/FEDOT contract integration")
    model = nn.Linear(8, 2)
    x = torch.randn(6, 8)
    y = model(x).detach()
    save_model_bundle(model, tmp_path / "model.fcb")
    save_dataset(tmp_path / "regression.fcb", x, y)
    save_dataset(tmp_path / "classification.fcb", x, y.argmax(1))
    payload = {"modern": modern, "root": str(tmp_path), "worker_python": sys.executable,
               "features": x.tolist(), "regression_targets": y.tolist(), "classification_targets": y.argmax(1).tolist()}
    program = r'''
import json, os, sys, numpy as np
from pathlib import Path
p=json.loads(sys.argv[1])
sys.path.insert(0,p['modern'])
from fedot.extensions import validate_extension_manifest, smoke_test_extension, get_registered_extensions, extension_scope, load_extension_manifest
from fedcore.external_runtime.adapters import build_industrial_extension_manifest
before=get_registered_extensions()
loaded=load_extension_manifest('fedcore.external_runtime.industrial_manifest')
assert not loaded.is_left(), loaded
manifest=build_industrial_extension_manifest()
assert not validate_extension_manifest(manifest).is_left()
root=Path(p['root'])
params={spec.name:{'model_bundle':str(root/'model.fcb'), 'validation_bundle':str(root/(task+'.fcb')), 'jobs_root':str(root/('jobs-'+task)), 'python_executable':p['worker_python']} for spec,task in zip(manifest.models,('classification','regression'))}
assert not smoke_test_extension(manifest,params).is_left()
assert smoke_test_extension(manifest,{}).is_left()
assert get_registered_extensions()==before
x=np.asarray(p['features'],dtype=np.float32)
environment=dict(os.environ)
for spec,task in zip(manifest.models,('classification','regression')):
    estimator=spec.factory(params[spec.name])
    target=np.asarray(p[task+'_targets'],dtype=np.int64 if task=='classification' else np.float32)
    estimator.fit(x,target)
    prediction=estimator.predict(x)
    assert prediction.shape==target.shape
    if task=='regression': np.testing.assert_allclose(prediction,target,rtol=1e-4,atol=1e-4)
    else: assert np.array_equal(prediction,target)
    assert get_registered_extensions()==before and dict(os.environ)==environment
with extension_scope(manifest):
    assert len(get_registered_extensions())==len(before)+1
assert get_registered_extensions()==before
try:
    with extension_scope(manifest): raise RuntimeError('isolated failure')
except RuntimeError: pass
assert get_registered_extensions()==before
print(json.dumps({'manifest':'validated','factory_smoke':'passed','real_fit_predict':'passed','registry_unchanged':True,'caller_python':sys.executable,'worker_python':p['worker_python']}))
'''
    completed = subprocess.run([caller_python, "-c", program, json.dumps(payload)], cwd=Path(__file__).resolve().parents[2],
                               capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1])["registry_unchanged"]
