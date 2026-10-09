"""Build both formats and restore/import the installed wheel outside the source tree."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
import pytest

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = {
    'fedcore/algorithm/low_rank/method_specs.py',
    'fedcore/algorithm/low_rank/method_execution.py',
    'fedcore/algorithm/low_rank/statistical_profiles.py',
    'fedcore/algorithm/low_rank/statistical_collectors.py',
    'fedcore/algorithm/low_rank/structured_profiles.py',
    'fedcore/algorithm/low_rank/structured_layers.py',
    'fedcore/algorithm/low_rank/factor_recovery.py',
    'fedcore/experiments/svd_rank_policies.py',
    'fedcore/experiments/svd_profile_measurement.py',
    'fedcore/algorithm/low_rank/execution.py',
    'fedcore/algorithm/low_rank/plans.py',
    'fedcore/algorithm/low_rank/statistics.py',
    'fedcore/algorithm/low_rank/approximation.py',
    'fedcore/algorithm/low_rank/allocation.py',
    'fedcore/algorithm/low_rank/topology.py',
    'fedcore/experiments/protocol.py',
    'fedcore/experiments/runner.py',
    'fedcore/experiments/scenarios.py',
    'fedcore/interfaces/search_trace.py',
    'fedcore/architecture/comptutaional/devices.py',
    'fedcore/repository/data/compression_model_repository.json',
    'fedcore/repository/data/compression_data_operation_repository.json',
    'fedcore/repository/data/compression_operation_params.json',
    'fedcore/repository/data/mode_capabilities.json',
    'model_exporter/templates/web_ui.html',
    'model_exporter/static/js/script.js',
    'model_exporter/static/css/style.css',
    'model_exporter/device_architectures/Jetson_arch.json',
}


def run(args, cwd, env=None):
    result = subprocess.run([sys.executable, *args], cwd=cwd, env=env,
                            capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_wheel_sdist_and_installed_package_outside_checkout(tmp_path, monkeypatch):
    pytest.importorskip('build', reason='The declared test extra includes build')
    temporary = tmp_path / 'temporary'
    temporary.mkdir()
    # Windows sandbox TEMP cannot atomically replace setuptools metadata.
    monkeypatch.setenv('TEMP', str(temporary))
    monkeypatch.setenv('TMP', str(temporary))
    source = tmp_path / 'source'
    source.mkdir()
    for name in ('fedcore', 'external', 'external_runtime', 'model_exporter'):
        if (ROOT / name).is_dir():
            shutil.copytree(ROOT / name, source / name,
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('README.md', 'LICENSE', 'pyproject.toml', 'setup.py', 'MANIFEST.in'):
        shutil.copy2(ROOT / name, source / name)
    artifacts = tmp_path / 'dist'
    run(['-m', 'build', '--no-isolation', '--outdir', str(artifacts)], source)
    wheel = next(artifacts.glob('*.whl'))
    sdist = next(artifacts.glob('*.tar.gz'))
    with zipfile.ZipFile(wheel) as archive:
        assert REQUIRED <= set(archive.namelist())
        metadata = archive.read(next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))).decode()
        assert 'Version: 0.0.5.4' in metadata and 'Requires-Python: <3.12,>=3.10' in metadata
    unpacked = tmp_path / 'sdist'
    with tarfile.open(sdist) as archive:
        names = {'/'.join(name.split('/')[1:]) for name in archive.getnames()}
        assert REQUIRED <= names
        # Builds were created locally; reject path traversal before extraction.
        assert all(not Path(name).is_absolute() and '..' not in Path(name).parts for name in archive.getnames())
        archive.extractall(unpacked)
    rebuilt = tmp_path / 'rebuilt'
    run(['-m', 'build', '--wheel', '--no-isolation', '--outdir', str(rebuilt)], next(unpacked.iterdir()))
    with zipfile.ZipFile(next(rebuilt.glob('*.whl'))) as archive:
        assert REQUIRED <= set(archive.namelist())
    installed = tmp_path / 'installed'
    run(['-m', 'pip', 'install', '--no-deps', '--target', str(installed), str(wheel)], tmp_path)
    outside = tmp_path / 'outside'
    outside.mkdir()
    env = dict(os.environ, PYTHONPATH=str(installed), HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    code = """from importlib import resources, metadata
from pathlib import Path
import json
import fedcore
assert metadata.version('fedcore') == fedcore.__version__ == '0.0.5.4'
from fedcore.architecture.comptutaional.devices import default_device
assert default_device('cpu').type == 'cpu'
for name in ['compression_model_repository.json', 'compression_data_operation_repository.json', 'compression_operation_params.json']:
    json.loads(resources.files('fedcore.repository.data').joinpath(name).read_text())
assert resources.files('model_exporter').joinpath('templates/web_ui.html').is_file()
from fedot.core.operations.operation import Operation
original = Operation.fit
from fedcore.api.main import FedCore
assert Operation.fit is original
from fedcore.experiments import ExperimentProtocol, ExperimentRunner
from fedcore.experiments.scenarios import build_tabular
bundle = build_tabular(seed=42)
assert bundle.task == 'classification'
assert ExperimentProtocol().device == 'cpu'
import torch
from torch import nn
from fedcore.algorithm.low_rank.low_rank_opt import LowRankModel
from fedcore.tools.registry.checkpoint_manager import CheckpointManager
from fedcore.tools.export import export_model
torch.set_num_threads(2)
model = nn.Linear(6, 4).double().eval()
x = torch.randn(5, 6, dtype=torch.double)
__import__('os').environ['FEDCORE_MODEL_REGISTRY_PATH'] = str(Path.cwd() / 'registry')
facade = LowRankModel({'device': 'cpu'})
result = facade.compress_weighted(model, x, rank=2)
assert result.model is facade.model_after
assert result.plan.request.statistics_refs
manager = CheckpointManager(str(Path.cwd() / 'checkpoint'), auto_cleanup=False)
path = str(Path.cwd() / 'checkpoint.pt')
manager.save_to_file(manager.serialize_to_bytes(result.model), path)
restored = manager.load_from_file(path)
torch.testing.assert_close(restored(x), result.model(x), rtol=1e-12, atol=1e-12)
artifact = export_model(restored.eval(), 'torchscript', Path.cwd() / 'export.pt', x[:1])
with artifact.open('rb') as stream:
    reloaded = torch.jit.load(stream)
torch.testing.assert_close(reloaded(x[:1]), restored(x[:1]), rtol=1e-12, atol=1e-12)
assert Path(fedcore.__file__).is_relative_to(Path(__import__('os').environ['PYTHONPATH']))
from fedcore.algorithm.low_rank.method_specs import ASVD, AFM, DRONE, SVDLLMV2
from fedcore.external_runtime import MethodOptions, SUPPORTED_CONTRACT_VERSIONS
assert SUPPORTED_CONTRACT_VERSIONS == (1, 2, 3)
for spec in (ASVD(), AFM(), DRONE(), SVDLLMV2()):
    result = facade.compress_profile(model, x, spec, rank=2, target_paths=('',))
    manager.save_to_file(manager.serialize_to_bytes(result.model), path)
    restored = manager.load_from_file(path)
    torch.testing.assert_close(restored(x), result.model(x))
    artifact = export_model(restored.eval(), 'torchscript', Path.cwd() / 'p2.pt', x[:1])
    with artifact.open('rb') as stream:
        reloaded = torch.jit.load(stream)
    torch.testing.assert_close(reloaded(x[:1]), restored(x[:1]))
"""
    run(['-c', code], outside, env)
