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


def test_wheel_sdist_and_installed_package_outside_checkout(tmp_path):
    pytest.importorskip('build', reason='The declared test extra includes build')
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
"""
    run(['-c', code], outside, env)
