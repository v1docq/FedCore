"""Explicit opt-in for legacy tests that download data or need ML extras."""
from pathlib import Path
import pytest


_ML_INTEGRATION_FILES = {
    'test_lr.py', 'test_pruners.py', 'test_quant.py', 'test_evaluate.py',
    'test_llm_galore.py', 'test_llm_hooks.py', 'test_llm_trainer.py',
    'test_transmla.py', 'test_transmla_qwen_integration.py', 'test_flatllm.py',
    'test_datasets_load.py', 'test_unet.py', 'test_trainer_factory.py',
}


def pytest_addoption(parser):
    parser.addoption('--run-ml-integration', action='store_true', default=False,
                     help='Collect legacy tests requiring downloaded datasets/models or optional ML extras')


def pytest_configure(config):
    config.addinivalue_line('markers', 'ml_integration: downloaded model/data or optional ML runtime')
    config.addinivalue_line('markers', 'slow: intentionally expensive ML test')


def pytest_ignore_collect(collection_path: Path, config):
    if not config.getoption('--run-ml-integration'):
        return 'unit' in collection_path.parts and collection_path.name in _ML_INTEGRATION_FILES
    return None


def pytest_collection_modifyitems(items):
    for item in items:
        if Path(str(item.path)).name in _ML_INTEGRATION_FILES:
            item.add_marker(pytest.mark.ml_integration)
