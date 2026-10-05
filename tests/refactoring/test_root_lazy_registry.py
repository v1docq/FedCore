import subprocess
import sys

import pytest

from fedcore.repository.lazy_registry import LazyFactory, LazyRegistry


def test_registry_enumeration_does_not_import_optional_models():
    script = """
import sys
from fedcore.repository.model_repository import BACKBONE_MODELS, AtomizedModel, default_fedcore_availiable_operation
assert 'ResNet18' in BACKBONE_MODELS
assert len(list(BACKBONE_MODELS.keys())) == 24
assert default_fedcore_availiable_operation('low_rank') == ['low_rank_model']
assert AtomizedModel.PRUNED_RESNET_MODELS is not AtomizedModel.RESNET_MODELS
assert not any(m in sys.modules for m in ('torch', 'torchvision', 'transformers', 'fastai', 'bitsandbytes'))
"""
    subprocess.run([sys.executable, '-c', script], check=True, timeout=30)


def test_lazy_registry_resolves_real_callable_and_allows_extension():
    registry = LazyRegistry({'sqrt': LazyFactory('math', 'sqrt')})
    assert registry['sqrt'](9) == 3
    registry['custom'] = lambda x: x + 1
    assert registry['custom'](9) == 10
    del registry['custom']
    assert 'custom' not in registry
    with pytest.raises(TypeError):
        registry['invalid'] = 'arbitrary.module'


def test_combination_does_not_force_resolution():
    registry = LazyRegistry({'optional': LazyFactory('missing_optional_module', 'Factory')})
    combined = LazyRegistry.combine(registry)
    assert list(combined) == ['optional']
    with pytest.raises(ModuleNotFoundError):
        combined['optional']
