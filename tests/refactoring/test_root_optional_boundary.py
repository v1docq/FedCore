"""Exercise real imports while optional distributions are unavailable."""
import os
import subprocess
import sys


def test_cpu_public_imports_do_not_require_optional_ml_or_service_extras():
    script = """
import importlib.abc
import importlib.util
import importlib.metadata
import sys
unavailable = {'transformers', 'datasets', 'accelerate', 'evaluate', 'onnx', 'onnxruntime', 'flask', 'bitsandbytes', 'segmentation_models_pytorch', 'opendatasets'}
original_find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, package=None: None if name.split('.')[0] in unavailable else original_find_spec(name, package)
original_version = importlib.metadata.version
original_distribution = importlib.metadata.distribution
original_distributions = importlib.metadata.distributions
importlib.metadata.distributions = lambda **kwargs: (d for d in original_distributions(**kwargs) if d.metadata['Name'].lower().replace('-', '_') not in unavailable)
def optional_metadata(name, original):
    if name.lower().replace('-', '_') in unavailable:
        raise importlib.metadata.PackageNotFoundError(name)
    return original(name)
importlib.metadata.version = lambda name: optional_metadata(name, original_version)
importlib.metadata.distribution = lambda name: optional_metadata(name, original_distribution)
class UnavailableOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in unavailable:
            raise ModuleNotFoundError('Optional dependency unavailable: ' + fullname, name=fullname)
sys.meta_path.insert(0, UnavailableOptional())
from fedcore.api.main import FedCore
from fedcore.algorithm.low_rank.low_rank_opt import LowRankModel
from fedcore.algorithm.low_rank.lora_operation import BaseLoRA
from fedcore.algorithm.pruning.pruners import BasePruner
from fedcore.algorithm.quantization.quantizers import BaseQuantizer
from external.flatllmcore.core.absorption import ActivationCollector
from fedcore.external_runtime.contracts import InputSpec
assert InputSpec((1, 4)).shape == (1, 4)
"""
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                            timeout=60, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    assert result.returncode == 0, result.stdout + result.stderr
