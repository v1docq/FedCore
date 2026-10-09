"""A warm pytest process can hide tdecomp's first-import environment mutation."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('wandb_mode', [None, 'online'])
def test_cold_decomposer_import_preserves_caller_environment(wandb_mode):
    environment = dict(os.environ)
    if wandb_mode is None:
        environment.pop('WANDB_MODE', None)
    else:
        environment['WANDB_MODE'] = wandb_mode
    program = '''
import os
before = dict(os.environ)
from fedcore.algorithm.low_rank.decomposer import SVDDecomposition
changed = sorted(k for k in set(before) | set(os.environ) if before.get(k) != os.environ.get(k))
assert not changed, changed
'''
    result = subprocess.run([sys.executable, '-c', program], env=environment,
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
