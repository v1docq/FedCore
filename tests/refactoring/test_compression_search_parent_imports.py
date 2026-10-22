"""The parent's timing boundary must precede every ML runtime import."""
from pathlib import Path
import subprocess
import sys


def test_fresh_parent_matrix_and_export_import_no_ml_runtime():
    repo = Path(__file__).resolve().parents[2]
    code = """
import sys, runpy
import fedcore.experiments.compression_search_local2
import fedcore.experiments.compression_search_matrix
runpy.run_path('docs/svd_compression/issue99-local2-report-source/export.py')
assert not any(name in sys.modules for name in ('torch', 'numpy', 'sklearn', 'fedot', 'golem')), sorted(sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=repo, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_lazy_experiment_facade_preserves_original_api_objects():
    import fedcore.experiments as facade
    from fedcore.experiments.protocol import CandidateSpec
    from fedcore.experiments.runner import ExperimentRunner
    assert facade.CandidateSpec is CandidateSpec
    assert facade.ExperimentRunner is ExperimentRunner
    assert set(facade.__all__) <= set(dir(facade))
