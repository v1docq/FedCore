"""Reusable, leakage-checked experiment protocol and measurement runner."""
from .protocol import (CandidateSpec, ExperimentBundle, ExperimentProtocol,
                       ProtocolError, TensorSplit, split_indices, validate_roles)
from .runner import ExperimentRunner, apply_candidate, load_run, quality_metrics
from .baseline import prepare_baseline
from .measurement import measure_artifact, predict_loaded_artifact
from .search import SearchConfig, compare_search_runs, hypervolume_2d

__all__ = ["CandidateSpec", "ExperimentBundle", "ExperimentProtocol", "ProtocolError",
           "TensorSplit", "split_indices", "validate_roles", "ExperimentRunner",
           "apply_candidate", "load_run", "quality_metrics", "measure_artifact",
           "predict_loaded_artifact", "SearchConfig", "compare_search_runs", "hypervolume_2d", "prepare_baseline"]
