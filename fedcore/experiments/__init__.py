"""Reusable experiment API, loaded on demand without initializing ML runtimes.

In particular, a standard-library parent watchdog can start its clock before
the first child imports Torch. Public names retain their original definitions.
"""
from importlib import import_module

__all__ = ["CandidateSpec", "ExperimentBundle", "ExperimentProtocol", "ProtocolError",
           "TensorSplit", "split_indices", "validate_roles", "ExperimentRunner",
           "apply_candidate", "load_run", "quality_metrics", "measure_artifact",
           "predict_loaded_artifact", "SearchConfig", "compare_search_runs", "hypervolume_2d", "prepare_baseline"]

_OWNERS = {name: module for module, names in (
    ("protocol", ("CandidateSpec", "ExperimentBundle", "ExperimentProtocol", "ProtocolError",
                  "TensorSplit", "split_indices", "validate_roles")),
    ("runner", ("ExperimentRunner", "apply_candidate", "load_run", "quality_metrics")),
    ("baseline", ("prepare_baseline",)),
    ("measurement", ("measure_artifact", "predict_loaded_artifact")),
    ("search", ("SearchConfig", "compare_search_runs", "hypervolume_2d")),
) for name in names}


def __getattr__(name):
    if name not in _OWNERS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_OWNERS[name]}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
