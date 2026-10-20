"""Worker boundary tests use synthetic collaborators, never experimental data."""
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import torch

from fedcore.experiments import compression_search_digits_worker as worker
from fedcore.experiments import compression_search_digits_evaluation as evaluation
from fedcore.experiments.compression_search_evaluation import PackageFatal
from fedcore.experiments.compression_search_digits_protocol import INITIAL_GENOTYPES, canonical_solution_id


def _job(tmp_path, **changes):
    now = time.perf_counter()
    return {"job": "synthetic-random", "kind": "search", "mode": "T", "budget": "attempts",
        "split_seed": 19901, "model_seed": 19901, "search_seed": 11,
        "output": str(tmp_path), "directory": str(tmp_path), "repo": str(tmp_path),
        "worker_started_at": now, "package_deadline": now+100, "compute_admission_deadline": now+100,
        "admission_deadline": now+100, "protocol_sha256": "p", "source_sha256": "s", **changes}


def test_random_worker_attempt_budget_includes_common_prefix_and_uses_owned_stream(tmp_path, monkeypatch):
    source_dir = tmp_path / "sources" / "19901"
    source_dir.mkdir(parents=True)
    (source_dir / "manifest.json").write_text('{}')
    seen = []
    class Evaluator:
        rows = []
        context_sha256 = "c"
        def __call__(self, genotype):
            seen.append(canonical_solution_id(genotype))
            self.rows.append({"status": "finite", "solution_id": seen[-1], "completed_at": time.perf_counter()})
    evaluator = Evaluator()
    source = SimpleNamespace(seed=19901, manifest={})
    monkeypatch.setattr(evaluation, "source_evaluator", lambda *args: (source, None, evaluator))
    import fedcore.experiments.compression_search_evaluation as shared
    monkeypatch.setattr(shared, "compile_checked", lambda genotype, space: genotype)
    result = worker._evaluate_job(_job(tmp_path))
    assert len(seen) == len(set(seen)) == 24
    assert seen[:4] == list(map(canonical_solution_id, INITIAL_GENOTYPES))
    assert result["search"]["created_proposals"] == 24
    assert result["stop_reason"] == "evaluation_budget"
    assert result["deadline"] == pytest.approx(result["search"].get("deadline", result["deadline"]))


def test_tampered_selection_is_rejected_before_opening_test_role(tmp_path, monkeypatch):
    path = tmp_path / "selected.json"
    worker.write_json(path, {"frozen_at": 0, "selected": []})
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text(path.read_text() + " ")
    calls = []
    monkeypatch.setattr(evaluation, "load_role", lambda *args: calls.append(args))
    source = SimpleNamespace(seed=19901)
    with pytest.raises(PackageFatal, match="Selected IDs changed"):
        worker._finalize(_job(tmp_path, selections_sha256=expected), source, None, None)
    assert calls == []


def test_worker_deadline_before_first_effect_does_not_open_data(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(worker, "_prepare", lambda job: calls.append(job))
    result = worker.run_job(_job(tmp_path, kind="prepare", admission_deadline=0))
    assert result["partial"] is True and result["stop_reason"] == "time_budget"
    assert result["evaluations"] == [] and calls == []


def test_worker_rejects_split_identity_drift_before_data_access(tmp_path):
    with pytest.raises(PackageFatal, match="initialization is separately fixed"):
        worker.run_job(_job(tmp_path, model_seed=101))


@pytest.mark.parametrize("case", ("deadline", "invalid_identity"))
def test_worker_keeps_parent_backend_configuration_on_early_exits(tmp_path, monkeypatch, case):
    def configuration():
        return (torch.get_num_threads(), torch.get_num_interop_threads(),
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled())
    before = configuration()
    setters = []
    # Backend configuration belongs to the fresh-process parent. Even a setter
    # that happens to preserve today's value would violate that boundary.
    for name in ("set_num_threads", "set_num_interop_threads", "use_deterministic_algorithms"):
        monkeypatch.setattr(torch, name, lambda *args, _name=name, **kwargs: setters.append(_name))
    if case == "deadline":
        result = worker.run_job(_job(tmp_path, admission_deadline=0))
        assert result["partial"] is True
    else:
        with pytest.raises(PackageFatal, match="initialization is separately fixed"):
            worker.run_job(_job(tmp_path, model_seed=101))
    assert configuration() == before
    assert setters == []
