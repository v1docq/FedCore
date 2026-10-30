"""Real typed CV search results must cross both pilot JSON boundaries safely."""
import json
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from fedcore.experiments import cv_sparse_pilot as pilot
from fedcore.experiments import cv_sparse_runtime as runtime
from fedcore.experiments.cv_sparse_protocol import PilotProtocol


def search_result():
    prefix = pilot.fixed_prefix(pilot.cv_menu())
    return {"status": "completed", "best_genotypes": [prefix[1], prefix[2]],
            "trace": [], "engine": "FedcoreEvoOptimizer", "charged_evaluations": 12}


def test_evolution_saves_portable_genotypes_without_mutating_typed_result(tmp_path, monkeypatch):
    from fedcore.algorithm.low_rank import grammar_search
    from fedcore.algorithm.low_rank.cv_grammar import CVGenotype, cv_genotype_payload
    result = search_result()
    identities = [pilot.solution_id(value) for value in result["best_genotypes"]]
    output = runtime.Output(tmp_path / "out")
    owner = SimpleNamespace(protocol=PilotProtocol(), records={}, space=object(), output=output,
                            prefix=pilot.fixed_prefix(pilot.cv_menu()), short=lambda *args: None)
    monkeypatch.setattr(grammar_search, "run_cv_grammar_search", lambda *args, **kwargs: result)
    returned = pilot.Pilot.evolution(owner, time.perf_counter() + 60)
    saved = json.loads((output.root / "search/evolution.json").read_text())
    assert returned is result
    assert all(type(value) is CVGenotype for value in result["best_genotypes"])
    assert [pilot.solution_id(value) for value in result["best_genotypes"]] == identities
    assert saved["best_genotypes"] == [cv_genotype_payload(value) for value in result["best_genotypes"]]
    assert [value["profile"] for value in saved["best_genotypes"]] == ["cv_internal_channels_v1"] * 2
    assert json.loads(json.dumps(saved, allow_nan=False)) == saved


def test_run_pilot_projects_typed_evolution_again_for_final_report(tmp_path, monkeypatch):
    from fedcore.experiments import cv_sparse_data
    result = search_result()
    prefix = pilot.fixed_prefix(pilot.cv_menu())
    data = SimpleNamespace(roles=SimpleNamespace(sha256="b" * 64, train=tuple(range(128)), payload=lambda: {}), source_hashes={})
    class FakeRuntime:
        def __init__(self, protocol, output, started):
            self.deadline = started + 60
            self.environment = {}
            self.torch = SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda: 0,
                                                             max_memory_reserved=lambda: 0))
        def phase(self, name):
            return nullcontext(self.deadline)
        def check(self, *args, **kwargs):
            pass
    class FakePilot:
        def __init__(self, runtime, data):
            self.prefix = prefix
            self.gpu = object()
            self.genotypes = {}
            self.records = {"random": [], "evolution": []}
            self.byte_ceiling = 80
            self.space = SimpleNamespace(source_storage_bytes=100)
        def bind_teacher(self, teacher):
            pass
        def short(self, *args):
            pass
        def random_search(self, deadline):
            return {"status": "completed"}
        def evolution(self, deadline):
            return result
    def stop_before_recovery(*args):
        raise runtime.PilotStop("recovery", "test_boundary_before_numerics")
    monkeypatch.setattr(pilot, "preparation", lambda *args: {"status": "prepared", "source": {}})
    monkeypatch.setattr(pilot, "admit_free_gpu", lambda: {})
    monkeypatch.setattr(pilot, "Runtime", FakeRuntime)
    monkeypatch.setattr(pilot, "Pilot", FakePilot)
    monkeypatch.setattr(cv_sparse_data, "CifarData", lambda *args, **kwargs: data)
    monkeypatch.setattr(pilot.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=""))
    monkeypatch.setattr(pilot, "verify_loaded_sources", lambda *args: {})
    monkeypatch.setattr(pilot, "train_baseline", lambda *args: (object(), {"top1": .95}, {}))
    monkeypatch.setattr(pilot, "checkpoint", lambda *args, **kwargs: ({"artifact_id": "a" * 64}, object()))
    monkeypatch.setattr(pilot, "torchscript_artifact", lambda *args: (object(), {}))
    monkeypatch.setattr(pilot, "independent_torchscript", lambda *args: object())
    monkeypatch.setattr(pilot, "paired_latency", lambda *args, **kwargs: {"accepted": True})
    monkeypatch.setattr(pilot, "recovery_tape", stop_before_recovery)
    returned = pilot.run_pilot(tmp_path / "data", tmp_path / "out", tmp_path / "source.json")
    saved = json.loads((tmp_path / "out/report.json").read_text())
    assert returned["status"] == saved["status"] == "pilot_incomplete_recovery", returned.get("traceback")
    assert all(type(value) is dict for value in saved["evolution"]["best_genotypes"])
    assert result["best_genotypes"] == list(prefix[1:3])


def test_report_projection_refuses_untyped_winners_without_stringifying_them():
    invalid = {"best_genotypes": [object()], "status": "completed"}
    with pytest.raises(ValueError, match="metadata-only CV genotypes"):
        pilot._cv_search_report(invalid)


def test_empty_search_archive_keeps_failure_and_trace_metadata():
    original = {"best_genotypes": [], "status": "failed", "reason": "bounded_failure", "trace": [{"event": "stopped"}]}
    projected = pilot._cv_search_report(original)
    assert projected == original and projected is not original
    assert json.loads(json.dumps(projected, allow_nan=False)) == original
