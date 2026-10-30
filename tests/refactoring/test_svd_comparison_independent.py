"""Independent boundary and evidence regressions for the local SVD study."""
import json
from copy import deepcopy
from types import SimpleNamespace
import sys

import pytest
import torch
from torch import nn

from fedcore.experiments import runner
from fedcore.experiments.protocol import CandidateSpec, ExperimentProtocol, ProtocolError
from fedcore.experiments import svd_comparison as study


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_physical_storage_counts_oversized_backing_once_across_parameter_buffer():
    backing = torch.arange(20., dtype=torch.float32)
    model = nn.Module()
    model.register_parameter("weight_view", nn.Parameter(backing[2:6]))
    model.register_buffer("buffer_view", backing[10:12])
    model.register_buffer("separate", torch.ones(3, dtype=torch.float64))
    assert study.exact_storage(model)["tensor_bytes"] == 20 * 4 + 3 * 8


def test_conflicting_per_alias_ranks_fail_without_touching_source():
    shared = nn.Linear(4, 4)
    source = nn.Sequential(shared, shared, nn.Linear(4, 2)).eval()
    before = deepcopy(source.state_dict())
    bundle = study.build_bundle("mlp", 41)
    candidate = CandidateSpec("svd", {"target_paths": ["0", "1"],
                                     "ranks": {"0": 1, "1": 2}, "finetune_epochs": 0})
    with pytest.raises(ProtocolError, match="(?i)alias|shared|conflict"):
        runner.apply_candidate(source, bundle, ExperimentProtocol(finetune_epochs=0), candidate)
    assert source[0] is source[1]
    for name, value in before.items():
        torch.testing.assert_close(source.state_dict()[name], value, rtol=0, atol=0)


def test_selecting_one_module_alias_changes_all_calls_and_preserves_unselected_head():
    torch.manual_seed(71)
    shared = nn.Linear(4, 4)
    source = nn.Sequential(shared, shared, nn.Linear(4, 2)).eval()
    original = deepcopy(source.state_dict())
    u, singular, vh = torch.linalg.svd(shared.weight.detach().double(), full_matrices=False)
    approximation = ((u[:, :2] * singular[:2]) @ vh[:2]).float()
    x = torch.arange(20., dtype=torch.float32).reshape(5, 4) / 10
    expected = torch.nn.functional.linear(x, approximation, shared.bias)
    expected = torch.nn.functional.linear(expected, approximation, shared.bias)
    expected = source[2](expected)
    bundle = study.build_bundle("mlp", 41)
    compressed, _ = runner.apply_candidate(source, bundle, ExperimentProtocol(finetune_epochs=0),
        CandidateSpec("svd", {"target_paths": ["1"], "ranks": {"1": 2}, "finetune_epochs": 0}))
    assert compressed[0] is compressed[1]
    assert compressed[0].U.shape[-1] == 2
    torch.testing.assert_close(compressed(x), expected, rtol=1e-5, atol=1e-6)
    for name, value in original.items():
        torch.testing.assert_close(source.state_dict()[name], value, rtol=0, atol=0)
        if name.startswith("2."):
            torch.testing.assert_close(compressed.state_dict()[name], value, rtol=0, atol=0)


def test_conflicting_ranks_for_distinct_modules_sharing_weight_are_rejected():
    first, second = nn.Linear(4, 4), nn.Linear(4, 4)
    second.weight = first.weight
    source = nn.Sequential(first, second, nn.Linear(4, 2)).eval()
    original = deepcopy(source.state_dict())
    bundle = study.build_bundle("mlp", 41)
    with pytest.raises((ProtocolError, ValueError), match="(?i)alias|shared|tie|conflict"):
        runner.apply_candidate(source, bundle, ExperimentProtocol(finetune_epochs=0),
            CandidateSpec("svd", {"target_paths": ["0", "1"],
                                  "ranks": {"0": 1, "1": 2}, "finetune_epochs": 0}))
    assert source[0].weight is source[1].weight
    for name, value in original.items():
        torch.testing.assert_close(source.state_dict()[name], value, rtol=0, atol=0)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_timing_retry_charges_previous_compute_and_timing_sessions(tmp_path, monkeypatch):
    _write(tmp_path / "protocol.json", {"source": {}})
    _write(tmp_path / "execution.json", {"architecture_workers": [
        {"resources": {"elapsed_seconds": 100.}}]})
    _write(tmp_path / "timing-execution.json", {"workers": [
        {"job": str(tmp_path / "mlp-seed-41/baseline/timing-job.json"),
         "resources": {"elapsed_seconds": 200., "returncode": 0, "violation": None}}]})
    _write(tmp_path / "mlp-seed-41/baseline/timing-job.json",
           {"output": str(tmp_path / "mlp-seed-41/baseline/timing.json")})
    deadlines = []
    monkeypatch.setattr(study, "source_identity", lambda: {})
    monkeypatch.setattr(study, "COMPUTE_SECONDS", 1000)
    monkeypatch.setattr(study.time, "monotonic", lambda: 1000.)
    monkeypatch.setattr(study, "consolidate_results", lambda output: {})

    def worker(command, directory, deadline):
        deadlines.append(deadline)
        return {"returncode": 0, "violation": None, "elapsed_seconds": 10.}

    monkeypatch.setattr(study, "monitored_process", worker)
    assert study.run_series(tmp_path, timing_only=True) == 0
    assert deadlines == [1700.]
    execution = json.loads((tmp_path / "timing-execution.json").read_text(encoding="utf-8"))
    assert sum(item["resources"]["elapsed_seconds"] for item in execution["workers"]) == 210.


def test_exhausted_time_budget_refuses_worker_before_spawning(tmp_path, monkeypatch):
    monkeypatch.setattr(study.time, "monotonic", lambda: 100.)

    def forbidden_spawn(*args, **kwargs):
        raise AssertionError("An exhausted series must not start another child")

    monkeypatch.setattr(study.subprocess, "Popen", forbidden_spawn)
    result = study.monitored_process(["unused"], tmp_path, 100.)
    assert result["returncode"] != 0
    assert result["violation"] and result["elapsed_seconds"] == 0


@pytest.mark.parametrize("child_mib,nested_mib", [(256, 32), (96, 96)])
def test_rss_guard_detects_interpreter_descendants_and_terminates_entire_tree(tmp_path, monkeypatch, child_mib, nested_mib):
    """The Windows launcher alone is small; only its child's RSS violates cap."""
    unit = 1024**2
    processes = {}

    class MissingProcess(Exception):
        pass

    class Process:
        def __init__(self, pid, rss, stubborn=False):
            self.pid, self.rss, self.stubborn = pid, rss, stubborn
            self.alive, self.terminated, self.killed, self.polls = True, False, False, 0
            processes[pid] = self

        def children(self, recursive=False):
            assert recursive, "The RSS limit must include nested interpreter children"
            return [processes[12], processes[13]] if self.pid == 11 and self.alive else []

        def memory_info(self):
            if not self.alive:
                raise MissingProcess()
            return SimpleNamespace(rss=self.rss)

        def terminate(self):
            self.terminated = True
            if not self.stubborn:
                self.alive = False

        def kill(self):
            self.killed, self.alive = True, False

        def is_running(self):
            return self.alive

        def create_time(self):
            return float(self.pid)

        def ppid(self):
            return {11: 0, 12: 11, 13: 12}[self.pid]

        def cpu_times(self):
            return SimpleNamespace(user=0., system=0.)

        def num_threads(self):
            return 1

        def poll(self):
            self.polls += 1
            # Bounds the regression test even for the old launcher-only monitor.
            return None if self.alive and self.polls < 3 else (0 if self.alive else -15)

        def wait(self, timeout=None):
            return 0 if self.alive else -15

    root, child, nested = Process(11, 4 * unit), Process(12, child_mib * unit), Process(13, nested_mib * unit, True)
    fake_psutil = SimpleNamespace(Process=lambda pid: processes[pid],
        virtual_memory=lambda: SimpleNamespace(available=32 * 1024**3), cpu_percent=lambda **kwargs: 0.,
        NoSuchProcess=MissingProcess, AccessDenied=PermissionError,
        wait_procs=lambda values, timeout=None: ([p for p in values if not p.alive], [p for p in values if p.alive]))
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(study.subprocess, "Popen", lambda *args, **kwargs: root)
    monkeypatch.setattr(study, "RSS_LIMIT", 128 * unit)
    monkeypatch.setattr(study.time, "monotonic", lambda: 100.)
    result = study.monitored_process(["unused"], tmp_path, 200.)
    assert result["violation"] and "rss" in result["violation"].lower()
    assert result["sampled_maximum_individual_rss_bytes"] == max(4, child_mib, nested_mib) * unit
    assert result["sampled_maximum_tree_sum_rss_bytes"] == (4 + child_mib + nested_mib) * unit
    assert root.terminated and child.terminated and nested.terminated
    assert nested.killed, "A stubborn interpreter descendant must be killed after termination timeout"
    assert not any(p.alive for p in processes.values())


def test_prior_attempt_charge_is_frozen_for_v2_and_preserved_during_timing_retry(tmp_path, monkeypatch):
    prior, current = tmp_path / "run-v1", tmp_path / "run-v2"
    _write(prior / "execution.json", {"architecture_workers": [
        {"resources": {"elapsed_seconds": 100.}}, {"resources": {"elapsed_seconds": 20.}}]})
    _write(prior / "timing-execution.json", {"workers": [{"resources": {"elapsed_seconds": 30.}}]})
    monkeypatch.setattr(study, "source_identity", lambda: {})
    monkeypatch.setattr(study.runner, "environment_manifest", lambda: {})
    monkeypatch.setattr(study, "COMPUTE_SECONDS", 1000)
    plan = study.predeclare_plan(current, prior_runs=(prior,))
    resource = plan["resources"]
    assert resource["prior_charged_worker_wall_seconds"] == 150.
    assert resource["remaining_at_release_seconds"] == 850.
    ledger = resource["prior_compute_ledger"][0]
    assert ledger["evidence_sha256"] == {
        name: study.measurement.file_hash(prior / name) for name in ("execution.json", "timing-execution.json")}
    _write(current / "execution.json", {"architecture_workers": [{"resources": {"elapsed_seconds": 100.}}]})
    job = current / "mlp-seed-41/baseline/timing-job.json"
    _write(job, {"output": str(job.parent / "timing.json")})
    _write(current / "timing-execution.json", {"workers": [
        {"job": str(job), "resources": {"elapsed_seconds": 200., "returncode": 0, "violation": None}}]})
    monkeypatch.setattr(study.time, "monotonic", lambda: 1000.)
    monkeypatch.setattr(study, "consolidate_results", lambda output: {})
    deadlines = []

    def worker(command, directory, deadline):
        deadlines.append(deadline)
        return {"returncode": 0, "violation": None, "elapsed_seconds": 10.}

    monkeypatch.setattr(study, "monitored_process", worker)
    assert study.run_series(current, timing_only=True) == 0
    assert deadlines == [1550.]


def test_prior_attempt_exhaustion_refuses_new_prospective_protocol(tmp_path, monkeypatch):
    prior, current = tmp_path / "run-v1", tmp_path / "run-v2"
    _write(prior / "execution.json", {"architecture_workers": [{"resources": {"elapsed_seconds": 1000.}}]})
    monkeypatch.setattr(study, "COMPUTE_SECONDS", 1000)
    with pytest.raises(ProtocolError, match="exhausted"):
        study.predeclare_plan(current, prior_runs=(prior,))
    assert not (current / "protocol.json").exists()


def _quality(accuracy, f1, cross_entropy):
    return {"test": {"value": accuracy, "macro_f1": f1, "cross_entropy": cross_entropy}}


def test_summary_pairs_by_architecture_and_seed_and_retains_failed_seed(tmp_path):
    for seed, baseline_accuracy, candidate_accuracy in ((43, .70, .80), (41, .90, .95), (42, .85, .65)):
        baseline = {"case_id": "baseline", "method": "baseline", "architecture": "mlp", "seed": seed,
                    "status": "succeeded", "quality": _quality(baseline_accuracy, .8, .2),
                    "storage": {"tensor_bytes": 1000, "artifact_file_bytes": 3000}}
        candidate = {"case_id": "candidate", "method": "svd", "architecture": "mlp", "seed": seed,
                     "cohort": "common_linear", "level": 0, "status": "succeeded",
                     "quality": _quality(candidate_accuracy, .7, .4), "compression_seconds": 1.,
                     "storage": {"tensor_bytes": 500, "artifact_file_bytes": 2000}}
        failed = dict(candidate, case_id="failed", method="asvd", status="failed")
        _write(tmp_path / f"mlp-seed-{seed}/run.json", {"architecture": "mlp", "seed": seed,
               "gate": {"passed": True}, "baseline_hash": str(seed), "baseline": baseline,
               "records": [candidate, failed]})
    summary = study.consolidate_results(tmp_path)
    groups = {row["method"]: row for row in summary["paired_summaries"]}
    assert groups["svd"]["paired_seeds"] == [41, 42, 43]
    assert groups["svd"]["test_accuracy_deltas"] == pytest.approx([.05, -.20, .10])
    assert groups["svd"]["descriptive"]["test_accuracy_deltas"]["mean"] == pytest.approx(-1/60)
    assert groups["asvd"]["paired_seeds"] == []
    assert groups["asvd"]["failed_seeds"] == [41, 42, 43]
    assert groups["asvd"]["descriptive"]["test_accuracy_deltas"]["mean"] is None
    assert summary["counts"]["succeeded_candidates"] == 3

