"""Pilot admission, immutable provenance and non-CUDA dry-run boundaries."""
import builtins
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext

import pytest

from fedcore.experiments import cv_sparse_pilot as pilot
from fedcore.experiments import cv_sparse_runtime as runtime
from fedcore.experiments.cv_sparse_protocol import PilotProtocol, eligible_finalists


def prepared_data(root):
    folder = root / "cifar-10-batches-py"
    folder.mkdir(parents=True)
    for name in ("batches.meta", "test_batch", *(f"data_batch_{index}" for index in range(1, 6))):
        (folder / name).write_bytes(b"prepared-file-placeholder")


def test_dry_run_has_no_cuda_ml_dataset_or_output_creation(tmp_path, monkeypatch, capsys):
    prepared_data(tmp_path / "data")
    monkeypatch.setattr(runtime, "source_manifest", lambda *args: {"verified": True})
    def forbidden(*args, **kwargs):
        pytest.fail("Dry-run crossed the numerical/runtime admission boundary")
    monkeypatch.setattr(pilot, "Runtime", forbidden)
    monkeypatch.setattr(pilot, "admit_free_gpu", forbidden)
    original = builtins.__import__
    def checked_import(name, *args, **kwargs):
        if name == "torch" or name.startswith(("torch.", "torchvision", "fedcore.algorithm")):
            forbidden()
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", checked_import)
    output = tmp_path / "new-output"
    result = pilot.main(["--data-root", str(tmp_path / "data"), "--output-root", str(output),
                         "--source-manifest", "not-read-by-test.json"])
    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["execute"] is False and payload["status"] == "prepared"
    assert not output.exists()


def test_existing_output_refuses_before_any_source_or_device_effect(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(runtime, "source_manifest", lambda *args: pytest.fail("Source touched after overwrite refusal"))
    with pytest.raises(FileExistsError):
        runtime.preparation(tmp_path, output, tmp_path / "source.json", PilotProtocol())


@pytest.mark.parametrize("external,expected", [(None, 14500.0), ("14000", 14000.0), ("99999", 14500.0)])
def test_supervisor_deadline_can_only_narrow_the_slot(external, expected):
    assert runtime.execution_deadline(100.0, PilotProtocol(), external) == expected


@pytest.mark.parametrize("external", ["nan", "inf", "-inf", "100", "-1", "invalid"])
def test_invalid_or_expired_external_deadline_is_refused(external):
    with pytest.raises(ValueError):
        runtime.execution_deadline(100.0, PilotProtocol(), external)


def test_gpu_admission_checks_compute_processes_before_inventory(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="12345\n", stderr="", returncode=0)
    monkeypatch.setattr(runtime.subprocess, "run", run)
    with pytest.raises(ValueError, match="zero existing compute"):
        runtime.admit_free_gpu()
    assert len(calls) == 1 and "--query-compute-apps=pid" in calls[0]


def test_free_gpu_admission_is_readonly_and_records_inventory(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="" if len(calls) == 1 else "0, UUID, A100, 40960, 0\n")
    monkeypatch.setattr(runtime.subprocess, "run", run)
    result = runtime.admit_free_gpu()
    assert result["compute_pids"] == []
    assert "A100" in result["gpu_inventory"]
    assert len(calls) == 2


def test_output_is_atomic_append_only_and_confined(tmp_path):
    output = runtime.Output(tmp_path / "new")
    output.json("nested/result.json", {"status": "succeeded"})
    output.append({"event": "first"})
    output.append({"event": "second"})
    assert json.loads((output.root / "nested/result.json").read_text())["status"] == "succeeded"
    assert [json.loads(line)["event"] for line in output.ledger.read_text().splitlines()] == ["first", "second"]
    assert not list(output.root.rglob("*.tmp"))
    with pytest.raises(ValueError):
        output.json("../escape.json", {})
    with pytest.raises(FileExistsError):
        runtime.Output(output.root)


def fake_package_manifest(tmp_path, monkeypatch):
    import fedcore
    root = tmp_path / "installed"
    names = {"fedcore/experiments/" + name + ".py" for name in
        ("cv_sparse_pilot", "cv_sparse_protocol", "cv_sparse_data", "cv_sparse_runtime", "cv_sparse_supervision")}
    names |= {"fedcore/algorithm/low_rank/" + name + ".py" for name in ("cv_grammar", "cv_execution", "grammar_search", "low_rank_opt")}
    names |= {"fedcore/algorithm/pruning/sparse/cv_channels.py", "fedcore/algorithm/pruning/sparse/cv_checkpoint.py",
              "fedcore/tools/registry/checkpoint_manager.py", "fedcore/interfaces/fedcore_optimizer.py", "fedcore/__init__.py"}
    hashes = {}
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixed package source\n", encoding="utf-8")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(fedcore, "__file__", str(root / "fedcore/__init__.py"))
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps({"source_commit": "a" * 40, "wheel_sha256": "b" * 64,
        "package_files_sha256": hashes, "protocol_sha256": PilotProtocol().sha256}), encoding="utf-8")
    return root, manifest


def test_installed_wheel_manifest_verifies_real_files_without_git(tmp_path, monkeypatch):
    root, manifest = fake_package_manifest(tmp_path, monkeypatch)
    result = runtime.source_manifest(manifest, PilotProtocol())
    assert result["git"] is None
    assert result["package_root"] == str(root)
    assert result["verified_package_files_sha256"]
    (root / "fedcore/algorithm/low_rank/cv_execution.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="differs"):
        runtime.source_manifest(manifest, PilotProtocol())


def test_old_or_incomplete_wheel_manifest_is_not_an_authority(tmp_path, monkeypatch):
    _root, manifest = fake_package_manifest(tmp_path, monkeypatch)
    payload = json.loads(manifest.read_text())
    del payload["package_files_sha256"]["fedcore/experiments/cv_sparse_pilot.py"]
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="complete current CV pilot"):
        runtime.source_manifest(manifest, PilotProtocol())


def test_menu_prefix_and_target_budget_are_explicit_metadata():
    menu = pilot.cv_menu()
    prefix = pilot.fixed_prefix(menu)
    assert len(menu) == 8 and all(len(choices) == 7 for choices in menu.values())
    assert len(prefix) == 4 and len({pilot.solution_id(genotype) for genotype in prefix}) == 4
    assert all(len(genotype.genes) == 8 for genotype in prefix)
    records = [{"status": "succeeded", "completed_within_deadline": True, "quality_accepted": True,
        "tensor_bytes": 80, "cross_entropy": 1.0, "solution_id": "a"},
        {"status": "succeeded", "completed_within_deadline": False, "quality_accepted": True,
        "tensor_bytes": 1, "cross_entropy": .01, "solution_id": "late"},
        {"status": "succeeded", "completed_within_deadline": True, "quality_accepted": True,
        "tensor_bytes": 81, "cross_entropy": .001, "solution_id": "over"}]
    assert [row["solution_id"] for row in eligible_finalists(records, byte_ceiling=80)] == ["a"]


def test_aa_refuses_a_shared_module_reference_before_measurement():
    same = object()
    with pytest.raises(ValueError, match="independently loaded"):
        runtime.paired_latency(same, same, None, deadline=1000, label="invalid", aa=True)


def test_timed_call_block_contains_no_deadline_vram_or_heartbeat_instrumentation():
    state = SimpleNamespace(timed=False, first_calls=0, second_calls=0)
    class Event:
        def __init__(self, **kwargs):
            pass
        def record(self):
            state.timed = not state.timed
        def synchronize(self):
            assert not state.timed
        def elapsed_time(self, other):
            return 1.0
    def check(*args, **kwargs):
        assert not state.timed, "Deadline/VRAM/heartbeat instrumentation entered the measured model span"
    def first(value):
        state.first_calls += 1
    def second(value):
        state.second_calls += 1
    torch = SimpleNamespace(inference_mode=nullcontext,
        cuda=SimpleNamespace(synchronize=lambda: None, Event=Event))
    output = SimpleNamespace(append=lambda *args: None, json=lambda *args: None)
    gpu = SimpleNamespace(torch=torch, runtime=SimpleNamespace(torch=torch, protocol=PilotProtocol(), check=check, output=output),
        data=SimpleNamespace(roles=SimpleNamespace(calibration=tuple(range(1000))),
            rows=lambda *args: SimpleNamespace(to=lambda *args: object())))
    result = runtime.paired_latency(first, second, gpu, deadline=1000, label="effect-boundary", aa=True)
    assert state.first_calls == state.second_calls == 2 * (20 + 20 * 20)
    assert len(result["batches"]["1"]["pairs"]) == len(result["batches"]["128"]["pairs"]) == 20
    assert not state.timed


def selected_artifacts(output):
    protocol = PilotProtocol()
    artifacts = {}
    for index, phase in enumerate(("dense_original", "recovered_200_updates")):
        record = output.bytes(f"checkpoints/selected_{index}.pt", b"fixed verified checkpoint " + bytes([index]))
        solution = "a" * 64
        record.update(phase=phase, solution_id=solution, tensor_bytes=100, completed_within_deadline=True)
        record["artifact_id"] = runtime.digest({"solution_id": solution, "checkpoint_sha256": record["sha256"], "phase": phase})
        if index == 0:
            record.update(training={"completed_epochs": 80, "all_parameters_received_finite_gradients": True,
                                    "all_parameter_tensors_changed": True}, validation={"top1": .91})
        else:
            record.update(recovery_updates=200, recovery_tape_sha256="b" * 64, roles=["dense_recovery"])
        artifacts[record["artifact_id"]] = record
    return protocol, artifacts


def test_fresh_selected_artifact_admission_accepts_complete_distinct_dense_versions(tmp_path):
    output = runtime.Output(tmp_path / "out")
    protocol, artifacts = selected_artifacts(output)
    admitted = runtime.validate_selected_artifacts(output, artifacts, protocol)
    assert admitted["artifact_count"] == 2
    assert admitted["roles"] == ["dense_recovery"]
    assert len(admitted["admission_sha256"]) == 64


@pytest.mark.parametrize("kind", ["corrupt_bytes", "missing", "outside", "wrong_size", "forged_identity", "incomplete_recovery", "late"])
def test_corrupt_or_incomplete_artifact_is_refused_before_seal_or_test_open(tmp_path, kind):
    output = runtime.Output(tmp_path / "out")
    protocol, artifacts = selected_artifacts(output)
    key = next(key for key, value in artifacts.items() if value["phase"] == "recovered_200_updates")
    artifact = artifacts[key]
    target = output.root / artifact["path"]
    if kind == "corrupt_bytes":
        original = target.read_bytes()
        target.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    elif kind == "missing":
        target.unlink()
    elif kind == "outside":
        artifact["path"] = "../outside.pt"
    elif kind == "wrong_size":
        artifact["file_bytes"] += 1
    elif kind == "forged_identity":
        artifact["solution_id"] = "c" * 64
    elif kind == "incomplete_recovery":
        artifact["recovery_updates"] = 199
    else:
        artifact["completed_within_deadline"] = False
    with pytest.raises(ValueError):
        runtime.validate_selected_artifacts(output, artifacts, protocol)
    assert not (output.root / "frozen_selection.json").exists()


@pytest.mark.parametrize("code", ["ResourceLimitExceeded", "EvaluationDeadline"])
def test_only_exact_declared_resource_failures_can_reject_a_candidate(code):
    from fedcore.algorithm.low_rank.grammar_execution import ResourceFailure
    failure = ResourceFailure(code, "known bounded resource refusal")
    assert runtime.classify_cv_candidate_failure(failure) == "candidate_failure"
    class Subclass(ResourceFailure):
        pass
    assert runtime.classify_cv_candidate_failure(Subclass(code, "unknown subtype")) == "fatal"


@pytest.mark.parametrize("code", ["RepresentationMismatch", "CheckpointMismatch", "SourceMutation",
    "StaleSourceContext", "InvalidWorkflow", "WorkflowBindingMismatch", "UnknownIntegrityCode"])
def test_all_structural_and_validation_failures_abort_the_prevalidated_pilot(code):
    from fedcore.algorithm.low_rank.grammar_execution import StructuralInvalid
    from fedcore.algorithm.low_rank.plans import ValidationFailure, Violation
    violations = (Violation(code, "model", "original integrity evidence"),)
    assert runtime.classify_cv_candidate_failure(StructuralInvalid(violations)) == "fatal"
    assert runtime.classify_cv_candidate_failure(ValidationFailure(violations)) == "fatal"


@pytest.mark.parametrize("code", ["CVExecutionFailed", "SolverFailure", "NonfiniteResult", "UnknownNumericalCode"])
def test_numerical_failures_have_no_admitted_solver_recovery_boundary(code):
    from fedcore.algorithm.low_rank.grammar_execution import NumericalFailure, ResourceFailure
    assert runtime.classify_cv_candidate_failure(NumericalFailure(code, "original failure")) == "fatal"
    assert runtime.classify_cv_candidate_failure(ResourceFailure(code, "wrong or unknown resource code")) == "fatal"
    assert runtime.classify_cv_candidate_failure(SimpleNamespace(code=code)) == "fatal"


def test_integrity_failure_keeps_charged_attempt_and_original_metadata_without_gpu_effects(tmp_path, monkeypatch):
    import time
    from fedcore.algorithm.low_rank.grammar_execution import StructuralInvalid
    from fedcore.algorithm.low_rank.plans import Violation
    failure = StructuralInvalid((Violation("CheckpointMismatch", "model", "exact original checkpoint evidence"),))
    deadline = time.perf_counter() + 60
    output = runtime.Output(tmp_path / "out")
    owner = SimpleNamespace(records={}, genotypes={}, output=output,
        runtime=SimpleNamespace(check=lambda *args, **kwargs: None, deadline=deadline, stage_limit=deadline),
        build=lambda genotype: failure)
    def forbidden(*args, **kwargs):
        pytest.fail("A fatal CPU integrity outcome reached GPU calibration or evaluation")
    monkeypatch.setattr(pilot, "calibrate_bn", forbidden)
    monkeypatch.setattr(pilot, "evaluate", forbidden)
    genotype = pilot.fixed_prefix(pilot.cv_menu())[0]
    with pytest.raises(runtime.PilotFatal, match="CheckpointMismatch"):
        pilot.Pilot.short(owner, genotype, "integrity-test", deadline)
    assert len(owner.records["integrity-test"]) == 1
    record = owner.records["integrity-test"][0]
    assert record["charged_attempt"] is True and record["attempt_id"] == 1
    assert record["status"] == record["failure_disposition"] == "fatal"
    assert record["failure_type"] == "StructuralInvalid"
    assert record["failure"]["violations"][0]["code"] == "CheckpointMismatch"
    assert record["failure"]["violations"][0]["message"] == "exact original checkpoint evidence"
    durable = json.loads((output.root / "search/integrity-test_attempts.json").read_text())
    assert durable[0]["status"] == "fatal" and durable[0]["charged_attempt"] is True
