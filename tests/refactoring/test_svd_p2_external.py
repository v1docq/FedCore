"""Version-three wire compatibility and real CPU worker profile parity."""
import json
import os
from pathlib import Path
import sys
import tempfile

import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.execution import WeightedProfileError
from fedcore.algorithm.low_rank.method_execution import transform_method
from fedcore.algorithm.low_rank.method_specs import AFM, ASVD, Bolaco, DRONE, FLARSVD, FWSVD, SVDLLMV1, SVDLLMV2
from fedcore.external_runtime.contracts import (
    CompressionRequest, ContractError, DataRoles, InputSpec, MethodOptions,
    Resources, WeightedOptions, plan_request,
)


@pytest.fixture(autouse=True)
def four_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


def _request(**kwargs):
    return CompressionRequest("model.fcb", "example.fcb", InputSpec((1, 5), "float64"),
        DataRoles("validation.fcb", calibration="calibration.fcb"), **kwargs)


@pytest.mark.parametrize("method,spec", [("asvd", ASVD(alpha=.7)), ("fwsvd", FWSVD(loss="mse")),
    ("afm", AFM()), ("bolaco", Bolaco(group_weighting="equal_groups")), ("flar_svd", FLARSVD()),
    ("drone", DRONE()), ("svdllm_v1", SVDLLMV1(ridge=.01)), ("svdllm_v2", SVDLLMV2())])
def test_v3_method_options_roundtrip_without_changing_v1_or_v2_wire(method, spec):
    old = _request()
    weighted = _request(method="weighted_svd", version=2, weighted=WeightedOptions(), rank=2)
    for legacy in (old, weighted):
        wire = legacy.to_dict()
        assert "method_profile" not in wire
        assert ("weighted" in wire) == (legacy.version == 2)
        assert CompressionRequest.parse(json.loads(json.dumps(wire))) == legacy
    role = "group_ids" if method == "bolaco" else "observed_labels"
    new = _request(method=method, version=3,
                   method_profile=MethodOptions(spec, target_paths=("0",), calibration_target_role=role), rank=2)
    wire = json.loads(json.dumps(new.to_dict(), allow_nan=False))
    assert "weighted" not in wire and isinstance(wire["method_profile"]["spec"], dict)
    assert CompressionRequest.parse(wire) == new
    assert method in plan_request(new).steps


@pytest.mark.parametrize("method,spec,role", [("bolaco", Bolaco(), "observed_labels"),
    ("fwsvd", FWSVD(), "group_ids"), ("asvd", ASVD(), "group_ids")])
def test_external_calibration_targets_have_explicit_role(method, spec, role):
    with pytest.raises(ContractError):
        MethodOptions(spec, target_paths=("0",), calibration_target_role=role)


@pytest.mark.parametrize("corruption", ["unknown_method", "bool_version", "bool_rank", "nonfinite_alpha",
    "huge_alpha", "bad_targets", "mismatched_version", "unknown_options", "missing_calibration"])
def test_malformed_v3_request_refuses_as_contract_data(corruption):
    wire = _request(method="asvd", version=3, method_profile=MethodOptions(ASVD()), rank=2).to_dict()
    if corruption == "unknown_method":
        wire["method"] = "unknown_svd"
    elif corruption == "bool_version":
        wire["version"] = True
    elif corruption == "bool_rank":
        wire["rank"] = True
    elif corruption == "nonfinite_alpha":
        wire["method_profile"]["spec"]["alpha"] = float("nan")
    elif corruption == "huge_alpha":
        wire["method_profile"]["spec"]["alpha"] = 10**1000
    elif corruption == "bad_targets":
        wire["method_profile"]["target_paths"] = ["0", False]
    elif corruption == "mismatched_version":
        wire["version"] = 2
    elif corruption == "unknown_options":
        wire["method_profile"]["spec"]["secret_solver"] = "other"
    else:
        wire["data"]["calibration"] = None
    with pytest.raises(ContractError):
        CompressionRequest.parse(wire)


def _model_data():
    model = nn.Sequential(nn.Linear(5, 3, dtype=torch.float64)).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1., -.2, .5, 2., 1.], [-.1, 2., 1., 0., 1.],
                                           [3., .1, -.7, 1., 2.]], dtype=torch.float64))
        model[0].bias.copy_(torch.tensor([.1, -.3, .5], dtype=torch.float64))
    grid = torch.arange(75, dtype=torch.float64).reshape(15, 5)
    calibration = (grid * .19).sin() + torch.tensor([2., -1., .1, 3., -.5], dtype=torch.float64)
    validation = (grid[:8] * .23).cos() + .2
    return model, calibration, validation


@pytest.mark.parametrize("kind", ["unknown_method", "unknown_target", "missing_calibration", "missing_fisher_labels",
                                        "bad_fisher_label_dtype", "bad_bolaco_label_dtype"])
def test_client_refuses_unsupported_requests_before_visible_job(monkeypatch, tmp_path, kind):
    from fedcore.external_runtime import client
    model, calibration, validation = _model_data()
    kwargs = {"method": "asvd", "method_options": MethodOptions(ASVD(), target_paths=("0",)),
              "calibration": (calibration, model(calibration).detach())}
    if kind == "unknown_method":
        kwargs = {"method": "unknown_svd"}
    elif kind == "unknown_target":
        kwargs["method_options"] = MethodOptions(ASVD(), target_paths=("missing",))
    elif kind == "missing_calibration":
        kwargs["calibration"] = None
    elif kind == "missing_fisher_labels":
        kwargs.update(method="fwsvd", method_options=MethodOptions(FWSVD(), target_paths=("0",)),
                      calibration=(calibration, None))
    elif kind == "bad_fisher_label_dtype":
        kwargs.update(method="fwsvd", method_options=MethodOptions(FWSVD(), target_paths=("0",)),
                      calibration=(calibration, torch.ones(len(calibration), dtype=torch.float64)))
    else:
        kwargs.update(method="bolaco", method_options=MethodOptions(Bolaco(), target_paths=("0",),
                      calibration_target_role="group_ids"),
                      calibration=(calibration, torch.ones(len(calibration), dtype=torch.float64)))

    def forbidden_store(*args, **kwargs):
        raise AssertionError("Visible job storage was created before invalid-request refusal")

    monkeypatch.setattr(client, "JobStore", forbidden_store)
    with pytest.raises((ContractError, WeightedProfileError)):
        client.compress(model, validation[:1], (validation, model(validation).detach()), jobs_root=tmp_path / "jobs",
                        rank=1, **kwargs)
    assert not (tmp_path / "jobs").exists()


@pytest.mark.parametrize("method,spec", [("asvd", ASVD(alpha=.7, epsilon=.01)), ("afm", AFM())])
def test_real_cpu_worker_v3_matches_inprocess_checkpoint_and_export(monkeypatch, tmp_path, method, spec):
    from fedcore.external_runtime import client, jobs
    from fedcore.tools.registry.checkpoint_manager import CheckpointManager
    model, calibration, validation = _model_data()
    expected = transform_method(model, calibration, spec, rank=1, target_paths=("0",)).model.eval()
    manager = CheckpointManager(str(tmp_path / "checkpoint"), auto_cleanup=False)
    restored = manager.deserialize_from_bytes(manager.serialize_to_bytes(expected))
    reference_output = restored(validation).detach()
    environment = dict(os.environ)
    processes = []
    real_popen = jobs.subprocess.Popen

    def record_real_worker(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process.pid)
        return process

    # Keep source staging under the same ASCII workspace as exported artifacts.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(jobs.subprocess, "Popen", record_real_worker)
    result = client.compress(model, validation[:1], (validation, model(validation).detach()),
        calibration=(calibration, model(calibration).detach()), jobs_root=tmp_path / "jobs",
        method=method, method_options=MethodOptions(spec, target_paths=("0",), batch_size=4),
        rank=1, max_relative_error=1e6, python_executable=sys.executable,
        resources=Resources(threads=4, repetitions=2, timeout_seconds=90))
    job_dir = Path(result["job_directory"])
    log = (job_dir / "worker.log").read_text(encoding="utf-8", errors="replace")
    assert result["status"] == "succeeded", (result, log)
    assert len(processes) == 1 and processes[0] != os.getpid()
    assert dict(os.environ) == environment
    assert result["request_version"] == 3
    assert result["provenance"]["request"]["method_profile"]["spec"] == json.loads(json.dumps(result["method_transform"]["method_spec"]["options"]))
    assert result["provenance"]["calibration_sha256"]
    assert result["method_transform"]["parameters_after"] == sum(parameter.numel() for parameter in restored.parameters())
    assert result["metrics"]["measurement"]["threads"] == 4
    with (job_dir / result["artifact"]).open("rb") as stream:
        artifact = torch.jit.load(stream)
    torch.testing.assert_close(artifact(validation), reference_output, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(artifact(validation[:1]), restored(validation[:1]), rtol=1e-10, atol=1e-10)
    json.dumps(result, allow_nan=False)
