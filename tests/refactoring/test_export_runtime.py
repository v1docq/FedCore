import json
import os
import sys
import time
from dataclasses import FrozenInstanceError
from pathlib import Path
import torch
from torch import nn
import pytest
from hypothesis import given, strategies as st
from fedcore.external_runtime.client import compress, save_dataset
from fedcore.external_runtime.contracts import CompressionRequest, ContractError, InputSpec, DataRoles, DeviceProfile, Resources, plan_request
from fedcore.external_runtime.jobs import JobStore, JobRunner, JobState, transition
from fedcore.external_runtime.models import load_model_bundle, save_model_bundle, build_model
from fedcore.external_runtime.security import safe_load, safe_save, confined_path
from fedcore.external_runtime.adapters import decompose_matrix_with_tdecomp, CompressedTensorEstimator


class UnknownObject:
    def __init__(self):
        self.value = 1


def test_tensor_only_model_factory_round_trip_and_unknown_type(tmp_path):
    model = nn.Sequential(nn.Conv1d(2, 4, 3), nn.ReLU(), nn.Flatten(), nn.Linear(24, 2)).double()
    path = save_model_bundle(model, tmp_path / "model.fcb")
    restored = load_model_bundle(path)
    x = torch.randn(2, 2, 8, dtype=torch.float64)
    torch.testing.assert_close(restored(x), model(x))
    torch.save(UnknownObject(), tmp_path / "unknown.fcb")
    with pytest.raises(ContractError) as error:
        safe_load(tmp_path / "unknown.fcb")
    assert error.value.code == "unsafe_format"
    torch.save(model, tmp_path / "full_model.fcb")
    with pytest.raises(ContractError):
        load_model_bundle(tmp_path / "full_model.fcb")


@pytest.mark.parametrize("path", ["../x.fcb", "/etc/x.fcb", "C:/x.fcb", "..\\x.fcb", "x/../y.fcb", "./x.fcb"])
def test_job_paths_are_confined(tmp_path, path):
    with pytest.raises(ContractError):
        confined_path(tmp_path, path, must_exist=False)


def test_size_and_factory_boundaries(tmp_path):
    safe_save(torch.ones(4096), tmp_path / "large.fcb")
    with pytest.raises(ContractError) as error:
        safe_load(tmp_path / "large.fcb", 1024)
    assert error.value.code == "size_limit"
    descriptor = {"type": "Linear", "config": {"in_features": 100000, "out_features": 100000, "bias": True}, "dtype": "float32"}
    with pytest.raises(ContractError, match="allocation"):
        build_model(descriptor)
    with pytest.raises(ContractError):
        build_model({"type": "UserModel", "config": {}, "dtype": "float32"})


@given(st.lists(st.integers(1, 12), min_size=1, max_size=4), st.sampled_from(["float32", "float64"]))
def test_request_round_trip_is_immutable_and_pure(shape, dtype):
    request = CompressionRequest("model.fcb", "example.fcb", InputSpec(tuple(shape), dtype), DataRoles("validation.fcb", "train.fcb"))
    original = request.to_dict()
    decoded = CompressionRequest.parse(json.loads(json.dumps(original)))
    assert decoded == request
    assert plan_request(decoded).request == request
    assert request.to_dict() == original
    with pytest.raises(FrozenInstanceError):
        request.task = "classification"


def test_roles_multinput_and_invalid_combinations():
    with pytest.raises(ContractError):
        DataRoles("validation.fcb", "validation.fcb")
    with pytest.raises(ContractError):
        DataRoles(None)
    with pytest.raises(ContractError):
        InputSpec.parse({"inputs": {"a": [1, 3], "b": [1, 2]}})
    with pytest.raises(ContractError):
        CompressionRequest("m.fcb", "e.fcb", InputSpec((1, 8)), DataRoles("v.fcb"), rank=1, retained_energy=.5)


@pytest.mark.parametrize("kind,groups,padding_mode", [("linear", 1, "zeros"), ("conv1d", 1, "zeros"), ("conv1d", 2, "reflect")])
def test_real_subprocess_compression_parity_and_measurements(tmp_path, kind, groups, padding_mode):
    if kind == "linear":
        model = nn.Linear(8, 8)
        model.weight.data.copy_(torch.arange(1, 9, dtype=torch.float32).view(8, 1) @ torch.linspace(.1, .8, 8).view(1, 8))
        x = torch.randn(6, 8)
    else:
        model = nn.Conv1d(4, 8, 3, stride=2, padding=2, dilation=2, groups=groups, padding_mode=padding_mode)
        for group in range(groups):
            out = 8 // groups
            matrix = torch.arange(1, out+1, dtype=torch.float32).view(out, 1) @ torch.linspace(.1, 1., 4//groups*3).view(1, -1)
            model.weight.data[group*out:(group+1)*out].copy_(matrix.reshape(out, 4//groups, 3))
        x = torch.randn(6, 4, 16)
    model.eval()
    targets = model(x).detach()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    environment, modules, import_path = os.environ.copy(), set(sys.modules), list(sys.path)
    result = compress(model, x[:1], (x, targets), train=(x+1, targets), jobs_root=tmp_path / "jobs", rank=1, max_relative_error=1e-4)
    assert result["status"] == "succeeded", result
    artifact = Path(result["job_directory"]) / result["artifact"]
    loaded = torch.jit.load(str(artifact))
    torch.testing.assert_close(loaded(x), targets, rtol=1e-4, atol=1e-4)
    assert result["metrics"]["compressed"]["parameters"] < result["metrics"]["baseline"]["parameters"]
    assert result["metrics"]["measurement"]["energy"]["status"] == "unsupported"
    assert result["metrics"]["measurement"]["validation_samples"] == 6
    assert result["provenance"]["request"]["data"]["train"] != result["provenance"]["request"]["data"]["validation"]
    assert os.environ == environment and sys.path == import_path
    assert not any(name.startswith("fedot") for name in set(sys.modules)-modules)
    for key in state:
        torch.testing.assert_close(state[key], model.state_dict()[key])


def test_error_budget_and_invalid_shape_fail_honestly(tmp_path):
    model = nn.Linear(8, 8)
    model.weight.data.copy_(torch.eye(8))
    x = torch.eye(8)
    result = compress(model, x[:1], (x, x), jobs_root=tmp_path, rank=1, max_relative_error=1e-4)
    assert result["status"] == "failed"
    assert result["error"]["code"] == "error_budget_exceeded"
    assert not list(Path(result["job_directory"]).glob("compressed.*"))


def test_terminal_state_law_and_restart_persistence(tmp_path):
    request = CompressionRequest("m.fcb", "e.fcb", InputSpec((1, 3)), DataRoles("v.fcb"))
    store = JobStore(tmp_path)
    completed = store.create(request)
    store.advance(completed, JobState.RUNNING)
    store.advance(completed, JobState.SUCCEEDED, {"status": "succeeded", "artifact": "compressed.fcb"})
    interrupted = store.create(request)
    store.advance(interrupted, JobState.RUNNING)
    restarted = JobStore(tmp_path)
    restarted.recover_interrupted()
    assert restarted.get(completed)["state"] == "succeeded"
    assert restarted.get(interrupted)["result"]["error"]["code"] == "interrupted"
    for terminal in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED):
        for event in JobState:
            with pytest.raises(ContractError):
                transition(terminal, event)


def test_actual_cancel_and_timeout_cleanup(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    model = nn.Linear(8, 8)
    x = torch.zeros(2, 8)
    save_model_bundle(model, source / "m.fcb")
    safe_save(x[:1], source / "e.fcb")
    save_dataset(source / "v.fcb", x, model(x).detach())
    store = JobStore(tmp_path / "jobs")
    runner = JobRunner(store, workers=1)
    try:
        request = CompressionRequest("m.fcb", "e.fcb", InputSpec((1, 8)), DataRoles("v.fcb"))
        first = runner.submit(request, source)
        second = runner.submit(request, source)
        runner.cancel(second)
        assert runner.wait(second)["state"] == "cancelled"
        runner.cancel(first)
        assert runner.wait(first)["state"] == "cancelled"
        assert not list(store.directory(first).glob("compressed.*"))
        timeout = runner.submit(CompressionRequest("m.fcb", "e.fcb", InputSpec((1, 8)), DataRoles("v.fcb"), resources=Resources(timeout_seconds=.01)), source)
        failure = runner.wait(timeout, 10)
        assert failure["state"] == "failed"
        assert failure["result"]["error"]["code"] == "timeout"
    finally:
        runner.close()


def test_tdecomp_adapter_uses_real_decomposer():
    matrix = torch.diag(torch.tensor([4., 2., 1.]))
    result = decompose_matrix_with_tdecomp(matrix, rank=2)
    torch.testing.assert_close(result["reconstruction"], torch.diag(torch.tensor([4., 2., 0.])))
    assert result["relative_error"] == pytest.approx(1 / 21**.5)


def test_closing_runner_cancels_its_queued_jobs_but_not_other_owners(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    model = nn.Linear(8, 8)
    x = torch.zeros(2, 8)
    save_model_bundle(model, source / "m.fcb")
    safe_save(x[:1], source / "e.fcb")
    save_dataset(source / "v.fcb", x, model(x).detach())
    request = CompressionRequest("m.fcb", "e.fcb", InputSpec((1, 8)), DataRoles("v.fcb"))
    store = JobStore(tmp_path / "jobs")
    unrelated = store.create(request)
    runner = JobRunner(store, workers=1)
    owned = [runner.submit(request, source) for _ in range(3)]
    runner.close()
    assert all(store.get(job)["state"] == "cancelled" for job in owned)
    assert store.get(unrelated)["state"] == "queued"


def test_industrial_estimator_smoke_preserves_regression_tensor_layout(tmp_path):
    model = nn.Conv1d(2, 3, 3)
    x = torch.randn(4, 2, 8)
    y = model(x).detach()
    save_model_bundle(model, tmp_path / "m.fcb")
    save_dataset(tmp_path / "v.fcb", x, y)
    adapter = CompressedTensorEstimator({"model_bundle": str(tmp_path / "m.fcb"), "validation_bundle": str(tmp_path / "v.fcb"), "jobs_root": str(tmp_path / "jobs")}, "regression")
    adapter.fit(x.numpy(), y.numpy())
    torch.testing.assert_close(torch.from_numpy(adapter.predict(x.numpy())), y)
