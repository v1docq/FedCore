import json
import os
from pathlib import Path
import sys
import pytest
import torch
from torch import nn
from fedcore.experiments.external_benchmark import benchmark_external, compare_decomposers, benchmark_extension_manifest


@pytest.mark.parametrize("rank", [1, 3])
def test_native_tdecomp_same_actual_rank_and_full_rank(rank):
    matrix = torch.diag(torch.tensor([7., 3., 1.], dtype=torch.float64))
    report = compare_decomposers(matrix, rank=rank)
    assert report["native_actual_rank"] == rank == report["tdecomp_actual_rank"]
    assert report["reconstruction_agreement"]["relative"] < 1e-12
    if rank == 3:
        assert report["native_error"]["relative"] < 1e-12
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("model,x", [(nn.Linear(4, 2), torch.randn(6, 4)),
    (nn.Conv1d(2, 3, 3, padding=1), torch.randn(6, 2, 8))])
def test_real_external_process_matches_local_artifact_and_keeps_environment(model, x, tmp_path):
    model.eval()
    environment, threads = dict(os.environ), torch.get_num_threads()
    y = model(x).detach()
    report = benchmark_external(model, x[:1], (x, y), tmp_path, worker_python=sys.executable, repetitions=1)
    assert report["two_environment_status"] == "same_environment_separate_process"
    call = report["external_calls"][0]
    assert call["status"] == "succeeded" and call["input_hashes_equal"]
    assert call["output_agreement"]["relative"] < 1e-5
    assert call["artifact_bytes"] > 0
    assert call["external_total_seconds"] >= call["result"]["timings"]["execute_plan_seconds"]
    assert call["result"]["timings"]["model_load_seconds"] > 0
    assert report["persistent_external_inference"]["status"] == "unsupported"
    assert dict(os.environ) == environment and torch.get_num_threads() == threads


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_actual_two_interpreters_and_extension_manifest_opt_in(tmp_path, task):
    modern = os.environ.get("FEDCORE_MODERN_FEDOT_ROOT")
    caller = os.environ.get("FEDCORE_MODERN_FEDOT_PYTHON")
    if not modern or not caller:
        pytest.skip("Requires explicit modern FEDOT caller and separate compression Python")
    from fedcore.external_runtime.models import save_model_bundle
    from fedcore.external_runtime.client import save_dataset
    model = nn.Linear(4, 2).eval()
    x = torch.randn(6, 4)
    scores = model(x).detach()
    y = scores if task == "regression" else scores.argmax(1)
    save_model_bundle(model, tmp_path / "model.fcb")
    save_dataset(tmp_path / "validation.fcb", x, y)
    result = benchmark_extension_manifest(modern_fedot_root=modern, caller_python=caller, worker_python=sys.executable,
        model_bundle=tmp_path / "model.fcb", validation_bundle=tmp_path / "validation.fcb", features=x, targets=y, output_dir=tmp_path / "manifest", task=task)
    assert result["status"] == "completed" and result["registry_unchanged"]
    assert Path(result["caller_python"]).resolve() != Path(result["worker_python"]).resolve()
    assert result["catalog_environment_restored"] and result["caller_environment_unchanged"]


def test_trained_external_adapter_real_unicode_workspace(tmp_path):
    from fedcore.external_runtime.adapters import CompressedTensorEstimator
    from fedcore.external_runtime.models import save_model_bundle
    from fedcore.external_runtime.client import save_dataset
    directory = tmp_path / "Исследование модели"
    directory.mkdir()
    model = nn.Linear(4, 2)
    train_x, train_y = torch.randn(8, 4), torch.randn(8, 2)
    before = model.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(3):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(train_x), train_y)
        loss.backward()
        optimizer.step()
    assert not torch.equal(before, model.weight)
    model.eval()
    x = torch.randn(6, 4)
    y = model(x).detach()
    save_model_bundle(model, directory / "Модель.fcb")
    save_dataset(directory / "Проверка.fcb", x, y)
    estimator = CompressedTensorEstimator({"model_bundle": directory / "Модель.fcb", "validation_bundle": directory / "Проверка.fcb",
        "jobs_root": directory / "Задания", "python_executable": sys.executable}, "regression")
    estimator.fit(train_x.numpy(), train_y.numpy())
    torch.testing.assert_close(torch.from_numpy(estimator.predict(x.numpy())), y, rtol=1e-4, atol=1e-4)
    assert estimator.result["status"] == "succeeded"
