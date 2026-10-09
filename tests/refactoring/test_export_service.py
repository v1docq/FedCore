import io
from pathlib import Path
import pytest
import torch
from torch import nn
pytest.importorskip("flask")
from model_exporter.api_server import create_app
from model_exporter.loader_bundle import LoaderBundle
from fedcore.external_runtime.models import save_model_bundle
from fedcore.external_runtime.client import save_dataset
from fedcore.external_runtime.security import safe_save
from torch.utils.data import DataLoader, TensorDataset

TOKEN = "a-test-service-token-with-sufficient-length"


@pytest.fixture
def service(tmp_path):
    app = create_app(storage_root=tmp_path / "service", token=TOKEN)
    app.config.update(TESTING=True)
    yield app, app.test_client(), {"Authorization": "Bearer " + TOKEN}
    app.config["JOB_RUNNER"].close()


def upload(client, headers, path, kind):
    with path.open("rb") as file:
        response = client.post("/upload", headers=headers, data={"kind": kind, "file": (file, path.name)})
    assert response.status_code == 201, response.json
    return response.json["id"]


def prepare(tmp_path, client, headers):
    model = nn.Linear(8, 2).eval()
    x = torch.randn(4, 8)
    save_model_bundle(model, tmp_path / "model.fcb")
    safe_save(x[:1], tmp_path / "example.fcb")
    save_dataset(tmp_path / "validation.fcb", x, model(x).detach())
    ids = {name: upload(client, headers, tmp_path / f"{name}.fcb", "model" if name == "model" else "example" if name == "example" else "dataset") for name in ("model", "example", "validation")}
    request = {"version": 1, "model": ids["model"], "example": ids["example"], "input_spec": {"shape": [1, 8]},
               "data": {"validation": ids["validation"]}, "task": "regression", "method": "svd"}
    return request, model, x


def test_authentication_and_all_operation_paths(service):
    app, client, headers = service
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200
    for path in ("/jobs", "/upload", "/fedcore_op", "/export", "/analyze_model"):
        assert client.post(path, json={}).status_code == 401
    assert client.post("/fedcore_op", headers=headers, json={"operation": "export_rknn"}).status_code == 400
    assert client.post("/export", headers=headers, json={"model_path": "../../somewhere.pt"}).status_code == 400
    assert client.get("/jobs/../artifact", headers=headers).status_code != 200


def test_upload_rejects_pickle_unknown_kind_and_oversize(service, tmp_path, monkeypatch):
    app, client, headers = service
    called = []
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: called.append(True))
    raw = io.BytesIO()
    torch.save(nn.Linear(2, 2), raw)
    raw.seek(0)
    response = client.post("/upload", headers=headers, data={"kind": "model", "file": (raw, "unknown.pt")})
    assert response.status_code == 400
    assert response.json["error"]["code"] == "unsafe_format"
    assert not called
    assert not list(app.config["UPLOAD_ROOT"].iterdir())
    response = client.post("/upload", headers=headers, data={"kind": "python", "file": (io.BytesIO(b"x"), "code.pt")})
    assert response.status_code == 400
    app.config["MAX_CONTENT_LENGTH"] = 1024
    response = client.post("/upload", headers=headers, data={"kind": "example", "file": (io.BytesIO(b"x"*2048), "large.fcb")})
    assert response.status_code == 413


def test_real_job_upload_status_artifact_and_restart(service, tmp_path):
    app, client, headers = service
    request, model, x = prepare(tmp_path, client, headers)
    submitted = client.post("/jobs", headers=headers, json=request)
    assert submitted.status_code == 202
    job_id = submitted.json["id"]
    job = app.config["JOB_RUNNER"].wait(job_id, 60)
    status = client.get(f"/jobs/{job_id}", headers=headers)
    assert status.json["state"] == "succeeded", status.json
    response = client.get(f"/jobs/{job_id}/artifact", headers=headers)
    assert response.status_code == 200
    torch.testing.assert_close(torch.jit.load(io.BytesIO(response.data))(x), model(x))
    app.config["JOB_RUNNER"].close()
    restarted = create_app(storage_root=app.config["STORAGE_ROOT"], token=TOKEN)
    try:
        response = restarted.test_client().get(f"/jobs/{job_id}", headers=headers)
        assert response.json["state"] == "succeeded"
        assert restarted.test_client().get(f"/jobs/{job_id}/artifact", headers=headers).status_code == 200
    finally:
        restarted.config["JOB_RUNNER"].close()


def test_explicit_multinput_path_and_roles_validation(service, tmp_path):
    app, client, headers = service
    request, _, _ = prepare(tmp_path, client, headers)
    request["input_spec"] = {"inputs": {"a": {"shape": [1, 8]}, "b": {"shape": [1, 2]}}}
    response = client.post("/jobs", headers=headers, json=request)
    assert response.status_code == 400
    request["input_spec"] = {"shape": [1, 8]}
    request["data"]["train"] = request["data"]["validation"]
    response = client.post("/jobs", headers=headers, json=request)
    assert response.status_code == 400
    assert response.json["error"]["code"] == "overlapping_data_roles"
    request["data"] = {"validation": "../outside.fcb"}
    assert client.post("/jobs", headers=headers, json=request).status_code == 400


def test_compatible_safe_loader_bundle_is_preserved(service, tmp_path):
    app, client, headers = service
    x, y = torch.randn(4, 2, 8), torch.randn(4, 3)
    path = LoaderBundle.save(tmp_path / "loader.fcb", DataLoader(TensorDataset(x, y), batch_size=2))
    with path.open("rb") as file:
        response = client.post("/upload_loader", headers=headers, data={"file": (file, path.name)})
    assert response.status_code == 201, response.json
    assert response.json["kind"] == "dataset"


def test_two_independent_profiles_and_cancel_cleanup(service, tmp_path):
    app, client, headers = service
    request, _, _ = prepare(tmp_path, client, headers)
    ids = []
    for name in ("A", "B", "A"):
        request["profile"] = {"name": name, "supported_ops": ["Gemm"] if name == "A" else []}
        response = client.post("/jobs", headers=headers, json=request)
        assert response.status_code == 202
        ids.append(response.json["id"])
    assert len(set(ids)) == 3
    cancelled = client.post(f"/jobs/{ids[-1]}/cancel", headers=headers)
    assert cancelled.status_code == 200
    assert cancelled.json["state"] == "cancelled"
    for job_id, name in zip(ids[:2], ("A", "B")):
        result = app.config["JOB_RUNNER"].wait(job_id, 60)
        assert result["state"] == "succeeded", result
        assert result["result"]["profile"]["name"] == name
    deleted = client.delete(f"/jobs/{ids[-1]}", headers=headers)
    assert deleted.status_code == 200
    assert not app.config["JOB_STORE"].directory(ids[-1]).exists()
