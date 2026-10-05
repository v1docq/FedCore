"""Authenticated local service over persistent, isolated v1 jobs.

Use ``create_app`` for WSGI/test clients or ``python -m model_exporter.api_server``.
Only allowlisted state_dict model bundles and tensor datasets may be uploaded.
"""
from __future__ import annotations
import argparse
import hmac
import json
import os
import re
import secrets
import uuid
from pathlib import Path
from flask import Flask, jsonify, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge, HTTPException

from fedcore.external_runtime.contracts import CompressionRequest, ContractError, DeviceProfile, InputSpec
from fedcore.external_runtime.jobs import JobRunner, JobStore
from fedcore.external_runtime.models import load_model_bundle
from fedcore.external_runtime.security import confined_path, safe_load, safe_save
from .loader_bundle import LoaderBundle
from .model_logic import ModelManager


def create_app(*, storage_root=None, token=None, python_executable=None, recover=True,
               max_bytes=64 * 1024 * 1024, max_timeout=120):
    root = Path(storage_root or os.environ.get("FEDCORE_SERVICE_ROOT", "results/service")).resolve()
    uploads = root / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    store = JobStore(root / "jobs")
    if recover:
        store.recover_interrupted()
    runner = JobRunner(store, python_executable=python_executable)
    app = Flask(__name__, template_folder="templates", static_folder="static")
    secret = token or os.environ.get("FEDCORE_API_TOKEN") or secrets.token_urlsafe(32)
    if not isinstance(secret, str) or not secret.isascii() or not 16 <= len(secret) <= 256:
        raise ValueError("Service bearer token must contain at least 16 characters")
    app.config.update(MAX_CONTENT_LENGTH=max_bytes + 16384, API_TOKEN=secret, JOB_STORE=store,
                      JOB_RUNNER=runner, STORAGE_ROOT=root, UPLOAD_ROOT=uploads)

    def upload_path(identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r"[0-9a-f]{32}\.fcb", identifier):
            raise ContractError("invalid_artifact_id", "Select an artifact id returned by /upload")
        return confined_path(uploads, identifier)

    @app.before_request
    def access_boundary():
        if request.endpoint in ("health", "index", "static"):
            return None
        supplied = request.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied, "Bearer " + secret):
            return jsonify({"error": {"code": "unauthorized", "message": "Valid bearer token required"}}), 401
        if request.is_json:
            if len(request.get_data(cache=True)) > 65536:
                raise ContractError("size_limit", "JSON request exceeds 64 KiB")
            data = request.get_json()
            if not isinstance(data, dict):
                raise ContractError("invalid_schema", "JSON requests must be objects")
            forbidden = {"model_path", "loader_path", "validation_loader_path", "export_dir", "architecture_file", "log_file", "path"}
            if set(data) & forbidden:
                raise ContractError("invalid_path", "Filesystem paths are not accepted; use uploaded artifact ids")

    @app.errorhandler(ContractError)
    def contract_error(error):
        return jsonify({"error": error.to_dict()}), 400

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        return jsonify({"error": {"code": "size_limit", "message": "Request exceeds upload byte limit"}}), 413

    @app.errorhandler(Exception)
    def unexpected(error):
        # Do not return server paths or traceback contents to clients.
        app.logger.exception("Service operation failed")
        return jsonify({"error": {"code": "service_error", "message": "Service operation failed"}}), 500

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify({"error": {"code": "http_error", "message": error.name}}), error.code

    @app.get("/health")
    def health():
        return jsonify({"status": "healthy", "contract_version": 1})

    @app.get("/")
    @app.get("/webui")
    def index():
        return send_file(Path(__file__).parent / "templates" / "web_ui.html")

    @app.post("/upload")
    @app.post("/upload_loader")
    def upload():
        kind = "loader" if request.path == "/upload_loader" else request.form.get("kind")
        if kind not in ("model", "dataset", "example", "loader"):
            raise ContractError("unsupported_kind", "Declare model, dataset, example or loader before upload")
        file = request.files.get("file")
        if file is None or not file.filename:
            raise ContractError("missing_file", "A tensor archive file is required")
        identifier = uuid.uuid4().hex + ".fcb"
        path = confined_path(uploads, identifier, must_exist=False)
        total = 0
        try:
            with path.open("wb") as output:
                while True:
                    chunk = file.stream.read(65536)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ContractError("size_limit", "Uploaded artifact exceeds byte limit")
                    output.write(chunk)
            if kind == "model":
                load_model_bundle(path, max_bytes)
            elif kind == "loader":
                bundle = LoaderBundle.load(path)
                # Convert legacy tensor loader schema to the v1 explicit dataset schema.
                import torch
                safe_save({"kind": "fedcore_tensor_dataset", "version": 1,
                            "features": bundle["features"], "targets": bundle["targets"]}, path)
                kind = "dataset"
            else:
                value = safe_load(path, max_bytes)
                import torch
                if kind == "example" and type(value) is not torch.Tensor:
                    raise ContractError("invalid_example", "Example archive must contain one tensor")
                if kind == "dataset":
                    if not isinstance(value, dict) or set(value) != {"kind", "version", "features", "targets"} or value["kind"] != "fedcore_tensor_dataset" or type(value["version"]) is not int or value["version"] != 1:
                        raise ContractError("invalid_dataset", "Invalid versioned tensor dataset")
                    x, y = value["features"], value["targets"]
                    if type(x) is not torch.Tensor or type(y) is not torch.Tensor or not x.ndim or not y.ndim or x.shape[0] != y.shape[0] or not len(x):
                        raise ContractError("invalid_dataset", "Features and targets need aligned nonempty sample axes")
            return jsonify({"id": identifier, "filename": identifier, "kind": kind, "size_bytes": path.stat().st_size}), 201
        except Exception:
            path.unlink(missing_ok=True)
            raise

    @app.get("/files")
    @app.get("/loaders")
    def files():
        return jsonify({"files": [{"id": p.name, "size_bytes": p.stat().st_size} for p in sorted(uploads.glob("*.fcb"))]})

    def enqueue(payload):
        contract = CompressionRequest.parse(payload)
        if contract.resources.max_bytes > max_bytes or contract.resources.timeout_seconds > max_timeout:
            raise ContractError("resource_limit", "Requested resources exceed service limits")
        for name in (contract.model, contract.example, contract.data.validation, contract.data.train, contract.data.calibration):
            if name is not None:
                upload_path(name)
        job_id = runner.submit(contract, uploads)
        return jsonify({"id": job_id, "state": "queued", "status_url": f"/jobs/{job_id}"}), 202

    @app.post("/jobs")
    def create_job():
        return enqueue(request.get_json())

    @app.get("/jobs/<job_id>")
    def get_job(job_id):
        value = store.get(job_id)
        # Result paths are job-relative; caller can only download this job's artifact.
        return jsonify(value)

    @app.post("/jobs/<job_id>/cancel")
    def cancel_job(job_id):
        runner.cancel(job_id)
        return jsonify(store.get(job_id))

    @app.delete("/jobs/<job_id>")
    def delete_job(job_id):
        store.delete(job_id)
        return jsonify({"id": job_id, "state": "deleted"})

    @app.get("/jobs/<job_id>/artifact")
    def artifact(job_id):
        job = store.get(job_id)
        if job["state"] != "succeeded":
            raise ContractError("artifact_unavailable", "Job has no successful artifact")
        return send_file(confined_path(store.directory(job_id), job["result"]["artifact"]), as_attachment=True)

    @app.post("/export")
    def export():
        payload = dict(request.get_json())
        payload["method"] = "export"
        return enqueue(payload)

    @app.post("/fedcore_op")
    def operation():
        payload = dict(request.get_json())
        selected = payload.pop("operation", None)
        allowed = {"low_rank": "svd", "export_torchscript": "export", "export_onnx": "export"}
        if selected not in allowed:
            raise ContractError("unsupported_operation", "Operation is outside the enabled service allowlist")
        payload["method"] = allowed[selected]
        if selected.startswith("export_"):
            payload["artifact_format"] = selected.removeprefix("export_")
        return enqueue(payload)

    @app.post("/analyze_model")
    def analyze():
        payload = request.get_json()
        if set(payload) - {"model_id", "example_id", "input_spec", "profile"}:
            raise ContractError("invalid_schema", "Unexpected analysis fields")
        model = load_model_bundle(upload_path(payload.get("model_id")), max_bytes)
        example = safe_load(upload_path(payload.get("example_id")), max_bytes)
        InputSpec.parse(payload.get("input_spec")).validate_tensor(example)
        profile = DeviceProfile.parse(payload.get("profile", {}))
        value = ModelManager(profile).analyze_model(model, example)
        return jsonify(value), 400 if "error" in value else 200

    @app.get("/architectures")
    def architectures():
        return jsonify(ModelManager().get_architectures())

    @app.post("/export_parts")
    def unsupported_legacy_partition():
        raise ContractError("unsupported_service_operation", "Partition export is a local ModelManager operation requiring actual intermediate inputs")

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-root", default="results/service")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", choices=("127.0.0.1", "0.0.0.0"), default="127.0.0.1")
    args = parser.parse_args()
    token = os.environ.get("FEDCORE_API_TOKEN")
    if not token:
        raise SystemExit("Set FEDCORE_API_TOKEN (at least 16 characters) before starting the local service")
    app = create_app(storage_root=args.storage_root, token=token)
    try:
        app.run(host=args.host, port=args.port, debug=False)
    finally:
        app.config["JOB_RUNNER"].close()


if __name__ == "__main__":
    main()
