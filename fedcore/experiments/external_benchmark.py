"""Real .fcb/JobRunner benchmark, preserving FedCore -> tdecomp dependency."""
from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch

from fedcore.external_runtime.client import save_dataset
from fedcore.external_runtime.contracts import CompressionRequest, DataRoles, InputSpec, Resources, plan_request
from fedcore.external_runtime.execution import execute_plan
from fedcore.external_runtime.jobs import JobRunner, JobStore
from fedcore.external_runtime.models import save_model_bundle
from fedcore.external_runtime.security import safe_save
from .math_checks import relative_error, truncated_svd


def compare_decomposers(matrix, *, rank):
    from fedcore.external_runtime.adapters import decompose_matrix_with_tdecomp
    start = time.perf_counter()
    native = truncated_svd(matrix, rank)
    native_seconds = time.perf_counter() - start
    start = time.perf_counter()
    other = decompose_matrix_with_tdecomp(matrix, rank=rank)
    tdecomp_seconds = time.perf_counter() - start
    return {"requested_rank": rank, "native_actual_rank": int(torch.linalg.matrix_rank(native)),
            "tdecomp_actual_rank": int(torch.linalg.matrix_rank(other["reconstruction"])),
            "native_error": relative_error(matrix, native), "tdecomp_error": relative_error(matrix, other["reconstruction"]),
            "reconstruction_agreement": relative_error(native, other["reconstruction"]),
            "native_seconds": native_seconds, "tdecomp_seconds": tdecomp_seconds,
            "tdecomp_version": importlib.metadata.version("tdecomp"), "device": str(matrix.device), "dtype": str(matrix.dtype),
            "scope": "single_real_call; no speed superiority claim", "dependency_direction": "FedCore -> tdecomp"}


def benchmark_external(model, example, validation, output_dir, *, task="regression", rank=None,
                       retained_energy=1.0, max_relative_error=1e-4, worker_python=None,
                       repetitions=2, resources=None):
    """Same persisted input, request, exporter and factorizer locally/in worker.

    Existing JobRunner starts a fresh process per job. Repeated external jobs
    therefore remain cold starts. Warm loaded-artifact inference is measured
    separately; persistent external worker inference is explicitly unsupported.
    """
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = root / "input"
    source.mkdir(exist_ok=True)
    environment = dict(os.environ)
    client_threads = torch.get_num_threads()
    worker_python = str(Path(worker_python or sys.executable).resolve())
    resource_config = resources or Resources(repetitions=3, threads=1)
    request = CompressionRequest("model.fcb", "example.fcb", InputSpec(tuple(example.shape), str(example.dtype).removeprefix("torch.")),
                                 DataRoles("validation.fcb"), task=task, rank=rank, retained_energy=retained_energy,
                                 max_relative_error=max_relative_error, resources=resource_config)
    start = time.perf_counter()
    save_model_bundle(deepcopy(model).cpu().eval(), source / request.model)
    safe_save(example.detach().cpu().clone(), source / request.example)
    save_dataset(source / request.data.validation, *validation)
    serialization_seconds = time.perf_counter() - start
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in (request.model, request.example, request.data.validation)}
    try:
        local_start = time.perf_counter()
        local = execute_plan(plan_request(request), source)
        local_seconds = time.perf_counter() - local_start
    finally:
        torch.set_num_threads(client_threads)
    with (source / local["artifact"]).open("rb") as stream:
        local_artifact = torch.jit.load(stream, map_location="cpu").eval()
    with torch.inference_mode():
        local_output = local_artifact(validation[0])
    calls = []
    for repeat in range(repetitions):
        store = JobStore(root / f"jobs-{repeat}")
        runner = JobRunner(store, python_executable=worker_python)
        call_start = time.perf_counter()
        try:
            submit_start = time.perf_counter()
            job_id = runner.submit(request, source)
            submission_seconds = time.perf_counter() - submit_start
            job = runner.wait(job_id, resource_config.timeout_seconds + 30)
            external_seconds = time.perf_counter() - call_start
            result = job["result"] or {"status": job["state"]}
            record = {"repeat": repeat, "status": job["state"], "job_id": job_id,
                      "submit_and_input_copy_seconds": submission_seconds, "external_total_seconds": external_seconds,
                      "cold_process_per_job": True, "result": result}
            if job["state"] == "succeeded":
                artifact = store.directory(job_id) / result["artifact"]
                load_start = time.perf_counter()
                with artifact.open("rb") as stream:
                    loaded = torch.jit.load(stream, map_location="cpu").eval()
                record["caller_artifact_load_seconds"] = time.perf_counter() - load_start
                with torch.inference_mode():
                    remote_output = loaded(validation[0])
                    loaded(example)
                    inference_start = time.perf_counter()
                    for _ in range(resource_config.repetitions):
                        loaded(example)
                    record["warm_loaded_artifact_latency_ms"] = (time.perf_counter() - inference_start) * 1000 / resource_config.repetitions
                record["output_agreement"] = relative_error(local_output, remote_output)
                record["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
                record["artifact_bytes"] = artifact.stat().st_size
                worker_seconds = result.get("timings", {}).get("execute_plan_seconds")
                record["unattributed_boundary_seconds"] = max(0., external_seconds - submission_seconds - worker_seconds) if worker_seconds is not None else None
                record["boundary_definition"] = "total minus submission/copy and execute_plan; includes process/imports, scheduling, persistence and result polling"
                record["input_hashes_equal"] = all(result["provenance"][key] == hashes[name] for key, name in
                                                      (("model_sha256", request.model), ("validation_sha256", request.data.validation)))
            calls.append(record)
        finally:
            runner.close()
    if dict(os.environ) != environment or torch.get_num_threads() != client_threads:
        raise RuntimeError("Benchmark changed caller environment or thread configuration")
    report = {"experiment": "external_runtime", "request": request.to_dict(), "input_hashes": hashes,
              "serialization_seconds": serialization_seconds, "local_total_seconds": local_seconds,
              "local_result": local, "external_calls": calls,
              "caller": {"python": sys.executable, "python_version": platform.python_version(), "torch": torch.__version__},
              "worker_python": worker_python, "distinct_python_executables": worker_python != str(Path(sys.executable).resolve()),
              "two_environment_status": "requested_distinct_interpreters" if worker_python != str(Path(sys.executable).resolve()) else "same_environment_separate_process",
              "persistent_external_inference": {"status": "unsupported", "reason": "Existing JobRunner launches one process per job"},
              "client_environment_unchanged": True, "client_registry": {"status": "not_imported", "reason": "FEDOT registry is exercised by opt-in manifest benchmark"},
              "claim_status": "engineering_equivalence; no external speedup claim"}
    (root / "external_benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def benchmark_extension_manifest(*, modern_fedot_root, caller_python, worker_python, model_bundle,
                                 validation_bundle, features, targets, output_dir, task="regression"):
    """Opt-in real modern ExtensionManifest lifecycle in the caller environment.

    No third-party source is edited. The child imports the explicit read-only
    modern FEDOT tree; validates, smoke tests, and only then publishes in a scope.
    Invalid parameters and exceptions must leave its registry unchanged.
    """
    if task not in ("classification", "regression"):
        raise ValueError("The manifest declares only classification/regression")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload = {"modern": str(Path(modern_fedot_root).resolve()), "worker": str(Path(worker_python).resolve()),
               "model": str(Path(model_bundle).resolve()), "validation": str(Path(validation_bundle).resolve()),
               "jobs": str(root / "jobs"), "features": features.detach().cpu().tolist(), "targets": targets.detach().cpu().tolist(), "task": task}
    program = r'''
import json, os, sys, time, platform, numpy as np
from dataclasses import replace
p=json.loads(sys.argv[1]); sys.path.insert(0,p['modern'])
from fedot.extensions import validate_extension_manifest, smoke_test_extension, get_registered_extensions, extension_scope
from fedcore.external_runtime.adapters import build_industrial_extension_manifest
catalog_environment=dict(os.environ); catalog_started=time.perf_counter()
# Real FEDOT catalog initialization may lazily import its Keras module, which
# sets TF_CPP_MIN_LOG_LEVEL. Keep that measured initialization explicit and
# restore its environment side effects in this exclusively owned child.
from fedot.core.repository.operation_types_repository import OperationTypesRepository
OperationTypesRepository('all').operations
catalog_seconds=time.perf_counter()-catalog_started
catalog_changed_keys=sorted(key for key in set(catalog_environment)|set(os.environ) if catalog_environment.get(key)!=os.environ.get(key))
for key in catalog_changed_keys:
    if key in catalog_environment: os.environ[key]=catalog_environment[key]
    else: os.environ.pop(key,None)
assert dict(os.environ)==catalog_environment
environment=dict(os.environ); before=get_registered_extensions(); manifest=build_industrial_extension_manifest()
assert not validate_extension_manifest(manifest).is_left()
selected=next(spec for spec in manifest.models if spec.name.endswith(p['task']))
params={'model_bundle':p['model'],'validation_bundle':p['validation'],'jobs_root':p['jobs'],'python_executable':p['worker']}
selected_manifest=replace(manifest,models=(selected,))
assert not validate_extension_manifest(selected_manifest).is_left()
assert not smoke_test_extension(selected_manifest,{selected.name:params}).is_left()
# Only the isolated, successfully smoke-tested model is published.
estimator=selected.factory(params); x=np.asarray(p['features'],dtype=np.float32)
y=np.asarray(p['targets'],dtype=np.int64 if p['task']=='classification' else np.float32)
start=time.perf_counter(); estimator.fit(x,y); fit_seconds=time.perf_counter()-start
assert estimator.predict(x).shape==y.shape
assert smoke_test_extension(manifest,{}).is_left()
assert get_registered_extensions()==before
with extension_scope(selected_manifest): assert len(get_registered_extensions())==len(before)+1
assert get_registered_extensions()==before
try:
    with extension_scope(selected_manifest): raise RuntimeError('check scoped rollback')
except RuntimeError: pass
registry_unchanged=get_registered_extensions()==before
changed_environment_keys=sorted(key for key in set(environment)|set(os.environ) if environment.get(key)!=os.environ.get(key))
if not registry_unchanged or changed_environment_keys:
    print(json.dumps({'status':'failed','phase':'caller_state_check','registry_unchanged':registry_unchanged,
        'changed_environment_keys':changed_environment_keys,'real_fit_predict':True,'invalid_smoke_rejected':True}))
    sys.exit(2)
print(json.dumps({'status':'completed','task':p['task'],'fit_seconds':fit_seconds,'registry_unchanged':True,
 'caller_environment_unchanged':True,'caller_python':sys.executable,'worker_python':p['worker'],
 'caller_prefix':sys.prefix,'caller_python_version':platform.python_version(),
 'catalog_initialization_seconds':catalog_seconds,'catalog_import_side_effect_keys':catalog_changed_keys,
 'catalog_environment_restored':True,
 'factory_smoke':'selected factory fit/predict before scoped publication','invalid_smoke_rejected':True,
 'result':estimator.result}))
'''
    start = time.perf_counter()
    completed = subprocess.run([str(caller_python), "-c", program, json.dumps(payload)], cwd=Path(__file__).resolve().parents[2],
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True, timeout=180)
    if completed.returncode:
        report = {"status": "failed", "returncode": completed.returncode, "stderr": completed.stderr[-6000:]}
        if completed.stdout.strip():
            try:
                diagnostic = json.loads(completed.stdout.strip().splitlines()[-1])
                if isinstance(diagnostic, dict) and diagnostic.get("phase") == "caller_state_check":
                    report.update(diagnostic)
            except json.JSONDecodeError:
                pass
    else:
        report = json.loads(completed.stdout.strip().splitlines()[-1])
    report["full_boundary_seconds"] = time.perf_counter() - start
    (root / "manifest_benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report
