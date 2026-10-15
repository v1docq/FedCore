"""Post-measurement transport tests; no Torch import or scientific execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import PurePosixPath
import subprocess
import sys

import pytest

from scripts import replay_svd_third_package_portable as adapter

GRAPHS = ("svd_factor", "svd_materialized", "weighted_factor", "weighted_materialized", "dense")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")
    return adapter.digest(path)


def units(stage):
    rows = []
    for block in range(8 if stage == "pilot" else 20):
        for graph in GRAPHS:
            rows.append({"unit_id": f"{stage}-{graph}-{block}", "kind": "control", "subject": graph,
                         "block": block, "order": ["a", "b"] if block % 2 == 0 else ["b", "a"],
                         "models": {"a": graph, "b": graph}})
        if stage == "main":
            rows.append({"unit_id": f"{stage}-comparison-{block}", "kind": "comparison",
                         "subject": "svd_factor_vs_materialized", "block": block,
                         "order": ["a", "b"] if block % 2 == 0 else ["b", "a"],
                         "models": {"a": "svd_factor", "b": "svd_materialized"}})
    return rows


def archive(root):
    profiles = [("cpu", 1), ("cpu", 64), ("cuda:0", 1), ("cuda:0", 64)]
    protocol = {"profiles": [{"device": d, "batch": b} for d, b in profiles],
        "schedules": {stage: units(stage) for stage in ("pilot", "main")},
        "prepared": {"graphs": {name: {"sha256": name, "path": f"Z:/unavailable/{name}.pt"} for name in GRAPHS},
                     "input": {"path": "Z:/unavailable/input.pt", "tensor_sha256_by_batch": {"1": "input1", "64": "input64"}}}}
    protocol["protocol_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    save(root / "protocol.json", protocol)
    run = {"protocol_sha256": protocol["protocol_sha256"], "profiles": []}
    for device, batch in profiles:
        entry = {"device": device, "batch": batch, "stages": {}}
        for stage in (("pilot", "main") if (device, batch) == ("cpu", 64) else ("pilot",)):
            identifier = f"{stage}-{device.replace(':', '')}-{batch}"
            job = {"stage": stage, "device": device, "batch": batch, "session_id": identifier,
                   "protocol_sha256": protocol["protocol_sha256"]}
            result = {"job": job, "status": "admission_failed", "records": []}
            relative = f"attempts/{identifier}/result.json"
            if (device, batch) == ("cpu", 64):
                result.update(status="succeeded", profile={"dtype": "float32", "threads": 1, "runtime": "torch.jit",
                    "profiler": False, "tf32": False, "cudnn_benchmark": False, "cudnn_deterministic": True},
                    admission={"status": "admitted", "session_id": identifier,
                               "protocol_sha256": protocol["protocol_sha256"], "before_worker_spawn": True})
                for unit in protocol["schedules"][stage]:
                    record = {"unit": deepcopy(unit), "status": "succeeded", "sides": {}}
                    for side, graph in unit["models"].items():
                        latency = 2. if unit["kind"] == "comparison" and side == "a" else 1.
                        record["sides"][side] = {"wall_ms": [latency] * (10 if stage == "pilot" else 20),
                            "full_path_ms": [latency] * (10 if stage == "pilot" else 20), "cuda_event_ms": [],
                            "throughput_seconds": [latency], "input_sha256": "input64", "source_sha256": graph}
                    result["records"].append(record)
                resource_sha = save(root / "attempts" / identifier / "resources.json", {
                    "admission": result["admission"], "returncode": 0, "violation": None,
                    "post_exit": {"status": "admitted", "worker_exited": True}})
                result["resources"] = {"path": "Z:/unavailable/resources.json", "sha256": resource_sha}
            entry["stages"][stage] = {"relative_path": relative.replace("/", "\\"),
                                      "sha256": save(root / relative, result)}
        run["profiles"].append(entry)
    save(root / "run.json", run)


@pytest.mark.parametrize("reference", ["../result.json", "attempts/../result.json", "..\\result.json", "/root/result.json",
    "C:\\root\\result.json", "C:result.json", "\\\\server\\share\\result.json", "a//result.json", "a/./result.json", "a:stream/result.json", ""])
def test_path_adapter_rejects_traversal_absolute_and_ambiguous_references(reference):
    with pytest.raises(ValueError):
        adapter.normalize_reference(reference)


def test_windows_refs_replay_with_posix_path_semantics_and_original_bytes_unchanged(tmp_path):
    root = tmp_path / "archive"
    archive(root)
    before = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    destination = tmp_path / "replay"
    proof = adapter.replay(root, destination)
    assert proof["original_files_modified"] is False
    assert proof["protocol_bytes_unchanged"] is True
    assert all("\\" not in row["normalized"] for row in proof["references"])
    assert all(PurePosixPath(row["normalized"]).parts[0] == "attempts" for row in proof["references"])
    analysis = json.loads((destination / "analysis.json").read_text(encoding="utf-8"))
    assert len(analysis["comparisons"]) == 3
    assert all(row["median"] == 2. and row["ranking_eligible"] for row in analysis["comparisons"])
    assert before == {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    second = adapter.replay(root, tmp_path / "second", reference_analysis=destination)
    assert second["numerical_status_invariance"]["verified_against_existing_analysis"] is True
    assert all(item["byte_identity"] for item in second["numerical_status_invariance"]["files"])


def test_isolated_stdlib_cli_works_without_torch_or_original_model_paths(tmp_path):
    root = tmp_path / "archive"
    archive(root)
    result = subprocess.run([sys.executable, "-I", "-S", str(adapter.Path(adapter.__file__).resolve()),
        "--root", str(root), "--destination", str(tmp_path / "output")], capture_output=True, text=True,
        check=False, timeout=30)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "output" / "replay-path-normalization.json").is_file()


def test_reference_index_traversal_fails_without_mutating_originals(tmp_path):
    root = tmp_path / "archive"
    archive(root)
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    run["profiles"][0]["stages"]["pilot"]["relative_path"] = "..\\outside.json"
    save(root / "run.json", run)
    before = (root / "run.json").read_bytes()
    with pytest.raises(ValueError, match="forbidden"):
        adapter.replay(root, tmp_path / "output")
    assert before == (root / "run.json").read_bytes()
