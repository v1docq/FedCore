"""Admission corrections preserve the original quota and immutable evidence."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from fedcore.experiments import compression_search_package as package


def prior_run(tmp_path):
    prior = tmp_path / "prior"
    prior.mkdir()
    source = tmp_path / "historical"
    digest = hashlib.sha256((package.REPO / "docs/svd_compression/PLAN_ISSUE_99_LOCAL_1.md").read_bytes()).hexdigest()
    package.write_json(prior / "protocol.json", {"plan_sha256": digest, "historical_root": str(source.resolve()),
        "protocol_sha256": "prior-protocol", "git_revision": "prior-build"})
    package.write_json(prior / "resource-ledger.json", {"status": "stopped", "started_at": 0.,
        "jobs": [{"kind": "admission", "status": "failed"}], "elapsed_seconds": 1.})
    (prior / "mlp-attempts.jsonl").write_text('{"attempt":1,"job":"admission-0","protocol_sha256":"prior-protocol"}\n', encoding="utf-8")
    return prior, source


def test_corrected_admission_keeps_original_time_counter_and_attempt_evidence(tmp_path, monkeypatch):
    prior, source = prior_run(tmp_path)
    original = {path.name: path.read_bytes() for path in prior.iterdir()}
    monkeypatch.setattr(package, "MODULE_STARTED_AT", 10.)
    monkeypatch.setattr(package.time, "perf_counter", lambda: 11.)
    monkeypatch.setattr(package, "source_hashes", lambda: {"new.py": "new-source"})
    monkeypatch.setattr(package.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="corrected-build\n"))
    calls = []

    def stopped(output, job, protocol, started):
        calls.append((started, package.count_attempts(output)))
        return {"job": "admission-0", "kind": "admission", "status": "failed", "error": "fixture stop", "resources": {}}

    monkeypatch.setattr(package, "_run_job", stopped)
    output = tmp_path / "continued"
    result = package.run_package(output, source, resume_from=prior)
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert calls == [(0., 1)]
    assert result["started_at"] == 0. and result["elapsed_seconds"] == 11. and result["mlp_attempts"] == 1
    assert protocol["resources"]["max_seconds"] == 7200 and protocol["resources"]["max_mlp_attempts"] == 3300
    assert protocol["resumed_admission"]["new_free_budget"] is False
    assert protocol["resumed_admission"]["protocol_sha256"] == "prior-protocol"
    assert protocol["git_revision"] == "corrected-build"
    assert (output / "mlp-attempts.jsonl").read_bytes() == original["mlp-attempts.jsonl"]
    assert {path.name: path.read_bytes() for path in prior.iterdir()} == original


@pytest.mark.parametrize("change", ("scientific_run", "plan", "source", "future_clock"))
def test_resume_refuses_replaying_experiments_or_changing_plan_source_or_clock(tmp_path, monkeypatch, change):
    prior, source = prior_run(tmp_path)
    monkeypatch.setattr(package, "MODULE_STARTED_AT", 10.)
    if change in ("scientific_run", "future_clock"):
        path = prior / "resource-ledger.json"
        value = package.read_json(path)
        if change == "scientific_run":
            value["jobs"].append({"kind": "search", "status": "succeeded"})
        else:
            value["started_at"] = 11.
        package.write_json(path, value)
    else:
        path = prior / "protocol.json"
        value = package.read_json(path)
        value["plan_sha256" if change == "plan" else "historical_root"] = "changed"
        package.write_json(path, value)
    output = tmp_path / "forbidden"
    with pytest.raises(RuntimeError):
        package.run_package(output, source, resume_from=prior)
    assert not output.exists()
