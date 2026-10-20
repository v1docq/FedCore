"""Admission corrections preserve the original quota and immutable evidence."""
import hashlib
import ast
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


def test_reuse_paths_require_identical_numerical_entrypoint_and_oracle_ast():
    from pathlib import Path
    original = Path(package.__file__).read_text(encoding="utf-8")
    proof = package.unchanged_reuse_paths(original, "# comment\n" + original)
    assert set(proof) == {"_toy", "composition_job", "_source_evaluator", "_run_job", "oracle_branch"}
    for function_name in ("_toy", "composition_job", "_source_evaluator", "_run_job", "execute_job"):
        tree = ast.parse(original)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
        if function_name == "execute_job":
            branch = next(node for node in function.body if isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare) and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value == "oracle")
            branch.body.append(ast.Pass())
        else:
            function.body.append(ast.Pass())
        with pytest.raises(RuntimeError, match="execution path changed"):
            package.unchanged_reuse_paths(original, ast.unparse(ast.fix_missing_locations(tree)))


def test_reuse_preserves_original_job_bytes_and_provenance_without_new_attempts(tmp_path, monkeypatch):
    from pathlib import Path
    old, target = tmp_path / "old", tmp_path / "new"
    source = old / "jobs/admissibility"
    source.mkdir(parents=True)
    target.mkdir()
    package.write_json(source / "result.json", {"status": "succeeded", "rows": [{"case": i} for i in range(163)]})
    package.write_json(source / "job.json", {"protocol_sha256": "old", "worker_started_at": 12.})
    files = {path.name: path.read_bytes() for path in source.iterdir()}
    source_map = {"numerical.py": "unchanged", "fedcore/interfaces/fedcore_dispatcher.py": "old"}
    before = {"source_files": source_map, "git_revision": "old-build", "protocol_sha256": "old"}
    after = {"source_files": dict(source_map, **{"fedcore/interfaces/fedcore_dispatcher.py": "fixed"})}
    original = Path(package.__file__).read_text(encoding="utf-8")
    monkeypatch.setattr(package.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=original))
    reused = package._reuse_unaffected(old, target, before, after,
        {"jobs": [{"job": "admissibility", "kind": "admissibility", "status": "succeeded"}]})
    record = reused["admissibility"]
    assert record["evidence_reused"] is True and record["new_attempts"] == 0
    assert record["origin_source_build"] == "old-build" and record["origin_protocol_sha256"] == "old"
    assert {path.name: path.read_bytes() for path in (target / "jobs/admissibility").iterdir()} == files
    assert {path.name: path.read_bytes() for path in source.iterdir()} == files
    assert not (target / "mlp-attempts.jsonl").exists()
    after["source_files"]["numerical.py"] = "changed"
    with pytest.raises(RuntimeError, match="numerical/source changes"):
        package._reuse_unaffected(old, target, before, after, {"jobs": []})
