"""S2 amendment invariants; execution belongs to the root guarded process."""
import copy
import json
from types import SimpleNamespace

import pytest

from fedcore.experiments.fashion_independent_worker import (
    IndependentFashionPlan, default_s2_manifest, unused_training_holdout, S1_COMMIT, M0_SHA256)
from fedcore.experiments.fashion_protocol import RAW_FILES, stratified_partition, stratified_subset

COMMIT = "a"*40
HASHES = {name: "b"*64 for name in RAW_FILES}
PARENT = {"source_commit": S1_COMMIT, "manifest_sha256": "c"*64, "prepared_sha256": "d"*64,
          "trained_sha256": "e"*64, "data_sha256": "f"*64, "code_lock_sha256": "1"*64, "source_sha256": M0_SHA256}


def test_s2_schema_requires_exact_m0_and_fixed_holdout_without_retraining():
    plan = IndependentFashionPlan.parse(default_s2_manifest(COMMIT, HASHES, PARENT))
    assert plan.payload["data"]["test_rows"] == 6000
    assert plan.payload["training_reuse"]["source_sha256"] == M0_SHA256
    for mutation in (lambda raw: raw["data"]["holdout"].update(seed=999),
                     lambda raw: raw["parent"].update(source_sha256="0"*64),
                     lambda raw: raw["search"].update(evaluations=48),
                     lambda raw: raw.update(unreviewed_model="other")):
        raw = default_s2_manifest(COMMIT, HASHES, PARENT); mutation(raw)
        with pytest.raises(ValueError): IndependentFashionPlan.parse(raw)


def test_unused_training_holdout_is_600_per_class_and_excludes_all_operational_roles():
    labels = [c for c in range(10) for _ in range(6000)]
    indices = stratified_partition(labels)
    indices["C"] = stratified_subset(indices["calibration"], labels, 256, seed=20261028)
    indices["Vsearch"] = stratified_subset(indices["search"], labels, 512, seed=20261029)
    original = copy.deepcopy(indices)
    selected, pool = unused_training_holdout(indices, labels)
    assert len(pool) == 6232
    assert len(selected) == len(set(selected)) == 6000
    assert set(selected) <= set(pool)
    for role in ("fit", "stop", "C", "Vsearch", "pick"):
        assert not set(selected) & set(indices[role])
        assert not set(pool) & set(indices[role])
    assert [sum(labels[i] == c for i in selected) for c in range(10)] == [600]*10
    assert indices == original
    assert unused_training_holdout(copy.deepcopy(indices), list(labels))[0] == selected
    bad = copy.deepcopy(indices); bad["C"][0] = bad["fit"][0]
    with pytest.raises(ValueError): unused_training_holdout(bad, labels)


def test_s2_closed_holdout_loader_runs_only_after_exclusive_lock_and_never_opens_raw(tmp_path, monkeypatch):
    from fedcore.experiments import fashion_worker as common
    from fedcore.experiments.fashion_independent_worker import heldout_loader
    import torch
    plan = IndependentFashionPlan.parse(default_s2_manifest(COMMIT, HASHES, PARENT))
    monkeypatch.setattr(common, "verify_code", lambda root: "fixed")
    monkeypatch.setattr(common, "file_hash", lambda path: "fixed")
    monkeypatch.setattr(common, "verified_raw", lambda *args, **kwargs: pytest.fail("S2 reached official raw test I/O"))
    observed = []
    def load_data(plan, root):
        assert (root/"selection.lock.json").exists()
        assert (root/"CLOSED_TEST_OPENED.json").exists()
        observed.append("load")
        return {}, {"images": torch.zeros(6000, 784, dtype=torch.uint8), "labels": torch.zeros(6000, dtype=torch.long),
                    "indices": {"heldout": torch.arange(6000)}}
    monkeypatch.setattr(common, "load_data", load_data)
    with pytest.raises(FileNotFoundError): common.closed_test(plan, tmp_path, SimpleNamespace(remaining=lambda: 1000), data_loader=heldout_loader)
    assert observed == []
    (tmp_path/"selection.lock.json").write_text(json.dumps({"manifest_sha256": plan.fingerprint,
        "trained_sha256": "fixed", "prepared_sha256": "fixed", "jobs": []}))
    result = common.closed_test(plan, tmp_path, SimpleNamespace(remaining=lambda: 1000), data_loader=heldout_loader)
    assert observed == ["load"]
    assert result["class_counts"] == [6000]+[0]*9
    with pytest.raises(FileExistsError): common.closed_test(plan, tmp_path, SimpleNamespace(remaining=lambda: 1000), data_loader=heldout_loader)
    assert observed == ["load"]


def test_artifact_copy_preserves_exact_bytes_and_rejects_mismatches(tmp_path):
    from fedcore.experiments.fashion_independent_worker import copy_artifact
    from fedcore.experiments.quantization_pilot_protocol import file_hash
    parent = tmp_path/"s1"; root = tmp_path/"s2"
    (parent/"cas").mkdir(parents=True); root.mkdir()
    original = parent/"cas"/"fixture.pt"; original.write_bytes(b"exact checkpoint bytes")
    metadata = {"checkpoint": "cas/fixture.pt", "sha256": file_hash(original), "physical_tensor_bytes": 123}
    copied = copy_artifact(parent, root, metadata)
    assert copied == metadata
    assert (root/"cas"/"fixture.pt").read_bytes() == original.read_bytes()
    (root/"cas"/"fixture.pt").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="destination artifact"): copy_artifact(parent, root, metadata)
    with pytest.raises(ValueError, match="escaped"): copy_artifact(parent, root, {**metadata, "checkpoint": "../escape.pt"})


def test_direct_s2_holdout_loader_rejects_unlocked_access_before_data_io(tmp_path, monkeypatch):
    from fedcore.experiments import fashion_worker as common
    from fedcore.experiments.fashion_independent_worker import heldout_loader
    monkeypatch.setattr(common, "load_data", lambda *args: pytest.fail("unlocked holdout reached data I/O"))
    plan = IndependentFashionPlan.parse(default_s2_manifest(COMMIT, HASHES, PARENT))
    with pytest.raises(PermissionError): heldout_loader(plan, tmp_path)
    (tmp_path/"selection.lock.json").write_text("{}")
    with pytest.raises(PermissionError): heldout_loader(plan, tmp_path)
    (tmp_path/"selection.lock.json").unlink()
    (tmp_path/"CLOSED_TEST_OPENED.json").write_text("{}")
    with pytest.raises(PermissionError): heldout_loader(plan, tmp_path)
