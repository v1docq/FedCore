"""Small-fixture protocol laws. The root orchestrator runs these under guard."""
import copy
import gzip
import json
import struct
from types import SimpleNamespace

import pytest
from hypothesis import given, strategies as st

from fedcore.experiments.fashion_protocol import (FashionPlan, RAW_FILES, default_manifest,
    stratified_partition, stratified_subset, classify_search_artifact, select_archive_candidates)

COMMIT = "4d7d9e1a73ab688cd3b8a496fdce4719025feaa5"
HASHES = {name: "a"*64 for name in RAW_FILES}


def test_manifest_finite_boundary_and_narrowing():
    raw = default_manifest(COMMIT, HASHES)
    plan = FashionPlan.parse(raw)
    raw["search_seconds"] = 120
    assert FashionPlan.parse(raw).fingerprint != plan.fingerprint
    for field, value in (("search_seconds", 601), ("job_seconds", True), ("pick_seconds", 301)):
        modified = default_manifest(COMMIT, HASHES); modified[field] = value
        with pytest.raises(ValueError): FashionPlan.parse(modified)
    for key, value in (("model", {"dimensions": [1, 2]}), ("unreviewed_callable", "eval")):
        modified = default_manifest(COMMIT, HASHES); modified[key] = value
        with pytest.raises(ValueError): FashionPlan.parse(modified)


@given(st.integers(min_value=0, max_value=2**31-1), st.permutations(tuple(range(10))))
def test_stratified_partition_conserves_rows_disjoint_quotas(seed, class_order):
    labels = [c for _ in range(10) for c in class_order]
    roles = {"fit": 40, "stop": 10, "calibration": 20, "search": 10, "pick": 20}
    actual = stratified_partition(labels, roles, seed=seed)
    flat = [i for ids in actual.values() for i in ids]
    assert sorted(flat) == list(range(100))
    assert len(set(flat)) == len(flat)
    for role, ids in actual.items():
        assert len(ids) == roles[role]
        assert [sum(labels[i] == c for i in ids) for c in range(10)] == [roles[role]//10]*10
    # An independent invocation must reproduce identities after unrelated mutation.
    saved = copy.deepcopy(actual); actual["fit"].reverse()
    assert stratified_partition(list(labels), dict(roles), seed=seed) == saved


def test_actual_protocol_60000_split_and_nondivisible_subsample_quotas():
    labels = [c for c in range(10) for _ in range(6000)]
    roles = stratified_partition(labels)
    assert {role: len(ids) for role, ids in roles.items()} == {"fit": 40000, "stop": 5000, "calibration": 5000, "search": 2000, "pick": 8000}
    calibration = stratified_subset(roles["calibration"], labels, 256, seed=12)
    search = stratified_subset(roles["search"], labels, 512, seed=13)
    assert set(calibration) <= set(roles["calibration"])
    assert set(search) <= set(roles["search"])
    assert not set(calibration) & set(search)
    assert [sum(labels[i] == c for i in calibration) for c in range(10)] == [26]*6+[25]*4
    assert [sum(labels[i] == c for i in search) for c in range(10)] == [52]*2+[51]*8


def test_test_role_cannot_reach_file_io_before_selection_lock(tmp_path, monkeypatch):
    from fedcore.experiments import fashion_worker as worker
    def forbidden(*args, **kwargs):
        raise AssertionError("raw file I/O was reached")
    monkeypatch.setattr(worker, "file_hash", forbidden)
    plan = FashionPlan.parse(default_manifest(COMMIT, HASHES))
    for allow in (False, True):
        with pytest.raises(PermissionError):
            worker.verified_raw(plan, tmp_path, RAW_FILES[2], allow_test=allow)
        with pytest.raises(PermissionError):
            worker.verified_raw(plan, tmp_path, RAW_FILES[3], allow_test=allow)


def test_idx_parser_rejects_wrong_type_shape_and_truncation(tmp_path):
    from fedcore.experiments.fashion_worker import read_idx
    path = tmp_path/"fixture.gz"
    with gzip.open(path, "wb") as stream: stream.write(struct.pack(">IIII", 2051, 2, 28, 28)+bytes(2*784))
    tensor = read_idx(path, images=True, expected_rows=2)
    assert tensor.shape == (2, 784)
    assert str(tensor.dtype) == "torch.uint8"
    for content in (struct.pack(">IIII", 2051, 2, 27, 28)+bytes(2*784), struct.pack(">II", 2049, 2)+bytes(1)):
        with gzip.open(path, "wb") as stream: stream.write(content)
        with pytest.raises(ValueError): read_idx(path, images=True, expected_rows=2)


def test_ground_truth_accuracy_is_distinct_from_teacher_agreement():
    import torch
    from torch import nn
    from fedcore.experiments.fashion_worker import evaluate
    class AlwaysZero(nn.Module):
        def forward(self, x):
            scores = torch.zeros(len(x), 10); scores[:, 0] = 8
            return scores
    result = evaluate(AlwaysZero(), torch.zeros(3, 784, dtype=torch.uint8), torch.tensor([1, 1, 1]),
        torch.arange(3), reference_predictions=[0, 0, 0], retain_predictions=True)
    assert result["accuracy"] == 0
    assert result["m0_agreement"] == 1
    assert result["cross_entropy"] > 7
    assert result["confusion"][1][0] == 3
    assert result["correct"] == [False]*3


def test_random_and_evo_share_initial_prefix_and_mask():
    import torch
    from fedcore.experiments.fashion_worker import make_source, alternatives, prefix_sample, unique_initial, seeded_request
    from fedcore.algorithm.quantization.grammar import prepare_quantization_space, genotype_payload, compile_genotype
    from fedcore.algorithm.quantization.graph_specs import Method
    plan = FashionPlan.parse(default_manifest(COMMIT, HASHES))
    source = make_source()
    choices = alternatives(plan)
    assert all(item.method is not Method.TENDER for p in ("0", "2") for item in choices[p])
    assert sum(item.method is Method.TENDER for item in choices["4"]) == 1
    space = prepare_quantization_space(source, choices, calibration=torch.zeros(4, 784),
        evaluation_context="fixture", max_peak_bytes=8*1024**3, prequantize_bits=(2, 4), training_choices=(None,))
    assert hasattr(space, "domains")
    random_initial = unique_initial(space, 24, 17); evo_initial = unique_initial(space, 4, 17)
    assert [genotype_payload(g) for g in random_initial[:4]] == [genotype_payload(g) for g in evo_initial]
    prefixes = {gene.transform.prequantize.bits if gene.transform.prequantize else None
        for seed in range(40) for gene in prefix_sample(space, seed=seed).genes
        if gene.transform.method not in (Method.DENSE, Method.TENDER)}
    assert prefixes == {None, 2, 4}
    first = compile_genotype(random_initial[0], space)
    second = compile_genotype(random_initial[1], space)
    before = genotype_payload(first.genotype)
    random_execution = seeded_request(first)
    seeded_request(second)  # Unrelated visit in the opposite algorithm/order.
    evo_execution = seeded_request(compile_genotype(evo_initial[0], space))
    assert random_execution == evo_execution
    assert genotype_payload(first.genotype) == before
    assert seeded_request(second)[1] != random_execution[1]
    assert all(item.transform.seed == random_execution[1] for item in random_execution[2].recipes)
    assert 0 <= random_execution[1] < 2**63


def test_checkpoint_roundtrip_keeps_dtype_and_unique_physical_storage(tmp_path):
    import torch
    from torch import nn
    from fedcore.experiments.quantization_pilot_worker import checkpoint
    from fedcore.algorithm.low_rank.candidate_storage import capture_inventory, inventory_cost
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4)).double()
    # Shared bias storage must be counted once and restored as shared storage.
    model[2].bias = model[0].bias
    loaded, metadata = checkpoint(model, tmp_path, torch.zeros(2, 4, dtype=torch.float64))
    assert all(p.dtype is torch.float64 for p in loaded.parameters())
    assert loaded[0].bias.untyped_storage().data_ptr() == loaded[2].bias.untyped_storage().data_ptr()
    assert metadata["physical_tensor_bytes"] == inventory_cost(capture_inventory(model), "bytes")


def test_frozen_data_and_source_code_mutations_are_detected(tmp_path, monkeypatch):
    import torch
    from fedcore.experiments import fashion_worker as worker
    from fedcore.experiments.quantization_pilot_protocol import file_hash
    plan = FashionPlan.parse(default_manifest(COMMIT, HASHES))
    monkeypatch.setattr(worker, "source_code_hashes", lambda: {"source.py": "first"})
    (tmp_path/"code.lock.json").write_text(json.dumps({"files": {"source.py": "first"}}))
    data_path = tmp_path/"data.pt"; torch.save({"images": torch.zeros(2, 784, dtype=torch.uint8)}, data_path)
    (tmp_path/"prepared.json").write_text(json.dumps({"manifest_sha256": plan.fingerprint, "data_sha256": file_hash(data_path)}))
    worker.load_data(plan, tmp_path)
    with data_path.open("ab") as stream: stream.write(b"mutation")
    with pytest.raises(ValueError, match="training data changed"): worker.load_data(plan, tmp_path)
    monkeypatch.setattr(worker, "source_code_hashes", lambda: {"source.py": "second"})
    with pytest.raises(ValueError, match="source changed"): worker.verify_code(tmp_path)


def test_in_memory_context_identity_covers_training_images_labels_and_role_ids():
    import torch
    from fedcore.experiments.fashion_worker import data_identity
    original = {"images": torch.zeros(4, 784, dtype=torch.uint8), "labels": torch.tensor([0, 1, 2, 3]),
                "indices": {"fit": torch.tensor([0, 1]), "C": torch.tensor([2]), "Vsearch": torch.tensor([3])}}
    frozen = data_identity(original)
    for field in ("images", "labels", "indices"):
        changed = copy.deepcopy(original)
        if field == "images": changed["images"][0, 0] = 1
        elif field == "labels": changed["labels"][0] = 9
        else: changed["indices"]["fit"][0] = 2
        assert data_identity(changed) != frozen
    assert data_identity(original) == frozen


def test_closed_test_exclusive_marker_prevents_second_decode(tmp_path, monkeypatch):
    from fedcore.experiments import fashion_worker as worker
    plan = FashionPlan.parse(default_manifest(COMMIT, HASHES))
    (tmp_path/"selection.lock.json").write_text(json.dumps({"manifest_sha256": plan.fingerprint,
        "trained_sha256": "fixed", "prepared_sha256": "fixed", "jobs": []}))
    monkeypatch.setattr(worker, "verify_code", lambda root: "fixed")
    monkeypatch.setattr(worker, "file_hash", lambda path: "fixed")
    calls = []
    def fail_after_opening(*args, **kwargs):
        calls.append(args[2]); raise ValueError("fixture decode failure")
    monkeypatch.setattr(worker, "verified_raw", fail_after_opening)
    with pytest.raises(ValueError, match="fixture decode failure"):
        worker.closed_test(plan, tmp_path, SimpleNamespace(remaining=lambda: 1000))
    assert calls == [RAW_FILES[2]]
    assert (tmp_path/"CLOSED_TEST_OPENED.json").exists()
    with pytest.raises(FileExistsError):
        worker.closed_test(plan, tmp_path, SimpleNamespace(remaining=lambda: 1000))
    assert calls == [RAW_FILES[2]]


def test_failed_engine_with_artifact_is_partial_and_shortlist_rule_is_fixed():
    assert classify_search_artifact({"status": "failed"}, {"sha256": "finite"}) == "partial_artifact"
    assert classify_search_artifact({"status": "succeeded"}, None) == "failed"
    def item(name, ce, size, archive=True, status="succeeded"):
        return {"canonical_solution_id": name, "search": {"cross_entropy": ce}, "artifact": {"physical_tensor_bytes": size},
                "finite_archive_member": archive, "status": status}
    actual = select_archive_candidates([item("z", .3, 2), item("a", .2, 5), item("b", .2, 4),
        item("c", .2, 4), item("too_good_but_late", .01, 1, False), item("infeasible", .01, 1, True, "infeasible")], 3)
    assert [row["canonical_solution_id"] for row in actual] == ["b", "c", "a"]


def test_paired_statistics_uses_paired_discordance():
    from fedcore.experiments.fashion_analysis import paired_correctness
    pair = paired_correctness([True, True, False, False], [True, False, True, False], resamples=0)
    assert pair["random_only_correct"] == pair["evo_only_correct"] == 1
    assert pair["accuracy_difference_evo_minus_random"] == 0
    assert pair["mcnemar_exact_two_sided_p_exploratory"] == pytest.approx(1)
    directional = paired_correctness([False]*6, [True]*6, resamples=0)
    assert directional["mcnemar_exact_two_sided_p_exploratory"] == pytest.approx(2/64)
