"""Finite pilot contracts. Run this module only below the shared resource guard."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
from hypothesis import given, strategies as st

from fedcore.experiments.quantization_pilot_guard import Limits, sampled_rss, directory_bytes
from fedcore.experiments.quantization_pilot_protocol import (PilotPlan, default_manifest, canonical_bytes,
    quantile, timing_summary, HARD_RSS)

COMMIT = "c70d37914b152fd6d8886c4e692e4c8355ed05c2"


def test_deadline_never_resets_and_both_clocks_can_stop():
    limit = Limits(1000, 50, 800, 700, 500)
    assert limit.remaining(epoch=980, monotonic=35) == 15
    assert limit.remaining(epoch=1001, monotonic=35) == -1
    assert limit.remaining(epoch=980, monotonic=51) == -1


@pytest.mark.parametrize("field,value", [("max_family_rss_bytes", 9*1024**3), ("safety_soft_rss_bytes", 8*1024**3),
    ("max_artifact_bytes", 6*1024**3), ("max_concurrent_numerical_workers", True)])
def test_guard_limits_cannot_expand_authorization(tmp_path, field, value):
    raw = {"deadline_unix": 1000., "deadline_monotonic": 1000., "max_family_rss_bytes": HARD_RSS,
           "safety_soft_rss_bytes": 7*1024**3, "max_artifact_bytes": 5*1024**3,
           "max_concurrent_numerical_workers": 1, "gpu_allowed": False, "cluster_allowed": False}
    raw[field] = value
    path = tmp_path / "limits.json"; path.write_text(json.dumps(raw))
    with pytest.raises(ValueError): Limits.read(path)


def test_vanished_member_does_not_hide_later_live_member():
    def process(pid):
        if pid == 10: raise psutil.NoSuchProcess(pid)
        return SimpleNamespace(create_time=lambda: 2., memory_info=lambda: SimpleNamespace(rss=123))
    samples = []
    assert sampled_rss({10: 1., 20: 2.}, process, samples=samples) == (123, {20: 2.})
    assert samples == [{"pid": 20, "create_time": 2., "rss_bytes": 123}]


def test_pid_reuse_is_not_family_membership():
    def process(pid): return SimpleNamespace(create_time=lambda: 99., memory_info=lambda: SimpleNamespace(rss=123))
    assert sampled_rss({20: 2.}, process) == (0, {})


def test_unobservable_live_process_fails_closed():
    def process(pid): raise psutil.AccessDenied(pid)
    with pytest.raises(psutil.AccessDenied): sampled_rss({20: 2.}, process)


def test_artifact_byte_count_includes_all_files(tmp_path):
    (tmp_path / "a").write_bytes(b"1234"); (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b").write_bytes(b"123")
    assert directory_bytes(tmp_path) == 7


@pytest.mark.parametrize("field,value", [("threads", True), ("budget_fraction", 1), ("max_peak_bytes", HARD_RSS+1),
    ("jobs", ["dense32"]), ("job_seconds", float("nan")), ("job_seconds", 181)])
def test_finite_manifest_rejects_protocol_drift(field, value):
    raw = default_manifest(COMMIT); raw[field] = value
    with pytest.raises((ValueError, TypeError)): PilotPlan.parse(raw)


def test_manifest_parse_is_idempotent_and_roles_disjoint():
    plan = PilotPlan.parse(default_manifest(COMMIT))
    assert PilotPlan.parse(json.loads(canonical_bytes(plan.payload))).fingerprint == plan.fingerprint
    assert len(plan.payload["jobs"]) == 16
    assert len({v["id"] for v in plan.payload["data"].values()}) == 4
    assert len({v["seed"] for v in plan.payload["data"].values()}) == 4
    assert len(PilotPlan.parse(default_manifest(COMMIT, include_recovery=True)).payload["jobs"]) == 18


@given(st.lists(st.floats(min_value=0, max_value=1e4, allow_nan=False, allow_infinity=False), min_size=1, max_size=100))
def test_quantiles_are_bounded_ordered_and_permutation_invariant(values):
    low, middle, high = [quantile(values, p) for p in (.05, .5, .95)]
    assert min(values) <= low <= middle <= high <= max(values)
    assert quantile(list(reversed(values)), .95) == high


def test_short_profile_does_not_claim_p95():
    assert timing_summary([.001]*199)["p95_seconds"] is None
    assert timing_summary([.001]*200)["p95_seconds"] == .001


def test_seeded_source_and_independent_splits_are_reproducible():
    import torch
    from fedcore.experiments.quantization_pilot_worker import make_source, make_inputs
    plan = PilotPlan.parse(default_manifest(COMMIT))
    first, second = make_source(plan), make_source(plan)
    assert type(first) is torch.nn.Sequential
    assert all(torch.equal(a, b) for a, b in zip(first.state_dict().values(), second.state_dict().values()))
    c, v = make_inputs(plan, "calibration"), make_inputs(plan, "selection")
    assert torch.equal(c, make_inputs(plan, "calibration"))
    assert not torch.equal(c, v)
    assert first(c[:4]).shape == (4, 10)
    s = torch.linalg.svdvals(first[0].weight.double())
    assert float(s[0]/s[-1]) > 1e4


def test_every_search_sampler_supports_prefixes_and_normalizes():
    import torch
    from fedcore.experiments.quantization_pilot_worker import prefix_sample
    from fedcore.algorithm.quantization import Method, TransformSpec, QuantizedTensorSpec, prepare_quantization_space
    from fedcore.algorithm.quantization.grammar import normalize_genotype
    model = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval(); x = torch.ones(4, 4)
    space = prepare_quantization_space(model, {"0": (TransformSpec(Method.PTQ, weight=QuantizedTensorSpec(4)),)},
                                       calibration=x, evaluation_context="sampler-test", prequantize_bits=(2, 4), max_peak_bytes=HARD_RSS)
    prefixes = set()
    for seed in range(30):
        genotype = prefix_sample(space, seed=seed)
        assert normalize_genotype(genotype, space) == genotype
        assert prefix_sample(space, seed=seed) == genotype
        pre = genotype.genes[0].transform.prequantize
        prefixes.add(None if pre is None else pre.bits)
    assert prefixes == {None, 2, 4}


def test_public_quantize_artifact_roundtrip_and_physical_accounting(tmp_path):
    import torch
    from fedcore.experiments.quantization_pilot_worker import checkpoint
    from fedcore.algorithm.quantization import (quantize, QuantizationGraphRequest, OperatorRecipe, TransformSpec,
        QuantizedTensorSpec, Method, ExportVerifiedArtifact, ResourceLedger, StageBudgets)
    model = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval(); x = torch.randn(5, 4)
    ledger = ResourceLedger(StageBudgets(max_seconds=10, max_training_steps=0, max_rss_bytes=HARD_RSS))
    artifact = quantize(model, QuantizationGraphRequest((OperatorRecipe("0", TransformSpec(Method.PTQ, weight=QuantizedTensorSpec(4))),),
                        max_peak_bytes=HARD_RSS), calibration=x, ledger=ledger)
    assert type(artifact) is ExportVerifiedArtifact
    loaded, metadata = checkpoint(artifact.model, tmp_path, x)
    assert metadata["physical_tensor_bytes"] == artifact.storage_bytes
    assert (tmp_path / metadata["checkpoint"]).stat().st_size == metadata["file_bytes"]
    assert torch.isfinite(loaded(x)).all()
    assert metadata["roundtrip_max_abs"] == 0


def test_dense64_checkpoint_preserves_dtype_and_complete_storage(tmp_path):
    import torch
    from fedcore.experiments.quantization_pilot_worker import checkpoint
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2)).double().eval()
    loaded, metadata = checkpoint(model, tmp_path, torch.randn(4, 4))
    assert next(loaded.parameters()).dtype is torch.float64
    assert metadata["physical_tensor_bytes"] == (4*3+3+3*2+2)*8


def test_freeze_prevents_missing_jobs_and_closed_test_opens_once(tmp_path):
    from fedcore.experiments.quantization_pilot_worker import freeze_selection, closed_test
    from fedcore.experiments.quantization_pilot_guard import atomic_json
    plan = PilotPlan.parse(default_manifest(COMMIT))
    with pytest.raises(ValueError, match="Every declared job"):
        freeze_selection(plan, tmp_path)
    (tmp_path / "jobs").mkdir()
    for job in plan.payload["jobs"]:
        atomic_json(tmp_path / "jobs" / (job+".json"), {"status": "failed"})
    freeze_selection(plan, tmp_path)
    with pytest.raises(FileExistsError): freeze_selection(plan, tmp_path)
    # Creating this marker is the irreversible test-opening transition. An
    # interrupted final assessment must preserve its partial report, not reopen.
    (tmp_path / "CLOSED_TEST_OPENED.json").write_text("{}")
    with pytest.raises(FileExistsError): closed_test(plan, tmp_path)
