"""Independent oracles for the second, inference-only local SVD packet.

Tiny CPU tensors and fake clocks test the protocol. These tests never launch
the scientific timing series, train a model, or modify first-packet evidence.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter, defaultdict
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fedcore.experiments import svd_second_package as packet, svd_second_timing as timing
from fedcore.experiments.protocol import ProtocolError
from scripts import analyze_svd_second_package as analysis


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("n", range(6, 13))
def test_median_interval_coverage_matches_exhaustive_sign_sequences(n):
    """Every equiprobable sign sequence is an independent coverage oracle.

    The common population median is between order statistics k and n-k+1
    exactly when the number of observations below it lies in [k, n-k].
    Counting sequences directly avoids reusing the production binomial sum.
    """
    ratios = [float(index + 1) for index in range(n)]
    result = analysis.median_interval(reversed(ratios))
    assert result is not None
    possible = []
    for tail in range(1, n // 2 + 1):
        covered = sum(tail <= sum(signs) <= n - tail
                      for signs in product((0, 1), repeat=n))
        coverage = covered / 2 ** n
        if coverage >= .95:
            possible.append((tail, coverage))
    tail, coverage = possible[-1]
    assert result["order_from_each_tail"] == tail
    assert result["lower"] == ratios[tail - 1]
    assert result["upper"] == ratios[n - tail]
    assert result["binomial_coverage"] == pytest.approx(coverage)
    assert result["minimum_coverage"] == .95
    assert "paired blocks" in result["assumptions"]
    assert "not across trained seeds" in result["scope"]


def test_ten_blocks_use_second_and_ninth_ratios_not_call_pseudoreplicates():
    result = analysis.describe([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    assert result["n"] == 10
    assert result["median"] == 5.5
    interval = result["median_interval"]
    assert (interval["lower"], interval["upper"]) == (2., 9.)
    assert interval["binomial_coverage"] == .978515625


@pytest.mark.parametrize("n", range(6))
def test_fewer_than_six_blocks_do_not_claim_95_percent_interval(n):
    values = [1. + index / 10 for index in range(n)]
    result = analysis.describe(values)
    assert result["n"] == n
    assert result["median_interval"] is None
    if n:
        assert result["min"] == min(values) and result["max"] == max(values)
    else:
        assert result["median"] is result["min"] is result["max"] is None


@pytest.mark.parametrize("bad", [0., -1., float("nan"), float("inf"), -float("inf")])
def test_nonpositive_or_nonfinite_ratio_is_never_silently_dropped(bad):
    with pytest.raises(ValueError):
        analysis.describe([1., bad, 2.])
    with pytest.raises(ValueError):
        analysis.median_interval([1., bad, 2.])


@pytest.mark.parametrize("coverage", [0., 1., -1., 1.01, float("nan")])
def test_invalid_nominal_coverage_is_rejected(coverage):
    with pytest.raises(ValueError):
        analysis.median_interval([1.] * 10, coverage)


def test_conv_descriptive_range_cannot_gain_interval_from_runtime_call_count():
    result = analysis.describe([.9, 1., 1.1], interval=False)
    assert result == {"n": 3, "median": 1., "min": .9, "max": 1.1,
                      "median_interval": None}


def _admission(tmp_path, **changes):
    token = {"status": "admitted", "session_id": "session-1",
             "protocol_sha256": "frozen-protocol", "before_worker_spawn": True}
    token.update(changes)
    path = tmp_path / "admission.json"
    path.write_text(json.dumps(token), encoding="utf-8")
    return {"admission_path": str(path), "session_id": "session-1",
            "protocol_sha256": "frozen-protocol"}, token


@pytest.mark.parametrize("changes", [{"status": "timeout"}, {"session_id": "other"},
    {"protocol_sha256": "changed"}, {"before_worker_spawn": False}])
def test_worker_refuses_unbound_external_admission_before_any_cuda_query(tmp_path, monkeypatch, changes):
    job, _ = _admission(tmp_path, **changes)
    job.update(directory=str(tmp_path / "session"), device="cuda:0")

    def forbidden_cuda():
        pytest.fail("Rejected admission must precede every CUDA query")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden_cuda)
    with pytest.raises(ProtocolError, match="admission token"):
        timing.timing_worker(job)
    assert not (tmp_path / "session").exists()


def test_external_admission_requires_exact_session_and_protocol_identity(tmp_path):
    job, token = _admission(tmp_path)
    assert timing.require_admission(job) == token


def test_other_batch_smoke_checks_values_even_when_shapes_match(tmp_path, monkeypatch):
    """Tracing a graph preserving shape can still alter the alternate batch."""
    model = nn.Linear(2, 2, bias=False).eval()
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    source = tmp_path / "source.pt"
    torch.jit.trace(model, torch.ones(1, 2)).save(str(source))
    inputs = tmp_path / "input.pt"
    torch.save(torch.tensor([[1., 2.]]), inputs)
    job, _ = _admission(tmp_path)
    job.update(directory=str(tmp_path / "session"), device="cpu", batch=1,
               source_artifact=str(source), source_artifact_sha256=timing.measurement.file_hash(source),
               input_path=str(inputs), input_file_sha256=timing.measurement.file_hash(inputs))

    class WrongAlternateBatch(nn.Module):
        def forward(self, x):
            return x if len(x) == 1 else 2 * x

    monkeypatch.setattr(timing, "precision_profile", lambda *args: None)
    monkeypatch.setattr(timing.atexit, "register", lambda *args: None)
    monkeypatch.setattr(timing.atexit, "unregister", lambda *args: None)
    monkeypatch.setattr(timing, "export_model", lambda *args: source)
    monkeypatch.setattr(timing.measurement, "load_artifact", lambda _: WrongAlternateBatch())
    with pytest.raises(AssertionError):
        timing.timing_worker(job)
    assert not (tmp_path / "session/worker-result.json").exists()


@pytest.mark.parametrize("batch", [1, 64])
def test_public_worker_reads_cyrillic_torchscript_source_through_binary_file(tmp_path, monkeypatch, batch):
    """Python file handles preserve Windows Unicode paths for the C++ loader."""
    model = nn.Linear(2, 3, bias=True).eval()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1., 2.], [-3., .5], [.25, -2.]]))
        model.bias.copy_(torch.tensor([.5, -.25, 1.]))
    directory = tmp_path / "данные модели"
    directory.mkdir()
    source = directory / "архив модели.pt"
    with source.open("wb") as stream:
        torch.jit.save(torch.jit.trace(model, torch.ones(1, 2)), stream)
    x = torch.arange(128, dtype=torch.float32).reshape(64, 2) / 16
    inputs = tmp_path / "inputs.pt"
    torch.save(x, inputs)
    job, _ = _admission(tmp_path)
    job.update(directory=str(tmp_path / "session"), device="cpu", batch=batch,
        source_artifact=str(source), source_artifact_sha256=timing.measurement.file_hash(source),
        input_path=str(inputs), input_file_sha256=timing.measurement.file_hash(inputs),
        input_tensor_sha256=timing.tensor_hash(x[:batch]), repeats=1, warmup=0,
        minimum_warmup_seconds=0., maximum_warmup_calls=10,
        throughput_blocks=1, throughput_calls=1, profiler=False)
    calls = []
    original_load = torch.jit.load

    def load_from_python_file(archive, *args, **kwargs):
        assert hasattr(archive, "read"), "The Unicode archive must use Python's binary stream"
        calls.append(Path(archive.name))
        return original_load(archive, *args, **kwargs)

    monkeypatch.setattr(torch.jit, "load", load_from_python_file)
    monkeypatch.setattr(timing, "precision_profile", lambda *args: None)
    monkeypatch.setattr(timing, "export_model", lambda *args: source)
    result = timing.timing_worker(job)
    assert result["status"] == "succeeded"
    assert calls == [source, source]
    assert result["source_artifact_sha256"] == timing.measurement.file_hash(source)
    assert result["input_sha256"] == timing.tensor_hash(x[:batch])
    with torch.inference_mode():
        expected = torch.nn.functional.linear(x[:batch], model.weight, model.bias)
    assert result["output_sha256"] == timing.tensor_hash(expected)
    assert (tmp_path / "session/worker-result.json").is_file()


@pytest.mark.parametrize("seed", [0, 42, 999])
def test_finite_schedule_conserves_comparisons_profiles_and_adjacent_members(seed):
    global_state = random.getstate()
    sessions = [asdict(row) for row in packet.timing_schedule(seed)]
    assert sessions == [asdict(row) for row in packet.timing_schedule(seed)]
    assert random.getstate() == global_state
    assert len(sessions) == len({row["session_id"] for row in sessions}) == 584
    assert Counter(row["kind"] for row in sessions) == {"artifact": 400, "control": 40, "conv": 144}
    units, positions = defaultdict(list), defaultdict(list)
    for index, row in enumerate(sessions):
        units[row["block_id"]].append(row)
        positions[row["block_id"]].append(index)
    expected = {
        ("artifact", name, device, batch, block)
        for name in ("conv2_svd", "conv2_weighted", "conv2_svd_linear16",
                     "conv2_weighted_linear16", "historical_afm_linear8")
        for device, batch, block in product(("cpu", "cuda:0"), (1, 64), range(10))}
    expected |= {("control", "dense_dense", device, batch, block)
                 for device, batch, block in product(("cpu", "cuda:0"), (1, 64), range(5))}
    expected |= {("conv", name, device, batch, block)
                 for name in ("digits_conv2", "synthetic_64_32_r16", "synthetic_128_16_r32", "synthetic_256_8_r64")
                 for device, batch, block in product(("cpu", "cuda:0"), (1, 64), range(3))}
    observed, starts, rotations = set(), defaultdict(list), defaultdict(list)
    for block_id, members in units.items():
        first = members[0]
        key = tuple(first[key] for key in ("kind", "subject", "device", "batch", "block"))
        observed.add(key)
        assert positions[block_id] == list(range(positions[block_id][0], positions[block_id][0] + len(members)))
        assert [row["order"] for row in members] == list(range(len(members)))
        sides = [row["side"] for row in members]
        if first["kind"] == "conv":
            assert set(sides) == {"original_dense", "factor", "materialized"}
            rotations[key[:4]].append(sides)
        else:
            assert set(sides) == {"baseline", "candidate"} and len(sides) == 2
            starts[key[:4]].append(sides[0])
    assert observed == expected
    for key, orders in starts.items():
        counts = Counter(orders)
        assert abs(counts["baseline"] - counts["candidate"]) <= 1
        if key[0] == "artifact":
            assert counts == {"baseline": 5, "candidate": 5}
    for orders in rotations.values():
        assert len(orders) == 3
        for position in range(3):
            assert {order[position] for order in orders} == {"original_dense", "factor", "materialized"}


def _quiet(cpu=0., gpu=0., available=32 * 1024 ** 3, gpu_free=4096.):
    return {"cpu_percent": cpu, "available_ram_bytes": available,
            "gpu": {"utilization_percent": gpu, "physical_free_mib": gpu_free}}


def test_external_quiet_thresholds_preserve_first_packet_boundaries():
    boundary = _quiet(cpu=25., gpu=10., available=6 * 1024 ** 3, gpu_free=2048.)
    assert packet.quiet_reasons(boundary, "cuda:0") == []
    assert packet.quiet_reasons(_quiet(cpu=25.01), "cpu")
    assert packet.quiet_reasons(_quiet(gpu=10.01), "cuda:0")
    assert packet.quiet_reasons(_quiet(gpu=100., gpu_free=0.), "cpu") == []
    assert packet.quiet_reasons(_quiet(available=6 * 1024 ** 3 - 1), "cpu")
    assert packet.quiet_reasons(_quiet(gpu_free=2047.99), "cuda:0")


@pytest.mark.parametrize("sample", [_quiet(cpu=None), _quiet(cpu=float("nan")),
    {"cpu_percent": 0., "available_ram_bytes": 32 * 1024 ** 3},
    _quiet(gpu=float("nan")), _quiet(gpu_free=float("inf"))])
def test_unavailable_external_observation_cannot_admit_cuda(sample):
    assert packet.quiet_reasons(sample, "cuda:0")


class _Clock:
    def __init__(self):
        self.now = 0.
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        assert duration >= 0
        self.sleeps.append(duration)
        self.now += duration


def test_external_quiet_streak_resets_and_retains_every_failed_observation():
    clock = _Clock()
    observations = iter((_quiet(), _quiet(), _quiet(gpu=90.), _quiet(), _quiet(), _quiet()))
    result = packet.wait_for_quiet("cuda:0", packet.AdmissionPolicy(timeout_seconds=10.),
        observer=lambda: next(observations), monotonic=clock.monotonic, sleep=clock.sleep)
    assert result["status"] == "admitted"
    assert [row["quiet_streak"] for row in result["samples"]] == [1, 2, 0, 1, 2, 3]
    assert len(result["samples"]) == 6
    assert result["samples"][2]["reasons"]
    assert result["elapsed_seconds"] == 5.
    assert clock.sleeps == [1.] * 5


def test_external_quiet_wait_is_finite_and_cannot_admit_after_deadline():
    clock = _Clock()
    result = packet.wait_for_quiet("cuda:0", packet.AdmissionPolicy(timeout_seconds=2.5),
        observer=lambda: _quiet(gpu=90.), monotonic=clock.monotonic, sleep=clock.sleep)
    assert result["status"] == "admission_failed"
    assert result["elapsed_seconds"] == 2.5
    assert len(result["samples"]) == 3
    assert all(row["quiet_streak"] == 0 for row in result["samples"])
    assert clock.sleeps == [1., 1., .5]


@pytest.mark.parametrize("channels,size,rank", [(2, 3, 1), (3, 4, 2)])
def test_materialized_conv_has_exact_factor_weight_and_independent_functional_oracle(channels, size, rank, monkeypatch):
    random_state, torch_state = random.getstate(), torch.get_rng_state().clone()

    def forbidden_cuda(*args, **kwargs):
        pytest.fail("CPU source preparation must not query or initialize CUDA")

    monkeypatch.setattr(torch.cuda, "device_count", forbidden_cuda)
    monkeypatch.setattr(torch.cuda, "get_rng_state", forbidden_cuda)
    monkeypatch.setattr(torch.cuda, "set_rng_state", forbidden_cuda)
    original, factor, dense, x = packet.synthetic_conv(channels, size, rank, 42)
    with torch.inference_mode():
        # Reconstruct the operator independently rather than calling factor_matrix.
        weight = (factor.U @ factor.Vh).reshape(channels, channels, 3, 3)
        oracle = torch.nn.functional.conv2d(x[:2], weight, factor.bias, padding=1)
        torch.testing.assert_close(dense.weight, weight, rtol=0, atol=0)
        torch.testing.assert_close(dense.bias, factor.bias, rtol=0, atol=0)
        torch.testing.assert_close(dense(x[:2]), oracle, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(factor(x[:2]), oracle, rtol=1e-4, atol=1e-5)
        assert not torch.equal(original.weight, dense.weight)
        original_bytes = (channels * channels * 9 + channels) * 4
        factor_bytes = (channels * rank + rank * channels * 9 + channels) * 4
        assert timing.measurement.tensor_state_bytes(original) == original_bytes
        assert timing.measurement.tensor_state_bytes(dense) == original_bytes
        assert timing.measurement.tensor_state_bytes(factor) == factor_bytes
    assert random.getstate() == random_state
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)


def _comparison_fixture(tmp_path):
    """An independently assembled seven-session, three-block evidence model."""
    source = lambda name: {"path": str(tmp_path / f"{name}.pt"), "sha256": f"source-{name}"}
    inputs = {"path": str(tmp_path / "inputs.pt"), "sha256": "input-file",
              "tensor_sha256_by_batch": {"1": "input-tensor"}}
    sources = {name: source(name) for name in ("dense", "conv2_svd")}
    comparisons = {"conv2_svd": {"baseline": "dense", "candidate": "conv2_svd"},
                   "dense_dense": {"baseline": "dense", "candidate": "dense"}}
    representations = {side: source(side) for side in ("original_dense", "factor", "materialized")}
    schedule, sessions = [], []
    for kind, subject, sides in (("artifact", "conv2_svd", ("baseline", "candidate")),
        ("control", "dense_dense", ("candidate", "baseline")),
        ("conv", "digits_conv2", ("factor", "original_dense", "materialized"))):
        block_id = f"{kind}-block"
        for order, side in enumerate(sides):
            plan = {"session_id": f"{block_id}-{side}", "block_id": block_id, "kind": kind,
                    "subject": subject, "device": "cpu", "batch": 1, "block": 0,
                    "side": side, "order": order}
            schedule.append(plan)
            frozen_source = representations[side] if kind == "conv" else sources[comparisons[subject][side]]
            job = dict(plan, directory=str(tmp_path / plan["session_id"]),
                protocol_sha256="protocol", source_artifact=frozen_source["path"],
                source_artifact_sha256=frozen_source["sha256"], input_path=inputs["path"],
                input_file_sha256=inputs["sha256"], input_tensor_sha256="input-tensor", profiler=False)
            quiet_samples = [{"sample": _quiet(), "reasons": [], "quiet_streak": index + 1,
                              "elapsed_seconds": float(index)} for index in range(3)]
            admission = {"status": "admitted", "session_id": plan["session_id"],
                "protocol_sha256": "protocol", "before_worker_spawn": True, "samples": quiet_samples}
            post = {"status": "admitted", "worker_exited": True, "worker_returncode": 0,
                    "drain_seconds": 2.2, "samples": quiet_samples}
            value = 2. if side in ("baseline", "original_dense", "materialized") else 1.
            sessions.append({"job": job, "status": "succeeded", "input_sha256": "input-tensor",
                "source_artifact_sha256": frozen_source["sha256"], "admission": admission,
                "post_exit_observation": post, "profile": {"profiler": False, "precision": "float32",
                    "threads": 1, "runtime": "torch.jit", "matmul_tf32": False, "cudnn_tf32": False,
                    "cudnn_benchmark": False, "cudnn_deterministic": True, "batch_shape_checked": True},
                "wall": {"p50_ms": value, "p95_ms": value + .5},
                "full_path": {"p50_ms": value * 3, "p95_ms": value * 3 + 1.}, "cuda_event": None,
                "blocked_throughput": [{"samples_per_second": 100. / value, "calls": 50} for _ in range(3)],
                "noise_reasons": []})
    protocol = {"timing_schedule": schedule, "protocol_sha256": "protocol",
        "prepared": {"sources": sources, "comparisons": comparisons, "model_input": inputs,
                     "conv_groups": {"digits_conv2": {"input": inputs, "representations": representations}}},
        "timing": {"individual_repeats": 100, "warmup": 50, "throughput_blocks": 3,
                   "calls_per_throughput_block": 50},
        "admission": asdict(packet.AdmissionPolicy())}
    return protocol, sessions


@pytest.mark.parametrize("status", ["admission_failed", "technical_failure", "invalid_noise", "not_run_budget"])
def test_failed_member_invalidates_whole_pair_and_all_three_conv_comparisons(tmp_path, status):
    protocol, sessions = _comparison_fixture(tmp_path)
    failed_artifact = next(row for row in sessions if row["job"]["kind"] == "artifact" and row["job"]["side"] == "candidate")
    failed_conv = next(row for row in sessions if row["job"]["kind"] == "conv" and row["job"]["side"] == "original_dense")
    for failed in (failed_artifact, failed_conv):
        failed["status"] = status
        failed["noise_reasons"] = ["retained failure"]
    before = deepcopy(sessions)
    blocks = packet.summarize_blocks(protocol, sessions)
    assert len(blocks) == 3
    assert Counter(row["status"] for row in blocks) == {"succeeded": 1, "invalid": 2}
    assert all(row["invalid_reasons"][0]["status"] == status for row in blocks if row["status"] == "invalid")
    rows, summaries = analysis.timing_analysis(protocol, sessions)
    assert len(rows) == 5
    assert sum(row["valid"] for row in rows) == 1
    assert all(not any(key.endswith("_ratio") for key in row) for row in rows if not row["valid"])
    assert all(row["planned_blocks"] == 1 for row in summaries)
    assert all(row["valid_blocks"] == (1 if row["kind"] == "control" else 0) for row in summaries)
    assert sessions == before


def test_missing_session_conserves_planned_block_and_analyzer_refuses_partial_packet(tmp_path):
    protocol, sessions = _comparison_fixture(tmp_path)
    omitted = sessions.pop(0)
    blocks = packet.summarize_blocks(protocol, sessions)
    block = next(row for row in blocks if omitted["job"]["session_id"] in row["session_ids"])
    assert block["status"] == "invalid" and block["invalid_reasons"][0]["status"] == "missing"
    assert len(blocks) == 3
    with pytest.raises(ValueError, match="retained outcome"):
        analysis.timing_analysis(protocol, sessions)


@pytest.mark.parametrize("field,replacement", [("source_artifact_sha256", "other-model"),
    ("input_file_sha256", "other-input"), ("protocol_sha256", "other-protocol"),
    ("side", "candidate"), ("block", 9), ("order", 9)])
def test_result_cannot_change_its_frozen_source_input_protocol_or_schedule(tmp_path, field, replacement):
    protocol, sessions = _comparison_fixture(tmp_path)
    sessions[0]["job"][field] = replacement
    with pytest.raises(ProtocolError):
        packet.summarize_blocks(protocol, sessions)
    with pytest.raises(ValueError):
        analysis.timing_analysis(protocol, sessions)


@pytest.mark.parametrize("where", ["job", "profile"])
def test_profiled_session_is_ineligible_for_unprofiled_latency(tmp_path, where):
    protocol, sessions = _comparison_fixture(tmp_path)
    sessions[0][where]["profiler"] = True
    with pytest.raises(ProtocolError, match="Profiler"):
        packet.summarize_blocks(protocol, sessions)
    with pytest.raises(ValueError):
        analysis.timing_analysis(protocol, sessions)


def test_tensor_fingerprint_cannot_differ_under_identical_input_file_identity(tmp_path):
    protocol, sessions = _comparison_fixture(tmp_path)
    sessions[0]["input_sha256"] = "changed-tensor"
    with pytest.raises(ProtocolError, match="input tensor"):
        packet.summarize_blocks(protocol, sessions)
    with pytest.raises(ValueError, match="input tensor"):
        analysis.timing_analysis(protocol, sessions)


def test_duplicate_session_is_rejected_instead_of_overwriting_one_attempt(tmp_path):
    protocol, sessions = _comparison_fixture(tmp_path)
    sessions.append(deepcopy(sessions[0]))
    with pytest.raises(ProtocolError, match="Duplicate"):
        packet.summarize_blocks(protocol, sessions)
    with pytest.raises(ValueError, match="retained outcome"):
        analysis.timing_analysis(protocol, sessions)


def test_analysis_ratios_count_blocks_and_keep_same_weight_control_distinct(tmp_path):
    protocol, sessions = _comparison_fixture(tmp_path)
    rows, summaries = analysis.timing_analysis(protocol, sessions)
    assert len(rows) == 5 and all(row["valid"] for row in rows)
    conv = {row["comparison"]: row for row in rows if row["kind"] == "conv"}
    assert set(conv) == {"original_dense/factor", "materialized/factor", "original_dense/materialized"}
    assert conv["materialized/factor"]["wall_p50_ms_ratio"] == 2.
    assert conv["original_dense/materialized"]["wall_p50_ms_ratio"] == 1.
    assert all(summary["planned_blocks"] == summary["valid_blocks"] == 1 for summary in summaries)
    assert all(summary["wall_p50_ms_ratio"]["n"] == 1 for summary in summaries)
    assert all(summary["wall_p50_ms_ratio"]["median_interval"] is None for summary in summaries)


@pytest.mark.parametrize("section,field,value", [("admission", "status", "admission_failed"),
    ("admission", "before_worker_spawn", False), ("admission", "session_id", "other"),
    ("admission", "protocol_sha256", "other"), ("post_exit_observation", "worker_exited", False),
    ("post_exit_observation", "worker_returncode", 1), ("post_exit_observation", "status", "admission_failed"),
    ("profile", "threads", 8), ("profile", "precision", "float16"),
    ("profile", "matmul_tf32", True), ("profile", "cudnn_tf32", True),
    ("profile", "runtime", "eager"), ("profile", "cudnn_benchmark", True),
    ("profile", "cudnn_deterministic", False), ("profile", "batch_shape_checked", False)])
def test_success_label_cannot_bypass_admission_or_frozen_precision(tmp_path, section, field, value):
    protocol, sessions = _comparison_fixture(tmp_path)
    sessions[0][section][field] = value
    with pytest.raises(ValueError):
        analysis.timing_analysis(protocol, sessions)


def _recovered_fixture(tmp_path):
    from fedcore.experiments import runner
    first = tmp_path / "first"
    directory = first / "recovery/toy-ce/update-400"
    directory.mkdir(parents=True)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        model = nn.Linear(2, 3).eval()
    architecture, checkpoint, artifact = (directory / name for name in ("architecture.pt", "state_dict.pt", "model.pt"))
    torch.save(model, architecture)
    torch.save(model.state_dict(), checkpoint)
    torch.jit.trace(model, torch.ones(1, 2)).save(str(artifact))
    record = lambda path: {"path": str(path), "sha256": timing.measurement.file_hash(path)}
    row = {"update": 400, "model_hash": runner.model_state_hash(model),
           "architecture": record(architecture), "checkpoint": record(checkpoint), "artifact": record(artifact)}
    curve = {"name": "toy", "objective": "ce", "status": "succeeded",
             "training": {"training_steps": 400}, "checkpoints": [row]}
    manifest = {"curves": [curve]}
    (first / "recovery.json").write_text(json.dumps(manifest), encoding="utf-8")
    return first, model, manifest


def test_recovered_source_is_exact_ce400_checkpoint_without_source_mutation(tmp_path):
    first, model, manifest = _recovered_fixture(tmp_path)
    paths = [first / "recovery.json"] + [first / f"recovery/toy-ce/update-400/{name}"
              for name in ("architecture.pt", "state_dict.pt", "model.pt")]
    before = {str(path): path.read_bytes() for path in paths}
    recovered, row = packet.recovered_checkpoint(first, "toy")
    assert row == manifest["curves"][0]["checkpoints"][0]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(recovered.state_dict()[key], value, rtol=0, atol=0)
    assert {str(path): path.read_bytes() for path in paths} == before


@pytest.mark.parametrize("mutation", ["wrong_objective", "wrong_updates", "wrong_checkpoint", "failed_curve", "duplicate_curve", "duplicate_checkpoint"])
def test_recovered_source_never_substitutes_other_objective_or_work_count(tmp_path, mutation):
    first, _, manifest = _recovered_fixture(tmp_path)
    curve = manifest["curves"][0]
    if mutation == "wrong_objective":
        curve["objective"] = "ce_kl"
    elif mutation == "wrong_updates":
        curve["training"]["training_steps"] = 399
    elif mutation == "wrong_checkpoint":
        curve["checkpoints"][0]["update"] = 100
    elif mutation == "failed_curve":
        curve["status"] = "technical_failure"
    elif mutation == "duplicate_curve":
        manifest["curves"].append(deepcopy(curve))
    else:
        curve["checkpoints"].append(deepcopy(curve["checkpoints"][0]))
    (first / "recovery.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ProtocolError):
        packet.recovered_checkpoint(first, "toy")


def test_checkpoint_corruption_is_rejected_even_with_matching_file_digest(tmp_path):
    first, _, manifest = _recovered_fixture(tmp_path)
    row = manifest["curves"][0]["checkpoints"][0]
    state = torch.load(row["checkpoint"]["path"], map_location="cpu", weights_only=True)
    state["weight"][0, 0] += 1.
    torch.save(state, row["checkpoint"]["path"])
    row["checkpoint"]["sha256"] = timing.measurement.file_hash(row["checkpoint"]["path"])
    (first / "recovery.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ProtocolError, match="checkpoint tensors differ"):
        packet.recovered_checkpoint(first, "toy")


def test_trusted_archive_cannot_escape_its_owning_directory(tmp_path):
    archive = tmp_path / "unrelated.pt"
    archive.write_bytes(b"valid bytes in unrelated directory")
    record = {"path": str(archive), "sha256": timing.measurement.file_hash(archive)}
    with pytest.raises(ProtocolError, match="escaped"):
        packet.checked_file(record, root=tmp_path / "trusted")


def test_resource_violation_cannot_succeed_on_zero_exit_and_post_observer_follows_reap(tmp_path, monkeypatch):
    protocol, sessions = _comparison_fixture(tmp_path)
    protocol["admission"]["total_extra_wait_budget_seconds"] = 600.
    job = sessions[0]["job"]
    job["directory"] = str(tmp_path / "attempt")
    ledger = {"attempts": [], "charged_seconds": 23., "admission_wait_seconds": 0.,
              "extra_noise_wait_seconds": 0., "new_training_updates": 0}
    events = []

    class Process:
        pid = 919
        alive = True
        waits = 0

        def poll(self):
            return None if self.alive else 0

        def wait(self, timeout=None):
            assert not self.alive
            self.waits += 1
            events.append("reaped")
            return 0

        def children(self, recursive=True):
            return []

        def is_running(self):
            return self.alive

        def memory_info(self):
            return SimpleNamespace(rss=13 * 1024 ** 3)

    process = Process()

    def admit(*args, **kwargs):
        if events:
            assert events == ["admitted", "spawned", "terminated", "reaped", "drained"]
            events.append("post_observed")
        else:
            events.append("admitted")
        return {"status": "admitted", "elapsed_seconds": 2., "extra_wait_seconds": 0.,
                "mandatory_observation_seconds": 2., "samples": []}

    def spawn(*args, **kwargs):
        assert events == ["admitted"]
        events.append("spawned")
        return process

    def terminate(owned):
        assert owned is process
        process.alive = False
        events.append("terminated")

    def drain(duration):
        assert duration == 2.2 and events[-1] == "reaped"
        events.append("drained")

    class MissingProcess(Exception):
        pass

    fake_psutil = SimpleNamespace(Process=lambda _: process, NoSuchProcess=MissingProcess,
        virtual_memory=lambda: SimpleNamespace(available=32 * 1024 ** 3))
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(packet, "wait_for_quiet", admit)
    monkeypatch.setattr(packet.subprocess, "Popen", spawn)
    monkeypatch.setattr(packet, "_terminate_owned", terminate)
    monkeypatch.setattr(packet.time, "sleep", drain)
    result = packet.monitored_session(tmp_path, protocol, job, ledger)
    assert result["status"] == "technical_failure"
    assert events == ["admitted", "spawned", "terminated", "reaped", "drained", "post_observed"]
    saved = packet.read_json(tmp_path / "ledger.json")
    resources = packet.read_json(tmp_path / "attempt/resources.json")
    assert resources["returncode"] == 0
    assert resources["violation"] == "owned recursive RSS above12GiB"
    assert saved["charged_seconds"] == pytest.approx(23. + resources["elapsed_seconds"])
    assert len(saved["attempts"]) == process.waits == 1
    assert saved["attempts"][0]["status"] == "technical_failure"
    assert resources["post_exit_observation"]["worker_exited"] is True


def test_termination_never_reaps_popen_root_via_psutil(tmp_path, monkeypatch):
    events = []

    class Owned:
        def __init__(self, name):
            self.name, self.alive = name, True

        def terminate(self):
            events.append(f"terminate-{self.name}")
            self.alive = False

        def kill(self):
            events.append(f"kill-{self.name}")
            self.alive = False

        def is_running(self):
            return self.alive

        def children(self, recursive=True):
            return [child]

    root, child = Owned("root"), Owned("child")

    class MissingProcess(Exception):
        pass

    def wait_procs(processes, timeout):
        assert processes == [child] and root not in processes
        events.append("reap-children-only")
        return [child], []

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(NoSuchProcess=MissingProcess, wait_procs=wait_procs))
    packet._terminate_owned(root)
    assert events == ["terminate-child", "terminate-root", "reap-children-only"]
