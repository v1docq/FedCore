"""Independent references for the inference-only third SVD packet.

Handwritten evidence and elementary algebra exercise the protocol without
launching its measurement series. These tests never initialize CUDA.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter, defaultdict
import hashlib
import json
import random
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fedcore.experiments import svd_third_package as packet
from fedcore.experiments import svd_third_timing as timing
from fedcore.experiments.protocol import ProtocolError
from scripts import analyze_svd_third_package as analysis


GRAPH_NAMES = ("svd_factor", "svd_materialized", "weighted_factor",
               "weighted_materialized", "dense")


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _controls(stage="pilot", device="cpu"):
    """Construct the fixed matrix independently of production scheduling."""
    n, calls = (8, 10) if stage == "pilot" else (20, 20)
    rows = []
    for graph in GRAPH_NAMES:
        for block in range(n):
            unit = {"unit_id": f"independent-{graph}-{block}", "kind": "control",
                    "subject": graph, "block": block,
                    "models": {"a": graph, "b": graph},
                    "order": ["a", "b"] if block % 2 == 0 else ["b", "a"]}
            side = {"wall_ms": [2.] * calls, "full_path_ms": [3.] * calls,
                    "cuda_event_ms": [1.] * calls if device.startswith("cuda") else [],
                    "throughput_seconds": [.25], "source_sha256": f"sha-{graph}",
                    "input_sha256": "identical-fixed-input"}
            rows.append({"unit": unit, "status": "succeeded",
                         "sides": {"a": deepcopy(side), "b": deepcopy(side)}})
    return rows


def _set_ratio(row, value, metric="wall_ms"):
    row["sides"]["a"][metric] = [v * value for v in row["sides"]["b"][metric]]


@pytest.mark.parametrize("stage", ["pilot", "main"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_identical_complete_controls_pass_without_cuda_or_source_access(stage, device, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Pure gate must not initialize CUDA or open model archives")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    rows = _controls(stage, device)
    before = deepcopy(rows)
    result = analysis.control_gate(rows, stage, device)
    assert result["passed"] is True
    assert not result["reasons"]
    assert result["observed_controls"] == result["planned_controls"] == len(rows)
    assert len(result["checks"]) == 5 * (4 if device.startswith("cuda") else 3)
    assert all(check["median"] == check["min"] == check["max"] == 1.
               for check in result["checks"])
    assert rows == before
    assert analysis.control_gate(list(reversed(rows)), stage, device) == result


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "outside", "failed", "unbalanced"])
def test_controls_cannot_replace_failed_or_missing_planned_blocks(mutation):
    rows = _controls()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(deepcopy(rows[-1]))
    elif mutation == "outside":
        rows[-1]["unit"]["block"] = 99
    elif mutation == "failed":
        rows[-1]["status"] = "technical_failure"
    else:
        rows[0]["unit"]["order"] = ["b", "a"]
    assert analysis.control_gate(rows, "pilot", "cpu")["passed"] is False


@pytest.mark.parametrize("mutation", ["input", "source", "model_label", "missing_side", "same_side_twice"])
def test_gate_rejects_label_and_identity_substitution(mutation):
    rows = _controls()
    row = rows[3]
    if mutation == "input":
        row["sides"]["b"]["input_sha256"] = "different-input"
    elif mutation == "source":
        row["sides"]["b"]["source_sha256"] = "different-model"
    elif mutation == "model_label":
        row["unit"]["models"]["b"] = "weighted_factor"
    elif mutation == "missing_side":
        del row["sides"]["b"]
    else:
        row["unit"]["order"] = ["a", "a"]
    assert analysis.control_gate(rows, "pilot", "cpu")["passed"] is False


@pytest.mark.parametrize("bad", [0., -1., float("nan"), float("inf"), True])
@pytest.mark.parametrize("metric", ["wall_ms", "full_path_ms", "cuda_event_ms", "throughput_seconds"])
def test_gate_never_silently_discards_bad_raw_observations(bad, metric):
    rows = _controls(device="cuda:0")
    rows[0]["sides"]["b"][metric][0] = bad
    assert analysis.control_gate(rows, "pilot", "cuda:0")["passed"] is False


@pytest.mark.parametrize("metric", ["wall_ms", "full_path_ms", "cuda_event_ms", "throughput_seconds"])
def test_every_metric_and_every_graph_participate_in_gate(metric):
    rows = _controls(device="cuda:0")
    for row in rows:
        if row["unit"]["subject"] == "weighted_materialized":
            _set_ratio(row, 1.06, metric)
    result = analysis.control_gate(rows, "pilot", "cuda:0")
    assert result["passed"] is False
    assert any("weighted_materialized" in reason and metric in reason for reason in result["reasons"])


@pytest.mark.parametrize("stage,allowed", [("pilot", 1), ("main", 2)])
def test_wide_band_counts_all_fixed_blocks_including_outliers(stage, allowed):
    rows = _controls(stage)
    for row in rows[:allowed]:
        _set_ratio(row, 1.16)
    assert analysis.control_gate(rows, stage, "cpu")["passed"] is True
    _set_ratio(rows[allowed], 1.16)
    assert analysis.control_gate(rows, stage, "cpu")["passed"] is False


def test_balanced_center_does_not_conceal_order_effect():
    rows = _controls()
    for row in rows[:8]:
        _set_ratio(row, 1.06 if row["unit"]["order"] == ["a", "b"] else .94)
    result = analysis.control_gate(rows, "pilot", "cpu")
    assert result["passed"] is False
    check = next(c for c in result["checks"] if c["graph"] == "svd_factor" and c["metric"] == "wall_ms")
    assert check["median"] == pytest.approx(1.)
    assert check["within_15_percent"] == 8
    assert check["ab_over_ba"] == pytest.approx(1.06 / .94)


def test_ratio_is_median_over_median_with_fixed_a_over_b_direction():
    row = _controls(device="cuda:0")[0]
    row["sides"]["a"]["wall_ms"] = list(range(1, 11))
    row["sides"]["b"]["wall_ms"] = list(range(2, 12))
    row["sides"]["a"]["throughput_seconds"] = [.4]
    row["sides"]["b"]["throughput_seconds"] = [.2]
    ratios = analysis.unit_ratios(row, "pilot", "cuda:0")
    assert ratios["wall_ms"] == pytest.approx(5.5 / 6.5)
    assert ratios["throughput_seconds"] == 2.
    reversed_order = deepcopy(row)
    reversed_order["unit"]["order"] = ["b", "a"]
    assert analysis.unit_ratios(reversed_order, "pilot", "cuda:0") == ratios


@pytest.mark.parametrize("stage", ["pilot", "main"])
def test_raw_call_count_is_contractual_and_not_a_population_sample_size(stage):
    row = _controls(stage)[0]
    row["sides"]["a"]["wall_ms"].append(2.)
    with pytest.raises(ValueError, match="incomplete|expected"):
        analysis.unit_ratios(row, stage, "cpu")


def test_cpu_does_not_fabricate_cuda_event_observations():
    row = _controls()[0]
    row["sides"]["a"]["cuda_event_ms"] = [1.] * 10
    with pytest.raises(ValueError, match="CPU CUDA"):
        analysis.unit_ratios(row, "pilot", "cpu")


def test_gate_rejects_unknown_stage():
    with pytest.raises(ValueError, match="stage"):
        analysis.control_gate([], "replacement", "cpu")


@pytest.mark.parametrize("stage,blocks", [("pilot", 8), ("main", 20)])
def test_schedule_conserves_whole_graph_matrix_balance_and_contiguous_blocks(stage, blocks):
    state = random.getstate()
    rows = packet.schedule(stage)
    assert rows == packet.schedule(stage)
    assert random.getstate() == state
    expected_pairs = {("control", graph, graph, graph) for graph in GRAPH_NAMES}
    if stage == "main":
        expected_pairs |= {("comparison", f"{name}_factor_vs_materialized", f"{name}_factor", f"{name}_materialized")
                           for name in ("svd", "weighted")}
        expected_pairs |= {("comparison", f"dense_vs_{name}", "dense", name) for name in GRAPH_NAMES[:4]}
    expected = {(kind, subject, a, b, block)
                for kind, subject, a, b in expected_pairs for block in range(blocks)}
    observed = {(r["kind"], r["subject"], r["models"]["a"], r["models"]["b"], r["block"]) for r in rows}
    assert observed == expected
    assert len(rows) == len({r["unit_id"] for r in rows}) == len(expected)
    by_pair, by_block = defaultdict(list), defaultdict(list)
    for index, row in enumerate(rows):
        assert sorted(row["order"]) == ["a", "b"]
        by_pair[row["kind"], row["subject"]].append("".join(row["order"]))
        by_block[row["block"]].append((index, row))
    assert all(Counter(orders) == {"ab": blocks // 2, "ba": blocks // 2} for orders in by_pair.values())
    for members in by_block.values():
        positions = [index for index, _ in members]
        assert positions == list(range(positions[0], positions[0] + len(expected_pairs)))
        assert [row["position"] for _, row in members] == list(range(len(expected_pairs)))
    # Controls move through the unit sequence, rather than all occurring only before the comparisons.
    if stage == "main":
        assert len({r["position"] for r in rows if r["kind"] == "control" and r["subject"] == "dense"}) == 11


def _factor_model(bias=True):
    from fedcore.experiments.scenarios import DigitsCNN
    from fedcore.models.network_impl.decomposed_layers import DecomposedConv2d, DecomposedLinear
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1701)
        model = DigitsCNN().eval()
        for path, rank, cls in (("features.2", 8, DecomposedConv2d), ("classifier.0", 16, DecomposedLinear)):
            base = model.get_submodule(path)
            if not bias:
                base.register_parameter("bias", None)
            layer = cls(base, decomposing_mode=None, compose_mode="two_layers")
            left = torch.randn(base.weight.shape[0], rank) / 8
            right = torch.randn(rank, base.weight.numel() // base.weight.shape[0]) / 8
            layer.set_U_S_Vh(left, torch.ones(rank), right)
            layer.compose_weight_for_inference()
            parent, index = path.split(".")
            getattr(model, parent)[int(index)] = layer
    return model.eval()


@pytest.mark.parametrize("bias", [True, False])
def test_conv_materialization_preserves_independent_operator_bias_linear_and_source(bias, monkeypatch):
    model = _factor_model(bias)
    original = {key: value.clone() for key, value in model.state_dict().items()}
    torch_rng, python_rng = torch.get_rng_state().clone(), random.getstate()

    def forbidden(*args, **kwargs):
        pytest.fail("CPU materialization must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "device_count", forbidden)
    monkeypatch.setattr(torch.cuda, "get_rng_state", forbidden)
    result = packet.materialize_conv2(model)
    assert type(result.features[2]) is nn.Conv2d
    assert type(result.features[0]) is nn.Conv2d
    assert result is not model and result.classifier[0] is not model.classifier[0]
    assert result.classifier[0].U.shape == (64, 16)
    assert result.classifier[0].Vh.shape == (16, 1024)
    conv = model.features[2]
    # Independent reconstruction bypasses both to_standard_module and factor_matrix.
    weight = torch.matmul(conv.U.detach(), conv.Vh.detach()).reshape(16, 16, 3, 3)
    torch.testing.assert_close(result.features[2].weight, weight, rtol=0, atol=0)
    if bias:
        torch.testing.assert_close(result.features[2].bias, conv.bias, rtol=0, atol=0)
    else:
        assert result.features[2].bias is None
    x = torch.arange(128, dtype=torch.float32).reshape(2, 1, 8, 8) / 128
    with torch.inference_mode():
        activation = torch.relu(torch.nn.functional.conv2d(x, model.features[0].weight, model.features[0].bias, padding=1))
        activation = torch.relu(torch.nn.functional.conv2d(activation, weight, conv.bias, padding=1)).flatten(1)
        linear = model.classifier[0]
        hidden = torch.relu(torch.nn.functional.linear(torch.nn.functional.linear(activation, linear.Vh), linear.U, linear.bias))
        expected = torch.nn.functional.linear(hidden, model.classifier[2].weight, model.classifier[2].bias)
        torch.testing.assert_close(result(x), expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(model(x), expected, rtol=1e-4, atol=1e-5)
    for key, value in original.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        if not key.startswith("features.2."):
            torch.testing.assert_close(result.state_dict()[key], value, rtol=0, atol=0)
    assert random.getstate() == python_rng
    torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
    if bias:
        # Count unique complete storage independently of measurement.tensor_state_bytes.
        count = lambda m: sum({tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
                               for tensor in m.state_dict().values()}.values())
        assert count(model) == 78312
        assert count(result) == 82408
    with torch.no_grad():
        result.classifier[0].U.add_(1.)
    torch.testing.assert_close(model.classifier[0].U, original["classifier.0.U"], rtol=0, atol=0)


@pytest.mark.parametrize("mutation", ["conv1", "conv_rank", "linear_rank"])
def test_materialization_rejects_other_architecture_or_rank_contract(mutation):
    model = _factor_model()
    if mutation == "conv1":
        model.features[0] = nn.Identity()
    elif mutation == "conv_rank":
        model.features[2].U = nn.Parameter(model.features[2].U[:, :7].detach().clone())
    else:
        model.classifier[0].U = nn.Parameter(model.classifier[0].U[:, :15].detach().clone())
    with pytest.raises(ProtocolError):
        packet.materialize_conv2(model)


def test_graph_loader_checks_frozen_bytes_and_uses_unicode_binary_stream(tmp_path, monkeypatch):
    path = tmp_path / "модель с пробелами.pt"
    path.write_bytes(b"independent artifact bytes")
    calls = []
    model = nn.Identity()

    def load(stream, *, map_location):
        assert hasattr(stream, "read") and stream.read() == b"independent artifact bytes"
        assert map_location == "cpu"
        calls.append(stream.name)
        return model

    monkeypatch.setattr(torch.jit, "load", load)
    record = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert timing._load(record, "cpu") is model
    assert calls == [str(path)]
    path.write_bytes(b"different graph")
    with pytest.raises(ProtocolError, match="changed"):
        timing._load(record, "cpu")
    assert calls == [str(record["path"])]


def test_worker_rejects_foreign_admission_before_cuda_or_protocol_access(tmp_path, monkeypatch):
    admission = tmp_path / "admission.json"
    admission.write_text(json.dumps({"status": "admitted", "session_id": "foreign",
        "protocol_sha256": "frozen", "before_worker_spawn": True}), encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("An unbound admission must precede every CUDA query")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    with pytest.raises(ProtocolError, match="admission token"):
        timing.timing_worker({"admission_path": str(admission), "session_id": "intended",
                              "protocol_sha256": "frozen", "device": "cuda:0"})


def test_mid_side_exception_keeps_completed_call_vectors_for_failure_record(monkeypatch):
    calls = []

    def failing_model(x):
        calls.append("call")
        if len(calls) == 2:
            raise RuntimeError("independent injected failure")
        return x

    clock = iter((1000000, 2000000, 3000000))
    monkeypatch.setattr(timing.time, "perf_counter_ns", lambda: next(clock))
    raw = {"wall_ms": [], "full_path_ms": [], "cuda_event_ms": [], "throughput_seconds": []}
    x = torch.ones(1, 2)
    with pytest.raises(RuntimeError, match="injected failure"):
        timing.measure(failing_model, x, x, "cpu", 2, 1, raw=raw)
    assert raw == {"wall_ms": [1.], "full_path_ms": [], "cuda_event_ms": [], "throughput_seconds": []}


def test_resource_violation_dominates_zero_exit_and_postcheck_follows_reap(tmp_path, monkeypatch):
    events = []

    class Process:
        pid = 3719
        alive = True

        def poll(self):
            return None if self.alive else 0

        def wait(self, timeout=None):
            assert not self.alive
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
            events.append("postcheck")
        else:
            events.append("admitted")
        return {"status": "admitted", "elapsed_seconds": 2., "extra_wait_seconds": 0., "samples": []}

    def spawn(*args, **kwargs):
        assert events == ["admitted"]
        events.append("spawned")
        return process

    def terminate(owned, worker):
        assert owned is process
        assert worker is process
        process.alive = False
        events.append("terminated")
        return worker.wait(timeout=30)

    def sleep(seconds):
        assert seconds == 2.2 and events[-1] == "reaped"
        events.append("drained")

    class MissingProcess(Exception):
        pass

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=lambda _: process,
        NoSuchProcess=MissingProcess, Error=MissingProcess,
        virtual_memory=lambda: SimpleNamespace(available=32 * 1024 ** 3)))
    monkeypatch.setattr(packet, "wait_for_quiet", admit)
    monkeypatch.setattr(packet.subprocess, "Popen", spawn)
    monkeypatch.setattr(packet, "stop_owned", terminate)
    monkeypatch.setattr(packet.time, "sleep", sleep)
    ledger = {"attempts": [], "charged_seconds": 7., "extra_noise_wait_seconds": 0.,
              "admission_wait_seconds": 0., "mandatory_drain_seconds": 0., "compute_by_stage_seconds": {}}
    job = {"directory": str(tmp_path / "attempt"), "session_id": "independent",
           "stage": "main", "device": "cpu", "protocol_sha256": "frozen"}
    result = packet.monitored(tmp_path, job, ledger)
    assert result["status"] == "technical_failure"
    assert events == ["admitted", "spawned", "terminated", "reaped", "drained", "postcheck"]
    assert len(ledger["attempts"]) == 1
    assert ledger["attempts"][0]["status"] == "technical_failure"
    resources = json.loads((tmp_path / "attempt/resources.json").read_text(encoding="utf-8"))
    assert resources["returncode"] == 0
    assert "RSS" in resources["violation"]
    assert ledger["charged_seconds"] == pytest.approx(7. + resources["elapsed_seconds"])
    assert ledger["mandatory_drain_seconds"] == 2.2


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")
    return {"relative_path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _records(units, stage, device, batch):
    rows = []
    latency = {"svd_factor": 2., "svd_materialized": 1., "weighted_factor": 3.,
               "weighted_materialized": 1.5, "dense": 4.}
    for unit in units:
        calls = 10 if stage == "pilot" else 20
        sides = {}
        for side, graph in unit["models"].items():
            value = latency[graph]
            sides[side] = {"wall_ms": [value] * calls, "full_path_ms": [3 * value] * calls,
                "cuda_event_ms": [value / 2] * calls if device.startswith("cuda") else [],
                "throughput_seconds": [value / 4], "samples_per_second": batch * 50 / (value / 4),
                "source_sha256": f"source-{graph}", "input_sha256": f"input-{batch}"}
        rows.append({"unit": deepcopy(unit), "status": "succeeded", "sides": sides})
    return rows


def _archive(tmp_path, mutation=None):
    profiles = [("cpu", 1), ("cpu", 64), ("cuda:0", 1), ("cuda:0", 64)]
    protocol = {"profiles": [{"device": d, "batch": b} for d, b in profiles],
        "schedules": {stage: packet.schedule(stage) for stage in ("pilot", "main")},
        "prepared": {"graphs": {graph: {"sha256": f"source-{graph}",
            "path": f"Z:/unavailable-originals/{graph}.pt"} for graph in GRAPH_NAMES},
            "input": {"path": "Z:/unavailable-originals/input.pt",
                      "tensor_sha256_by_batch": {str(b): f"input-{b}" for b in (1, 64)}}}}
    protocol["protocol_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    _write_json(tmp_path / "protocol.json", protocol)
    entries = []
    for device, batch in profiles:
        entry = {"device": device, "batch": batch, "stages": {}}
        for stage in ("pilot", "main"):
            job = {"protocol_sha256": protocol["protocol_sha256"], "stage": stage, "device": device,
                   "batch": batch, "session_id": f"{stage}-{device}-{batch}"}
            result = {"job": job, "status": "succeeded", "records": _records(protocol["schedules"][stage], stage, device, batch),
                "profile": {"dtype": "float32", "threads": 1, "runtime": "torch.jit", "profiler": False,
                            "tf32": False, "cudnn_benchmark": False, "cudnn_deterministic": True},
                "control_runtime": "two independently loaded TorchScript objects from identical bytes, both warmed",
                "admission": {"status": "admitted", "session_id": job["session_id"],
                              "protocol_sha256": protocol["protocol_sha256"], "before_worker_spawn": True}}
            directory = tmp_path / "attempts" / f"{stage}-{device.replace(':', '')}-{batch}"
            resource_path = directory / "resources.json"
            resource = {"returncode": 0, "violation": None, "elapsed_seconds": .1, "samples": [],
                        "admission": deepcopy(result["admission"]),
                        "post_exit": {"status": "admitted", "worker_exited": True, "worker_returncode": 0}}
            resource_record = _write_json(resource_path, resource)
            result["resources"] = {"sha256": resource_record["sha256"],
                                   "path": f"Z:/unavailable-originals/{directory.name}/resources.json"}
            if mutation is not None:
                mutation(device, batch, stage, result)
            path = directory / "result.json"
            entry["stages"][stage] = _write_json(path, result)
            entry["stages"][stage]["relative_path"] = path.relative_to(tmp_path).as_posix()
        entries.append(entry)
    _write_json(tmp_path / "run.json", {"protocol_sha256": protocol["protocol_sha256"], "profiles": entries})
    return protocol, entries


def test_portable_replay_honors_pilot_before_alphabetically_serialized_main_and_independent_ratios(tmp_path):
    _archive(tmp_path)
    # Canonical JSON sort places "main" before "pilot". Analysis must use protocol order.
    run = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert list(run["profiles"][0]["stages"]) == ["main", "pilot"]
    result = analysis.analyze(tmp_path, tmp_path / "analysis")
    assert len(result["profiles"]) == 4
    for profile in result["profiles"]:
        assert profile["stages"]["pilot"]["gate"]["passed"] is True
        assert profile["stages"]["main"]["ranking_eligible"] is True
    for row in result["comparisons"]:
        assert row["n"] == 20
        assert row["ab"]["n"] == row["ba"]["n"] == 10
        assert row["ranking_eligible"] is True
        if row["subject"] in ("svd_factor_vs_materialized", "weighted_factor_vs_materialized"):
            assert row["min"] == row["median"] == row["max"] == 2.
        assert "median_interval" not in row
    assert "no iid or population CI" in result["scope"]


def test_partial_failure_is_retained_without_ranking_or_block_replacement(tmp_path):
    def mutation(device, batch, stage, result):
        if (device, batch, stage) == ("cuda:0", 1, "main"):
            result["status"] = "technical_failure"
            last = result["records"][-1]
            last["status"] = "incomplete"
            keep = last["unit"]["order"][0]
            last["sides"] = {keep: last["sides"][keep]}
            for metric in analysis.METRICS:
                last["sides"][keep][metric] = last["sides"][keep][metric][:1] if metric == "wall_ms" else []

    _archive(tmp_path, mutation)
    result = analysis.analyze(tmp_path, tmp_path / "analysis")
    failed = next(p for p in result["profiles"] if (p["device"], p["batch"]) == ("cuda:0", 1))
    assert failed["stages"]["main"]["status"] == "technical_failure"
    assert failed["stages"]["main"]["complete"] is False
    assert failed["stages"]["main"]["ranking_eligible"] is False
    assert all(not r["ranking_eligible"] for r in result["comparisons"] if (r["device"], r["batch"]) == ("cuda:0", 1))
    assert all(p["stages"]["main"]["ranking_eligible"] for p in result["profiles"] if p is not failed)


def test_failed_pilot_blocks_only_its_own_device_batch_profile(tmp_path):
    def mutation(device, batch, stage, result):
        if (device, batch, stage) == ("cuda:0", 1, "pilot"):
            for row in result["records"]:
                if row["unit"]["subject"] == "svd_materialized":
                    _set_ratio(row, 1.06)

    _archive(tmp_path, mutation)
    run_path = tmp_path / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    failed_entry = next(p for p in run["profiles"] if (p["device"], p["batch"]) == ("cuda:0", 1))
    del failed_entry["stages"]["main"]
    _write_json(run_path, run)
    result = analysis.analyze(tmp_path, tmp_path / "analysis")
    failed = next(p for p in result["profiles"] if (p["device"], p["batch"]) == ("cuda:0", 1))
    assert failed["stages"]["pilot"]["gate"]["passed"] is False
    assert "main" not in failed["stages"]
    assert all(p["stages"]["main"]["ranking_eligible"] for p in result["profiles"] if p is not failed)
    assert not any((r["device"], r["batch"]) == ("cuda:0", 1) for r in result["comparisons"])


@pytest.mark.parametrize("mutation", ["source", "input", "duplicate", "reorder"])
def test_raw_provenance_tampering_is_fatal_instead_of_noise_filtering(tmp_path, mutation):
    def change(device, batch, stage, result):
        if (device, batch, stage) != ("cpu", 1, "main"):
            return
        if mutation in ("source", "input"):
            result["records"][0]["sides"]["a"][f"{mutation}_sha256"] = "changed"
        elif mutation == "duplicate":
            result["records"].append(deepcopy(result["records"][-1]))
        else:
            result["records"][0], result["records"][1] = result["records"][1], result["records"][0]

    _archive(tmp_path, change)
    with pytest.raises(ValueError):
        analysis.analyze(tmp_path, tmp_path / "analysis")


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "escape"])
def test_portable_archive_rejects_profile_omission_duplication_and_external_result(tmp_path, mutation):
    _archive(tmp_path)
    run_path = tmp_path / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        run["profiles"].pop()
    elif mutation == "duplicate":
        run["profiles"].append(deepcopy(run["profiles"][0]))
    else:
        run["profiles"][0]["stages"]["pilot"]["relative_path"] = "../outside-result.json"
    _write_json(run_path, run)
    with pytest.raises(ValueError):
        analysis.analyze(tmp_path, tmp_path / "analysis")


def test_script_replays_without_site_packages_or_original_artifacts(tmp_path):
    from fedcore.tools import svd_third_analysis
    assert analysis.analyze is svd_third_analysis.analyze
    _archive(tmp_path)
    script = Path(__file__).resolve().parents[2] / "scripts/analyze_svd_third_package.py"
    completed = subprocess.run([sys.executable, "-I", "-S", str(script), "--root", str(tmp_path),
        "--destination", str(tmp_path / "isolated-analysis")], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    result = json.loads((tmp_path / "isolated-analysis/analysis.json").read_text(encoding="utf-8"))
    assert result["new_training_updates"] == 0
    assert len(result["profiles"]) == 4


@pytest.mark.parametrize("mutation", ["missing", "foreign_session", "foreign_protocol", "not_admitted", "after_spawn"])
def test_success_cannot_rank_with_missing_or_foreign_pre_cuda_admission(tmp_path, mutation):
    def change(device, batch, stage, result):
        if (device, batch, stage) != ("cpu", 1, "main"):
            return
        if mutation == "missing":
            del result["admission"]
        elif mutation == "foreign_session":
            result["admission"]["session_id"] = "other-session"
        elif mutation == "foreign_protocol":
            result["admission"]["protocol_sha256"] = "other-protocol"
        elif mutation == "not_admitted":
            result["admission"]["status"] = "admission_failed"
        else:
            result["admission"]["before_worker_spawn"] = False

    _archive(tmp_path, change)
    with pytest.raises(ValueError, match="admission|pre-CUDA|pre-spawn"):
        analysis.analyze(tmp_path, tmp_path / "analysis")


@pytest.mark.parametrize("mutation", ["exit", "violation", "post_failed", "still_alive", "missing_marker", "resource_admission", "resource_digest"])
def test_success_cannot_override_failed_resource_or_post_exit_proof(tmp_path, mutation):
    _archive(tmp_path)
    run_path = tmp_path / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    record = run["profiles"][0]["stages"]["main"]
    result_path = tmp_path / record["relative_path"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    resource_path = result_path.with_name("resources.json")
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    if mutation == "exit":
        resource["returncode"] = 7
    elif mutation == "violation":
        resource["violation"] = "owned recursive RSS above12GiB"
    elif mutation == "post_failed":
        resource["post_exit"]["status"] = "admission_failed"
    elif mutation == "still_alive":
        resource["post_exit"]["worker_exited"] = False
    elif mutation == "missing_marker":
        del resource["post_exit"]["worker_exited"]
    elif mutation == "resource_admission":
        resource["admission"]["session_id"] = "other-session"
    else:
        resource["changed_without_rebinding"] = True
    _write_json(resource_path, resource)
    if mutation != "resource_digest":
        result["resources"]["sha256"] = hashlib.sha256(resource_path.read_bytes()).hexdigest()
    _write_json(result_path, result)
    record["sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    _write_json(run_path, run)
    with pytest.raises(ValueError, match="resource|Resource|post-exit|exit"):
        analysis.analyze(tmp_path, tmp_path / "analysis")
