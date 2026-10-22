"""Contract tests for exact-factor materialization and finite local execution."""
import json

import pytest
import torch
from torch import nn

from fedcore.experiments import svd_third_package as packet
from fedcore.experiments.protocol import ProtocolError
from fedcore.experiments.scenarios import DigitsCNN
from fedcore.experiments.svd_third_timing import measure
from fedcore.models.network_impl.decomposed_layers import DecomposedConv2d, DecomposedLinear
from fedcore.tools import svd_third_analysis as analysis


def factor_model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        model = DigitsCNN().eval()
        conv = DecomposedConv2d(model.features[2], decomposing_mode=None, compose_mode="two_layers")
        conv.set_U_S_Vh(torch.randn(16, 8) * .1, torch.ones(8), torch.randn(8, 144) * .1)
        conv.compose_weight_for_inference()
        model.features[2] = conv
        linear = DecomposedLinear(model.classifier[0], decomposing_mode=None, compose_mode="two_layers")
        linear.set_U_S_Vh(torch.randn(64, 16) * .02, torch.ones(16), torch.randn(16, 1024) * .02)
        linear.compose_weight_for_inference()
        model.classifier[0] = linear
    return model


def test_independent_conv_algebra_preserved_and_whole_storage_exact():
    model = factor_model()
    old = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    dense = packet.materialize_conv2(model)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert packet.exact_storage(model)["tensor_bytes"] == 78312
    assert packet.exact_storage(dense)["tensor_bytes"] == 82408
    conv = model.features[2]
    expected = (conv.U.detach() @ conv.Vh.detach()).reshape(16, 16, 3, 3)
    assert torch.equal(expected, dense.features[2].weight)
    assert type(dense.features[2]) is nn.Conv2d
    assert isinstance(dense.classifier[0], DecomposedLinear)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in old.items())
    for batch in (1, 64):
        x = torch.linspace(0, 1, batch * 64).reshape(batch, 1, 8, 8)
        with torch.inference_mode():
            torch.testing.assert_close(model(x), dense(x), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("stage,count,units", [("pilot", 8, 5), ("main", 20, 11)])
def test_schedule_complete_balanced_deterministic_and_per_block_controls(stage, count, units):
    rows = packet.schedule(stage)
    assert rows == packet.schedule(stage)
    assert len(rows) == count * units == len({r["unit_id"] for r in rows})
    for subject in {r["subject"] for r in rows}:
        group = [r for r in rows if r["subject"] == subject]
        assert {r["block"] for r in group} == set(range(count))
        assert sum(r["order"] == ["a", "b"] for r in group) == count // 2
    for block in range(count):
        controls = [r for r in rows if r["block"] == block and r["kind"] == "control"]
        assert {r["subject"] for r in controls} == set(analysis.GRAPHS)


def test_measure_retains_partial_raw_vector_when_model_raises():
    class Failing:
        count = 0
        def __call__(self, x):
            self.count += 1
            if self.count == 3:
                raise RuntimeError("planned test failure")
            return x
    raw = {metric: [] for metric in analysis.METRICS}
    with pytest.raises(RuntimeError, match="planned"):
        measure(Failing(), torch.ones(1), torch.ones(1), "cpu", 10, 50, raw)
    assert len(raw["wall_ms"]) == 2
    assert raw["full_path_ms"] == raw["cuda_event_ms"] == raw["throughput_seconds"] == []


def ledger():
    return {"charged_seconds": 0., "compute_by_stage_seconds": {}, "admission_wait_seconds": 0.,
            "extra_noise_wait_seconds": 0., "mandatory_drain_seconds": 0., "attempts": []}


def test_admission_failure_retained_and_never_spawns_cuda_owner(tmp_path, monkeypatch):
    job = packet.job_for(tmp_path, "pilot", "cuda:0", 1, "frozen-protocol")
    counters = ledger()
    monkeypatch.setattr(packet, "wait_for_quiet", lambda *a, **k: {
        "status": "admission_failed", "elapsed_seconds": 60., "extra_wait_seconds": 60.,
        "samples": [{"gpu": "busy"}]})
    monkeypatch.setattr(packet.subprocess, "Popen", lambda *a, **k: pytest.fail("Denied admission must not create CUDA owner"))
    result = packet.monitored(tmp_path, job, counters)
    assert result["status"] == "admission_failed"
    assert counters["charged_seconds"] == 0.
    assert counters["extra_noise_wait_seconds"] == 60.
    assert len(counters["attempts"]) == 1
    saved = json.loads((tmp_path / "attempts/pilot-cuda0-b1/admission.json").read_text())
    assert saved["before_worker_spawn"] is True
    assert saved["protocol_sha256"] == "frozen-protocol"
    with pytest.raises(ProtocolError, match="retries"):
        packet.monitored(tmp_path, job, counters)


def test_exhausted_compute_budget_preserves_attempt_without_observation(tmp_path, monkeypatch):
    counters = ledger()
    counters["charged_seconds"] = packet.COMPUTE_SECONDS
    monkeypatch.setattr(packet, "wait_for_quiet", lambda *a, **k: pytest.fail("Exhausted budget must not wait"))
    result = packet.monitored(tmp_path, packet.job_for(tmp_path, "main", "cpu", 64, "protocol"), counters)
    assert result["status"] == "not_run_budget"
    assert (tmp_path / "attempts/main-cpu-b64/result.json").is_file()


@pytest.mark.parametrize("seconds", [-1., float("nan"), float("inf"), 7200.])
def test_freeze_invalid_preflight_never_creates_archive(tmp_path, seconds):
    path = tmp_path / "archive"
    with pytest.raises(ProtocolError):
        packet.freeze(path, tmp_path / "first", seconds)
    assert not path.exists()
