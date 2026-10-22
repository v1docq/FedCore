"""Behavioral tests of the finite inference-only controller, without CUDA runs."""
from dataclasses import asdict
import json

import pytest

from fedcore.experiments import svd_second_package as packet
from fedcore.experiments.protocol import ProtocolError


def quiet(cpu=0., gpu=0.):
    return {"cpu_percent": cpu, "available_ram_bytes": 8 * 1024**3,
            "gpu": {"utilization_percent": gpu, "physical_free_mib": 4096.}}


class Clock:
    def __init__(self):
        self.seconds = 0.

    def now(self):
        return self.seconds

    def sleep(self, seconds):
        self.seconds += seconds


def test_quiet_window_does_not_charge_required_observations_as_noise_wait():
    clock = Clock()
    result = packet.wait_for_quiet("cuda:0", observer=quiet, monotonic=clock.now, sleep=clock.sleep)
    assert result["status"] == "admitted"
    assert result["elapsed_seconds"] == result["mandatory_observation_seconds"] == 2.
    assert result["extra_wait_seconds"] == 0.
    assert len(result["samples"]) == 3


def test_reset_quiet_streak_counts_all_discarded_wait_as_extra():
    clock = Clock()
    samples = iter([quiet(), quiet(), quiet(gpu=11), quiet(), quiet(), quiet()])
    result = packet.wait_for_quiet("cuda:0", observer=lambda: next(samples), monotonic=clock.now, sleep=clock.sleep)
    assert [row["quiet_streak"] for row in result["samples"]] == [1, 2, 0, 1, 2, 3]
    assert result["elapsed_seconds"] == 5.
    assert result["extra_wait_seconds"] == 3.
    assert result["mandatory_observation_seconds"] == 2.


def test_admission_timeout_bounds_last_sleep_and_retains_every_observation():
    clock = Clock()
    policy = packet.AdmissionPolicy(timeout_seconds=2.5, interval_seconds=1.)
    result = packet.wait_for_quiet("cuda:0", policy, observer=lambda: quiet(gpu=100),
                                   monotonic=clock.now, sleep=clock.sleep)
    assert result["status"] == "admission_failed"
    assert result["elapsed_seconds"] == result["extra_wait_seconds"] == 2.5
    assert len(result["samples"]) == 3


def protocol_for(rows):
    source = {"path": "/trusted/model.pt", "sha256": "frozen-model"}
    inputs = {"path": "/trusted/input.pt", "sha256": "frozen-input",
              "tensor_sha256_by_batch": {"1": "tensor-one", "64": "tensor-64"}}
    return {"protocol_sha256": "protocol", "timing_schedule": rows,
        "timing": {"individual_repeats": 100, "warmup": 50,
                   "throughput_blocks": 3, "calls_per_throughput_block": 50},
        "prepared": {"sources": {"dense": source}, "model_input": inputs,
            "comparisons": {"dense_dense": {"baseline": "dense", "candidate": "dense"}},
            "conv_groups": {"digits_conv2": {"input": inputs,
                "representations": {side: source for side in packet.REPRESENTATIONS}}}}}


def result_for(protocol, row, status="succeeded"):
    job = packet.session_job(protocol, row, "/new/" + row["session_id"])
    return {"job": job, "status": status, "input_sha256": job["input_tensor_sha256"]}


def test_missing_member_and_failed_side_never_leave_survivor_pair():
    rows = [asdict(s) for s in packet.timing_schedule() if s.kind == "control" and s.device == "cpu" and s.batch == 1]
    protocol = protocol_for(rows)
    results = [result_for(protocol, row) for row in rows[:-1]]
    results[0]["status"] = "technical_failure"
    blocks = packet.summarize_blocks(protocol, results)
    assert len(blocks) == 5
    assert sum(row["status"] == "invalid" for row in blocks) == 2
    assert {reason["status"] for block in blocks for reason in block["invalid_reasons"]} == {"missing", "technical_failure"}


def test_failed_original_dense_invalidates_entire_same_weight_conv_triple():
    rows = [asdict(s) for s in packet.timing_schedule() if s.kind == "conv" and s.subject == "digits_conv2"
            and s.device == "cpu" and s.batch == 1 and s.block == 0]
    protocol = protocol_for(rows)
    results = [result_for(protocol, row, "admission_failed" if row["side"] == "original_dense" else "succeeded") for row in rows]
    blocks = packet.summarize_blocks(protocol, results)
    assert len(blocks) == 1 and blocks[0]["status"] == "invalid"
    assert len(blocks[0]["session_ids"]) == 3


@pytest.mark.parametrize("mutation", ["duplicate", "input", "source", "tensor", "profiler"])
def test_block_summary_rejects_unbound_or_profiled_evidence(mutation):
    rows = [asdict(s) for s in packet.timing_schedule() if s.kind == "control" and s.device == "cpu" and s.batch == 1 and s.block == 0]
    protocol = protocol_for(rows)
    results = [result_for(protocol, row) for row in rows]
    if mutation == "duplicate":
        results.append(results[0])
    elif mutation == "tensor":
        results[0]["input_sha256"] = "other-tensor"
    elif mutation == "profiler":
        results[0]["job"]["profiler"] = True
    else:
        key = "input_file_sha256" if mutation == "input" else "source_artifact_sha256"
        results[0]["job"][key] = "other"
    with pytest.raises(ProtocolError):
        packet.summarize_blocks(protocol, results)


def test_refused_admission_retains_ledger_and_never_spawns_worker(tmp_path, monkeypatch):
    rows = [asdict(s) for s in packet.timing_schedule() if s.kind == "control"]
    protocol = protocol_for(rows)
    protocol["admission"] = {"total_extra_wait_budget_seconds": 600., "drain_seconds": 2.2}
    ledger = {"charged_seconds": 0., "admission_wait_seconds": 0., "extra_noise_wait_seconds": 0., "attempts": []}
    job = packet.session_job(protocol, rows[0], tmp_path / "session")
    monkeypatch.setattr(packet, "wait_for_quiet", lambda *args, **kwargs: {
        "status": "admission_failed", "elapsed_seconds": 60., "extra_wait_seconds": 60.,
        "samples": [{"sample": quiet(gpu=90)}], "reasons": ["no quiet GPU"]})
    monkeypatch.setattr(packet.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Rejected admission must not spawn a worker"))
    result = packet.monitored_session(tmp_path, protocol, job, ledger)
    assert result["status"] == "admission_failed"
    saved = json.loads((tmp_path / "ledger.json").read_text())
    assert saved["charged_seconds"] == 0.
    assert saved["extra_noise_wait_seconds"] == 60.
    assert len(saved["attempts"]) == 1
    assert (tmp_path / "session/admission.json").is_file()
    assert (tmp_path / "session/result.json").is_file()


def test_negative_and_nonfinite_admission_policy_are_refused():
    for params in ({"timeout_seconds": float("inf")}, {"interval_seconds": 0}, {"consecutive": 0}):
        with pytest.raises(ProtocolError):
            packet.AdmissionPolicy(**params)
