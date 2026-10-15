"""Independent numerical and evidence oracles for the approved local packet.

These tests use tiny synthetic tensors.  They do not execute the scientific
series, time GPU kernels, or touch historical evidence.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import permutations
import json
import random
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fedcore.experiments import runner, svd_local_package as packet, svd_local_timing as timing
from fedcore.experiments.protocol import ExperimentBundle, ProtocolError, TensorSplit, stable_hash, tensor_hash


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _background(cpu=0., gpu=0.):
    return {"cpu_percent": cpu, "gpu": {"utilization_percent": gpu}}


def test_noise_admission_uses_predeclared_device_specific_thresholds():
    # Boundary equality remains admissible; one excess snapshot invalidates.
    assert timing.noise_reasons(_background(25, 10), _background(25, 10), "cuda:0") == []
    assert timing.noise_reasons(_background(25.01, 0), _background(), "cpu")
    assert timing.noise_reasons(_background(), _background(0, 10.01), "cuda:0")
    assert timing.noise_reasons(_background(0, 99), _background(0, 99), "cpu") == []


def test_missing_gpu_observation_is_explicit_invalid_cuda_evidence():
    missing = {"cpu_percent": 0., "gpu_observation_error": "TimeoutExpired"}
    assert timing.noise_reasons(missing, _background(), "cuda:0")
    assert timing.noise_reasons(missing, missing, "cpu") == []


def test_secondary_batch_smoke_requires_numerical_parity(tmp_path, monkeypatch):
    """A graph may preserve shape while silently changing its other batch."""
    source_model = nn.Linear(2, 2, bias=False).eval()
    with torch.no_grad():
        source_model.weight.copy_(torch.eye(2))
    source = tmp_path / "source.pt"
    torch.jit.trace(source_model, torch.ones(1, 2)).save(str(source))
    inputs = tmp_path / "input.pt"
    torch.save(torch.tensor([[1., 2.]]), inputs)

    class IncorrectOtherBatch(nn.Module):
        def forward(self, x):
            return x if len(x) == 1 else 2 * x

    monkeypatch.setattr(timing, "precision_profile", lambda: None)
    monkeypatch.setattr(timing, "background_snapshot", _background)
    monkeypatch.setattr(timing, "export_model", lambda *args, **kwargs: source)
    monkeypatch.setattr(timing.measurement, "load_artifact", lambda artifact: IncorrectOtherBatch())
    job = {"directory": str(tmp_path / "session"), "device": "cpu", "batch": 1,
           "source_artifact": str(source),
           "source_artifact_sha256": timing.measurement.file_hash(source),
           "input_path": str(inputs), "input_file_sha256": timing.measurement.file_hash(inputs),
           "warmup": 0, "repeats": 1}
    with pytest.raises(AssertionError):
        timing.timing_worker(job)
    assert not (tmp_path / "session/result.json").exists()


def _manual_full_batch_adam(weight, bias, x, y, updates):
    """Closed-form CE gradient and Adam recurrences, without autograd/Adam."""
    weight, bias = weight.clone(), bias.clone()
    moments = [torch.zeros_like(weight), torch.zeros_like(bias)]
    variances = [torch.zeros_like(weight), torch.zeros_like(bias)]
    target = torch.zeros(len(y), len(bias), dtype=x.dtype)
    target[torch.arange(len(y)), y] = 1.
    for step in range(1, updates + 1):
        error = torch.softmax(x @ weight.T + bias, dim=1) - target
        gradients = [error.T @ x / len(x), error.mean(dim=0)]
        for index, parameter in enumerate((weight, bias)):
            moments[index] = .9 * moments[index] + .1 * gradients[index]
            variances[index] = .999 * variances[index] + .001 * gradients[index].square()
            parameter.sub_(1e-4 * (moments[index] / (1 - .9 ** step)) /
                           ((variances[index] / (1 - .999 ** step)).sqrt() + 1e-8))
    return weight, bias, moments, variances


def test_after_step_checkpoint_100_preserves_one_adam_through_update_400():
    # A complete batch makes the independent recurrence insensitive to shuffle.
    model = nn.Linear(2, 3).double()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[.2, -.3], [.5, .1], [-.4, .8]], dtype=torch.float64))
        model.bias.copy_(torch.tensor([.1, -.2, .4], dtype=torch.float64))
    initial = deepcopy(model.state_dict())
    x = torch.tensor([[1., -2.], [2., 1.], [-1., 3.], [.5, -.7],
                      [4., 2.], [-3., .2], [.3, .9], [-.8, -2.]], dtype=torch.float64)
    y = torch.tensor([0, 1, 2, 1, 0, 2, 1, 2])
    split = TensorSplit(x, y, tuple(f"synthetic:{i}" for i in range(len(x))))

    class SnapshotObjective:
        def __init__(self):
            self.steps, self.optimizer_ids, self.checkpoints = 0, set(), {}
            self.evidence = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __call__(self, student, batch, target):
            return nn.functional.cross_entropy(student(batch), target)

        def after_step(self, student, optimizer):
            self.steps += 1
            self.optimizer_ids.add(id(optimizer))
            if self.steps in (100, 400):
                self.checkpoints[self.steps] = deepcopy(optimizer.state_dict())

    objective = SnapshotObjective()
    result = runner.train_model(model, split, "classification", epochs=400, batch_size=8,
                                learning_rate=1e-4, seed=42, max_updates=400,
                                batch_objective=objective)
    expected_weight, expected_bias, moments, variances = _manual_full_batch_adam(
        initial["weight"], initial["bias"], x, y, 400)
    torch.testing.assert_close(model.weight, expected_weight, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(model.bias, expected_bias, rtol=1e-12, atol=1e-12)
    assert result["training_steps"] == objective.steps == 400
    assert len(objective.optimizer_ids) == 1
    assert set(objective.checkpoints) == {100, 400}
    for step, checkpoint in objective.checkpoints.items():
        assert {int(state["step"]) for state in checkpoint["state"].values()} == {step}
    for index, state in enumerate(objective.checkpoints[400]["state"].values()):
        torch.testing.assert_close(state["exp_avg"], moments[index], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(state["exp_avg_sq"], variances[index], rtol=1e-12, atol=1e-12)
    assert result["gradient_tensor_bytes"] == 9 * 8
    assert result["optimizer_state_bytes"] == 2 * 9 * 8 + 2 * 4
    split.verify_integrity()


class _FixedLogits(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(logits.clone())
        self.register_buffer("offset", torch.tensor(.125, dtype=logits.dtype))
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.logits + self.offset


@pytest.mark.parametrize("temperature", [1., 2., 4.])
def test_distillation_matches_direct_probability_sum_and_analytic_gradient(temperature):
    student_logits = torch.tensor([[1.2, -.4, .7], [-1.1, 2., .2], [.3, -.8, 1.5]], dtype=torch.float64)
    teacher_logits = torch.tensor([[-.2, 1.1, .6], [.7, -.3, 1.4], [-.5, 1.2, -.7]], dtype=torch.float64)
    source_teacher = _FixedLogits(teacher_logits).train()
    original_teacher = deepcopy(source_teacher.state_dict())
    teacher, student = deepcopy(source_teacher), _FixedLogits(student_logits)
    y, x = torch.tensor([2, 0, 1]), torch.zeros(3, 1, dtype=torch.float64)
    alpha = .5
    objective = packet.RecoveryObjective(teacher, alpha=alpha, temperature=temperature)
    with objective:
        loss = objective(student, x, y)
        teacher_probabilities = torch.softmax(teacher_logits / temperature, dim=1)
        student_probabilities = torch.softmax(student_logits / temperature, dim=1)
        ce = -(student_logits.log_softmax(1)[torch.arange(3), y]).mean()
        kl = (teacher_probabilities * (teacher_probabilities.log() - student_probabilities.log())).sum() / 3
        expected = (1 - alpha) * ce + alpha * temperature ** 2 * kl
        torch.testing.assert_close(loss, expected, rtol=1e-13, atol=1e-13)
        loss.backward()
        one_hot = torch.zeros_like(student_logits)
        one_hot[torch.arange(3), y] = 1.
        expected_gradient = ((1 - alpha) * (student_logits.softmax(1) - one_hot) +
                             alpha * temperature * (student_probabilities - teacher_probabilities)) / 3
        torch.testing.assert_close(student.logits.grad, expected_gradient, rtol=1e-12, atol=1e-12)
        assert not teacher.training
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in teacher.parameters())
    assert objective.evidence["teacher_unchanged"]
    assert objective.evidence["teacher_forwards"] == 1
    assert source_teacher.training and source_teacher.logits.requires_grad and source_teacher.calls == 0
    for name, value in original_teacher.items():
        torch.testing.assert_close(source_teacher.state_dict()[name], value, rtol=0, atol=0)


def test_ce_control_has_full_ce_coefficient_and_no_teacher_calls():
    values = torch.tensor([[1., -.5, .7], [-.4, 1.2, .3]], dtype=torch.float64)
    student = _FixedLogits(values)
    objective = packet.RecoveryObjective()
    with objective:
        loss = objective(student, torch.zeros(2, 1), torch.tensor([2, 1]))
        expected = -values.log_softmax(1)[torch.arange(2), torch.tensor([2, 1])].mean()
        torch.testing.assert_close(loss, expected, rtol=1e-13, atol=1e-13)
    assert objective.evidence["teacher_forwards"] == 0


def test_teacher_buffer_mutation_and_optimizer_contamination_are_observable():
    teacher = _FixedLogits(torch.ones(2, 3, dtype=torch.float64))
    student = _FixedLogits(torch.zeros(2, 3, dtype=torch.float64))
    objective = packet.RecoveryObjective(teacher)
    with objective:
        optimizer = torch.optim.Adam([student.logits, teacher.logits], lr=1e-4)
        with pytest.raises(ProtocolError, match="Teacher parameters"):
            objective.after_step(student, optimizer)
    objective = packet.RecoveryObjective(teacher)
    with pytest.raises(ProtocolError, match="Frozen teacher"):
        with objective:
            teacher.offset.add_(1.)


def test_schedule_conserves_132_sessions_and_balances_comparable_orders_without_rng_mutation():
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    schedule = packet.timing_schedule(42)
    assert random.getstate() == python_rng
    torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
    assert schedule == packet.timing_schedule(42)
    assert len(schedule) == len({item.session_id for item in schedule}) == 132
    assert Counter(item.kind for item in schedule) == {"artifact": 96, "conv2": 36}
    pairs, micros = defaultdict(list), defaultdict(list)
    for item in schedule:
        assert item.device in ("cpu", "cuda:0") and item.batch in (1, 64)
        assert item.block in (0, 1, 2)
        if item.kind == "artifact":
            pairs[item.subject, item.device, item.batch, item.block].append(item)
        else:
            micros[item.device, item.batch].append(item)
    first_sides = []
    cells = defaultdict(list)
    for (subject, device, batch, block), members in pairs.items():
        ordered = sorted(members, key=lambda item: item.order)
        assert [item.order for item in ordered] == [0, 1]
        assert {item.side for item in ordered} == {"baseline", "candidate"}
        first_sides.append(ordered[0].side)
        cells[subject, device, batch].append(ordered[0].side)
    assert Counter(first_sides) == {"baseline": 24, "candidate": 24}
    assert len(cells) == 16
    for orders in cells.values():
        assert len(orders) == 3 and sorted(Counter(orders).values()) == [1, 2]
    for members in micros.values():
        assert len(members) == 9
        assert {item.subject for item in members} == {"original_dense", "factor_r8", "materialized_r8"}
        for subject in {item.subject for item in members}:
            assert {item.order for item in members if item.subject == subject} == {0, 1, 2}


def test_policy_order_uses_validation_only_and_is_permutation_invariant():
    def row(case_id, accuracy=.9, ce=.2, n=90, alpha=1e-4):
        return {"status": "succeeded", "spec": {"case_id": case_id, "calibration_n": n, "alpha": alpha},
                "quality": {"validation": {"value": accuracy, "cross_entropy": ce},
                            "test": {"value": 1. if case_id == "worst" else 0.}}}
    rows = [row("worst", accuracy=.89), row("larger", n=180), row("chosen", n=32)]
    before = deepcopy(rows)
    for order in permutations(rows):
        assert packet.select_validation_policy(order)["spec"]["case_id"] == "chosen"
    assert rows == before
    assert packet.select_validation_policy([row("z"), row("a")])["spec"]["case_id"] == "a"
    with pytest.raises(ProtocolError, match="finite"):
        packet.select_validation_policy([row("nan", ce=float("nan"))])
    with pytest.raises(ProtocolError, match="No successful"):
        packet.select_validation_policy([{"status": "unsupported"}])


def test_conv_ridge_uses_uncentered_padded_window_energy_and_retains_image_count():
    # Nine 3x3 windows over a 3x3 image contain 49 actual ones after padding.
    conv = nn.Conv2d(1, 2, 3, padding=1).double()
    ridge, evidence = packet.ridge_from_inputs(conv, torch.ones(1, 1, 3, 3), .01)
    assert ridge == pytest.approx(.01 * 49 / 81, rel=1e-14)
    assert evidence["images"] == 1
    assert evidence["correlated_windows"] == 9
    assert evidence["window_independence_claim"] is False
    linear = nn.Linear(2, 3).double()
    ridge, evidence = packet.ridge_from_inputs(linear, torch.full((5, 2), 2.), .01)
    assert ridge == pytest.approx(.04)


def test_nested_calibration_keeps_prefix_lineage_and_does_not_mutate_pool():
    @dataclass(frozen=True)
    class Bundle:
        calibration: TensorSplit

    x, y = torch.arange(360.).reshape(180, 2), torch.arange(180)
    bundle = Bundle(TensorSplit(x, y, tuple(f"calibration:{i}" for i in range(180))))
    subsets = [packet.nested_calibration(bundle, count) for count in (32, 90, 180)]
    for count, subset in zip((32, 90, 180), subsets):
        assert subset.calibration.ids == bundle.calibration.ids[:count]
        torch.testing.assert_close(subset.calibration.x, x[:count], rtol=0, atol=0)
        torch.testing.assert_close(subset.calibration.y, y[:count], rtol=0, atol=0)
    subsets[0].calibration.x.add_(1000.)
    bundle.calibration.verify_integrity()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize("kind", ["transforms", "recovery", "retry_cap", "budget"])
def test_refused_attempt_never_spawns_or_overwrites_scientific_evidence(tmp_path, monkeypatch, kind):
    ledger = {"attempts": [], "charged_seconds": 12., "transform_attempts": 0}
    job = "transforms"
    scientific_file = tmp_path / "retained.bin"
    scientific_file.write_bytes(b"first attempt scientific evidence")
    if kind == "transforms":
        ledger["transform_attempts"] = 1
    elif kind == "recovery":
        job = "recovery-conv2_svd-ce"
        scientific_file = tmp_path / "recovery/conv2_svd-ce/update-100/state.pt"
        scientific_file.parent.mkdir(parents=True)
        scientific_file.write_bytes(b"first continuous Adam checkpoint")
    elif kind == "retry_cap":
        job = "timing-session"
        ledger["attempts"] = [{"job_id": job, "attempt": 1}, {"job_id": job, "attempt": 2}]
    else:
        job = "timing-session"
        ledger["charged_seconds"] = packet.COMPUTE_SECONDS
    _write_json(tmp_path / "ledger.json", ledger)
    original = scientific_file.read_bytes()
    monkeypatch.setattr(packet, "verify_protocol", lambda output: {})

    def forbidden_spawn(*args, **kwargs):
        pytest.fail("A refused attempt must not launch a process")

    monkeypatch.setattr(packet.subprocess, "Popen", forbidden_spawn)
    with pytest.raises(ProtocolError):
        packet.monitored_job(tmp_path, job, ["--worker", "unused"])
    assert json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8")) == ledger
    assert scientific_file.read_bytes() == original


def test_zero_exit_resource_failure_is_charged_and_child_transform_counter_survives(tmp_path, monkeypatch):
    ledger = {"attempts": [], "charged_seconds": 23., "transform_attempts": 0}
    _write_json(tmp_path / "ledger.json", ledger)
    monkeypatch.setattr(packet, "verify_protocol", lambda output: {})
    monkeypatch.setattr(packet, "background_snapshot", lambda: {"available_ram_bytes": 32 * 1024 ** 3})

    class OwnedProcess:
        pid = 919
        alive = True
        waits = 0

        def poll(self):
            return None if self.alive else 0

        def wait(self, timeout=None):
            self.waits += 1
            assert not self.alive
            return 0

    process = OwnedProcess()

    def launch(*args, **kwargs):
        # An actual transform worker advances its own ledger while monitored.
        active = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
        active["transform_attempts"] = 7
        _write_json(tmp_path / "ledger.json", active)
        return process

    class MissingProcess(Exception):
        pass

    fake_psutil = SimpleNamespace(Process=lambda pid: process,
                                 virtual_memory=lambda: SimpleNamespace(available=32 * 1024 ** 3),
                                 NoSuchProcess=MissingProcess)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(packet.subprocess, "Popen", launch)
    monkeypatch.setattr(packet.historical, "_owned_process_sample", lambda root: [{"pid": 919, "rss_bytes": 13 * 1024 ** 3}])
    terminated = []

    def stop(root):
        terminated.append(root.pid)
        process.alive = False

    monkeypatch.setattr(packet.historical, "_terminate_owned_tree", stop)
    with pytest.raises(RuntimeError, match="Retained computational attempt failed"):
        packet.monitored_job(tmp_path, "transforms", ["--worker", "transforms"])
    after = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert after["transform_attempts"] == 7
    assert len(after["attempts"]) == 1 and after["attempts"][0]["returncode"] == 0
    record = after["attempts"][0]
    assert record["violation"] == "recursive RSS exceeds 12GiB"
    assert after["charged_seconds"] == pytest.approx(23. + record["elapsed_seconds"])
    assert terminated == [919] and process.waits == 1
    assert (tmp_path / "processes/transforms/attempt-1/resources.json").exists()


def test_paired_objectives_visit_identical_400_full_batches_and_leave_source_and_rng_unchanged():
    x = torch.stack((torch.arange(131.) / 131, torch.sin(torch.arange(131.))), dim=1)
    y = torch.arange(131) % 3
    split = TensorSplit(x, y, tuple(f"train:{i}" for i in range(131)))
    source = nn.Linear(2, 3)
    source_hash = runner.model_state_hash(source)
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    indices = packet.planned_batches(split, updates=400, batch_size=64)
    assert len(indices) == 400
    assert all(len(batch) == len(set(batch)) == 64 for batch in indices)
    expected = [stable_hash({"x": tensor_hash(split.x[batch]), "y": tensor_hash(split.y[batch])})
                for batch in indices]
    observed = []
    for use_teacher in (False, True):
        student = deepcopy(source)
        objective = packet.RecoveryObjective(deepcopy(source) if use_teacher else None,
                                             expected_batch_hashes=expected)
        with runner._seeded(42, "cpu"):
            result = runner.train_model(student, split, "classification", epochs=200, batch_size=64,
                                        learning_rate=1e-4, max_updates=400, drop_last=True,
                                        seed=42, batch_objective=objective)
        assert result["training_steps"] == 400
        assert objective.evidence["visited_samples"] == 400 * 64
        assert objective.evidence["teacher_forwards"] == (400 if use_teacher else 0)
        assert objective.evidence["batch_hashes"] == expected
        observed.append(objective.evidence["batch_hashes"])
    assert observed[0] == observed[1]
    assert runner.model_state_hash(source) == source_hash
    assert random.getstate() == python_rng
    torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
    split.verify_integrity()


def test_original_current_conv_statistics_keep_the_same_first_conv_and_linear_head():
    generator = torch.Generator().manual_seed(17)
    roles = {}
    for role, count in (("train", 2), ("validation", 2), ("calibration", 180), ("test", 2)):
        roles[role] = TensorSplit(torch.rand(count, 1, 8, 8, generator=generator),
                                  torch.arange(count) % 10, tuple(f"{role}:{i}" for i in range(count)))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        baseline = packet.scenarios.DigitsCNN().eval()
    bundle = ExperimentBundle(roles["train"], roles["validation"], roles["calibration"], roles["test"],
                              "classification", baseline)
    original = deepcopy(baseline.state_dict())
    results = []
    for graph in ("original", "current"):
        spec = packet.TransformSpec(f"toy-{graph}", "weighted_svd", "features.2", 8,
                                    calibration_n=32, first_rank=3, linear_rank=16, statistics_graph=graph)
        model, evidence = packet._transform(baseline, bundle, spec)
        results.append((model, evidence))
    for path in ("features.0", "classifier.0", "classifier.2"):
        left = results[0][0].get_submodule(path).state_dict()
        right = results[1][0].get_submodule(path).state_dict()
        assert left.keys() == right.keys()
        for name in left:
            torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
    assert results[0][1]["statistics_graph"] != results[1][1]["statistics_graph"]
    assert results[0][1]["calibration"] == results[1][1]["calibration"]
    for name, value in original.items():
        torch.testing.assert_close(baseline.state_dict()[name], value, rtol=0, atol=0)
