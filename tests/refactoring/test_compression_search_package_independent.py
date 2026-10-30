"""Independent finite-language, numerical and accounting witnesses for issue 99."""
from dataclasses import replace
import hashlib
from itertools import product
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.compression_composition import (
    AddResidual, CheckedComposition, CompositionRecipe, CompositionRequest,
    CompositionSuccess, LowRank, MaterializeDense, PruneMask, QuantizeRemapped,
    execute_composition, prepare_composition,
)
from fedcore.algorithm.low_rank.plans import ValidationFailure
from fedcore.experiments.compression_composition_reference import (
    assert_reference_parity, dense_weight_bias, execute_reference,
    independent_inventory, inspect_contract,
)


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _model():
    # Do not consume any process-global RNG, even during Linear construction.
    generator = torch.Generator(device="cpu").manual_seed(9900)
    with torch.random.fork_rng(devices=[]):
        model = nn.Sequential(nn.Linear(8, 8, dtype=torch.float64)).eval()
        with torch.no_grad():
            model[0].weight.copy_(torch.randn((8, 8), generator=generator, dtype=torch.float64))
            model[0].bias.copy_(torch.randn(8, generator=generator, dtype=torch.float64))
    return model


def _request(word):
    alphabet = {"P": PruneMask((0,)), "L": LowRank(2),
        "Q": QuantizeRemapped(tail_dtype="float64", scale_dtype="float64", compute_dtype="float64"),
        "D": MaterializeDense(), "R": AddResidual(1)}
    return CompositionRequest((CompositionRecipe("0", tuple(alphabet[letter] for letter in word)),))


WORDS = tuple("".join(chars) for length in (1, 2, 3) for chars in product("PLQDR", repeat=length))


def _negatives():
    return (
        replace(_request("L"), recipes=(CompositionRecipe("0", (LowRank(0),)),)),
        replace(_request("L"), recipes=(CompositionRecipe("0", (LowRank(9),)),)),
        replace(_request("P"), recipes=(CompositionRecipe("0", (PruneMask((0, 0)),)),)),
        replace(_request("P"), recipes=(CompositionRecipe("0", (PruneMask((64,)),)),)),
        replace(_request("Q"), recipes=(CompositionRecipe("0", (QuantizeRemapped(compute_dtype="float32"),)),)),
        replace(_request("D"), recipes=(CompositionRecipe("missing", (MaterializeDense(),)),)),
        replace(_request("D"), recipes=(CompositionRecipe("0", (MaterializeDense(),)), CompositionRecipe("0", (MaterializeDense(),)))),
        replace(_request("D"), export_format="onnx"),
    )


def test_literal_finite_language_has_exactly_72_acceptances_and_83_refusals():
    model = _model()
    accepted = [word for word in WORDS if inspect_contract(_request(word), model).accepted]
    assert len(WORDS) == len(set(WORDS)) == 155
    assert [sum(len(word) == length for word in accepted) for length in (1, 2, 3)] == [5, 16, 51]
    assert len(accepted) == 72 and len(WORDS)-len(accepted) == 83
    # Golden examples distinguish explicit materialization, repeated packing,
    # nested residuals and a mask that follows real factors.
    assert all(word in accepted for word in ("PLQ", "LDR", "QDP", "RDL", "DDD"))
    assert all(word not in accepted for word in ("LP", "LL", "QQ", "RQ", "RR", "QRL"))


@pytest.mark.parametrize("word", WORDS)
def test_reached_primitives_match_the_literal_contract_for_every_word(word):
    model, request = _model(), _request(word)
    contract = inspect_contract(request, model)
    result = execute_reference(model, request)
    assert result.accepted == contract.accepted, (word, contract, result)
    assert result.timings.total_seconds >= 0
    assert sum((result.timings.admission_seconds, result.timings.copy_seconds,
                result.timings.numerical_seconds, result.timings.verification_seconds,
                result.timings.rejection_seconds)) <= result.timings.total_seconds + 1e-9
    assert result.decompositions_started == sum(event.decompositions_started for event in result.events)
    assert result.completed_steps == sum(event.completed for event in result.events)
    assert result.decompositions_started == sum(letter in "LQR" for letter in word[:result.completed_steps])
    if result.accepted:
        assert result.completed_steps == len(word)
        assert result.storage_bytes == contract.expected_storage_bytes
        assert result.storage_bytes == independent_inventory(result.model)[0]
        assert result.final_states == contract.final_states
    else:
        assert result.model is None and result.rejection.category == "structural"
        assert result.rejection.code == contract.rejection.code
        assert len(result.events) == result.completed_steps + 1
        assert not result.events[-1].completed


@pytest.mark.parametrize("case_request,code", tuple(zip(_negatives(), (
    "InvalidRank", "InvalidRank", "InvalidMask", "InvalidMask", "UnsupportedCapability",
    "InvalidScope", "OverlappingScopes", "UnsupportedExport"))))
def test_all_eight_negative_cases_refuse_explicitly(case_request, code):
    model = _model()
    contract, result = inspect_contract(case_request, model), execute_reference(model, case_request)
    assert not contract.accepted and contract.rejection.code == code
    assert not result.accepted and result.rejection.code == code
    assert result.rejection.category == "structural"
    if code == "UnsupportedExport":
        # Late mode does real safe work before final-format publication.
        assert result.completed_steps == 1


def test_bad_late_rank_is_rejected_after_valid_prefix_without_silent_svd_slicing():
    model = _model()
    request = CompositionRequest((CompositionRecipe("0", (LowRank(2), MaterializeDense(), LowRank(9))),))
    result = execute_reference(model, request)
    assert result.rejection.code == "InvalidRank"
    assert result.rejection.step_index == 2
    assert result.completed_steps == 2 and result.decompositions_started == 1
    assert [event.completed for event in result.events] == [True, True, False]


def test_contract_prediction_does_not_copy_factorize_or_execute_a_model(monkeypatch):
    import fedcore.experiments.compression_composition_reference as reference
    model = _model()

    def forbidden(*args, **kwargs):
        pytest.fail("Metadata contract prediction reached a numerical or copying effect")

    monkeypatch.setattr(torch.linalg, "svd", forbidden)
    monkeypatch.setattr(nn.Linear, "forward", forbidden)
    monkeypatch.setattr(reference, "deepcopy", forbidden)
    decisions = [inspect_contract(_request(word), model).accepted for word in WORDS]
    assert sum(decisions) == 72


def test_reference_never_calls_production_planning_execution_or_inventory(monkeypatch):
    import fedcore.algorithm.low_rank.compression_composition as production
    import fedcore.algorithm.low_rank.candidate_storage as storage
    import fedcore.algorithm.low_rank.grammar_execution as grammar
    import fedcore.algorithm.low_rank.compression_grammar as compiler

    def forbidden(*args, **kwargs):
        pytest.fail("Reference reached a production decision, executor or tensor inventory")

    for name in ("prepare_composition", "execute_composition", "_transition", "_apply", "_factors", "_inspect_operator"):
        monkeypatch.setattr(production, name, forbidden)
    monkeypatch.setattr(storage, "capture_inventory", forbidden)
    monkeypatch.setattr(compiler, "compile_genotype", forbidden)
    for name in ("compile_workflow", "compile_candidate", "compile_genotype"):
        if hasattr(grammar, name):
            monkeypatch.setattr(grammar, name, forbidden)
    model = _model()
    for word in ("PLQ", "LQR", "QRD", "RDP", "LL"):
        request = _request(word)
        assert execute_reference(model, request).accepted == inspect_contract(request, model).accepted


@pytest.mark.parametrize("word", ("P", "L", "Q", "R", "PLQ", "LQR", "QRD", "RDL", "DPP", "LDP"))
def test_production_and_its_checkpoint_match_independent_representation_and_outputs(word):
    model, request = _model(), _request(word)
    expected = execute_reference(model, request)
    checked = prepare_composition(model, request)
    assert type(checked) is CheckedComposition, checked
    actual = execute_composition(model, checked)
    assert type(actual) is CompositionSuccess, actual
    assert_reference_parity(expected, actual.model, request)
    from fedcore.tools.registry.checkpoint_manager import CheckpointManager
    restored = CheckpointManager("", auto_cleanup=False).deserialize_from_bytes(actual.checkpoint_bytes, device="cpu")
    assert_reference_parity(expected, restored, request)


def test_mask_rank_residual_numerics_have_an_independent_numpy_oracle():
    model = _model()
    original = model[0].weight.detach().numpy().copy()
    masked = original.copy()
    masked.flat[0] = 0
    left, singular, right = np.linalg.svd(masked, full_matrices=False)
    truncated = (left[:, :2]*singular[:2]) @ right[:2]
    delta_left, delta_singular, delta_right = np.linalg.svd(original-truncated, full_matrices=False)
    corrected = truncated + (delta_left[:, :1]*delta_singular[:1]) @ delta_right[:1]
    result = execute_reference(model, _request("PLR"))
    assert result.accepted
    weight, bias = dense_weight_bias(result.model[0])
    np.testing.assert_allclose(weight.numpy(), corrected, rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(bias.numpy(), model[0].bias.detach().numpy())
    assert result.storage_bytes == (2*(8+8) + 1*(8+8) + 8)*8


@pytest.mark.parametrize("word,bytes_", (("D", 576), ("P", 576), ("L", 320),
    ("Q", 320), ("LQ", 128), ("LR", 448), ("LQR", 256), ("LRD", 576)))
def test_named_components_have_independent_exact_byte_costs(word, bytes_):
    result = execute_reference(_model(), _request(word))
    assert result.accepted and result.storage_bytes == bytes_
    assert inspect_contract(_request(word), _model()).expected_storage_bytes == bytes_


def test_reference_preserves_source_modes_parameter_identity_values_and_global_rng():
    model = _model().train()
    model[0].eval()
    parameters = tuple((id(value), value.untyped_storage()._cdata, value.detach().clone(), value.requires_grad)
                       for value in model.parameters())
    modes = tuple(layer.training for layer in model.modules())
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    for word in ("PLQ", "LQR", "RDP", "LL"):
        execute_reference(model, _request(word))
    assert tuple(layer.training for layer in model.modules()) == modes
    for value, (identity, storage, expected, requires_grad) in zip(model.parameters(), parameters):
        assert id(value) == identity and value.untyped_storage()._cdata == storage
        assert value.requires_grad == requires_grad and torch.equal(value, expected)
    assert random.getstate() == python_rng
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_rng[0] and np.array_equal(after_numpy[1], numpy_rng[1])
    assert after_numpy[2:] == numpy_rng[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng)


def test_numerical_failures_are_not_misclassified_as_structural(monkeypatch):
    model = _model()

    def fails(*args, **kwargs):
        raise torch.linalg.LinAlgError("injected nonconvergence")

    monkeypatch.setattr(torch.linalg, "svd", fails)
    request = _request("L")
    assert inspect_contract(request, model).accepted
    result = execute_reference(model, request)
    assert result.rejection.category == "numerical"
    assert result.rejection.code == "NumericalFailure"
    assert result.decompositions_started == 1 and result.completed_steps == 0
    assert len(result.events) == 1 and not result.events[0].completed


def test_memory_and_unknown_runtime_errors_remain_distinct(monkeypatch):
    import fedcore.experiments.compression_composition_reference as reference
    model = _model()
    for error, category in ((MemoryError("injected"), "resource"), (RuntimeError("injected"), "shell"),
            (RuntimeError("[enforce fail at alloc_cpu.cpp] DefaultCPUAllocator: can't allocate memory"), "resource")):
        def fails(*args, _error=error, **kwargs):
            raise _error
        monkeypatch.setattr(reference, "_primitive", fails)
        result = execute_reference(model, _request("L"))
        assert result.rejection.category == category and result.decompositions_started == 0


def test_final_byte_limit_is_enforced_after_execution_but_peak_limit_before_copy():
    model = _model()
    storage_refusal = execute_reference(model, replace(_request("L"), max_storage_bytes=319))
    assert storage_refusal.rejection.category == "resource"
    assert storage_refusal.rejection.code == "BudgetExceeded"
    assert storage_refusal.completed_steps == 1
    peak_refusal = execute_reference(model, replace(_request("L"), max_peak_bytes=1))
    assert peak_refusal.rejection.category == "resource"
    assert peak_refusal.completed_steps == peak_refusal.decompositions_started == 0
    assert peak_refusal.timings.copy_seconds == 0


@pytest.mark.parametrize("values,targets", (
    ([[3., 1., -1.], [-1., 3., 1.], [3., 2., 1.], [1., 2., 3.], [1., 1., 1.], [1., -2., 0.]],
     [0, 2, 1, 2, 0, 2]),
    ([[3., 1.], [4., -2.], [1., 0.]], [0, 0, 0]),
))
def test_validation_metrics_match_independent_sklearn_and_numpy_counts(values, targets):
    from sklearn.metrics import accuracy_score, f1_score, log_loss
    from fedcore.experiments.compression_search_evaluation import metrics

    logits = torch.tensor(values, dtype=torch.float64)
    labels = torch.tensor(targets)
    actual = metrics(logits, labels)
    matrix, truth = np.asarray(values), np.asarray(targets)
    probabilities = np.exp(matrix-matrix.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    prediction, classes = matrix.argmax(axis=1), list(range(matrix.shape[1]))
    assert actual["cross_entropy"] == pytest.approx(log_loss(truth, probabilities, labels=classes), rel=1e-13, abs=1e-13)
    assert actual["accuracy"] == accuracy_score(truth, prediction)
    assert actual["accuracy"] == sum(int(a == b) for a, b in zip(prediction, truth))/len(truth)
    assert actual["macro_f1"] == pytest.approx(f1_score(truth, prediction, labels=classes,
        average="macro", zero_division=0), rel=0, abs=1e-15)
    assert actual["samples"] == len(truth)


def _historical_fixture(tmp_path, *, target_dtype=np.int64):
    """Synthetic provenance fixture; no historical weights or dataset needed."""
    from fedcore.experiments.protocol import TensorSplit
    from fedcore.experiments.scenarios import TrainPreprocessor

    raw = np.arange(569*3, dtype=np.float64).reshape(569, 3)/1707
    labels = (np.arange(569) % 2).astype(target_dtype)
    partitions = {"train": np.arange(0, 341), "validation": np.arange(341, 426),
                  "calibration": np.arange(426, 483), "test": np.arange(483, 569)}
    ids = {role: tuple(f"tabular:{index}" for index in indices) for role, indices in partitions.items()}
    preprocessor = TrainPreprocessor.fit(raw[partitions["train"]])
    normalized = preprocessor.transform(raw)
    # The historical builder hashes raw labels in their source dtype, then
    # explicitly normalizes each role to int64 before constructing TensorSplit.
    roles = {role: TensorSplit(torch.from_numpy(normalized[indices]), torch.from_numpy(labels[indices].astype(np.int64)), ids[role]).manifest()
             for role, indices in partitions.items()}
    digest = lambda array: hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    archive = {"data": {"roles": roles, "metadata": {"data_sha256": digest(raw),
        "target_sha256": digest(labels), "preprocessing_fit_ids": list(ids["train"]),
        "preprocessing": preprocessor.metadata()}}}
    path = tmp_path / "docs/svd_compression/results/local-2026-10-13/runs/mlp-seed-41.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(archive), encoding="utf-8")
    return SimpleNamespace(data=raw, target=labels), archive, path


@pytest.mark.parametrize("target_dtype", (np.int32, np.int64))
def test_historical_loader_preserves_raw_label_hash_and_restores_archived_int64_roles(tmp_path, monkeypatch, target_dtype):
    import sklearn.datasets
    import fedcore.experiments.compression_search_evaluation as evaluation

    raw, archive, _ = _historical_fixture(tmp_path, target_dtype=target_dtype)
    before = raw.target.copy()
    raw_hash = hashlib.sha256(np.ascontiguousarray(raw.target).tobytes()).hexdigest()
    assert archive["data"]["metadata"]["target_sha256"] == raw_hash
    if target_dtype == np.int32:
        canonical_hash = hashlib.sha256(raw.target.astype(np.int64).tobytes()).hexdigest()
        assert canonical_hash != raw_hash
    monkeypatch.setattr(sklearn.datasets, "load_breast_cancer", lambda: raw)
    splits, restored_archive = evaluation.historical_splits(41, tmp_path, include_test=True)
    assert raw.target.dtype == target_dtype and np.array_equal(raw.target, before)
    assert restored_archive["data"]["metadata"]["target_sha256"] == raw_hash
    for role, split in splits.items():
        indices = [int(identifier.split(":")[1]) for identifier in split.ids]
        assert split.y.dtype == torch.int64
        np.testing.assert_array_equal(split.y.numpy(), before[indices].astype(np.int64))
        assert split.manifest() == archive["data"]["roles"][role]


def test_historical_roles_do_not_construct_test_tensors_until_explicit_finalization(tmp_path, monkeypatch):
    import sklearn.datasets
    import fedcore.experiments.compression_search_evaluation as evaluation

    raw, _, _ = _historical_fixture(tmp_path)
    monkeypatch.setattr(sklearn.datasets, "load_breast_cancer", lambda: raw)
    original_split, constructed = evaluation.TensorSplit, []

    def record_split(features, labels, ids):
        constructed.append(tuple(ids))
        return original_split(features, labels, ids)

    monkeypatch.setattr(evaluation, "TensorSplit", record_split)
    splits, _ = evaluation.historical_splits(41, tmp_path)
    assert set(splits) == {"train", "validation", "calibration"}
    assert all("tabular:483" not in identifiers for identifiers in constructed)
    constructed.clear()
    reopened, _ = evaluation.historical_splits(41, tmp_path, include_test=True)
    assert set(reopened) == {"train", "validation", "calibration", "test"}
    assert sum("tabular:483" in identifiers for identifiers in constructed) == 1


@pytest.mark.parametrize("tamper", ("raw_hash", "duplicate_id", "preprocessing_fit_ids", "preprocessing_values"))
def test_historical_provenance_mismatches_stop_instead_of_resplitting(tmp_path, monkeypatch, tamper):
    import sklearn.datasets
    import fedcore.experiments.compression_search_evaluation as evaluation

    raw, archive, path = _historical_fixture(tmp_path)
    monkeypatch.setattr(sklearn.datasets, "load_breast_cancer", lambda: raw)
    data = archive["data"]
    if tamper == "raw_hash":
        data["metadata"]["data_sha256"] = "0"*64
    elif tamper == "duplicate_id":
        data["roles"]["test"]["ids"][0] = data["roles"]["train"]["ids"][0]
    elif tamper == "preprocessing_fit_ids":
        data["metadata"]["preprocessing_fit_ids"].reverse()
    else:
        data["metadata"]["preprocessing"]["mean"][0] += .25
    path.write_text(json.dumps(archive), encoding="utf-8")
    with pytest.raises(evaluation.PackageFatal):
        evaluation.historical_splits(41, tmp_path)


@pytest.mark.parametrize("failure", ("hash", "size"))
def test_checkpoint_provenance_is_verified_before_weights_or_role_loading(tmp_path, monkeypatch, failure):
    import fedcore.experiments.compression_search_evaluation as evaluation

    path = tmp_path / "mlp-seed-41/baseline/state_dict.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"synthetic invalid source")
    if failure == "size":
        monkeypatch.setattr(evaluation, "file_sha256", lambda _: evaluation.SOURCE_HASHES[41])

    def forbidden(*args, **kwargs):
        pytest.fail("Bad checkpoint provenance reached data or model loading")

    monkeypatch.setattr(evaluation, "historical_splits", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(evaluation.PackageFatal, match="hash/size"):
        evaluation.load_source(41, tmp_path, tmp_path)


@pytest.mark.parametrize("mutation", ("model", "calibration", "validation"))
def test_fatal_source_invariant_keeps_a_durable_terminal_row_and_charged_attempt(tmp_path, monkeypatch, mutation):
    import fedcore.experiments.compression_search_evaluation as evaluation
    from fedcore.algorithm.low_rank.grammar_execution import NumericalFailure
    from fedcore.experiments.compression_search_protocol import INITIAL_GENOTYPES
    from fedcore.experiments.protocol import TensorSplit

    model = _model()
    calibration = TensorSplit(torch.zeros((2, 8)), torch.zeros(2, dtype=torch.long), ("cal:0", "cal:1"))
    validation = TensorSplit(torch.ones((2, 8)), torch.zeros(2, dtype=torch.long), ("val:0", "val:1"))
    source = evaluation.EvaluationSource(41, model, calibration, validation, {}, {}, torch.zeros((2, 8)))
    ledger = evaluation.AttemptLedger(tmp_path / "attempts.jsonl", "invariant-test", "protocol")
    evaluator = evaluation.FullEvaluator(source, None, ledger, tmp_path / "events.jsonl",
        context_sha256="context", job_id="invariant-test")

    def mutate_then_refuse(*args, **kwargs):
        with torch.no_grad():
            if mutation == "model":
                model[0].weight[0, 0] += 1
            else:
                getattr(source, mutation).x[0, 0] += 1
        return NumericalFailure("injected", "source mutation must override this ordinary failure")

    monkeypatch.setattr(evaluation, "execute_workflow", mutate_then_refuse)
    workflow = SimpleNamespace(genotype=INITIAL_GENOTYPES[0], fingerprint="workflow")
    with pytest.raises(evaluation.PackageFatal):
        evaluator(workflow)
    persisted = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    attempts = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ledger.used == len(attempts) == len(persisted) == len(evaluator.rows) == 1
    assert persisted[0]["status"] == "package_fatal"
    assert persisted[0]["failure"]
    assert persisted[0]["global_attempt"] == attempts[0]["attempt"] == 1
    assert persisted[0]["completed_at"] >= persisted[0]["started_at"]


@pytest.mark.parametrize("now,hard_seconds,expected_stop", (
    (7200., 10000., "package_time_limit"),
    (46., 45., "worker_hard_deadline"),
))
def test_orchestrator_watchdog_kills_the_whole_owned_tree_at_absolute_deadlines(
        tmp_path, monkeypatch, now, hard_seconds, expected_stop):
    """Exercise the parent loop with a fake clock/process, never launch ML."""
    import psutil
    import fedcore.experiments.compression_search_package as package

    killed, waits = [], []

    class Process:
        pid = 10
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            waits.append(timeout)
            return self.returncode

    process = Process()

    class Observed:
        def __init__(self, pid):
            self.pid = pid

        def children(self, recursive):
            assert recursive
            return [Observed(11)]

        def memory_info(self):
            return SimpleNamespace(rss=1024)

        def is_running(self):
            return True

        def kill(self):
            killed.append(self.pid)
            if self.pid == 10:
                process.returncode = -9

    calls = []

    def fake_popen(*args, **kwargs):
        calls.append((args, kwargs))
        return process

    clock_values = iter((0., 1., now))
    monkeypatch.setattr(package.time, "perf_counter", lambda: next(clock_values, now))
    monkeypatch.setattr(package, "source_hashes", lambda: {})
    monkeypatch.setattr(package.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(psutil, "Process", Observed)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=16*1024**3))
    protocol = {"source_files": {}, "historical_root": "unused", "protocol_sha256": "protocol", "source_sha256": "sources"}
    result = package._run_job(tmp_path, {"job": "fake", "kind": "search", "hard_seconds": hard_seconds}, protocol, 0.)
    assert result["status"] == "failed" and result["stop"] == expected_stop
    assert killed == [11, 10] and waits == [10]
    assert result["resources"]["peak_tree_rss_bytes"] == 2048
    job = json.loads((tmp_path / "jobs/fake/job.json").read_text(encoding="utf-8"))
    assert job["package_deadline"] == 7200 and job["compute_admission_deadline"] == 6900
    assert len(calls) == 1
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.parametrize("limit", ("compute_deadline", "global_attempts"))
def test_orchestrator_does_not_launch_work_after_115_minutes_or_3300_attempts(tmp_path, monkeypatch, limit):
    import fedcore.experiments.compression_search_package as package

    monkeypatch.setattr(package, "source_hashes", lambda: {})
    monkeypatch.setattr(package.time, "perf_counter", lambda: 6900. if limit == "compute_deadline" else 0.)
    if limit == "global_attempts":
        (tmp_path / "mlp-attempts.jsonl").write_text("{}\n"*3300, encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("Exhausted package allowance launched a new worker")

    monkeypatch.setattr(package.subprocess, "Popen", forbidden)
    result = package._run_job(tmp_path, {"job": "forbidden"}, {"source_files": {}}, 0.)
    assert result["status"] == "package_budget"
    assert not (tmp_path / "jobs").exists()


def _cache_replay_fixture(tmp_path, monkeypatch, *, branch="A", failing=False, forged=False, switch_context=False):
    """Reach the real cache loop with tiny trusted evaluator substitutes."""
    import fedcore.experiments.compression_search_evaluation as evaluation
    import fedcore.experiments.compression_search_package as package
    import fedcore.experiments.compression_search_protocol as protocol

    stream = protocol.TUNED_CATALOG.genotypes[:12]*4
    failure_id = protocol.canonical_solution_id(stream[0])

    class Evaluator:
        def __init__(self):
            self.context_sha256 = "context-A"
            self.rows = []
            self.artifact_dir = None

        def __call__(self, workflow):
            identity = workflow.fingerprint
            if failing and identity == failure_id:
                row = {"status": "NumericalFailure"}
            else:
                self.artifact_dir.mkdir(parents=True, exist_ok=True)
                path = self.artifact_dir / f"{identity}.pt"
                payload = ("verified-tiny-artifact:" + identity).encode()
                path.write_bytes(payload)
                row = {"status": "finite", "quality": int(identity[:6], 16)/16**6,
                    "tensor_bytes": 12_488, "artifact_path": str(path),
                    "checkpoint_sha256": "0"*64 if forged else hashlib.sha256(payload).hexdigest(),
                    "workflow_fingerprint": identity, "evaluation_context_sha256": self.context_sha256}
            self.rows.append(row)

    evaluator = Evaluator()
    compile_calls = [0]

    def compile_candidate(genotype, space):
        compile_calls[0] += 1
        if switch_context and compile_calls[0] == 13:
            evaluator.context_sha256 = "context-B"
        return SimpleNamespace(fingerprint=protocol.canonical_solution_id(genotype))

    monkeypatch.setattr(protocol, "cache_replay_stream", lambda seed: stream)
    monkeypatch.setattr(package, "_source_evaluator", lambda job, catalog: (SimpleNamespace(seed=41), None, evaluator))
    monkeypatch.setattr(evaluation, "compile_checked", compile_candidate)
    tmp_path.mkdir(parents=True, exist_ok=True)
    job = {"kind": "cache", "job": "cache-stub", "directory": str(tmp_path), "model_seed": 41,
           "stream_seed": 9921, "branch": branch}
    return package, evaluator, job


def test_real_cache_replay_loop_has_12_evaluations_with_cache_and_48_without(tmp_path, monkeypatch):
    package, cached_evaluator, cached_job = _cache_replay_fixture(tmp_path / "on", monkeypatch)
    cached = package.execute_job(cached_job)
    package, plain_evaluator, plain_job = _cache_replay_fixture(tmp_path / "off", monkeypatch, branch="B")
    plain = package.execute_job(plain_job)
    assert len(cached_evaluator.rows) == 12 and len(plain_evaluator.rows) == 48
    assert len(cached["requests"]) == len(plain["requests"]) == 48
    assert sum(row["cache_hit"] for row in cached["requests"]) == 36
    assert sum(row["cache_hit"] for row in plain["requests"]) == 0
    signature = lambda result: [(row["solution_id"], row["status"], row["quality"], row["tensor_bytes"])
                                for row in result["requests"]]
    assert signature(cached) == signature(plain)


def test_real_cache_loop_does_not_cache_numerical_failures(tmp_path, monkeypatch):
    package, evaluator, job = _cache_replay_fixture(tmp_path, monkeypatch, failing=True)
    result = package.execute_job(job)
    assert len(evaluator.rows) == 15
    assert sum(row["status"] == "NumericalFailure" for row in evaluator.rows) == 4
    assert sum(row["cache_hit"] for row in result["requests"]) == 33
    assert all(not row["cache_hit"] for row in result["requests"] if row["status"] == "NumericalFailure")


def test_real_cache_loop_refuses_a_forged_artifact_reference(tmp_path, monkeypatch):
    from fedcore.experiments.compression_search_evaluation import PackageFatal
    package, _, job = _cache_replay_fixture(tmp_path, monkeypatch, forged=True)
    with pytest.raises(PackageFatal, match="forged"):
        package.execute_job(job)


def test_real_cache_loop_misses_when_evaluation_context_changes(tmp_path, monkeypatch):
    package, evaluator, job = _cache_replay_fixture(tmp_path, monkeypatch, switch_context=True)
    result = package.execute_job(job)
    assert len(evaluator.rows) == 24
    assert sum(row["cache_hit"] for row in result["requests"]) == 24
    assert all(not row["cache_hit"] for row in result["requests"][:24])
    assert all(row["cache_hit"] for row in result["requests"][24:])


@pytest.mark.parametrize("change", ("source", "calibration_values", "validation_labels", "data_role", "code", "environment"))
def test_actual_evaluation_context_covers_source_data_roles_code_and_environment(tmp_path, monkeypatch, change):
    import fedcore.experiments.compression_search_evaluation as evaluation
    import fedcore.experiments.compression_search_package as package
    from fedcore.experiments.compression_search_protocol import TUNED_CATALOG

    class Split:
        def __init__(self, prefix):
            self.value = {"ids": [prefix+":0"], "x_sha256": prefix+"-features", "y_sha256": prefix+"-labels"}

        def manifest(self):
            return dict(self.value)

    source = SimpleNamespace(manifest={"normalized_identity": "source-A"},
        calibration=Split("cal"), validation=Split("val"))
    monkeypatch.setattr(evaluation, "AttemptLedger", lambda *args, **kwargs: SimpleNamespace(charge=lambda *args: None))
    monkeypatch.setattr(evaluation, "load_source", lambda *args: source)
    monkeypatch.setattr(evaluation, "prepare_space", lambda *args: None)
    monkeypatch.setattr(evaluation, "FullEvaluator", lambda *args, **kwargs: SimpleNamespace(context_sha256=kwargs["context_sha256"]))
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps({"environment": {"torch": "pinned-A"}}), encoding="utf-8")
    job = {"output": str(tmp_path), "directory": str(tmp_path / "jobs/context"), "job": "context",
        "protocol_sha256": "protocol", "compute_admission_deadline": float("inf"), "model_seed": 41,
        "historical_root": "unused", "repo": "unused", "source_sha256": "code-A"}
    original = package._source_evaluator(job, TUNED_CATALOG)[2].context_sha256
    assert package._source_evaluator(job, TUNED_CATALOG)[2].context_sha256 == original
    if change == "source":
        source.manifest["normalized_identity"] = "source-B"
    elif change == "calibration_values":
        source.calibration.value["x_sha256"] = "changed-features"
    elif change == "validation_labels":
        source.validation.value["y_sha256"] = "changed-labels"
    elif change == "data_role":
        source.calibration, source.validation = source.validation, source.calibration
    elif change == "code":
        job["source_sha256"] = "code-B"
    else:
        protocol_path.write_text(json.dumps({"environment": {"torch": "pinned-B"}}), encoding="utf-8")
    assert package._source_evaluator(job, TUNED_CATALOG)[2].context_sha256 != original
