"""Independent public-boundary regressions for heterogeneous integration.

The optimizer is real; candidate compilation/evaluation are recording fakes.
No model, training loop, GPU, host process, or external service is required.
"""

from types import SimpleNamespace
from collections import Counter
from dataclasses import replace
import math
import random
import json

import pytest
from hypothesis import given, settings, strategies as st

from fedcore.algorithm.low_rank import grammar_search as search_module
from fedcore.algorithm.low_rank.grammar_execution import ResourceFailure
from fedcore.algorithm.low_rank.grammar_search import (
    GrammarAdapter,
    GrammarFitness,
    GrammarMutationDecision,
    GrammarSearchConfig,
    run_grammar_search,
)
from fedcore.algorithm.low_rank.plans import ValidationFailure, Violation
from fedcore.algorithm.heterogeneous.policy import Action, SearchArm, policy_for
from fedcore.algorithm.heterogeneous.protocol import ExperimentProtocol, pilot_budget


def _adapter(compile_candidate, *, guard=None):
    """A finite foreign grammar checks adapter compatibility after the merge."""
    def normalize(choice, space):
        if type(choice) is int and choice in space:
            return choice
        return ValidationFailure((Violation("UnknownChoice", "scope", "Unknown recipe"),))

    return GrammarAdapter(
        validate_space=lambda space: None,
        normalize=normalize,
        generate=lambda space, seed=0: space[seed % len(space)],
        mutate=lambda choice, space, **options: space[(space.index(choice) + 1) % len(space)],
        crossover=lambda first, second, space, **options: (first, second),
        compile=compile_candidate,
        payload=lambda choice: {
            "schema_version": 1,
            "genes": [{"aliases": ["scope"], "recipe": choice}],
        },
        domain="independent finite foreign grammar",
        context_guard=guard,
    )


def _search(adapter, evaluate, *, initial=(0, 1), charged=2, **options):
    return run_grammar_search(
        (0, 1, 2),
        GrammarSearchConfig(
            population_size=len(initial),
            generations=1,
            max_evaluations=charged,
            max_proposals=16,
            max_seconds=30,
        ),
        evaluate,
        fitness_context="independent fixed heldout context",
        initial_genotypes=initial,
        grammar_adapter=adapter,
        **options,
    )


@pytest.mark.parametrize("charge_before_compile, expected", [(False, 1), (True, 2)])
def test_attempt_identity_and_workflow_identity_remain_distinct(charge_before_compile, expected):
    """Opt-in charging uses canonical plans; legacy callers retain workflow caching."""
    evaluated = []

    def compile_candidate(choice, space):
        return SimpleNamespace(fingerprint="same-executable-workflow", choice=choice)

    def evaluate(workflow):
        evaluated.append(workflow.choice)
        return GrammarFitness(0.5, 0.5)

    result = _search(
        _adapter(compile_candidate), evaluate,
        charge_before_compile=charge_before_compile,
    )

    assert result["charged_evaluations"] == len(evaluated) == expected
    assert len(result["finite_archive"]) == expected
    assert {row["workflow_fingerprint"] for row in result["finite_archive"]} == {
        "same-executable-workflow"
    }
    if charge_before_compile:
        assert len({row["canonical_attempt_key"] for row in result["finite_archive"]}) == 2
        assert len(set(evaluated)) == 2
    else:
        assert all("canonical_attempt_key" not in row for row in result["finite_archive"])


def test_compile_exception_is_charged_cached_and_cannot_hide_a_valid_neighbor():
    """A repeated failed plan is one paid failure; a distinct plan can still succeed."""
    compiled, evaluated = [], []

    def compile_candidate(choice, space):
        compiled.append(choice)
        if choice == 0:
            raise ArithmeticError("independent compiler failure")
        return SimpleNamespace(fingerprint=f"workflow-{choice}", choice=choice)

    def evaluate(workflow):
        evaluated.append(workflow.choice)
        return GrammarFitness(0.25, 0.4)

    result = _search(
        _adapter(compile_candidate), evaluate,
        initial=(0, 0, 1), charge_before_compile=True,
    )

    assert result["charged_evaluations"] == 2
    assert compiled.count(0) == compiled.count(1) == 1
    assert evaluated == [1]
    assert result["failure_count"] == 1
    failures = [row for row in result["trace"]
                if row["event"] == "evaluated" and row["outcome"] == "CallbackFailure"]
    assert len(failures) == 1
    assert failures[0]["failure_details"]["code"] == "ArithmeticError"
    assert failures[0]["workflow_fingerprint"] is None
    assert len(result["finite_archive"]) == 1
    assert result["finite_archive"][0]["workflow_fingerprint"] == "workflow-1"


def test_source_invalidated_by_terminal_trace_cannot_leave_a_finite_archive():
    """The immutable context remains guarded through the final public return."""
    invalidated, evaluated = [], []

    def guard():
        if invalidated:
            return ResourceFailure("SourceChanged", "Source changed at terminal reporting")
        return None

    def trace_sink(event):
        if event["event"] == "stopped":
            invalidated.append(True)

    def evaluate(workflow):
        evaluated.append(workflow.fingerprint)
        return GrammarFitness(0.1, 0.2)

    result = _search(
        _adapter(lambda choice, space: SimpleNamespace(fingerprint=f"workflow-{choice}"), guard=guard),
        evaluate, initial=(0,), charged=1,
        trace_callback=trace_sink, charge_before_compile=True,
    )

    assert len(evaluated) == result["charged_evaluations"] == 1
    assert result["status"] == "failed"
    assert result["stop_reason"] == "context_invalidated"
    assert result["context_failure"]["code"] == "SourceChanged"
    assert result["finite_archive"] == result["best_genotypes"] == []
    assert result["trace"][-1]["stop_reason"] == "context_invalidated"


def test_domain_planner_receives_paid_progress_and_its_action_reaches_the_public_trace():
    """A generation number, proposal count, or fitness-component count cannot drive staging."""
    events, planner_calls = [], []

    def record(event):
        events.append(event)

    def plan(parent, space, **context):
        starts = sum(event["event"] == "evaluation_started" for event in events)
        assert context["charged"] == starts
        assert context["initial"] == 1 and context["maximum"] == 3
        assert type(context["seed"]) is int and context["seed"] >= 0
        planner_calls.append(context)
        return GrammarMutationDecision((parent + 1) % 3, "independent_group_move",
                                       (("stage", "fixed_test"), ("charged_seen", starts)))

    adapter = replace(
        _adapter(lambda choice, space: SimpleNamespace(fingerprint=f"workflow-{choice}", choice=choice)),
        mutate=lambda *args, **kwargs: pytest.fail("Explicit domain planner fell back to generic mutation"),
        mutation_planner=plan,
    )
    result = run_grammar_search(
        (0, 1, 2),
        GrammarSearchConfig(population_size=1, generations=4, max_evaluations=3,
                            max_proposals=16, max_seconds=30, mutation_only=True),
        lambda workflow: GrammarFitness(float(workflow.choice + 1), float(3 - workflow.choice)),
        fitness_context="independent planner progress", initial_genotypes=(0,),
        grammar_adapter=adapter, charge_before_compile=True, trace_callback=record,
    )
    assert planner_calls and planner_calls[0]["charged"] == 1
    assert result["charged_evaluations"] >= 2
    variations = [event for event in events if event["event"] == "variation"]
    assert variations
    assert all(event["kind"] == "independent_group_move" for event in variations)
    assert all(event["mutation_details"]["stage"] == "fixed_test" for event in variations)
    assert all(event["mutation_details"]["charged_seen"] >= 1 for event in variations)


@pytest.mark.parametrize("finished_at, eligible", [(101.0, True), (101.001, False)])
def test_deadline_eligibility_is_decided_on_completion_without_refunding_attempts(
        monkeypatch, finished_at, eligible):
    """A completed late value stays charged and cannot become an archived finalist."""
    clock = [100.0]
    monkeypatch.setattr(search_module, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    evaluated = []

    def evaluate(workflow):
        evaluated.append(workflow.fingerprint)
        clock[0] = finished_at
        return GrammarFitness(0.1, 0.2)

    result = _search(
        _adapter(lambda choice, space: SimpleNamespace(fingerprint=f"workflow-{choice}")),
        evaluate, initial=(0,), charged=1,
        absolute_deadline=101.0, strict_deadline=True, charge_before_compile=True,
    )

    assert len(evaluated) == result["charged_evaluations"] == 1
    assert len(result["finite_archive"]) == int(eligible)
    assert result["late_evaluation_count"] == int(not eligible)
    completed = [row for row in result["trace"] if row["event"] == "evaluated"]
    assert len(completed) == 1
    assert completed[0]["completed_within_deadline"] is eligible


def test_approved_evo_panel_counts_roles_without_silently_authorizing_compute():
    protocol = ExperimentProtocol()
    budget = protocol.budget()
    expected = {
        "searches": 27,
        "charged_attempts": 1728,
        "max_proposals": 17280,
        "baseline_sgd": 75120,
        "search_sgd": 345600,
        "control_sgd": 3000,
        "total_sgd": 423720,
        "max_finalists": 45,
        "latency_sessions": 204,
        "primary_contrast": "E2-E0",
        "secondary_contrasts": ("E1-E0", "E2-E1"),
        "cluster_launch_authorized": False,
        "main_gpu_seconds": None,
    }
    assert {key: budget[key] for key in expected} == expected
    assert "one tape per M0" in budget["recovery_tape"]
    jobs = protocol.jobs()
    assert len(jobs) == len(set(jobs)) == 27
    expected_blocks = {(model, seed) for model in protocol.model_seeds
                       for seed in protocol.search_seeds}
    assert {(model, seed) for model, seed, _ in jobs} == expected_blocks
    for block in expected_blocks:
        assert Counter(arm for model, seed, arm in jobs if (model, seed) == block) == Counter(SearchArm)
    assert all(type(arm) is SearchArm for _, _, arm in jobs)
    assert all(count == 9 for count in Counter(arm for _, _, arm in jobs).values())


@pytest.mark.parametrize("arms", [("E0", "E1", "E2"), list(SearchArm),
                                  (SearchArm.BASIC, SearchArm.STAGED),
                                  (SearchArm.BASIC,) * 3])
def test_panel_rejects_untyped_or_incomplete_search_arms(arms):
    with pytest.raises(ValueError):
        ExperimentProtocol(arms=arms)


@pytest.mark.parametrize("model_count", [1, 2, 4])
def test_each_baseline_has_pre_and_post_controls_for_both_batches_in_both_windows(model_count):
    protocol = replace(ExperimentProtocol(), model_seeds=tuple(range(model_count)))
    budget = protocol.budget()
    model_sessions = budget["max_finalists"] * 4
    # Two batches, two windows, and controls at both boundaries of every window.
    assert budget["latency_sessions"] - model_sessions == 8 * model_count


def test_pilot_is_separate_bounded_work_without_main_panel_authorization():
    budget = pilot_budget()
    assert (budget["models"], budget["recipes"], budget["max_seconds"]) == (1, 12, 7200)
    assert (budget["baseline_sgd"], budget["recovery_sgd"], budget["total_sgd"]) == (
        25040, 2400, 27440
    )
    assert budget["cluster_launch_authorized"] is False
    assert ExperimentProtocol().budget()["total_sgd"] == 423720


@given(initial=st.integers(min_value=1, max_value=16), remaining=st.integers(min_value=10, max_value=160))
@settings(max_examples=30, deadline=None)
def test_fixed_extended_mix_matches_the_discrete_staged_mix_after_the_prefix(initial, remaining):
    """The control averages charged slots, not generations or a continuous 30/40/30 split."""
    maximum = initial + remaining
    staged = [policy_for(SearchArm.STAGED, charged=initial + index,
                         maximum=maximum, initial=initial) for index in range(remaining)]
    # Integer comparisons are an independent oracle for the three interval boundaries.
    expected_stages = ["explore" if 10 * index < 3 * remaining else
                       "joint" if 10 * index < 7 * remaining else "refine"
                       for index in range(remaining)]
    assert [decision.stage for decision in staged] == expected_stages
    fixed = policy_for(SearchArm.EXTENDED, charged=maximum - 1,
                       maximum=maximum, initial=initial)
    for action, weight in fixed.weights:
        expected = math.fsum(dict(decision.weights)[action] for decision in staged) / remaining
        assert weight == pytest.approx(expected, abs=1e-14, rel=1e-14)
    assert set(dict(fixed.weights)) == set(Action)
    assert math.fsum(dict(fixed.weights).values()) == pytest.approx(1.0)
    assert dict(staged[0].weights)[Action.GROUP] > dict(staged[-1].weights)[Action.GROUP]
    assert dict(staged[0].weights)[Action.LOCAL] < dict(staged[-1].weights)[Action.LOCAL]
    assert policy_for(SearchArm.EXTENDED, charged=initial, maximum=maximum, initial=initial) == fixed


def _g1_metadata_space():
    """Build a checked CV capability description without creating any tensors."""
    from fedcore.algorithm.low_rank.compression_graph import CalibrationRef, ModelVersion
    from fedcore.algorithm.low_rank.cv_grammar import CVBlockDomain, CVGrammarSpace
    from fedcore.algorithm.heterogeneous.cv import (
        BlockRecipe, HeterogeneousDomain, HeterogeneousSpace,
    )

    recipes = tuple(BlockRecipe(keep, rank, ptq) for keep in (None, 32)
                    for rank in (None, 16) for ptq in (False, True))
    paths = ("layer1.0", "layer1.1", "layer2.0")
    base = CVGrammarSpace(
        ModelVersion("independent-g1-source", "a" * 64),
        CalibrationRef("b" * 64, (8, 3, 32, 32), "float32"),
        tuple(CVBlockDomain(path, tuple(dict.fromkeys(recipe.structural for recipe in recipes)))
              for path in paths),
        "c" * 64,
        44734408,
    )
    return HeterogeneousSpace(base, tuple(HeterogeneousDomain(path, recipes) for path in paths))


@given(indices=st.tuples(*(st.integers(min_value=0, max_value=7) for _ in range(3))),
       seed=st.integers(min_value=0, max_value=2**32 - 1), arm=st.sampled_from(tuple(SearchArm)))
@settings(max_examples=30, deadline=None)
def test_domain_mutations_are_atomic_closed_and_do_not_consume_global_rng(indices, seed, arm):
    from fedcore.algorithm.heterogeneous.cv import (
        HeterogeneousGene, HeterogeneousGenotype, normalize_genotype,
    )
    from fedcore.algorithm.heterogeneous.policy import mutate_with_policy

    space = _g1_metadata_space()
    parent = HeterogeneousGenotype(tuple(HeterogeneousGene(domain.path, domain.alternatives[index])
                                       for domain, index in zip(space.domains, indices)))
    before = random.getstate()
    decision = mutate_with_policy(parent, space, arm=arm, seed=seed, charged=20, maximum=64)
    replay = mutate_with_policy(parent, space, arm=arm, seed=seed, charged=20, maximum=64)
    assert random.getstate() == before
    assert decision == replay
    child = decision.genotype
    assert normalize_genotype(child, space) == child
    assert len(child.genes) == len(parent.genes) == 3
    changed = sum(left != right for left, right in zip(parent.genes, child.genes))
    assert changed >= 1
    assert all(gene.path == domain.path and gene.recipe in domain.alternatives
               for gene, domain in zip(child.genes, space.domains))
    if arm is SearchArm.BASIC:
        assert decision.kind == Action.LOCAL.value and changed == 1
    if decision.kind == Action.GROUP.value:
        assert 2 <= changed <= 3
    if decision.kind == Action.BUDGET.value:
        assert changed == 2
    assert dict(decision.details)["changed_scopes"] == changed


def test_ptq_choice_changes_evaluation_identity_while_preserving_the_float_structural_plan():
    from fedcore.algorithm.heterogeneous.cv import (
        BlockRecipe, HeterogeneousGene, HeterogeneousGenotype, compile_genotype,
        genotype_from_payload, genotype_payload,
    )
    from fedcore.algorithm.heterogeneous.contracts import Operation

    space = _g1_metadata_space()
    plain = HeterogeneousGenotype(tuple(HeterogeneousGene(domain.path, BlockRecipe(32, 16))
                                       for domain in space.domains))
    quantized = replace(plain, genes=(replace(plain.genes[0], recipe=BlockRecipe(32, 16, True)),)
                       + plain.genes[1:])
    assert genotype_from_payload(genotype_payload(quantized), space) == quantized
    first, second = compile_genotype(plain, space), compile_genotype(quantized, space)
    assert first.structural_workflow.fingerprint == second.structural_workflow.fingerprint
    assert first.fingerprint != second.fingerprint
    assert first.ptq_regions == () and second.ptq_regions == (space.domains[0].path,)
    assert len(second.deferred_ptq) == 1
    stage = second.deferred_ptq[0]
    assert stage.operation is Operation.PTQ
    assert stage.dependency.observation_mode == "current_student"
    assert second.fingerprint in stage.dependency.predecessor


def _host_fixture(tmp_path, monkeypatch, *, nspid="9001 21", gpu_namespace="pid:[400]",
                  births=(1500.0, 1500.0)):
    from fedcore.experiments import heterogeneous_host as host
    from fedcore.experiments.cv_independent_supervision import ProcessIdentity

    proc = tmp_path / "recorded-proc"
    (proc / "9001").mkdir(parents=True)
    (proc / "9001" / "status").write_text(f"Name:\towned-worker\nNSpid:\t{nspid}\n", encoding="ascii")
    namespaces = {
        str(proc / "9000" / "ns/pid"): "pid:[400]",
        str(proc / "9001" / "ns/pid"): gpu_namespace,
        str(proc / "self/ns/pid"): "pid:[100]",
    }
    monkeypatch.setattr(host.os, "readlink", lambda path: namespaces[str(path)])
    commands, probes = [], []
    samples = iter(births)

    def inspect(command, **kwargs):
        commands.append((command, kwargs))
        return SimpleNamespace(stdout="9000\n")

    def birth(pid):
        probes.append(pid)
        return next(samples)

    request = host.HostRequest("a" * 64, "b" * 32, ("9001, 10",),
                               (ProcessIdentity(21, 1500.0),))
    return host, request, proc, inspect, birth, commands, probes, namespaces


def test_host_mapping_uses_namespace_nspid_and_both_lifetime_samples(tmp_path, monkeypatch):
    host, request, proc, inspect, birth, commands, probes, _ = _host_fixture(tmp_path, monkeypatch)
    response = host.collect_host_binding(request, proc_root=proc, run=inspect, process_birth=birth)
    assert response["verified"] is True
    assert response["nonce"] == request.nonce and response["rows"] == list(request.rows)
    assert response["entries"] == [{"gpu_pid": 9001, "local": {"pid": 21, "birth": 1500.0}}]
    assert probes == [9001, 9001]
    assert len(commands) == 1
    command, options = commands[0]
    assert command == ["docker", "inspect", "--format", "{{.State.Pid}}", request.container_id]
    assert options["shell"] is False and options["timeout"] <= 1


@pytest.mark.parametrize("violation", ["pid_reused", "foreign_namespace", "nested_namespace", "unowned_local"])
def test_host_mapping_rejects_lifetime_namespace_and_ownership_violations(tmp_path, monkeypatch, violation):
    host, request, proc, inspect, birth, _, _, _ = _host_fixture(
        tmp_path, monkeypatch,
        nspid="9001 8 21" if violation == "nested_namespace" else
              "9001 22" if violation == "unowned_local" else "9001 21",
        gpu_namespace="pid:[999]" if violation == "foreign_namespace" else "pid:[400]",
        births=(1500.0, 1500.001) if violation == "pid_reused" else (1500.0, 1500.0),
    )
    with pytest.raises(RuntimeError):
        host.collect_host_binding(request, proc_root=proc, run=inspect, process_birth=birth)


def test_host_probe_detects_container_namespace_change_before_return(tmp_path, monkeypatch):
    host, request, proc, inspect, birth, _, _, namespaces = _host_fixture(tmp_path, monkeypatch)

    def changing_birth(pid):
        result = birth(pid)
        namespaces[str(proc / "9000" / "ns/pid")] = "pid:[777]"
        return result

    with pytest.raises(RuntimeError, match="namespace changed"):
        host.collect_host_binding(request, proc_root=proc, run=inspect, process_birth=changing_birth)


def test_empty_gpu_preflight_still_verifies_the_container_without_guessing_process_ownership(tmp_path, monkeypatch):
    host, request, proc, inspect, _, commands, _, _ = _host_fixture(tmp_path, monkeypatch)
    response = host.collect_host_binding(replace(request, rows=()), proc_root=proc, run=inspect,
                                         process_birth=lambda pid: pytest.fail("Empty GPU sample resolved a process"))
    assert response["verified"] is True and response["entries"] == []
    assert len(commands) == 1


@pytest.mark.parametrize("violation", ["old_nonce", "different_container", "different_rows", "foreign_local", "unverified"])
def test_binding_transport_rejects_a_replayed_or_mismatched_host_response(monkeypatch, violation):
    from fedcore.experiments import heterogeneous_host as host
    from fedcore.experiments.cv_independent_supervision import ProcessIdentity

    calls = []

    def transport(command, payload, timeout):
        calls.append((command, payload, timeout))
        request = json.loads(payload)
        response = {
            "schema": "fedcore-host-binding-v1", "container_id": request["container_id"],
            "nonce": request["nonce"], "rows": request["rows"], "verified": True,
            "entries": [{"gpu_pid": 9001, "local": {"pid": 21, "birth": 1500.0}}],
            "evidence": "independent authenticated test transport",
        }
        if violation == "old_nonce":
            response["nonce"] = "old-request-nonce"
        elif violation == "different_container":
            response["container_id"] = "c" * 64
        elif violation == "different_rows":
            response["rows"] = ["9002, 10"]
        elif violation == "foreign_local":
            response["entries"][0]["local"]["pid"] = 22
        else:
            response["verified"] = False
        return json.dumps(response)

    monkeypatch.setattr(host, "_run_bounded", transport)
    monkeypatch.setattr(host.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Test started a process"))
    provider = host.BoundedBindingProvider(("explicit-host-observer",), "a" * 64, timeout=1.0)
    with pytest.raises(RuntimeError):
        provider(("9001, 10",), (ProcessIdentity(21, 1500.0),))
    assert len(calls) == 1
    assert calls[0][0] == ("explicit-host-observer",) and calls[0][2] == 1.0


def test_recovery_updates_the_parameters_which_belong_to_the_transferred_model():
    """A CPU double reproduces legal parameter rebinding during a device transfer."""
    import torch
    from torch import nn
    from fedcore.experiments.heterogeneous_runtime import RecoveryConfig, recovery_tape, recover_owned

    class RebindingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros((2, 1)))

        def to(self, *args, **kwargs):
            self.weight = nn.Parameter(self.weight.detach().clone())
            return self

        def forward(self, inputs):
            return inputs @ self.weight.T

    config = RecoveryConfig(updates=1, batch_size=2)
    tape = recovery_tape((0, 1), model_seed=7, config=config)
    model = RebindingModel()
    previous_parameter = model.weight
    result = recover_owned(
        model, tape,
        lambda indices, seed: (torch.ones((2, 1), dtype=torch.float32),
                               torch.zeros(2, dtype=torch.int64)),
        config=config, device="cpu", deadline=10.0, clock=lambda: 0.0,
    )
    assert model.weight is not previous_parameter
    assert result["updates"] == 1
    assert not torch.equal(model.weight.detach(), torch.zeros_like(model.weight))
    assert torch.equal(previous_parameter.detach(), torch.zeros_like(previous_parameter))
    assert model.training is False and model.weight.device.type == "cpu"
