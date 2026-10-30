"""Closed grammar laws and the actual FedcoreEvoOptimizer boundary."""
from dataclasses import replace

import pytest
import torch
from torch import nn
from hypothesis import given, settings, strategies as st

from fedcore.algorithm.low_rank.compression_graph import CalibrationRef, ModelVersion
from fedcore.algorithm.low_rank.compression_grammar import (
    ASVDChoice, AffinePCAChoice, CompressionGenotype, GrammarSpace, MutationKind,
    ScopeDomain, ScopeGene, WeightedChoice, compile_genotype, crossover_genotypes,
    generate_genotype, genotype_from_payload, genotype_payload, mutate_genotype,
    normalize_genotype, validate_workflow)
from fedcore.algorithm.low_rank.grammar_execution import (
    PoorQuality, ResourceFailure, StructuralInvalid, WorkflowSuccess, assess_quality,
    execute_workflow, prepare_grammar_space)
from fedcore.algorithm.low_rank.grammar_search import (
    GrammarFitness, GrammarSearchConfig, graph_genotype, grammar_graph, run_grammar_search)
from fedcore.algorithm.low_rank.method_specs import ASVD
from fedcore.algorithm.low_rank.plans import MetricPolicy, OperatorDescriptor, ValidationFailure


def space_fixture():
    choices = (WeightedChoice(1), WeightedChoice(2),
        WeightedChoice(1, "one_layer", MetricPolicy(ridge=.01)),
        ASVDChoice(1, ASVD(abs_mode="abs_max", alpha=.7)), AffinePCAChoice(1))
    return GrammarSpace(ModelVersion("test-v1", "a"*64),
        CalibrationRef("b"*64, (6, 4), "float32"),
        tuple(ScopeDomain(OperatorDescriptor(path, (4, 4), "Linear"), (path,), choices)
              for path in ("left", "right")))


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 2**16), st.sampled_from(tuple(MutationKind)))
def test_valid_construction_and_normalization(seed, kind):
    space = space_fixture()
    genotype = generate_genotype(space, seed=seed)
    assert normalize_genotype(normalize_genotype(genotype, space), space) == genotype
    changed = mutate_genotype(genotype, space, kind=kind, seed=seed)
    if kind is MutationKind.RECOVERY:
        assert isinstance(changed, ValidationFailure)
        return
    assert not isinstance(changed, ValidationFailure)
    workflow = compile_genotype(changed, space)
    assert validate_workflow(workflow) is None
    children = crossover_genotypes(genotype, changed, space, seed=seed)
    assert all(validate_workflow(compile_genotype(child, space)) is None for child in children)


def test_centering_cannot_be_repaired_and_stale_graphs_are_recompiled():
    space = space_fixture()
    bad = CompressionGenotype((ScopeGene(("left",), AffinePCAChoice(1, bias_policy="preserve")),))
    assert normalize_genotype(bad, space).violations[0].code == "IncompatibleCenteringBias"
    original = CompressionGenotype((ScopeGene(("left",), WeightedChoice(1)),))
    changed = mutate_genotype(original, space, kind=MutationKind.CENTERING)
    assert type(changed.genes[0].choice) is AffinePCAChoice
    old, new = compile_genotype(original, space), compile_genotype(changed, space)
    assert old.external_artifacts != new.external_artifacts
    assert validate_workflow(replace(new, nodes=old.nodes)) is not None
    assert isinstance(normalize_genotype(CompressionGenotype((ScopeGene(("left",), WeightedChoice(8)),)), space), ValidationFailure)


def test_tagged_transport_round_trip_and_conditional_unknown_fields():
    import json
    space = space_fixture()
    for choice in space.domains[0].alternatives:
        genotype = CompressionGenotype((ScopeGene(("left",), choice),))
        payload = json.loads(json.dumps(genotype_payload(genotype), allow_nan=False))
        assert genotype_from_payload(payload, space) == genotype
        payload["genes"][0]["choice"]["unexpected"] = 1
        assert isinstance(genotype_from_payload(payload, space), ValidationFailure)
    assert isinstance(genotype_from_payload({"schema_version": 1, "genes": [None]}, space), ValidationFailure)


@pytest.mark.parametrize("kind", ["weighted", "afm", "asvd"])
def test_execute_real_profile_and_preserve_source(kind):
    torch.manual_seed(11)
    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Tanh(), nn.Linear(4, 3)).eval()
    data = torch.randn(8, 4) + 2
    before = {key: value.clone() for key, value in model.state_dict().items()}
    choice = {"weighted": WeightedChoice(1), "afm": AffinePCAChoice(1), "asvd": ASVDChoice(1)}[kind]
    space = prepare_grammar_space(model, data, alternatives={"0": (choice,)})
    assert not isinstance(space, ValidationFailure), space
    workflow = compile_genotype(CompressionGenotype((ScopeGene(("0",), choice),)), space)
    result = execute_workflow(model, data, workflow)
    assert type(result) is WorkflowSuccess, result
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    assert result.model(data).shape == (8, 3)
    assert result.evidence["apply_count"] == 1
    if kind == "afm":
        # Affine correction preserves mean output even for a biasless source.
        assert result.model[0].bias is not None
        torch.testing.assert_close(result.model[0](data).mean(0), model[0](data).mean(0), atol=2e-5, rtol=2e-5)
    assert type(assess_quality(result, 10., 1.)) is PoorQuality
    changed_data = data.clone()
    changed_data[0, 0] += 1
    assert type(execute_workflow(model, changed_data, workflow)) is StructuralInvalid


def test_actual_inventory_budget_and_afm_observation_guard():
    class Mutating(nn.Sequential):
        def forward(self, value):
            value.add_(1)
            return super().forward(value)
    model, data = Mutating(nn.Linear(4, 4)).eval(), torch.randn(6, 4)
    original = data.clone()
    space = prepare_grammar_space(model, data, alternatives={"0": (AffinePCAChoice(1),)})
    genotype = CompressionGenotype((ScopeGene(("0",), AffinePCAChoice(1)),))
    outcome = execute_workflow(model, data, compile_genotype(genotype, space))
    assert type(outcome) is StructuralInvalid
    assert outcome.violations[0].code == "UnsupportedObservationEffect"
    assert torch.equal(data, original)
    ordinary = nn.Sequential(nn.Linear(4, 4)).eval()
    options = replace(space.options, tensor_byte_budget=1)
    bounded = prepare_grammar_space(ordinary, data, alternatives={"0": (AffinePCAChoice(1),)}, options=options)
    assert type(execute_workflow(ordinary, data, compile_genotype(genotype, bounded))) is ResourceFailure


def test_actual_golem_reproduction_uses_grammar_and_native_round_trip():
    space = space_fixture()
    initial = tuple(CompressionGenotype((ScopeGene(("left",), choice),))
                    for choice in space.domains[0].alternatives[:4])
    for genotype in initial:
        assert graph_genotype(grammar_graph(genotype), space) == genotype
    called = []
    def evaluate(workflow):
        called.append(workflow)
        quality = sum(gene.choice.rank for gene in workflow.genotype.genes)
        return GrammarFitness(quality, 10. / quality, quality_accepted=False)
    result = run_grammar_search(space, GrammarSearchConfig(seed=13, population_size=4,
        generations=2, max_evaluations=12, max_seconds=40), evaluate,
        fitness_context="explicit synthetic heldout evaluator; no local solver metric",
        initial_genotypes=initial)
    assert result["status"] == "succeeded", result
    assert result["charged_evaluations"] == len(called)
    assert any(event.get("operator") == "grammar_mutation" for event in result["trace"])
    assert any(event.get("operator") == "grammar_crossover" for event in result["trace"])
    assert any(event["event"] == "population" and event["generation"] > 0 for event in result["trace"])
    assert result["quality_rejection_count"] == len(called)


def test_all_failed_callbacks_are_typed_charged_and_bounded():
    space = space_fixture()
    result = run_grammar_search(space, GrammarSearchConfig(population_size=2,
        generations=1, max_evaluations=2, max_seconds=10),
        lambda workflow: (_ for _ in ()).throw(RuntimeError("callback failed")),
        fitness_context="explicit failure fixture")
    assert result["status"] == "no_valid_fitness"
    assert 0 < result["charged_evaluations"] <= 2
    assert result["failure_count"] == result["charged_evaluations"]
    assert all(event["fitness"] is None for event in result["trace"] if event["event"] == "evaluated")
