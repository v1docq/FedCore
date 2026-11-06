"""Closed recipes preserve coordinates, order and complete CV scope ownership."""
from dataclasses import replace
import pytest

from fedcore.algorithm.heterogeneous.contracts import (
    Dependency, JoinApplyReplacements, Operation, ParallelDisjoint, Representation,
    Scope, Sequence, Step, TensorRepresentation)
from fedcore.algorithm.heterogeneous.cv import (
    BlockRecipe, HeterogeneousDomain, HeterogeneousGene, HeterogeneousGenotype,
    HeterogeneousSpace, compile_genotype, crossover_genotypes, generate_genotype,
    genotype_from_payload, genotype_payload, normalize_genotype, validate_workflow)
from fedcore.algorithm.low_rank.compression_graph import CalibrationRef, ModelVersion
from fedcore.algorithm.low_rank.cv_grammar import CVBlockDomain, CVGrammarSpace
from fedcore.algorithm.low_rank.plans import ValidationFailure


def space():
    choices = tuple(BlockRecipe(k, r, q) for k in (None, 32) for r in (None, 8) for q in (False, True))
    domains = tuple(HeterogeneousDomain(path, choices) for path in ("layer1.0", "layer1.1"))
    structural = CVGrammarSpace(ModelVersion("test", "a" * 64), CalibrationRef("b" * 64, (2, 3, 32, 32), "float32"),
        tuple(CVBlockDomain(d.path, tuple(dict.fromkeys(c.structural for c in d.alternatives))) for d in domains),
        "c" * 64, 32 * 1024**2)
    return HeterogeneousSpace(structural, domains)


def test_closed_menu_compiles_each_combination_and_defers_ptq_until_recovery():
    s = space()
    for choice in s.domains[0].alternatives:
        genotype = HeterogeneousGenotype((HeterogeneousGene("layer1.0", choice), HeterogeneousGene("layer1.1", BlockRecipe())))
        compiled = compile_genotype(genotype, s)
        assert not isinstance(compiled, ValidationFailure)
        assert validate_workflow(compiled) is None
        assert compiled.procedure.parallel.branches[0].external_channels == (64, 64)
        assert all(step.operation is not Operation.PTQ for branch in compiled.procedure.parallel.branches for step in branch.recipe.steps)
        assert bool(compiled.deferred_ptq) == choice.ptq
        if choice.ptq:
            assert compiled.deferred_ptq[0].dependency.observation_mode == "current_student"


def test_payload_roundtrip_canonical_order_and_complete_scope_crossover():
    s = space()
    first, second = generate_genotype(s, seed=2), generate_genotype(s, seed=7)
    assert normalize_genotype(replace(first, genes=first.genes[::-1]), s) == first
    assert genotype_from_payload(genotype_payload(first), s) == first
    children = crossover_genotypes(first, second, s, seed=3)
    for child in children:
        assert normalize_genotype(child, s) == child
        assert {g.path for g in child.genes} == {"layer1.0", "layer1.1"}
    malformed = genotype_payload(first)
    malformed["genes"][0]["recipe"]["unknown"] = 1
    assert isinstance(genotype_from_payload(malformed, s), ValidationFailure)


def test_invalid_rank_width_and_tampered_dependencies_fail_before_execution():
    s = space()
    genotype = generate_genotype(s, seed=0)
    assert isinstance(normalize_genotype(replace(genotype, genes=genotype.genes[:1]), s), ValidationFailure)
    assert isinstance(normalize_genotype(HeterogeneousGenotype((HeterogeneousGene("layer1.0", BlockRecipe(65, 8)), genotype.genes[1])), s), ValidationFailure)
    workflow = compile_genotype(genotype, s)
    assert isinstance(validate_workflow(replace(workflow, fingerprint="forged")), ValidationFailure)


def test_typed_sequences_reject_noncommuting_or_mismatched_transitions():
    original = TensorRepresentation(Representation.DENSE, (64, 576), 64)
    compact = TensorRepresentation(Representation.COMPACT, (64, 288), 32,
                                   channel_coordinates="frozen_M0_conv1_l1_selected_original_order")
    factors = replace(compact, kind=Representation.FACTORS, rank=8)
    dependency = Dependency("source", "data", "coordinates")
    prune = Step(Operation.PRUNE, original, compact, dependency)
    svd = Step(Operation.SVD, compact, factors, dependency)
    assert Sequence((prune, svd))
    with pytest.raises(ValueError, match="incompatible"):
        Sequence((svd, prune))
    with pytest.raises(ValueError, match="SVD"):
        Step(Operation.SVD, original, original, dependency)
    with pytest.raises(ValueError, match="predecessor"):
        Dependency("source", "data", "coordinates", "current_student")


def test_parallel_join_rejects_nested_alias_writes_and_mixed_source_versions():
    branch = Scope("layer1.0", Sequence(()), "source", ("layer1.0",), ("layer1.0",), (64, 64))
    other = Scope("layer1.1", Sequence(()), "source", ("layer1.1",), ("layer1.1",), (64, 64))
    assert JoinApplyReplacements(ParallelDisjoint((branch, other), "source"), "candidate")
    with pytest.raises(ValueError, match="overlap"):
        ParallelDisjoint((branch, replace(other, write_closure=("layer1.0.conv1",))), "source")
    with pytest.raises(ValueError, match="source"):
        ParallelDisjoint((branch, replace(other, source_version="different")), "source")
