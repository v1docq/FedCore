"""Frozen local-2 catalog laws, independently counted storage and role seals."""
from dataclasses import FrozenInstanceError, replace
from itertools import product
import math
import random

import pytest
import torch

from fedcore.algorithm.low_rank.compression_grammar import ASVDChoice, compile_genotype
from fedcore.algorithm.low_rank.plans import ValidationFailure
from fedcore.experiments.compression_search_digits_protocol import (
    ADMISSION_GENOTYPES, ATTEMPT_BUDGET, COMMON_READY_BYTES, DENSE_TENSOR_BYTES,
    INITIAL_GENOTYPES, READY_CATALOG, SEARCH_SEEDS, SELECTION_BYTE_BUDGETS,
    TIME_BUDGET, TENSOR_BYTE_BUDGET, TUNED_CATALOG, canonical_solution_id,
    catalog_by_id, expected_tensor_bytes, physical_mode_order, random_search_stream)


def test_exact_catalog_cross_product_and_shared_prefix_are_frozen():
    for catalog, count in ((READY_CATALOG, 15), (TUNED_CATALOG, 440)):
        observed = {tuple((gene.aliases[0], gene.choice) for gene in value.genes) for value in catalog.genotypes}
        reference = {((path, choice),) for path in ("0", "2") for choice in catalog.alternatives}
        reference |= {(("0", left), ("2", right)) for left, right in product(catalog.alternatives, repeat=2)}
        assert observed == reference
        assert len(catalog_by_id(catalog)) == count
        assert set(map(canonical_solution_id, INITIAL_GENOTYPES)) <= set(catalog_by_id(catalog))
        with pytest.raises(FrozenInstanceError):
            catalog.name = "changed"
    assert set(catalog_by_id(READY_CATALOG)) < set(catalog_by_id(TUNED_CATALOG))
    assert len(TUNED_CATALOG.alternatives) == 20
    assert {choice.rank for choice in TUNED_CATALOG.alternatives} == {4, 8, 16, 24, 32}
    assert {choice.spec.alpha for choice in TUNED_CATALOG.alternatives if type(choice) is ASVDChoice} == {.25, .5}
    assert all(len(item.genes) == 1 for item in ADMISSION_GENOTYPES)
    assert [item.genes[0].choice.rank for item in ADMISSION_GENOTYPES] == [4, 32, 32]


@pytest.mark.parametrize("catalog", (READY_CATALOG, TUNED_CATALOG))
@pytest.mark.parametrize("seed", SEARCH_SEEDS)
def test_uniform_controls_cover_each_candidate_once_and_do_not_touch_global_rng(catalog, seed):
    before = random.getstate()
    stream = random_search_stream(catalog, seed)
    remaining = [value for value in catalog.genotypes if value not in INITIAL_GENOTYPES]
    random.Random(seed).shuffle(remaining)
    assert stream[:4] == INITIAL_GENOTYPES
    assert stream[4:] == tuple(remaining)
    assert len(stream) == len(set(map(canonical_solution_id, stream))) == len(catalog.genotypes)
    assert random.getstate() == before


def test_integer_storage_formula_includes_biases_and_protected_classifier():
    assert DENSE_TENSOR_BYTES == 4 * (64*128 + 128 + 128*64 + 64 + 64*10 + 10)
    for value in TUNED_CATALOG.genotypes:
        ranks = {gene.aliases[0]: gene.choice.rank for gene in value.genes}
        reference_elements = 64*10 + 10
        for path, inputs, outputs in (("0", 64, 128), ("2", 128, 64)):
            reference_elements += outputs
            reference_elements += ranks[path]*(inputs+outputs) if path in ranks else inputs*outputs
        assert expected_tensor_bytes(value) == reference_elements * 4
        assert expected_tensor_bytes(value) <= TENSOR_BYTE_BUDGET
    assert min(map(expected_tensor_bytes, TUNED_CATALOG.genotypes)) == 9512
    assert max(map(expected_tensor_bytes, TUNED_CATALOG.genotypes)) == 60_712
    assert {expected_tensor_bytes(value) for value in READY_CATALOG.genotypes} == set(COMMON_READY_BYTES)
    assert SELECTION_BYTE_BUDGETS == tuple(math.floor(DENSE_TENSOR_BYTES*fraction) for fraction in (.3, .6, .85))


def test_all_440_catalog_entries_compile_against_the_frozen_dense_source():
    from fedcore.experiments.compression_search_digits_evaluation import fresh_model, prepare_space
    from fedcore.experiments.compression_search_evaluation import EvaluationSource
    from fedcore.experiments.protocol import TensorSplit
    model = fresh_model()
    # Synthetic data keep the experimental Digits source unopened in tests.
    calibration = TensorSplit(torch.ones(3, 64), torch.zeros(3, dtype=torch.long), ("a", "b", "c"))
    source = EvaluationSource(19901, model, calibration, calibration, {}, {}, torch.empty(0))
    space = prepare_space(source, TUNED_CATALOG)
    for value in TUNED_CATALOG.genotypes:
        assert not isinstance(compile_genotype(value, space), ValidationFailure)
    assert canonical_solution_id(replace(INITIAL_GENOTYPES[2], genes=INITIAL_GENOTYPES[2].genes[::-1])) == canonical_solution_id(INITIAL_GENOTYPES[2])


def test_frozen_search_limits_and_crossover_proposal_capacity():
    assert (ATTEMPT_BUDGET.max_evaluations, ATTEMPT_BUDGET.max_proposals,
        ATTEMPT_BUDGET.max_seconds, ATTEMPT_BUDGET.hard_seconds) == (24, 192, 45, 60)
    assert (TIME_BUDGET.max_evaluations, TIME_BUDGET.max_proposals,
        TIME_BUDGET.max_seconds, TIME_BUDGET.hard_seconds) == (440, 1024, 30, 45)
    assert ["".join(physical_mode_order(seed)) for seed in SEARCH_SEEDS] == ["RTE", "TER", "ERT"]
