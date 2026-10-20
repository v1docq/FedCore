"""Frozen protocol counts, identities, paired ordering and RNG ownership."""
from collections import Counter
from dataclasses import FrozenInstanceError, replace
from itertools import product
import random

import pytest

from fedcore.algorithm.low_rank.compression_grammar import ASVDChoice, WeightedChoice
from fedcore.experiments.compression_search_protocol import (
    ADMISSIBILITY_STREAM_SEEDS, ATTEMPT_BUDGET, CACHE_STREAM_SEEDS,
    G5_CASES, G5_NEGATIVE_CASES, INITIAL_GENOTYPES, MODEL_SEEDS,
    READY_CATALOG, SEARCH_SEEDS, TIME_BUDGET, TUNED_CATALOG,
    admissibility_branch_order, admissibility_stream, cache_branch_order,
    cache_replay_stream, canonical_solution_id, catalog_by_id,
    physical_mode_order, random_search_stream)


def test_catalogs_have_exact_cross_product_and_menu_nesting():
    ready, tuned = READY_CATALOG, TUNED_CATALOG
    assert len(ready.alternatives) == 3
    assert len(tuned.alternatives) == 8
    for catalog, expected in ((ready, 15), (tuned, 80)):
        assert len(catalog.genotypes) == len(catalog_by_id(catalog)) == expected
        observed = {tuple((gene.aliases[0], gene.choice) for gene in item.genes)
                    for item in catalog.genotypes}
        independent = {((path, choice),) for path in ("0", "2") for choice in catalog.alternatives}
        independent |= {(("0", left), ("2", right))
                        for left, right in product(catalog.alternatives, repeat=2)}
        assert observed == independent
        assert catalog.initial_genotypes == INITIAL_GENOTYPES
        assert set(map(canonical_solution_id, INITIAL_GENOTYPES)) <= set(catalog_by_id(catalog))
    assert set(catalog_by_id(ready)) < set(catalog_by_id(tuned))
    assert {choice.rank for choice in ready.alternatives} == {8}
    assert {choice.rank for choice in tuned.alternatives} == {8, 16}
    assert {choice.spec.alpha for choice in tuned.alternatives if type(choice) is ASVDChoice} == {.25, .5}
    assert all((choice.policy.ridge, choice.policy.rcond, choice.policy.nullspace_policy)
               == (0, None, "support_only") for choice in tuned.alternatives if type(choice) is WeightedChoice)
    with pytest.raises(FrozenInstanceError):
        ready.name = "changed"
    lookup = catalog_by_id(tuned)
    lookup.clear()
    assert len(catalog_by_id(tuned)) == 80


def test_canonical_identity_ignores_scope_order_and_preserves_choice_semantics():
    pair = INITIAL_GENOTYPES[2]
    assert canonical_solution_id(replace(pair, genes=pair.genes[::-1])) == canonical_solution_id(pair)
    changed = replace(pair, genes=(replace(pair.genes[0], choice=ASVDChoice(8, replace(pair.genes[0].choice.spec, alpha=.25))), pair.genes[1]))
    assert canonical_solution_id(changed) != canonical_solution_id(pair)


@pytest.mark.parametrize("catalog", (READY_CATALOG, TUNED_CATALOG))
@pytest.mark.parametrize("seed", SEARCH_SEEDS)
def test_random_controls_are_complete_uniform_permutations_after_shared_prefix(catalog, seed):
    before = random.getstate()
    stream = random_search_stream(catalog, seed)
    assert stream == random_search_stream(catalog, seed)
    assert stream[:4] == INITIAL_GENOTYPES
    assert len(stream) == len(set(map(canonical_solution_id, stream))) == len(catalog.genotypes)
    assert set(map(canonical_solution_id, stream)) == set(catalog_by_id(catalog))
    # The independent construction permutes each remaining catalog item once.
    remaining = [item for item in catalog.genotypes if item not in INITIAL_GENOTYPES]
    random.Random(seed).shuffle(remaining)
    assert stream[4:] == tuple(remaining)
    assert random.getstate() == before


def test_frozen_g5_words_and_negative_metadata_are_separate():
    assert len(G5_CASES) == 155
    assert {case.word for case in G5_CASES} == {"".join(word)
        for length in (1, 2, 3) for word in product("PLQDR", repeat=length)}
    assert len({case.case_id for case in G5_CASES}) == 155
    assert all(not case.schema_negative for case in G5_CASES)
    assert len(G5_NEGATIVE_CASES) == 8
    assert len({case.case_id for case in G5_NEGATIVE_CASES}) == 8
    assert all(case.schema_negative and case.expected_code for case in G5_NEGATIVE_CASES)
    expected_ids = {case.case_id for case in G5_CASES + G5_NEGATIVE_CASES}
    for seed in ADMISSIBILITY_STREAM_SEEDS:
        stream = admissibility_stream(seed)
        assert len(stream) == 163
        assert {case.case_id for case in stream} == expected_ids
        assert stream == admissibility_stream(seed)
    assert ["".join(admissibility_branch_order(seed)) for seed in ADMISSIBILITY_STREAM_SEEDS] == ["AB", "BA", "BA", "AB", "AB", "BA"]


@pytest.mark.parametrize("seed", CACHE_STREAM_SEEDS)
def test_cache_stream_has_twelve_sources_four_occurrences_and_no_rng_effect(seed):
    before = random.getstate()
    stream = cache_replay_stream(seed)
    assert stream == cache_replay_stream(seed)
    counts = Counter(map(canonical_solution_id, stream))
    assert len(stream) == 48 and len(counts) == 12 and set(counts.values()) == {4}
    assert set(counts) <= set(catalog_by_id())
    assert random.getstate() == before


def test_search_orders_and_budgets_match_the_approved_protocol():
    assert ["".join(physical_mode_order(seed)) for seed in SEARCH_SEEDS] == ["RTE", "TER", "ERT"]
    assert (ATTEMPT_BUDGET.max_evaluations, ATTEMPT_BUDGET.max_proposals,
            ATTEMPT_BUDGET.max_seconds, ATTEMPT_BUDGET.hard_seconds) == (12, 96, 60, 75)
    assert (TIME_BUDGET.max_evaluations, TIME_BUDGET.max_proposals,
            TIME_BUDGET.max_seconds, TIME_BUDGET.hard_seconds) == (80, 256, 30, 45)
    assert ["".join(cache_branch_order(model, stream)) for model in MODEL_SEEDS
            for stream in CACHE_STREAM_SEEDS] == ["AB", "BA", "AB", "BA", "AB", "BA", "AB", "BA", "AB"]
