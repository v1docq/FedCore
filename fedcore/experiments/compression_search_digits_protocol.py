"""Frozen DigitsMLP experiment metadata, separate from the historical package.

Catalogs own immutable genotype descriptions, never tensors or measurements.
Uniform controls shuffle each remaining candidate once after the shared prefix.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import random

from fedcore.algorithm.low_rank.compression_grammar import (
    ASVDChoice, AffinePCAChoice, CompressionGenotype, ScopeGene, WeightedChoice)
from fedcore.algorithm.low_rank.method_specs import ASVD
from .compression_search_protocol import canonical_solution_id

SPLIT_SEEDS = (19901, 19911, 19921)
MODEL_SEEDS = SPLIT_SEEDS
SEARCH_SEEDS = (11, 22, 33)
INITIALIZATION_SEED = TRAIN_SHUFFLE_SEED = 101
SCOPE_PATHS = ("0", "2")
RANKS = (4, 8, 16, 24, 32)
DENSE_TENSOR_BYTES = 68_904
TENSOR_BYTE_BUDGET = 65_458
SELECTION_BYTE_BUDGETS = (20_671, 41_342, 58_568)
COMMON_READY_BYTES = (15_656, 42_280)
MAX_WORKSPACE_BYTES = 64 * 1024**2
MAX_PEAK_BYTES = 8 * 1024**3
ATTEMPT_LIMIT = 10_300
UPDATES_PER_MODEL = 680
UPDATE_LIMIT = 2040
ROLE_SIZES = {"train": 1078, "validation": 269, "calibration": 180, "test": 270}
HYPERVOLUME_REFERENCE = (2., 1.05)


@dataclass(frozen=True)
class SearchBudget:
    name: str
    max_evaluations: int
    max_proposals: int
    max_seconds: float
    hard_seconds: float


ATTEMPT_BUDGET = SearchBudget("attempts", 24, 192, 45., 60.)
TIME_BUDGET = SearchBudget("time", 440, 1024, 30., 45.)

_weighted, _asvd, _afm = WeightedChoice(8), ASVDChoice(8, ASVD(alpha=.5)), AffinePCAChoice(8)
INITIAL_GENOTYPES = (
    CompressionGenotype((ScopeGene(("0",), _weighted),)),
    CompressionGenotype((ScopeGene(("2",), _weighted),)),
    CompressionGenotype((ScopeGene(("0",), _asvd), ScopeGene(("2",), _afm))),
    CompressionGenotype((ScopeGene(("0",), _afm), ScopeGene(("2",), _asvd))),
)


@dataclass(frozen=True)
class SearchCatalog:
    name: str
    alternatives: tuple
    genotypes: tuple[CompressionGenotype, ...]
    initial_genotypes: tuple[CompressionGenotype, ...] = INITIAL_GENOTYPES


def _catalog(name, alternatives):
    singles = tuple(CompressionGenotype((ScopeGene((path,), choice),))
                    for path in SCOPE_PATHS for choice in alternatives)
    pairs = tuple(CompressionGenotype((ScopeGene(("0",), left), ScopeGene(("2",), right)))
                  for left, right in product(alternatives, repeat=2))
    return SearchCatalog(name, alternatives, singles + pairs)


READY_CATALOG = _catalog("digits-ready", (_weighted, _asvd, _afm))
TUNED_CATALOG = _catalog("digits-tuned", tuple(choice for rank in RANKS for choice in (
    WeightedChoice(rank), ASVDChoice(rank, ASVD(alpha=.25)),
    ASVDChoice(rank, ASVD(alpha=.5)), AffinePCAChoice(rank))))
ADMISSION_GENOTYPES = (
    CompressionGenotype((ScopeGene(("0",), WeightedChoice(4)),)),
    CompressionGenotype((ScopeGene(("0",), ASVDChoice(32, ASVD(alpha=.25))),)),
    CompressionGenotype((ScopeGene(("2",), AffinePCAChoice(32)),)),
)


def catalog_by_id(catalog=TUNED_CATALOG):
    return {canonical_solution_id(value): value for value in catalog.genotypes}


def random_search_stream(catalog, seed):
    if type(catalog) is not SearchCatalog or seed not in SEARCH_SEEDS:
        raise ValueError("Use a frozen Digits catalog and search seed")
    initial = {canonical_solution_id(value) for value in INITIAL_GENOTYPES}
    remaining = [value for value in catalog.genotypes if canonical_solution_id(value) not in initial]
    random.Random(seed).shuffle(remaining)
    return INITIAL_GENOTYPES + tuple(remaining)


def physical_mode_order(seed):
    try:
        return {11: ("R", "T", "E"), 22: ("T", "E", "R"), 33: ("E", "R", "T")}[seed]
    except KeyError as error:
        raise ValueError("Use a frozen search seed") from error


def expected_tensor_bytes(genotype):
    """Independent integer storage count including biases and classifier."""
    if not genotype.genes or len(genotype.genes) > 2:
        raise ValueError("One or two declared compressed scopes are required")
    scopes = [gene.aliases for gene in genotype.genes]
    if len(set(scopes)) != len(scopes) or any(scope not in (("0",), ("2",)) for scope in scopes):
        raise ValueError("Only distinct scopes 0 and 2 may be compressed")
    if any(gene.choice.rank not in RANKS for gene in genotype.genes):
        raise ValueError("A frozen rank is required")
    # Both 64x128 and 128x64 replace 8192 weight values by 192*r.
    return 4 * (17_226 + sum(192 * gene.choice.rank - 8192 for gene in genotype.genes))


def protocol_metadata():
    return {"dataset": "sklearn.load_digits", "split_seeds": list(SPLIT_SEEDS),
        "initialization_seed": INITIALIZATION_SEED, "train_shuffle_seed": TRAIN_SHUFFLE_SEED,
        "classes": 10, "dense_tensor_bytes": DENSE_TENSOR_BYTES,
        "tensor_byte_budget": TENSOR_BYTE_BUDGET, "selection_byte_caps": list(SELECTION_BYTE_BUDGETS),
        "common_ready_bytes": list(COMMON_READY_BYTES), "catalog_sizes": [15, 440],
        "catalog_ids": {catalog.name: list(catalog_by_id(catalog)) for catalog in (READY_CATALOG, TUNED_CATALOG)},
        "initial_ids": [canonical_solution_id(value) for value in INITIAL_GENOTYPES],
        "max_mlp_attempts": ATTEMPT_LIMIT, "updates_per_model": UPDATES_PER_MODEL,
        "max_updates": UPDATE_LIMIT, "quality_normalization": "CE/ln(10), bytes/68904",
        "quality_rule": "CE<=M0+.05 and accuracy>=M0-.02",
        "evolution": {"engine": "FedcoreEvoOptimizer", "selection": "SPEA2", "population_size": 4,
            "generations": 64, "mutation_only": False, "mutation_prob": 1., "crossover_prob": 1.,
            "max_num_of_operator_attempts": 4, "early_stopping_iterations": 64},
        "training": {"epochs": 40, "batch_size": 64, "drop_last": False, "optimizer": "Adam",
            "lr": .001, "weight_decay": 0., "betas": [.9, .999], "eps": 1e-8, "amsgrad": False}}
