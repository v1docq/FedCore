"""Frozen metadata for the first local issue-99 compression experiment.

No model, data, measured fitness or mutable RNG state lives in these catalogs.
The ready menu is a strict subset of the tuned menu. Random controls traverse a
uniform permutation after the four common initial solutions, rather than using
the grammar's scope-count-biased genotype generator.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from itertools import product
import json
import random

from fedcore.algorithm.low_rank.compression_composition import (
    AddResidual, CompositionRecipe, CompositionRequest, LowRank,
    MaterializeDense, PruneMask, QuantizeRemapped)
from fedcore.algorithm.low_rank.compression_grammar import (
    ASVDChoice, AffinePCAChoice, CompressionChoice, CompressionGenotype,
    ScopeGene, WeightedChoice, genotype_payload)
from fedcore.algorithm.low_rank.method_specs import ASVD
from fedcore.algorithm.low_rank.plans import ValidationFailure


MODEL_SEEDS = (41, 42, 43)
SEARCH_SEEDS = (11, 22, 33)
ADMISSIBILITY_SEED = 9900
ADMISSIBILITY_STREAM_SEEDS = (9911, 9912, 9913, 9914, 9915, 9916)
CACHE_STREAM_SEEDS = (9921, 9922, 9923)
SCOPE_PATHS = ("0", "2")
DENSE_TENSOR_BYTES = 49_416
G3_TENSOR_BYTE_BUDGET = 46_945
SELECTION_BYTE_BUDGETS = (14_824, 29_649, 42_003)
COMMON_READY_BYTES = (12_488, 22_792, 39_112)
MAX_WORKSPACE_BYTES = 64 * 1024**2
MAX_PEAK_BYTES = 8 * 1024**3
HYPERVOLUME_REFERENCE = (2.0, 1.05)


@dataclass(frozen=True)
class SearchBudget:
    name: str
    max_evaluations: int
    max_proposals: int
    max_seconds: float
    hard_seconds: float


ATTEMPT_BUDGET = SearchBudget("attempts", 12, 96, 60.0, 75.0)
TIME_BUDGET = SearchBudget("time", 80, 256, 30.0, 45.0)
SEARCH_BUDGETS = (ATTEMPT_BUDGET, TIME_BUDGET)


def canonical_solution_id(genotype: CompressionGenotype) -> str:
    """Identify executable choices, independently of a menu/source namespace."""
    if type(genotype) is not CompressionGenotype:
        raise ValueError("A metadata-only CompressionGenotype is required")
    ordered = replace(genotype, genes=tuple(sorted(genotype.genes, key=lambda item: item.aliases)))
    payload = genotype_payload(ordered)
    if isinstance(payload, ValidationFailure):
        raise ValueError("A valid metadata-only CompressionGenotype is required")
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False,
        separators=(",", ":")).encode()).hexdigest()


def _initial_genotypes() -> tuple[CompressionGenotype, ...]:
    weighted, asvd, afm = WeightedChoice(8), ASVDChoice(8, ASVD()), AffinePCAChoice(8)
    return (
        CompressionGenotype((ScopeGene(("0",), weighted),)),
        CompressionGenotype((ScopeGene(("2",), weighted),)),
        CompressionGenotype((ScopeGene(("0",), asvd), ScopeGene(("2",), afm))),
        CompressionGenotype((ScopeGene(("0",), afm), ScopeGene(("2",), asvd))),
    )


INITIAL_GENOTYPES = _initial_genotypes()


@dataclass(frozen=True)
class SearchCatalog:
    name: str
    alternatives: tuple[CompressionChoice, ...]
    genotypes: tuple[CompressionGenotype, ...]
    initial_genotypes: tuple[CompressionGenotype, ...] = INITIAL_GENOTYPES


def _catalog(name, alternatives):
    singles = tuple(CompressionGenotype((ScopeGene((path,), choice),))
        for path in SCOPE_PATHS for choice in alternatives)
    pairs = tuple(CompressionGenotype((ScopeGene(("0",), left), ScopeGene(("2",), right)))
        for left, right in product(alternatives, repeat=2))
    return SearchCatalog(name, alternatives, singles + pairs)


READY_CATALOG = _catalog("ready", (WeightedChoice(8), ASVDChoice(8, ASVD()), AffinePCAChoice(8)))
TUNED_CATALOG = _catalog("tuned", tuple(choice for rank in (8, 16) for choice in (
    WeightedChoice(rank), ASVDChoice(rank, ASVD(alpha=0.25)),
    ASVDChoice(rank, ASVD(alpha=0.5)), AffinePCAChoice(rank))))


def random_search_stream(catalog: SearchCatalog, seed: int) -> tuple[CompressionGenotype, ...]:
    """Return a deterministic complete no-repeat stream with the common prefix."""
    if type(catalog) is not SearchCatalog or type(seed) is not int or seed < 0:
        raise ValueError("A frozen catalog and nonnegative integer seed are required")
    prefix_ids = {canonical_solution_id(item) for item in catalog.initial_genotypes}
    remaining = [item for item in catalog.genotypes if canonical_solution_id(item) not in prefix_ids]
    random.Random(seed).shuffle(remaining)
    return catalog.initial_genotypes + tuple(remaining)


def catalog_by_id(catalog: SearchCatalog = TUNED_CATALOG) -> dict[str, CompressionGenotype]:
    """Return an owned lookup; mutating it cannot alter the frozen catalog."""
    return {canonical_solution_id(item): item for item in catalog.genotypes}


def physical_mode_order(seed: int) -> tuple[str, ...]:
    try:
        return {11: ("R", "T", "E"), 22: ("T", "E", "R"), 33: ("E", "R", "T")}[seed]
    except KeyError as error:
        raise ValueError("Use a frozen search seed: 11, 22 or 33") from error


@dataclass(frozen=True)
class CompositionCase:
    case_id: str
    word: str
    request: CompositionRequest
    schema_negative: bool = False
    expected_code: str | None = None


def _request(steps, *, path="0"):
    return CompositionRequest((CompositionRecipe(path, tuple(steps)),), max_peak_bytes=MAX_PEAK_BYTES)


_G5_ALPHABET = (
    ("P", PruneMask((0,))), ("L", LowRank(2)),
    ("Q", QuantizeRemapped("float64", "float64", "float64")),
    ("D", MaterializeDense()), ("R", AddResidual(1)),
)
G5_CASES = tuple(CompositionCase("word-" + "".join(letter for letter, _ in letters),
    "".join(letter for letter, _ in letters), _request(step for _, step in letters))
    for length in (1, 2, 3) for letters in product(_G5_ALPHABET, repeat=length))
G5_NEGATIVE_CASES = (
    CompositionCase("rank-zero", "L0", _request((LowRank(0),)), True, "InvalidRank"),
    CompositionCase("rank-too-large", "L9", _request((LowRank(9),)), True, "InvalidRank"),
    CompositionCase("duplicate-mask", "P00", _request((PruneMask((0, 0)),)), True, "InvalidMask"),
    CompositionCase("out-of-range-mask", "P64", _request((PruneMask((64,)),)), True, "InvalidMask"),
    CompositionCase("compute-dtype", "Q32", _request((QuantizeRemapped("float64", "float64", "float32"),)), True, "UnsupportedCapability"),
    CompositionCase("unknown-scope", "L", _request((LowRank(2),), path="missing"), True, "InvalidScope"),
    CompositionCase("overlapping-scopes", "LL", CompositionRequest((
        CompositionRecipe("0", (LowRank(2),)), CompositionRecipe("0", (LowRank(2),))),
        max_peak_bytes=MAX_PEAK_BYTES), True, "OverlappingScopes"),
    CompositionCase("onnx", "L", replace(_request((LowRank(2),)), export_format="onnx"), True, "UnsupportedExport"),
)


def admissibility_stream(seed: int) -> tuple[CompositionCase, ...]:
    if seed not in ADMISSIBILITY_STREAM_SEEDS:
        raise ValueError("Use one of the six frozen admissibility stream seeds")
    cases = list(G5_CASES + G5_NEGATIVE_CASES)
    random.Random(seed).shuffle(cases)
    return tuple(cases)


def admissibility_branch_order(seed: int) -> tuple[str, str]:
    if seed not in ADMISSIBILITY_STREAM_SEEDS:
        raise ValueError("Use one of the six frozen admissibility stream seeds")
    return (("A", "B"), ("B", "A"), ("B", "A"),
            ("A", "B"), ("A", "B"), ("B", "A"))[ADMISSIBILITY_STREAM_SEEDS.index(seed)]


def cache_replay_stream(seed: int) -> tuple[CompressionGenotype, ...]:
    if seed not in CACHE_STREAM_SEEDS:
        raise ValueError("Use one of the three frozen cache stream seeds")
    rng = random.Random(seed)
    stream = rng.sample(TUNED_CATALOG.genotypes, 12) * 4
    rng.shuffle(stream)
    return tuple(stream)


def cache_branch_order(model_seed: int, stream_seed: int) -> tuple[str, str]:
    if model_seed not in MODEL_SEEDS or stream_seed not in CACHE_STREAM_SEEDS:
        raise ValueError("Use frozen model and cache stream seeds")
    index = MODEL_SEEDS.index(model_seed) * len(CACHE_STREAM_SEEDS) + CACHE_STREAM_SEEDS.index(stream_seed)
    return ("A", "B") if index % 2 == 0 else ("B", "A")
