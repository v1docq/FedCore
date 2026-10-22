"""Pure serial schedule and independent cost boundaries for issue 99 local 3."""
from __future__ import annotations

SPLITS = (19901, 19911, 19921)
SEARCH_SEEDS = (11, 22, 33)
MLP_LIMIT = 2700
MATRIX_LIMIT = 4350
PLANNED_MLP = 2676
PLANNED_MATRIX = 4301
PLANNED_MATRIX_SVD = 3535
STAGE_SECONDS = dict(prepare=300, matrix=2520, search=1500, statistics=1080, finalize=600)
RESOURCE_MAX_GAP = 2.0


def metadata_tapes():
    """Freeze tagged choices before loading sources; import only pure contracts.

    The legacy experiment catalog's import chain loads Torch. This independent
    metadata construction is checked against the numerical worker's canonical
    catalog, never against measured quality or a saved oracle table.
    """
    import hashlib
    import json
    import random
    from itertools import product
    from fedcore.algorithm.low_rank.compression_grammar import (
        ASVDChoice, AffinePCAChoice, CompressionGenotype, ScopeGene, WeightedChoice, genotype_payload)
    from fedcore.algorithm.low_rank.method_specs import ASVD

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    def pair(left, right):
        return CompressionGenotype((ScopeGene(("0",), left), ScopeGene(("2",), right)))
    def identity(value):
        return digest(genotype_payload(value))
    alternatives = tuple(choice for rank in (4, 8, 16, 24, 32) for choice in (
        WeightedChoice(rank), ASVDChoice(rank, ASVD(alpha=.25)),
        ASVDChoice(rank, ASVD(alpha=.5)), AffinePCAChoice(rank)))
    catalog = tuple(CompressionGenotype((ScopeGene((path,), choice),))
                    for path in ("0", "2") for choice in alternatives)
    catalog += tuple(pair(left, right) for left, right in product(alternatives, repeat=2))
    weighted, asvd, afm = WeightedChoice(8), ASVDChoice(8, ASVD(alpha=.5)), AffinePCAChoice(8)
    initial = (CompressionGenotype((ScopeGene(("0",), weighted),)),
               CompressionGenotype((ScopeGene(("2",), weighted),)), pair(asvd, afm), pair(afm, asvd))
    initial_ids = [identity(value) for value in initial]
    remaining = [value for value in catalog if identity(value) not in initial_ids]
    random.Random(11).shuffle(remaining)
    statistics_ids = [identity(value) for value in (*initial, *remaining)[:24]]
    admission = (pair(WeightedChoice(4), WeightedChoice(4)), pair(WeightedChoice(32), WeightedChoice(32)),
        pair(ASVDChoice(8, ASVD(alpha=.25)), ASVDChoice(8, ASVD(alpha=.25))),
        pair(ASVDChoice(32, ASVD(alpha=.5)), ASVDChoice(32, ASVD(alpha=.5))),
        pair(AffinePCAChoice(4), AffinePCAChoice(4)), pair(WeightedChoice(8), AffinePCAChoice(32)))
    return dict(catalog_ids=[identity(value) for value in catalog], initial_ids=initial_ids,
                statistics_ids=statistics_ids, statistics_sha256=digest(statistics_ids),
                admission_ids=[identity(value) for value in admission], rng_seed=990011)


def schedule():
    jobs = [dict(job=f"statistics-admission-{seed}", kind="statistics_admission",
                 stage="prepare", split_seed=seed, hard_seconds=90) for seed in SPLITS]
    jobs.append(dict(job="matrix-admission", kind="matrix_admission", stage="matrix",
                     session_id="matrix-admission", sizes=[32, 512, 1024], hard_seconds=180))
    jobs.extend(dict(job=session, kind="matrix", stage="matrix", session_id=session,
                     n=n, hard_seconds=seconds)
                for session, n, seconds in (("M32", 32, 60), ("M1-512", 512, 300),
                    ("M1-1024", 1024, 780), ("M2-1024", 1024, 780), ("M2-512", 512, 300)))
    pair = 0
    for split in SPLITS:
        for seed in SEARCH_SEEDS:
            order = "TE" if pair % 2 == 0 else "ET"
            pair += 1
            jobs.extend(dict(job=f"search-{split}-{seed}-{mode}", kind="search_diagnostic",
                             stage="search", split_seed=split, search_seed=seed, mode=mode,
                             hard_seconds=75, budget_seconds=60, max_evaluations=64,
                             max_proposals=512) for mode in order)
    jobs.extend(dict(job=f"statistics-{seed}", kind="statistics_streams", stage="statistics",
                     split_seed=seed, hard_seconds=360) for seed in SPLITS)
    return tuple(jobs)


def resource_coverage(samples, started_at, completed_at, *, maximum_gap=RESOURCE_MAX_GAP):
    """Observed coverage law; missing RSS is never a zero-memory observation."""
    times = [row["timestamp"] for row in samples]
    gaps = ([times[0]-started_at, completed_at-times[-1]] +
            [right-left for left, right in zip(times, times[1:])]) if times else []
    known = bool(samples) and all(type(row.get("rss_bytes")) is int and row["rss_bytes"] >= 0
                                  and row.get("monitor_error") is None for row in samples)
    ordered = all(value >= 0 for value in gaps)
    maximum = max(gaps) if gaps else None
    return dict(complete=known and ordered and maximum <= maximum_gap,
                sample_count=len(samples), max_gap_seconds=maximum,
                allowed_max_gap_seconds=maximum_gap,
                peak_rss_bytes=max((row["rss_bytes"] for row in samples
                                   if type(row.get("rss_bytes")) is int), default=None),
                first_sample_at=times[0] if times else None,
                last_sample_at=times[-1] if times else None)
