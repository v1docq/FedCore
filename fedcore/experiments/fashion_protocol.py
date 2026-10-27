"""Finite, predeclared Fashion-MNIST protocol; no numerical dependencies."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random

from .quantization_pilot_protocol import canonical_bytes, digest, file_hash, HARD_RSS

PATHS = ("0", "2", "4")
ROLES = {"fit": 40000, "stop": 5000, "calibration": 5000, "search": 2000, "pick": 8000}
RAW_FILES = ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
MD5 = dict(zip(RAW_FILES, ("8d4fb7e6c68d591d4c3dfef9ec88bf0d", "25c81989df183df01b3e8a0aad5dffbe",
                          "bef4ecab320f06d8554ea6380940ec79", "bb300cfdad3c16e7a12a480ee83cd310")))
CONTROLS = ("dense32", "svd", "ptq4", "static_int8", "manual_hetero")
SEARCH_JOBS = tuple(f"{method}_{seed}" for seed in (17, 29, 43) for method in ("random", "evo"))


def default_manifest(source_commit, raw_sha256):
    return {
        "schema": "pr166_fashion_v1", "source_commit": source_commit, "raw_sha256": raw_sha256,
        "model": {"dimensions": [784, 256, 128, 10], "activation": "ReLU", "seed": 20261027},
        "data": {"partition": dict(ROLES), "partition_seed": 20261027, "calibration_rows": 256,
                 "search_rows": 512, "pick_rows": 8000, "test_rows": 10000,
                 "preprocessing": "uint8 / 255, flatten 784; no augmentation", "labels": "official ground truth"},
        "training": {"epochs": 20, "optimizer": "Adam", "lr": .001, "batch_size": 256,
                     "threads": 8, "max_seconds": 1200, "minimum_stop_accuracy": .85,
                     "checkpoint_rule": "minimum stop CE; earlier epoch wins ties"},
        "budget_fraction": .5, "max_peak_bytes": HARD_RSS, "threads": 8,
        "search": {"evaluations": 24, "evo_population": 4, "generations": 24, "max_proposals": 128,
                   "prequantize_bits": [2, 4], "training_choices": [None], "cache_bytes": 64*1024**2,
                   "ranks": [[32, 64, 96], [16, 32, 48], [2, 4, 6]], "pick_candidates": 4,
                   "selection_rule": "top four feasible finite archive by search CE, bytes, canonical ID; pick minimum pick CE, bytes, canonical ID",
                   "tender_paths": ["4"], "factor_bits": 4, "lplr_comparison_rank": "same as factor rank",
                   "caldera": {"outer_iterations": 1, "inner_iterations": 1, "ridge": .0001},
                   "seed_rule": "execution seed = first 15 hexadecimal SHA256 digits of canonical complete genotype"},
        "job_seconds": 600, "search_seconds": 600, "pick_seconds": 300,
        "jobs": list(CONTROLS + SEARCH_JOBS),
        "test_policy": "single exclusive opening after all jobs and selection lock; full 10000 for every eligible artifact",
        "test_batch_size": 128, "test_model_seconds": 300, "preflight_safety_factor": 2,
        "inference_policy": "preflight on C before test; reject predicted full-pass time above cap; never reduce test rows",
        "inference": "one trained M0; three paired search seeds are descriptive conditional comparisons, not independent M0 replications",
        "statistics": {"paired_example_bootstrap_resamples": 2000, "bootstrap_seed_base": 20261030,
                       "mcnemar": "exact two sided exploratory, no multiplicity correction"},
    }


@dataclass(frozen=True)
class FashionPlan:
    payload: dict
    fingerprint: str

    @classmethod
    def parse(cls, raw):
        if type(raw) is not dict or raw.get("schema") != "pr166_fashion_v1":
            raise ValueError("Unknown Fashion protocol")
        commit = raw.get("source_commit")
        if type(commit) is not str or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise ValueError("A full lowercase commit identity is required")
        hashes = raw.get("raw_sha256")
        if (type(hashes) is not dict or set(hashes) != set(RAW_FILES)
                or any(type(h) is not str or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes.values())):
            raise ValueError("Four canonical raw SHA256 identities are required")
        reference = default_manifest(commit, hashes)
        if set(raw) != set(reference):
            raise ValueError("Unknown or missing manifest fields")
        for key in reference:
            if key not in ("job_seconds", "search_seconds") and canonical_bytes(raw[key]) != canonical_bytes(reference[key]):
                raise ValueError("Protocol field differs from approved finite Fashion plan: " + key)
        for key in ("job_seconds", "search_seconds"):
            value = raw[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not 1 <= value <= reference[key]:
                raise ValueError("Caps may only narrow the approved maximum")
        normalized = json.loads(canonical_bytes(raw))
        return cls(normalized, digest(normalized))

    @classmethod
    def read(cls, path):
        return cls.parse(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def stratified_partition(labels, roles=ROLES, *, seed=20261027):
    """Exhaustive disjoint partition with exact equal per-class role quotas."""
    labels = [int(label) for label in labels]
    if not labels or set(labels) != set(range(10)) or sum(roles.values()) != len(labels):
        raise ValueError("Ten classes and exhaustive role sizes are required")
    if any(type(n) is not int or n <= 0 or n % 10 for n in roles.values()):
        raise ValueError("Role sizes must be positive multiples of ten")
    by_class = {c: [i for i, label in enumerate(labels) if label == c] for c in range(10)}
    if any(len(ids) != len(labels)//10 for ids in by_class.values()):
        raise ValueError("Balanced source classes are required")
    result = {role: [] for role in roles}
    for c, ids in by_class.items():
        random.Random(seed+c).shuffle(ids)
        start = 0
        for role, size in roles.items():
            end = start + size//10
            result[role].extend(ids[start:end]); start = end
    for role, ids in result.items():
        random.Random(seed + 100 + list(roles).index(role)).shuffle(ids)
    return result


def stratified_subset(ids, labels, rows, *, seed):
    if type(rows) is not int or rows < 10 or rows > len(ids):
        raise ValueError("A bounded stratified subset is required")
    result = []
    for c in range(10):
        pool = [int(i) for i in ids if int(labels[i]) == c]
        random.Random(seed+c).shuffle(pool)
        count = rows//10 + (c < rows%10)
        if len(pool) < count:
            raise ValueError("Insufficient class quota")
        result.extend(pool[:count])
    random.Random(seed+100).shuffle(result)
    return result


def independent_initial(space, count, seed, *, sample, payload):
    """One independently seeded proposal stream per search replicate.

    A run's proposal seeds are random 63-bit values, rather than consecutive
    integers whose shifted windows overlap across nearby search seeds.
    """
    if type(seed) is not int or not 0 <= seed < 2**63 or type(count) is not int or not 1 <= count <= 128:
        raise ValueError("Bounded search seed and initial population required")
    rng = random.Random(seed)
    result, seen = [], set()
    for _ in range(4096):
        item = sample(space, seed=rng.getrandbits(63))
        identity = digest(payload(item))
        if identity not in seen:
            seen.add(identity); result.append(item)
        if len(result) == count:
            return tuple(result)
    raise ValueError("Finite space could not supply unique initial proposals")


def classify_search_artifact(search, artifact):
    if artifact is None:
        return "failed"
    return "succeeded" if search.get("status") == "succeeded" else "partial_artifact"


def select_archive_candidates(attempts, count=4):
    """Unique feasible artifacts; deterministic public selection rule."""
    unique = {}
    for item in attempts:
        if item.get("status") == "succeeded" and item.get("finite_archive_member"):
            unique[item["canonical_solution_id"]] = item
    return sorted(unique.values(), key=lambda x: (x["search"]["cross_entropy"],
        x["artifact"]["physical_tensor_bytes"], x["canonical_solution_id"]))[:count]


def source_code_hashes():
    base = Path(__file__).resolve().parents[1]
    paths = ["experiments/fashion_protocol.py", "experiments/fashion_worker.py", "experiments/fashion_analysis.py",
             "experiments/quantization_pilot_worker.py", "experiments/quantization_pilot_guard.py",
             "algorithm/quantization/grammar.py", "algorithm/quantization/graph_execution.py",
             "algorithm/quantization/graph_specs.py", "algorithm/quantization/resources.py",
             "algorithm/low_rank/grammar_search.py", "tools/registry/checkpoint_manager.py",
             "interfaces/fedcore_dispatcher.py", "experiments/fashion_independent_worker.py"]
    paths.extend(path.relative_to(base).as_posix() for path in (base/"algorithm"/"quantization").rglob("*.py"))
    return {path: file_hash(base/path) for path in sorted(set(paths))}
