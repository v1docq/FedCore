"""Frozen metadata and safe JSON boundary for the PR166 local CPU pilot."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

HARD_RSS = 8 * 1024**3
PATHS = ("0", "2", "4")
BASE_JOBS = ("dense32", "dense64", "static_int8", "ptq4", "svd", "dsvd4", "lplr4", "lplr_svd4",
             "caldera4", "tender_w4a8", "manual_hetero", "exact_catalogue", "random_17", "evo_17", "random_29", "evo_29")
ROLES = {"training": (4096, 701), "calibration": (256, 702), "selection": (256, 703), "closed_test": (2048, 704)}


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def default_manifest(source_commit, *, include_recovery=False):
    return {
        "schema": "pr166_pilot_v1", "source_commit": source_commit,
        "model": {"dimensions": [64, 128, 64, 10], "activation": "ReLU", "seed": 20261027,
                  "regime": "fast_spectrum", "singular_decay": 6., "slow_regime": "diagnostic_only_unrun"},
        "data": {role: {"rows": rows, "seed": seed, "id": "pr166-synthetic-" + role + "-v1"} for role, (rows, seed) in ROLES.items()},
        "labels": "argmax frozen M0; agreement is teacher fidelity, not trained task quality",
        "budget_fraction": .5, "max_peak_bytes": HARD_RSS, "threads": 1,
        "search": {"evaluations": 8, "evo_population": 4, "generations": 8, "max_proposals": 64,
                   "prequantize_bits": [2, 4], "training_choices": [None], "cache_bytes": 8 * 1024**2},
        "catalogue": {"max_options": 256, "max_cumulative_bytes": 128 * 1024**2,
                      "objective": "sum frozen local output SSE; not whole-model CE"},
        "job_seconds": 180, "search_seconds": 600,
        "timing": {"batches": [1, 128], "threads": [1, 8], "warmups": 20, "requests": 200,
                   "process_repeats": 5, "aa_ratio_min": .95, "aa_ratio_max": 1.05},
        "jobs": list(BASE_JOBS) + (["recovery_lplr", "recovery_factor_control"] if include_recovery else []),
        "recovery": {"steps": 4, "batch_size": 64, "criterion": "logits_mse", "seed": 919},
        "test_policy": "Open closed_test exactly once, after artifact selection lock; no adaptive protocol changes",
    }


@dataclass(frozen=True)
class PilotPlan:
    payload: dict
    fingerprint: str

    @classmethod
    def parse(cls, raw):
        if type(raw) is not dict or raw.get("schema") != "pr166_pilot_v1":
            raise ValueError("Unknown pilot manifest")
        reference = default_manifest(raw.get("source_commit"), include_recovery="recovery_lplr" in raw.get("jobs", []))
        if set(raw) != set(reference):
            raise ValueError("Unknown or missing manifest fields")
        if type(raw["source_commit"]) is not str or len(raw["source_commit"]) != 40 or any(c not in "0123456789abcdef" for c in raw["source_commit"]):
            raise ValueError("A full lowercase commit identity is required")
        # This finite pilot deliberately exposes no unreviewed arbitrary model,
        # dataset, optimizer, recipe or search-space parser.
        for key in reference:
            if key not in ("source_commit", "job_seconds", "search_seconds") and canonical_bytes(raw[key]) != canonical_bytes(reference[key]):
                raise ValueError("Protocol field differs from the approved finite pilot: " + key)
        for key in ("job_seconds", "search_seconds"):
            value = raw[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not 1 <= value <= reference[key]:
                raise ValueError("Job caps can only narrow the approved maximum")
        normalized = json.loads(canonical_bytes(raw))
        return cls(normalized, digest(normalized))

    @classmethod
    def read(cls, path):
        return cls.parse(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def quantile(values, probability):
    if not values or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Finite nonnegative observations required")
    ordered = sorted(values)
    at = (len(ordered) - 1) * probability
    low = math.floor(at); high = math.ceil(at)
    return ordered[low] + (ordered[high] - ordered[low]) * (at - low)


def timing_summary(values, *, target_requests=200):
    return {"observed_requests": len(values), "p50_seconds": quantile(values, .5),
            "p95_seconds": quantile(values, .95) if len(values) >= target_requests else None,
            "p95_eligible": len(values) >= target_requests, "request_seconds": list(values)}


def search_execution_summary(result):
    """Separate preserved finite artifacts from the engine's terminal state."""
    search = result.get("search")
    if search is None:
        return {"search_status": None, "search_reason": None, "search_completed": None,
                "search_charged_evaluations": None, "search_variation_candidates_assessed": None}
    varied = {event["proposal_id"] for event in search.get("trace", [])
              if event.get("event") == "proposal" and event.get("operator") != "initial"}
    return {"search_status": search.get("status"), "search_reason": search.get("reason"),
            "search_stop_reason": search.get("stop_reason"), "search_completed": search.get("status") == "succeeded",
            "search_charged_evaluations": search.get("charged_evaluations"),
            "search_created_proposals": search.get("created_proposals"),
            "search_variation_candidates_assessed": sum(event.get("event") == "evaluated" and event.get("proposal_id") in varied
                                                         for event in search.get("trace", []))}
