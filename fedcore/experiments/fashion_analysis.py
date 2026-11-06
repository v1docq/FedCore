"""Descriptive analysis conditional on one trained M0 and one fixed dataset."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .quantization_pilot_protocol import file_hash
from .quantization_pilot_guard import atomic_json


def paired_correctness(first, second, *, seed=20261030, resamples=2000):
    """Paired test-example analysis; first=Random, second=Evo.

    Resampling examples does not replicate training or search randomness.
    The McNemar value is exploratory and carries no multiplicity correction.
    """
    if not first or len(first) != len(second) or any(type(x) is not bool for x in list(first)+list(second)):
        raise ValueError("Identical nonempty ordered boolean correctness arrays required")
    both = sum(a and b for a, b in zip(first, second))
    random_only = sum(a and not b for a, b in zip(first, second))
    evo_only = sum(b and not a for a, b in zip(first, second))
    neither = len(first)-both-random_only-evo_only
    discordant = random_only+evo_only
    if discordant:
        tail = min(random_only, evo_only)
        logs = [math.lgamma(discordant+1)-math.lgamma(k+1)-math.lgamma(discordant-k+1)-discordant*math.log(2)
                for k in range(tail+1)]
        maximum = max(logs)
        p = min(1., 2*math.exp(maximum)*sum(math.exp(v-maximum) for v in logs))
    else:
        p = 1.
    result = {"examples": len(first), "both_correct": both, "random_only_correct": random_only,
        "evo_only_correct": evo_only, "both_wrong": neither,
        "accuracy_difference_evo_minus_random": (evo_only-random_only)/len(first),
        "mcnemar_exact_two_sided_p_exploratory": p,
        "interpretation": "conditional paired examples for one M0; no claim of search superiority or independent training replication"}
    if resamples:
        import numpy as np
        draws = np.random.default_rng(seed).multinomial(len(first),
            [both/len(first), random_only/len(first), evo_only/len(first), neither/len(first)], size=resamples)
        deltas = (draws[:, 2]-draws[:, 1])/len(first)
        result["paired_example_bootstrap"] = {"seed": seed, "resamples": resamples,
            "percentile_95_interval": np.quantile(deltas, [.025, .975]).tolist(),
            "scope": "test-example uncertainty conditional on the fixed trained M0 and selected artifacts"}
    return result


def analyze(plan, root):
    root = Path(root)
    prepared = json.loads((root/"prepared.json").read_text())
    trained = json.loads((root/"trained.json").read_text())
    closed_path = root/"closed_test.json"
    closed = json.loads(closed_path.read_text()) if closed_path.exists() else {"status": "unopened", "jobs": {}}
    rows = []
    for job in plan.payload["jobs"]:
        path = root/"jobs"/(job+".json")
        raw = json.loads(path.read_text()) if path.exists() else {"status": "missing"}
        artifact = raw.get("artifact", {})
        search = raw.get("search", {})
        test = closed["jobs"].get(job, {})
        rows.append({"job": job, "status": raw["status"], "artifact_sha256": artifact.get("sha256"),
            "physical_tensor_bytes": artifact.get("physical_tensor_bytes"), "within_budget": raw.get("within_budget"),
            "search_status": search.get("status"), "charged_evaluations": search.get("charged_evaluations"),
            "created_proposals": search.get("created_proposals"), "finite_archive_count": len(search.get("finite_archive", [])),
            "failure_count": search.get("failure_count"), "search_reason": search.get("reason"),
            "search_stop_reason": search.get("stop_reason"), "pick_candidates": raw.get("pick_candidates", []),
            "test": test, "wall_seconds": raw.get("wall_seconds"), "resources": raw.get("resources")})
    pairs = {}
    for seed in (17, 29, 43):
        random_result = closed["jobs"].get(f"random_{seed}", {})
        evo_result = closed["jobs"].get(f"evo_{seed}", {})
        if "correct" in random_result and "correct" in evo_result:
            pairs[str(seed)] = paired_correctness(random_result["correct"], evo_result["correct"], seed=20261030+seed)
        else:
            pairs[str(seed)] = {"status": "unavailable", "reason": "both full closed-test artifacts required"}
    result = {"schema": "pr166_fashion_analysis_v1", "manifest_sha256": plan.fingerprint,
        "prepared_sha256": file_hash(root/"prepared.json"), "trained_sha256": file_hash(root/"trained.json"),
        "closed_test_sha256": file_hash(closed_path) if closed_path.exists() else None,
        "m0": {"source": trained.get("source"), "status": trained["status"], "selected_epoch": trained.get("selected_epoch"),
               "stop": trained.get("stop"), "epochs_completed": trained.get("epochs_completed")},
        "budget_bytes": trained.get("budget_bytes"), "closed_test_status": closed["status"],
        "roles": {name: {key: value for key, value in details.items() if key != "ids"} for name, details in prepared["roles"].items()},
        "jobs": rows, "paired_search_seeds": pairs,
        "limitations": ["One trained M0, three paired search seeds: descriptive conditional comparisons only.",
            "Fashion-MNIST is a public historical benchmark. A bounded local prior-use audit cannot establish global test-set pristineness.",
            "Checkpoint storage counts unique physical tensor bytes; this is distinct from file bytes and guardian process-family RSS.",
            "Reference decode and Python kernels do not imply hardware inference speedups.",
            "Failed or deadline-limited engines remain failed/partial even if a finite artifact was preserved."]}
    atomic_json(root/"analysis.json", result)
    lines = ["# Fashion-MNIST: trained M0 and paired Evo/Random", "",
        f"M0 status: **{trained['status']}**; selected epoch {trained.get('selected_epoch')}; completed epochs {trained.get('epochs_completed')}.",
        f"Frozen tensor-storage budget: {trained.get('budget_bytes')} bytes. Closed test: {closed['status']}.", "",
        "The objective is cross-entropy against official labels. M0 agreement is reported separately.", "",
        "| Job | Engine/job status | Charged K | Storage bytes | Test rows | Accuracy | CE | M0 agreement |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(value): return "—" if value is None else f"{value:.6f}" if type(value) is float else str(value)
    for row in rows:
        test = row["test"]
        lines.append("| " + " | ".join(fmt(x) for x in (row["job"], row["status"], row["charged_evaluations"],
            row["physical_tensor_bytes"], test.get("examples"), test.get("accuracy"), test.get("cross_entropy"), test.get("m0_agreement"))) + " |")
    lines.extend(["", "Each paired seed uses the same initial four proposals and the same finite grammar; Random uses 24 unique initial draws in the common engine.",
        "Top four feasible finite archive artifacts are shortlisted on Vsearch; each job selects its artifact on the common Vpick before test opening.", ""])
    for seed, pair in pairs.items():
        if "accuracy_difference_evo_minus_random" in pair:
            lines.append(f"Seed {seed}: Evo − Random accuracy = {pair['accuracy_difference_evo_minus_random']:.6f}; "
                f"discordant correct examples Random-only={pair['random_only_correct']}, Evo-only={pair['evo_only_correct']}. "
                f"Conditional paired-example bootstrap 95% interval: {pair['paired_example_bootstrap']['percentile_95_interval']}.")
    lines.extend(["", *result["limitations"], "", "Full confusion matrices, predictions, resource evidence, failed proposals and traces are retained in the JSON records."])
    (root/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result
