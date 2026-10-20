"""Read-only, standard-library evidence audit. Never imports FedCore or ML code.

Prepared while run-3 was active; execute only after the coordinator's signal.
The only optional write is this audit's own JSON report outside the run folder.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import random


MODELS = (41, 42, 43)
SEEDS = (11, 22, 33)
CAPS = (14824, 29649, 42003)
DENSE = 49416
REFERENCE = (2., 1.05)
NEGATIVES = {
    "rank-zero": "InvalidRank", "rank-too-large": "InvalidRank",
    "duplicate-mask": "InvalidMask", "out-of-range-mask": "InvalidMask",
    "compute-dtype": "UnsupportedCapability", "unknown-scope": "InvalidScope",
    "overlapping-scopes": "OverlappingScopes", "onnx": "UnsupportedExport",
}
WORDS = tuple("".join(word) for length in (1, 2, 3) for word in product("PLQDR", repeat=length))
CASES = tuple("word-"+word for word in WORDS) + tuple(NEGATIVES)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def lines(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()] if path.exists() else []


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
        separators=(",", ":")).encode()).hexdigest()


def near(a, b):
    return (isinstance(a, (int, float)) and isinstance(b, (int, float))
            and math.isfinite(a) and math.isfinite(b)
            and math.isclose(a, b, rel_tol=2e-12, abs_tol=2e-12))


class Audit:
    def __init__(self):
        self.failures, self.warnings, self.counts, self.facts = [], [], Counter(), {}

    def check(self, condition, category, detail):
        self.counts[category] += 1
        if not condition:
            self.failures.append({"category": category, "detail": detail})

    def warn(self, detail):
        self.warnings.append(detail)


def gene(path, recipe, rank, alpha=None):
    choice = {"rank": rank, "representation": "two_layers", "centering": "none", "bias_policy": "preserve"}
    if recipe == "weighted_svd":
        choice["policy"] = {"ridge": 0.0, "rcond": None, "nullspace_policy": "support_only"}
    elif recipe == "asvd":
        choice["spec"] = {"abs_mode": "abs_mean", "alpha": alpha, "epsilon": 0.0, "zero_policy": "support_only"}
    else:
        choice.update(centering="output_mean", bias_policy="affine_mean_correction")
    return {"aliases": [path], "choice": choice, "recipe": recipe, "recovery": "none"}


def genotype(*genes):
    return {"schema_version": 1, "genes": sorted(genes, key=lambda item: item["aliases"])}


def catalogs():
    def make(alternatives):
        values = [genotype(gene(path, *choice)) for path in ("0", "2") for choice in alternatives]
        values += [genotype(gene("0", *a), gene("2", *b)) for a, b in product(alternatives, repeat=2)]
        return {identity(item): item for item in values}
    ready = (("weighted_svd", 8, None), ("asvd", 8, .5), ("afm", 8, None))
    tuned = tuple(choice for rank in (8, 16) for choice in (
        ("weighted_svd", rank, None), ("asvd", rank, .25), ("asvd", rank, .5), ("afm", rank, None)))
    prefix = (genotype(gene("0", "weighted_svd", 8)), genotype(gene("2", "weighted_svd", 8)),
        genotype(gene("0", "asvd", 8, .5), gene("2", "afm", 8)),
        genotype(gene("0", "afm", 8), gene("2", "asvd", 8, .5)))
    return make(ready), make(tuned), tuple(identity(item) for item in prefix)


def byte_formula(value):
    result = DENSE
    for item in value["genes"]:
        n, m = {"0": (30, 128), "2": (128, 64)}[item["aliases"][0]]
        result += 4*(item["choice"]["rank"]*(n+m)-n*m)
    return result


def g5(word):
    allowed = {"dense": "PLQDR", "dense_masked": "PLQDR", "real_factors": "QDR",
               "packed_factors": "DR", "residual_branch": "D"}
    representation, rank, size, completed, decompositions = "dense", None, 576, 0, 0
    for symbol in word:
        if symbol not in allowed[representation]:
            return False, representation, size, completed, decompositions
        if symbol in "PD":
            representation, rank, size = ("dense_masked" if symbol == "P" else "dense"), None, 576
        elif symbol == "L":
            representation, rank, size = "real_factors", 2, 320
        elif symbol == "Q":
            rank = rank if representation == "real_factors" else 8
            representation, size = "packed_factors", 32*rank+64
        else:
            representation, rank, size = "residual_branch", None, size+128
        completed += 1
        decompositions += symbol in "LQR"
    return True, representation, size, completed, decompositions


def finite(rows, deadline=None):
    return [row for row in rows if row.get("status") == "finite"
        and (deadline is None or row["completed_at"] <= deadline)]


def points(rows):
    return [(row["quality"], row["cost"]) for row in rows]


def rectangle_union(values):
    """Independent vertical-cell union, distinct from the published HV sweep."""
    values = [(x, y) for x, y in values if x < REFERENCE[0] and y < REFERENCE[1]]
    coordinates = sorted({x for x, _ in values} | {REFERENCE[0]})
    result = 0.
    for left, right in zip(coordinates, coordinates[1:]):
        bottoms = [y for x, y in values if x <= left]
        if bottoms:
            result += (right-left)*(REFERENCE[1]-min(bottoms))
    return result


def describe(values):
    values = sorted(values)
    n = len(values)
    return {"count": n, "median": (values[(n-1)//2]+values[n//2])/2 if n else None,
        "minimum": values[0] if n else None, "maximum": values[-1] if n else None}


def diagnostic(rows):
    seen, unique = set(), []
    for row in sorted(rows, key=lambda row: (row["completed_at"], row["global_attempt"], row["solution_id"])):
        if row["solution_id"] not in seen:
            seen.add(row["solution_id"])
            unique.append(row)
    previous, flat = 0., 0
    for index in range(1, len(unique)+1):
        area = rectangle_union(points(unique[:index]))
        if index > 12:
            flat = 0 if area-previous > 1e-6 else flat+1
            if flat == 6:
                return "triggered", index
        previous = area
    return "not_triggered", None


def nondominated(rows):
    return {(row["solution_id"], row["quality"], row["cost"]) for row in rows if not any(
        other["quality"] <= row["quality"] and other["cost"] <= row["cost"]
        and (other["quality"] < row["quality"] or other["cost"] < row["cost"]) for other in rows)}


def selection(rows, cap):
    feasible = [row for row in rows if row.get("quality_accepted") and row["tensor_bytes"] <= cap]
    return min(feasible, key=lambda row: (row["quality"], row["tensor_bytes"], row["solution_id"])) if feasible else None


def expected_searches():
    return [f"search-{model}-{budget}-{seed}-{mode}" for model in MODELS for budget in ("attempts", "time")
        for seed in SEEDS for mode in {11: "RTE", 22: "TER", 33: "ERT"}[seed]]


def main(run, published=None):
    audit = Audit()
    run = Path(run).resolve()
    protocol, ledger = load(run / "protocol.json"), load(run / "resource-ledger.json")
    ready, tuned, initial = catalogs()
    attempts = lines(run / "mlp-attempts.jsonl")
    origin = ledger["started_at"]
    audit.check(ledger["status"] == "succeeded", "completeness", "Final package status")
    audit.check(len(ready) == 15 and len(tuned) == 80 and set(ready) < set(tuned), "catalog", "Independent15/80 nested catalogs")
    audit.check([row["attempt"] for row in attempts] == list(range(1, len(attempts)+1)), "budget", "Continuous cumulative attempt IDs")
    audit.check(len(attempts) == ledger["mlp_attempts"] <= 3300, "budget", "Ledger count equals actual lines and stays≤3300")
    audit.check(ledger["elapsed_seconds"] <= 7200.5, "budget", "Original package real-time≤7200s (sampling tolerance0.5s)")
    audit.check(all(row["started_at"] < origin+6900 for row in attempts), "budget", "No charged computation admitted after115minutes")
    audit.check(abs((ledger["completed_at"]-origin)-ledger["elapsed_seconds"]) <= .5, "budget", "Original origin retained in elapsed time (separate-clock-read tolerance)")
    frozen = dict(protocol)
    frozen.pop("protocol_sha256")
    audit.check(hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest() == protocol["protocol_sha256"],
        "provenance", "Protocol SHA-256")

    # Traverse original revisions exactly once; reused records keep their origin.
    revisions, resources, visited = [], {}, set()
    directory = run
    while directory not in visited:
        visited.add(directory)
        old_protocol, old_ledger = load(directory / "protocol.json"), load(directory / "resource-ledger.json")
        revisions.append({"directory": str(directory), "protocol": old_protocol["protocol_sha256"],
            "status": old_ledger["status"], "attempts": old_ledger.get("mlp_attempts")})
        audit.check(old_ledger["started_at"] == origin, "lineage", f"Original origin:{directory.name}")
        for record in old_ledger["jobs"]:
            resource_origin = Path(record.get("origin_execution", directory)).resolve()
            resources.setdefault((str(resource_origin), record["job"]), record)
        prior = old_protocol.get("resumed_admission")
        if not prior:
            break
        previous = Path(prior["directory"]).resolve()
        audit.check(prior["new_free_budget"] is False, "lineage", f"No free budget:{directory.name}")
        audit.check(sha(previous / "resource-ledger.json") == prior["ledger_sha256"], "lineage", f"Immutable prior ledger:{directory.name}")
        audit.check((directory / "mlp-attempts.jsonl").read_bytes().startswith((previous / "mlp-attempts.jsonl").read_bytes()),
            "lineage", f"Exact prior attempt prefix:{directory.name}")
        audit.check(load(previous / "protocol.json")["protocol_sha256"] == prior["protocol_sha256"], "lineage", f"Prior protocol binding:{directory.name}")
        directory = previous
    inherited = protocol["resumed_admission"]["prior_attempts"]
    audit.check(inherited == 560, "lineage", "All560 pre-replay charges retained")
    audit.facts.update(cumulative_attempts=len(attempts), inherited_attempts=inherited,
        new_attempts=len(attempts)-inherited, original_origin=origin, elapsed_seconds=ledger["elapsed_seconds"], revisions=revisions)

    expected_reuse = {"admissibility"} | {f"early-late-{seed}-{branch}" for seed in range(9911, 9917) for branch in "AB"} | {f"oracle-{model}" for model in MODELS}
    reused = load(run / "reused-evidence.json")
    audit.check(set(reused) == expected_reuse, "reuse", "Exactly16 unaffected witnesses")
    for name, record in reused.items():
        audit.check(record["new_attempts"] == 0 and record["evidence_reused"], "reuse", f"No new attempt:{name}")
        original = Path(record["origin_execution"]) / "jobs" / name
        copied = run / "jobs" / name
        for relative, digest in record["evidence_file_hashes"].items():
            audit.check(sha(original / relative) == sha(copied / relative) == digest, "reuse", f"Original/copied byte identity:{name}/{relative}")
        audit.check(load(copied / "job.json")["protocol_sha256"] == record["origin_protocol_sha256"], "reuse", f"Original protocol retained:{name}")

    jobs = {}
    for path in sorted((run / "jobs").iterdir()):
        if not path.is_dir():
            continue
        job, result = load(path / "job.json"), load(path / "result.json")
        rows = result.get("evaluations", [])
        jobs[path.name] = {"path": path, "job": job, "result": result, "rows": rows,
            "trace": lines(path / "search-trace.jsonl")}
        audit.check(result.get("status") == "succeeded", "completeness", f"Successful job:{path.name}")
        rng = result.get("rng_verification", {})
        audit.check(rng.get("unchanged") is True and rng.get("before") == rng.get("after"), "immutability", f"Restored global RNG:{path.name}")
        resource = load(path / "resources.json")
        audit.check(resource["resources"]["peak_tree_rss_bytes"] <= 8*1024**3, "resources", f"RSS≤8GiB:{path.name}")
        audit.check(resource["resources"]["minimum_available_ram_bytes"] >= 4*1024**3, "resources", f"RAMreserve≥4GiB:{path.name}")
        if path.name not in expected_reuse:
            audit.check(job["protocol_sha256"] == protocol["protocol_sha256"], "provenance", f"Fresh execution protocol:{path.name}")
        row_attempts = {row.get("global_attempt") for row in rows}
        charged = {row["attempt"] for row in attempts if row["job"] == path.name and row["purpose"] == "full_validation"
            and row["protocol_sha256"] == job["protocol_sha256"]}
        audit.check(row_attempts == charged, "budget", f"Every full evaluation charged exactly once:{path.name}")

    planned = {value["job"] for value in protocol["jobs"]} | {f"finalize-{model}" for model in MODELS}
    audit.check(set(jobs) == planned and len(jobs) == 97, "completeness", "94 planned jobs +3 finalizations")
    audit.check([row["job"] for row in ledger["jobs"]] == [row["job"] for row in protocol["jobs"]] + [f"finalize-{m}" for m in MODELS],
        "completeness", "Ledger respects frozen logical schedule")
    audit.facts["job_counts"] = dict(Counter(item["job"]["kind"] for item in jobs.values()))

    # Literal G5 model, independent of either production or experimental code.
    admission_rows = jobs["admissibility"]["result"]["rows"]
    audit.check({row["case_id"] for row in admission_rows} == set(CASES) and len(admission_rows) == 163, "g5", "Complete155+8 language")
    observed_accepted = 0
    for row in admission_rows:
        name = row["case_id"]
        expected = g5(name[5:])[0] if name.startswith("word-") else False
        audit.check(row["contract_accepted"] == row["production_accepted"] == row["execution_accepted"] == expected,
            "g5", f"Independent structural decision:{name}")
        if expected:
            _, representation, size, _, _ = g5(name[5:])
            observed_accepted += 1
            audit.check(row["tensor_bytes"] == size and row["final_states"][0][1]["representation"] == representation,
                "g5", f"Exact final representation/bytes:{name}")
        elif name in NEGATIVES:
            audit.check(row["reference_rejection"]["code"] == NEGATIVES[name], "g5", f"Typed negative:{name}")
    audit.check(observed_accepted == 72, "g5", "72accepted/83rejected words,8separate negatives")
    early_late_facts = []
    for index, seed in enumerate(range(9911, 9917)):
        a, b = (jobs[f"early-late-{seed}-{branch}"] for branch in "AB")
        expected_stream = list(CASES)
        random.Random(seed).shuffle(expected_stream)
        ar, br = a["result"]["rows"], b["result"]["rows"]
        audit.check([row["case_id"] for row in ar] == [row["case_id"] for row in br] == expected_stream,
            "early_late", f"Frozen identical163-request permutation:{seed}")
        expected_first = ("A", "B", "B", "A", "A", "B")[index]
        audit.check((a["job"]["worker_started_at"] < b["job"]["worker_started_at"]) == (expected_first == "A"),
            "early_late", f"Prespecified physical branch order:{seed}")
        for left, right in zip(ar, br):
            name = left["case_id"]
            expected = g5(name[5:])[0] if name.startswith("word-") else False
            audit.check(left["accepted"] == right["accepted"] == expected, "early_late", f"Same structural decision:{seed}/{name}")
            if expected:
                audit.check((left["output_identity"], left["tensor_bytes"], left["final_states"]) ==
                    (right["output_identity"], right["tensor_bytes"], right["final_states"]),
                    "early_late", f"Identical actual output:{seed}/{name}")
            for branch, row in (("A", left), ("B", right)):
                if row.get("stage_timings"):
                    stage = row["stage_timings"]
                    audit.check(near(row["execution_seconds"], stage["total_seconds"]), "early_late", f"No timing double count:{seed}/{branch}/{name}")
                    audit.check(sum(value for key, value in stage.items() if key != "total_seconds") <= stage["total_seconds"]+1e-7,
                        "early_late", f"Disjoint phase times:{seed}/{branch}/{name}")
                if name.startswith("word-"):
                    _, _, _, complete, decompositions = g5(name[5:])
                    audit.check(row["completed_steps"] == (complete if branch == "B" or expected else 0), "early_late", f"Reached prefix:{seed}/{branch}/{name}")
                    audit.check(row["decompositions_started"] == (decompositions if branch == "B" or expected else 0), "early_late", f"Started SVDs:{seed}/{branch}/{name}")
        early_late_facts.append({"seed": seed, "early_over_late": a["result"]["stream_seconds"]/b["result"]["stream_seconds"]})
    audit.facts["early_late"] = early_late_facts

    manifests = defaultdict(list)
    full_rows = []
    for name, item in jobs.items():
        manifest_path = item["path"] / "source-manifest.json"
        if manifest_path.exists():
            manifest = load(manifest_path)
            manifests[manifest["seed"]].append((name, manifest))
            audit.check(sha(manifest["source_path"]) == manifest["checkpoint_sha256"], "provenance", f"Historical weights hash:{name}")
        for row in item["rows"]:
            if row.get("status") != "finite":
                continue
            value = tuned[row["solution_id"]]
            audit.check(row["tensor_bytes"] == row["expected_tensor_bytes"] == byte_formula(value) <= 46945,
                "storage", f"Independent G3 bytes:{name}/{row['global_attempt']}")
            audit.check(row["inventory_verified_after_reload"] is True and near(row["cost"], row["tensor_bytes"]/DENSE)
                and near(row["quality"], row["metrics"]["cross_entropy"]/math.log(2)),
                "metrics", f"Recorded normalized metrics/reload:{name}/{row['global_attempt']}")
            if row.get("artifact_path"):
                audit.check(sha(row["artifact_path"]) == row["checkpoint_sha256"], "artifacts", f"Durable checkpoint:{name}/{row['global_attempt']}")
            full_rows.append(row)
    for model, values in manifests.items():
        reference_manifest = values[0][1]
        for name, manifest in values:
            audit.check(all(manifest[key] == reference_manifest[key] for key in
                ("normalized_identity", "checkpoint_sha256", "historical_data", "baseline_validation", "tensor_bytes")),
                "immutability", f"Same M0/data/roles/baseline:{model}/{name}")
            baseline = manifest["baseline_validation"]
            for row in finite(jobs[name]["rows"]):
                accepted = row["metrics"]["cross_entropy"] <= baseline["cross_entropy"]+.05 and row["metrics"]["accuracy"] >= baseline["accuracy"]-.02
                audit.check(row["quality_accepted"] == accepted, "selection", f"Frozen quality policy:{name}/{row['global_attempt']}")

    for model in MODELS:
        item = jobs[f"oracle-{model}"]
        catalog = load(item["path"] / "catalog.json")
        audit.check({row["solution_id"]: row["genotype"] for row in catalog} == tuned, "catalog", f"Independent exact80catalog:{model}")
        audit.check(len(item["rows"]) == 80 and {row["solution_id"] for row in finite(item["rows"])} == set(tuned),
            "catalog", f"Complete finite oracle:{model}")

    searches = expected_searches()
    actual_order = sorted(searches, key=lambda name: jobs[name]["job"]["worker_started_at"])
    audit.check(actual_order == searches, "search", "All54 fresh searches in frozen physical order")
    within, search_facts = {}, []
    for name in searches:
        item = jobs[name]
        job, result, rows, trace = item["job"], item["result"], item["rows"], item["trace"]
        budget, mode = job["budget"], job["mode"]
        cap_attempts, cap_proposals, seconds = (12, 96, 60) if budget == "attempts" else (80, 256, 30)
        catalog = ready if mode == "R" else tuned
        audit.check(all(row["global_attempt"] > inherited for row in rows), "search", f"No old R/T/E observations reused:{name}")
        proposal = [event for event in trace if event["event"] == "proposal"]
        audit.check(tuple(event["canonical_solution_id"] for event in proposal[:4]) == initial, "search", f"Common4 initial proposals:{name}")
        audit.check(all(event["canonical_solution_id"] in catalog for event in proposal), "search", f"Proposals inside frozen catalog:{name}")
        audit.check(all(row["solution_id"] in catalog for row in rows) and len(rows) <= cap_attempts
            and len(proposal) <= cap_proposals, "search", f"Catalog/attempt/proposal bounds:{name}")
        deadline = min(job["package_deadline"], job["worker_started_at"]+seconds)
        audit.check(near(result["deadline"], deadline), "deadlines", f"Parent absolute deadline:{name}")
        audit.check(all(row["started_at"] < deadline for row in rows), "deadlines", f"No new evaluation after deadline:{name}")
        audit.check(result["completed_at"] <= job["worker_started_at"]+job["hard_seconds"]+.5,
            "deadlines", f"Worker hard deadline:{name}")
        within[name] = finite(rows, deadline)
        search = result["search"]
        audit.check(search["charged_evaluations"] == len(rows) and search["created_proposals"] == len(proposal),
            "search", f"Actual trace counters:{name}")
        if mode != "E":
            remainder = [key for key in catalog if key not in set(initial)]
            random.Random(job["search_seed"]).shuffle(remainder)
            expected_order = initial + tuple(remainder)
            audit.check(tuple(event["canonical_solution_id"] for event in proposal) == expected_order[:len(proposal)],
                "search", f"Uniform remaining-catalog stream:{name}")
        else:
            populations = [event for event in trace if event["event"] == "population"]
            audit.check(all(event["size"] > 0 for event in populations), "search", f"No empty E generation:{name}")
            audit.check(any(event["generation"] >= 2 for event in populations), "search", f"Real later E generation:{name}")
            audit.check(any(event["operator"] == "grammar_crossover" for event in proposal)
                and any(event["operator"] == "grammar_mutation" for event in proposal), "search", f"Real mutation/crossover:{name}")
            audit.check(search["engine"] == "FedcoreEvoOptimizer" and trace[0].get("cache_entries") == 0,
                "search", f"Real engine with fresh observations:{name}")
            audit.check(search["cache_hits"] == sum(event.get("cache_hit") is True for event in trace if event["event"] == "cache_observation")
                and search["component_reads"] == sum(event["event"] == "fitness_read" for event in trace),
                "search", f"Useful hits/technical component reads:{name}")
            for event in trace:
                if event["event"] == "proposal":
                    audit.check(identity(genotype(*event["genotype"]["genes"])) == event["canonical_solution_id"],
                        "search", f"Independent executable ID:{name}/{event['proposal_id']}")
            audit.check(len({row["solution_id"] for row in rows}) == len(rows), "search", f"Each uncached unique evaluation charged once:{name}")
        search_facts.append({"job": name, "attempts": len(rows), "proposals": len(proposal), "stop_reason": search["stop_reason"],
            "within": len(within[name]), "late": len(finite(rows))-len(within[name]), "hypervolume": rectangle_union(points(within[name]))})
    audit.facts["searches"] = search_facts

    order_count, order_pairs = 0, 0
    for model in MODELS:
        item = jobs[f"order-{model}"]
        rows = item["result"]["rows"]
        grouped = defaultdict(list)
        audit.check(len(rows) == 12 and set(item["result"]["masks_sha256"]) == {"0.25", "0.5"}, "order", f"12outputs/fixed masks:{model}")
        for row in rows:
            grouped[row["pair"]].append(row)
            audit.check(row["tensor_bytes"] == DENSE and row["global_attempt"] > inherited, "order", f"Fresh dense49416:{model}/{row['word']}")
            audit.check(0 <= row["zero_fraction"] <= 1 and len(row["singular_values"]) == 64
                and all(math.isfinite(value) and value >= 0 for value in row["singular_values"]), "order", f"Finite64-value spectrum:{model}/{row['word']}")
            if row["word"] == "LDP":
                audit.check(row["zero_fraction"] >= row["mask_fraction"], "order", f"Explicit final mask retained:{model}/{row['pair']}")
        for pair, values in grouped.items():
            expected = {"PLD", "LDP"} if pair.startswith("mask-") else {"LDQD", "QDLD"}
            audit.check(len(values) == 2 and {row["word"] for row in values} == expected, "order", f"Same-operation paired order:{model}/{pair}")
            order_pairs += 1
        order_count += len(rows)
    audit.facts.update(order_outputs=order_count, order_pairs=order_pairs)

    cache_facts = []
    for model in MODELS:
        for seed in (9921, 9922, 9923):
            a, b = (jobs[f"cache-{model}-{seed}-{branch}"] for branch in "AB")
            ar, br = a["result"]["requests"], b["result"]["requests"]
            rng = random.Random(seed)
            expected = rng.sample(list(tuned), 12)*4
            rng.shuffle(expected)
            audit.check([row["solution_id"] for row in ar] == [row["solution_id"] for row in br] == expected,
                "cache", f"Frozen48-request stream:{model}/{seed}")
            audit.check(set(Counter(expected).values()) == {4} and len(set(expected)) == 12, "cache", f"12unique×4:{model}/{seed}")
            seen_success, expected_attempts = set(), 0
            for left, right in zip(ar, br):
                same = left["status"] == right["status"] and left.get("tensor_bytes") == right.get("tensor_bytes")
                if left["status"] == "finite":
                    same = same and near(left["quality"], right["quality"])
                audit.check(same, "cache", f"Same actual paired outcome:{model}/{seed}/{left['index']}")
                expected_hit = left["solution_id"] in seen_success
                audit.check(left["cache_hit"] == expected_hit and right["cache_hit"] is False, "cache", f"Only successful results cached:{model}/{seed}/{left['index']}")
                if not expected_hit:
                    expected_attempts += 1
                if left["status"] == "finite":
                    seen_success.add(left["solution_id"])
            audit.check(len(a["rows"]) == expected_attempts and len(b["rows"]) == 48, "cache", f"Actual12/48 accounting (or repeated failures):{model}/{seed}")
            first_a = ((model-41)*3 + seed-9921) % 2 == 0
            audit.check((a["job"]["worker_started_at"] < b["job"]["worker_started_at"]) == first_a, "cache", f"FrozenAB/BA:{model}/{seed}")
            cache_facts.append({"model": model, "seed": seed, "on_evaluations": len(a["rows"]), "off_evaluations": len(b["rows"]),
                "hits": sum(row["cache_hit"] for row in ar), "on_over_off": a["result"]["stream_seconds"]/b["result"]["stream_seconds"]})
    audit.facts["cache_pairs"] = cache_facts

    selected = load(run / "selected.json")
    chosen = {}
    for name in searches + [f"oracle-{model}" for model in MODELS]:
        rows = within[name] if name in within else finite(jobs[name]["rows"])
        for cap in CAPS:
            row = selection(rows, cap)
            chosen[(name, cap)] = row
    audit.check(len(selected["selected"]) == len(chosen) == 171, "selection", "57trajectories×3caps before test")
    for value in selected["selected"]:
        row = chosen[(value["job"], value["byte_cap"])]
        audit.check(value["solution_id"] == (row["solution_id"] if row else None), "selection", f"Independent frozen choice:{value['job']}/{value['byte_cap']}")
        if row:
            audit.check(near(value["validation_quality"], row["quality"]), "selection", f"Frozen validation value:{value['job']}/{value['byte_cap']}")
    nonfinal_end = max(item["result"]["completed_at"] for item in jobs.values() if item["job"]["kind"] != "finalize")
    audit.check(selected["frozen_at"] >= nonfinal_end, "selection", "Selection frozen after all non-test series")
    selected_hash = sha(run / "selected.json")
    final_count = 0
    for model in MODELS:
        item = jobs[f"finalize-{model}"]
        job, result = item["job"], item["result"]
        expected = [value for value in selected["selected"] if value["model_seed"] == model and value["solution_id"] is not None]
        audit.check(job["selections_sha256"] == selected_hash and job["worker_started_at"] >= selected["frozen_at"]
            and result["selection_frozen_at"] == selected["frozen_at"], "selection", f"Test opened after immutable selection:{model}")
        audit.check(result["historical_test"] is True and result["baseline_test"]["samples"] == 86,
            "selection", f"Historical baseline test recorded:{model}")
        expected_keys = {(value["job"], value["byte_cap"], value["solution_id"]) for value in expected}
        actual_keys = {(value["job"], value["byte_cap"], value["solution_id"]) for value in result["selected"]}
        audit.check(actual_keys == expected_keys and len(result["selected"]) == len(expected), "selection", f"Every selected solution reconstructed once:{model}")
        for value in result["selected"]:
            audit.check(value["tensor_bytes"] <= value["byte_cap"] and value["test"]["samples"] == 86
                and sha(value["artifact_path"]) == value["checkpoint_sha256"], "selection", f"Selected artifact/test constraints:{model}/{value['job']}/{value['byte_cap']}")
        final_count += len(result["selected"])
    audit.facts.update(selected_test_rows=final_count, baseline_test_rows=3)

    if published and Path(published).exists():
        analysis = load(published)
        audit.check(analysis["status"] == "complete" and not analysis["issues"] and not analysis["missing_jobs"] and not analysis["failed_jobs"],
            "published", "Published completeness/issues")
        audit.check(analysis["protocol_sha256"] == protocol["protocol_sha256"] and analysis["hypervolume_reference"] == [2., 1.05]
            and analysis["normalization_area"] == 2.1, "published", "Fixed objective/reference/protocol")
        frontiers = {value["job"]: value for value in analysis["frontiers"]}
        published_runs = {row["job"]: row for row in analysis["search_runs"] + analysis["oracle_runs"]}
        for summary in published_runs.values():
            name = summary["job"]
            rows = within[name] if name in within else finite(jobs[name]["rows"])
            area = rectangle_union(points(rows))
            accepted_area = rectangle_union(points([row for row in rows if row["quality_accepted"]]))
            audit.check(near(summary["hypervolume"], area) and near(summary["normalized_hypervolume"], area/2.1)
                and near(summary["accepted_hypervolume"], accepted_area), "published", f"Independent rectangle union:{name}")
            reported_frontier = {(row["solution_id"], row["quality"], row["cost"]) for row in frontiers[name]["all"]}
            audit.check(reported_frontier == nondominated(rows), "published", f"Independent quadratic Pareto set:{name}")
            job = jobs[name]["job"]
            correct_attempts = [row for row in attempts if row["job"] == name and row["purpose"] == "full_validation"
                and row["protocol_sha256"] == job["protocol_sha256"]]
            audit.check(summary["full_evaluations_started"] == len(correct_attempts), "published", f"No duplicate logical-job charges:{name}")
            if name in within:
                expected_stop = ("not_applicable", None) if job["budget"] == "attempts" or job["mode"] == "R" else diagnostic(rows)
                observed = summary["virtual_stop"]
                audit.check(observed["status"] == expected_stop[0] and observed.get("stop_after_finite") == expected_stop[1],
                    "published", f"Independent12+6 diagnostic:{name}")
                for step in summary["curves"]:
                    prefix = [row for row in rows if row["completed_at"] <= step["completed_at"]]
                    audit.check(near(step["hypervolume"], rectangle_union(points(prefix))), "published", f"Completion-time archive curve:{name}/{step['global_attempt']}")
        pair_differences = defaultdict(list)
        for pair in analysis["paired"]:
            left, right = published_runs[pair["left_job"]], published_runs[pair["right_job"]]
            expected = right["hypervolume"]-left["hypervolume"]
            expected_accepted = right["accepted_hypervolume"]-left["accepted_hypervolume"]
            audit.check(pair["complete"] and near(pair["delta_hypervolume_second_minus_first"], expected)
                and near(pair["delta_accepted_hypervolume_second_minus_first"], expected_accepted),
                "published", f"Paired comparison delta:{pair['left_job']}/{pair['right_job']}")
            pair_differences[(pair["model_seed"], pair["budget"], pair["comparison"])].append(expected)
        for summary in analysis["paired_summary_by_model"]:
            expected = describe(pair_differences[(summary["model_seed"], summary["budget"], summary["comparison"])])
            observed = summary["delta_hypervolume_second_minus_first"]
            audit.check(observed["count"] == expected["count"] and all(near(observed[key], expected[key])
                for key in ("median", "minimum", "maximum")), "published", f"Per-model paired median/range:{summary['model_seed']}/{summary['budget']}/{summary['comparison']}")
        for pair in analysis["common_byte_pairs"]:
            values = {}
            for mode in ("R", "T"):
                name = f"search-{pair['model_seed']}-{pair['budget']}-{pair['search_seed']}-{mode}"
                feasible = [row for row in within[name] if row["tensor_bytes"] == pair["tensor_bytes"]]
                values[mode] = min(feasible, key=lambda row: (row["quality"], row["solution_id"])) if feasible else None
                audit.check((pair[mode] or {}).get("solution_id") == (values[mode] or {}).get("solution_id"),
                    "published", f"Actual equal-byte choice:{name}/{pair['tensor_bytes']}")
            if values["R"] and values["T"]:
                delta = values["T"]["quality"]-values["R"]["quality"]
                audit.check(near(pair["delta_quality_T_minus_R"], delta) and near(pair["delta_CE_T_minus_R"], delta*math.log(2)),
                    "published", f"Actual equal-byte CE delta:{pair['model_seed']}/{pair['budget']}/{pair['search_seed']}/{pair['tensor_bytes']}")
        for pair in analysis["early_late_pairs"]:
            expected = next(row["early_over_late"] for row in early_late_facts if row["seed"] == pair["stream_seed"])
            audit.check(near(pair["early_over_late"], expected), "published", f"Measured early/late ratio:{pair['stream_seed']}")
        for pair in analysis["cache_pairs"]:
            expected = next(row for row in cache_facts if row["model"] == pair["model_seed"] and row["seed"] == pair["stream_seed"])
            audit.check(near(pair["enabled_over_disabled"], expected["on_over_off"])
                and pair["A_full_evaluations"] == expected["on_evaluations"] and pair["B_full_evaluations"] == expected["off_evaluations"]
                and pair["A_cache_hits"] == expected["hits"], "published", f"Measured cache ratio/count:{pair['model_seed']}/{pair['stream_seed']}")
        for key, actual_values in (("early_late_ratio_summary", [row["early_over_late"] for row in early_late_facts]),
                                  ("cache_ratio_summary", [row["on_over_off"] for row in cache_facts])):
            expected, observed = describe(actual_values), analysis[key]
            audit.check(observed["count"] == expected["count"] and all(near(observed[name], expected[name])
                for name in ("median", "minimum", "maximum")), "published", f"Measured paired-ratio median/range:{key}")
        audit.facts["paired_summary_by_model"] = [{"model": key[0], "budget": key[1], "comparison": key[2], **describe(values)}
            for key, values in sorted(pair_differences.items())]
        audit.facts["early_over_late_summary"] = describe([row["early_over_late"] for row in early_late_facts])
        audit.facts["cache_on_over_off_summary"] = describe([row["on_over_off"] for row in cache_facts])
        audit.check(analysis["resources"]["recorded_mlp_attempts"] == len(attempts), "published", "All superseded/replay costs counted")
        maximum = max(record.get("resources", {}).get("peak_tree_rss_bytes", 0) for record in resources.values())
        audit.check(analysis["resources"]["maximum_worker_tree_rss_bytes"] == maximum, "published", "Global RSS maximum across original revisions")
        audit.check(len(analysis["early_late_pairs"]) == 6 and all(row["valid"] for row in analysis["early_late_pairs"])
            and len(analysis["cache_pairs"]) == 9 and all(row["valid"] for row in analysis["cache_pairs"]),
            "published", "All independent paired series represented")
        audit.facts["published_analysis_sha256"] = sha(published)
    else:
        audit.warn("Published analysis was absent/not requested; rerun this audit after publication to verify its numerical summaries.")
    audit.warn("No model, dataset, logits or checkpoint tensor was loaded. CE/F1 and tensor content parity rely on recorded execution witnesses and earlier independent tests; this audit checks stored-normalization arithmetic, provenance and hashes.")
    audit.warn("Absence of unlogged oracle/test reads cannot be proved from event files alone; frozen source review plus fresh charged evaluations provide the isolation evidence.")
    return {"status": "passed" if not audit.failures else "failed", "run": str(run), "verifier_sha256": sha(__file__),
        "checks": dict(audit.counts), "failures": audit.failures, "limitations": audit.warnings, "facts": audit.facts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = main(arguments.run, arguments.analysis)
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n"
    if arguments.output:
        arguments.output.write_text(payload, encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": sum(result["checks"].values()),
        "failures": result["failures"], "facts": {key: value for key, value in result["facts"].items()
        if key not in ("searches", "early_late", "cache_pairs")}}, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if result["status"] == "passed" else 1)
