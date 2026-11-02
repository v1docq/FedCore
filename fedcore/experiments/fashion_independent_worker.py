"""S2: independent search streams, exact S1 M0, unused-training holdout.

No command in this module trains a model or opens the official t10k files.
All numerical commands remain owned by the root resource guardian.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
import time

from . import fashion_worker as common
from .fashion_protocol import (FashionPlan, CONTROLS, default_manifest, stratified_subset,
                               source_code_hashes)
from .quantization_pilot_protocol import canonical_bytes, digest, file_hash
from .quantization_pilot_guard import Limits, atomic_json

S1_COMMIT = "9295f6b36151b1d163b06abc27f3d3ab182a624c"
M0_SHA256 = "b333cdaa33f29d07e9be27249145aa95ffba6459969c66ce481bf8ae15282429"


def default_s2_manifest(source_commit, raw_sha256, parent):
    result = default_manifest(source_commit, dict(raw_sha256))
    result["schema"] = "pr166_fashion_s2_v1"
    result["parent"] = dict(parent)
    result["data"]["test_rows"] = 6000
    result["data"]["holdout"] = {"source": "unused official training rows", "pool_rows": 6232,
        "rows": 6000, "per_class": 600, "seed": 20261031,
        "rule": "sorted union(calibration minus C, search minus Vsearch); stratified_subset",
        "exclude": ["fit", "stop", "C", "Vsearch", "pick"]}
    result["search"]["initialization"] = "independent random.Random(search_seed) stream of 63-bit proposal seeds; canonical genotype deduplication; Evo shares Random first four"
    result["test_policy"] = "single exclusive unused-training holdout opening after S2 selection lock; full 6000 per eligible artifact; never reopen official t10k"
    result["training_reuse"] = {"mode": "exact checkpoint copy; no retraining", "source_sha256": M0_SHA256,
        "controls": "copy all five unchanged S1 control artifacts with SHA256 verification"}
    return result


@dataclass(frozen=True)
class IndependentFashionPlan:
    payload: dict
    fingerprint: str

    @classmethod
    def parse(cls, raw):
        if type(raw) is not dict or raw.get("schema") != "pr166_fashion_s2_v1":
            raise ValueError("Unknown S2 protocol")
        # The original finite parser validates source identity and four raw
        # hashes without permitting arbitrary models or recipes in this schema.
        base = FashionPlan.parse(default_manifest(raw.get("source_commit"), raw.get("raw_sha256")))
        parent = raw.get("parent")
        keys = {"source_commit", "manifest_sha256", "prepared_sha256", "trained_sha256", "data_sha256",
                "code_lock_sha256", "source_sha256"}
        if type(parent) is not dict or set(parent) != keys or parent["source_commit"] != S1_COMMIT:
            raise ValueError("Exact S1 parent identity required")
        if any(type(parent[key]) is not str or len(parent[key]) != 64
               or any(c not in "0123456789abcdef" for c in parent[key]) for key in keys-{"source_commit"}):
            raise ValueError("Full parent SHA256 identities required")
        if parent["source_sha256"] != M0_SHA256:
            raise ValueError("S2 must reuse the approved exact M0 checkpoint")
        reference = default_s2_manifest(base.payload["source_commit"], base.payload["raw_sha256"], parent)
        if canonical_bytes(raw) != canonical_bytes(reference):
            raise ValueError("S2 fields differ from the approved finite amendment")
        normalized = json.loads(canonical_bytes(raw))
        return cls(normalized, digest(normalized))

    @classmethod
    def read(cls, path):
        return cls.parse(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def parent_reference(parent_root, parent_manifest=None):
    parent_root = Path(parent_root).resolve()
    manifest_path = Path(parent_manifest) if parent_manifest else parent_root/"manifest.json"
    if not manifest_path.resolve().is_relative_to(parent_root):
        raise ValueError("Parent manifest must belong to the S1 run")
    plan = FashionPlan.read(manifest_path)
    prepared = json.loads((parent_root/"prepared.json").read_text())
    trained = json.loads((parent_root/"trained.json").read_text())
    if (plan.payload["source_commit"] != S1_COMMIT or trained["status"] != "accepted"
            or prepared["manifest_sha256"] != plan.fingerprint or trained["manifest_sha256"] != plan.fingerprint
            or trained["prepared_sha256"] != file_hash(parent_root/"prepared.json")
            or trained["data_sha256"] != prepared["data_sha256"] or trained["source"]["sha256"] != M0_SHA256):
        raise ValueError("S1 M0 or data provenance differs from the approved parent")
    reference = {"source_commit": S1_COMMIT, "manifest_sha256": file_hash(manifest_path),
        "prepared_sha256": file_hash(parent_root/"prepared.json"), "trained_sha256": file_hash(parent_root/"trained.json"),
        "data_sha256": prepared["data_sha256"], "code_lock_sha256": file_hash(parent_root/"code.lock.json"),
        "source_sha256": trained["source"]["sha256"]}
    return reference, plan, prepared, trained


def unused_training_holdout(indices, labels, *, rows=6000, seed=20261031):
    """Select only rows outside every operational training/selection role."""
    lists = {name: [int(i) for i in values] for name, values in indices.items()}
    used_roles = ("fit", "stop", "C", "Vsearch", "pick")
    used = set()
    for role in used_roles:
        current = set(lists[role])
        if len(current) != len(lists[role]) or used & current:
            raise ValueError("Operational roles must be disjoint and unique")
        used.update(current)
    if not set(lists["C"]) <= set(lists["calibration"]) or not set(lists["Vsearch"]) <= set(lists["search"]):
        raise ValueError("Operational subsets must belong to their declared pools")
    pool = sorted((set(lists["calibration"])-set(lists["C"])) |
                  (set(lists["search"])-set(lists["Vsearch"])))
    if used & set(pool):
        raise ValueError("Unused holdout pool overlaps an operational role")
    selected = stratified_subset(pool, labels, rows, seed=seed)
    if len(set(selected)) != rows or set(selected) & used:
        raise ValueError("Heldout identities violated the exclusion contract")
    return selected, pool


def copy_artifact(parent_root, root, metadata):
    parent_root, root = Path(parent_root).resolve(), Path(root).resolve()
    relative = Path(metadata["checkpoint"])
    source, target = (parent_root/relative).resolve(), (root/relative).resolve()
    if not source.is_relative_to(parent_root) or not target.is_relative_to(root):
        raise ValueError("Artifact path escaped its run root")
    if file_hash(source) != metadata["sha256"]:
        raise ValueError("Parent artifact SHA256 mismatch")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_hash(target) != metadata["sha256"]:
            raise ValueError("Existing destination artifact identity mismatch")
    else:
        shutil.copyfile(source, target)
    if file_hash(target) != metadata["sha256"]:
        raise ValueError("Copied artifact identity mismatch")
    return dict(metadata)


def reuse_m0(plan, root, parent_root, parent_manifest=None):
    import torch
    from fedcore.algorithm.low_rank.candidate_storage import capture_inventory, inventory_cost
    root, parent_root = Path(root).resolve(), Path(parent_root).resolve()
    if root == parent_root or root.is_relative_to(parent_root) or parent_root.is_relative_to(root):
        raise ValueError("S1 and S2 must use separate sibling run roots")
    common.verify_code(root)
    if (root/"reuse.started.json").exists() or (root/"prepared.json").exists():
        raise FileExistsError("S2 source reuse cannot be replaced or restarted")
    reference, _, parent_prepared, parent_trained = parent_reference(parent_root, parent_manifest)
    if reference != plan.payload["parent"] or file_hash(parent_root/"data.pt") != reference["data_sha256"]:
        raise ValueError("S1 parent changed after S2 manifest freeze")
    atomic_json(root/"reuse.started.json", {"parent": reference, "started_unix": time.time()})
    data = torch.load(parent_root/"data.pt", map_location="cpu", weights_only=True)
    if (common.tensor_hash(data["images"]) != parent_prepared["images_sha256"]
            or common.tensor_hash(data["labels"]) != parent_prepared["labels_sha256"]):
        raise ValueError("S1 image or ground-truth label tensor identity mismatch")
    for role, ids in data["indices"].items():
        if ids.tolist() != parent_prepared["roles"][role]["ids"]:
            raise ValueError("S1 role identities changed")
    heldout, pool = unused_training_holdout(data["indices"], data["labels"].tolist())
    if len(pool) != 6232:
        raise ValueError("Expected the predeclared 6232 unused training rows")
    data["indices"]["heldout"] = torch.tensor(heldout, dtype=torch.int64)
    torch.save(data, root/"data.pt")
    roles = {name: {"rows": len(ids), "ids": ids.tolist(), "ids_sha256": digest(ids.tolist()),
                   "class_counts": common.class_counts(data["labels"], ids)} for name, ids in data["indices"].items()}
    prepared = {"manifest_sha256": plan.fingerprint, "code_lock_sha256": common.verify_code(root),
        "data_sha256": file_hash(root/"data.pt"), "images_sha256": parent_prepared["images_sha256"],
        "labels_sha256": parent_prepared["labels_sha256"], "roles": roles, "parent": reference,
        "unused_pool_rows": len(pool), "unused_pool_ids_sha256": digest(pool), "closed_test_opened": False,
        "runtime": common.runtime_info(), "overlap_audit": "heldout disjoint from fit/stop/C/Vsearch/pick; original operational IDs unchanged"}
    atomic_json(root/"prepared.json", prepared)
    metadata = copy_artifact(parent_root, root, parent_trained["source"])
    loaded = common.load_artifact(root, metadata)
    if (inventory_cost(capture_inventory(loaded), "bytes") != metadata["physical_tensor_bytes"]
            or any(parameter.dtype is not torch.float32 for parameter in loaded.parameters())):
        raise ValueError("Reused M0 dtype or unique physical storage changed")
    trained = {**parent_trained, "manifest_sha256": plan.fingerprint, "prepared_sha256": file_hash(root/"prepared.json"),
        "code_lock_sha256": common.verify_code(root), "data_sha256": prepared["data_sha256"], "source": metadata,
        "reused_from": reference, "training_performed_in_s2": False, "runtime": common.runtime_info()}
    atomic_json(root/"trained.json", trained)
    (root/"jobs").mkdir(exist_ok=True)
    for job in CONTROLS:
        parent_path = parent_root/"jobs"/(job+".json")
        record = json.loads(parent_path.read_text())
        if record["manifest_sha256"] != parent_prepared["manifest_sha256"] or record["status"] != "succeeded":
            raise ValueError("Only successful unchanged S1 controls may be imported")
        artifact = copy_artifact(parent_root, root, record["artifact"])
        imported = {**record, "manifest_sha256": plan.fingerprint, "artifact": artifact,
            "trained_sha256": file_hash(root/"trained.json"), "imported_from_s1": True,
            "parent_result_sha256": file_hash(parent_path), "parent_source_sha256": M0_SHA256}
        atomic_json(root/"jobs"/(job+".json"), imported)
    return {"manifest_sha256": plan.fingerprint, "source_sha256": metadata["sha256"], "training_performed": False,
        "heldout_rows": len(heldout), "heldout_ids_sha256": roles["heldout"]["ids_sha256"], "imported_controls": list(CONTROLS)}


def heldout_loader(plan, root):
    root = Path(root)
    if not (root/"selection.lock.json").exists() or not (root/"CLOSED_TEST_OPENED.json").exists():
        raise PermissionError("S2 holdout metrics require selection lock and exclusive opening marker")
    import torch
    _, data = common.load_data(plan, root)
    ids = data["indices"]["heldout"]
    # Only this post-lock callback creates a metric input/label view of S2's
    # selected holdout. No official t10k file is read anywhere in this module.
    return data["images"][ids], data["labels"][ids], torch.arange(len(ids))


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("manifest", "reuse-m0", "job", "record-abort", "freeze", "closed-test", "analyze"))
    parser.add_argument("--manifest", required=True); parser.add_argument("--run-root", required=True); parser.add_argument("--limits", required=True)
    parser.add_argument("--parent-root"); parser.add_argument("--parent-manifest"); parser.add_argument("--source-commit")
    parser.add_argument("--job"); parser.add_argument("--guard-report")
    args = parser.parse_args(); root = Path(args.run_root).resolve(); common.require_guard(root)
    limits = Limits.read(args.limits)
    if limits.remaining() <= 0: raise TimeoutError("Original shared absolute deadline exhausted")
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    if args.command == "manifest":
        if not args.parent_root: raise ValueError("--parent-root is required")
        parent, parent_plan, _, _ = parent_reference(args.parent_root, args.parent_manifest)
        plan = IndependentFashionPlan.parse(default_s2_manifest(args.source_commit, parent_plan.payload["raw_sha256"], parent))
        if Path(args.manifest).exists() or (root/"code.lock.json").exists():
            raise FileExistsError("S2 manifest cannot be replaced")
        atomic_json(args.manifest, plan.payload)
        atomic_json(root/"code.lock.json", {"source_commit": args.source_commit, "files": source_code_hashes()})
        result = {"manifest_sha256": plan.fingerprint, "code_lock_sha256": common.verify_code(root)}
    else:
        plan = IndependentFashionPlan.read(args.manifest)
        if args.command == "reuse-m0":
            if not args.parent_root: raise ValueError("--parent-root is required")
            result = reuse_m0(plan, root, args.parent_root, args.parent_manifest)
        elif args.command == "job": result = common.run_job(plan, root, args.job, limits, initial_sampler=common.unique_initial)
        elif args.command == "record-abort": result = common.record_abort(plan, root, args.job, args.guard_report)
        elif args.command == "freeze": result = common.freeze_selection(plan, root, limits)
        elif args.command == "closed-test":
            result = common.closed_test(plan, root, limits, data_loader=heldout_loader)
            prepared = json.loads((root/"prepared.json").read_text())
            result["holdout"] = {**plan.payload["data"]["holdout"], "original_train_ids_sha256": prepared["roles"]["heldout"]["ids_sha256"]}
            atomic_json(root/"closed_test.json", result)
        else:
            from .fashion_analysis import analyze
            result = analyze(plan, root)
            result["series"] = "S2"; result["parent"] = plan.payload["parent"]
            result["holdout"] = plan.payload["data"]["holdout"]
            result["limitations"].append("S2 uses a new 6000-row holdout from unused training rows; S1 and S2 outcomes must not be pooled.")
            atomic_json(root/"analysis.json", result)
            with (root/"report.md").open("a", encoding="utf-8") as stream:
                stream.write("\nS2: independent proposal-seed streams; exact S1 M0 reused without training. "
                    "The closed holdout contains 6000 previously unused training rows, 600 per class. "
                    "The official S1 test was never reopened. S1 and S2 results must not be pooled.\n")
    print(json.dumps(common.jsonable(result), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
