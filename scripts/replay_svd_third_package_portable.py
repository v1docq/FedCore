"""Post-measurement path adapter for the unchanged frozen third-packet analyzer.

Windows run indexes contain backslash-separated relative references. This
stdlib-only adapter makes a temporary POSIX-separated index; protocol, worker
results and resource evidence retain their exact original bytes and hashes.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedcore.tools.svd_third_analysis import analyze

OUTPUTS = ("analysis.json", "blocks.csv", "comparisons.csv")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_reference(reference):
    """Accept a relative file reference and change separators only."""
    if not isinstance(reference, str) or not reference or "\x00" in reference:
        raise ValueError("Relative result reference must be a nonempty string")
    normalized = reference.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (path.is_absolute() or PureWindowsPath(reference).drive
            or any(part in ("", ".", "..") or ":" in part for part in normalized.split("/"))
            or path.as_posix() != normalized):
        raise ValueError("Absolute, ambiguous, or traversal result reference is forbidden")
    return normalized


def contained(root, reference):
    path = (root / Path(*PurePosixPath(reference).parts)).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Result reference escapes its owning archive")
    return path


def semantic_output(path):
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def replay(root, destination, reference_analysis=None):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("Replay destination must be fresh")
    original_run_path, protocol_path = root / "run.json", root / "protocol.json"
    original_run_bytes = original_run_path.read_bytes()
    original_run = json.loads(original_run_bytes)
    view_run = copy.deepcopy(original_run)
    references, occupied = [], set()
    with tempfile.TemporaryDirectory(prefix="svd-third-portable-") as temporary:
        view = Path(temporary)
        shutil.copyfile(protocol_path, view / "protocol.json")
        for profile in view_run["profiles"]:
            for stage, record in profile["stages"].items():
                original = record["relative_path"]
                normalized = normalize_reference(original)
                if normalized in occupied:
                    raise ValueError("Duplicate normalized result reference")
                occupied.add(normalized)
                source = contained(root, normalized)
                target = view / Path(*PurePosixPath(normalized).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                if digest(source) != record["sha256"] or digest(source) != digest(target):
                    raise ValueError("Result bytes differ from original run identity")
                result = json.loads(source.read_text(encoding="utf-8"))
                mapping = {"device": profile["device"], "batch": profile["batch"], "stage": stage,
                           "original": original, "normalized": normalized, "result_sha256": digest(source)}
                if result.get("resources"):
                    resources = contained(root, str(PurePosixPath(normalized).parent / "resources.json"))
                    shutil.copyfile(resources, target.parent / "resources.json")
                    mapping["resources_sha256"] = digest(resources)
                    if mapping["resources_sha256"] != result["resources"]["sha256"]:
                        raise ValueError("Resources bytes differ from result identity")
                references.append(mapping)
                record["relative_path"] = normalized
        view_run_bytes = (json.dumps(view_run, ensure_ascii=False, sort_keys=True, indent=2,
                                     allow_nan=False) + "\n").encode("utf-8")
        (view / "run.json").write_bytes(view_run_bytes)
        analyze(view, destination)
    # Restore only separators in the copied index and prove semantic identity.
    restored = copy.deepcopy(view_run)
    for old_profile, restored_profile in zip(original_run["profiles"], restored["profiles"]):
        for stage in restored_profile["stages"]:
            restored_profile["stages"][stage]["relative_path"] = old_profile["stages"][stage]["relative_path"]
    if restored != original_run:
        raise ValueError("Temporary index changed fields beyond relative-path separators")
    reference = Path(reference_analysis).resolve() if reference_analysis else root / "analysis"
    comparisons = []
    if reference.exists():
        for name in OUTPUTS:
            expected, actual = reference / name, destination / name
            identical = semantic_output(expected) == semantic_output(actual)
            if not identical:
                raise ValueError(f"Replay changed numerical/status output: {name}")
            comparisons.append({"file": name, "semantic_identity": identical,
                                "byte_identity": expected.read_bytes() == actual.read_bytes(),
                                "reference_sha256": digest(expected), "replay_sha256": digest(actual)})
    if original_run_path.read_bytes() != original_run_bytes:
        raise ValueError("Original run index changed during replay")
    provenance = {"post_measurement_adapter": True,
        "frozen_scientific_sources_modified": False, "original_files_modified": False,
        "original_run_sha256": hashlib.sha256(original_run_bytes).hexdigest(),
        "temporary_view_run_sha256": hashlib.sha256(view_run_bytes).hexdigest(),
        "protocol_bytes_unchanged": True, "protocol_file_sha256": digest(protocol_path),
        "only_index_field_changes": "profiles[].stages.*.relative_path backslash to slash",
        "references": references,
        "numerical_status_invariance": {"verified_against_existing_analysis": bool(comparisons),
            "files": comparisons, "without_reference_basis": "exact same protocol/result/resources bytes and frozen analyzer"},
        "scope": "transport compatibility only; no new timings, training, thresholds, or metric selection"}
    (destination / "replay-path-normalization.json").write_text(json.dumps(provenance, ensure_ascii=False,
        sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--reference-analysis", type=Path)
    arguments = parser.parse_args()
    replay(arguments.root, arguments.destination, arguments.reference_analysis)
