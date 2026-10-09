"""Rebuild the published example ZIP from exactly the checked source files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def build_archive(root: Path | None = None) -> Path:
    root = root or Path(__file__).resolve().parents[2]
    names = ("examples/__init__.py", "examples/README.md", "examples/petra/__init__.py", "examples/petra/run.py",
             "examples/petra/view_results.py", "examples/petra/sequence.py", "examples/petra/inventory.json",
             "examples/petra/package.py", "docs/petra/examples.md")
    payload = {name: (root / name).read_bytes() for name in names}
    try:
        source_revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True,
                                         timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        source_revision = None
    passport = {"schema_version": 1, "purpose": "measured example entrypoints; install matching FedCore source first",
                "source_revision": source_revision,
                "library_source_sha256": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in sorted((root / "fedcore").rglob("*.py"))},
                "source_files": {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()},
                "contains_measured_results": False, "credentials_scan": "must pass scripts/check_example_secrets.py"}
    payload["PACKAGE_MANIFEST.json"] = (json.dumps(passport, indent=2, sort_keys=True) + "\n").encode()
    destination = root / "examples/fedcore_examples_export_onnx_and_docker.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 5, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    return destination


if __name__ == "__main__":
    print(build_archive())
