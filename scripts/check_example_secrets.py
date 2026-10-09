"""Fail closed on credentials in examples, including notebook outputs and ZIPs.

Only filenames are printed: matching text must never appear in CI logs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import zipfile


PATTERNS = (
    re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    re.compile(rb"https?://[^\s/<>\"']+@(?:[^\s/<>\"']+)", re.IGNORECASE),
)


def contains_credentials(data: bytes) -> bool:
    """Inspect raw bytes and decoded JSON strings without returning a match."""
    def matches(value):
        return any(pattern.search(value) is not None for pattern in PATTERNS)
    if matches(data):
        return True
    # JSON/notebooks may escape URL slashes or token characters. Scan the decoded
    # string values too, so encoding a notebook output cannot bypass the check.
    try:
        value = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        return False
    def walk(item):
        if isinstance(item, str):
            return matches(item.encode("utf-8"))
        if isinstance(item, list):
            return any(walk(child) for child in item)
        if isinstance(item, dict):
            return any(walk(key) or walk(child) for key, child in item.items())
        return False
    return walk(value)


def scan_path(root: Path) -> list[str]:
    findings: list[str] = []
    paths = sorted(root.rglob("*")) if root.is_dir() else [root]
    for path in paths:
        if not path.is_file():
            continue
        label = path.as_posix()
        if contains_credentials(path.read_bytes()):
            findings.append(label)
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if not member.is_dir() and contains_credentials(archive.read(member)):
                        findings.append(f"{label}!{member.filename}")
    return sorted(set(findings))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="*", default=[Path("examples")])
    args = parser.parse_args(argv)
    try:
        findings = sorted({item for path in args.paths for item in scan_path(path)})
    except (OSError, zipfile.BadZipFile):
        # Deliberately omit exception text because filenames may contain secrets.
        print("Credential scan could not inspect an input file.", file=sys.stderr)
        return 2
    for filename in findings:
        print(filename)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
