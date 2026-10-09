"""Published evidence retains its exact recorded bytes on every platform."""
import hashlib
import json
from pathlib import Path


def test_published_readiness_evidence_files_exist_and_match_hashes():
    root = Path(__file__).resolve().parents[2] / 'docs/petra/results/readiness-2026-10-06'
    hashes = json.loads((root / 'EVIDENCE_SHA256.json').read_text(encoding='utf-8'))
    assert hashes
    for name, expected in hashes.items():
        path = root / name
        assert path.is_file(), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
