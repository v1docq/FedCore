"""Observed Windows sharing failure preserves old records until publication."""
import json
from pathlib import Path

import pytest
from fedcore.tools.atomic_json import write_atomic_json


def test_transient_sharing_failure_retried_without_corrupting_previous_record(tmp_path, monkeypatch):
    target = tmp_path / 'паспорт.json'
    write_atomic_json(target, {'status': 'old'})
    original = Path.replace
    attempts = []
    def locked(path, destination):
        if Path(destination) == target:
            attempts.append(path)
            if len(attempts) < 3:
                assert json.loads(target.read_text(encoding='utf-8')) == {'status': 'old'}
                raise PermissionError('simulated Windows sharing violation')
        return original(path, destination)
    monkeypatch.setattr(Path, 'replace', locked)
    write_atomic_json(target, {'status': 'new'})
    assert len(attempts) == 3
    assert json.loads(target.read_text(encoding='utf-8')) == {'status': 'new'}
    assert not list(tmp_path.glob('*.tmp'))


def test_persistent_failure_and_invalid_data_do_not_overwrite_record(tmp_path, monkeypatch):
    target = tmp_path / 'manifest.json'
    write_atomic_json(target, {'status': 'old'})
    def locked(*args):
        raise PermissionError('persistent lock')
    monkeypatch.setattr(Path, 'replace', locked)
    with pytest.raises(PermissionError):
        write_atomic_json(target, {'status': 'new'})
    with pytest.raises(ValueError):
        write_atomic_json(target, {'metric': float('nan')})
    assert json.loads(target.read_text()) == {'status': 'old'}
    assert not list(tmp_path.glob('*.tmp'))
