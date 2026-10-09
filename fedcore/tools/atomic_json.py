"""Publish JSON atomically, tolerating short-lived Windows reader locks."""
import json
import os
from pathlib import Path
import tempfile
import time


def write_atomic_json(path, value):
    path = Path(path)
    # Serialize before touching the filesystem: invalid data cannot replace a
    # previously valid scientific record.
    serialized = json.dumps(value, indent=2, sort_keys=True, allow_nan=False,
                            ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix='.' + path.name + '-',
                                          suffix='.tmp', dir=path.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(serialized)
        for attempt in range(6):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(.02 * 2 ** attempt)
    finally:
        temporary.unlink(missing_ok=True)
