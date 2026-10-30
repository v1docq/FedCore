"""Prepare declared CPU contracts and immutable historical fixtures."""
import json
from pathlib import Path
import tarfile
root=Path(__file__).resolve().parent;repo=root/'checkout'
names=json.loads((root/'independent-test-list.json').read_text())
names=sorted(set(name.replace('\\','/') for name in names)|{'tests/refactoring/test_cv_compression_grammar.py'})
(root/'independent-installed-test-list.json').write_text(json.dumps(names,indent=2)+'\n')
files={repo/name for name in names}
for name in ('a100-pilot-2-2026-10-22','a100-completion-2026-10-26',
             'bn-ts-diagnostic-2026-10-26','recovery-bn-matrix-2026-10-26'):
    folder=repo/'docs/svd_compression/results'/name/'raw'
    files.update(path for path in folder.rglob('*') if path.is_file() and path.suffix in ('.json','.npz','.txt'))
assert all(path.is_file() for path in files)
with tarfile.open(root/'independent-checks.tar.gz','w:gz') as archive:
    for path in sorted(files):archive.add(path,arcname=path.relative_to(repo).as_posix())
print(json.dumps({'test_modules':len(names),'files':len(files),'archive_bytes':(root/'independent-checks.tar.gz').stat().st_size}))
