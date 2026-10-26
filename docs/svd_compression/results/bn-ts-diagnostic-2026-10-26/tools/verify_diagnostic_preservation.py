"""Read-only audit of the original A100 inputs/environments after diagnostics."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

root = Path('/root/fedcore_bn_ts_diagnostic_20261026')
completion = Path('/root/fedcore_sparse_cv_completion_20261026')
original = Path('/root/fedcore_sparse_cv_20261021')

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()

admission = json.loads((root/'results/diagnostic-1/completion_admission.json').read_text())
consumed = admission['verified_files']
print(json.dumps({'admission_files_shape': type(consumed).__name__}), flush=True)
for name, row in consumed.items():
    path = completion/'results/completion-1'/name
    assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], name
assert sha(completion/'results/completion-1/report.json') == admission['report_sha256']
assert sha(root/'completion-file-inventory.json') == admission['inventory_sha256']

counts = {}
for label, parent in (('original', original), ('completion', completion)):
    manifest = json.loads((parent/'source-manifest.json').read_text())
    for name, digest in manifest['package_files_sha256'].items():
        assert sha(parent/'.venv/lib/python3.10/site-packages'/name) == digest, (label, name)
    assert sha(parent/'fedcore-0.0.5.4-py3-none-any.whl') == manifest['wheel_sha256']
    counts[label] = len(manifest['package_files_sha256'])

snapshot = json.loads((completion/'logs/parent-snapshot.json').read_text())
for row in snapshot['copied']:
    path = Path(row['source'])
    assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], row['source']

gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True).strip()
result = {
    'schema': 'fedcore-bn-ts-postrun-preservation-v1',
    'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'completion_consumed_files_verified': len(consumed),
    'old_parent_files_verified': len(snapshot['copied']),
    'packaged_sources_verified': counts,
    'original_wheels_unchanged': True,
    'gpu_compute_processes_after': gpu,
}
(root/'logs/diagnostic-postrun-preservation.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
