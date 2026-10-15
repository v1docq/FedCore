"""Add final checks separately; never mutate the science ledger referenced by run.json."""
from pathlib import Path
import json
import hashlib

RAW = Path('C:/Users/user/Documents/Codex/FedCore-SVD-Third-Package-2026-10-15')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    original = RAW / 'ledger.json'
    science, run = read(original), read(RAW / 'run.json')
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    if digest != run['ledger']['sha256']:
        raise RuntimeError('Original scientific ledger changed')
    if read(RAW / 'validation.json')['status'] != 'passed':
        raise RuntimeError('Final validation must pass first')
    attempts = []
    for folder in sorted(RAW.glob('validation-process*')):
        if not folder.is_dir():
            continue
        resource = read(folder / 'resources.json')
        attempts.append({'directory': folder.name,
            'elapsed_seconds': resource['elapsed_seconds'], 'returncode': resource['returncode'],
            'violation': resource['violation']})
    # This reserve exceeds the separately reported diagnostic + analysis + audit
    # runtimes; it is explicitly conservative, not a measured execution time.
    reserve = 300.
    total = science['charged_seconds'] + sum(x['elapsed_seconds'] for x in attempts) + reserve
    if total > 7200:
        raise RuntimeError('Total packet charge exceeds its fixed limit')
    value = {'schema': 3, 'original_ledger_sha256': digest,
        'science_charged_seconds': science['charged_seconds'],
        'science_preflight_reserve_seconds': 300., 'validation_attempts': attempts,
        'postprocessing_diagnostic_publication_reserve_seconds': reserve,
        'reserve_scope': 'Conservative 300s for stdlib analysis, independent raw audit, portable tests/replay, Dask diagnostic, rendering and publication; excludes separately charged full validation processes',
        'charged_total_seconds': total, 'fixed_limit_seconds': 7200.,
        'original_science_ledger_unchanged': True,
        'new_scientific_training_updates': 0}
    (RAW / 'final-ledger.json').write_bytes((json.dumps(value, ensure_ascii=False,
        sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8'))
    print(json.dumps(value, sort_keys=True))


if __name__ == '__main__':
    main()
