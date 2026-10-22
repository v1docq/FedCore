"""Preserve third-packet evidence bytes and record the portable publication."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

CHECKOUT = Path('C:/Users/user/Documents/Codex/FedCore-SVD-Comparisons-2026-10-13/checkout')
RAW = Path('C:/Users/user/Documents/Codex/FedCore-SVD-Third-Package-2026-10-15')
QA = Path(__file__).resolve().parent
DEST = CHECKOUT / 'docs/svd_compression/results/third-local-2026-10-15'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, sort_keys=True,
                    indent=2, allow_nan=False) + '\n').encode('utf-8'))


def main():
    if not (RAW / 'run.json').is_file() or not (RAW / 'validation.json').is_file():
        raise RuntimeError('Scientific run and final tests must finish first')
    if DEST.exists():
        raise RuntimeError('Publication destination must be fresh')
    sources = json.loads((RAW / 'source-freeze.json').read_text(encoding='utf-8'))
    for relative, expected in sources['lf_source_hashes'].items():
        frozen = RAW / 'frozen-source' / relative
        current = CHECKOUT / relative
        actual = hashlib.sha256(current.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if digest(frozen) != expected or actual != expected:
            raise RuntimeError('Frozen source differs: ' + relative)
    zip_path = RAW / 'frozen-source.zip'
    if zip_path.exists():
        raise RuntimeError('Frozen ZIP already exists; never replace it')
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted(sources['lf_source_hashes']):
            info = zipfile.ZipInfo(relative, date_time=(2026, 10, 15, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (RAW / 'frozen-source' / relative).read_bytes())
    science_paths = [RAW / name for name in ('prospective.json', 'protocol.json',
        'source-freeze.json', 'run.json', 'run-progress.json', 'ledger.json')]
    for folder in ('sources', 'frozen-source', 'attempts', 'analysis'):
        science_paths.extend(p for p in (RAW / folder).rglob('*') if p.is_file())
    science_paths = sorted(set(p for p in science_paths if p.exists()))
    scientific_manifest = {'schema': 3, 'scope': 'Immutable scientific files; excludes final QA and publication',
        'source_sha256': sources['sha256'], 'files': {p.relative_to(RAW).as_posix():
            {'sha256': digest(p), 'bytes': p.stat().st_size} for p in science_paths}}
    write(RAW / 'scientific-raw-manifest.json', scientific_manifest)
    selected = [p for p in science_paths if p.suffix in ('.json', '.jsonl', '.log', '.csv')
                and 'frozen-source' not in p.relative_to(RAW).parts]
    for name in ('validation.json', 'independent-review.json', 'final-ledger.json',
                 'scientific-raw-manifest.json', 'frozen-source.zip', 'html-validation.json'):
        if (RAW / name).is_file():
            selected.append(RAW / name)
    for name in ('replay-path-normalization.json', 'postprocessing-source-audit.json'):
        if (RAW / 'portable-replay' / name).is_file():
            selected.append(RAW / 'portable-replay' / name)
    selected += list(RAW.glob('validation-attempt-*.json'))
    for validation in RAW.glob('validation-process*'):
        if validation.is_dir():
            selected += [p for p in validation.iterdir() if p.is_file()]
    DEST.mkdir(parents=True)
    records = []
    for source in sorted(set(selected)):
        relative = source.relative_to(RAW).as_posix()
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if digest(source) != digest(target):
            raise RuntimeError('Copy changed evidence bytes')
        records.append({'path': relative, 'source_sha256': digest(source),
                        'published_sha256': digest(target), 'bytes': target.stat().st_size})
    qa_records = []
    for source in sorted(QA.rglob('*')):
        if (not source.is_file() or source.suffix not in ('.json', '.log', '.txt', '.py', '.html', '.xml')
                or any('temp' in part.lower() or part.lower().startswith('pytest')
                       for part in source.relative_to(QA).parts[:-1])):
            continue
        relative = 'qa/' + source.relative_to(QA).as_posix()
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        qa_records.append({'path': relative, 'source_sha256': digest(source),
                           'published_sha256': digest(target), 'bytes': target.stat().st_size})
    all_records = sorted(records + qa_records, key=lambda row: row['path'])
    passport = {'schema': 3, 'source_sha256': sources['sha256'],
        'protocol_sha256': json.loads((RAW / 'protocol.json').read_text(encoding='utf-8'))['protocol_sha256'],
        'policy': 'Exact original evidence bytes; absolute local artifact paths are metadata only; numerical replay uses archive-relative result paths',
        'files': all_records, 'file_count': len(all_records),
        'total_bytes': sum(row['bytes'] for row in all_records)}
    write(DEST / 'publication-manifest.json', passport)
    write(RAW / 'publication-manifest.json', passport)
    print(json.dumps({'files': len(all_records), 'bytes': passport['total_bytes'],
                      'scientific_files': len(science_paths), 'destination': str(DEST)}, sort_keys=True))


if __name__ == '__main__':
    main()
