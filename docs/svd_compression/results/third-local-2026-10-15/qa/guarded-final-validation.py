"""Run the established CPU regression suite after the immutable science run."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
import argparse

CHECKOUT = Path('C:/Users/user/Documents/Codex/FedCore-SVD-Comparisons-2026-10-13/checkout')
ARCHIVE = Path('C:/Users/user/Documents/Codex/FedCore-SVD-Third-Package-2026-10-15')
sys.path.insert(0, str(CHECKOUT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', default='02', choices=('02', '03'))
    args = parser.parse_args()
    if not (ARCHIVE / 'run.json').is_file():
        raise RuntimeError('All measurement stages must finish before final tests')
    from fedcore.experiments import svd_comparison
    from fedcore.experiments.svd_third_package import verify
    frozen = verify(ARCHIVE)
    destination = ARCHIVE / ('validation-process-' + args.attempt)
    if destination.exists():
        raise RuntimeError('Never replace an earlier validation attempt')
    temporary = ARCHIVE / ('validation-temp-' + args.attempt)
    temporary.mkdir(exist_ok=False)
    os.chdir(CHECKOUT)
    os.environ.update(PYTHONUTF8='1', TEMP=str(temporary), TMP=str(temporary),
                      CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                      HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    svd_comparison.THREADS = 8
    svd_comparison.RSS_LIMIT = 12 * 1024**3
    svd_comparison.OS_RESERVE = 6 * 1024**3
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/refactoring',
               'tests/petra', 'tests/unit/algorithm/low_rank',
               'tests/unit/low_rank/test_statistical_profiles.py',
               'tests/unit/low_rank/test_statistical_collectors.py',
               '--basetemp=' + str(temporary / 'pytest'),
               '--junitxml=' + str(destination / 'pytest.xml')]
    observed = svd_comparison.monitored_process(command, destination, time.monotonic() + 600.)
    xml = ET.parse(destination / 'pytest.xml').getroot()
    suites = list(xml.iter('testsuite'))
    total = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
             for key in ('tests', 'failures', 'errors', 'skipped')}
    total['passed'] = total['tests'] - total['failures'] - total['errors'] - total['skipped']
    verify(ARCHIVE)
    source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    passed = not observed['returncode'] and not observed['violation'] and not total['failures'] and not total['errors']
    result = {'schema': 3, 'status': 'passed' if passed else 'failed', 'counts': total,
              'attempt': args.attempt, 'process_directory': destination.name,
              'scientific_source_sha256': frozen['source']['sha256'],
              'protocol_sha256': frozen['protocol_sha256'],
              'observed': {k: v for k, v in observed.items() if k != 'samples'},
              'command': command, 'guard_seconds': 600., 'guard_rss_bytes': 12 * 1024**3,
              'minimum_ram_bytes': 6 * 1024**3,
              'script_sha256': source,
              'new_scientific_training_updates': 0,
              'scope': 'Postmeasurement CPU regressions; ephemeral test fixtures are not scientific training'}
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    (ARCHIVE / ('validation-attempt-' + args.attempt + '.json')).write_text(encoded, encoding='utf-8')
    if passed:
        (ARCHIVE / 'validation.json').write_text(encoded, encoding='utf-8')
    print(json.dumps({'status': result['status'], 'counts': total,
                      'elapsed_seconds': observed['elapsed_seconds']}, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
