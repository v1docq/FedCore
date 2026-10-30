"""Check artifact data and JavaScript syntax without browser execution."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib
import json
import subprocess

SUPPORT = Path(__file__).resolve().parent
RAW = Path('C:/Users/user/Documents/Codex/FedCore-SVD-Third-Package-2026-10-15')
HTML = Path('C:/Users/user/Documents/Codex/FedCore-SVD-Comparisons-2026-10-13/checkout/docs/svd_compression/third-local-package-infographic.html')


class Parser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.scripts, self.active, self.external = [], [], None, []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        if tag == 'script':
            self.active = {'attrs': attrs, 'text': ''}
            if attrs.get('src'):
                self.external.append(attrs['src'])

    def handle_data(self, data):
        if self.active is not None:
            self.active['text'] += data

    def handle_endtag(self, tag):
        if tag == 'script':
            self.scripts.append(self.active)
            self.active = None


def main():
    p = Parser()
    p.feed(HTML.read_text(encoding='utf-8'))
    assert len(p.ids) == len(set(p.ids)), 'Duplicate element IDs'
    assert not p.external, 'Unexpected external script dependency'
    payload = next(s for s in p.scripts if s['attrs'].get('id') == 'data')
    embedded = json.loads(payload['text'])
    analysis = json.loads((RAW / 'analysis/analysis.json').read_text(encoding='utf-8'))
    protocol = json.loads((RAW / 'protocol.json').read_text(encoding='utf-8'))
    assert embedded['comparisons'] == analysis['comparisons']
    assert embedded['profiles'] == analysis['profiles']
    assert embedded['graphs'] == analysis['prepared']['graphs']
    assert embedded['source'] == protocol['source']['sha256']
    assert embedded['protocol'] == protocol['protocol_sha256']
    assert embedded['counts'] == {'processes': 7, 'timing_processes': 6, 'pair_units': 600,
        'wall_ms': 20800, 'full_path_ms': 20800, 'cuda_event_ms': 10400, 'throughput_seconds': 1200}
    eligible = [x for x in embedded['profiles'] if x['stages'].get('main', {}).get('ranking_eligible')]
    assert [(x['device'], x['batch']) for x in eligible] == [('cpu', 64)]
    for script in p.scripts:
        if script['attrs'].get('type') == 'application/json':
            continue
        target = SUPPORT / 'third-inline-script.js'
        target.write_text(script['text'], encoding='utf-8')
        checked = subprocess.run(['C:/Program Files/nodejs/node.exe', '--check', str(target)],
                                 capture_output=True, text=True, timeout=15)
        assert checked.returncode == 0, checked.stderr
    result = {'schema': 3, 'status': 'passed', 'element_ids': len(p.ids),
        'embedded_comparisons': len(embedded['comparisons']),
        'embedded_profiles': len(embedded['profiles']), 'external_scripts': p.external,
        'html_sha256': hashlib.sha256(HTML.read_bytes()).hexdigest(),
        'javascript_syntax': 'node --check passed; browser code not executed',
        'embedded_data': 'exact equality with original analysis, protocol and verified counts',
        'visual_inspection': 'not performed: browser policy rejected file://; no workaround attempted'}
    (RAW / 'html-validation.json').write_text(json.dumps(result, ensure_ascii=False,
        sort_keys=True, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
