"""Deployment repeats preserve exact artifacts, ordering and trained weights."""
import json

import pytest
import torch

from fedcore.experiments import CandidateSpec, ExperimentRunner, ProtocolError
from fedcore.experiments.remeasure import remeasure
from tests.petra.test_protocol_runner import bundle, protocol


def test_real_fresh_process_remeasures_same_file_and_rejects_modified_weights(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = ExperimentRunner(bundle(), protocol(), tmp_path / 'run').run([CandidateSpec('baseline')])
    finally:
        torch.set_num_threads(previous)
    rows = remeasure(tmp_path / 'run/manifest.json', tmp_path / 'timings',
                     sessions=1, batch_sizes=(1, 4), repeats=3, warmup=1)
    session = json.loads((tmp_path / 'timings/session-0.json').read_text(encoding='utf-8'))
    assert len(rows) == 1
    assert len(session['records']) == 2
    assert all(r['artifact_sha256'] == result['candidates'][0]['measurement']['artifact']['sha256']
               for r in session['records'])
    assert all(r['prediction_parity'] == 'checked' and len(r['raw_inference_ms']) == 3 for r in session['records'])
    with pytest.raises(ProtocolError, match='empty'):
        remeasure(tmp_path / 'run/manifest.json', tmp_path / 'timings', sessions=1)
    artifact = result['candidates'][0]['measurement']['artifact']
    from pathlib import Path
    Path(artifact['path']).write_bytes(b'tampered')
    with pytest.raises(RuntimeError, match='session failed'):
        remeasure(tmp_path / 'run/manifest.json', tmp_path / 'tampered', sessions=1, repeats=1)
    assert json.loads((tmp_path / 'tampered/failure.json').read_text())['status'] == 'failed'


def test_empty_or_duplicate_batches_do_not_start_sessions(tmp_path):
    for batches in ((), (1, 1)):
        with pytest.raises(ProtocolError):
            remeasure(tmp_path / 'missing.json', tmp_path / 'none', batch_sizes=batches)
    assert not (tmp_path / 'none').exists()
