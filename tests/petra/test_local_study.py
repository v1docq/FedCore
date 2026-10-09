"""Paired descriptive summaries conserve failures and repeat identities."""
import copy

import pytest

from fedcore.experiments.local_study import summarize_pairs, run_local_study
from fedcore.experiments.protocol import ProtocolError


def manifest(seed, quality=.9, candidate_status='succeeded'):
    def record(method, value, size, latency):
        return {'configuration': {'method': method}, 'status': 'succeeded',
                'validation': {'metric': 'accuracy', 'direction': 'maximize', 'value': value},
                'measurement': {'metrics': {'file_bytes': size, 'latency_p50_ms': latency}}}
    baseline = record('baseline', .9, 100, 2.)
    candidate = record('svd', quality, 50, 3.)
    candidate['status'] = candidate_status
    return {'data': {'metadata': {'scenario': 'tabular'}}, 'protocol': {'seed': seed},
            'candidates': [baseline, candidate]}


def test_paired_summary_retains_missing_seeds_and_speed_regression():
    result = summarize_pairs([manifest(1, .92), manifest(2, .88), manifest(3, candidate_status='failed')])[0]
    assert result['paired_seeds'] == [1, 2]
    assert result['missing_seeds'] == [3]
    assert result['mean_validation_difference'] == pytest.approx(0.)
    assert result['sample_sd'] == pytest.approx(.02 * 2 ** .5)
    assert result['file_size_change_percent'] == [-50., -50.]
    assert result['latency_change_percent'] == [50., 50.]


def test_duplicate_repeat_and_mismatched_metric_cannot_form_summary():
    first = manifest(1)
    with pytest.raises(ProtocolError, match='Duplicate'):
        summarize_pairs([first, copy.deepcopy(first)])
    second = manifest(2)
    second['candidates'][1]['validation']['metric'] = 'mse'
    with pytest.raises(ProtocolError, match='metrics differ'):
        summarize_pairs([second])


def test_invalid_prospective_seeds_create_no_output(tmp_path):
    with pytest.raises(ProtocolError, match='unique'):
        run_local_study(tmp_path / 'run', seeds=(41, 41))
    assert not (tmp_path / 'run').exists()
