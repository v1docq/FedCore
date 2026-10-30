"""Independent recount and strict aggregation; no model evaluation required."""
from dataclasses import replace

import numpy as np
import pytest

from fedcore.experiments.cv_independent_analysis import hypervolume_2d, primary_results, recount_predictions
from fedcore.experiments.cv_independent_protocol import SearchOutcome


def test_hypervolume_is_union_area_with_fixed_reference_and_keeps_dominated_points_harmless():
    assert hypervolume_2d([(1., .4)]) == pytest.approx(.2)
    assert hypervolume_2d([(1., .4), (1.1, .5), (2., .2), (.5, .9)]) == pytest.approx(.2)
    assert hypervolume_2d([(.5, .7), (1., .4)]) == pytest.approx(.25)


def test_prediction_recount_uses_saved_row_losses_and_checks_hash(tmp_path):
    from fedcore.experiments.cv_sparse_runtime import sha256_file
    path = tmp_path/'rows.npz'
    np.savez(path, row_ids=[4, 5], labels=[0, 1], predictions=[0, 0], cross_entropy=[.2, .8])
    metrics = {'prediction_file': path.name, 'prediction_sha256': sha256_file(path),
               'rows': 2, 'correct': 1, 'top1': .5, 'cross_entropy': .5}
    assert recount_predictions(tmp_path, metrics)['top1'] == .5
    with pytest.raises(ValueError, match='recount differs'):
        recount_predictions(tmp_path, {**metrics, 'cross_entropy': .4})
    with pytest.raises(ValueError, match='changed'):
        recount_predictions(tmp_path, {**metrics, 'prediction_sha256': 'f'*64})


def test_partial_M0_never_gets_a_mean_from_only_the_successful_seeds():
    from types import SimpleNamespace
    from fedcore.experiments.cv_independent_protocol import search_jobs
    key = 'a'*64
    outcomes = [SearchOutcome(j.name, 'completed', 32, key, True) for j in search_jobs()]
    quality = {key: {'eager': {'cross_entropy': .5, 'top1': .9}}}
    full = primary_results(SimpleNamespace(outcomes=outcomes), quality)
    assert full['complete'] and full['panel_mean_delta_ce'] == 0.
    outcomes[0] = replace(outcomes[0], charged=31, status='partial', complete_q32=False)
    partial = primary_results(SimpleNamespace(outcomes=outcomes), quality)
    assert not partial['complete'] and partial['panel_mean_delta_ce'] is None
    assert partial['models']['M1']['mean_delta_ce'] is None
    assert partial['models']['M2']['mean_delta_ce'] == 0.
