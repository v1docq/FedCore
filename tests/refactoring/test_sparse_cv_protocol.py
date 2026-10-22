"""Scientific role boundaries, budget caps and late-result exclusion."""
from dataclasses import replace

import numpy as np
import pytest

from fedcore.experiments.cv_sparse_data import FrozenSelection, split_roles
from fedcore.experiments.cv_sparse_protocol import (
    PilotProtocol, STAGES, aa_admission, eligible_finalists, stage_deadline,
)


@pytest.mark.parametrize('field,value', [
    ('max_slot_seconds', 14401), ('max_slot_seconds', 14400.0),
    ('tf32', 0), ('baseline_epochs', 79), ('search_evaluations', 13),
])
def test_named_protocol_cannot_silently_change_a_resource_or_data_contract(field, value):
    with pytest.raises(ValueError, match='frozen'):
        replace(PilotProtocol(), **{field: value})


def test_all_stage_windows_fit_the_single_approved_slot_and_never_extend_it():
    protocol = PilotProtocol()
    assert sum(minutes for _, minutes in STAGES) * 60 == protocol.max_slot_seconds
    for name, minutes in STAGES:
        assert stage_deadline(100., 105., name) == 105.
        assert stage_deadline(100., 99999., name) == 100 + minutes * 60
    with pytest.raises(ValueError):
        stage_deadline(1., 99., 'another_free_retry')


def test_role_partition_is_complete_stratified_disjoint_deterministic_and_rng_isolated():
    labels = np.repeat(np.arange(10), 5000)
    before = np.random.get_state()
    first = split_roles(labels, seed=PilotProtocol().split_seed)
    second = split_roles(labels, seed=PilotProtocol().split_seed)
    after = np.random.get_state()
    assert first == second and first.sha256 == second.sha256
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    roles = (first.train, first.calibration, first.search_validation)
    assert tuple(map(len, roles)) == (44000, 1000, 5000)
    assert sorted(i for role in roles for i in role) == list(range(50000))
    for role, per_class in zip(roles, (4400, 100, 500)):
        assert np.array_equal(np.bincount(labels[list(role)]), [per_class] * 10)


def test_bad_source_class_counts_cannot_be_reinterpreted_as_official_cifar_train():
    labels = np.repeat(np.arange(10), 5000)
    labels[0] = 1
    with pytest.raises(ValueError, match='5000'):
        split_roles(labels, seed=1)


def test_selection_excludes_late_overbudget_nonfinite_and_unaccepted_results():
    base = {'status': 'succeeded', 'completed_within_deadline': True,
            'quality_accepted': True, 'tensor_bytes': 80, 'cross_entropy': 1., 'solution_id': 'b'}
    records = [dict(base, solution_id='late', cross_entropy=.01, completed_within_deadline=False),
               dict(base, solution_id='large', cross_entropy=.01, tensor_bytes=81),
               dict(base, solution_id='invalid', cross_entropy=.01, quality_accepted=False),
               dict(base, solution_id='nan', cross_entropy=float('nan')),
               dict(base, solution_id='bool-bytes', tensor_bytes=True),
               base, dict(base, solution_id='a'), dict(base, solution_id='c', tensor_bytes=70)]
    assert [row['solution_id'] for row in eligible_finalists(records, byte_ceiling=80)] == ['c', 'a', 'b']
    assert records[0]['cross_entropy'] == .01  # diagnostics are not rewritten


def test_test_seal_requires_every_selected_artifact_to_have_a_completed_hash():
    protocol = PilotProtocol()
    with pytest.raises(ValueError, match='Seal'):
        FrozenSelection(protocol.sha256, (('dense', 'a' * 64),), (('b' * 64, 'c' * 64),))
    seal = FrozenSelection(protocol.sha256, (('dense', 'a' * 64), ('same-finalist', 'a' * 64)),
                           (('a' * 64, 'b' * 64),))
    assert seal.protocol_sha256 == protocol.sha256


def test_test_loader_refuses_before_reading_dataset_without_this_protocol_seal(monkeypatch, tmp_path):
    from fedcore.experiments.cv_sparse_data import CifarData
    import torchvision.datasets
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('Test data opened before the seal')

    monkeypatch.setattr(torchvision.datasets, 'CIFAR10', forbidden)
    data = CifarData.__new__(CifarData)
    data.root, data.protocol = tmp_path, PilotProtocol()
    with pytest.raises(ValueError, match='sealed'):
        data.open_test(None)
    other = FrozenSelection('f' * 64, (('dense', 'a' * 64),), (('a' * 64, 'b' * 64),))
    with pytest.raises(ValueError, match='sealed'):
        data.open_test(other)
    assert not calls


def test_aa_admission_uses_the_fixed_twenty_pairs_and_both_order_groups():
    assert aa_admission([1.] * 20, ['AB', 'BA'] * 10)['accepted']
    assert not aa_admission([1.] * 40, ['AB', 'BA'] * 20)['accepted']
    assert not aa_admission([1.] * 20, ['AB'] * 20)['accepted']
    assert not aa_admission([.8, 1.2] * 10, ['AB', 'BA'] * 10)['accepted']
    assert not aa_admission([float('inf')] * 20, ['AB', 'BA'] * 10)['accepted']
