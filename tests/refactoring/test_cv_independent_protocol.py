"""Independent-M0 laws: roles, charging, seeds, finalist eligibility and seal."""
from dataclasses import asdict, replace

from hypothesis import given, strategies as st
import numpy as np
import pytest

from fedcore.experiments.cv_independent_data import recovery_tape, split_roles, tape_payload
from fedcore.experiments.cv_independent_protocol import (
    AttemptBudget, GlobalSeal, IndependentProtocol, MODELS, Reservation, SearchOutcome,
    STAGES, count_limits, derived_seed, digest, eligible_finalists, primary_aggregate,
    search_jobs, stage_deadline, timing_comparison_admitted,
)


@pytest.mark.parametrize('field,value', [
    ('recovery_lr', .01), ('recovery_updates', 199), ('max_slot_seconds', 14400),
    ('search_evaluations', 32.), ('bn_calibration', True), ('tf32', 0),
    ('search_generations', 3), ('max_vram_bytes', 36*1024**3),
])
def test_named_protocol_rejects_old_pilot_and_type_drift(field, value):
    with pytest.raises(ValueError, match='frozen'):
        replace(IndependentProtocol(), **{field: value})


def test_exact_counts_and_stage_caps_fit_only_the_new_eight_hour_slot():
    limits = count_limits()
    assert limits['baseline_updates'] == 75120
    assert limits['search_attempts'] == 576
    assert limits['recovery_updates'] == 117600
    assert limits['total_updates'] == 192720
    assert limits['Vreport_eager_rows'] == limits['Vreport_ts_rows'] == 165000
    assert limits['test_ts_rows'] == 330000
    assert limits['timing_sessions'] == 189 and limits['timing_pairs'] == 7560
    assert sum(minutes for _, minutes in STAGES)*60 == IndependentProtocol().max_slot_seconds == 28800
    for name, minutes in STAGES:
        assert stage_deadline(10., 20., name) == 20.
        assert stage_deadline(10., 99999., name) == 10+minutes*60


def test_fixed_job_order_and_independent_seed_domains():
    jobs = search_jobs()
    assert len(jobs) == len({job.name for job in jobs}) == 18
    assert [job.name for job in jobs[:6]] == [
        'M1_140201_random', 'M1_140201_evolution', 'M1_140202_evolution',
        'M1_140202_random', 'M1_140203_random', 'M1_140203_evolution']
    assert jobs[6].strategy.value == 'evolution'
    assert len({x for m in MODELS for x in asdict(m).values()}) == 15
    for m in MODELS:
        assert derived_seed('order', m.train_order, 0) != derived_seed('augmentation', m.augmentation, 0)
        assert derived_seed('order', m.train_order, 1) != derived_seed('order', m.train_order, 0)


@given(st.lists(st.integers(min_value=0, max_value=100), min_size=0, max_size=330))
def test_charging_matches_a_small_reference_model_with_duplicate_and_failure_inputs(proposals):
    budget = AttemptBudget(100.)
    expected_seen, expected_proposed = [], 0
    for number in proposals:
        identity = str(number)
        before = budget
        budget, result = budget.reserve(identity, 1.)
        if len(expected_seen) >= 32 or expected_proposed >= 256:
            assert result is Reservation.STOPPED and budget == before
            continue
        expected_proposed += 1
        if identity in expected_seen:
            assert result is Reservation.DUPLICATE
        else:
            expected_seen.append(identity)
            assert result is Reservation.CHARGED
        # No refund operation exists, regardless of success/failure afterward.
        assert budget.proposed == expected_proposed
        assert budget.identities == tuple(expected_seen)


def test_deadline_and_proposal_cap_do_not_become_success_count_limits():
    budget = AttemptBudget(2.)
    for _ in range(256):
        budget, _ = budget.reserve('same failed candidate', 1.)
    assert len(budget.identities) == 1 and budget.proposed == 256
    assert budget.reserve('next', 1.)[1] is Reservation.STOPPED
    assert AttemptBudget(2.).reserve('first', 2.)[1] is Reservation.STOPPED
    with pytest.raises(ValueError):
        AttemptBudget(float('nan'))


def test_roles_are_stratified_disjoint_complete_and_rng_isolated():
    labels = np.repeat(np.arange(10), 5000)
    before = np.random.get_state()
    roles = split_roles(labels)
    assert roles == split_roles(labels)
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    groups = (roles.train, roles.calibration, roles.search_validation, roles.report_validation)
    assert [len(group) for group in groups] == [40000, 1000, 4000, 5000]
    assert sorted(x for group in groups for x in group) == list(range(50000))
    for group, count in zip(groups, (4000, 100, 400, 500)):
        assert np.array_equal(np.bincount(labels[list(group)]), [count]*10)
    labels[0] = 1
    with pytest.raises(ValueError, match='5000'):
        split_roles(labels)


def test_recovery_tape_contains_all_crop_parameters_and_is_shared_within_one_M0():
    roles = split_roles(np.repeat(np.arange(10), 5000))
    tape = recovery_tape(roles.train, MODELS[0])
    assert tape == recovery_tape(roles.train, MODELS[0])
    assert digest(tape_payload(tape)) != digest(tape_payload(recovery_tape(roles.train, MODELS[1])))
    assert len(tape) == 200 and all(len(batch.row_ids) == 128 for batch in tape)
    ids = [row for batch in tape for row in batch.row_ids]
    assert len(set(ids)) == 25600 and set(ids) <= set(roles.train)
    assert all(len(batch.offsets) == len(batch.flips) == 128 for batch in tape)


def test_finalist_excludes_incomplete_recovery_late_oversized_and_nonfinite_outcomes():
    base = dict(status='succeeded', completed_within_deadline=True, quality_accepted=True,
                recovery_updates=200, tensor_bytes=80, cross_entropy=1., solution_id='b')
    rows = [base, dict(base, solution_id='a'), dict(base, solution_id='c', tensor_bytes=70),
            dict(base, solution_id='late', cross_entropy=.01, completed_within_deadline=False),
            dict(base, solution_id='short', cross_entropy=.01, recovery_updates=199),
            dict(base, solution_id='big', tensor_bytes=81), dict(base, cross_entropy=float('nan'))]
    assert [r['solution_id'] for r in eligible_finalists(rows, byte_ceiling=80)] == ['c', 'a', 'b']


def make_seal(*, complete=True):
    key, sha = 'a'*64, 'b'*64
    outcomes = tuple(SearchOutcome(job.name, 'completed' if complete else 'not_started',
        32 if complete else 0, key if complete else None, complete) for job in search_jobs())
    roles = [(f'{m.name}_{role}', key) for m in MODELS for role in
             ('dense_original', 'dense_recovery', 'fixed_prune', 'fixed_svd', 'fixed_hybrid')]
    if complete:
        roles.extend((f'{job.name}_finalist', key) for job in search_jobs())
    return GlobalSeal(IndependentProtocol().sha256, 'c'*64, 'd'*64, outcomes, tuple(roles), ((key, sha),))


def test_global_seal_requires_all_eighteen_outcomes_and_all_fifteen_fixed_roles():
    seal = make_seal()
    assert len(seal.roles) == 33 and len(seal.artifacts) == 1  # roles survive deduplication
    with pytest.raises(ValueError, match='entire'):
        replace(seal, outcomes=seal.outcomes[:-1])
    with pytest.raises(ValueError, match='entire'):
        replace(seal, roles=seal.roles[1:])
    with pytest.raises(ValueError, match='entire'):
        replace(seal, artifacts=(('e'*64, 'b'*64),))
    assert make_seal(complete=False).outcomes[-1].status == 'not_started'


def test_primary_panel_needs_every_model_every_seed_Q32_and_both_finalists():
    outcomes = list(make_seal().outcomes)
    assert primary_aggregate(outcomes)['panel_complete']
    outcomes[0] = replace(outcomes[0], status='partial', charged=31, complete_q32=False)
    result = primary_aggregate(outcomes)
    assert result['model_complete'] == {'M1': False, 'M2': True, 'M3': True}
    assert not result['panel_complete']
    outcomes = list(make_seal().outcomes)
    outcomes[-1] = replace(outcomes[-1], finalist=None)
    assert not primary_aggregate(outcomes)['panel_complete']


@pytest.mark.parametrize('failed', range(5))
def test_AB_final_admission_requires_all_controls_and_completed_whole_window(failed):
    values = dict(initial_dense=True, own=True, dense_pre=True, dense_post=True, window_completed=True)
    assert timing_comparison_admitted(**values)
    values[list(values)[failed]] = False
    assert not timing_comparison_admitted(**values)
