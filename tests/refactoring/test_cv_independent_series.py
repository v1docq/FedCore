"""CPU boundaries and one complete, tiny numerical recovery without extra BN."""
import builtins
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fedcore.experiments import cv_independent_series as series
from fedcore.experiments.cv_independent_protocol import IndependentProtocol, MODELS, digest
from fedcore.experiments.cv_independent_data import RecoveryBatch, recovery_tape, tape_payload


def test_noexecute_checks_source_and_data_integrity_without_ML_or_output(tmp_path, monkeypatch, capsys):
    from fedcore.experiments import cv_sparse_runtime
    data = tmp_path/'data'/'cifar-10-batches-py'
    data.mkdir(parents=True)
    monkeypatch.setattr(series, 'OFFICIAL_MD5', {'test_batch': hashlib.md5(b'bytes').hexdigest()})
    (data/'test_batch').write_bytes(b'bytes')
    files = {f'fedcore/experiments/{name}.py': 'a'*64 for name in series.OWN_MODULES}
    files['fedcore/interfaces/fedcore_dispatcher.py'] = 'a'*64
    monkeypatch.setattr(cv_sparse_runtime, 'source_manifest', lambda *a: {
        'manifest_sha256': 'a'*64, 'verified_package_files_sha256': files})
    original = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name == 'torch' or name.startswith(('torch.', 'torchvision', 'fedcore.algorithm')):
            pytest.fail('Dry-run crossed ML/CUDA boundary')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', checked_import)
    output = tmp_path/'unused_output'
    assert series.main(['--data-root', str(tmp_path/'data'), '--output-root', str(output),
                        '--source-manifest', 'stub.json']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['execute'] is False and result['status'] == 'prepared'
    assert result['data_files']['test_batch']['sha256'] == hashlib.sha256(b'bytes').hexdigest()
    assert not output.exists()


def test_noexecute_rejects_corrupted_official_data_before_creating_output(tmp_path, monkeypatch):
    from fedcore.experiments import cv_sparse_runtime
    data = tmp_path/'cifar-10-batches-py'
    data.mkdir()
    (data/'test_batch').write_bytes(b'corrupt')
    monkeypatch.setattr(cv_sparse_runtime, 'source_manifest', lambda *a: {
        'manifest_sha256': 'a'*64, 'verified_package_files_sha256': {
            **{f'fedcore/experiments/{n}.py': 'a'*64 for n in series.OWN_MODULES},
            'fedcore/interfaces/fedcore_dispatcher.py': 'a'*64}})
    with pytest.raises(ValueError, match='integrity mismatch'):
        series.preparation(tmp_path, tmp_path/'new_output', 'stub.json')
    assert not (tmp_path/'new_output').exists()


@pytest.mark.parametrize('external,expected', [(None, 28890.), ('20000', 20000.), ('99999', 28890.)])
def test_external_deadline_never_extends_internal_termination_reserve(monkeypatch, external, expected):
    monkeypatch.delenv('FEDCORE_PILOT_ABSOLUTE_DEADLINE', raising=False)
    assert series._deadline(100., external) == expected
    monkeypatch.setenv('FEDCORE_PILOT_ABSOLUTE_DEADLINE', '500')
    assert series._deadline(100., external) == min(500., expected)


@pytest.mark.parametrize('external', ('nan', 'inf', '100', '-1'))
def test_invalid_deadline_is_not_a_request_for_unbounded_execution(monkeypatch, external):
    monkeypatch.delenv('FEDCORE_PILOT_ABSOLUTE_DEADLINE', raising=False)
    with pytest.raises(ValueError):
        series._deadline(100., external)


def test_recovery_runs_exactly_200_updates_with_fresh_optimizer_and_no_bn_reset(monkeypatch):
    import torch
    from fedcore.experiments.cv_independent_runtime import recover_no_extra_bn
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4),
                                    torch.nn.ReLU(), torch.nn.Linear(4, 2))
        x, y = torch.randn(128, 4), torch.arange(128) % 2
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    events, checks, optimizers = [], [], []
    bn = model[1]
    bn.running_mean.fill_(.25)
    monkeypatch.setattr(bn, 'reset_running_stats', lambda: pytest.fail('Unexpected extra BN reset'))
    real_sgd = torch.optim.SGD

    def optimizer(*args, **kwargs):
        optimizers.append(kwargs)
        return real_sgd(*args, **kwargs)

    monkeypatch.setattr(torch.optim, 'SGD', optimizer)
    runtime = SimpleNamespace(protocol=IndependentProtocol(), check=lambda *a, **kw: checks.append(kw),
                              output=SimpleNamespace(append=events.append))
    # A tiny CPU capability satisfies the same effect boundary as CUDA data.
    gpu = SimpleNamespace(runtime=runtime, torch=torch, training_tape=lambda batch: (x, y))
    original_to = model.to
    monkeypatch.setattr(model, 'to', lambda device: original_to('cpu'))
    batch = RecoveryBatch(tuple(range(128)), ((0, 0),)*128, (False,)*128)
    proof = recover_no_extra_bn(model, gpu, (batch,)*200, deadline=999., role='tiny')
    assert proof['updates'] == 200 and proof['extra_bn_passes'] == 0
    assert proof['bn_mode'] == 'inherited_state_ordinary_SGD_updates_only'
    assert proof['tape_sha256'] == digest(tape_payload((batch,)*200))
    assert len(optimizers) == 1 and optimizers[0] == {
        'lr': .001, 'momentum': .9, 'weight_decay': .0005, 'nesterov': True}
    assert len(events) == 200 and events[-1]['update'] == 200
    assert int(bn.num_batches_tracked) == 200 and not model.training
    assert len(checks) >= 600
    assert all(not torch.equal(value, initial[name]) for name, value in model.named_parameters())


def test_incomplete_tape_is_refused_before_model_training_or_optimizer_creation():
    from fedcore.experiments.cv_independent_runtime import recover_no_extra_bn
    gpu = SimpleNamespace(runtime=SimpleNamespace(protocol=IndependentProtocol()), torch=None)
    with pytest.raises(ValueError, match='200'):
        recover_no_extra_bn(None, gpu, (), deadline=1., role='bad')


def test_report_data_gate_rejects_before_torchvision_or_test_dataset_read(monkeypatch, tmp_path):
    from fedcore.experiments.cv_independent_data import IndependentCifarData
    data = IndependentCifarData.__new__(IndependentCifarData)
    data.protocol = IndependentProtocol()
    with pytest.raises(ValueError, match='global seal'):
        data.open_report(None)
    with pytest.raises(ValueError, match='global seal'):
        data.open_test(None)


def test_reconcile_hard_kill_keeps_common_compile_charge_SGD_and_partial_rows(tmp_path):
    events = [
        {'event': 'evolution_trace', 'trace': {'event': 'evaluation_started', 'canonical_solution_id': 'a'*64,
            'attempt_id': 1, 'genotype': {}, 'started_at': 1.}},
        {'event': 'evolution_trace', 'trace': {'event': 'evaluated', 'canonical_solution_id': 'a'*64,
            'attempt_id': 1, 'genotype': {}, 'started_at': 1., 'ended_at': 2.,
            'completed_within_deadline': True, 'failure': 'compile rejected'}},
        {'event': 'evolution_trace', 'trace': {'event': 'evaluation_started', 'canonical_solution_id': 'b'*64,
            'attempt_id': 2, 'genotype': {}, 'started_at': 2.}},
        {'event': 'candidate_charged', 'solution_id': 'b'*64, 'attempt_id': 1, 'genotype': {}, 'started_at': 2.},
        {'event': 'recovery_update', 'role': 'b'*64, 'update': 17},
        {'event': 'evaluation_batch', 'label': 'Vsearch', 'completed_rows': 256},
    ]
    (tmp_path/'ledger.jsonl').write_text('\n'.join(json.dumps(row) for row in events)+'\n', encoding='utf-8')
    result = series.reconcile_worker_output(tmp_path, 'search')
    assert result['charged'] == 2 and result['confirmed_recovery_updates'] == 17
    assert result['confirmed_evaluation_rows'] == {'Vsearch': 256}
    assert [r['attempt_id'] for r in result['attempts']] == [1, 2]
    assert result['attempts'][1]['status'] == 'incomplete'
    assert result['attempts'][1]['confirmed_sgd_updates'] == 17
    assert not result['complete_q32']


def test_reconcile_keeps_quality_prefix_when_worker_dies_before_terminal_result(tmp_path):
    (tmp_path/'quality_partial.json').write_text(json.dumps({'key': {'eager': {'rows': 5000}}}), encoding='utf-8')
    result = series.reconcile_worker_output(tmp_path, 'quality')
    assert result['status'] == 'partial' and result['quality']['key']['eager']['rows'] == 5000


def test_reconcile_never_revives_late_attempt_from_earlier_successful_ledger_event(tmp_path):
    key = 'a'*64
    base = {'solution_id': key, 'attempt_id': 1, 'status': 'succeeded',
        'completed_within_deadline': True, 'quality_accepted': True, 'recovery_updates': 200,
        'cross_entropy': .1, 'tensor_bytes': 50, 'checkpoint': {'solution_id': key}}
    (tmp_path/'attempts.json').write_text(json.dumps([
        {**base, 'status': 'late', 'completed_within_deadline': False}]), encoding='utf-8')
    events = [{'event': 'candidate_charged', 'solution_id': key, 'attempt_id': 1},
              {'event': 'candidate_ended', **base},
              {'event': 'evolution_trace', 'trace': {'event': 'evaluated', 'canonical_solution_id': key,
               'attempt_id': 1, 'completed_within_deadline': False, 'failure_details': {'code': 'EvaluationDeadline'}}}]
    (tmp_path/'ledger.jsonl').write_text('\n'.join(json.dumps(row) for row in events)+'\n', encoding='utf-8')
    (tmp_path/'result.json').write_text(json.dumps({'status': 'completed', 'byte_ceiling': 80,
        'charged': 1, 'finalist': base['checkpoint']}), encoding='utf-8')
    result = series.reconcile_worker_output(tmp_path, 'search')
    assert result['attempts'][0]['status'] == 'late'
    assert result['attempts'][0]['completed_within_deadline'] is False
    assert result['charged'] == 1 and result['finalist'] is None


def raw_session(*, aa=True, ratio=1., pair_count=20):
    batches = {}
    for size in ('1', '128'):
        pairs = [{'pair': i, 'order': 'AB' if i % 2 == 0 else 'BA',
                  'A': {'cuda_event_seconds': 1., 'synchronized_wall_seconds': 1.},
                  'B': {'cuda_event_seconds': ratio, 'synchronized_wall_seconds': ratio}}
                 for i in range(pair_count)]
        batches[size] = {'pairs': pairs, 'orders': [p['order'] for p in pairs], 'accepted': True}
    return {'aa': aa, 'batches': batches, 'accepted': True}


def timing_window():
    return {'status': 'completed', 'window_completed': True, 'models': {'M1': {
        'status': 'completed', 'initial_dense_AA': raw_session(), 'dense_pre_AA': raw_session(),
        'dense_post_AA': raw_session(), 'candidates': {'key': {'own_AA': raw_session(),
            'AB': raw_session(aa=False, ratio=.9), 'comparison_admitted': True}}}}}


@pytest.mark.parametrize('exitcode,late', [(9, False), (0, True)])
def test_terminal_process_failure_revokes_even_previous_completed_AB_admission(tmp_path, exitcode, late):
    (tmp_path/'result.json').write_text(json.dumps(timing_window()), encoding='utf-8')
    result = series.reconcile_worker_output(tmp_path, 'timing', process_returncode=exitcode, force_partial=late)
    assert result['status'] != 'completed' and result['window_completed'] is False
    assert result['models']['M1']['candidates']['key']['comparison_admitted'] is False


@pytest.mark.parametrize('mutation', ['missing_AB', 'short_AB', 'bad_raw_AA'])
def test_terminal_timing_recomputes_raw_gates_and_complete_AB_instead_of_cached_booleans(tmp_path, mutation):
    window = timing_window()
    row = window['models']['M1']['candidates']['key']
    if mutation == 'missing_AB':
        row.pop('AB')
    elif mutation == 'short_AB':
        row['AB'] = raw_session(aa=False, pair_count=19)
    else:
        row['own_AA'] = raw_session(ratio=1.2)
    (tmp_path/'result.json').write_text(json.dumps(window), encoding='utf-8')
    result = series.reconcile_worker_output(tmp_path, 'timing', process_returncode=0)
    assert result['models']['M1']['candidates']['key']['comparison_admitted'] is False


def test_evaluation_timeout_preserves_latest_NPZ_prefix_and_durable_batch_counter(tmp_path):
    import torch
    import numpy as np
    from fedcore.experiments.cv_independent_runtime import evaluate
    from fedcore.experiments.cv_sparse_runtime import Output, PilotStop
    output = Output(tmp_path/'worker')
    calls = []

    def check(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 3:
            raise PilotStop('search', 'absolute_deadline')

    gpu = SimpleNamespace(torch=torch, images=torch.ones(256, 4), labels=torch.zeros(256, dtype=torch.long),
        normalized=lambda x: x, runtime=SimpleNamespace(check=check, output=output))
    with pytest.raises(PilotStop):
        evaluate(torch.nn.Linear(4, 2), gpu, tuple(range(256)), deadline=1., label='partial')
    with np.load(output.root/'predictions/partial.npz') as rows:
        assert np.array_equal(rows['row_ids'], np.arange(128))
        assert len(rows['cross_entropy']) == 128
    progress = json.loads((output.root/'prediction_progress/partial.json').read_text())
    assert progress['status'] == 'partial' and progress['completed_rows'] == 128
    result = series.reconcile_worker_output(output.root, 'search')
    assert result['confirmed_evaluation_rows'] == {'partial': 128}


def test_real_timing_worker_uses_raw_AA_and_keeps_AB_ineligible_without_all_twenty_pairs(tmp_path, monkeypatch):
    from fedcore.experiments import cv_independent_runtime, cv_sparse_runtime
    seal = SimpleNamespace(roles=tuple((f'M{i}_dense_recovery', f'dense{i}') for i in (1, 2, 3)))
    keys = [f'dense{i}' for i in (1, 2, 3)] + [f'compact{i}' for i in (1, 2, 3)]
    quality = {key: {'export_eligible': True, 'torchscript': {'path': key+'.pt', 'sha256': 'a'*64}} for key in keys}
    (tmp_path/'quality.json').write_text(json.dumps({'quality': quality}), encoding='utf-8')
    (tmp_path/'timing_order.json').write_text(json.dumps({'0': {f'M{i}': [f'compact{i}'] for i in (1, 2, 3)}}), encoding='utf-8')
    for key in keys:
        (tmp_path/(key+'.pt')).write_bytes(b'fixed-ts')
    monkeypatch.setattr(series, '_sealed_inputs', lambda *a: (seal, {}))
    monkeypatch.setattr(cv_sparse_runtime, 'sha256_file', lambda *a: 'a'*64)
    monkeypatch.setattr(cv_independent_runtime, 'IndependentGPUData', lambda *a: object())
    sessions = []

    def paired(first, second, gpu, **kwargs):
        sessions.append(kwargs['label'])
        return raw_session(aa=kwargs.get('aa', False), pair_count=20 if kwargs.get('aa') else 19)

    monkeypatch.setattr(cv_sparse_runtime, 'paired_latency', paired)
    runtime = SimpleNamespace(protocol=IndependentProtocol(), check=lambda *a, **kw: None,
        torch=SimpleNamespace(jit=SimpleNamespace(load=lambda *a, **kw: SimpleNamespace(eval=lambda: object()))),
        output=SimpleNamespace(json=lambda *a: None))
    result = series._timing_worker(tmp_path, runtime, object(), 999., 0)
    assert len(sessions) == 15  # 3 M0 x initial, own, dense-pre, AB, dense-post
    assert result['window_completed']
    assert all(not row['candidates'][f'compact{i}']['comparison_admitted']
               for i, row in enumerate(result['models'].values(), 1))


def test_controller_first_pair_failure_stops_other_sixteen_searches_without_retries(tmp_path, monkeypatch):
    from fedcore.experiments.cv_independent_protocol import search_jobs
    output = tmp_path/'series'
    monkeypatch.setattr(series, 'preparation', lambda *a: {'status': 'prepared', 'source': {}, 'source_sha256': 'f'*64})
    monkeypatch.setattr(series.shutil, 'disk_usage', lambda *a: SimpleNamespace(free=200*1024**3))
    monkeypatch.setattr(series, '_artifact', lambda *a: tmp_path/'placeholder.pt')
    calls = []

    def artifact(label):
        return {'artifact_id': digest(['artifact', label]), 'sha256': digest(['checkpoint', label]),
            'path': label+'.pt', 'file_bytes': 1, 'tensor_bytes': 100,
            'completed_within_deadline': True, 'solution_id': digest(['genotype', label])}

    class Process:
        def __init__(self, command, **kwargs):
            self.returncode = 0
            options = {command[i]: command[i+1] for i in range(3, len(command)-1)
                       if command[i].startswith('--') and command[i] != '--execute'}
            kind, job = options['--worker'], options['--job-id']
            calls.append((kind, job))
            child_output = Path(options['--output-root'])
            child_output.mkdir(parents=True)
            result = {'status': 'completed'}
            if kind == 'manifest':
                result['identity'] = 'e'*64
            elif kind == 'baseline':
                result['baseline'] = {**artifact(job), 'training': {'initial_fingerprint': digest(job)}}
            elif kind == 'controls':
                result['controls'] = {role: artifact(job+role) for role in
                    ('dense_recovery', 'fixed_prune', 'fixed_svd', 'fixed_hybrid')}
            elif kind == 'search':
                number = len([call for call in calls if call[0] == 'search'])
                result.update(charged=31 if number == 1 else 32, complete_q32=number != 1, finalist=None)
            elif kind == 'quality':
                result['status'] = 'partial'
            else:
                pytest.fail('Unexpected worker after the partial quality stop')
            (child_output/'result.json').write_text(json.dumps(result), encoding='utf-8')

        def wait(self, **kwargs):
            return self.returncode

    monkeypatch.setattr(series.subprocess, 'Popen', Process)
    result = series.run_series(tmp_path/'data', output, tmp_path/'source.json')
    assert [job for kind, job in calls if kind == 'search'] == [job.name for job in search_jobs()[:2]]
    assert result['feasibility_gate'] == 'failed_first_pair_Q32'
    assert [row['charged'] for row in result['outcomes'][:2]] == [31, 32]
    assert all(row['status'] == 'not_started' for row in result['outcomes'][2:])


def durable_context():
    from fedcore.experiments.cv_independent_protocol import search_jobs
    return series.search_context(search_jobs()[0], source_tensor_bytes=100,
        source_m0_fingerprint='a'*64, data_sha256='b'*64, tape_sha256='c'*64,
        source_sha256='d'*64, deadline=100., process_started_at=1.)


def test_hard_kill_context_restores_ceiling_and_keeps_valid_partial_finalist_after_file_admission(tmp_path):
    from fedcore.experiments.cv_sparse_runtime import sha256_file
    checkpoint_path = tmp_path/'candidate.pt'
    checkpoint_path.write_bytes(b'independent checkpoint')
    key = 'e'*64
    checkpoint = {'path': checkpoint_path.name, 'sha256': sha256_file(checkpoint_path),
        'file_bytes': checkpoint_path.stat().st_size, 'tensor_bytes': 70,
        'completed_within_deadline': True, 'solution_id': key, 'artifact_id': 'f'*64}
    attempt = {'attempt_id': 1, 'solution_id': key, 'status': 'succeeded', 'cross_entropy': .2,
        'tensor_bytes': 70, 'quality_accepted': True, 'completed_within_deadline': True,
        'recovery_updates': 200, 'checkpoint': checkpoint}
    (tmp_path/'search_context.json').write_text(json.dumps(durable_context()), encoding='utf-8')
    (tmp_path/'attempts.json').write_text(json.dumps([attempt]), encoding='utf-8')
    (tmp_path/'ledger.jsonl').write_text(json.dumps({
        'event': 'candidate_charged', 'attempt_id': 1, 'solution_id': key})+'\n', encoding='utf-8')
    result = series.reconcile_worker_output(tmp_path, 'search')
    assert result['status'] == 'partial' and result['charged'] == 1
    assert result['source_tensor_bytes'] == 100 and result['byte_ceiling'] == 80
    assert result['finalist'] == checkpoint and not result['complete_q32']
    assert result['finalist_admission'] == 'pending_controller_fresh_file_check'
    assert series._artifact(tmp_path, result['finalist']) == checkpoint_path
    checkpoint_path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='fresh file admission'):
        series._artifact(tmp_path, result['finalist'])


def test_reconcile_context_rejects_conflicting_terminal_budget_or_tampered_identity(tmp_path):
    context = durable_context()
    path = tmp_path/'search_context.json'
    path.write_text(json.dumps(context), encoding='utf-8')
    (tmp_path/'result.json').write_text(json.dumps({'status': 'partial', 'byte_ceiling': 90}), encoding='utf-8')
    with pytest.raises(ValueError, match='differs from its durable context'):
        series.reconcile_worker_output(tmp_path, 'search')
    (tmp_path/'result.json').unlink()
    context['context']['tape_sha256'] = 'f'*64
    path.write_text(json.dumps(context), encoding='utf-8')
    with pytest.raises(ValueError, match='durable independent search context'):
        series.reconcile_worker_output(tmp_path, 'search')


def test_cost_accounting_partitions_wall_time_without_adding_nested_engine_phases():
    events = [
        {'event': 'search_engine_started', 'timestamp': 2.},
        {'event': 'search_engine_ready', 'timestamp': 5.},
        {'event': 'search_phase', 'phase': 'optimizer_call', 'started_at': 5., 'completed_at': 18.},
        {'event': 'search_phase', 'phase': 'objective_component', 'started_at': 6., 'completed_at': 10.},
        {'event': 'search_phase', 'phase': 'candidate_compilation', 'started_at': 6., 'completed_at': 7.},
        {'event': 'search_phase', 'phase': 'numerical_callback', 'callback_id': 1,
            'started_at': 7., 'completed_at': 10.},
        {'event': 'search_phase', 'phase': 'numerical_callback', 'callback_id': 2,
            'started_at': 12., 'completed_at': 16.},
        {'event': 'search_engine_ended', 'timestamp': 18.},
    ]
    cost = series.search_cost_accounting(events, process_started_at=0., process_completed_at=20.)
    assert cost['process_wall_seconds'] == 20.
    assert cost['engine_wall_seconds'] == 16.
    assert cost['engine_import_setup_seconds'] == 3.
    assert cost['callback_sum_seconds'] == cost['callback_union_seconds'] == 7.
    assert cost['engine_residual_seconds'] == 6.
    assert cost['process_outside_engine_seconds'] == 4.
    assert sum(cost[key] for key in ('engine_import_setup_seconds', 'callback_union_seconds',
        'engine_residual_seconds', 'process_outside_engine_seconds')) == cost['process_wall_seconds']


def test_cost_accounting_uses_union_for_overlap_and_retains_open_killed_callback():
    events = [
        {'event': 'search_engine_started', 'timestamp': 1.},
        {'event': 'search_engine_ready', 'timestamp': 3.},
        {'event': 'search_phase', 'phase': 'numerical_callback', 'callback_id': 1,
            'started_at': 4., 'completed_at': 8.},
        {'event': 'search_phase', 'phase': 'numerical_callback', 'callback_id': 2,
            'started_at': 6., 'completed_at': 9.},
        {'event': 'search_callback_started', 'callback_id': 3, 'timestamp': 10.},
    ]
    cost = series.search_cost_accounting(events, process_started_at=0., process_completed_at=12.)
    assert cost['engine_incomplete'] and cost['open_callbacks'] == 1
    assert cost['callback_sum_seconds'] == 9.
    assert cost['callback_union_seconds'] == 7.
    assert cost['engine_import_setup_seconds'] == 2.
    assert cost['engine_residual_seconds'] == 2.
    assert cost['process_outside_engine_seconds'] == 1.


def test_random_wiring_records_setup_proposals_and_exactly_one_phase_per_actual_callback(monkeypatch):
    import time
    from fedcore.experiments.cv_independent_runtime import IndependentEvaluator
    from fedcore.experiments import cv_sparse_pilot
    from fedcore.algorithm.low_rank import cv_grammar
    events = []
    evaluator = IndependentEvaluator.__new__(IndependentEvaluator)
    evaluator.output = SimpleNamespace(append=events.append)
    evaluator.runtime = SimpleNamespace(check=lambda *a, **kw: None)
    evaluator.records, evaluator.space = [], object()
    evaluator.prefix = tuple(f'prefix{i}' for i in range(4))
    counter = iter(range(100))
    monkeypatch.setattr(cv_sparse_pilot, 'solution_id', lambda genotype: genotype)
    monkeypatch.setattr(cv_grammar, 'cv_genotype_payload', lambda genotype: {'value': genotype})
    monkeypatch.setattr(cv_grammar, 'generate_cv_genotype', lambda *a, **kw: f'proposal{next(counter)}')
    evaluator.evaluate = lambda *a: evaluator.records.append({'status': 'fake'})
    result = evaluator.random_search(140201, time.perf_counter()+100.)
    assert result['charged_evaluations'] == 32
    assert len([e for e in events if e['event'] == 'search_callback_started']) == 32
    assert len([e for e in events if e['event'] == 'search_phase' and e['phase'] == 'numerical_callback']) == 32
    assert len([e for e in events if e['event'] == 'search_phase' and e['phase'] == 'proposal_graph_creation']) == 32
    assert [e['event'] for e in events[:2]] == ['search_engine_started', 'search_engine_ready']
    assert events[-1]['event'] == 'search_engine_ended'


def test_evolution_wiring_connects_common_phase_callback_and_import_ready_event(monkeypatch):
    import torch
    import time
    from fedcore.experiments.cv_independent_runtime import IndependentEvaluator
    from fedcore.algorithm.low_rank import grammar_search
    from fedcore.algorithm.low_rank.candidate_storage import capture_inventory, inventory_fingerprint
    events = []
    evaluator = IndependentEvaluator.__new__(IndependentEvaluator)
    evaluator.output = SimpleNamespace(append=events.append, json=lambda *a: None)
    evaluator.records, evaluator.space, evaluator.prefix = [], object(), ()
    evaluator.data = SimpleNamespace(identity='a'*64)
    evaluator.teacher = torch.nn.Linear(2, 2).eval().requires_grad_(False)
    evaluator.teacher_fingerprint = inventory_fingerprint(capture_inventory(evaluator.teacher))

    def run(*args, **kwargs):
        assert kwargs['context_guard']() is None
        now = time.perf_counter()
        kwargs['trace_callback']({'event': 'started', 'timestamp': now})
        kwargs['phase_callback']({'phase': 'optimizer_call', 'started_at': now, 'completed_at': now+1.})
        return {'status': 'succeeded', 'trace': [], 'best_genotypes': []}

    monkeypatch.setattr(grammar_search, 'run_cv_grammar_search', run)
    evaluator.evolution(140201, time.perf_counter()+100.)
    assert any(e['event'] == 'search_engine_ready' and e['strategy'] == 'evolution' for e in events)
    phase = next(e for e in events if e['event'] == 'search_phase')
    assert phase['phase'] == 'optimizer_call' and phase['seconds'] == 1.
    assert events[-1]['event'] == 'search_engine_ended'
