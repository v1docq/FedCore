"""State/data provenance, safe replay, and independent consumer ownership."""
from dataclasses import replace
import json

import pytest
import torch

from fedcore.experiments import CandidateSpec, ExperimentRunner, ProtocolError, prepare_baseline
from fedcore.experiments.runner import model_state_hash
from tests.petra.test_protocol_runner import bundle, protocol, single_cpu_thread


def test_checkpoint_reuse_does_not_train_and_returns_independent_state(tmp_path, monkeypatch):
    source, settings = bundle(), protocol()
    initial = model_state_hash(source.original_model)
    trained, record = prepare_baseline(source, settings, tmp_path / 'common')
    assert record['status'] == 'trained_on_train' and record['training_steps'] == 1
    assert record['state_sha256'] != initial
    assert model_state_hash(source.original_model) == initial
    import fedcore.experiments.runner as runtime
    monkeypatch.setattr(runtime, 'train_model', lambda *_args, **_kwargs: pytest.fail('Checkpoint was retrained'))
    left, reused = prepare_baseline(source, settings, tmp_path / 'left', checkpoint=record['checkpoint'])
    right, second = prepare_baseline(source, replace(settings, finetune_epochs=3, measurement_repeats=5),
                                     tmp_path / 'right', checkpoint=record['checkpoint'])
    assert reused['reused'] and second['reused']
    assert reused['wall_seconds'] == record['wall_seconds']
    assert reused['actual_work_seconds'] >= 0
    assert model_state_hash(left) == model_state_hash(right) == model_state_hash(trained)
    with torch.no_grad():
        next(left.parameters()).add_(1)
    assert model_state_hash(right) == model_state_hash(trained)
    assert model_state_hash(source.original_model) == initial


@pytest.mark.parametrize('change', ['initial', 'train', 'validation', 'calibration', 'seed', 'epochs', 'batch', 'lr', 'environment'])
def test_checkpoint_provenance_rejects_relevant_changes(tmp_path, change):
    source, settings = bundle(), protocol()
    _, record = prepare_baseline(source, settings, tmp_path / 'common', environment={'version': 'one'})
    environment = {'version': 'one'}
    if change == 'initial':
        with torch.no_grad():
            next(source.original_model.parameters()).add_(1)
    elif change in ('train', 'validation', 'calibration'):
        split = getattr(source, change)
        source = replace(source, **{change: replace(split, x=split.x + .1)})
    elif change == 'environment':
        environment = {'version': 'two'}
    else:
        key, value = {'seed': ('seed', 7), 'epochs': ('baseline_epochs', 2),
                      'batch': ('batch_size', 6), 'lr': ('learning_rate', .01)}[change]
        settings = replace(settings, **{key: value})
    with pytest.raises(ProtocolError, match='provenance'):
        prepare_baseline(source, settings, tmp_path / 'attempt', checkpoint=record['checkpoint'], environment=environment)


def test_changing_test_does_not_affect_training_checkpoint_identity(tmp_path):
    _, record = prepare_baseline(bundle(), protocol(), tmp_path / 'common')
    _, reused = prepare_baseline(bundle(test_flip=True), protocol(), tmp_path / 'reuse', checkpoint=record['checkpoint'])
    assert reused['state_sha256'] == record['state_sha256']


def test_untrained_state_cannot_be_reused_as_trained(tmp_path):
    settings = replace(protocol(), baseline_epochs=0)
    _, record = prepare_baseline(bundle(), settings, tmp_path / 'provided')
    assert record['status'] == 'provided_state; pretraining_not_verified_by_this_helper'
    assert record['training_steps'] == 0
    with pytest.raises(ProtocolError, match='positive training history'):
        prepare_baseline(bundle(), settings, tmp_path / 'reuse', checkpoint=record['checkpoint'])


@pytest.mark.parametrize('tamper', ['state_bytes', 'state_hash', 'steps', 'loss'])
def test_checkpoint_integrity_and_history_tampering_are_rejected(tmp_path, tamper):
    _, record = prepare_baseline(bundle(), protocol(), tmp_path / 'common')
    path = tmp_path / 'common' / 'baseline.json'
    value = json.loads(path.read_text())
    if tamper == 'state_bytes':
        (tmp_path / 'common' / 'baseline_state.pt').write_bytes(b'bad')
    elif tamper == 'state_hash':
        value['state']['model_sha256'] = 'bad'
    elif tamper == 'steps':
        value['training']['training_steps'] = 0
    else:
        value['training']['train_loss'] = ['unverified']
    path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ProtocolError, match='hash mismatch|positive training history'):
        prepare_baseline(bundle(), protocol(), tmp_path / 'reuse', checkpoint=path)


def test_runner_and_ablation_reuse_actual_common_checkpoint(tmp_path):
    from fedcore.experiments.ablations import run_rank_ablation
    source, settings = bundle(), protocol()
    _, common = prepare_baseline(source, settings, tmp_path / 'common')
    first = ExperimentRunner(source, settings, tmp_path / 'run', baseline_checkpoint=common['checkpoint'])
    manifest = first.run((CandidateSpec('train', {'epochs': 1}),))
    assert manifest['baseline_training']['reused']
    assert manifest['baseline_training']['state_sha256'] == common['state_sha256']
    assert len({row['source_model_sha256'] for row in manifest['candidates']}) == 1
    assert manifest['baseline_cost_accounting']['charged_common_baseline_seconds'] == common['wall_seconds']
    assert manifest['baseline_cost_accounting']['charged_training_seconds'] == common['training_seconds']
    assert {'baseline.json', 'baseline_state.pt'} <= set(manifest['replay_files'])
    report = run_rank_ablation(source, settings, tmp_path / 'ablation', layer_path='0', ranks=(1,),
                              baseline_checkpoint=common['checkpoint'])
    assert report['baseline_training']['reused']
    assert report['baseline_training']['state_sha256'] == common['state_sha256']


def test_loading_uses_weights_only(tmp_path, monkeypatch):
    _, record = prepare_baseline(bundle(), protocol(), tmp_path / 'common')
    original, values = torch.load, []
    def observed(*args, **kwargs):
        values.append(kwargs.get('weights_only'))
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, 'load', observed)
    prepare_baseline(bundle(), protocol(), tmp_path / 'reuse', checkpoint=record['checkpoint'])
    assert values == [True]
