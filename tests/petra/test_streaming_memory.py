"""Independent dense references for reductions and file-backed tensor replay."""
from dataclasses import replace
import hashlib

import numpy as np
import pytest
import torch
from torch import nn

from fedcore.experiments.ablations import _quality
from fedcore.experiments.math_checks import activation_rows, approximate_layer, relative_error, second_moment
from fedcore.experiments.protocol import ProtocolError, TensorSplit, canonical_json, role_content_identity, tensor_hash
from fedcore.experiments.streaming import ActivationMoments, capture_moments, layer_error
from tests.petra.test_protocol_runner import bundle, protocol, model_factory


@pytest.mark.parametrize('task,features', [('classification', 3), ('regression', 3), ('forecasting', 5)])
@pytest.mark.parametrize('batch_size', [1, 4, 12])
def test_batched_quality_matches_full_reference_with_unequal_last_batch(task, features, batch_size):
    torch.manual_seed(7)
    model = nn.Linear(features, 3).eval()
    x = torch.randn(11, features)
    y = torch.arange(11) % 3 if task == 'classification' else torch.randn(11, 3)
    split = TensorSplit(x, y, tuple(map(str, range(11))))
    full = model(x).detach()
    expected = float((full.argmax(-1) == y).double().mean()) if task == 'classification' else float((full.double() - y).square().mean())
    sizes = []
    handle = model.register_forward_pre_hook(lambda _model, values: sizes.append(len(values[0])))
    result = _quality(model, split, task, batch_size)
    handle.remove()
    assert result['samples'] == 11
    assert result['value'] == pytest.approx(expected, abs=2e-7)
    assert max(sizes) <= batch_size


@pytest.mark.parametrize('layer,x', [
    (nn.Linear(4, 3, dtype=torch.float64), torch.randn(7, 5, 4, dtype=torch.float64)),
    (nn.Conv1d(4, 6, 3, groups=2, padding=1, dtype=torch.float64), torch.randn(7, 4, 9, dtype=torch.float64)),
    (nn.Conv2d(4, 6, 3, groups=2, padding=1, dtype=torch.float64), torch.randn(7, 4, 8, 9, dtype=torch.float64)),
])
def test_streamed_moments_and_weighted_factors_match_dense_reference(layer, x):
    accumulator = ActivationMoments(layer, row_chunk_size=3)
    for batch in x.split(2):
        accumulator.add(batch)
    moments, memory = accumulator.finish()
    for expected, actual in zip(activation_rows(layer, x), moments):
        torch.testing.assert_close(actual, second_moment(expected), rtol=1e-12, atol=1e-12)
    dense, _ = approximate_layer(layer, x, 1, weighted=True)
    streaming, _ = approximate_layer(layer, None, 1, weighted=True, calibration_moments=moments)
    torch.testing.assert_close(dense(x), streaming(x), rtol=1e-10, atol=1e-10)
    assert memory['forwards'] == 4
    assert memory['moment_tensor_bytes'] <= memory['workspace_limit_bytes']


def test_capture_and_layer_error_use_selected_layer_inputs_and_remove_hooks():
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    split = TensorSplit(torch.randn(11, 3), torch.zeros(11, 2), tuple(map(str, range(11))))
    layer = model[2]
    expected_inputs = model[:2](split.x).detach()
    moments, _ = capture_moments(model, layer, split, 3)
    torch.testing.assert_close(moments[0], second_moment(expected_inputs), rtol=1e-6, atol=1e-8)
    replacement, _ = approximate_layer(layer, None, 1)
    expected = relative_error(layer(expected_inputs), replacement(expected_inputs))
    actual = layer_error(model, layer, replacement, split, 3)
    assert actual['relative'] == pytest.approx(expected['relative'], abs=1e-6)
    assert not layer._forward_hooks and not layer._forward_pre_hooks
    with pytest.raises(ValueError, match='exceeds'):
        capture_moments(model, layer, split, 3, max_bytes=1)
    assert not layer._forward_pre_hooks


def test_wide_moment_and_single_image_patch_limit_fail_before_unfold(monkeypatch):
    with pytest.raises(ValueError, match='second moment'):
        ActivationMoments(nn.Linear(128, 2), max_bytes=128)
    accumulator = ActivationMoments(nn.Conv2d(3, 2, 3), max_bytes=100_000, row_chunk_size=1)
    monkeypatch.setattr('fedcore.experiments.streaming.activation_rows', lambda *_: pytest.fail('Too large patch allocated'))
    with pytest.raises(ValueError, match='patch workspace'):
        accumulator.add(torch.zeros(1, 3, 128, 128))


def test_tensor_hash_preserves_original_bytes_for_noncontiguous_and_scalar_tensors():
    for value in (torch.arange(60).reshape(3, 4, 5).transpose(1, 2), torch.tensor(2.), torch.empty(0, 3)):
        descriptor = canonical_json({'shape': list(value.shape), 'dtype': str(value.dtype)})
        digest = hashlib.sha256(descriptor.encode())
        digest.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        assert tensor_hash(value) == digest.hexdigest()


def test_file_backed_split_is_independent_and_replays_without_dense_copy(tmp_path):
    from fedcore.experiments.runner import ExperimentRunner
    source = bundle()
    replacements = {}
    for role in ('train', 'validation', 'calibration', 'test'):
        split = getattr(source, role)
        xp, yp = tmp_path / f'{role}-x.npy', tmp_path / f'{role}-y.npy'
        np.save(xp, split.x.numpy()); np.save(yp, split.y.numpy())
        replacements[role] = TensorSplit.from_npy(xp, yp, split.ids)
    source = replace(source, **replacements)
    ordinary = bundle()
    assert role_content_identity(source.manifest()['roles']) == role_content_identity(ordinary.manifest()['roles'])
    changed = replace(source.train, x=source.train.x + 1)
    assert not changed._backing
    torch.testing.assert_close(changed.x, source.train.x + 1)
    source.train.x[0, 0] += 2
    assert np.load(tmp_path / 'train-x.npy')[0, 0] != source.train.x[0, 0]
    with pytest.raises(ProtocolError, match='modified in place'):
        source.manifest()
    source = replace(source, train=TensorSplit.from_npy(tmp_path / 'train-x.npy', tmp_path / 'train-y.npy', source.train.ids))
    result = ExperimentRunner(source, protocol(), tmp_path / 'run').run(())
    data = torch.load(tmp_path / 'run' / 'data.pt', weights_only=True)
    assert set(data['train']) == {'files'}
    replay = ExperimentRunner.from_manifest(tmp_path / 'run', model_factory, tmp_path / 'replay')
    assert replay.bundle.train._backing
    torch.testing.assert_close(replay.bundle.train.x, source.train.x)
    assert replay.bundle.train.ids == source.train.ids
    assert result['data']['roles']['train']['storage']['kind'] == 'npy_copy_on_write'
    # Windows disallows truncating an actively mapped file; an in-place edit is
    # also the integrity threat that the replay hash must detect.
    with (tmp_path / 'run' / 'train-x.npy').open('r+b') as stream:
        stream.seek(-1, 2)
        stream.write(b'!')
    with pytest.raises(ProtocolError, match='hash mismatch'):
        ExperimentRunner.from_manifest(tmp_path / 'run', model_factory, tmp_path / 'bad-replay')


def test_cv_builder_bounds_loading_and_explicitly_requires_disk_for_large_tensors(tmp_path, monkeypatch):
    from torchvision import datasets, models
    from fedcore.experiments.scenarios import ScenarioUnavailable, build_cv_dataset
    class Images:
        targets = list(range(10)) * 10
        classes = list(map(str, range(10)))
        class_to_idx = {name: index for index, name in enumerate(classes)}
        reads = 0
        def __init__(self, *_args, **_kwargs):
            pass
        def __len__(self):
            return len(self.targets)
        def __getitem__(self, index):
            Images.reads += 1
            return torch.full((3, 16, 16), index / 100.), self.targets[index]
    monkeypatch.setattr(datasets, 'CIFAR10', Images)
    monkeypatch.setattr(models, 'resnet18', lambda **_: nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 10)))
    with pytest.raises(ScenarioUnavailable, match='storage_dir'):
        build_cv_dataset('cifar10', tmp_path, resolution=16, max_materialized_bytes=1)
    assert Images.reads == 0
    result = build_cv_dataset('cifar10', tmp_path, resolution=16, storage_dir=tmp_path / 'backing', max_materialized_bytes=1)
    assert Images.reads == 200
    assert result.train._backing and result.test._backing
    assert sum(len(getattr(result, role).ids) for role in ('train', 'validation', 'calibration', 'test')) == 200
    assert result.metadata['tensor_storage'] == 'npy_copy_on_write'
