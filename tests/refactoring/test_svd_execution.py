"""Public profile, mathematical oracle, real persistence and process boundaries."""
import copy
import json
import sys
from types import SimpleNamespace
import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.execution import transform_weighted, WeightedProfileError
from fedcore.algorithm.low_rank.plans import MetricPolicy
from fedcore.algorithm.low_rank.topology import TopologyError
from fedcore.external_runtime.contracts import (CompressionRequest, ContractError, DataRoles,
    InputSpec, Resources, WeightedOptions)


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(8)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('kind', ['linear', 'conv1', 'conv2', 'grouped', 'reflect'])
@pytest.mark.parametrize('representation', ['one_layer', 'two_layers', 'three_layers'])
def test_full_rank_copy_persistence_export(kind, representation, tmp_path):
    from fedcore.tools.registry.checkpoint_manager import CheckpointManager
    from fedcore.tools.export import export_model
    if kind == 'linear':
        model, x, rank = nn.Linear(5, 3).double(), torch.randn(5, 5, dtype=torch.double), 3
    elif kind == 'conv1':
        model, x, rank = nn.Conv1d(2, 3, 3, padding=2, dilation=2).double(), torch.randn(5, 2, 9, dtype=torch.double), 3
    else:
        groups = 2 if kind == 'grouped' else 1
        model = nn.Conv2d(4, 6, (3, 2), stride=(2, 1), padding=(1, 1), groups=groups,
                          padding_mode='reflect' if kind == 'reflect' else 'zeros').double()
        x, rank = torch.randn(5, 4, 8, 7, dtype=torch.double), 6 // groups
    model.train()
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    result = transform_weighted(model, x, rank=rank, representation=representation, batch_size=2)
    assert torch.equal(rng, torch.get_rng_state()) and model.training and result.model.training
    assert not model._forward_pre_hooks and not result.model._forward_pre_hooks
    torch.testing.assert_close(result.model(x), model(x), atol=1e-11, rtol=1e-11)
    for key, tensor in before.items():
        torch.testing.assert_close(model.state_dict()[key], tensor, rtol=0, atol=0)
    assert result.plan.request.statistics_refs
    assert result.evidence['layers'][0]['requested_rank'] == rank
    assert result.evidence['parameters_after'] == sum(p.numel() for p in result.model.parameters())
    json.dumps(result.evidence, allow_nan=False)
    manager = CheckpointManager(str(tmp_path), auto_cleanup=False)
    path = tmp_path / 'checkpoint.pt'
    manager.save_to_file(manager.serialize_to_bytes(result.model), str(path))
    restored = manager.load_from_file(str(path))
    torch.testing.assert_close(restored(x), result.model(x), atol=1e-11, rtol=1e-11)
    artifact = export_model(restored.eval(), 'torchscript', tmp_path / 'export.pt', x[:1])
    # File objects avoid a known old TorchScript Windows Unicode-path bug.
    with artifact.open('rb') as stream:
        loaded = torch.jit.load(stream)
    torch.testing.assert_close(loaded(x[:1]), restored(x[:1]))


@pytest.mark.parametrize('groups', [1, 2])
def test_truncated_conv_matches_independent_output_oracle(groups):
    from fedcore.experiments.math_checks import activation_rows, layer_matrices, weighted_svd
    model = nn.Conv2d(4, 6, (3, 2), padding=(1, 1), groups=groups).double()
    x = torch.randn(7, 4, 8, 6, dtype=torch.double)
    actual = transform_weighted(model, x, rank=1, batch_size=3)
    matrices = layer_matrices(model)
    rows = activation_rows(model, x)
    # The oracle stores [groups, observations, features] and grouped operators.
    if isinstance(rows, torch.Tensor) and rows.ndim == 2:
        rows = rows.unsqueeze(0)
    if matrices.ndim == 2:
        matrices = matrices.unsqueeze(0)
    operator = actual.model.factor_matrix()
    if operator.ndim == 2:
        operator = operator.unsqueeze(0)
    for index, (matrix, observations) in enumerate(zip(matrices, rows)):
        moment = observations.T @ observations / len(observations)
        expected = weighted_svd(matrix, moment, 1)['approximation']
        torch.testing.assert_close(operator[index], expected, atol=1e-10, rtol=1e-10)


def test_actual_factor_operator_not_trained_s_and_passport_invalidates():
    from fedcore.models.network_impl.decomposed_layers import DecomposedLinear
    model, x = DecomposedLinear(nn.Linear(5, 4).double()), torch.randn(9, 5, dtype=torch.double)
    with torch.no_grad():
        model.U.mul_(7)
        model.Vh.div_(7)
        model.S.add_(.3)
    result = transform_weighted(model, x, rank=4)
    torch.testing.assert_close(result.model(x), model(x), atol=1e-11, rtol=1e-11)
    with torch.no_grad():
        model.U.add_(.1)
    changed = transform_weighted(model, x, rank=4)
    assert changed.evidence['checkpoint_id'] != result.evidence['checkpoint_id']
    new_data = transform_weighted(model, x + 1, rank=4)
    assert changed.evidence['data_id'] != new_data.evidence['data_id']


def test_nonstate_predecessor_configuration_invalidates_snapshot():
    model = nn.Sequential(nn.LeakyReLU(.1), nn.Linear(4, 3)).double()
    x = torch.randn(9, 4, dtype=torch.double)
    old = transform_weighted(model, x, rank=1)
    model[0].negative_slope = .9
    new = transform_weighted(model, x, rank=1)
    assert old.evidence['checkpoint_id'] != new.evidence['checkpoint_id']
    assert old.plan.request.statistics_refs != new.plan.request.statistics_refs


class AliasModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = self.a
    def forward(self, x):
        return self.a(x) + self.b(2 * x)


def test_module_alias_calls_pooled_once_without_parameter_cloning():
    model, x = AliasModel(), torch.randn(5, 4)
    result = transform_weighted(model, x, rank=2)
    assert result.model.a is result.model.b and model.a is model.b
    assert len(result.evidence['layers']) == 1
    assert result.evidence['layers'][0]['statistics'][0]['count'] == 2 * len(x)
    assert result.evidence['parameters_after'] == sum(p.numel() for p in result.model.parameters())


def test_tied_parameter_refused_and_failure_leaves_caller_untouched():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    model[1].weight = model[0].weight
    parameter = model[0].weight
    with pytest.raises(TopologyError, match='pooled'):
        transform_weighted(model, torch.randn(3, 4), rank=2)
    assert model[1].weight is model[0].weight is parameter
    assert not hasattr(model, '_fedcore_requires_optimizer_rebuild')


def test_petra_rejects_untouched_storage_views_before_copy(monkeypatch):
    from fedcore.experiments import CandidateSpec, ExperimentProtocol
    from fedcore.experiments import runner
    model = nn.Sequential(nn.Linear(4, 4))
    storage = torch.randn(4, 4)
    model.register_parameter('a', nn.Parameter(storage))
    model.register_parameter('b', nn.Parameter(storage.T))
    def forbidden(*args, **kwargs):
        pytest.fail('Shared views must be rejected before candidate copying')
    monkeypatch.setattr(runner.copy, 'deepcopy', forbidden)
    with pytest.raises(runner.UnsupportedOperation, match='storage views'):
        runner.apply_candidate(model, None, ExperimentProtocol(), CandidateSpec('weighted_svd', {'rank': 1}))
    assert model.a.untyped_storage()._cdata == model.b.untyped_storage()._cdata


def test_resource_preflight_before_copy_or_hook(monkeypatch):
    import fedcore.algorithm.low_rank.execution as execution
    model, x = nn.Linear(32, 16), torch.randn(2, 32)
    def forbidden(*args, **kwargs):
        pytest.fail('Allocation must be refused before deepcopy')
    monkeypatch.setattr(execution, 'deepcopy', forbidden)
    monkeypatch.setattr(torch, 'isfinite', forbidden)
    with pytest.raises(WeightedProfileError, match='ResourceLimitExceeded'):
        transform_weighted(model, x, rank=2, max_workspace_bytes=1024, max_peak_bytes=2048)
    assert not model._forward_pre_hooks


class FailingModel(nn.Sequential):
    def forward(self, x):
        super().forward(x)
        raise RuntimeError('intentional forward failure')


def test_failed_forward_cleans_work_hook_and_original(monkeypatch):
    import fedcore.algorithm.low_rank.execution as execution
    model = FailingModel(nn.Linear(4, 3))
    cloned = copy.deepcopy(model)
    monkeypatch.setattr(execution, 'deepcopy', lambda _: cloned)
    with pytest.raises(RuntimeError, match='intentional'):
        transform_weighted(model, torch.randn(3, 4), rank=2)
    assert not model[0]._forward_pre_hooks and not cloned[0]._forward_pre_hooks
    assert not hasattr(model, '_fedcore_requires_optimizer_rebuild')


def request(**kwargs):
    return CompressionRequest('model.fcb', 'example.fcb', InputSpec((1, 4)),
        DataRoles('validation.fcb', calibration='calibration.fcb'), **kwargs)


def test_external_v1_wire_unchanged_v2_roundtrip_and_bad_versions():
    old = request()
    assert 'weighted' not in old.to_dict()
    assert CompressionRequest.parse(old.to_dict()) == old
    new = request(method='weighted_svd', version=2, weighted=WeightedOptions(), rank=2)
    assert CompressionRequest.parse(new.to_dict()) == new
    for version in (0, 3, True):
        with pytest.raises(ContractError, match='version'):
            request(version=version)
    with pytest.raises(ContractError):
        request(method='weighted_svd', version=1, weighted=WeightedOptions(), rank=1)
    with pytest.raises(ContractError, match='calibration'):
        CompressionRequest('m.fcb', 'x.fcb', InputSpec((1, 4)), DataRoles('v.fcb'),
            method='weighted_svd', version=2, weighted=WeightedOptions(), rank=1)
    for kwargs in ({'ridge': float('nan')}, {'ridge': True}, {'rcond': -1}, {'method_version': 2}):
        with pytest.raises(ContractError):
            WeightedOptions(**kwargs)


def test_weighted_job_cancel_timeout_and_failed_execution_cleanup(tmp_path):
    from fedcore.external_runtime.client import save_dataset
    from fedcore.external_runtime.models import save_model_bundle
    from fedcore.external_runtime.security import safe_save
    from fedcore.external_runtime.jobs import JobStore, JobRunner
    source = tmp_path / 'source'
    source.mkdir()
    model, x = nn.Linear(4, 3), torch.randn(3, 4)
    save_model_bundle(model, source / 'model.fcb')
    safe_save(x[:1], source / 'example.fcb')
    save_dataset(source / 'validation.fcb', x, model(x).detach())
    save_dataset(source / 'calibration.fcb', x + 1, model(x + 1).detach())
    store = JobStore(tmp_path / 'jobs')
    runner = JobRunner(store, workers=1)
    try:
        job_request = request(method='weighted_svd', version=2, weighted=WeightedOptions(), rank=1)
        first = runner.submit(job_request, source)
        second = runner.submit(job_request, source)
        runner.cancel(second)
        runner.cancel(first)
        assert runner.wait(first)['state'] == runner.wait(second)['state'] == 'cancelled'
        assert not list(store.directory(first).glob('compressed.*'))
        timed = request(method='weighted_svd', version=2, weighted=WeightedOptions(), rank=1,
                        resources=Resources(timeout_seconds=.01))
        timeout = runner.wait(runner.submit(timed, source), 10)
        assert timeout['state'] == 'failed' and timeout['result']['error']['code'] == 'timeout'
        invalid = request(method='weighted_svd', version=2, weighted=WeightedOptions(), rank=999)
        failed_id = runner.submit(invalid, source)
        failed = runner.wait(failed_id, 120)
        assert failed['state'] == 'failed'
        assert not list(store.directory(failed_id).glob('compressed.*'))
    finally:
        runner.close()


def test_low_rank_facade_updates_root_and_trainer_ownership(tmp_path, monkeypatch):
    from fedcore.algorithm.low_rank.low_rank_opt import LowRankModel
    model, x = nn.Linear(4, 3), torch.randn(6, 4)
    monkeypatch.setenv('FEDCORE_MODEL_REGISTRY_PATH', str(tmp_path / 'registry'))
    facade = LowRankModel({'device': 'cpu'})
    facade.trainer = SimpleNamespace(model=model)
    result = facade.compress_weighted(model, x, rank=2)
    assert facade.model_after is facade.trainer.model is result.model
    assert facade.trainer._fedcore_requires_optimizer_rebuild
    optimizer = torch.optim.SGD(result.model.parameters(), lr=.01)
    loss = result.model(x).square().mean()
    loss.backward()
    optimizer.step()
    assert all(p.grad is not None for p in result.model.parameters())


def test_petra_uses_separate_calibration_and_current_candidate_runner():
    from fedcore.experiments.protocol import ExperimentBundle, ExperimentProtocol, CandidateSpec, TensorSplit
    from fedcore.experiments.runner import apply_candidate
    model = nn.Linear(4, 3)
    splits = []
    for role in ('train', 'validation', 'calibration', 'test'):
        x = torch.randn(6, 4)
        splits.append(TensorSplit(x, model(x).detach(), tuple(f'{role}-{i}' for i in range(6))))
    bundle = ExperimentBundle(*splits, 'regression', model)
    result, evidence = apply_candidate(model, bundle, ExperimentProtocol(finetune_epochs=0),
                                       CandidateSpec('weighted_svd', {'rank': 2}))
    expected = transform_weighted(model, bundle.calibration.x, rank=2)
    torch.testing.assert_close(result(bundle.validation.x), expected.model(bundle.validation.x))
    assert evidence['weighted_transform']['data_id'] == expected.evidence['data_id']
    bundle.test.verify_integrity()


def test_petra_trains_common_baseline_and_freezes_selection_before_test(tmp_path):
    from dataclasses import replace
    from fedcore.experiments import ExperimentRunner, ExperimentProtocol, CandidateSpec
    from tests.petra.test_protocol_runner import bundle
    source = bundle()
    settings = ExperimentProtocol(baseline_epochs=1, finetune_epochs=0, batch_size=12,
        measurement_repeats=1, warmup=1, quality_tolerance=1.0, threads=2)
    candidates = (CandidateSpec('weighted_svd', {'rank': 1}), CandidateSpec('baseline'))
    one = ExperimentRunner(source, settings, tmp_path / 'one').run(candidates)
    changed = replace(source, test=replace(source.test, y=1 - source.test.y))
    two = ExperimentRunner(changed, settings, tmp_path / 'two').run(candidates)
    assert all(row['status'] == 'succeeded' for row in one['candidates']), one['candidates']
    assert len({row['source_model_sha256'] for row in one['candidates']}) == 1
    assert one['test_gate']['selection_role'] == 'validation'
    assert one['selection']['archive_ids'] == two['selection']['archive_ids']
    assert [row['validation'] for row in one['candidates']] == [row['validation'] for row in two['candidates']]


@pytest.mark.parametrize('kind', ['linear', 'conv'])
def test_real_weighted_worker_matches_local_export(kind, tmp_path):
    from fedcore.external_runtime.client import compress
    model = nn.Linear(4, 3).eval() if kind == 'linear' else nn.Conv1d(2, 3, 3, padding=1).eval()
    calibration = torch.randn(9, 4) if kind == 'linear' else torch.randn(9, 2, 8)
    validation = torch.randn_like(calibration)
    with torch.no_grad():
        targets, calibration_targets = model(validation), model(calibration)
    local = transform_weighted(model, calibration, rank=1)
    result = compress(model, validation[:1], (validation, targets), jobs_root=tmp_path / 'jobs',
        method='weighted_svd', rank=1, max_relative_error=1e6, calibration=(calibration, calibration_targets),
        weighted_options=WeightedOptions(batch_size=3), resources=Resources(threads=2, repetitions=1),
        python_executable=sys.executable)
    assert result['status'] == 'succeeded', result
    assert result['request_version'] == 2 and result['weighted_transform']['method_version'] == 1
    assert result['parameters']['layers'][0]['requested_rank'] == 1
    from pathlib import Path
    with (Path(result['job_directory']) / result['artifact']).open('rb') as stream:
        loaded = torch.jit.load(stream)
    torch.testing.assert_close(loaded(validation[:1]), local.model(validation[:1]), atol=1e-5, rtol=1e-5)
    assert result['metrics']['compressed']['parameters'] == local.evidence['parameters_after']
    json.dumps(result, allow_nan=False)


def test_unsupported_request_before_job_registration(tmp_path):
    from fedcore.external_runtime.client import compress
    model, x = nn.Linear(4, 3), torch.randn(3, 4)
    with pytest.raises(ContractError):
        compress(model, x[:1], (x, model(x).detach()), jobs_root=tmp_path / 'jobs',
                 method='weighted_svd', rank=1)
    assert not (tmp_path / 'jobs').exists()
    with pytest.raises(WeightedProfileError, match='InvalidRank'):
        compress(model, x[:1], (x, model(x).detach()), jobs_root=tmp_path / 'jobs',
                 method='weighted_svd', rank=100, calibration=(x, model(x).detach()))
    assert not (tmp_path / 'jobs').exists()
