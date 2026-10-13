"""Public execution, physical checkpoints and versioned external boundaries."""
import copy
from dataclasses import FrozenInstanceError
import io

import pytest
import torch
from torch import nn

from fedcore.algorithm.pruning.sparse.execution import plan_sparse, transform_sparse
from fedcore.algorithm.pruning.sparse.layers import PackedNMLinear, IndexedResidualLinear
from fedcore.algorithm.pruning.sparse.specs import HASSLEFreeSpec, NMStructure, SparseDomainError
from fedcore.tools.registry.checkpoint_manager import CheckpointError, CheckpointManager
from fedcore.external_runtime.contracts import (CompressionRequest, InputSpec, DataRoles,
    SparseOptions, ContractError, Resources, plan_request)


def test_public_pruner_profile_boundary():
    from fedcore.algorithm.pruning.pruners import BasePruner
    source = nn.Linear(8, 4)
    result = BasePruner.compress_profile(source, 'hassle_free', spec(), calibration=torch.randn(12, 8))
    assert type(result.model) is PackedNMLinear
    assert result.evidence['method'] == 'hassle_free'
    assert type(source) is nn.Linear


def test_runner_losparse_uses_training_callback_and_finalizes():
    from types import SimpleNamespace
    from fedcore.experiments.protocol import TensorSplit, CandidateSpec
    from fedcore.experiments.runner import apply_candidate
    source = nn.Linear(8, 4)
    x = torch.randn(4, 8)
    split = TensorSplit(x, source(x).detach(), ('a', 'b', 'c', 'd'))
    bundle = SimpleNamespace(train=split, calibration=split, task='regression')
    protocol = SimpleNamespace(batch_size=4, learning_rate=1e-3, seed=7, device='cpu')
    candidate = CandidateSpec('losparse', {'spec': {'rank': 1, 'residual_rows': 2, 'total_steps': 2}})
    compact, evidence = apply_candidate(source, bundle, protocol, candidate)
    assert type(compact) is IndexedResidualLinear
    assert compact.residual_count == 2
    assert evidence['method'] == 'losparse'


def spec():
    return HASSLEFreeSpec(structure=NMStructure(rank=1), outer_steps=1, inner_steps=1)


def test_execution_is_independent_and_plan_contains_only_values():
    torch.manual_seed(2)
    original = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 3)).double()
    snapshot = copy.deepcopy(original.state_dict())
    x = torch.randn(12, 8, dtype=torch.float64)
    plan = plan_sparse(original, 'hassle_free', spec())
    with pytest.raises(FrozenInstanceError):
        plan.method = 'other'
    result = transform_sparse(original, 'hassle_free', spec(), calibration=x)
    assert result.model is not original
    assert type(result.model[0]) is PackedNMLinear
    assert type(original[0]) is nn.Linear
    assert not any(isinstance(value, (nn.Module, torch.Tensor)) for value in vars(plan).values())
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, snapshot[key], rtol=0, atol=0)


def test_tedepr_refuses_before_model_or_data_access():
    class Explosive:
        def __getattribute__(self, name):
            raise AssertionError('No data/model traversal permitted')
    with pytest.raises(SparseDomainError, match='TEDEPR') as error:
        transform_sparse(Explosive(), 'tedepr', Explosive(), calibration=Explosive())
    assert error.value.code == 'insufficient_specification'


@pytest.mark.parametrize('kind', ['packed', 'indexed'])
def test_physical_checkpoint_roundtrip_without_factory(kind, tmp_path):
    layer = PackedNMLinear(8, 5, rank=1, dtype=torch.float64) if kind == 'packed' else IndexedResidualLinear(8, 5, 1, 2, dtype=torch.float64)
    x = torch.randn(3, 8, dtype=torch.float64)
    manager = CheckpointManager(tmp_path)
    data = manager.serialize_to_bytes(nn.Sequential(layer))
    restored = manager.deserialize_from_bytes(data)
    torch.testing.assert_close(restored(x), layer(x), rtol=0, atol=0)
    assert restored[0].representation_config() == layer.representation_config()
    assert not hasattr(restored[0], 'weight')
    payload = torch.load(io.BytesIO(data), weights_only=True)
    key = '0.indices' if kind == 'packed' else '0.row_indices'
    payload['state_dict'][key].fill_(99)
    with pytest.raises(CheckpointError):
        manager.restore(payload)


def test_failed_preflight_does_not_collect_or_mutate():
    model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 3))
    state = copy.deepcopy(model.state_dict())
    with pytest.raises(SparseDomainError):
        transform_sparse(model, 'hassle_free', spec(), target_paths=('0', 'missing'))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)


def request(**changes):
    values = dict(model='model.fcb', example='example.fcb', input_spec=InputSpec((2, 8)),
        data=DataRoles('validation.fcb', calibration='calibration.fcb'),
        version=4, method='hassle_free', sparse_profile=SparseOptions(spec()),
        max_relative_error=10, resources=Resources(repetitions=1))
    values.update(changes)
    return CompressionRequest(**values)


def test_external_v4_roundtrip_and_strict_allowlist():
    value = request()
    assert CompressionRequest.parse(value.to_dict()) == value
    assert 'hassle_free' in plan_request(value).steps
    for changes in ({'version': 3}, {'artifact_format': 'onnx'}, {'rank': 1}, {'method': 'export'}):
        with pytest.raises(ContractError):
            request(**changes)
    with pytest.raises(ContractError):
        SparseOptions.parse('hassle_free', {'spec': {'invented': True}})


def test_moe_recursive_checkpoint_preserves_factorized_experts(tmp_path):
    from fedcore.algorithm.pruning.sparse.moe import ReferenceMoE, remove_experts, compress_experts
    from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear
    source = ReferenceMoE(4, 6, 3, top_k=1)
    x = torch.randn(12, 4)
    compact, _ = remove_experts(source, (0, 2))
    compact, _ = compress_experts(compact, x, (1, 1))
    manager = CheckpointManager(tmp_path)
    saved = manager.serialize_to_bytes(compact)
    restored = manager.deserialize_from_bytes(saved)
    assert type(restored.experts[0].up_proj) is FactorizedLinear
    assert restored.expert_ids.tolist() == [0, 2]
    torch.testing.assert_close(restored(x), compact(x), rtol=0, atol=0)


def test_hassle_sequential_targets_observe_current_predecessors(monkeypatch):
    from fedcore.algorithm.pruning.sparse import linear
    source = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 3))
    x = torch.randn(12, 8)
    original = linear.hassle_free
    records = []
    def record(layer, inputs, options):
        result = original(layer, inputs, options)
        records.append((inputs.clone(), result[0]))
        return result
    monkeypatch.setattr(linear, 'hassle_free', record)
    transform_sparse(source, 'hassle_free', spec(), calibration=x)
    torch.testing.assert_close(records[1][0], records[0][1](x))


def test_external_workflow_executes_and_measures_artifact(tmp_path):
    from fedcore.external_runtime.models import save_model_bundle
    from fedcore.external_runtime.security import safe_save
    from fedcore.external_runtime.execution import execute_plan
    torch.manual_seed(9)
    model = nn.Linear(8, 4)
    save_model_bundle(model, tmp_path / 'model.fcb')
    safe_save(torch.randn(2, 8), tmp_path / 'example.fcb')
    for role in ('validation', 'calibration'):
        x = torch.randn(12, 8)
        safe_save({'kind': 'fedcore_tensor_dataset', 'version': 1,
            'features': x, 'targets': model(x).detach()}, tmp_path / (role+'.fcb'))
    from fedcore.tools.export import ExportError
    try:
        result = execute_plan(plan_request(request()), tmp_path)
    except ExportError as error:
        pytest.fail(str(error.causes))
    assert result['request_version'] == 4
    assert result['artifact_measurement']['status'] == 'succeeded'
    assert result['artifact_measurement']['metrics']['tensor_state_bytes'] == result['metrics']['compressed']['tensor_bytes']
    assert (tmp_path / result['artifact']).is_file()
