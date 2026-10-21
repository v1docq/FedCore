"""Low-rank and physical sparse contracts must coexist without implicit adapters."""
import copy
from dataclasses import asdict
import io
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.candidate_storage import capture_inventory, inventory_cost
from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear
from fedcore.algorithm.pruning.sparse.layers import IndexedResidualLinear, PackedNMLinear
from fedcore.algorithm.pruning.sparse.specs import HASSLEFreeSpec, NMStructure
from fedcore.tools.registry.checkpoint_manager import CheckpointError, CheckpointManager


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _inputs(rows=12):
    grid = torch.arange(rows * 8, dtype=torch.float64).reshape(rows, 8)
    return (grid * .17).sin() + (grid * .31).cos() + .2


def _fill_parameters(model):
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            grid = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape(parameter.shape)
            parameter.copy_((grid * .23 + index).sin() / 4)
    return model


def _source():
    return _fill_parameters(nn.Sequential(
        nn.Linear(8, 8, dtype=torch.float64), nn.Tanh(),
        nn.Linear(8, 4, dtype=torch.float64))).eval()


def _spec():
    return HASSLEFreeSpec(structure=NMStructure(n=1, m=4, rank=1),
                          outer_steps=1, inner_steps=1)


def _mixed(kind, *, aliases=True):
    low_rank = FactorizedLinear(8, 8, 1, dtype=torch.float64)
    if kind == 'packed':
        sparse = PackedNMLinear(8, 8, n=1, m=4, rank=1, dtype=torch.float64)
        with torch.no_grad():
            sparse.values.copy_(torch.arange(16, dtype=torch.float64).reshape(8, 2, 1) / 30)
    else:
        sparse = IndexedResidualLinear(8, 8, 1, 2, dtype=torch.float64)
    if aliases:
        sparse.factors.right = low_rank.right
    model = _fill_parameters(nn.Sequential(low_rank, nn.Tanh(), sparse,
                                           sparse if aliases else nn.Identity())).eval()
    model[0].left.weight.requires_grad_(False)
    model[1].train()
    return model


@pytest.mark.parametrize('kind', ['packed', 'indexed'])
def test_mixed_checkpoint_preserves_runtime_storage_and_cross_branch_aliases(tmp_path, kind):
    model, inputs = _mixed(kind), _inputs()
    inventory = capture_inventory(model)
    state_bytes = sum(value.numel() * value.element_size() for value in model.state_dict().values())
    assert inventory_cost(inventory, 'bytes') < state_bytes
    assert any(view.category == 'buffer' and view.dtype == 'torch.int64'
               for record in inventory for view in record.views)
    manager = CheckpointManager(str(tmp_path), auto_cleanup=False)
    checkpoint = manager.serialize_to_bytes(model)
    path = str(tmp_path / 'mixed.pt')
    manager.save_to_file(checkpoint, path)
    for restored in (manager.deserialize_from_bytes(checkpoint), manager.load_from_file(path)):
        assert type(restored[0]) is FactorizedLinear
        assert type(restored[2]) is type(model[2])
        assert restored[2] is restored[3]
        assert restored[0].right is restored[2].factors.right
        assert not hasattr(restored[2], 'weight')
        assert not restored[0].left.weight.requires_grad
        assert restored[1].training and not restored.training
        assert capture_inventory(restored) == inventory
        torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)


@pytest.mark.parametrize('kind', ['packed', 'indexed'])
@pytest.mark.parametrize('alias_kind', ['parameter', 'buffer'])
def test_mixed_checkpoint_rejects_conflicting_alias_state_without_mutating_source(tmp_path, kind, alias_kind):
    source = _mixed(kind)
    inventory = capture_inventory(source)
    manager = CheckpointManager(str(tmp_path), auto_cleanup=False)
    payload = torch.load(io.BytesIO(manager.serialize_to_bytes(source)), weights_only=True)
    if alias_kind == 'parameter':
        payload['state_dict']['2.factors.right.weight'].add_(1)
        message = 'tied Parameters'
    else:
        key = '3.indices' if kind == 'packed' else '3.row_indices'
        payload['state_dict'][key].bitwise_xor_(1)
        message = 'shared buffer'
    with pytest.raises(CheckpointError, match=message):
        manager.restore(payload, model=source)
    assert capture_inventory(source) == inventory


@pytest.mark.parametrize('kind', ['packed', 'indexed'])
def test_mixed_checkpoint_refuses_distinct_parameters_sharing_backing_storage(tmp_path, kind):
    model = _mixed(kind, aliases=False)
    model[0].right.weight = nn.Parameter(model[2].factors.right.weight.detach())
    with pytest.raises(CheckpointError, match='Shared storage views'):
        CheckpointManager(str(tmp_path), auto_cleanup=False).serialize_to_bytes(model)


def test_checkpoint_preserves_factorized_remapped_and_both_sparse_representations(tmp_path):
    from fedcore.algorithm.low_rank.dobi_remapping import RemappedLinear
    dense = _fill_parameters(nn.Linear(8, 8, dtype=torch.float64))
    remapped = RemappedLinear.from_weight(dense.weight.detach(), 2, dense.bias.detach(),
        tail_dtype='float64', scale_dtype='float64', compute_dtype='float64')
    model = _fill_parameters(nn.Sequential(
        FactorizedLinear(8, 8, 2, dtype=torch.float64), remapped,
        PackedNMLinear(8, 8, n=1, m=4, rank=1, dtype=torch.float64),
        IndexedResidualLinear(8, 4, 1, 2, dtype=torch.float64))).eval()
    with torch.no_grad():
        model[2].values.fill_(.2)
    inventory = capture_inventory(model)
    assert {'torch.int8', 'torch.int64', 'torch.float64'} <= {
        view.dtype for record in inventory for view in record.views}
    manager = CheckpointManager(str(tmp_path), auto_cleanup=False)
    restored = manager.deserialize_from_bytes(manager.serialize_to_bytes(model))
    assert tuple(type(layer) for layer in restored) == tuple(type(layer) for layer in model)
    assert capture_inventory(restored) == inventory
    torch.testing.assert_close(restored(_inputs()), model(_inputs()), rtol=0, atol=0)
    model[2].bias = model[1].bias
    with pytest.raises(CheckpointError, match='cannot alias|storage sharing'):
        manager.serialize_to_bytes(model)


@pytest.mark.parametrize('kind', ['packed', 'indexed'])
@pytest.mark.parametrize('persistent', [False, True])
def test_checkpoint_refuses_sparse_buffer_views_including_nonpersistent_state(tmp_path, kind, persistent):
    model = _mixed(kind, aliases=False)
    component = model[2].values if kind == 'packed' else model[2].residual_rows
    model.register_buffer('mirror', component.detach().reshape(-1), persistent=persistent)
    with pytest.raises(CheckpointError, match='Sparse checkpoint storage sharing'):
        CheckpointManager(str(tmp_path), auto_cleanup=False).serialize_to_bytes(model)


@pytest.mark.parametrize('kind', ['packed', 'indexed'])
@pytest.mark.parametrize('corruption', ['cross_branch_storage', 'padded_sparse_storage'])
def test_checkpoint_refuses_forged_sparse_storage_before_assign(tmp_path, kind, corruption):
    model = _mixed(kind, aliases=False)
    inventory = capture_inventory(model)
    manager = CheckpointManager(str(tmp_path), auto_cleanup=False)
    payload = torch.load(io.BytesIO(manager.serialize_to_bytes(model)), weights_only=True)
    key = '2.values' if kind == 'packed' else '2.residual_rows'
    component = payload['state_dict'][key]
    if corruption == 'cross_branch_storage':
        payload['state_dict']['0.right.weight'] = component.reshape(2, 8)[:1]
    else:
        padded = torch.empty(component.numel() + 1, dtype=component.dtype)
        padded[1:].copy_(component.reshape(-1))
        payload['state_dict'][key] = padded[1:].reshape_as(component)
    with pytest.raises(CheckpointError, match='Sparse checkpoint'):
        manager.restore(payload, model=model)
    assert capture_inventory(model) == inventory


@pytest.mark.parametrize('buffer_kind', ['packed', 'persistent', 'nonpersistent'])
def test_sparse_preflight_refuses_parameter_buffer_storage_before_copy_or_calibration(monkeypatch, buffer_kind):
    from fedcore.algorithm.pruning.pruners import BasePruner
    from fedcore.algorithm.pruning.sparse import execution
    from fedcore.algorithm.pruning.sparse.specs import SparseDomainError
    if buffer_kind == 'packed':
        packed = PackedNMLinear(8, 4, n=2, m=4, dtype=torch.float64)
        model = nn.Sequential(packed, nn.Linear(4, 4, dtype=torch.float64))
        model[1].weight = nn.Parameter(packed.values.reshape(4, 4))
        targets = ('1',)
    else:
        model = nn.Sequential(nn.Linear(8, 4, dtype=torch.float64))
        model.register_buffer('mirror', model[0].weight.detach(), persistent=buffer_kind == 'persistent')
        targets = ('0',)
    inventory, rng = capture_inventory(model), torch.random.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        pytest.fail('Unsupported storage topology reached copying or calibration')

    monkeypatch.setattr(execution, 'deepcopy', forbidden)
    monkeypatch.setattr(execution, '_inputs', forbidden)
    with pytest.raises(SparseDomainError) as error:
        BasePruner.compress_profile(model, 'hassle_free', _spec(),
                                    calibration=object(), target_paths=targets)
    assert error.value.code == 'unsupported_topology'
    assert capture_inventory(model) == inventory
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize('boundary', ['weighted_facade', 'profile_facade', 'runner'])
def test_low_rank_boundaries_refuse_mixed_parameter_buffer_storage_before_copy(tmp_path, monkeypatch, boundary):
    from fedcore.algorithm.low_rank import execution, method_execution
    from fedcore.algorithm.low_rank.low_rank_opt import LowRankModel
    from fedcore.algorithm.low_rank.method_specs import ASVD
    from fedcore.experiments import runner
    from fedcore.experiments.protocol import CandidateSpec
    monkeypatch.setenv('FEDCORE_MODEL_REGISTRY_PATH', str(tmp_path / 'registry'))
    packed = PackedNMLinear(8, 4, n=2, m=4, dtype=torch.float64)
    model = nn.Sequential(packed, nn.Linear(4, 4, dtype=torch.float64))
    model[1].weight = nn.Parameter(packed.values.reshape(4, 4))
    inventory = capture_inventory(model)
    facade = LowRankModel({'device': 'cpu'})
    inputs, rng = _inputs(), torch.random.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        pytest.fail('Unsupported mixed storage topology reached model copying')

    with monkeypatch.context() as guarded:
        guarded.setattr(execution, 'deepcopy', forbidden)
        guarded.setattr(method_execution, 'deepcopy', forbidden)
        if boundary == 'runner':
            guarded.setattr(runner.copy, 'deepcopy', forbidden)
        with pytest.raises((ValueError, runner.UnsupportedOperation), match='Sparse checkpoint storage sharing'):
            if boundary == 'weighted_facade':
                facade.compress_weighted(model, inputs, rank=1, target_paths=('1',))
            elif boundary == 'profile_facade':
                facade.compress_profile(model, inputs, ASVD(), rank=1, target_paths=('1',))
            else:
                runner.apply_candidate(model, object(), object(), CandidateSpec('baseline'))
    assert capture_inventory(model) == inventory
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize('sparse_first', [False, True])
def test_public_facades_compose_disjoint_scopes_without_mutating_the_other_branch(tmp_path, monkeypatch, sparse_first):
    from fedcore.algorithm.low_rank.low_rank_opt import LowRankModel
    from fedcore.algorithm.pruning.pruners import BasePruner
    from fedcore.models.network_impl.decomposed_layers import DecomposedLinear
    monkeypatch.setenv('FEDCORE_MODEL_REGISTRY_PATH', str(tmp_path / 'registry'))
    source, inputs = _source(), _inputs()
    original = capture_inventory(source)
    facade = LowRankModel({'device': 'cpu'})
    if sparse_first:
        sparse = BasePruner.compress_profile(source, 'hassle_free', _spec(),
                                            calibration=inputs, target_paths=('2',))
        previous = capture_inventory(sparse.model[2])
        result = facade.compress_weighted(sparse.model, inputs, rank=2, target_paths=('0',))
        assert capture_inventory(result.model[2]) == previous
    else:
        low_rank = facade.compress_weighted(source, inputs, rank=2, target_paths=('0',))
        previous = capture_inventory(low_rank.model[0])
        result = BasePruner.compress_profile(low_rank.model, 'hassle_free', _spec(),
                                            calibration=inputs, target_paths=('2',))
        assert capture_inventory(result.model[0]) == previous
    assert type(result.model[0]) is DecomposedLinear
    assert type(result.model[2]) is PackedNMLinear
    assert capture_inventory(source) == original
    manager = CheckpointManager(str(tmp_path), auto_cleanup=False)
    restored = manager.deserialize_from_bytes(manager.serialize_to_bytes(result.model))
    assert capture_inventory(restored) == capture_inventory(result.model)
    torch.testing.assert_close(restored(inputs), result.model(inputs), rtol=0, atol=0)


@pytest.mark.parametrize('low_rank_method', ['svd', 'weighted_svd', 'asvd'])
@pytest.mark.parametrize('sparse_first', [False, True])
def test_public_runner_dispatches_mixed_chains_in_declared_order(tmp_path, low_rank_method, sparse_first):
    from fedcore.experiments.protocol import CandidateSpec, TensorSplit
    from fedcore.experiments.runner import apply_candidate
    from fedcore.models.network_impl.decomposed_layers import DecomposedLinear
    source, inputs = _source(), _inputs()
    original = capture_inventory(source)
    split = TensorSplit(inputs, source(inputs).detach(), tuple(str(i) for i in range(len(inputs))))
    bundle = SimpleNamespace(train=split, calibration=split, task='regression')
    protocol = SimpleNamespace(batch_size=4, learning_rate=.001, seed=11,
                               device='cpu', finetune_epochs=0)
    low_rank = CandidateSpec(low_rank_method, {'rank': 2, 'target_paths': ['0'], 'finetune_epochs': 0})
    sparse = CandidateSpec('hassle_free', {'spec': asdict(_spec()), 'target_paths': ['2']})
    steps = (sparse, low_rank) if sparse_first else (low_rank, sparse)
    result, evidence = apply_candidate(source, bundle, protocol, CandidateSpec('chain', chain=steps))
    assert type(result[0]) is DecomposedLinear
    assert type(result[2]) is PackedNMLinear
    sparse_index = 0 if sparse_first else 1
    assert evidence['steps'][sparse_index]['method'] == 'hassle_free'
    assert evidence['steps'][1 - sparse_index]['training_steps'] == 0
    assert capture_inventory(source) == original
    manager = CheckpointManager(str(tmp_path), auto_cleanup=False)
    restored = manager.deserialize_from_bytes(manager.serialize_to_bytes(result))
    torch.testing.assert_close(restored(inputs), result(inputs), rtol=0, atol=0)


def _external_request(version):
    from fedcore.algorithm.low_rank.method_specs import ASVD
    from fedcore.external_runtime import (CompressionRequest, DataRoles, InputSpec,
        MethodOptions, Resources, SparseOptions, WeightedOptions)
    options = {
        1: {'method': 'svd', 'rank': 1},
        2: {'method': 'weighted_svd', 'rank': 1, 'weighted': WeightedOptions()},
        3: {'method': 'asvd', 'rank': 1, 'method_profile': MethodOptions(ASVD())},
        4: {'method': 'hassle_free', 'sparse_profile': SparseOptions(_spec())},
    }[version]
    return CompressionRequest(model='model.fcb', example='example.fcb',
        input_spec=InputSpec((2, 8), dtype='float64'),
        data=DataRoles('validation.fcb', calibration='calibration.fcb'),
        version=version, max_relative_error=1e6, resources=Resources(repetitions=1), **options)


@pytest.mark.parametrize('version', [1, 2, 3, 4])
def test_external_contract_versions_keep_distinct_profiles_and_reject_cross_version_options(version):
    from fedcore.external_runtime import CompressionRequest, ContractError, SUPPORTED_CONTRACT_VERSIONS, plan_request
    assert SUPPORTED_CONTRACT_VERSIONS == (1, 2, 3, 4)
    request = _external_request(version)
    wire = json.loads(json.dumps(request.to_dict(), allow_nan=False))
    assert CompressionRequest.parse(wire) == request
    assert request.method in plan_request(request).steps
    profile_keys = set(wire).intersection({'weighted', 'method_profile', 'sparse_profile'})
    assert profile_keys == {1: set(), 2: {'weighted'}, 3: {'method_profile'}, 4: {'sparse_profile'}}[version]
    for other_version in set(SUPPORTED_CONTRACT_VERSIONS) - {version}:
        invalid = copy.deepcopy(wire)
        invalid['version'] = other_version
        with pytest.raises(ContractError):
            CompressionRequest.parse(invalid)


@pytest.mark.parametrize('version', [1, 2, 3, 4])
def test_external_dispatcher_executes_all_versions_and_exports_the_reported_operator(tmp_path, version):
    from fedcore.external_runtime import plan_request
    from fedcore.external_runtime.execution import execute_plan
    from fedcore.external_runtime.models import save_model_bundle
    from fedcore.external_runtime.security import safe_save
    model, inputs = _fill_parameters(nn.Linear(8, 4, dtype=torch.float64)).eval(), _inputs()
    save_model_bundle(model, tmp_path / 'model.fcb')
    safe_save(inputs[:2], tmp_path / 'example.fcb')
    for role, features in (('calibration', inputs), ('validation', _inputs(9) + .1)):
        safe_save({'kind': 'fedcore_tensor_dataset', 'version': 1, 'features': features,
                   'targets': model(features).detach()}, tmp_path / (role + '.fcb'))
    request = _external_request(version)
    result = execute_plan(plan_request(request), tmp_path)
    assert result['status'] == 'succeeded'
    assert result['provenance']['request'] == request.to_dict()
    assert result.get('request_version', 1) == version
    assert ('weighted_transform' in result) == (version == 2)
    assert ('method_transform' in result) == (version in (3, 4))
    if version == 4:
        assert result['method_transform']['method'] == 'hassle_free'
        measurement = result['artifact_measurement']
        assert measurement['status'] == 'succeeded'
        assert measurement['metrics']['tensor_state_bytes'] == result['metrics']['compressed']['tensor_bytes']
    deployed = torch.jit.load(str(tmp_path / result['artifact']))
    validation = _inputs(9) + .1
    with torch.inference_mode():
        output, expected = deployed(validation), model(validation)
    error = (output - expected).square().mean().item()
    assert result['metrics']['compressed']['quality']['value'] == pytest.approx(error, rel=1e-12, abs=1e-14)
    assert bool(torch.isfinite(output).all())
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('kind', ['packed', 'indexed'])
def test_g5_still_refuses_a_model_containing_physical_sparse_operators(kind):
    from fedcore.algorithm.low_rank.compression_composition import (
        CompositionRecipe, CompositionRequest, LowRank, prepare_composition)
    from fedcore.algorithm.low_rank.plans import ValidationFailure
    model = _mixed(kind, aliases=False)
    inventory = capture_inventory(model)
    request = CompositionRequest((CompositionRecipe('0', (LowRank(1),)),))
    outcome = prepare_composition(model, request)
    assert type(outcome) is ValidationFailure
    assert outcome.violations[0].code == 'UnsupportedExport'
    assert outcome.violations[0].path == '2'
    assert capture_inventory(model) == inventory
