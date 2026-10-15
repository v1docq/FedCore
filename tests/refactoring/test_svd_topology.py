"""Sharing is determined by identity and storage, never by equal weight values."""
import copy
import io
import importlib
import pytest
import torch
from torch import nn
from fedcore.algorithm.low_rank.topology import (
    TopologyError, inspect_topology, module_paths, replace_modules_atomically,
)
from fedcore.algorithm.low_rank.svd_tools import decompose_module, load_svd_state_dict
from fedcore.models.network_modules.layers.lora import apply_lora
from fedcore.tools.registry.checkpoint_manager import CheckpointManager, CheckpointError


def shared_linears():
    first, second = nn.Linear(4, 4), nn.Linear(4, 4)
    second.weight = first.weight
    return nn.Sequential(first, second)


def test_module_aliases_and_independent_equal_values():
    shared = nn.Linear(4, 4)
    independent = copy.deepcopy(shared)
    model = nn.Sequential(shared, shared, independent)
    before = inspect_topology(model)
    assert ('0', '1') in before.module_aliases
    assert before.unique_parameter_numel == 40
    decompose_module(model)
    assert model[0] is model[1] and model[0] is not model[2]
    assert model[0].U is not model[2].U
    assert model._fedcore_requires_optimizer_rebuild


def test_identical_parameters_share_factors_and_checkpoint_storage(tmp_path):
    model = shared_linears()
    decompose_module(model)
    assert model[0] is not model[1]
    assert model[0].U is model[1].U and model[0].S is model[1].S and model[0].Vh is model[1].Vh
    assert inspect_topology(model).unique_parameter_numel == 44
    manager = CheckpointManager(str(tmp_path), auto_cleanup=False)
    restored = manager.deserialize_from_bytes(manager.serialize_to_bytes(model))
    assert restored[0].U is restored[1].U and restored[0].Vh is restored[1].Vh
    path = tmp_path / 'tied.pt'
    torch.save(model.state_dict(), path)
    restored_raw = load_svd_state_dict(shared_linears(), True, path)
    assert restored_raw[0].U is restored_raw[1].U
    x = torch.randn(2, 4)
    torch.testing.assert_close(restored(x), model(x))
    torch.testing.assert_close(restored_raw(x), model(x))


def test_checkpoint_named_children_do_not_drop_alias_edges(tmp_path):
    linear = nn.Linear(4, 4)
    model = nn.Sequential(linear, linear)
    manager = CheckpointManager(str(tmp_path), auto_cleanup=False)
    restored = manager.deserialize_from_bytes(manager.serialize_to_bytes(model))
    assert len(restored) == 2 and restored[0] is restored[1]
    x = torch.randn(2, 4)
    torch.testing.assert_close(restored(x), model(x))


@pytest.mark.parametrize('transformation', [decompose_module, apply_lora])
def test_mixed_embedding_head_tie_rejects_atomically(transformation):
    embedding, head = nn.Embedding(6, 4), nn.Linear(4, 6, bias=False)
    head.weight = embedding.weight
    model = nn.ModuleDict({'embedding': embedding, 'head': head})
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(TopologyError):
        transformation(model)
    assert model['embedding'] is embedding and model['head'] is head and embedding.weight is head.weight
    for name, tensor in before.items():
        torch.testing.assert_close(model.state_dict()[name], tensor)


@pytest.mark.parametrize('fixture', ['basis_sharing', 'group_reduce', 'lightformer'])
def test_sharing_fixtures_describe_storage_without_implementing_methods(fixture):
    storage = torch.randn(8, 4)
    model = nn.Module()
    if fixture == 'basis_sharing':
        model.register_parameter('basis', nn.Parameter(storage))
        model.register_parameter('projection', nn.Parameter(storage[:4]))
    elif fixture == 'group_reduce':
        model.register_parameter('left', nn.Parameter(storage[:4]))
        model.register_parameter('right', nn.Parameter(storage[4:]))
    else:
        model.register_parameter('queries', nn.Parameter(storage))
        model.register_parameter('keys', nn.Parameter(storage.T))
    topology = inspect_topology(model)
    assert len(topology.storage_aliases) == 1
    assert topology.unique_storage_bytes == storage.untyped_storage().nbytes()
    assert not topology.parameter_aliases


@pytest.mark.parametrize('transformation', [decompose_module, apply_lora])
def test_storage_view_rejected_before_any_module_replacement(transformation):
    first, last = nn.Linear(4, 4), nn.Linear(4, 4)
    last.weight = nn.Parameter(first.weight.T)
    model = nn.Sequential(nn.Linear(4, 4), first, last)
    original = tuple(model.modules())
    with pytest.raises(TopologyError, match='storage'):
        transformation(model)
    assert tuple(model.modules()) == original and not hasattr(model, '_fedcore_requires_optimizer_rebuild')


def test_alias_lora_merge_unmerge_preserves_reference_and_source():
    layer = nn.Linear(4, 4)
    source = nn.Sequential(layer, layer)
    adapted = apply_lora(source, rank=2, target_layers=['1']).eval()
    assert adapted[0] is adapted[1] and source[0] is source[1] is layer
    with torch.no_grad():
        adapted[0].lora_B['default'].weight.fill_(.1)
    x = torch.randn(2, 4)
    expected = adapted(x)
    adapted[1].merge(); torch.testing.assert_close(adapted(x), expected)
    adapted[0].unmerge(); torch.testing.assert_close(adapted(x), expected)


def test_invalid_last_replacement_is_atomic():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    original = model[0]
    with pytest.raises(TopologyError):
        replace_modules_atomically(model, {'0': nn.Identity(), 'missing': nn.Identity()})
    assert model[0] is original and not hasattr(model, '_fedcore_requires_optimizer_rebuild')
    with pytest.raises(TopologyError, match='cycle'):
        replace_modules_atomically(model, {'0': nn.Identity(), '1': model})
    assert model[0] is original and not hasattr(model, '_fedcore_requires_optimizer_rebuild')


def test_alias_replacement_conflict_and_cyclic_graph_fail_before_mutation():
    layer = nn.Linear(4, 4)
    model = nn.Sequential(layer, layer)
    with pytest.raises(TopologyError, match='Conflicting'):
        replace_modules_atomically(model, {'0': nn.Identity(), '1': nn.ReLU()})
    assert model[0] is model[1] is layer
    model.add_module('cycle', model)
    with pytest.raises(TopologyError, match='Cyclic'):
        module_paths(model)


def test_conflicting_checkpoint_tied_values_are_rejected(tmp_path):
    manager = CheckpointManager(str(tmp_path), auto_cleanup=False)
    source = shared_linears()
    payload = torch.load(io.BytesIO(manager.serialize_to_bytes(source)), weights_only=True)
    payload['state_dict']['1.weight'].add_(1)
    with pytest.raises(CheckpointError, match='Conflicting'):
        manager.restore(payload)
    assert source[0].weight is source[1].weight


def test_individual_tied_composition_rejects_before_mutation(tmp_path):
    model = decompose_module(shared_linears(), compose_mode='two_layers')
    manager = CheckpointManager(str(tmp_path), auto_cleanup=False)
    for candidate in (model, manager.deserialize_from_bytes(manager.serialize_to_bytes(model))):
        original = candidate[0].U, candidate[0].S, candidate[0].Vh
        with pytest.raises(ValueError, match='jointly'):
            candidate[0].compose_weight_for_inference()
        with pytest.raises(ValueError, match='jointly'):
            candidate[0].compose()
        assert (candidate[0].U, candidate[0].S, candidate[0].Vh) == original
        assert candidate[0].U is candidate[1].U and candidate[0].S is candidate[1].S


def test_portable_bundle_explicitly_rejects_aliases(tmp_path):
    from fedcore.external_runtime.models import save_model_bundle
    from fedcore.external_runtime.contracts import ContractError
    layer = nn.Linear(4, 4)
    for model in (nn.Sequential(layer, layer), shared_linears()):
        path = tmp_path / 'unsupported.fcb'
        with pytest.raises(ContractError, match='Portable model bundles'):
            save_model_bundle(model, path)
        assert not path.exists()


def test_parameter_replacement_requires_a_fresh_optimizer():
    model = shared_linears()
    old_optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    old_ids = {id(parameter) for group in old_optimizer.param_groups for parameter in group['params']}
    decompose_module(model)
    current_ids = {id(parameter) for parameter in model.parameters()}
    assert model._fedcore_requires_optimizer_rebuild and not old_ids & current_ids
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    owned_ids = [id(parameter) for group in optimizer.param_groups for parameter in group['params']]
    assert set(owned_ids) == current_ids and len(owned_ids) == len(current_ids)
    before = model[0].U.detach().clone()
    model(torch.randn(3, 4)).square().mean().backward()
    optimizer.step()
    assert not torch.equal(before, model[0].U)
    assert model[0].U is model[1].U


@pytest.mark.parametrize('boundary', ['lora', 'svd_load', 'checkpoint_restore'])
def test_untouched_storage_views_rejected_before_model_copy(tmp_path, monkeypatch, boundary):
    model = nn.Module()
    model.target = nn.Linear(4, 4)
    model.aux = nn.Module()
    storage = torch.randn(4, 4)
    model.aux.a = nn.Parameter(storage)
    model.aux.b = nn.Parameter(storage.T)
    original_modules = tuple(model.modules())
    original_parameters = tuple(model.parameters())
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    original_topology = inspect_topology(model)
    def forbidden_copy(_model):
        raise AssertionError('Unsupported storage views must be detected before deepcopy')
    if boundary == 'lora':
        module = importlib.import_module('fedcore.models.network_modules.layers.lora')
        monkeypatch.setattr(module, 'deepcopy', forbidden_copy)
        operation = lambda: apply_lora(model, rank=2, target_layers=['target'])
    elif boundary == 'svd_load':
        path = tmp_path / 'views.pt'
        torch.save(model.state_dict(), path)
        module = importlib.import_module('fedcore.algorithm.low_rank.svd_tools')
        monkeypatch.setattr(module, 'deepcopy', forbidden_copy)
        operation = lambda: load_svd_state_dict(model, True, path)
    else:
        module = importlib.import_module('fedcore.tools.registry.checkpoint_manager')
        monkeypatch.setattr(module, 'deepcopy', forbidden_copy)
        operation = lambda: CheckpointManager.restore({'state_dict': model.state_dict()}, model=model)
    with pytest.raises(ValueError, match='storage views'):
        operation()
    assert tuple(model.modules()) == original_modules
    assert tuple(model.parameters()) == original_parameters
    assert inspect_topology(model) == original_topology
    assert not hasattr(model, '_fedcore_requires_optimizer_rebuild')
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


def test_in_place_decomposition_preserves_untouched_storage_views():
    model = nn.Module()
    model.target = nn.Linear(4, 4)
    model.aux = nn.Module()
    storage = torch.randn(4, 4)
    model.aux.a = nn.Parameter(storage)
    model.aux.b = nn.Parameter(storage.T)
    original_a, original_b = model.aux.a, model.aux.b
    before = inspect_topology(model).storage_aliases
    assert decompose_module(model) is model
    assert model.aux.a is original_a and model.aux.b is original_b
    assert inspect_topology(model).storage_aliases == before
