"""Real checkpoint round trips for each supported stored low-rank form."""
import copy
import gc
import io
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from fedcore.algorithm.low_rank.svd_tools import decompose_module, load_svd_state_dict
from fedcore.tools.registry.checkpoint_manager import CheckpointManager, CheckpointError


@pytest.mark.parametrize('mode', ['one_layer', 'two_layers', 'three_layers'])
@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('assembled', [False, True])
def test_actual_representation_roundtrip_after_cache_clear(tmp_path, mode, nested, assembled):
    dense = nn.Linear(5, 4).double().eval()
    source = nn.Sequential(copy.deepcopy(dense)) if nested else copy.deepcopy(dense)
    fitted = decompose_module(source, compose_mode=mode)
    layer = fitted[0] if nested else fitted
    if assembled:
        layer.compose_weight_for_inference()
    x = torch.randn(3, 5, dtype=torch.double)
    expected = dense(x).detach()
    path = tmp_path / 'svd.pt'
    torch.save(fitted.state_dict(), path)
    del fitted, source, layer
    gc.collect()
    template = nn.Sequential(nn.Linear(5, 4).double()) if nested else nn.Linear(5, 4).double()
    original = copy.deepcopy(template.state_dict())
    restored = load_svd_state_dict(template, True, path)
    torch.testing.assert_close(restored(x), expected, atol=1e-12, rtol=1e-12)
    actual = restored[0] if nested else restored
    assert actual._representation == (mode if assembled else 'three_layers')
    assert actual.representation_metadata()['version'] == 1
    for key, value in original.items():
        torch.testing.assert_close(template.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['one_layer', 'two_layers', 'three_layers'])
@pytest.mark.parametrize('profile', ['linear', 'grouped_conv', 'grouped_conv_spatial', 'grouped_conv1d', 'embedding'])
def test_manager_restores_allowlisted_profiles_from_real_file(tmp_path, mode, profile):
    if profile in ('grouped_conv', 'grouped_conv_spatial'):
        dense = nn.Conv2d(4, 6, (3, 2), groups=2, padding=(1, 1)).double()
        x = torch.randn(2, 4, 8, 7, dtype=torch.double)
    elif profile == 'grouped_conv1d':
        dense = nn.Conv1d(4, 6, 3, groups=2, padding=1).double()
        x = torch.randn(2, 4, 8, dtype=torch.double)
    elif profile == 'embedding':
        dense = nn.Embedding(9, 4, padding_idx=2).double()
        x = torch.tensor([0, 2, 7])
    else:
        dense = nn.Linear(5, 4).double()
        x = torch.randn(3, 5, dtype=torch.double)
    expected = dense(x).detach()
    fitted = decompose_module(dense, decomposing_mode='spatial' if profile == 'grouped_conv_spatial' else True,
                              compose_mode=mode)
    fitted.compose_weight_for_inference()
    fitted.eval()
    fitted.U.requires_grad_(False) if fitted.U is not None else fitted.weight.requires_grad_(False)
    manager = CheckpointManager(str(tmp_path), auto_cleanup=False)
    path = str(tmp_path / 'local.pt')
    manager.save_to_file(manager.serialize_to_bytes(fitted), path)
    del fitted
    gc.collect()
    restored = manager.load_from_file(path)
    torch.testing.assert_close(restored(x), expected, atol=1e-12, rtol=1e-12)
    assert restored._representation == mode and not restored.training
    assert not (restored.U if restored.U is not None else restored.weight).requires_grad


def test_legacy_three_factor_checkpoint_remains_supported(tmp_path):
    dense = nn.Linear(5, 4).double()
    fitted = decompose_module(copy.deepcopy(dense))
    path = tmp_path / 'legacy.pt'
    torch.save(dict(fitted.state_dict()), path)
    restored = load_svd_state_dict(nn.Linear(5, 4).double(), True, path)
    x = torch.randn(3, 5, dtype=torch.double)
    torch.testing.assert_close(restored(x), dense(x))


@pytest.mark.parametrize('corruption', ['version', 'shape', 'missing', 'representation'])
def test_invalid_last_node_does_not_mutate_template(tmp_path, corruption):
    fitted = decompose_module(nn.Sequential(nn.Linear(5, 4), nn.Linear(4, 3)))
    state = fitted.state_dict()
    if corruption == 'version':
        state._metadata['1']['fedcore_svd']['version'] = 999
    elif corruption == 'shape':
        state['1.Vh'] = torch.zeros(3, 99)
    elif corruption == 'missing':
        del state['1.Vh']
    else:
        state._metadata['1']['fedcore_svd']['representation'] = 'one_layer'
    path = tmp_path / 'bad.pt'
    torch.save(state, path)
    template = nn.Sequential(nn.Linear(5, 4), nn.Linear(4, 3))
    modules = tuple(template.modules())
    before = copy.deepcopy(template.state_dict())
    with pytest.raises(ValueError):
        load_svd_state_dict(template, True, path)
    assert tuple(template.modules()) == modules
    assert not hasattr(template, '_fedcore_requires_optimizer_rebuild')
    for name, value in before.items():
        torch.testing.assert_close(template.state_dict()[name], value, rtol=0, atol=0)


def test_low_rank_owner_receives_restored_root(tmp_path):
    from fedcore.algorithm.low_rank.low_rank_opt import LowRankModel
    dense = nn.Linear(5, 4)
    fitted = decompose_module(copy.deepcopy(dense), compose_mode='two_layers')
    fitted.compose_weight_for_inference()
    path = tmp_path / 'root.pt'
    torch.save(fitted.state_dict(), path)
    owner = SimpleNamespace(decomposing_mode=True, compose_mode=None, decomposer_params={}, device='cpu',
                            trainer=SimpleNamespace(model=dense))
    LowRankModel.load_model(owner, dense, str(path))
    assert owner.model_after is owner.trainer.model and owner.model_after is not dense
    assert owner.trainer._fedcore_requires_optimizer_rebuild
    x = torch.randn(2, 5)
    torch.testing.assert_close(owner.model_after(x), fitted(x))


def test_lora_local_roundtrip_and_explicit_merged_rejection(tmp_path):
    from fedcore.models.network_modules.layers.lora import apply_lora
    source = apply_lora(nn.Linear(4, 3).double(), rank=2).eval()
    with torch.no_grad():
        source.lora_B['default'].weight.fill_(.2)
    x = torch.randn(3, 4, dtype=torch.double)
    manager = CheckpointManager(str(tmp_path), auto_cleanup=False)
    restored = manager.deserialize_from_bytes(manager.serialize_to_bytes(source))
    torch.testing.assert_close(restored(x), source(x), atol=1e-12, rtol=1e-12)
    restored.merge(); torch.testing.assert_close(restored(x), source(x), atol=1e-12, rtol=1e-12)
    with pytest.raises(CheckpointError, match='Unmerge'):
        manager.serialize_to_bytes(restored)
    restored.unmerge(); torch.testing.assert_close(restored(x), source(x), atol=1e-12, rtol=1e-12)


def test_corrupt_manager_representation_rejects_without_mutation(tmp_path):
    manager = CheckpointManager(str(tmp_path), auto_cleanup=False)
    payload = torch.load(io.BytesIO(manager.serialize_to_bytes(decompose_module(nn.Linear(4, 3)))), weights_only=True)
    payload['architecture']['representation']['version'] = 99
    with pytest.raises(CheckpointError, match='version'):
        manager.restore(payload)


def test_legacy_solver_instance_and_rank_fraction_use_metadata_only_plan():
    from fedcore.algorithm.low_rank.decomposer import SVDDecomposition
    from fedcore.algorithm.low_rank.plans import TransformPlan
    instance = SVDDecomposition(rank=2)
    fitted = decompose_module(nn.Linear(5, 4), decomposer=instance)
    assert isinstance(fitted._fedcore_transform_plan, TransformPlan)
    assert isinstance(fitted._fedcore_transform_plan.request.solver, str)
    assert fitted.U.shape[-1] == fitted._fedcore_transform_plan.steps[0].rank == 2
    fraction = decompose_module(nn.Sequential(nn.Linear(6, 4), nn.Linear(4, 2)), decomposer_params={'rank': .5})
    assert [step.rank for step in fraction._fedcore_transform_plan.steps] == [2, 1]
    assert [fraction[index].U.shape[-1] for index in (0, 1)] == [2, 1]
