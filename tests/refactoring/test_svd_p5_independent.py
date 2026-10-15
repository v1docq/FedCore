"""Independent P5 review: executable algebra, state integrity and resource guards.

These small CPU cases use alternative attention and coordinate oracles rather
than calling the profile's verification helpers as their expected result.
"""
from copy import deepcopy
from dataclasses import replace
import io
import math
import types

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fedcore.algorithm.low_rank.architecture_checkpoint import build_p5, describe_p5
from fedcore.algorithm.low_rank.architecture_profiles import build_architecture, plan_architecture
from fedcore.algorithm.low_rank.lightformer_layers import LightFormerTransformer
from fedcore.algorithm.low_rank.lightformer_specs import LightFormer
from fedcore.algorithm.low_rank.lrt_layers import LEDLinear, LRTAttention
from fedcore.algorithm.low_rank.lrt_profiles import build_lrt, lrt_costs
from fedcore.algorithm.low_rank.lrt_specs import LRT
from fedcore.algorithm.low_rank.slicegpt_layers import SliceGPTModel
from fedcore.algorithm.low_rank.slicegpt_profiles import rotate_slicegpt
from fedcore.algorithm.low_rank.slicegpt_specs import SliceGPT
from fedcore.tools.registry.checkpoint_manager import CheckpointError, CheckpointManager


@pytest.fixture(autouse=True)
def isolated_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(140512)
        yield
    torch.set_num_threads(previous)


def slice_source():
    model = SliceGPTModel(11, 6, 2, 2, 1, 4, 9, eps=.013,
                          dtype=torch.float64).eval()
    with torch.no_grad():
        for index, block in enumerate(model.blocks):
            block.norm.weight.copy_(torch.linspace(.2 + index / 7, 1.8, 6))
        model.final_norm.weight.copy_(torch.linspace(.4, 1.6, 6))
    return model


def light_source():
    model = LightFormerTransformer(11, 11, 6, 2, 9, (0, 1), (0, 1),
                                  max_positions=9, dtype=torch.float64).eval()
    # Analytically rank-one projections permit exact transfer at rank 1.
    with torch.no_grad():
        for index, (name, module) in enumerate(model.named_modules()):
            if type(module) is nn.Linear and name != 'output_projection':
                a = torch.linspace(-.23, .31, module.out_features, dtype=torch.float64)
                b = torch.linspace(.17, .39 + index / 100, module.in_features,
                                   dtype=torch.float64)
                module.weight.copy_(a[:, None] * b)
                if module.bias is not None:
                    module.bias.copy_(torch.linspace(-.11, .19, module.out_features))
    return model


def lrt_spec(**changes):
    return replace(LRT(vocab_size=11, d_model=6, num_heads=2, d_k=3, d_v=2,
                       d_ff=9, num_layers=2, max_seq_len=9, q_rank=2, k_rank=2,
                       v_rank=1, out_rank=2, ffn_in_rank=2, ffn_out_rank=2,
                       seed=83), **changes)


def independent_tensor_bytes(model):
    storages = {}
    for tensor in (*model.parameters(), *model.buffers()):
        storage = tensor.untyped_storage()
        storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def payload_for(model):
    manager = CheckpointManager('unused-independent-review', auto_cleanup=False)
    return manager, torch.load(io.BytesIO(manager.serialize_to_bytes(model)), weights_only=True)


def test_slicegpt_householder_coordinate_equivalence_and_tied_head_bridge():
    source = slice_source()
    vectors = torch.sin(torch.arange(30, dtype=torch.float64).reshape(5, 6) + .27)
    bases = tuple(torch.eye(6, dtype=torch.float64) - 2 * v[:, None] @ v[None] / v.square().sum()
                  for v in vectors)
    rotated = rotate_slicegpt(source, bases)
    tokens = torch.tensor([[1, 3, 8, 2], [7, 6, 4, 0]])
    # Derivation: z=xA; N(z)=N(x)A for orthogonal A. Branch input
    # weights W diag(s) A and residual A.T B preserve each boundary.
    original, observed = source.embedding(tokens), rotated.embedding(tokens)
    torch.testing.assert_close(observed, original @ bases[0], atol=2e-13, rtol=2e-13)
    for i, (old, new) in enumerate(zip(source.blocks, rotated.blocks)):
        original, observed = old(original), new(observed)
        torch.testing.assert_close(observed, original @ bases[i + 1], atol=2e-12, rtol=2e-12)
    expected_bridge = bases[-1].T @ torch.diag(source.final_norm.weight) @ bases[0]
    torch.testing.assert_close(rotated.head_bridge, expected_bridge, atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(rotated(tokens), source(tokens), atol=3e-12, rtol=3e-12)
    assert rotated.embedding.weight is rotated.lm_head.weight


def test_slicegpt_compact_cache_and_exact_storage_accounting():
    source = slice_source()
    tokens = torch.tensor([[1, 3, 8, 2, 5], [7, 6, 4, 0, 1]])
    spec = SliceGPT(widths=(4, 5, 3, 4, 2))
    result = build_architecture(spec, source=source, calibration=tokens)
    assert result.plan.tensor_bytes == independent_tensor_bytes(result.model)
    assert result.plan.parameter_count == sum(p.numel() for p in result.model.parameters())
    assert result.model.embedding.weight.shape == (11, 4)
    assert result.model.head_bridge.shape == (2, 4)
    cached = result.model.causal_forward(tokens[:, :2], use_cache=True)
    suffix = result.model.causal_forward(tokens[:, 2:], caches=cached.caches, use_cache=True)
    torch.testing.assert_close(suffix.logits, result.model(tokens)[:, 2:], atol=3e-12, rtol=3e-12)
    actual_cache_bytes = sum(cache.tensor_bytes for cache in suffix.caches)
    declared = result.evidence['kv_layout']['bytes_per_batch_token'] * tokens.numel()
    assert actual_cache_bytes == declared
    assert not result.evidence['kv_layout']['cache_width_reduced']
    assert result.evidence['approximation']['calibration_logit_mse'] > 0
    assert not result.evidence['claims']['quality_guarantee']


def dense_attention(module, query, memory=None, *, causal=False):
    memory = query if memory is None else memory
    batch, length, width = query.shape
    heads = module.num_heads
    def projection(layer, x):
        return F.linear(x, layer.weight, layer.bias)
    q = projection(module.q_proj, query).reshape(batch, length, heads, width // heads).transpose(1, 2)
    k = projection(module.k_proj, memory).reshape(batch, memory.shape[1], heads, width // heads).transpose(1, 2)
    v = projection(module.v_proj, memory).reshape(batch, memory.shape[1], heads, width // heads).transpose(1, 2)
    context = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return projection(module.o_proj, context.transpose(1, 2).reshape(batch, length, width))


def independent_changed_depth(source, spec, src, tgt):
    norm = lambda module, x: F.layer_norm(x, module.normalized_shape, module.weight, module.bias, module.eps)
    x = F.embedding(src, source.src_embedding.weight) * math.sqrt(source.d_model) + source.positions[:src.shape[1]]
    for group in spec.encoder_groups:
        block = source.encoder_layers[spec.encoder_sources[group]]
        x = x + dense_attention(block.self_attention, norm(block.self_norm, x))
        x = x + F.linear(F.relu(F.linear(norm(block.ffn_norm, x), block.ffn_up.weight,
                                        block.ffn_up.bias)), block.ffn_down.weight, block.ffn_down.bias)
    memory = norm(source.encoder_norm, x)
    x = F.embedding(tgt, source.tgt_embedding.weight) * math.sqrt(source.d_model) + source.positions[:tgt.shape[1]]
    for group in spec.decoder_groups:
        block = source.decoder_layers[spec.decoder_sources[group]]
        x = x + dense_attention(block.self_attention, norm(block.self_norm, x), causal=True)
        x = x + dense_attention(block.cross_attention, norm(block.cross_norm, x), memory)
        x = x + F.linear(F.relu(F.linear(norm(block.ffn_norm, x), block.ffn_up.weight,
                                        block.ffn_up.bias)), block.ffn_down.weight, block.ffn_down.bias)
    return F.linear(norm(source.decoder_norm, x), source.output_projection.weight, source.output_projection.bias)


def test_lightformer_changed_depth_matches_independent_dense_source_execution():
    source = light_source()
    before = deepcopy(source.state_dict())
    spec = LightFormer(1, (0, 0, 1, 1, 1), (0, 0, 0), (1, 0), (1,))
    result = build_architecture(spec, source=source)
    src, tgt = torch.tensor([[1, 2, 8], [3, 5, 7]]), torch.tensor([[4, 3], [2, 1]])
    torch.testing.assert_close(result.model(src, tgt), independent_changed_depth(source, spec, src, tgt),
                               atol=5e-12, rtol=5e-12)
    assert all(torch.equal(value, source.state_dict()[name]) for name, value in before.items())
    assert result.plan.tensor_bytes == independent_tensor_bytes(result.model)
    assert result.model.encoder_layers[0] is result.model.encoder_layers[1]
    assert result.model.encoder_layers[2] is result.model.encoder_layers[3] is result.model.encoder_layers[4]
    assert result.model.decoder_layers[0] is result.model.decoder_layers[1] is result.model.decoder_layers[2]
    optimizer = torch.optim.SGD(result.model.parameters(), lr=.01)
    owned = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(owned) == len(set(owned)) == len(list(result.model.parameters()))


def test_lrt_standard_attention_matches_dense_effective_weight_sdpa():
    model = LRTAttention(lrt_spec(num_layers=1), dtype=torch.float64)
    inputs = torch.cos(torch.arange(48, dtype=torch.float64).reshape(2, 4, 6) / 13)
    def project(layer, x):
        return F.linear(x, layer.decoder.weight @ layer.encoder.weight, layer.decoder.bias)
    q = torch.stack([project(layer, inputs) for layer in model.q_proj], 1)
    k = torch.stack([project(layer, inputs) for layer in model.k_proj], 1)
    v = torch.stack([project(layer, inputs) for layer in model.v_proj], 1)
    context = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    expected = project(model.out_proj, context.transpose(1, 2).reshape(2, 4, 4))
    torch.testing.assert_close(model(inputs), expected, atol=2e-13, rtol=2e-13)


def test_lrt_build_without_svd_and_optimizer_updates_every_factor(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('LRT must train directly from initialized factors')
    monkeypatch.setattr(torch.linalg, 'svd', forbidden)
    monkeypatch.setattr(torch.linalg, 'svdvals', forbidden)
    monkeypatch.setattr(torch, 'svd', forbidden)
    result = build_architecture(lrt_spec(), dtype=torch.float64)
    model = result.model
    tokens = torch.tensor([[1, 3, 8, 2], [7, 6, 4, 0]])
    optimizer = torch.optim.SGD(model.parameters(), lr=.03)
    factors = {name: value for name, value in model.named_parameters()
               if name.endswith(('encoder.weight', 'decoder.weight'))}
    before = {name: value.detach().clone() for name, value in factors.items()}
    F.cross_entropy(model(tokens)[:, :-1].reshape(-1, 11), tokens[:, 1:].reshape(-1)).backward()
    assert factors
    for name, parameter in factors.items():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), name
        assert bool(parameter.grad.abs().sum() > 0), name
    optimizer.step()
    assert all(not torch.equal(value, before[name]) for name, value in factors.items())
    assert result.plan.tensor_bytes == independent_tensor_bytes(model)
    costs = lrt_costs(lrt_spec(), sequence_length=4)
    smaller = lrt_costs(lrt_spec(q_rank=1, k_rank=1, out_rank=1), sequence_length=4)
    assert smaller['attention_macs'] == costs['attention_macs']
    assert smaller['hypothetical_kv_elements_all_layers'] == costs['hypothetical_kv_elements_all_layers']


@pytest.mark.parametrize('kind', ['slice', 'light', 'lrt'])
def test_checkpoint_rejects_same_shape_executable_mutations(kind):
    if kind == 'slice':
        model = slice_source()
        model.blocks[0].consumer.rope_base *= 2
    elif kind == 'light':
        model = light_source()
        model.decoder_layers[0].cross_attention.num_heads = 3
    else:
        model = build_lrt(lrt_spec(), dtype=torch.float64).model
        model.blocks[0].attention_norm.eps = .02
    manager = CheckpointManager('unused-independent-review', auto_cleanup=False)
    with pytest.raises(CheckpointError):
        manager.serialize_to_bytes(model)


@pytest.mark.parametrize('kind', ['slice', 'light', 'lrt'])
def test_checkpoint_rejects_finite_fixed_buffer_tampering(kind):
    if kind == 'slice':
        model, path = slice_source(), 'blocks.0.norm.width_denominator'
    elif kind == 'light':
        model, path = light_source(), 'positions'
    else:
        model, path = build_lrt(lrt_spec(), dtype=torch.float64).model, 'position_encoding'
    manager, payload = payload_for(model)
    payload['state_dict'][path].reshape(-1)[0] += .25
    with pytest.raises(CheckpointError):
        manager.restore(payload)


def test_slicegpt_checkpoint_rejects_added_parameter_alias():
    manager, payload = payload_for(slice_source())
    key, value = 'blocks.0.consumer.k_proj.weight', 'blocks.0.consumer.v_proj.weight'
    payload['state_dict'][value] = payload['state_dict'][key].clone()
    payload['topology']['parameter_aliases'] = (*payload['topology']['parameter_aliases'], (key, value))
    with pytest.raises(CheckpointError):
        manager.restore(payload)


def test_slicegpt_checkpoint_descriptor_caps_allocation_before_embedding(monkeypatch):
    description = describe_p5(slice_source())
    description['config']['vocab_size'] = 10**10
    def allocation_forbidden(*args, **kwargs):
        raise AssertionError('Malformed checkpoint reached tensor allocation')
    monkeypatch.setattr(torch, 'empty', allocation_forbidden)
    with pytest.raises(ValueError):
        build_p5(description)


@pytest.mark.parametrize('kind', ['light', 'lrt'])
def test_other_checkpoint_descriptors_cap_allocation_before_embedding(monkeypatch, kind):
    model = light_source() if kind == 'light' else build_lrt(lrt_spec(), dtype=torch.float64).model
    description = describe_p5(model)
    if kind == 'light':
        description['config']['src_vocab_size'] = 10**10
    else:
        description['config']['spec']['vocab_size'] = 10**10
    def allocation_forbidden(*args, **kwargs):
        raise AssertionError('Malformed checkpoint reached tensor allocation')
    monkeypatch.setattr(torch, 'empty', allocation_forbidden)
    with pytest.raises(ValueError):
        build_p5(description)


@pytest.mark.parametrize('container', [nn.Sequential, nn.ModuleDict])
def test_nested_p5_checkpoint_roundtrip_preserves_aliases_and_fixed_buffers(container):
    tokens = torch.tensor([[1, 3, 8, 2], [7, 6, 4, 0]])
    sliced = build_architecture(SliceGPT(widths=(4, 5, 3, 4, 2)),
                                source=slice_source(), calibration=tokens).model
    light = build_architecture(LightFormer(1, (0, 0, 1), (0, 0), (1, 0), (1,)),
                               source=light_source()).model
    lrt = build_lrt(lrt_spec(), dtype=torch.float64).model
    if container is nn.Sequential:
        graph = container(sliced, light, lrt)
        originals = {'0': sliced, '1': light, '2': lrt}
    else:
        originals = {'slice': sliced, 'light': light, 'lrt': lrt}
        graph = container(originals)
    manager, payload = payload_for(graph)
    restored = manager.restore(payload)
    for name, original in originals.items():
        target = restored._modules[name]
        inputs = (tokens, tokens[:, :3]) if isinstance(original, LightFormerTransformer) else (tokens,)
        torch.testing.assert_close(target(*inputs), original(*inputs), rtol=0, atol=0)
        assert independent_tensor_bytes(target) == independent_tensor_bytes(original)
    restored_slice = restored[0] if container is nn.Sequential else restored['slice']
    restored_light = restored[1] if container is nn.Sequential else restored['light']
    assert restored_slice.embedding.weight is restored_slice.lm_head.weight
    assert restored_light.encoder_layers[0] is restored_light.encoder_layers[1]
    assert restored_light.decoder_layers[0] is restored_light.decoder_layers[1]


@pytest.mark.parametrize('kind', ['slice', 'light', 'lrt'])
def test_nested_p5_checkpoint_rejects_added_cross_group_parameter_tie(kind):
    if kind == 'slice':
        model = slice_source()
        a, b = 'blocks.0.consumer.k_proj.weight', 'blocks.0.consumer.v_proj.weight'
    elif kind == 'light':
        model = build_architecture(LightFormer(1, (0, 0, 1), (0, 0), (1, 0), (1,)),
                                   source=light_source()).model
        a, b = 'encoder_layers.0.ffn_up.right.weight', 'encoder_layers.2.ffn_up.right.weight'
    else:
        model = build_lrt(lrt_spec(), dtype=torch.float64).model
        a, b = 'blocks.0.attention.q_proj.0.encoder.weight', 'blocks.0.attention.k_proj.0.encoder.weight'
    manager, payload = payload_for(nn.ModuleDict({'architecture': model, 'other': nn.Identity()}))
    a, b = 'architecture.' + a, 'architecture.' + b
    state = payload['state_dict']
    state[b] = state[a].clone()
    # When b is an application alias, update the entire existing alias group
    # before merging it with a. This isolates the extra topology mutation from
    # a mere equal-value consistency failure.
    remaining = []
    merged = [a, b]
    for group in payload['topology']['parameter_aliases']:
        if a in group or b in group:
            merged.extend(path for path in group if path not in merged)
        else:
            remaining.append(group)
    for path in merged:
        state[path] = state[a].clone()
    payload['topology']['parameter_aliases'] = (*remaining, tuple(merged))
    with pytest.raises(CheckpointError):
        manager.restore(payload)


@pytest.mark.parametrize('traversal', [None, 'modules', 'named_modules'])
def test_lightformer_checkpoint_rejects_encode_override_missing_from_descriptor(traversal):
    model = light_source()
    original = LightFormerTransformer.encode
    model.encode = types.MethodType(lambda self, *args, **kwargs: original(self, *args, **kwargs) * 2,
                                   model)
    if traversal is not None:
        setattr(model, traversal, types.MethodType(lambda *args, **kwargs: iter(()), model))
    manager = CheckpointManager('unused-independent-review', auto_cleanup=False)
    with pytest.raises(CheckpointError):
        manager.serialize_to_bytes(model)


@pytest.mark.parametrize('kind', ['slice', 'light', 'lrt'])
def test_checkpoint_refuses_state_hooks_that_change_saved_weight_values(kind):
    model = slice_source() if kind == 'slice' else (
        light_source() if kind == 'light' else build_lrt(lrt_spec(), dtype=torch.float64).model)
    key = 'lm_head.bias' if kind == 'slice' else ('output_projection.bias' if kind == 'light' else 'output.bias')
    def alter_saved_weights(module, state, prefix, metadata):
        state[prefix + key] = state[prefix + key] + 1
    # PyTorch 2.2 exposes this post-hook registration under its older name.
    handle = model._register_state_dict_hook(alter_saved_weights)
    manager = CheckpointManager('unused-independent-review', auto_cleanup=False)
    try:
        with pytest.raises(CheckpointError):
            manager.serialize_to_bytes(model)
    finally:
        handle.remove()
