"""Independent encoder/decoder algebra and complete group-sharing invariants."""
from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
import math
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fedcore.algorithm.low_rank.lightformer_layers import LightFormerTransformer
from fedcore.algorithm.low_rank.lightformer_profiles import (build_lightformer,
    plan_lightformer, transfer_lightformer)
from fedcore.algorithm.low_rank.lightformer_specs import LightFormer
from fedcore.algorithm.low_rank.method_errors import MethodSpecError
from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear


def _spec(**kwargs):
    return LightFormer(**{**dict(rank=2, encoder_groups=(0, 0, 0, 1, 1, 1),
        decoder_groups=(0, 0), encoder_sources=(1, 0), decoder_sources=(2,)), **kwargs})


def _source(*, low_rank=False, dtype=torch.float64, bias=True):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(401)
        model = LightFormerTransformer(11, 13, 8, 2, 12, (0, 1), (0, 1, 2),
                                      max_positions=12, dtype=dtype, bias=bias)
        # Every parameter, including norm biases, carries nonzero source data.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.uniform_(.02, .12)
            if low_rank:
                for name, module in model.named_modules():
                    if type(module) is nn.Linear and name != "output_projection":
                        a = torch.randn(module.out_features, 2, dtype=dtype) * .12
                        b = torch.randn(2, module.in_features, dtype=dtype) * .12
                        module.weight.copy_(a @ b)
        return model


def _linear(module, x):
    # Independent dense reconstruction checks execution of both compact factors.
    if type(module) is FactorizedLinear:
        return F.linear(x, module.left.weight @ module.right.weight, module.left.bias)
    return F.linear(x, module.weight, module.bias)


def _norm(module, x):
    return F.layer_norm(x, module.normalized_shape, module.weight, module.bias, module.eps)


def _attention(module, query, memory=None, *, valid=None, causal=False):
    memory = query if memory is None else memory
    batch, qlen, width = query.shape
    klen, heads = memory.shape[1], module.num_heads
    q = _linear(module.q_proj, query).view(batch, qlen, heads, width // heads).transpose(1, 2)
    k = _linear(module.k_proj, memory).view(batch, klen, heads, width // heads).transpose(1, 2)
    v = _linear(module.v_proj, memory).view(batch, klen, heads, width // heads).transpose(1, 2)
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * (width // heads) ** -.5
    if causal:
        scores = scores.masked_fill(torch.arange(klen)[None, :] > torch.arange(qlen)[:, None], -torch.inf)
    if valid is not None:
        scores = scores.masked_fill(~valid[:, None, None, :], -torch.inf)
    weights = torch.softmax(scores, -1)
    weights = torch.nan_to_num(weights, nan=0.)
    context = torch.einsum("bhqk,bhkd->bhqd", weights, v).transpose(1, 2).reshape(batch, qlen, width)
    return _linear(module.o_proj, context)


def _manual(source, spec, src, tgt, src_mask=None, tgt_mask=None):
    # Use original dense source weights at the declared changed target depths.
    x = F.embedding(src, source.src_embedding.weight) * math.sqrt(source.d_model) + source.positions[:src.shape[1]]
    for group in spec.encoder_groups:
        block = source.encoder_layers[spec.encoder_sources[group]]
        x = x + _attention(block.self_attention, _norm(block.self_norm, x), valid=src_mask)
        x = x + _linear(block.ffn_down, F.relu(_linear(block.ffn_up, _norm(block.ffn_norm, x))))
    memory = _norm(source.encoder_norm, x)
    x = F.embedding(tgt, source.tgt_embedding.weight) * math.sqrt(source.d_model) + source.positions[:tgt.shape[1]]
    for group in spec.decoder_groups:
        block = source.decoder_layers[spec.decoder_sources[group]]
        x = x + _attention(block.self_attention, _norm(block.self_norm, x), valid=tgt_mask, causal=True)
        x = x + _attention(block.cross_attention, _norm(block.cross_norm, x), memory, valid=src_mask)
        x = x + _linear(block.ffn_down, F.relu(_linear(block.ffn_up, _norm(block.ffn_norm, x))))
    return _linear(source.output_projection, _norm(source.decoder_norm, x))


def test_spec_is_metadata_only_and_has_immutable_explicit_maps():
    subprocess.run([sys.executable, "-c", "import sys; from fedcore.algorithm.low_rank.lightformer_specs import LightFormer; assert 'torch' not in sys.modules"], check=True)
    for kwargs in ({"rank": True}, {"rank": 0}, {"encoder_groups": (0, 1, 0)},
                   {"decoder_groups": (1,)}, {"encoder_groups": [0]},
                   {"encoder_sources": ()}, {"decoder_sources": (-1,)},
                   {"max_parameters": 0}, {"max_peak_bytes": True}, {"variant": "guessed"}):
        with pytest.raises(MethodSpecError):
            _spec(**kwargs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("bias", [False, True])
def test_exact_full_cost_and_separate_build_transfer(dtype, bias):
    source = _source(dtype=dtype, bias=bias)
    spec = _spec()
    plan = plan_lightformer(source, spec)
    fresh = build_lightformer(plan)
    result = transfer_lightformer(source, spec, plan=plan)
    assert plan.parameter_count == sum(p.numel() for p in result.model.parameters())
    assert plan.tensor_bytes == sum(t.numel() * t.element_size() for t in (*result.model.parameters(), *result.model.buffers()))
    assert plan.source_parameter_elements == sum(p.numel() for p in source.parameters())
    assert plan.parameter_count < plan.source_parameter_elements
    fixed = sum(p.numel() for name in ("src_embedding", "tgt_embedding", "encoder_norm", "decoder_norm", "output_projection")
                for p in getattr(result.model, name).parameters())
    assert plan.fixed_parameter_elements == fixed == result.evidence["fixed_parameters"]
    assert not torch.equal(fresh.src_embedding.weight, source.src_embedding.weight)
    torch.testing.assert_close(result.model.src_embedding.weight, source.src_embedding.weight)
    json.dumps(asdict(plan), allow_nan=False)
    json.dumps(result.evidence, allow_nan=False)


def test_changed_depth_manual_source_reference_and_standard_attention():
    source, spec = _source(low_rank=True), _spec()
    target = transfer_lightformer(source, spec).model
    src, tgt = torch.tensor([[1, 3, 2], [4, 5, 2]]), torch.tensor([[2, 7], [2, 1]])
    sm, tm = torch.tensor([[True, True, False], [True, True, True]]), torch.tensor([[True, False], [True, True]])
    actual = target(src, tgt, src_padding_mask=sm, tgt_padding_mask=tm)
    expected = _manual(source, spec, src, tgt, sm, tm)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    generator = torch.Generator().manual_seed(9)
    query = torch.randn(2, 3, 8, dtype=torch.float64, generator=generator)
    memory = torch.randn(2, 5, 8, dtype=torch.float64, generator=generator)
    block = target.decoder_layers[0]
    torch.testing.assert_close(block.self_attention(query, causal=True), _attention(block.self_attention, query, causal=True))
    torch.testing.assert_close(block.cross_attention(query, memory), _attention(block.cross_attention, query, memory))
    all_masked = torch.zeros(2, 5, dtype=torch.bool)
    assert torch.isfinite(block.cross_attention(query, memory, padding_mask=all_masked)).all()
    # Repeated calls use distinct states and do not retain hidden application state.
    torch.testing.assert_close(target(src, tgt), target(src, tgt))
    assert not torch.equal(block.self_attention(query), block.self_attention(query + 1))
    with pytest.raises(ValueError, match="KV cache"):
        target(src, tgt, use_cache=True)
    with pytest.raises(ValueError, match="KV cache"):
        block.self_attention(query, cache=object())


def test_complete_group_objects_optimizer_uniqueness_and_clone_negative():
    result = transfer_lightformer(_source(), _spec())
    model = result.model
    assert model.encoder_layers[0] is model.encoder_layers[1] is model.encoder_layers[2]
    assert model.encoder_layers[3] is model.encoder_layers[4] is model.encoder_layers[5]
    assert model.encoder_layers[0] is not model.encoder_layers[3]
    assert model.decoder_layers[0] is model.decoder_layers[1]
    for name in ("self_attention", "cross_attention", "self_norm", "cross_norm", "ffn_norm", "ffn_up", "ffn_down"):
        assert getattr(model.decoder_layers[0], name) is getattr(model.decoder_layers[1], name)
    optimizer = torch.optim.AdamW(model.parameters())
    entries = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(entries) == len({id(p) for p in entries})
    calls = []
    hook = model.encoder_layers[0].register_forward_hook(lambda *_: calls.append(1))
    model(torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3]]))
    hook.remove()
    assert len(calls) == 3
    linear_calls = []
    hooks = [m.register_forward_hook(lambda *_: linear_calls.append(1)) for m in model.modules() if type(m) is nn.Linear]
    model(torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3]]))
    for hook in hooks:
        hook.remove()
    assert len(linear_calls) == result.evidence["linear_operator_applications"]
    assert result.evidence["encoder_applications"] == 6
    assert result.evidence["unique_encoder_groups"] == 2
    assert result.evidence["cross_attention_applications"] == 2
    assert result.evidence["factor_linear_applications"] == 2 * (6 * 6 + 10 * 2)
    model.encoder_layers[1] = deepcopy(model.encoder_layers[0])
    with pytest.raises(ValueError, match="alias|sharing"):
        model.validate_checkpoint_state(model.state_dict())


def test_svd_initialization_uses_declared_source_and_real_factor_gradients():
    source, spec = _source(), _spec()
    result = transfer_lightformer(source, spec)
    model = result.model
    original = source.encoder_layers[1].self_attention.q_proj
    compact = model.encoder_layers[0].self_attention.q_proj
    u, s, vh = torch.linalg.svd(original.weight.detach(), full_matrices=False)
    torch.testing.assert_close(compact.left.weight @ compact.right.weight, (u[:, :2] * s[:2]) @ vh[:2])
    torch.testing.assert_close(compact.left.bias, original.bias)
    assert model.encoder_layers[0].self_norm.weight.data_ptr() != source.encoder_layers[1].self_norm.weight.data_ptr()
    before = compact.right.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    src, tgt = torch.tensor([[1, 3, 2], [4, 5, 6]]), torch.tensor([[2, 7], [2, 1]])
    loss = model(src, tgt).square().mean()
    loss.backward()
    factors = [p for name, p in model.named_parameters() if name.endswith("left.weight") or name.endswith("right.weight")]
    assert factors and all(p.grad is not None and torch.isfinite(p.grad).all() for p in factors)
    assert all(torch.count_nonzero(p.grad) for p in factors)
    optimizer.step()
    assert not torch.equal(before, compact.right.weight)
    assert len(optimizer.state) == len(list(model.parameters()))
    moments = sum(v.numel() * v.element_size() for state in optimizer.state.values()
                  for key, v in state.items() if key in ("exp_avg", "exp_avg_sq"))
    assert moments == result.evidence["training_resources"]["adam_two_moment_bytes_estimate"]


def test_source_storage_modes_and_all_rng_streams_are_preserved():
    source = _source().eval()
    before = {name: value.clone() for name, value in source.state_dict().items()}
    torch_state, python_state, numpy_state = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    result = transfer_lightformer(source, _spec())
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert not source.training and not result.model.training
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    source_storages = {p.untyped_storage().data_ptr() for p in source.parameters()}
    assert not source_storages & {p.untyped_storage().data_ptr() for p in result.model.parameters()}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_constructor_checkpoint_roundtrip_retains_aliases_and_rejects_corruption(dtype):
    model = transfer_lightformer(_source(dtype=dtype), _spec()).model
    data = io.BytesIO()
    torch.save({"config": model.representation_config(), "state": model.state_dict()}, data)
    data.seek(0)
    restored_data = torch.load(data, weights_only=True)
    restored = LightFormerTransformer.from_config(restored_data["config"], dtype=dtype)
    restored.validate_checkpoint_state(restored_data["state"])
    restored.load_state_dict(restored_data["state"], strict=True)
    assert restored.encoder_layers[0] is restored.encoder_layers[2]
    assert restored.decoder_layers[0] is restored.decoder_layers[1]
    src, tgt = torch.tensor([[1, 2, 3]]), torch.tensor([[2, 1]])
    torch.testing.assert_close(restored(src, tgt), model(src, tgt), atol=0, rtol=0)
    corrupt = {k: v.clone() for k, v in restored_data["state"].items()}
    corrupt["encoder_layers.1.self_norm.weight"][0] += .1
    with pytest.raises(ValueError, match="shared parameter"):
        restored.validate_checkpoint_state(corrupt)
    for key in ("_encoder_source_map", "_decoder_group_map", "positions"):
        corrupt = {k: v.clone() for k, v in restored_data["state"].items()}
        corrupt[key].view(-1)[0] += 1
        with pytest.raises(ValueError, match="metadata|map"):
            restored.validate_checkpoint_state(corrupt)
    corrupt = dict(restored_data["state"])
    corrupt["output_projection.weight"] = torch.full_like(corrupt["output_projection.weight"], float("nan"))
    with pytest.raises(ValueError, match="tensor mismatch"):
        restored.validate_checkpoint_state(corrupt)
    restored.decoder_layers[0].cross_norm.eps *= 2
    with pytest.raises(ValueError, match="normalization topology"):
        restored.validate_checkpoint_state(restored_data["state"])


@pytest.mark.parametrize("spec", [_spec(rank=4), _spec(encoder_sources=(5, 0)),
                                   _spec(max_parameters=1), _spec(max_peak_bytes=1)])
def test_refusal_happens_before_architecture_build_or_svd(monkeypatch, spec):
    source = _source()
    def forbidden(*args, **kwargs):
        pytest.fail("invalid transformation allocated architecture or started SVD")
    monkeypatch.setattr("fedcore.algorithm.low_rank.lightformer_profiles.build_lightformer", forbidden)
    monkeypatch.setattr(torch.linalg, "svd", forbidden)
    with pytest.raises(MethodSpecError):
        transfer_lightformer(source, spec)


def test_unsupported_graph_stale_plan_and_source_modification_refuse():
    with pytest.raises(MethodSpecError, match="arbitrary|HF"):
        plan_lightformer(nn.Transformer(d_model=8, nhead=2, batch_first=True), _spec())
    source, spec = _source(), _spec()
    plan = plan_lightformer(source, spec)
    with torch.no_grad():
        source.src_embedding.weight[0, 0] += 1
    with pytest.raises(MethodSpecError, match="stale"):
        transfer_lightformer(source, spec, plan=plan)
    source = _source()
    source.decoder_layers[0].cross_attention.q_proj.weight = source.decoder_layers[1].cross_attention.q_proj.weight
    with pytest.raises(MethodSpecError, match="ties"):
        plan_lightformer(source, spec)
    with pytest.raises(MethodSpecError, match="inconsistent"):
        build_lightformer(replace(plan, target_parameter_elements=1))
    with pytest.raises(MethodSpecError, match="inconsistent"):
        build_lightformer(replace(plan, architecture=replace(plan.architecture, max_positions=10 ** 12)))


def test_dtype_conversion_preserves_fixed_position_semantics_and_cross_group_ties_refuse():
    source = _source(dtype=torch.float32).double()
    model = transfer_lightformer(source, _spec()).model
    model.validate_checkpoint_state(model.state_dict())
    model.encoder_layers[3].self_norm.weight = model.encoder_layers[0].self_norm.weight
    with pytest.raises(ValueError, match="alias semantics"):
        model.validate_checkpoint_state(model.state_dict())
    source = _source()
    source.decoder_layers[1].self_norm.weight = nn.Parameter(source.decoder_layers[0].self_norm.weight.detach())
    with pytest.raises(MethodSpecError, match="storage ties"):
        plan_lightformer(source, _spec())
    model = transfer_lightformer(_source(), _spec()).model
    model._encoder_group_map = model._encoder_group_map.float()
    with pytest.raises(ValueError, match="metadata buffer layout"):
        model.validate_checkpoint_state(model.state_dict())


@pytest.mark.parametrize("change", [{"src_vocab_size": 10 ** 9}, {"max_positions": 10 ** 9},
    {"encoder_groups": list(range(129)), "encoder_sources": list(range(129))}])
def test_standalone_checkpoint_constructor_caps_refuse_before_tensor_allocation(monkeypatch, change):
    config = _source().representation_config()
    config.update(change)
    def forbidden(*args, **kwargs):
        pytest.fail("oversized standalone checkpoint started tensor allocation")
    monkeypatch.setattr(nn.Linear, "__init__", forbidden)
    monkeypatch.setattr(nn.Embedding, "__init__", forbidden)
    monkeypatch.setattr(torch, "empty", forbidden)
    with pytest.raises(ValueError, match="before allocation"):
        LightFormerTransformer.from_config(config)
