"""Independent tiny CPU oracles for coupled reference architecture transforms."""
from copy import deepcopy
import io
import json

import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.coupled_specs import (
    FlatValueOutput, MoDeValueOutput, FlatMLPChannels, MoDeMLPChannels,
    FlatQueryKey, MoDeQueryKey,
)
from fedcore.algorithm.low_rank.coupled_layers import ReferenceAttention, ReferenceMLP
from fedcore.algorithm.low_rank.coupled_profiles import (
    mode_product_factors, mlp_channel_fit, validate_rope_intertwining,
)
from fedcore.algorithm.low_rank.coupled_execution import (
    plan_coupled_profile, prepare_coupled_replacements, CoupledProfileError,
)
from fedcore.algorithm.low_rank.approximation import NumericalDomainViolation


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _rows(width=8, length=5):
    return torch.randn((3, length, width), generator=torch.Generator().manual_seed(318), dtype=torch.float64)


def _attention(groups=2, *, rope=True, bias=True):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        return ReferenceAttention(8, 4, groups, 4, rope=rope, bias=bias, dtype=torch.float64)


def _mlp(*, gated=True, activation="silu", bias=True):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(66)
        return ReferenceMLP(8, 7, gated=gated, activation=activation, bias=bias, dtype=torch.float64)


def _transform(layer, spec, rank, *, data=None, kwargs=None):
    data = _rows() if data is None else data
    plan = plan_coupled_profile(layer, spec, rank=rank, target_paths=[""])
    replacements, evidence = prepare_coupled_replacements(layer, data, spec, plan, calibration_kwargs=kwargs)
    return replacements[""], evidence[0]


def _reload(layer):
    config = json.loads(json.dumps(layer.representation_config()))
    kind = config.pop("layer_type"); config.pop("version")
    cls = {"ReferenceAttention": ReferenceAttention, "ReferenceMLP": ReferenceMLP}[kind]
    with torch.random.fork_rng(devices=[]):
        result = cls(**config, dtype=torch.float64)
    buffer = io.BytesIO(); torch.save(layer.state_dict(), buffer); buffer.seek(0)
    result.load_state_dict(torch.load(buffer, weights_only=True))
    return result


def _independent_rope(states, positions, coords, original=4, base=10000.):
    result = torch.empty_like(states)
    for head in range(states.shape[1]):
        for pair in range(states.shape[-1] // 2):
            angle = positions * base ** (-coords[head][2 * pair] / original)
            even = states[:, head, :, 2 * pair]; odd = states[:, head, :, 2 * pair + 1]
            result[:, head, :, 2 * pair] = even * angle.cos() - odd * angle.sin()
            result[:, head, :, 2 * pair + 1] = even * angle.sin() + odd * angle.cos()
    return result


def _dense_attention_oracle(layer, x, padding=None):
    b, t, _ = x.shape
    q = nn.functional.linear(x, layer.q_proj.weight, layer.q_proj.bias).reshape(b, t, 4, layer.qk_dim).transpose(1, 2)
    k = nn.functional.linear(x, layer.k_proj.weight, layer.k_proj.bias).reshape(b, t, layer.num_kv_heads, layer.qk_dim).transpose(1, 2)
    v = nn.functional.linear(x, layer.v_proj.weight, layer.v_proj.bias).reshape(b, t, layer.num_kv_heads, layer.value_dim).transpose(1, 2)
    if layer.rope:
        positions = torch.arange(t, dtype=x.dtype)[None].expand(b, -1)
        k = _independent_rope(k, positions, layer.group_coordinates.tolist())
        q = _independent_rope(q, positions, layer.group_coordinates[layer.head_map].tolist())
    logits = torch.stack([q[:, h] @ k[:, int(layer.head_map[h])].transpose(-2, -1) for h in range(4)], 1) * layer.logits_scale
    mask = torch.ones((t, t), dtype=torch.bool).tril()[None, None].expand(b, 1, t, t)
    if padding is not None:
        mask = mask & padding[:, None, None]
    probabilities = logits.masked_fill(~mask, -torch.inf).softmax(-1)
    contexts = torch.stack([probabilities[:, h] @ v[:, int(layer.head_map[h])] for h in range(4)], 1)
    return logits, nn.functional.linear(contexts.transpose(1, 2).reshape(b, t, -1), layer.o_proj.weight, layer.o_proj.bias)


@pytest.mark.parametrize("groups", [2, 4])
def test_reference_attention_matches_independent_rope_gqa_mask_formula_and_decode(groups):
    layer = _attention(groups); x = _rows()
    padding = torch.ones((3, 5), dtype=torch.bool); padding[1, -1] = False
    answer = layer.attention(x, padding_mask=padding, use_cache=True)
    logits, outputs = _dense_attention_oracle(layer, x, padding)
    torch.testing.assert_close(answer.logits, logits)
    torch.testing.assert_close(answer.output, outputs)
    prefill = layer.attention(x[:, :3], use_cache=True)
    decode = layer.attention(x[:, 3:], cache=prefill.cache, use_cache=True)
    torch.testing.assert_close(decode.output, layer(x)[:, 3:])
    assert decode.cache.keys.shape == (3, groups, 5, 4)


@pytest.mark.parametrize("spec", [FlatValueOutput(), FlatValueOutput(observation_point="post_softmax"), MoDeValueOutput()])
@pytest.mark.parametrize("groups", [2, 4])
def test_vo_full_width_preserves_function_and_export(spec, groups):
    layer = _attention(groups); x = _rows()
    transformed, record = _transform(layer, spec, 4)
    torch.testing.assert_close(transformed(x), layer(x))
    torch.testing.assert_close(_reload(transformed)(x), layer(x))
    assert record["diagnostics"]["group_ranks"] == [4] * groups
    if isinstance(spec, FlatValueOutput):
        assert not torch.equal(transformed.v_proj.weight, layer.v_proj.weight)


def test_flat_projection_absorption_is_explicit_and_reduces_actual_value_cache():
    layer = _attention(); x = _rows()
    transformed, record = _transform(layer, FlatValueOutput(), 2)
    before = layer.attention(x, use_cache=True); after = transformed.attention(x, use_cache=True)
    contexts = before.contexts.clone()
    for group_info in record["diagnostics"]["groups"]:
        basis = x.new_tensor(group_info["basis"])
        for h in group_info["query_heads"]:
            contexts[:, h] = contexts[:, h] @ basis @ basis.T
    expected = layer.o_proj(contexts.transpose(1, 2).reshape(3, 5, -1))
    torch.testing.assert_close(after.output, expected)
    assert transformed.v_proj.weight.shape == (4, 8)
    assert transformed.o_proj.weight.shape == (8, 8)
    assert after.cache.values.shape == (3, 2, 5, 2)
    assert after.cache.tensor_bytes < before.cache.tensor_bytes
    torch.testing.assert_close(_reload(transformed).attention(x, use_cache=True).cache.values, after.cache.values)
    prefill = transformed.attention(x[:, :3], use_cache=True)
    decoded = transformed.attention(x[:, 3:], cache=prefill.cache)
    torch.testing.assert_close(decoded.output, after.output[:, 3:])


def test_mode_product_thin_factors_match_dense_weighted_svd_and_differ_from_value_pca():
    a = torch.diag(torch.tensor([.01, 100.], dtype=torch.float64))
    b = torch.diag(torch.tensor([10., 1.], dtype=torch.float64)); x = torch.eye(2, dtype=torch.float64)
    fitted = mode_product_factors(a, b, x, 1)
    u, s, vh = torch.linalg.svd(x @ (a @ b).T)
    oracle = (u[:, :1] * s[:1]) @ vh[:1]
    torch.testing.assert_close(x @ fitted.matrix().T, oracle)
    assert fitted.diagnostics["product_materialized"] is False
    assert fitted.matrix()[1, 1] > 99
    # Value PCA retains e1; its product reconstruction instead retains .1.
    assert float(((a @ b) - a[:, :1] @ b[:1]).square().sum()) > 9999


def test_mode_gqa_product_objective_matches_independent_dense_group_reference():
    layer = _attention(bias=False); x = _rows()
    transformed, record = _transform(layer, MoDeValueOutput(), 2, data=x)
    rows = x.reshape(-1, 8)
    for group, details in enumerate(record["diagnostics"]["groups"]):
        heads = details["query_heads"]
        old_left = torch.cat([layer.o_proj.weight[:, h * 4:(h + 1) * 4] for h in heads])
        new_left = torch.cat([transformed.o_proj.weight[:, h * 2:(h + 1) * 2] for h in heads])
        target = rows @ layer.v_proj.weight[group * 4:(group + 1) * 4].T @ old_left.T
        actual = rows @ transformed.v_proj.weight[group * 2:(group + 1) * 2].T @ new_left.T
        u, s, vh = torch.linalg.svd(target, full_matrices=False)
        torch.testing.assert_close(actual, (u[:, :2] * s[:2]) @ vh[:2])
        assert details["variant_status"] == "reconstructed_joint_gqa_product"


@pytest.mark.parametrize("gated,activation", [(True, "silu"), (True, "gelu"), (False, "relu"), (False, "gelu")])
@pytest.mark.parametrize("spec", [FlatMLPChannels(), MoDeMLPChannels()])
def test_mlp_full_channel_parity_preserves_original_nonlinearity_and_bias(gated, activation, spec):
    layer = _mlp(gated=gated, activation=activation); x = _rows()
    answer, _ = _transform(layer, spec, 7)
    torch.testing.assert_close(answer(x), layer(x))
    torch.testing.assert_close(_reload(answer)(x), layer(x))
    assert answer.activation == activation and answer.gated == gated


@pytest.mark.parametrize("spec", [FlatMLPChannels(indices=(1, 4, 6)), MoDeMLPChannels(indices=(1, 4, 6))])
def test_mlp_selected_real_channels_and_fixed_design_lstsq_oracle(spec):
    layer = _mlp(); x = _rows()
    compressed, record = _transform(layer, spec, 3, data=x)
    indices = list(spec.indices)
    design = (nn.functional.silu(layer.gate_proj(x)[..., indices]) * layer.up_proj(x)[..., indices]).reshape(-1, 3)
    target = (layer(x) - layer.down_proj.bias).reshape(-1, 8)
    independent = torch.linalg.lstsq(design, target, driver="gelsd").solution
    torch.testing.assert_close(compressed.down_proj.weight.T, independent)
    torch.testing.assert_close(compressed(x), (design @ independent).reshape(3, 5, 8) + layer.down_proj.bias)
    torch.testing.assert_close(compressed.up_proj.weight, layer.up_proj.weight[indices])
    torch.testing.assert_close(compressed.gate_proj.weight, layer.gate_proj.weight[indices])
    details = record["diagnostics"]
    assert details["gate_indices"] == details["up_indices"] == indices
    assert details["residual_squared"] <= details["baseline_residual_squared"] + 1e-10
    torch.testing.assert_close(_reload(compressed)(x), compressed(x))


def test_mlp_dependent_features_and_ridge_have_explicit_independent_objectives():
    h = torch.tensor([[1., 1., 3.], [2., 2., 1.], [3., 3., -1.]], dtype=torch.float64)
    w = torch.tensor([[1., 2., 3.], [-2., 1., 1.]], dtype=torch.float64)
    fitted = mlp_channel_fit(h, w, 2, indices=(0, 1))
    torch.testing.assert_close(fitted.down_weight.T, torch.linalg.lstsq(h[:, :2], h @ w.T, driver="gelsd").solution)
    regularized = mlp_channel_fit(h, w, 2, indices=(0, 1), ridge=.7)
    oracle = torch.linalg.solve(h[:, :2].T @ h[:, :2] + .7 * torch.eye(2, dtype=h.dtype), h[:, :2].T @ h @ w.T)
    torch.testing.assert_close(regularized.down_weight.T, oracle)
    assert regularized.diagnostics["objective"].endswith("ridge")


@pytest.mark.parametrize("spec", [FlatQueryKey(), MoDeQueryKey()])
@pytest.mark.parametrize("groups", [2, 4])
def test_qk_full_width_preserves_logits_probabilities_outputs_and_export(spec, groups):
    layer = _attention(groups); x = _rows()
    answer, _ = _transform(layer, spec, 4)
    before = layer.attention(x); after = answer.attention(x)
    torch.testing.assert_close(after.logits, before.logits)
    torch.testing.assert_close(after.probabilities, before.probabilities)
    torch.testing.assert_close(after.output, before.output)
    torch.testing.assert_close(_reload(answer)(x), before.output)


@pytest.mark.parametrize("spec", [FlatQueryKey(coordinates=(2, 3)), MoDeQueryKey(coordinates=(2, 3))])
def test_qk_retains_original_pair_frequencies_scale_and_compact_key_cache(spec):
    layer = _attention(); x = _rows()
    compressed, record = _transform(layer, spec, 2)
    result = compressed.attention(x, use_cache=True)
    logits, outputs = _dense_attention_oracle(compressed, x)
    torch.testing.assert_close(result.logits, logits)
    torch.testing.assert_close(result.output, outputs)
    assert compressed.logits_scale == layer.logits_scale == .5
    assert compressed.group_coordinates.tolist() == [[2, 3], [2, 3]]
    assert result.cache.keys.shape == (3, 2, 5, 2)
    assert record["diagnostics"]["qk_cache_width"] == 2
    loaded = _reload(compressed)
    prefill = loaded.attention(x[:, :2], use_cache=True)
    decode = loaded.attention(x[:, 2:], cache=prefill.cache)
    torch.testing.assert_close(decode.output, result.output[:, 2:])


def test_rope_generator_identity_accepts_pair_selector_rejects_noncommuting_rotation():
    frequencies = torch.tensor([1., .01], dtype=torch.float64)
    selector = torch.eye(4, dtype=torch.float64)[:, 2:]
    assert validate_rope_intertwining(selector, frequencies, frequencies[1:])
    rotation = torch.eye(4, dtype=torch.float64)[:, [0, 2]]
    with pytest.raises(NumericalDomainViolation, match="does not satisfy"):
        validate_rope_intertwining(rotation, frequencies, frequencies[:1])


@pytest.mark.parametrize("spec,rank", [(FlatQueryKey(coordinates=(0, 2)), 2), (MoDeQueryKey(), 3), (FlatValueOutput(), 5)])
def test_invalid_coupled_requests_refuse_without_mutation_or_rng_consumption(spec, rank):
    model = _attention(); before = deepcopy(model.state_dict()); rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        _transform(model, spec, rank)
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
    assert not model._forward_pre_hooks


def test_mask_refusal_and_bad_gqa_map_are_atomic_and_restore_hooks_modes_rng():
    model = _attention(); model.train(); before = deepcopy(model.state_dict()); rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="attention_mask"):
        _transform(model, FlatValueOutput(), 2, kwargs={"attention_mask": torch.ones(5, 5)})
    assert model.training and not model._forward_pre_hooks and torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    model.head_map[0] = 2
    with pytest.raises(CoupledProfileError, match="head map"):
        plan_coupled_profile(model, MoDeQueryKey(), rank=2, target_paths=[""])


def test_success_preserves_source_graph_modes_rng_and_parameter_trainability():
    model = _attention(); model.train(); model.q_proj.weight.requires_grad_(False)
    before = deepcopy(model.state_dict()); rng = torch.get_rng_state().clone()
    answer, _ = _transform(model, FlatValueOutput(), 2)
    assert model.training and answer.training and torch.equal(rng, torch.get_rng_state())
    assert not answer.q_proj.weight.requires_grad
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())


def test_plan_rejects_arbitrary_architectures_ties_and_tiny_workspace():
    with pytest.raises(CoupledProfileError, match="exact declared"):
        plan_coupled_profile(nn.Linear(8, 8), FlatValueOutput(), rank=2, target_paths=[""])
    model = _attention()
    plan = plan_coupled_profile(model, FlatValueOutput(), rank=2, target_paths=[""])
    with pytest.raises(CoupledProfileError, match="scratch"):
        prepare_coupled_replacements(model, _rows(), FlatValueOutput(), plan, max_workspace_bytes=1)


def test_nested_current_inputs_are_captured_without_inserting_nonlinearities():
    with torch.random.fork_rng(devices=[]):
        model = nn.Sequential(nn.Linear(8, 8, dtype=torch.float64), _mlp())
    x = _rows(); plan = plan_coupled_profile(model, MoDeMLPChannels(indices=(0, 2, 4)), rank=3, target_paths=["1"])
    mapping, records = prepare_coupled_replacements(model, x, MoDeMLPChannels(indices=(0, 2, 4)), plan)
    target = model[1]; current = model[0](x)
    design = target.features(current)[..., [0, 2, 4]].reshape(-1, 3)
    y = (target(current) - target.down_proj.bias).reshape(-1, 8)
    torch.testing.assert_close(mapping["1"].down_proj.weight.T, torch.linalg.lstsq(design, y, driver="gelsd").solution)
    assert records[0]["path"] == "1"


@pytest.mark.parametrize("spec", [FlatValueOutput(), MoDeValueOutput(), FlatMLPChannels(),
                                   MoDeMLPChannels(), FlatQueryKey(), MoDeQueryKey()])
def test_float32_profile_observations_and_compact_factors_remain_finite(spec):
    layer = (_mlp() if isinstance(spec, (FlatMLPChannels, MoDeMLPChannels)) else _attention()).float()
    x = _rows().float()
    answer, evidence = _transform(layer, spec, 2, data=x)
    assert all(p.dtype == torch.float32 and bool(torch.isfinite(p).all()) for p in answer.parameters())
    assert bool(torch.isfinite(answer(x)).all())
    json.dumps(evidence, allow_nan=False)
