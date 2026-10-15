"""Independent LED/attention reference, from-scratch training and passports."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
import json
import math
import random

from hypothesis import given, settings, strategies as st
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fedcore.algorithm.low_rank.lrt_specs import LRT, lrt_spec_from_config
from fedcore.algorithm.low_rank.lrt_layers import (
    LEDLinear, LRTAttention, LRTProfileError, LRTTransformer)
from fedcore.algorithm.low_rank.lrt_profiles import (
    AUTHOR_COMMIT, build_lrt, build_dense_lrt, lrt_costs, plan_lrt)
from fedcore.algorithm.low_rank.method_errors import MethodSpecError
from fedcore.algorithm.low_rank.plans import Goal


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def tiny(**changes):
    values = dict(vocab_size=11, d_model=5, num_heads=2, d_k=3, d_v=2,
                  d_ff=7, num_layers=2, max_seq_len=8, q_rank=2, k_rank=1,
                  v_rank=2, out_rank=2, ffn_in_rank=2, ffn_out_rank=3, seed=17)
    return LRT(**{**values, **changes})


def tokens():
    return torch.tensor([[1, 2, 4, 6, 3], [2, 7, 1, 8, 5]], dtype=torch.int64)


@pytest.mark.parametrize("changes", [
    {"vocab_size": 0}, {"vocab_size": True}, {"vocab_size": 8193},
    {"d_model": -1}, {"d_model": 1.5}, {"d_model": 10**100},
    {"num_heads": 0}, {"num_heads": 17}, {"d_k": 0}, {"d_v": 0},
    {"num_layers": 9}, {"max_seq_len": 513}, {"q_rank": 0}, {"q_rank": 4},
    {"k_rank": 4}, {"v_rank": 3}, {"out_rank": 5}, {"ffn_in_rank": 6},
    {"ffn_out_rank": 6}, {"bias": 1}, {"seed": -1}, {"seed": True}, {"seed": 2**63},
    {"initialization": "svd"}, {"initialization": None}, {"max_parameters": 1},
    {"max_tensor_bytes": 0}, {"max_peak_bytes": True}, {"max_parameters": 10_000_001},
])
def test_invalid_architecture_and_rank_boundaries_are_refused(changes):
    with pytest.raises(MethodSpecError):
        tiny(**changes)


def test_spec_is_immutable_and_requires_exact_json_description():
    spec = tiny()
    with pytest.raises(FrozenInstanceError):
        spec.q_rank = 1
    assert lrt_spec_from_config(json.loads(json.dumps(asdict(spec)))) == spec
    for config in ({}, {**asdict(spec), "pretrained": True}, {**asdict(spec), "q_rank": False}):
        with pytest.raises(MethodSpecError):
            lrt_spec_from_config(config)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("factorized", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_plan_counts_whole_model_parameters_and_fixed_buffers(dtype, factorized, bias):
    spec = tiny(bias=bias)
    plan = plan_lrt(spec, dtype=dtype, factorized=factorized)
    model = LRTTransformer(spec, factorized=factorized, dtype=dtype)
    assert plan.goal is Goal.BUILD_ARCHITECTURE
    assert plan.parameter_count == sum(p.numel() for p in model.parameters())
    assert plan.tensor_bytes == sum(t.numel()*t.element_size() for t in model.state_dict().values())
    assert plan.config() == model.representation_config()
    assert model(tokens()).shape == (2, 5, spec.vocab_size)
    assert model(tokens()).dtype == dtype
    assert all(p.requires_grad for p in model.parameters())
    costs = lrt_costs(spec, factorized=factorized, dtype=dtype, batch_size=2, sequence_length=5)
    assert costs["parameter_count"] == plan.parameter_count
    assert costs["attention_score_elements_per_layer"] == 2*spec.num_heads*5*5
    assert costs["attention_macs"] == 2*spec.num_layers*spec.num_heads*25*(spec.d_k+spec.d_v)
    assert costs["forward_macs"] == costs["linear_macs"]+costs["attention_macs"]


def test_rank_does_not_change_head_widths_attention_size_or_hypothetical_kv():
    spec = tiny()
    smaller = replace(spec, q_rank=1, v_rank=1, out_rank=1, ffn_in_rank=1, ffn_out_rank=1)
    large_cost, small_cost = lrt_costs(spec), lrt_costs(smaller)
    assert small_cost["parameter_count"] < large_cost["parameter_count"]
    for name in ("attention_score_elements_per_layer", "hypothetical_kv_elements_all_layers", "attention_macs"):
        assert small_cost[name] == large_cost[name]
    for current in (spec, smaller):
        attention = LRTAttention(current, dtype=torch.float64)
        inputs = torch.arange(2*4*current.d_model, dtype=torch.float64).reshape(2, 4, -1)/31
        output, probabilities = attention(inputs, return_attention=True)
        assert output.shape == inputs.shape
        assert probabilities.shape == (2, current.num_heads, 4, 4)
        assert all(layer.out_features == current.d_k for layer in attention.q_proj)
        assert all(layer.out_features == current.d_v for layer in attention.v_proj)
    assert large_cost["kv_cache_implemented"] is False


def test_full_rank_is_allowed_without_claiming_parameter_compression():
    spec = tiny(q_rank=3, k_rank=3, v_rank=2, out_rank=4, ffn_in_rank=5, ffn_out_rank=5)
    plan = plan_lrt(spec)
    assert plan.parameter_count > plan.dense_parameter_count


def test_led_is_two_linear_factors_without_hidden_activation():
    layer = LEDLinear(2, 2, 1, dtype=torch.float64)
    with torch.no_grad():
        layer.encoder.weight.copy_(torch.tensor([[1., -2.]], dtype=torch.float64))
        layer.decoder.weight.copy_(torch.tensor([[3.], [-4.]], dtype=torch.float64))
        layer.decoder.bias.copy_(torch.tensor([.5, -.25], dtype=torch.float64))
    data = torch.tensor([[-1., 1.], [2., -3.]], dtype=torch.float64)
    actual = layer(data)
    expected = data @ layer.encoder.weight.T @ layer.decoder.weight.T+layer.decoder.bias
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual[0, 0] < 0  # an intermediate ReLU would destroy this result
    clone = LEDLinear.from_config(layer.representation_config(), dtype=torch.float64)
    clone.load_state_dict(layer.state_dict())
    torch.testing.assert_close(clone(data), actual, rtol=0, atol=0)
    assert layer.encoder.bias is None


def scalar_projection(layer, vector):
    """Python scalar reference; does not call a module or torch matmul/softmax."""
    def linear(weights, bias, values):
        return [sum(w*x for w, x in zip(row, values))+(0.0 if bias is None else bias[i])
                for i, row in enumerate(weights)]
    if type(layer) is LEDLinear:
        hidden = linear(layer.encoder.weight.detach().tolist(), None, vector)
        return linear(layer.decoder.weight.detach().tolist(),
                      None if layer.decoder.bias is None else layer.decoder.bias.detach().tolist(), hidden)
    return linear(layer.weight.detach().tolist(),
                  None if layer.bias is None else layer.bias.detach().tolist(), vector)


def scalar_attention(attention, inputs, attention_mask, padding, causal):
    spec = attention.spec
    values = inputs.tolist()
    expected, all_probabilities = [], []
    for b, sequence in enumerate(values):
        head_outputs, head_probabilities = [], []
        for h in range(spec.num_heads):
            q = [scalar_projection(attention.q_proj[h], x) for x in sequence]
            k = [scalar_projection(attention.k_proj[h], x) for x in sequence]
            v = [scalar_projection(attention.v_proj[h], x) for x in sequence]
            outputs, probabilities = [], []
            for i in range(len(sequence)):
                allowed = [j for j in range(len(sequence)) if not (
                    (causal and j > i) or bool(attention_mask[b, i, j]) or bool(padding[b, j]))]
                scores = {j: sum(x*y for x, y in zip(q[i], k[j]))/math.sqrt(spec.d_k) for j in allowed}
                maximum = max(scores.values())
                weights = {j: math.exp(score-maximum) for j, score in scores.items()}
                total = sum(weights.values())
                row = [weights.get(j, 0.0)/total for j in range(len(sequence))]
                probabilities.append(row)
                outputs.append([sum(row[j]*v[j][c] for j in range(len(sequence))) for c in range(spec.d_v)])
            head_outputs.append(outputs)
            head_probabilities.append(probabilities)
        expected.append([scalar_projection(attention.out_proj,
            [value for h in range(spec.num_heads) for value in head_outputs[h][i]]) for i in range(len(sequence))])
        all_probabilities.append(head_probabilities)
    return torch.tensor(expected, dtype=inputs.dtype), torch.tensor(all_probabilities, dtype=inputs.dtype)


@pytest.mark.parametrize("dtype, tolerance", [(torch.float32, 3e-6), (torch.float64, 3e-14)])
@pytest.mark.parametrize("factorized", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_attention_matches_independent_scalar_bilinear_reference(dtype, tolerance, factorized, causal):
    spec = tiny(num_layers=1)
    attention = LRTAttention(spec, factorized=factorized, dtype=dtype)
    inputs = torch.arange(2*4*spec.d_model, dtype=dtype).reshape(2, 4, -1)/11-1.3
    # Nonzero output biases test their exact placement in the pair.
    with torch.no_grad():
        for name, parameter in attention.named_parameters():
            if "bias" in name:
                parameter.copy_(torch.linspace(-.2, .3, parameter.numel(), dtype=dtype))
    mask = torch.zeros(2, 4, 4, dtype=torch.bool)
    mask[0, 2, 1] = True
    mask[1, 3, 2] = True
    padding = torch.tensor([[False, False, False, True], [False, False, False, False]])
    expected, expected_probabilities = scalar_attention(attention, inputs, mask, padding, causal)
    actual, probabilities = attention(inputs, attention_mask=mask, key_padding_mask=padding,
                                      causal=causal, return_attention=True)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(probabilities, expected_probabilities, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones_like(probabilities.sum(-1)),
                               atol=tolerance, rtol=tolerance)
    assert torch.count_nonzero(probabilities[0, :, :, 3]) == 0


@pytest.mark.parametrize("dtype, tolerance", [(torch.float32, 2e-6), (torch.float64, 2e-13)])
@pytest.mark.parametrize("factorized", [False, True])
def test_full_model_is_causal_and_prefix_agrees_with_short_sequence(dtype, tolerance, factorized):
    spec = tiny()
    model = LRTTransformer(spec, factorized=factorized, dtype=dtype)
    data = tokens()
    changed = data.clone()
    changed[:, 3:] = (changed[:, 3:]+3) % spec.vocab_size
    with torch.no_grad():
        original = model(data)
        changed_output = model(changed)
        short = model(data[:, :3])
    torch.testing.assert_close(original[:, :3], changed_output[:, :3], atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(original[:, :3], short, atol=tolerance, rtol=tolerance)
    assert not torch.allclose(original[:, 3:], changed_output[:, 3:])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_every_factor_has_real_finite_gradient_and_optimizer_update(dtype):
    model = build_lrt(tiny(), dtype=dtype).model
    data = tokens()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    logits = model(data)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, model.spec.vocab_size), data[:, 1:].reshape(-1))
    loss.backward()
    for name, parameter in model.named_parameters():
        if "encoder.weight" in name or "decoder.weight" in name:
            assert parameter.grad is not None, name
            assert bool(torch.isfinite(parameter.grad).all()), name
            assert torch.count_nonzero(parameter.grad) > 0, name
    optimizer.step()
    for name, parameter in model.named_parameters():
        if "encoder.weight" in name or "decoder.weight" in name:
            assert not torch.equal(parameter, before[name]), name
    assert bool(torch.isfinite(model(data)).all())


def test_build_and_training_never_call_svd(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("From-scratch construction must not invoke SVD")
    for owner, name in ((torch.linalg, "svd"), (torch.linalg, "svdvals"), (torch, "svd")):
        monkeypatch.setattr(owner, name, forbidden)
    for builder in (build_lrt, build_dense_lrt):
        result = builder(tiny())
        optimizer = torch.optim.SGD(result.model.parameters(), lr=.01)
        output = result.model(tokens())
        F.cross_entropy(output[:, :-1].reshape(-1, 11), tokens()[:, 1:].reshape(-1)).backward()
        optimizer.step()
        assert result.evidence["source_weights_required"] is False
        assert result.evidence["svd_required"] is False


@pytest.mark.parametrize("initialization", ["xavier_uniform", "xavier_normal"])
def test_seed_determinism_isolated_from_global_rng_and_default_dtype(initialization):
    spec = tiny(initialization=initialization)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    first = build_lrt(spec).model
    dense = build_dense_lrt(spec).model
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert numpy_state[0] == after_numpy[0]
    np.testing.assert_array_equal(numpy_state[1], after_numpy[1])
    assert numpy_state[2:] == after_numpy[2:]
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        second = build_lrt(spec).model
    finally:
        torch.set_default_dtype(default_dtype)
    for name, parameter in first.named_parameters():
        assert parameter.dtype == torch.float32
        torch.testing.assert_close(parameter, dict(second.named_parameters())[name], rtol=0, atol=0)
    assert first.embedding.weight.dtype == dense.embedding.weight.dtype
    other_seed = build_lrt(replace(spec, seed=spec.seed+1)).model
    assert not torch.equal(first.embedding.weight, other_seed.embedding.weight)
    q = first.blocks[0].attention.q_proj
    assert not torch.equal(q[0].encoder.weight, q[1].encoder.weight)
    first.validate_checkpoint_state(first.state_dict())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("factorized", [False, True])
def test_trained_architecture_config_and_state_round_trip(dtype, factorized):
    model = LRTTransformer(tiny(), factorized=factorized, dtype=dtype)
    optimizer = torch.optim.SGD(model.parameters(), lr=.03)
    output = model(tokens())
    F.cross_entropy(output[:, :-1].reshape(-1, 11), tokens()[:, 1:].reshape(-1)).backward()
    optimizer.step()
    config = json.loads(json.dumps(model.representation_config()))
    restored = LRTTransformer.from_config(config, dtype=dtype)
    state = deepcopy(model.state_dict())
    restored.validate_checkpoint_state({"model."+k: v for k, v in state.items()} | {"sibling.value": torch.ones(1)}, "model.")
    restored.load_state_dict(state)
    torch.testing.assert_close(model(tokens()), restored(tokens()), rtol=0, atol=0)
    assert restored.factorized == factorized
    assert sum(type(module) is LEDLinear for module in restored.modules()) == (
        model.spec.num_layers*(3*model.spec.num_heads+3) if factorized else 0)


def test_dtype_conversion_preserves_canonical_position_buffer_on_reload():
    model = build_lrt(tiny()).model.to(dtype=torch.float64)
    restored = LRTTransformer.from_config(model.representation_config(), dtype=torch.float64)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(tokens()), restored(tokens()), rtol=0, atol=0)


@pytest.mark.parametrize("mutation", ["position", "metadata", "missing", "extra", "nan", "dtype", "shape"])
def test_checkpoint_refuses_corrupt_state_before_loading(mutation):
    model = build_lrt(tiny()).model
    state = deepcopy(model.state_dict())
    if mutation == "position":
        state["position_encoding"][0, 0] += 1
    elif mutation == "metadata":
        state["_architecture_metadata"][2] += 1
    elif mutation == "missing":
        state.pop("blocks.0.attention.q_proj.0.encoder.weight")
    elif mutation == "extra":
        state["unexpected.weight"] = torch.ones(1)
    elif mutation == "nan":
        state["output.weight"][0, 0] = torch.nan
    elif mutation == "dtype":
        state["output.weight"] = state["output.weight"].to(torch.float64)
    else:
        state["output.weight"] = state["output.weight"][:1]
    before = deepcopy(model.state_dict())
    with pytest.raises(LRTProfileError):
        model.load_state_dict(state, strict=False)
    for name, tensor in before.items():
        assert torch.equal(tensor, model.state_dict()[name])


@pytest.mark.parametrize("mutation", ["linear_width", "norm_eps", "embedding_options", "shared", "class"])
def test_checkpoint_refuses_executable_attributes_missing_from_descriptor(mutation):
    model = build_lrt(tiny()).model
    if mutation == "linear_width":
        model.blocks[0].attention.q_proj[0].encoder.in_features += 1
    elif mutation == "norm_eps":
        model.blocks[0].attention_norm.eps = .1
    elif mutation == "embedding_options":
        model.embedding.max_norm = 1
    elif mutation == "shared":
        model.blocks[0].attention.q_proj[1] = model.blocks[0].attention.q_proj[0]
    else:
        class ChangedLinear(nn.Linear):
            pass
        model.output.__class__ = ChangedLinear
    with pytest.raises(LRTProfileError):
        model.validate_checkpoint_state(model.state_dict())


@pytest.mark.parametrize("argument", ["source", "calibration"])
def test_from_scratch_refuses_source_and_calibration_before_allocation(monkeypatch, argument):
    def forbidden(*args, **kwargs):
        raise AssertionError("Must reject before constructing tensors")
    monkeypatch.setattr(torch, "empty", forbidden)
    with pytest.raises(LRTProfileError):
        build_lrt(tiny(), **{argument: object()})


@pytest.mark.parametrize("changes", [{"max_tensor_bytes": 1}, {"max_peak_bytes": 1}])
def test_persistent_and_build_memory_limits_are_checked_before_allocation(monkeypatch, changes):
    spec = tiny(**changes)
    def forbidden(*args, **kwargs):
        raise AssertionError("Must plan before constructing tensors")
    monkeypatch.setattr(torch, "empty", forbidden)
    with pytest.raises(LRTProfileError):
        build_lrt(spec)


def test_dense_control_checks_its_own_whole_model_budget():
    spec = LRT(max_parameters=2500)
    assert build_lrt(spec).plan.parameter_count == 2032
    with pytest.raises(LRTProfileError, match="parameters"):
        build_dense_lrt(spec)


def test_runtime_batch_budget_precedes_embedding_allocation():
    model = build_lrt(tiny(max_peak_bytes=100_000)).model
    called = []
    handle = model.embedding.register_forward_pre_hook(lambda *args: called.append(True))
    try:
        with pytest.raises(LRTProfileError, match="Forward tensor reserve"):
            model(torch.ones(500, 8, dtype=torch.int64))
    finally:
        handle.remove()
    assert not called


@pytest.mark.parametrize("data", [torch.ones(2, 3), torch.ones(2, 3, dtype=torch.int32),
    torch.ones(2, 3, 1, dtype=torch.int64), torch.empty(0, 3, dtype=torch.int64),
    torch.empty(2, 0, dtype=torch.int64), torch.ones(2, 9, dtype=torch.int64),
    torch.tensor([[1, -1]]), torch.tensor([[11, 1]])])
def test_tokens_validate_dtype_shape_length_and_vocabulary(data):
    with pytest.raises(LRTProfileError):
        build_lrt(tiny()).model(data)


@pytest.mark.parametrize("data", [torch.ones(2, 4), torch.ones(2, 3, 6),
    torch.ones(2, 3, 5, dtype=torch.int64), torch.ones(2, 3, 5, dtype=torch.float64),
    torch.full((2, 3, 5), torch.inf), torch.full((2, 3, 5), torch.nan),
    torch.empty(0, 3, 5), torch.ones(2, 9, 5)])
def test_attention_validates_float_dtype_shape_finiteness_and_length(data):
    with pytest.raises(LRTProfileError):
        LRTAttention(tiny())(data)


@pytest.mark.parametrize("mask_name, mask", [
    ("attention_mask", torch.zeros(5, 5)), ("attention_mask", torch.zeros(2, 5, dtype=torch.bool)),
    ("attention_mask", torch.zeros(2, 1, 5, 5, dtype=torch.bool)),
    ("key_padding_mask", torch.zeros(2, 4, dtype=torch.bool)),
    ("key_padding_mask", torch.zeros(2, 5)), ("attention_mask", torch.ones(5, 5, dtype=torch.bool)),
    ("key_padding_mask", torch.ones(2, 5, dtype=torch.bool)),
])
def test_attention_masks_reject_ambiguous_or_all_masked_rows(mask_name, mask):
    with pytest.raises(LRTProfileError):
        build_lrt(tiny()).model(tokens(), **{mask_name: mask})


def test_two_dimensional_and_batch_masks_have_same_true_blocks_semantics():
    model = build_lrt(tiny()).model
    mask = torch.zeros(5, 5, dtype=torch.bool)
    mask[4, 2] = True
    with torch.no_grad():
        a = model(tokens(), attention_mask=mask)
        b = model(tokens(), attention_mask=mask.expand(2, -1, -1))
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("dtype, device", [(torch.float16, "cpu"), (torch.int64, "cpu"),
    (torch.float32, "meta"), (torch.float64, "cuda"), (torch.float32, "cpu:1")])
def test_unsupported_device_and_dtype_refused_without_allocation(dtype, device):
    with pytest.raises(LRTProfileError):
        build_lrt(tiny(), dtype=dtype, device=device)


@st.composite
def architecture_specs(draw):
    m = draw(st.integers(2, 8))
    k, v, f = draw(st.integers(1, 5)), draw(st.integers(1, 5)), draw(st.integers(2, 10))
    heads = draw(st.integers(1, 3))
    return LRT(vocab_size=draw(st.integers(3, 12)), d_model=m, num_heads=heads, d_k=k, d_v=v,
        d_ff=f, num_layers=draw(st.integers(1, 2)), max_seq_len=5,
        q_rank=draw(st.integers(1, min(m, k))), k_rank=draw(st.integers(1, min(m, k))),
        v_rank=draw(st.integers(1, min(m, v))), out_rank=draw(st.integers(1, min(heads*v, m))),
        ffn_in_rank=draw(st.integers(1, min(m, f))), ffn_out_rank=draw(st.integers(1, min(m, f))),
        bias=draw(st.booleans()), seed=draw(st.integers(0, 100)))


@given(architecture_specs())
@settings(max_examples=20, deadline=None)
def test_generated_valid_architectures_conserve_plan_and_config(spec):
    model = LRTTransformer(spec)
    plan = plan_lrt(spec)
    assert plan.parameter_count == sum(p.numel() for p in model.parameters())
    assert plan.tensor_bytes == sum(t.numel()*t.element_size() for t in model.state_dict().values())
    config = json.loads(json.dumps(model.representation_config()))
    reconstructed = LRTTransformer.from_config(config)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, reconstructed.state_dict()[name])
    data = torch.arange(5).remainder(spec.vocab_size).unsqueeze(0)
    assert model(data).shape == (1, 5, spec.vocab_size)
    model.validate_checkpoint_state(model.state_dict())


def test_source_evidence_labels_reconstruction_and_defers_asr_metrics():
    result = build_lrt(tiny())
    evidence = json.loads(json.dumps(result.evidence))
    assert evidence["source"]["paper_version"] == "1910.13923v3"
    assert evidence["source"]["author_commit"] == AUTHOR_COMMIT
    assert evidence["source"]["printed_equation_2_corrected"] is True
    assert "reconstruction" in evidence["source"]["status"]
    assert evidence["training_performed"] is False
    assert evidence["initialized_from_scratch"] is True
    assert "cer" not in evidence and "speedup" not in evidence
