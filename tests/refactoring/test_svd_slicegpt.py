"""SliceGPT invariants: whole-graph rotation precedes approximate slicing.

Independent end-to-end logits, projector-energy, changed residual statistics,
checkpoint sharing and cache layouts detect plausible architectural defects.
"""
from copy import deepcopy
import io
import json
import random
import types

import numpy as np
import pytest
import torch
from torch import nn
from hypothesis import given, settings, strategies as st

from fedcore.algorithm.low_rank import slicegpt_profiles as profiles
from fedcore.algorithm.low_rank.slicegpt_layers import (
    SliceGPTBlock, SliceGPTModel, SliceGPTRMSNorm,
)
from fedcore.algorithm.low_rank.slicegpt_specs import SliceGPT


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def _model(dtype=torch.float64, *, tied=True, bias=True, rope=True):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(731)
        model = SliceGPTModel(23, 8, 2, 2, 1, 4, 12,
            tie_word_embeddings=tied, bias=bias, rope=rope, dtype=dtype)
        with torch.no_grad():
            for i, block in enumerate(model.blocks):
                # Nonconstant, nonzero scales cannot commute with general Q.
                block.norm.weight.copy_(torch.linspace(.4 + i / 8, 1.8 + i / 8, 8, dtype=dtype))
            model.final_norm.weight.copy_(torch.linspace(.6, 1.7, 8, dtype=dtype))
        return model


def _data():
    return (torch.tensor([[1, 2, 4], [2, 3, 5]]),
            torch.tensor([[4, 1, 7, 2, 9], [3, 6, 1, 4, 2]]))


def _bases(dtype):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(613)
        return tuple(torch.linalg.qr(torch.randn(8, 8, dtype=dtype))[0] for _ in range(5))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("tied", [True, False])
def test_multiple_full_bases_preserve_complete_causal_function(dtype, tied):
    source = _model(dtype, tied=tied); bases = _bases(dtype)
    before = deepcopy(source.state_dict())
    result = profiles.rotate_slicegpt(source, bases, verification_inputs=_data())
    tolerance = 5e-5 if dtype == torch.float32 else 2e-10
    for batch in _data():
        torch.testing.assert_close(result(batch), source(batch), rtol=tolerance, atol=tolerance)
    for i, block in enumerate(result.blocks):
        torch.testing.assert_close(block.transition.double(), bases[i].double().T @ bases[i + 1].double(),
                                   rtol=tolerance, atol=tolerance)
        assert block.norm.weight is None
    assert result.final_norm.weight is None
    assert (result.lm_head.weight is result.embedding.weight) == tied
    assert all(torch.equal(before[key], value) for key, value in source.state_dict().items())
    assert all(value.data_ptr() not in {p.data_ptr() for p in source.parameters()} for value in result.parameters())


def test_wrong_residual_transition_is_detected_by_independent_logits_oracle():
    source = _model(); result = profiles.rotate_slicegpt(source, _bases(torch.float64))
    with torch.no_grad():
        result.blocks[1].transition.copy_(torch.eye(8, dtype=torch.float64))
    with pytest.raises(profiles.SliceGPTProfileError, match="FullWidthEquivalenceFailed"):
        profiles.verify_full_width(source, result, _data())


def test_arbitrary_rotation_does_not_commute_with_layernorm():
    q = _bases(torch.float64)[0]
    x = torch.tensor([[1., 2., 4., 8., -1., 3., 6., 7.]], dtype=torch.float64)
    norm = nn.LayerNorm(8, elementwise_affine=False, dtype=torch.float64)
    assert not torch.allclose(norm(x @ q), norm(x) @ q, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("tied", [True, False])
def test_physical_slices_have_consistent_shapes_and_honest_evidence(dtype, tied):
    source = _model(dtype, tied=tied)
    widths = (5, 6, 4, 5, 6)
    result = profiles.transform_slicegpt(source, _data(), SliceGPT(widths=widths))
    target = result.model; evidence = result.evidence
    assert target.embedding.weight.shape == (23, 5)
    for i, block in enumerate(target.blocks):
        ki, ko = widths[i:i + 2]
        assert block.transition.shape == (ki, ko)
        assert block.norm.reference_width == 8 and block.norm.width == ki
        if block.kind == "attention":
            assert block.consumer.q_proj.weight.shape == (8, ki)
            assert block.consumer.k_proj.weight.shape == (4, ki)
            assert block.consumer.v_proj.weight.shape == (4, ki)
            assert block.consumer.o_proj.weight.shape == (ko, 8)
        else:
            assert block.consumer.up_proj.weight.shape == (12, ki)
            assert block.consumer.gate_proj.weight.shape == (12, ki)
            assert block.consumer.down_proj.weight.shape == (ko, 12)
    assert target.head_bridge.shape == (6, 5 if tied else 6)
    assert evidence["pre_slicing_equivalence_gate"]["passed"]
    assert evidence["full_width_equivalence"]["passed"]
    assert {v["tokens"] for v in evidence["full_width_equivalence"]["cases"]} >= {1, 3, 7}
    assert evidence["approximation"]["is_width_reduced"]
    assert evidence["approximation"]["calibration_logit_mse"] > 0
    assert evidence["approximation"]["tied_head_extra_projection"] == tied
    assert evidence["kv_layout"]["cache_width_reduced"] is False
    assert evidence["cost"]["unique_parameters"] == result.plan.estimated_unique_parameters
    assert evidence["cost"]["tensor_bytes"] == result.plan.estimated_tensor_bytes
    assert evidence["resources"]["observed_peak_bytes"] is None
    assert evidence["claims"]["quality_guarantee"] is False
    assert all(s["centered"] is False for s in evidence["statistics"])
    json.dumps(evidence, allow_nan=False)


def test_full_width_profile_keeps_exactness_separate_from_slicing():
    source = _model()
    result = profiles.transform_slicegpt(source, _data(), SliceGPT())
    assert not result.evidence["approximation"]["is_width_reduced"]
    assert result.evidence["approximation"]["calibration_logit_mse"] < 1e-22
    torch.testing.assert_close(result.model(_data()[0]), source(_data()[0]), atol=2e-10, rtol=2e-10)


def test_new_width_rmsnorm_uses_original_denominator_and_epsilon():
    norm = SliceGPTRMSNorm(3, reference_width=8, eps=.02, learnable_scale=False, dtype=torch.float64)
    x = torch.tensor([[[1., -2., 3.]]], dtype=torch.float64)
    expected = x / torch.sqrt(x.square().sum(-1, keepdim=True) / 8 + .02)
    torch.testing.assert_close(norm(x), expected, atol=0, rtol=0)
    ordinary = x / torch.sqrt(x.square().mean(-1, keepdim=True) + .02)
    assert not torch.allclose(norm(x), ordinary)


@given(width=st.integers(2, 7), rank_fraction=st.integers(1, 5))
@settings(max_examples=12, deadline=None)
def test_uncentered_projector_energy_matches_discarded_spectrum(width, rank_fraction):
    rows = torch.sin(torch.arange(5 * width, dtype=torch.float64).reshape(5, width) / 3) + 2.0
    rank = min(rank_fraction, width)
    basis, eigenvalues, count = profiles.residual_second_moment_basis([rows[None]])
    retained = basis[:, :rank]
    residual = rows - rows @ retained @ retained.T
    torch.testing.assert_close(residual.square().sum(), eigenvalues[rank:].sum(), atol=2e-11, rtol=2e-9)
    singular_values = torch.linalg.svdvals(rows)
    torch.testing.assert_close(eigenvalues[:len(singular_values)], singular_values.square(), atol=2e-11, rtol=2e-9)
    assert count == 5
    centered_eigenvalues = torch.linalg.eigvalsh((rows - rows.mean(0)).T @ (rows - rows.mean(0))).flip(0)
    assert not torch.allclose(eigenvalues, centered_eigenvalues)


def test_sequential_statistics_follow_current_sliced_upstream_graph():
    source = _model(); batches = _data()
    result = profiles.transform_slicegpt(source, batches, SliceGPT(widths=(3, 5, 4, 5, 4)))
    # Recover the initial embedding basis independently by an overdetermined solve.
    a = torch.linalg.lstsq(source.embedding.weight.detach(), result.model.embedding.weight.detach()).solution
    current = []
    for tokens in batches:
        x = result.model.embedding(tokens).detach() @ a.T
        branch, _ = source.blocks[0].apply_branch(x)
        current.append((x + branch).detach().reshape(-1, 8))
    gram = sum(x.T @ x for x in current)
    expected = torch.linalg.eigvalsh(gram).flip(0).clamp_min(0)
    actual = torch.tensor(result.evidence["statistics"][1]["eigenvalues"], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-10)
    original = [source.blocks[0](source.embedding(tokens)).detach().reshape(-1, 8) for tokens in batches]
    original_spectrum = torch.linalg.eigvalsh(sum(x.T @ x for x in original)).flip(0).clamp_min(0)
    assert not torch.allclose(actual, original_spectrum, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("tied", [True, False])
def test_compact_config_and_state_roundtrip_rebuilds_graph_and_sharing(tied):
    source = _model(tied=tied)
    result = profiles.transform_slicegpt(source, _data(), SliceGPT(widths=(4, 6, 5, 4, 6))).model
    config = json.loads(json.dumps(result.representation_config(), allow_nan=False))
    restored = SliceGPTModel.from_config(config, device="cpu", dtype=torch.float64)
    restored.validate_checkpoint_state(result.state_dict())
    restored.load_state_dict(result.state_dict(), strict=True)
    assert (restored.embedding.weight is restored.lm_head.weight) == tied
    for tokens in _data():
        torch.testing.assert_close(restored(tokens), result(tokens), rtol=0, atol=0)
    # Prefix use is required for nesting in another checked model.
    restored.validate_checkpoint_state({"child." + k: v for k, v in result.state_dict().items()}, "child.")


def test_block_and_norm_configs_roundtrip_without_external_source():
    model = profiles.transform_slicegpt(_model(), _data(), SliceGPT(widths=(4, 6, 5, 4, 6))).model
    for original in (model.blocks[0], model.blocks[1], model.final_norm):
        restored = type(original).from_config(original.representation_config())
        restored.validate_checkpoint_state(original.state_dict()); restored.load_state_dict(original.state_dict())
        assert restored.representation_config() == original.representation_config()


@pytest.mark.parametrize("path", ["head_bridge", "blocks.0.transition", "blocks.0.norm.width_denominator",
                                  "blocks.0.consumer.head_map", "blocks.0.consumer.group_coordinates"])
def test_checkpoint_rejects_tampered_fixed_buffers(path):
    target = profiles.transform_slicegpt(_model(), _data(), SliceGPT(widths=(4, 6, 5, 4, 6))).model
    restored = SliceGPTModel.from_config(target.representation_config())
    state = deepcopy(target.state_dict()); state[path].reshape(-1)[0] += 1
    with pytest.raises(ValueError, match="fixed-buffer"):
        restored.validate_checkpoint_state(state)


def test_checkpoint_rejects_conflicting_tie_nonfinite_and_wrong_dtype():
    model = profiles.transform_slicegpt(_model(), _data(), SliceGPT()).model
    state = {k: v.clone() for k, v in model.state_dict().items()}
    state["lm_head.weight"][0, 0] += 1
    with pytest.raises(ValueError, match="conflicting tied"):
        model.validate_checkpoint_state(state)
    state = deepcopy(model.state_dict()); state["blocks.0.consumer.q_proj.weight"][0, 0] = torch.inf
    with pytest.raises(ValueError, match="Invalid SliceGPT checkpoint"):
        model.validate_checkpoint_state(state)
    with pytest.raises(ValueError, match="dtype conflicts"):
        SliceGPTModel.from_config(model.representation_config(), dtype=torch.float32)


@pytest.mark.parametrize("modification", ["norm", "extra_consumer", "forward", "hook", "sharing", "kv_layout"])
def test_unsupported_graph_rejected_before_constructor_or_execution(monkeypatch, modification):
    source = _model(); called = []
    if modification == "norm":
        source.blocks[0].norm = nn.LayerNorm(8, dtype=torch.float64)
    elif modification == "extra_consumer":
        source.hidden_consumer = nn.Linear(8, 8, dtype=torch.float64)
    elif modification == "forward":
        source.forward = types.MethodType(lambda self, x: called.append("forward"), source)
    elif modification == "hook":
        source.blocks[0].consumer.register_forward_hook(lambda *args: called.append("hook"))
    elif modification == "sharing":
        source.blocks[0].consumer.k_proj.weight = source.blocks[0].consumer.v_proj.weight
    else:
        source.blocks[0].consumer.rope_base = 9999.0
    monkeypatch.setattr(profiles, "_constructor", lambda *args, **kwargs: called.append("copy"))
    with pytest.raises((profiles.SliceGPTProfileError, ValueError)):
        profiles.transform_slicegpt(source, _data(), SliceGPT(widths=(4,) * 5))
    assert not called


def test_gate_failure_prevents_construction_of_any_reduced_width_graph(monkeypatch):
    original = profiles._constructor; widths = []
    def wrong_constructor(source, values, **kwargs):
        widths.append(tuple(values))
        model = original(source, values, **kwargs)
        # Corrupt one original branch, surviving weight installation: norm eps.
        model.blocks[1].norm.eps = 1.0
        return model
    monkeypatch.setattr(profiles, "_constructor", wrong_constructor)
    with pytest.raises(profiles.SliceGPTProfileError, match="FullWidthEquivalenceFailed"):
        profiles.transform_slicegpt(_model(), _data(), SliceGPT(widths=(4,) * 5))
    assert widths == [(8,) * 5]


@pytest.mark.parametrize("case", ["samples", "tokens", "bytes", "width_count", "width_value", "nonfinite"])
def test_resource_and_shape_preflight_refusals(case):
    source = _model(); data = _data()
    kwargs = {"samples": {"max_samples": 1}, "tokens": {"max_tokens": 1},
              "bytes": {"max_peak_bytes": 1}, "width_count": {"widths": (4, 4)},
              "width_value": {"widths": (9,) * 5}, "nonfinite": {}}[case]
    if case == "nonfinite":
        with torch.no_grad():
            source.embedding.weight[0, 0] = torch.nan
    with pytest.raises((profiles.SliceGPTProfileError, ValueError)):
        profiles.transform_slicegpt(source, data, SliceGPT(**kwargs))


def test_transformation_is_deterministic_source_independent_and_rng_isolated():
    source = _model(); source.train(); source.blocks[0].consumer.eval()
    training = {key: value.training for key, value in source.named_modules()}
    before = deepcopy(source.state_dict()); rng = torch.get_rng_state().clone()
    python_rng = random.getstate(); numpy_rng = np.random.get_state()
    spec = SliceGPT(widths=(4, 6, 5, 4, 6))
    first = profiles.transform_slicegpt(source, _data(), spec)
    second = profiles.transform_slicegpt(source, _data(), spec)
    assert torch.equal(torch.get_rng_state(), rng)
    assert random.getstate() == python_rng
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_rng[0] and np.array_equal(after_numpy[1], numpy_rng[1])
    assert after_numpy[2:] == numpy_rng[2:]
    assert all(torch.equal(before[key], value) for key, value in source.state_dict().items())
    assert training == {key: value.training for key, value in source.named_modules()}
    assert all(torch.equal(value, second.model.state_dict()[key]) for key, value in first.model.state_dict().items())
    assert first.evidence["provenance"] == second.evidence["provenance"]
    with torch.no_grad():
        first.model.embedding.weight.add_(1)
    assert all(torch.equal(before[key], value) for key, value in source.state_dict().items())


def test_calibration_provenance_records_order_and_current_state():
    source = _model(); data = _data()
    original = profiles.transform_slicegpt(source, data, SliceGPT(widths=(4,) * 5))
    reversed_data = profiles.transform_slicegpt(source, tuple(reversed(data)), SliceGPT(widths=(4,) * 5))
    assert original.evidence["provenance"]["calibration_fingerprint"] != reversed_data.evidence["provenance"]["calibration_fingerprint"]
    assert len({s["state_fingerprint"] for s in original.evidence["statistics"]}) == 5
    with torch.no_grad():
        source.blocks[1].consumer.up_proj.bias.add_(.01)
    changed = profiles.plan_slicegpt(source, SliceGPT(), data)
    assert changed.source_fingerprint != original.plan.source_fingerprint


def test_compact_causal_cache_has_true_original_kv_width_and_matches_full_sequence():
    source = _model()
    target = profiles.transform_slicegpt(source, _data(), SliceGPT(widths=(4, 6, 5, 4, 6))).model
    tokens = _data()[1]
    prefix = target.causal_forward(tokens[:, :3], use_cache=True)
    suffix = target.causal_forward(tokens[:, 3:], caches=prefix.caches, use_cache=True)
    torch.testing.assert_close(suffix.logits, target(tokens)[:, 3:], atol=2e-10, rtol=2e-10)
    assert len(suffix.caches) == 2
    for cache in suffix.caches:
        assert cache.keys.shape == (2, 1, 5, 4)
        assert cache.values.shape == (2, 1, 5, 4)
    assert sum(c.tensor_bytes for c in suffix.caches) == sum(
        c.tensor_bytes for c in source.causal_forward(tokens, use_cache=True).caches)


@pytest.mark.parametrize("setting", ["eps", "rope_base", "activation", "extra", "parameter"])
def test_config_refuses_child_behavior_omitted_from_root_configuration(setting):
    model = _model()
    if setting == "eps":
        model.blocks[0].norm.eps *= 10
    elif setting == "rope_base":
        model.blocks[0].consumer.rope_base = 20000.0
    elif setting == "activation":
        model.blocks[1].consumer.activation = "relu"
    elif setting == "extra":
        model.blocks[1].consumer.extra = nn.Linear(8, 8, dtype=torch.float64)
    else:
        model.blocks[1].consumer.register_parameter("extra", nn.Parameter(torch.ones(1, dtype=torch.float64)))
    with pytest.raises(ValueError, match="SliceGPT representation"):
        model.representation_config()


def test_biasless_non_rope_graph_is_declared_and_works():
    source = _model(bias=False, rope=False)
    result = profiles.transform_slicegpt(source, _data(), SliceGPT(widths=(4,) * 5))
    assert result.model.lm_head.bias is None
    assert not result.model.blocks[0].consumer.rope
    assert result.evidence["full_width_equivalence"]["passed"]


def test_verification_cannot_be_disabled_by_loose_or_nonfinite_tolerances():
    source = _model(); rotated = profiles.rotate_slicegpt(source, _bases(torch.float64))
    for tolerance in (1.0, torch.inf, float("nan"), -1.0):
        with pytest.raises(profiles.SliceGPTProfileError, match="tolerances"):
            profiles.verify_full_width(source, rotated, rtol=tolerance)


def test_zero_second_moment_has_finite_complete_basis_without_division_by_zero():
    states = torch.zeros(2, 3, 4, dtype=torch.float64)
    basis, spectrum, count = profiles.residual_second_moment_basis([states])
    torch.testing.assert_close(basis.T @ basis, torch.eye(4, dtype=torch.float64))
    assert torch.equal(spectrum, torch.zeros(4, dtype=torch.float64)) and count == 6


def test_traversal_override_is_refused_without_executing_it():
    source = _model(); called = []
    source.named_modules = types.MethodType(lambda *args, **kwargs: called.append("traversal"), source)
    with pytest.raises(profiles.SliceGPTProfileError, match="overrides"):
        profiles.transform_slicegpt(source, _data(), SliceGPT())
    assert not called


@pytest.mark.parametrize("field,value", [("vocab_size", 10**10), ("d_model", 10**10),
    ("num_layers", 10**10), ("num_heads", 10**10), ("num_kv_heads", 10**10),
    ("head_dim", 10**10), ("intermediate_size", 10**10), ("vocab_size", 200000)])
def test_untrusted_config_resource_caps_precede_all_module_allocators(monkeypatch, field, value):
    config = _model().representation_config()
    config[field] = value
    def forbidden(*args, **kwargs):
        pytest.fail("Module allocation occurred before rejecting an unbounded configuration")
    monkeypatch.setattr(nn, "Embedding", forbidden)
    monkeypatch.setattr(nn, "Linear", forbidden)
    with pytest.raises(ValueError, match="bounded|widths"):
        SliceGPTModel.from_config(config)


def test_direct_model_and_standalone_layers_have_preallocation_caps(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Tensor allocation occurred before rejecting unbounded dimensions")
    monkeypatch.setattr(torch, "ones", forbidden)
    monkeypatch.setattr(torch, "tensor", forbidden)
    monkeypatch.setattr(nn, "Embedding", forbidden)
    monkeypatch.setattr(nn, "Linear", forbidden)
    with pytest.raises(ValueError, match="bounded"):
        SliceGPTModel(10**10, 8, 2, 2, 1, 4, 12)
    with pytest.raises(ValueError, match="bounded"):
        SliceGPTRMSNorm(10**10)
    with pytest.raises(ValueError, match="bounded"):
        SliceGPTBlock("attention", 8, 8, 8, num_heads=2, num_kv_heads=1,
                      head_dim=10**10, intermediate_size=12)
    with pytest.raises(ValueError, match="resource limit"):
        SliceGPTBlock("mlp", 1024, 1024, 1024, num_heads=2, num_kv_heads=1,
                      head_dim=4, intermediate_size=4096)


def test_fixed_matrix_description_shape_is_checked_before_tensor_allocation(monkeypatch):
    from fedcore.algorithm.low_rank.slicegpt_layers import _fixed_tensor
    def forbidden(*args, **kwargs):
        pytest.fail("Malformed matrix reached torch.tensor")
    monkeypatch.setattr(torch, "tensor", forbidden)
    for value in ([[0.] * 9] * 8, [[0.] * 8] * 9, [["bad"] * 8] * 8):
        with pytest.raises(ValueError, match="description"):
            _fixed_tensor(value, (8, 8), dtype=torch.float64, device="cpu")


def test_attention_inner_activation_reservation_does_not_assume_residual_width(monkeypatch):
    source = SliceGPTModel(8, 8, 1, 128, 128, 128, 8, dtype=torch.float64)
    tokens = torch.zeros(8, 8, dtype=torch.long)
    def forbidden(*args, **kwargs):
        pytest.fail("Wide Q/K/V execution escaped its activation reservation")
    monkeypatch.setattr(profiles, "_constructor", forbidden)
    with pytest.raises(profiles.SliceGPTProfileError, match="ResourceLimitExceeded"):
        profiles.transform_slicegpt(source, tokens, SliceGPT(max_peak_bytes=32 * 1024 * 1024))


def test_checkpoint_semantic_validation_rejects_undeclared_live_kv_retie():
    model = profiles.transform_slicegpt(_model(), _data(), SliceGPT(widths=(4,) * 5)).model
    state = deepcopy(model.state_dict())
    restored = SliceGPTModel.from_config(model.representation_config())
    # Emulate topology.parameter_aliases applied by CheckpointManager before
    # state assignment. Identical shapes/dtypes and finite values must not make
    # this undeclared sharing legal.
    restored.blocks[0].consumer.v_proj.weight = restored.blocks[0].consumer.k_proj.weight
    with pytest.raises(ValueError, match="undeclared parameter sharing"):
        restored.validate_checkpoint_state(state)


@pytest.mark.parametrize("alias_kind", ["parameters", "buffers"])
def test_checkpoint_manager_rejects_injected_alias_metadata(tmp_path, alias_kind):
    from fedcore.tools.registry.checkpoint_manager import CheckpointError, CheckpointManager
    source = _model()
    manager = CheckpointManager(str(tmp_path / "registry"), auto_cleanup=False)
    payload = torch.load(io.BytesIO(manager.serialize_to_bytes(source)), weights_only=True)
    if alias_kind == "parameters":
        paths = ("blocks.0.consumer.k_proj.weight", "blocks.0.consumer.v_proj.weight")
        key = "parameter_aliases"
        # Equal finite values pass the generic tied-state validator. Only the
        # architectural sharing contract can reject this malicious declaration.
        payload["state_dict"][paths[1]] = payload["state_dict"][paths[0]].clone()
    else:
        paths = ("blocks.0.norm.width_denominator", "blocks.1.norm.width_denominator")
        key = "buffer_aliases"
    payload["topology"][key] = tuple(payload["topology"][key]) + (paths,)
    with pytest.raises(CheckpointError, match="Invalid P5 architecture state"):
        manager.restore(payload)
