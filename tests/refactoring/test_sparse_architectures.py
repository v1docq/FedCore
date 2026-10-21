"""FlexiGPT projection, sharing, cache and checkpoint invariants."""
from copy import deepcopy
import json

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fedcore.algorithm.pruning.sparse.architectures import (
    SparseReferenceTransformer, FlexiGPTTransformer, FlexiGPTLinear, FlexiGPTSpec,
    GammaNorm, block_influence, flexigpt_distance, flexigpt,
)


@pytest.fixture(autouse=True)
def cpu_scope():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(148007)
        yield
    torch.set_num_threads(previous)


def source():
    return SparseReferenceTransformer(4, 7, 3, num_heads=2, dtype=torch.float64)


def truncation(weight, rank):
    u, singular, vh = torch.linalg.svd(weight.double(), full_matrices=False)
    return (u[:, :rank]*singular[:rank]) @ vh[:rank]


def test_selection_uses_tail_of_truncated_difference_not_initialization_difference():
    left = torch.diag(torch.tensor([4., 3., 1.], dtype=torch.float64))
    right = torch.diag(torch.tensor([3., 4., 2.], dtype=torch.float64))
    residual = truncation(left, 1) - truncation(right, 1)
    independent_tail = float(torch.linalg.svdvals(residual)[1:].square().sum())
    assert flexigpt_distance(left, right, 1) == independent_tail == 16.
    full_tail = float(torch.linalg.svdvals(left-right)[1:].square().sum())
    assert full_tail != independent_tail


@pytest.mark.parametrize("alpha", [.7, 3.5])
def test_projection_matches_base_plus_full_difference_truncation_with_correct_scale(alpha):
    graph = source()
    x = torch.randn(2, 3, 4, dtype=torch.float64)
    spec = FlexiGPTSpec(retained_positions=(0, 2), rank=2, alpha=alpha)
    compact, evidence = flexigpt(graph, x, spec)
    for name in compact._projections():
        projection = compact.layers[1].get_submodule(name)
        assert type(projection) is FlexiGPTLinear
        base = projection.inner.base
        original = graph.layers[1].get_submodule(name)
        probe = torch.randn(2, original.in_features, dtype=torch.float64)
        expected = F.linear(probe, base.weight + truncation(original.weight-base.weight, 2), original.bias)
        torch.testing.assert_close(projection(probe), expected, rtol=1e-12, atol=1e-12)
    assert evidence["adapter_initialization"] == "rank_r_of_full_difference"
    assert evidence["shared_base_trainable"] is True
    assert evidence["execution_depth"] == 3
    assert isinstance(compact.layers[1].attention_output_norm, GammaNorm)
    assert isinstance(compact.layers[0].attention_output_norm, nn.Identity)


def test_shared_base_gradient_sums_positions_and_optimizer_has_each_parameter_once():
    graph = source()
    x = torch.randn(2, 3, 4, dtype=torch.float64)
    compact, _ = flexigpt(graph, x, FlexiGPTSpec(retained_positions=(0,), rank=2))
    base = compact.layers[0].attention.q_proj
    projections = [compact.layers[i].attention.q_proj for i in range(3)]
    assert projections[1].inner.base is projections[2].inner.base is base
    probes = [torch.randn(3, 4, dtype=torch.float64) for _ in projections]
    sum(projection(probe).sum() for projection, probe in zip(projections, probes)).backward()
    expected = sum(probe.sum(0) for probe in probes).repeat(4, 1)
    torch.testing.assert_close(base.weight.grad, expected, rtol=1e-13, atol=1e-13)
    optimizer = torch.optim.Adam(compact.parameters(), lr=.001)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(parameters) == len({id(p) for p in parameters})
    assert sum(p is base.weight for p in parameters) == 1
    assert all(p.requires_grad for p in base.parameters())


def test_block_influence_actually_selects_retained_positions_and_stable_ties():
    graph = source()
    x = torch.randn(2, 3, 4, dtype=torch.float64)
    with torch.no_grad():
        # Identity residual block has BI=0; the other source blocks change x.
        graph.layers[0].attention.o_proj.weight.zero_()
        graph.layers[0].attention.o_proj.bias.zero_()
        graph.layers[0].mlp.down_proj.weight.zero_()
        graph.layers[0].mlp.down_proj.bias.zero_()
    compact, evidence = flexigpt(graph, x, FlexiGPTSpec(retained_positions=None, retained_count=1))
    assert abs(evidence["block_influence"][0]) < 1e-14
    expected = min(range(3), key=lambda i: (-evidence["block_influence"][i], i))
    assert expected != 0
    assert compact.retained_positions == (expected,)
    assert set(compact.base_sources) == {expected}
    assert evidence["retained_selection"] == "highest_block_influence_stable_position_ties"
    with torch.no_grad():
        for layer in graph.layers:
            layer.attention.o_proj.weight.zero_(); layer.attention.o_proj.bias.zero_()
            layer.mlp.down_proj.weight.zero_(); layer.mlp.down_proj.bias.zero_()
    tied, _ = flexigpt(graph, x, FlexiGPTSpec(retained_positions=None, retained_count=2))
    assert tied.retained_positions == (0, 1)
    with pytest.raises(ValueError):
        FlexiGPTSpec(retained_positions=(0,), retained_count=1)


def test_constructor_roundtrip_restores_aliases_distinct_adapters_and_position_caches():
    graph = source()
    x = torch.randn(2, 3, 4, dtype=torch.float64)
    compact, evidence = flexigpt(graph, x, FlexiGPTSpec(extension_sources=(1,), rank=2))
    config = json.loads(json.dumps(compact.representation_config()))
    restored = FlexiGPTTransformer.from_config(config, dtype=torch.float64)
    restored.validate_state(compact.state_dict())
    restored.load_state_dict(compact.state_dict())
    restored.validate_state()
    assert torch.equal(restored(x), compact(x))
    base = restored.layers[0].mlp.up_proj
    first = restored.layers[1].mlp.up_proj
    second = restored.layers[2].mlp.up_proj
    assert first.inner.base is second.inner.base is base
    assert first.inner.residual.left.weight is not second.inner.residual.left.weight
    assert not torch.equal(first.inner.residual.left.weight, second.inner.residual.left.weight)
    output, caches = restored(x, use_cache=True)
    assert len(caches) == evidence["cache_slots"] == 4
    assert len({cache.keys.data_ptr() for cache in caches}) == 4
    assert torch.equal(output, restored(x))
    # Cached one-token decoding agrees with the same full causal graph.
    token = torch.randn(2, 1, 4, dtype=torch.float64)
    step, next_caches = restored(token, caches=caches, use_cache=True)
    all_at_once = restored(torch.cat((x, token), dim=1))[:, -1:]
    torch.testing.assert_close(step, all_at_once, rtol=1e-11, atol=1e-11)
    assert all(cache.length == 4 for cache in next_caches)


def test_checkpoint_rejects_disagreeing_alias_values_or_live_alias_clones():
    graph = source()
    compact, _ = flexigpt(graph, torch.randn(1, 3, 4, dtype=torch.float64))
    # Clone entries independently: deepcopy intentionally preserves aliases.
    state = {name: value.clone() for name, value in compact.state_dict().items()}
    state["layers.1.attention.q_proj.inner.base.weight"] += 1
    with pytest.raises(ValueError, match="aliases"):
        compact.validate_state(state)
    compact.layers[1].attention.q_proj.inner.base = deepcopy(compact.layers[0].attention.q_proj)
    with pytest.raises(ValueError, match="alias topology"):
        compact.validate_state()


def test_frozen_inference_checkpoint_keeps_aliases_and_values():
    compact, _ = flexigpt(source(), torch.randn(1, 3, 4, dtype=torch.float64))
    compact.requires_grad_(False)
    compact.validate_state()
    restored = FlexiGPTTransformer.from_config(compact.representation_config(), dtype=torch.float64)
    restored.validate_state(compact.state_dict())
    restored.load_state_dict(compact.state_dict())
    restored.requires_grad_(False)
    restored.validate_state()
    x = torch.randn(1, 3, 4, dtype=torch.float64)
    assert torch.equal(restored(x), compact(x))
    assert restored.layers[1].attention.q_proj.inner.base is restored.layers[0].attention.q_proj


def test_block_influence_zero_norms_and_gamma_population_normalization():
    zeros = torch.zeros(1, 2, 4, dtype=torch.float64)
    assert block_influence(zeros, zeros) == 0.
    assert block_influence(zeros, torch.ones_like(zeros)) == 1.
    assert block_influence(torch.ones_like(zeros), -torch.ones_like(zeros)) == 2.
    normalization = GammaNorm(4, .01, 1e-5, dtype=torch.float64)
    x = torch.tensor([[[1., 3., 4., 8.]]], dtype=torch.float64)
    expected = .01*(x-x.mean(-1, keepdim=True))/torch.sqrt(x.var(-1, keepdim=True, unbiased=False)+1e-5)
    torch.testing.assert_close(normalization(x), expected, rtol=1e-13, atol=1e-13)
    assert torch.equal(normalization(zeros), zeros)


def test_preflight_rejects_shape_budget_and_wrong_reference_without_mutation():
    graph = source()
    x = torch.randn(2, 3, 4, dtype=torch.float64)
    before, rng = deepcopy(graph.state_dict()), torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="workspace"):
        flexigpt(graph, x, FlexiGPTSpec(max_workspace_bytes=1))
    with pytest.raises(ValueError, match="rank"):
        flexigpt(graph, x, FlexiGPTSpec(rank=5))
    with pytest.raises(ValueError, match="retained"):
        flexigpt(graph, x, FlexiGPTSpec(retained_positions=(3,)))
    with pytest.raises(ValueError, match="persistent-element"):
        SparseReferenceTransformer(1000, 1000, 3)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(before[k], v) for k, v in graph.state_dict().items())
