"""Independent coordinate and weighted-operator oracles for sparse FFNs."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fedcore.algorithm.low_rank.coupled_layers import ReferenceMLP
from fedcore.algorithm.low_rank.statistics import StatisticsPassport
from fedcore.algorithm.pruning.sparse.channels import (
    SparseMLP, PartitionedMLP, SoLASpec, inspect_mlp, slice_mlp, partition_mlp, sola,
)


@pytest.fixture(autouse=True)
def cpu_scope():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(141143)
        yield
    torch.set_num_threads(previous)


def dense_selected(module, x, channels, *, output_bias=True):
    index = torch.tensor(channels, dtype=torch.long)
    up = F.linear(x, module.up_proj.weight[index], None if module.up_proj.bias is None else module.up_proj.bias[index])
    activation = {"silu": F.silu, "relu": F.relu, "gelu": F.gelu}[module.activation]
    if module.gated:
        gate = F.linear(x, module.gate_proj.weight[index], None if module.gate_proj.bias is None else module.gate_proj.bias[index])
        features = activation(gate)*up
    else:
        features = activation(up)
    return F.linear(features, module.down_proj.weight[:, index], module.down_proj.bias if output_bias else None)


@pytest.mark.parametrize("gated,activation", [(True, "silu"), (False, "gelu"), (False, "relu")])
@pytest.mark.parametrize("channels", [(), (5, 1, 3), (5, 4, 3, 2, 1, 0)])
def test_slice_keeps_original_coefficients_and_permuted_coordinates(gated, activation, channels):
    model = ReferenceMLP(4, 6, gated=gated, activation=activation, bias=True, dtype=torch.float64,
                          channel_indices=(9, 8, 7, 6, 5, 4))
    original = deepcopy(model.state_dict())
    x = torch.randn(2, 3, 4, dtype=torch.float64)
    result, evidence = slice_mlp(model, channels)
    torch.testing.assert_close(result(x), dense_selected(model, x, channels), rtol=1e-13, atol=1e-13)
    assert result.channel_indices == tuple(model.channel_indices[i] for i in channels)
    assert result.current_indices == channels
    assert evidence["down_refit"] is False
    for key, value in original.items():
        assert torch.equal(value, model.state_dict()[key])
    assert result.up_proj.weight.data_ptr() != model.up_proj.weight.data_ptr() or not channels
    if len(channels) == 6:
        torch.testing.assert_close(result(x), model(x), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("prime", [(), (4, 1), tuple(range(6))])
@pytest.mark.parametrize("gated", [True, False])
def test_partition_is_complete_and_adds_external_bias_once(prime, gated):
    model = ReferenceMLP(4, 6, gated=gated, dtype=torch.float64)
    x = torch.randn(3, 4, dtype=torch.float64)
    result, _ = partition_mlp(model, prime)
    torch.testing.assert_close(result(x), model(x), rtol=1e-13, atol=1e-13)
    assert result.prime.down_proj.bias is result.marginal.down_proj.bias is None
    assert result.prime.current_indices == prime
    assert result.marginal.current_indices == tuple(i for i in range(6) if i not in prime)
    assert result.output_bias.shape == (4,)
    result.validate_state()
    restored = PartitionedMLP.from_config(json.loads(json.dumps(result.representation_config())), dtype=torch.float64)
    restored.validate_state(result.state_dict())
    restored.load_state_dict(result.state_dict())
    assert torch.equal(restored(x), result(x))


@pytest.mark.parametrize("invalid", [(1, 1), (-1,), (6,), (True,), (1.0,)])
def test_bad_selection_rejects_before_source_or_rng_changes(invalid):
    model = ReferenceMLP(4, 6, dtype=torch.float64)
    before, rng = deepcopy(model.state_dict()), torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        slice_mlp(model, invalid)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


def test_real_llama_adapter_uses_actual_class_and_exact_gated_function():
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaMLP
    config = LlamaConfig(hidden_size=4, intermediate_size=7, num_attention_heads=1, mlp_bias=True)
    model = LlamaMLP(config).double()
    x = torch.randn(2, 3, 4, dtype=torch.float64)
    all_channels, evidence = slice_mlp(model, list(range(7)))
    torch.testing.assert_close(all_channels(x), model(x), rtol=1e-13, atol=1e-13)
    assert evidence["adapter"] == "transformers-4.44-LlamaMLP"
    class Lookalike(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj, self.gate_proj, self.down_proj = model.up_proj, model.gate_proj, model.down_proj
    with pytest.raises(ValueError, match="Supported FFN adapters"):
        inspect_mlp(Lookalike())


def independent_weighted(weight, rows, rank, ridge):
    moment = rows.double().T @ rows.double()/len(rows)
    moment = moment + ridge*torch.eye(moment.shape[0], dtype=torch.float64)
    values, vectors = torch.linalg.eigh(moment)
    root = (vectors*values.sqrt()) @ vectors.T
    inverse = (vectors*values.rsqrt()) @ vectors.T
    u, s, vh = torch.linalg.svd(weight.double() @ root, full_matrices=False)
    return ((u[:, :rank]*s[:rank]) @ vh[:rank] @ inverse).to(weight)


@pytest.mark.parametrize("gated", [False, True])
def test_sola_keeps_marginal_nonlinearity_and_matches_independent_weighted_oracle(gated):
    model = ReferenceMLP(4, 7, gated=gated, dtype=torch.float64)
    x = torch.randn(4, 3, 4, dtype=torch.float64)
    spec = SoLASpec(prime_count=2, rank=2, activity_point="gated_product" if gated else "post_activation", ridge=1e-4)
    result, evidence = sola(model, x, spec)
    prime, marginal = result.prime.current_indices, result.marginal.current_indices
    rows = x.reshape(-1, 4)
    features = model.features(rows)
    wm = independent_weighted(model.up_proj.weight[list(marginal)], rows, 2, spec.ridge)
    up = F.linear(x, wm, model.up_proj.bias[list(marginal)])
    if gated:
        wg = independent_weighted(model.gate_proj.weight[list(marginal)], rows, 2, spec.ridge)
        gate = F.linear(x, wg, model.gate_proj.bias[list(marginal)])
        hidden = F.silu(gate)*up
    else:
        hidden = F.silu(up)
    wd = independent_weighted(model.down_proj.weight[:, list(marginal)], features[:, list(marginal)], 2, spec.ridge)
    expected = dense_selected(model, x, prime, output_bias=False) + F.linear(hidden, wd, model.down_proj.bias)
    torch.testing.assert_close(result(x), expected, rtol=2e-10, atol=2e-10)
    assert result.marginal.up_proj.out_features == len(marginal)
    assert evidence["marginal_deleted"] is False
    assert evidence["parameter_elements"] == sum(v.numel() for v in result.state_dict().values())
    clone = PartitionedMLP.from_config(result.representation_config(), dtype=torch.float64)
    clone.load_state_dict(result.state_dict())
    clone.validate_state()
    assert torch.equal(clone(x), result(x))


def test_sola_selects_gram_before_root_and_counts_dense_prime_and_exclusions():
    # Cholesky slicing fails for a non-leading principal subset.
    gram = torch.tensor([[4., 2., 1.], [2., 5., 2.], [1., 2., 6.]], dtype=torch.float64)
    indices = torch.tensor([1, 2])
    sliced_root = torch.linalg.cholesky(gram)[indices][:, indices]
    selected_root = torch.linalg.cholesky(gram[indices][:, indices])
    assert not torch.allclose(sliced_root @ sliced_root.T, gram[indices][:, indices])
    torch.testing.assert_close(selected_root @ selected_root.T, gram[indices][:, indices])
    model = ReferenceMLP(4, 7, dtype=torch.float64)
    x = torch.randn(20, 4, dtype=torch.float64)
    spec = SoLASpec(prime_count=2, rank=1, excluded_projections=("gate_proj",))
    result, evidence = sola(model, x, spec)
    assert type(result.marginal.gate_proj) is nn.Linear
    assert evidence["prime_elements"] > 0
    budget = evidence["parameter_elements"]
    same, _ = sola(model, x, replace(spec, parameter_budget=budget))
    assert sum(v.numel() for v in same.state_dict().values()) == budget
    with pytest.raises(ValueError, match="budget infeasible"):
        sola(model, x, replace(spec, parameter_budget=budget-1))


def test_sola_energy_zero_statistics_and_preflight_provenance():
    model = ReferenceMLP(4, 7, dtype=torch.float64)
    x = torch.zeros(2, 3, 4, dtype=torch.float64)
    spec = SoLASpec(prime_count=0, energy_target=.5, ridge=0, allow_dense=False)
    result, evidence = sola(model, x, spec)
    assert all(v >= .5 for v in evidence["retained_energy"].values())
    assert result.marginal.intermediate_size == 7
    passport = StatisticsPassport("a", "g", "p", "mlp_input", "d")
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="StaleStatistics"):
        sola(model, x, spec, passport=passport, expected_passport=replace(passport, graph_version="new"))
    with pytest.raises(ValueError, match="workspace"):
        sola(model, x, replace(spec, max_workspace_bytes=1))
    assert torch.equal(rng, torch.get_rng_state())


def test_compact_config_rejects_unknown_fields_and_coordinate_tampering():
    source = ReferenceMLP(4, 6, dtype=torch.float64)
    compact, _ = slice_mlp(source, (4, 1))
    config = compact.representation_config()
    with pytest.raises(ValueError):
        SparseMLP.from_config(dict(config, surprising=True))
    state = deepcopy(compact.state_dict())
    state["_current_map"][0] = 0
    with pytest.raises(ValueError, match="coordinate"):
        compact.validate_state(state)


def test_adapter_rejects_undeclared_source_state_and_passport_fingerprints_all_weights():
    model = ReferenceMLP(4, 7, dtype=torch.float64)
    x = torch.randn(8, 4, dtype=torch.float64)
    _, first = sola(model, x, SoLASpec(rank=1))
    assert first["passport"]["count"] == 8
    assert first["passport"]["weight_sum"] == 8.
    with torch.no_grad():
        model.gate_proj.bias.add_(.01)
    _, second = sola(model, x, SoLASpec(rank=1))
    assert first["passport"]["checkpoint_id"] != second["passport"]["checkpoint_id"]
    model.register_buffer("undeclared", torch.ones(1))
    with pytest.raises(ValueError, match="undeclared buffers"):
        slice_mlp(model, (0, 1))
