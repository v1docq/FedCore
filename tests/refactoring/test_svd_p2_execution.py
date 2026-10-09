"""Public P2 regression tests with independent numerical and storage evidence."""
from copy import deepcopy
from dataclasses import replace
import io
import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.approximation import solve_weighted
from fedcore.algorithm.low_rank.execution import WeightedProfileError
from fedcore.algorithm.low_rank.method_execution import MethodProfileError, plan_method, transform_method
from fedcore.algorithm.low_rank.method_specs import (
    AFM, ASVD, BasisSharing, Bolaco, DRONE, EoRA, FLARSVD, FWSVD,
    GroupReduce, MethodSpecError, MixedRank, SVDLLMV1, SVDLLMV2, SVDLLMV5,
    method_payload, parse_method,
)
from fedcore.algorithm.low_rank.statistics import (
    StatisticsPassport, empty_centered_moments, finish_centered_moments,
    finish_within_group_covariance, update_centered_moments,
)
from fedcore.algorithm.low_rank.statistical_profiles import (
    GradientCollectionMetadata, channel_abs_statistics, fwsvd_statistics,
    ledoit_wolf_metric, solve_affine_pca, solve_asvd, solve_flar, solve_fwsvd,
)
from fedcore.algorithm.low_rank.structured_profiles import (
    balanced_factors, basis_sharing_factors, drone_factors, eora_factors,
    groupreduce_factors, mixed_rank_metric, svdllm_v1_fit, svdllm_v2_factors,
)
from fedcore.algorithm.low_rank.structured_layers import (
    GroupedEmbedding, GroupedLMHead, ResidualLinear, SharedBasisLinear,
)
from fedcore.algorithm.low_rank.topology import TopologyError


@pytest.fixture(autouse=True)
def four_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


def _passport():
    return StatisticsPassport("reference-checkpoint", "reference-graph", "predecessor", "input", "calibration")


def _centered(rows):
    return finish_centered_moments(update_centered_moments(empty_centered_moments(rows.shape[-1], _passport()), rows))


def _layer(in_features=4, out_features=3, *, bias=True):
    # Local generator makes test data independent of the caller's RNG state.
    layer = nn.Linear(in_features, out_features, bias=bias, dtype=torch.float64)
    generator = torch.Generator().manual_seed(271 + in_features + out_features)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator, dtype=torch.float64))
        if layer.bias is not None:
            layer.bias.copy_(torch.linspace(-.3, .4, out_features, dtype=torch.float64))
    return layer


def _rows(width=4, length=13):
    generator = torch.Generator().manual_seed(19 + width)
    return torch.randn((length, width), generator=generator, dtype=torch.float64) + torch.arange(width, dtype=torch.float64)


def _moment(rows):
    return rows.T @ rows / len(rows)


def _pair_matrix(answer):
    return answer.approximation if hasattr(answer, "approximation") else answer.left @ answer.right


def _reference(spec, layer, rows, rank, *, labels=None, group_labels=None, original_rows=None):
    weight, bias = layer.weight.detach(), layer.bias
    if isinstance(spec, ASVD):
        answer = solve_asvd(weight, channel_abs_statistics(rows, _passport(), mode=spec.abs_mode),
                            rank, alpha=spec.alpha, epsilon=spec.epsilon, zero_policy=spec.zero_policy)
    elif isinstance(spec, FWSVD):
        # Analytic individual MSE gradients, independent of the model collector.
        residual = rows @ weight.T + bias.detach() - labels
        multiplier = 2 / layer.out_features if spec.reduction == "mean" else 2
        gradients = multiplier * residual[:, :, None] * rows[:, None, :]
        stamp = replace(_passport(), count=len(rows), weight_sum=float(len(rows)))
        stats = fwsvd_statistics(gradients.square().mean(0), stamp,
                                 GradientCollectionMetadata("mse", "labels", reduction=spec.reduction))
        answer = solve_fwsvd(weight, stats, rank, epsilon=spec.epsilon, zero_policy=spec.zero_policy)
    elif isinstance(spec, (AFM, Bolaco)):
        outputs = rows @ weight.T + (bias.detach() if bias is not None else 0)
        moments = _centered(outputs)
        if isinstance(spec, Bolaco):
            groups = tuple((str(int(key)), update_centered_moments(
                empty_centered_moments(layer.out_features, _passport()), outputs[group_labels == key]))
                for key in torch.unique(group_labels, sorted=True))
            moments = finish_within_group_covariance(groups)
            if spec.group_weighting == "equal_groups":
                weights = tuple(1 / len(groups) for _ in groups)
                covariance = sum((item.covariance * mass for item, mass in zip(moments.group_moments, weights)),
                                 torch.zeros_like(moments.covariance))
                moments = replace(moments, covariance=covariance, group_weights=weights)
        answer = solve_affine_pca(weight, bias, moments, rank)
    elif isinstance(spec, FLARSVD):
        answer = solve_flar(weight, ledoit_wolf_metric(rows, _passport()), rank)
    elif isinstance(spec, DRONE):
        answer = drone_factors(weight, rows.T, rank, rcond=spec.rcond)
    elif isinstance(spec, SVDLLMV1):
        initial = balanced_factors(solve_weighted(weight, _moment(rows if original_rows is None else original_rows), rank))
        answer = svdllm_v1_fit(weight, initial.right, rows, rcond=spec.rcond, ridge=spec.ridge)
    else:
        answer = svdllm_v2_factors(weight, _moment(rows), rank)
    output_bias = getattr(answer, "bias", None)
    return _pair_matrix(answer), bias.detach() if output_bias is None and bias is not None else output_bias


@pytest.mark.parametrize("spec", [ASVD(abs_mode="abs_max", alpha=.7, epsilon=.01), FWSVD(loss="mse"),
    FWSVD(loss="mse", reduction="sum"), AFM(), Bolaco(), Bolaco(group_weighting="equal_groups"),
    FLARSVD(), DRONE(), SVDLLMV1(ridge=.01), SVDLLMV2()])
def test_public_single_layer_matches_pure_reference_and_actual_storage(spec):
    layer, rows = _layer(), _rows()
    labels = _rows(3) * .2
    groups = torch.tensor([0] * 4 + [1] * 9)
    kwargs = {"labels": labels} if isinstance(spec, FWSVD) else ({"group_labels": groups} if isinstance(spec, Bolaco) else {})
    result = transform_method(layer, rows, spec, rank=2, target_paths=("",), **kwargs)
    weight, bias = _reference(spec, layer, rows, 2, labels=labels, group_labels=groups)
    expected = rows @ weight.T + (bias if bias is not None else 0)
    torch.testing.assert_close(result.model(rows), expected, rtol=1e-10, atol=1e-10)
    actual_parameters = sum(parameter.numel() for parameter in result.model.parameters())
    actual_bytes = sum(parameter.numel() * parameter.element_size() for parameter in result.model.parameters())
    assert result.evidence["parameters_after"] == actual_parameters == 2 * (4 + 3) + 3
    assert result.evidence["parameter_bytes_after"] == actual_bytes
    assert result.evidence["tensor_bytes_after"] == actual_bytes
    assert result.model is not layer and result.plan.request.statistics_refs
    assert result.model._fedcore_requires_optimizer_rebuild
    json.dumps(result.evidence, allow_nan=False)


SPECS = [ASVD(), FWSVD(loss="mse", max_examples=7), AFM(), Bolaco(), FLARSVD(), DRONE(rcond=1e-8),
         SVDLLMV1(ridge=.1), SVDLLMV2(), EoRA(), MixedRank(seed=17),
         BasisSharing(metric_mode="proportional_moments", proportional_scales=(1., 2.)),
         GroupReduce(groups=((0, 2), (1, 3)), ranks=(1, 1)), SVDLLMV5()]


@pytest.mark.parametrize("spec", SPECS)
def test_method_specs_roundtrip_through_strict_json(spec):
    payload = json.loads(json.dumps(method_payload(spec), allow_nan=False))
    assert parse_method(**payload) == spec


@pytest.mark.parametrize("method,options", [
    ("asvd", {"alpha": True}), ("asvd", {"alpha": float("nan")}), ("asvd", {"epsilon": 10**1000}),
    ("fwsvd", {"max_examples": True}), ("fwsvd", {"reduction": "none"}),
    ("drone", {"rcond": float("inf")}), ("svdllm_v1", {"ridge": -1.}),
    ("mixed_rank", {"residual_rank": True}), ("basis_sharing", {"proportional_scales": [float("nan")]}),
    ("groupreduce", {"groups": [[0, True], [2]], "ranks": [1, 1]}),
    ("groupreduce", {"groups": [[0, 1], [1, 2]], "ranks": [1, 1]}),
    ("svdllm_v5", {"merge": 1}), ("flar_svd", {"estimator_version": "literal_paper"}),
])
def test_method_specs_refuse_malformed_types_and_nonfinite_numbers(method, options):
    with pytest.raises(MethodSpecError):
        parse_method(method, options)


@pytest.mark.parametrize("version", [True, 1., 0, 2])
def test_method_specs_require_exact_supported_version(version):
    with pytest.raises(MethodSpecError):
        parse_method("asvd", version=version)


def test_original_mixed_modes_parameters_buffers_grad_objects_and_rng_are_unchanged():
    model = nn.Sequential(_layer(4, 4), nn.Dropout(.5), _layer(4, 3))
    model.register_buffer("original_buffer", torch.tensor([3., 4.], dtype=torch.float64))
    model.train()
    model[0].eval()
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, .125)
    modes = tuple(module.training for module in model.modules())
    parameters = tuple(model.parameters())
    gradients = tuple(parameter.grad for parameter in parameters)
    state = deepcopy(model.state_dict())
    rng = torch.random.get_rng_state().clone()
    result = transform_method(model, _rows(), ASVD(), rank=2, target_paths=("0", "2"))
    assert tuple(model.parameters()) == parameters
    assert all(parameter.grad is gradient for parameter, gradient in zip(parameters, gradients))
    assert all(bool((gradient == .125).all()) for gradient in gradients)
    assert tuple(module.training for module in model.modules()) == modes
    assert tuple(module.training for module in result.model.modules())[:2] == modes[:2]
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, state[name])
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert all(not module._forward_pre_hooks and not module._forward_hooks for module in model.modules())
    assert all(not module._forward_pre_hooks and not module._forward_hooks for module in result.model.modules())


class _RandomCalibration(nn.Module):
    def forward(self, inputs):
        return inputs + .01 * (random.random() + np.random.random() + torch.rand(()))


def test_public_collection_restores_python_numpy_and_torch_rng():
    model = nn.Sequential(_RandomCalibration(), _layer())
    inputs = _rows()
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    transform_method(model, inputs, ASVD(), rank=2, target_paths=("1",))
    assert random.getstate() == python_rng
    actual = np.random.get_state()
    assert actual[0] == numpy_rng[0] and actual[2:] == numpy_rng[2:]
    np.testing.assert_array_equal(actual[1], numpy_rng[1])
    assert torch.equal(torch.random.get_rng_state(), torch_rng)


@pytest.mark.parametrize("spec", [ASVD(), DRONE(), SVDLLMV1()])
def test_sequential_public_solve_recollects_after_predecessor_replacement(spec):
    model = nn.Sequential(_layer(4, 4), nn.Tanh(), _layer(4, 3)).eval()
    inputs = _rows()
    weight0, bias0 = _reference(spec, model[0], inputs, 1)
    current_rows = torch.tanh(inputs @ weight0.T + bias0)
    original_rows = torch.tanh(model[0](inputs)).detach()
    weight2, bias2 = _reference(spec, model[2], current_rows, 2, original_rows=original_rows)
    result = transform_method(model, inputs, spec, ranks={"0": 1, "2": 2}, target_paths=("0", "2"))
    torch.testing.assert_close(result.model(inputs), current_rows @ weight2.T + bias2, atol=1e-10, rtol=1e-10)
    first, second = result.evidence["layers"]
    assert first["graph_version"].endswith("accepted=0")
    assert second["graph_version"].endswith("accepted=1")
    assert first["predecessor_version"] != second["predecessor_version"]
    assert first["snapshot_id"] != second["snapshot_id"]
    assert second["statistics"]["graph_version"] == second["graph_version"]
    if isinstance(spec, SVDLLMV1):
        assert second["numerics"]["initialization_input"] == "original_reference_graph"
        assert second["numerics"]["fit_input"] == "current_graph_after_predecessor_replacements"


@pytest.mark.parametrize("kind", ["unsupported_operator", "resource", "bad_rank", "test_data", "storage_alias", "missing_fisher_labels"])
def test_invalid_public_requests_refuse_before_private_model_copy(monkeypatch, kind):
    import fedcore.algorithm.low_rank.method_execution as execution
    model, inputs, spec = _layer(), _rows(), ASVD()
    kwargs = {"rank": 1, "target_paths": ("",)}
    if kind == "unsupported_operator":
        model = nn.Conv1d(2, 3, 1).double()
        inputs = torch.ones(3, 2, 5, dtype=torch.float64)
    elif kind == "resource":
        kwargs["max_workspace_bytes"] = 8
    elif kind == "bad_rank":
        kwargs["rank"] = True
    elif kind == "test_data":
        kwargs["data_role"] = "test"
    elif kind == "storage_alias":
        model = nn.Sequential(_layer(4, 4), _layer(4, 4))
        model[1].weight = nn.Parameter(model[0].weight[:])
        kwargs["target_paths"] = ("0", "1")
    else:
        spec = FWSVD(loss="mse")

    def forbidden_copy(*args, **kwargs):
        raise AssertionError("Private model copy happened before capability/resource refusal")

    monkeypatch.setattr(execution, "deepcopy", forbidden_copy)
    with pytest.raises((WeightedProfileError, TopologyError)):
        transform_method(model, inputs, spec, **kwargs)


def test_affine_parameter_fraction_includes_new_bias_cost():
    layer, inputs = _layer(4, 4, bias=False), _rows()
    with pytest.raises(MethodProfileError, match="[Bb]udget|[Pp]arameter|[Rr]ank"):
        transform_method(layer, inputs, AFM(), parameter_fraction=.6, target_paths=("",))


def test_biased_affine_fraction_uses_requested_total_parameter_budget():
    layer, inputs = _layer(10, 10), _rows(10)
    result = transform_method(layer, inputs, AFM(), parameter_fraction=.42, target_paths=("",))
    assert result.evidence["parameters_after"] == sum(p.numel() for p in result.model.parameters()) == 30
    assert result.evidence["parameters_after"] <= int(.42 * 110) == 46
    assert result.evidence["layers"][0]["requested_rank"] == 1


def test_explicit_budget_is_checked_against_installed_factors():
    layer, inputs = _layer(), _rows()
    with pytest.raises(MethodProfileError, match="BudgetInfeasible"):
        transform_method(layer, inputs, ASVD(), rank=2, target_paths=("",), parameter_budget=16)
    result = transform_method(layer, inputs, ASVD(), rank=2, target_paths=("",), parameter_budget=17)
    assert result.evidence["parameters_after"] == 17


def test_residual_eora_and_mixed_rank_use_real_base_and_single_bias():
    reference = nn.Sequential(_layer())
    base = deepcopy(reference)
    with torch.no_grad():
        base[0].weight.mul_(.6)
    inputs = _rows()
    result = transform_method(reference, inputs, EoRA(), rank=1, target_paths=("0",), base_model=base)
    pair = eora_factors(reference[0].weight, base[0].weight, _moment(inputs), 1)
    expected = base(inputs) + inputs @ pair.matrix().T
    torch.testing.assert_close(result.model(inputs), expected, rtol=1e-10, atol=1e-10)
    assert isinstance(result.model[0], ResidualLinear)
    assert result.model[0].base is not base[0]
    assert not result.model[0].base.weight.requires_grad
    assert result.evidence["parameters_after"] == 4 * 3 + 3 + 4 + 3
    mixed = transform_method(reference, inputs, MixedRank(residual_rank=1), rank=2, target_paths=("0",))
    metric, _ = mixed_rank_metric(reference[0].weight, inputs, objective="normalized_output")
    expected_weight = solve_weighted(reference[0].weight, metric, 2).approximation
    torch.testing.assert_close(mixed.model(inputs), inputs @ expected_weight.T + reference[0].bias,
                               atol=1e-10, rtol=1e-10)
    assert mixed.evidence["parameters_after"] == 2 * (4 + 3) + 3 + 4 + 3
    assert not mixed.model[0].base.left.weight.requires_grad
    mixed.model(inputs).square().mean().backward()
    assert bool(torch.count_nonzero(mixed.model[0].residual.left.weight.grad))


@pytest.mark.parametrize("sequence_axis", [False, True])
def test_eora_author_fixed_n_preserves_original_batch_axes_and_reference(sequence_axis):
    reference = nn.Sequential(_layer())
    base = deepcopy(reference)
    with torch.no_grad():
        base[0].weight.mul_(.6)
    rows = _rows(length=12)
    inputs = rows.reshape(4, 3, 4) if sequence_axis else rows[:6]
    batch_size = 2 if sequence_axis else 3
    original = {name: value.clone() for name, value in reference.state_dict().items()}
    original_base = {name: value.clone() for name, value in base.state_dict().items()}
    result = transform_method(reference, inputs, EoRA(gram_update_version="author_fixed_n_v1"),
                              rank=1, target_paths=("0",), base_model=base, batch_size=batch_size)
    gram = torch.zeros((4, 4), dtype=torch.float64)
    for batch in inputs.split(batch_size):
        batch_samples = len(batch) if sequence_axis else 1
        flattened = batch.reshape(-1, 4)
        gram = gram * (len(inputs) / (len(inputs) + batch_samples)) + flattened.T @ flattened / len(inputs)
    pair = eora_factors(reference[0].weight, base[0].weight, gram, 1)
    torch.testing.assert_close(result.model(inputs), base(inputs) + inputs @ pair.matrix().T,
                               atol=1e-10, rtol=1e-10)
    diagnostics = result.evidence["layers"][0]["numerics"]
    assert diagnostics["gram_interpretation"] == "author_fixed_n_forgetting_not_empirical_second_moment"
    assert len(diagnostics["updates"]) == 2
    for update in diagnostics["updates"]:
        assert update["calibration_samples"] == len(inputs)
        assert update["batch_samples"] == (2 if sequence_axis else 1)
        assert update["input_rows"] == (6 if sequence_axis else 3)
        assert update["order_dependent"] is True and update["mergeable"] is False
    for name, value in reference.state_dict().items():
        assert torch.equal(value, original[name])
    for name, value in base.state_dict().items():
        assert torch.equal(value, original_base[name])


def _basis_case(rank=2):
    model = nn.Sequential(_layer(4, 4), nn.Identity(), _layer(4, 3)).eval()
    inputs = _rows()
    result = transform_method(model, inputs, BasisSharing(), rank=rank, target_paths=("0", "2"))
    return model, inputs, result


def _group_case(ranks=(1, 1)):
    embedding = nn.Embedding(6, 3, dtype=torch.float64)
    with torch.no_grad():
        embedding.weight.copy_(_rows(3, length=6))
    head = nn.Linear(3, 6, bias=True, dtype=torch.float64)
    head.weight = embedding.weight
    model = nn.Sequential(embedding, head).eval()
    tokens = torch.tensor([[0, 1, 2], [3, 4, 5], [2, 2, 5]])
    spec = GroupReduce(groups=((0, 2, 4), (1, 3, 5)), ranks=ranks)
    return model, tokens, transform_method(model, tokens, spec, target_paths=("0", "1"))


def test_basis_shared_parameter_identity_and_unique_cost():
    model, inputs, result = _basis_case()
    metric = (_moment(inputs) + _moment(model[0](inputs).detach())) / 2
    reference = basis_sharing_factors((model[0].weight, model[2].weight), metric, 2)
    current = inputs @ (reference.lefts[0] @ reference.right).T + model[0].bias
    expected = current @ (reference.lefts[1] @ reference.right).T + model[2].bias
    torch.testing.assert_close(result.model(inputs), expected, atol=1e-10, rtol=1e-10)
    assert isinstance(result.model[0], SharedBasisLinear)
    assert result.model[0].right is result.model[2].right
    assert result.evidence["parameters_after"] == 2 * 4 + 4 * 2 + 3 * 2 + 4 + 3 == 29
    assert result.evidence["parameters_after"] == sum(parameter.numel() for parameter in result.model.parameters())
    assert all(record["numerics"]["exact_individual_objective"] is False for record in result.evidence["layers"])


def test_groupreduce_keeps_tied_head_vocabulary_order_and_map_buffer_cost():
    model, tokens, result = _group_case()
    frequencies = torch.bincount(tokens.flatten(), minlength=6).double()
    factors = groupreduce_factors(model[0].weight, frequencies, torch.tensor([0, 1, 0, 1, 0, 1]), (1, 1))
    table = torch.empty_like(model[0].weight)
    for ids, pair in zip(factors.group_tokens, factors.factors):
        table[torch.as_tensor(ids, dtype=torch.long)] = pair.matrix()
    expected = nn.functional.embedding(tokens, table) @ table.T + model[1].bias
    torch.testing.assert_close(result.model(tokens), expected, atol=1e-10, rtol=1e-10)
    assert isinstance(result.model[0], GroupedEmbedding) and isinstance(result.model[1], GroupedLMHead)
    assert result.model[1].table is result.model[0]
    parameter_elements = sum(parameter.numel() for parameter in result.model.parameters())
    assert result.evidence["parameters_after"] == parameter_elements == 18
    assert result.evidence["tensor_bytes_after"] == 18 * 8 + 2 * 6 * 8
    assert result.evidence["layers"][0]["vocabulary_map_bytes"] == 2 * 6 * 8


def test_checkpoint_refuses_conflicting_shared_vocabulary_buffer_state(tmp_path):
    from fedcore.tools.registry.checkpoint_manager import CheckpointManager, CheckpointError
    _, _, result = _group_case()
    manager = CheckpointManager(str(tmp_path), auto_cleanup=False)
    serialized = manager.serialize_to_bytes(result.model)
    payload = torch.load(io.BytesIO(serialized), weights_only=True)
    # This alternate map is valid for a single table, but conflicts with the
    # embedding's map for the same shared table. Last-write-wins is invalid.
    altered = payload["state_dict"]["1.table.token_to_group"].clone()
    altered[0], altered[1] = altered[1].clone(), altered[0].clone()
    payload["state_dict"]["1.table.token_to_group"] = altered
    original_maps = result.model[0].token_to_group.clone()
    with pytest.raises(CheckpointError, match="[Bb]uffer|[Tt]ied|[Ss]hared"):
        manager.restore(payload, model=result.model)
    assert torch.equal(result.model[0].token_to_group, original_maps)
    assert result.model[1].table is result.model[0]


@pytest.mark.parametrize("kind", ["asvd", "fwsvd", "afm", "bolaco", "flar", "drone", "v1", "v2", "residual", "basis", "group"])
def test_fullrank_public_checkpoint_reload_and_torchscript_export(tmp_path, kind):
    from fedcore.tools.registry.checkpoint_manager import CheckpointManager
    from fedcore.tools.export import export_model
    if kind == "basis":
        original, inputs, result = _basis_case(rank=4)
    elif kind == "group":
        original, inputs, result = _group_case(ranks=(3, 3))
    else:
        original, inputs = nn.Sequential(_layer()).eval(), _rows()
        specs = {"asvd": ASVD(), "fwsvd": FWSVD(loss="mse"), "afm": AFM(), "bolaco": Bolaco(),
                 "flar": FLARSVD(), "drone": DRONE(), "v1": SVDLLMV1(), "v2": SVDLLMV2(), "residual": EoRA()}
        kwargs = {}
        if kind == "fwsvd":
            kwargs["labels"] = torch.zeros((len(inputs), 3), dtype=torch.float64)
        elif kind == "bolaco":
            kwargs["group_labels"] = torch.arange(len(inputs)) % 2
        elif kind == "residual":
            kwargs["base_model"] = deepcopy(original)
            with torch.no_grad():
                kwargs["base_model"][0].weight.mul_(.8)
        result = transform_method(original, inputs, specs[kind], rank=3, target_paths=("0",), **kwargs)
    result.model.eval()
    torch.testing.assert_close(result.model(inputs), original(inputs), rtol=1e-10, atol=1e-10)
    manager = CheckpointManager(str(tmp_path), auto_cleanup=False)
    path = tmp_path / "checkpoint.pt"
    manager.save_to_file(manager.serialize_to_bytes(result.model), str(path))
    restored = manager.load_from_file(str(path))
    torch.testing.assert_close(restored(inputs), result.model(inputs), rtol=1e-10, atol=1e-10)
    assert parse_method(**restored._fedcore_method_evidence["method_spec"]) == parse_method(**result.evidence["method_spec"])
    if kind == "basis":
        assert restored[0].right is restored[2].right
    if kind == "group":
        assert restored[1].table is restored[0]
    artifact = export_model(restored.eval(), "torchscript", tmp_path / "export.pt", inputs[:1])
    with artifact.open("rb") as stream:
        exported = torch.jit.load(stream)
    torch.testing.assert_close(exported(inputs[:1]), restored(inputs[:1]), rtol=1e-10, atol=1e-10)
