"""Public facade and checked transition contracts for finite G5 composition."""
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.compression_composition import (
    AddResidual, CheckedComposition, CompositionRecipe, CompositionRequest,
    CompositionSuccess, FinalFormat, LowRank, MaterializeDense, PruneMask,
    QuantizeRemapped, Representation, execute_composition, prepare_composition,
)
from fedcore.algorithm.low_rank.low_rank_opt import LowRankModel
from fedcore.algorithm.low_rank.plans import ValidationFailure
from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def source():
    with torch.random.fork_rng(devices=[]):
        model = nn.Sequential(nn.Linear(4, 3, dtype=torch.float64)).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[3., 1., -2., .5], [1., -2., .25, 4.],
                                          [2., .5, 1., -1.]], dtype=torch.float64))
        model[0].bias.copy_(torch.tensor([.1, -.2, .3], dtype=torch.float64))
    return model


def request(*steps, **options):
    return CompositionRequest((CompositionRecipe("0", steps),), **options)


def result(model, *steps):
    checked = prepare_composition(model, request(*steps))
    assert type(checked) is CheckedComposition, checked
    output = execute_composition(model, checked)
    assert type(output) is CompositionSuccess, output
    return output


def weight(layer):
    inputs = torch.eye(layer.in_features, dtype=torch.float64)
    zero = layer(torch.zeros(1, layer.in_features, dtype=torch.float64))
    return (layer(inputs) - zero).T.detach().numpy()


def test_remapping_and_rank_truncation_do_not_commute_and_record_both_orders():
    model = source()
    quantize = QuantizeRemapped(tail_dtype="float64", scale_dtype="float64", compute_dtype="float64")
    quantized = result(model, quantize)
    quantize_then_rank = result(model, quantize, MaterializeDense(), LowRank(1))
    rank_then_quantize = result(model, LowRank(1), quantize)
    u, singular, vh = np.linalg.svd(weight(quantized.model[0]), full_matrices=False)
    oracle = (u[:, :1] * singular[:1]) @ vh[:1]
    np.testing.assert_allclose(weight(quantize_then_rank.model[0]), oracle, rtol=2e-13, atol=2e-13)
    assert np.max(np.abs(weight(quantize_then_rank.model[0]) - weight(rank_then_quantize.model[0]))) > 1e-4
    assert quantize_then_rank.fingerprint != rank_then_quantize.fingerprint
    assert "reconstruct_then_SVD" in rank_then_quantize.history[-1].semantics
    transitions = quantize_then_rank.workflow.transitions
    assert [item.output_state.representation for item in transitions] == [
        Representation.PACKED_FACTORS, Representation.DENSE, Representation.REAL_FACTORS]
    assert transitions[1].output_state.rank is None
    assert transitions[1].output_state.runtime == "cpu_float"


def test_ordered_transitions_are_metadata_bound_and_revalidated_before_execution(monkeypatch):
    model = source()
    checked = prepare_composition(model, request(LowRank(1), MaterializeDense(), PruneMask((0,))))
    assert type(checked) is CheckedComposition
    forged_transition = replace(checked.transitions[-1], output_state=replace(
        checked.transitions[-1].output_state, representation=Representation.PACKED_FACTORS))
    forged = replace(checked, transitions=(*checked.transitions[:-1], forged_transition))

    def forbidden(*args, **kwargs):
        pytest.fail("Forged metadata reached a solver")

    monkeypatch.setattr(torch.linalg, "svd", forbidden)
    output = execute_composition(model, forged)
    assert type(output) is ValidationFailure
    assert output.violations[0].code == "StaleComposition"


def test_public_facade_publishes_success_and_leaves_prior_state_on_refusal():
    model = source()
    facade = LowRankModel.__new__(LowRankModel)
    registry_calls = []
    facade._model_id_before, facade._model_id_after = "before", "after"
    facade._model_before_cached, facade._model_after_cached = "before", "after"
    facade._fedcore_id = "composition-test"
    facade._registry = SimpleNamespace(register_changes=lambda **options: registry_calls.append(options))
    facade.transform_result = "previous"
    facade.trainer = SimpleNamespace(model=model)
    checked = facade.prepare_compression_composition(model, request(LowRank(1)))
    good = facade.compress_composition(model, checked)
    assert type(good) is CompositionSuccess
    assert facade.model_before is model and facade.model_after is good.model
    assert facade.trainer.model is good.model
    assert facade.trainer._fedcore_requires_optimizer_rebuild
    assert len(registry_calls) == 2
    prior = facade.transform_result
    with torch.no_grad():
        model[0].weight.add_(1)
    refused = facade.compress_composition(model, checked)
    assert type(refused) is ValidationFailure
    assert facade.model_after is good.model and facade.transform_result is prior
    assert len(registry_calls) == 2


@pytest.mark.parametrize("steps", [(AddResidual(1), AddResidual(1)),
    (LowRank(1), PruneMask((0,))), (QuantizeRemapped(compute_dtype="float64"), LowRank(1))])
def test_unsupported_transitions_are_refused_before_transform(monkeypatch, steps):
    model = source()
    monkeypatch.setattr(torch.linalg, "svd", lambda *args, **kwargs: pytest.fail("Solver called during validation"))
    output = prepare_composition(model, request(*steps))
    assert type(output) is ValidationFailure
    assert output.violations[0].code == "UnsupportedTransition"


def test_finite_grammar_rejects_sparse_backend_and_unknown_export_without_effects(monkeypatch):
    @dataclass(frozen=True)
    class PhysicalSparse:
        format: str = "csr"

    model = source()
    monkeypatch.setattr(torch.linalg, "svd", lambda *args, **kwargs: pytest.fail("Unsupported branch transformed"))
    sparse = prepare_composition(model, request(PhysicalSparse()))
    assert type(sparse) is ValidationFailure and sparse.violations[0].code == "UnsupportedCapability"
    onnx = prepare_composition(model, request(LowRank(1), export_format="onnx"))
    assert type(onnx) is ValidationFailure and onnx.violations[0].code == "UnsupportedExport"
    supported = prepare_composition(model, request(LowRank(1), export_format=FinalFormat.CHECKPOINT))
    assert type(supported) is CheckedComposition


@pytest.mark.parametrize("change", ["nonpersistent", "extra_persistent", "method"])
def test_unsupported_source_export_state_is_refused_before_transform(monkeypatch, change):
    model = source()
    if change == "nonpersistent":
        model[0].register_buffer("cache", torch.ones(3), persistent=False)
    elif change == "extra_persistent":
        model[0].register_buffer("indices", torch.arange(3))
    else:
        model[0].state_dict = lambda *args: {}
    monkeypatch.setattr(torch.linalg, "svd", lambda *args, **kwargs: pytest.fail("Unsupported state transformed"))
    output = prepare_composition(model, request(LowRank(1)))
    assert type(output) is ValidationFailure
    assert output.violations[0].code in ("UnsupportedExport", "UnsupportedCapability")


def test_module_aliases_are_replaced_together_but_duplicate_alias_recipes_are_refused():
    model = source()
    model.add_module("alias", model[0])
    checked = prepare_composition(model, request(LowRank(1)))
    assert type(checked) is CheckedComposition
    output = execute_composition(model, checked)
    assert type(output) is CompositionSuccess, output
    assert output.model[0] is output.model.alias
    assert output.storage_bytes == 8 * (4 + 3 + 3)
    duplicate = CompositionRequest((CompositionRecipe("0", (LowRank(1),)),
                                    CompositionRecipe("alias", (LowRank(1),))))
    rejected = prepare_composition(model, duplicate)
    assert type(rejected) is ValidationFailure
    assert rejected.violations[0].code == "OverlappingScopes"


@pytest.mark.parametrize("placement", ["root", "sequential"])
def test_nonexecutable_collections_are_refused_as_final_inference_positions(monkeypatch, placement):
    collection = nn.ModuleDict({"projection": source()[0]})
    if placement == "root":
        model, path = collection, "projection"
    else:
        model, path = nn.Sequential(collection), "0.projection"
    monkeypatch.setattr(torch.linalg, "svd", lambda *args, **kwargs: pytest.fail("Nonexecutable graph transformed"))
    output = prepare_composition(model, CompositionRequest((CompositionRecipe(path, (LowRank(1),)),)))
    assert type(output) is ValidationFailure
    assert output.violations[0].code == "UnsupportedExport"


def test_checkpoint_cannot_drop_an_undeclared_child_without_tensor_state(monkeypatch):
    model = source()
    with torch.random.fork_rng(devices=[]):
        model.add_module("unchanged", nn.Linear(3, 3, dtype=torch.float64))
    model.unchanged.add_module("extra", nn.ReLU())
    monkeypatch.setattr(torch.linalg, "svd", lambda *args, **kwargs: pytest.fail("Lossy graph transformed"))
    output = prepare_composition(model, request(LowRank(1)))
    assert type(output) is ValidationFailure
    assert output.violations[0].code == "UnsupportedExport"
    assert "module graph" in output.violations[0].message


@pytest.mark.parametrize("invalid_bias", [torch.zeros(3, dtype=torch.float32),
                                        torch.zeros(4, dtype=torch.float64)])
def test_source_real_factors_require_bias_shape_and_dtype_compatible_with_weights(monkeypatch, invalid_bias):
    with torch.random.fork_rng(devices=[]):
        layer = FactorizedLinear(4, 3, 1, dtype=torch.float64)
    layer.left.bias = nn.Parameter(invalid_bias.clone())
    model = nn.Sequential(layer).eval()
    monkeypatch.setattr(torch.linalg, "svd", lambda *args, **kwargs: pytest.fail("Invalid factor bias reached solver"))
    output = prepare_composition(model, request(MaterializeDense(), LowRank(1)))
    assert type(output) is ValidationFailure
    assert output.violations[0].code == "UnsupportedCapability"
