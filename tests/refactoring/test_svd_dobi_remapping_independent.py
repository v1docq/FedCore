"""Independent adversarial checks for the declared portable remapping format.

Analytical rank-one inputs avoid a second call to the production SVD path.
The dense decoder below uses only named payload values and scalar indexing.
"""
from collections import OrderedDict
from copy import deepcopy
import io
import math
import types

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fedcore.algorithm.low_rank.approximation import NumericalDomainViolation
from fedcore.algorithm.low_rank.dobi_profiles import prepare_dobi
from fedcore.algorithm.low_rank.dobi_remapping import RemappedLinear, remapped_tensor_bytes
from fedcore.algorithm.low_rank.dobi_specs import DobiSVD
from fedcore.algorithm.low_rank.learned_layers import ZeroLinear
from fedcore.algorithm.low_rank.method_execution import _tensor_state_bytes
from fedcore.tools.registry.checkpoint_manager import (
    CheckpointError, CheckpointManager, build_model, describe_model,
)


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def assert_state_identical(model, before):
    actual = model.state_dict()
    assert actual.keys() == before.keys()
    for name, value in before.items():
        assert actual[name].dtype == value.dtype
        assert actual[name].device == value.device
        assert torch.equal(actual[name], value), name


def scalar_decode(layer):
    """Dense FP64 oracle, independent of decoded_factors and F.linear."""
    core = getattr(torch, layer.compute_dtype)
    left = torch.empty(layer.out_features, layer.rank, dtype=core)
    right = torch.empty(layer.rank, layer.in_features, dtype=core)
    short = min(layer.in_features, layer.out_features)
    for j in range(layer.rank):
        for i in range(layer.out_features):
            left[i, j] = (layer.left_codes[i, j].to(core) * layer.left_scale[j].to(core)
                          if i < short else layer.tail[i-short, j].to(core))
        for i in range(layer.in_features):
            right[j, i] = (layer.right_codes[j, i].to(core) * layer.right_scale[j].to(core)
                           if i < short else layer.tail[j, i-short].to(core))
    return left.double() @ right.double()


def quantize_known_vector(vector, scale_dtype, core_dtype):
    maximum = max(abs(float(x)) for x in vector)
    scale = torch.tensor(maximum / 127 if maximum else 1, dtype=scale_dtype)
    return torch.tensor([max(-127, min(127, round(float(x) / float(scale))))
                         for x in vector], dtype=core_dtype) * scale.to(core_dtype)


@pytest.mark.parametrize("shape", [(5, 3), (3, 5), (4, 4)])
@pytest.mark.parametrize("tail_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("scale_dtype,core_dtype", [(torch.float32, torch.float32),
                                                   (torch.float64, torch.float64)])
def test_analytic_rank_one_orientation_without_svd_oracle(shape, tail_dtype, scale_dtype, core_dtype):
    m, n = shape
    u = torch.tensor([(-1)**i * (i+1) for i in range(m)], dtype=torch.float64)
    # Stay away from half-code rounding boundaries: SVD may perturb an exact
    # mathematical half-tie to either side by one floating-point ulp.
    v = torch.tensor([1.13+.73*i+.17*i*i for i in range(n)], dtype=torch.float64)
    u, v = u / torch.linalg.vector_norm(u), v / torch.linalg.vector_norm(v)
    singular = 7.0
    weight = singular * u[:, None] * v[None, :]
    layer = RemappedLinear.from_weight(weight, 1, tail_dtype=str(tail_dtype)[6:],
        scale_dtype=str(scale_dtype)[6:], compute_dtype=str(core_dtype)[6:])
    if m >= n:
        a = quantize_known_vector(singular*u[:n], scale_dtype, core_dtype)
        a = torch.cat((a, (singular*u[n:]).to(tail_dtype).to(core_dtype)))
        b = quantize_known_vector(v, scale_dtype, core_dtype)
    else:
        a = quantize_known_vector(u, scale_dtype, core_dtype)
        b = quantize_known_vector(singular*v[:m], scale_dtype, core_dtype)
        b = torch.cat((b, (singular*v[m:]).to(tail_dtype).to(core_dtype)))
    expected = a.double()[:, None] * b.double()[None, :]
    torch.testing.assert_close(scalar_decode(layer), expected, rtol=2e-7, atol=2e-7)
    x = torch.arange(2*n, dtype=torch.float64).reshape(2, n) / 10
    torch.testing.assert_close(layer(x), x @ expected.T, rtol=3e-7, atol=3e-7)


@pytest.mark.parametrize("shape,rank", [((7, 4), 1), ((4, 7), 1), ((4, 4), 1),
                                        ((7, 4), 4), ((4, 7), 4), ((4, 4), 4)])
def test_float64_tail_owns_only_declared_storage(shape, rank):
    m, n = shape
    weight = torch.arange(1, m*n+1, dtype=torch.float64).reshape(m, n) / 7
    layer = RemappedLinear.from_weight(weight, rank, tail_dtype="float64",
                                      scale_dtype="float64", compute_dtype="float64")
    expected = 2*min(m,n)*rank + 8*abs(m-n)*rank + 16*rank
    assert layer.tail.untyped_storage().nbytes() == layer.tail.numel()*8
    assert _tensor_state_bytes(layer) == expected
    assert remapped_tensor_bytes(n, m, rank, tail_dtype="float64",
                                 scale_dtype="float64", bias=False) == expected


@pytest.mark.parametrize("shape", [(7, 4), (4, 7), (4, 4)])
def test_public_finalize_accepts_float64_tail_with_exact_byte_budget(shape):
    m, n = shape
    source = nn.Sequential(nn.Linear(n, m, dtype=torch.float64))
    calibration = torch.arange(1, 1+8*n, dtype=torch.float64).reshape(8, n) / 100
    objective = prepare_dobi(source, ["0"], DobiSVD(initial_fraction=1/min(shape),
        remapping="symmetric_int8_v1", remap_tail_dtype="float64",
        remap_scale_dtype="float64", remap_compute_dtype="float64"))
    budget = 2*min(m,n) + 8*abs(m-n) + 16 + 8*m
    record = objective.finalize(objective.model, calibration, 4, tensor_byte_budget=budget)
    assert record["tensor_bytes"] == budget
    assert record["layers"]["0"]["rank"] == 1


@pytest.mark.parametrize("remapping", ["none", "symmetric_int8_v1"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_budget_gradient_matches_independent_byte_slope(remapping, dtype):
    source = nn.Sequential(nn.Linear(3, 5, dtype=dtype), nn.Tanh(), nn.Linear(5, 2, dtype=dtype))
    x = torch.tensor([[1., 2., 3.], [2., -1., 1.], [-3., 2., 1.], [1., 1., -1.]], dtype=dtype)
    labels = source(x).detach()
    gradients = []
    for budget_weight in (0., 3.):
        objective = prepare_dobi(deepcopy(source), ["0", "2"], DobiSVD(
            initial_fraction=.6, remapping=remapping, budget_weight=budget_weight, target_fraction=.01))
        with objective:
            loss = objective(objective.model, x, labels)
            loss.backward()
        gradients.append([float(layer.k.grad) for layer in objective.layers.values()])
    if remapping == "none":
        slopes = [8/25, 7/25]
    else:
        # Two int8 short blocks, fp16 tail and two fp32 scales per rank.
        denominator = 25 * torch.empty((), dtype=dtype).element_size()
        slopes = [(2*3 + 2*2 + 8)/denominator, (2*2 + 2*3 + 8)/denominator]
    tolerance = 3e-6 if dtype == torch.float32 else 1e-12
    for base, penalized, slope in zip(*gradients, slopes):
        assert penalized-base == pytest.approx(3*slope, rel=tolerance, abs=tolerance)


def payload_for(layer):
    manager = CheckpointManager("unused", auto_cleanup=False)
    return manager, torch.load(io.BytesIO(manager.serialize_to_bytes(layer)), weights_only=True)


@pytest.mark.parametrize("field,value", [("version", True), ("rank", True), ("bias", 1),
    ("left_dtype", "uint8"), ("right_dtype", "uint8"), ("scale_dtype", "float16"),
    ("tail_dtype", "int8"), ("compute_dtype", "float16"), ("remapping", "none")])
def test_strict_descriptor_rejects_corrupt_precision_and_behavior(field, value):
    layer = RemappedLinear(3, 5, 2)
    description = describe_model(layer)
    description["config"][field] = value
    with pytest.raises(CheckpointError):
        build_model(description)


@pytest.mark.parametrize("method", ["decoded_factors", "validate_state", "representation_config"])
def test_checkpoint_refuses_overridden_remapping_methods(method):
    layer = RemappedLinear.from_weight(torch.eye(3, dtype=torch.float64), 3,
                                       scale_dtype="float64", compute_dtype="float64")
    if method == "decoded_factors":
        replacement = lambda self: (torch.zeros(3,3,dtype=torch.float64), torch.eye(3,dtype=torch.float64))
    elif method == "validate_state":
        replacement = lambda self, *args, **kwargs: None
    else:
        original = layer.representation_config()
        replacement = lambda self: original
    setattr(layer, method, types.MethodType(replacement, layer))
    with pytest.raises(CheckpointError, match="unmodified|declared"):
        CheckpointManager("unused", auto_cleanup=False).serialize_to_bytes(layer)


@pytest.mark.parametrize("group", [("left_codes", "right_codes"), ("left_scale", "right_scale")])
def test_checkpoint_refuses_aliases_between_independent_remap_components(group):
    layer = RemappedLinear.from_weight(torch.eye(3, dtype=torch.float64), 3,
                                       scale_dtype="float64", compute_dtype="float64")
    manager, payload = payload_for(layer)
    payload["topology"]["buffer_aliases"] = (group,)
    with pytest.raises(CheckpointError, match="alias|independent|storage|remap|Remap"):
        manager.restore(payload)


@pytest.mark.parametrize("corruption", ["scale_dtype", "scale_overflow", "compute_underflow",
                                        "tail_nan", "code_range", "bias_overflow"])
def test_invalid_incoming_state_is_checked_before_any_tensor_assignment(corruption):
    layer = RemappedLinear(2, 3, 2, tail_dtype="float64", scale_dtype="float64", bias_dtype="float64")
    before = deepcopy(layer.state_dict())
    state = deepcopy(before)
    state["left_codes"].fill_(17)  # An observable earlier key that must remain unchanged.
    if corruption == "scale_dtype":
        state["right_scale"] = state["right_scale"].float()
    elif corruption == "scale_overflow":
        state["right_codes"].fill_(127)
        state["right_scale"].fill_(torch.finfo(torch.float32).max)
    elif corruption == "compute_underflow":
        state["right_scale"].fill_(1e-100)
    elif corruption == "tail_nan":
        state["tail"][0,0] = math.nan
    elif corruption == "code_range":
        state["right_codes"][0,0] = -128
    else:
        state["bias"].fill_(1e100)
    state = OrderedDict(reversed(list(state.items())))
    with pytest.raises(RuntimeError):
        layer.load_state_dict(state, strict=True)
    assert_state_identical(layer, before)


def test_checkpoint_serialization_and_successful_restore_are_read_only():
    layer = RemappedLinear.from_weight(torch.eye(3, dtype=torch.float64), 3,
                                       scale_dtype="float64", compute_dtype="float64")
    layer.train()
    layer.bias = None
    before = deepcopy(layer.state_dict())
    manager, payload = payload_for(layer)
    restored = manager.restore(payload, model=layer)
    assert restored is not layer
    assert restored.training is True and layer.training is True
    assert_state_identical(layer, before)
    for name, value in restored.state_dict().items():
        assert value.data_ptr() != before[name].data_ptr() or not value.numel()


def test_multi_layer_local_errors_and_end_to_end_error_have_distinct_inputs():
    first = nn.Linear(3, 5, dtype=torch.float64)
    second = nn.Linear(5, 2, dtype=torch.float64)
    with torch.no_grad():
        first.weight.zero_()
        first.weight[:3,:3].copy_(torch.diag(torch.tensor([4., 2., 1.], dtype=torch.float64)))
        first.bias.copy_(torch.tensor([.1, -.2, .3, .4, -.5], dtype=torch.float64))
        second.weight.copy_(torch.tensor([[.8, -.6, .2, .4, -.3], [-.4, .1, .9, -.2, .7]], dtype=torch.float64))
        second.bias.copy_(torch.tensor([.2, -.1], dtype=torch.float64))
    source = nn.Sequential(first, nn.Tanh(), second)
    original = deepcopy(source)
    x = torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[0.,0.,0.]], dtype=torch.float64).repeat(3,1)
    objective = prepare_dobi(source, ["0", "2"], DobiSVD(remapping="symmetric_int8_v1",
        remap_scale_dtype="float64", remap_compute_dtype="float64"))
    with torch.no_grad():
        objective.layers["0"].k.fill_(1)
        objective.layers["2"].k.fill_(2)
    record = objective.finalize(objective.model, x, 4)
    hard_first = torch.zeros_like(original[0].weight)
    hard_first[0,0] = 4
    dense_inputs = {"0": x, "2": torch.tanh(original[0](x))}
    hard_weights = {"0": hard_first, "2": original[2].weight}
    for path in ("0", "2"):
        dense_layer = original[int(path)]
        input_rows = dense_inputs[path]
        dense = dense_layer(input_rows)
        hard = F.linear(input_rows, hard_weights[path], dense_layer.bias)
        decoded = F.linear(input_rows, scalar_decode(objective.model[int(path)]), dense_layer.bias)
        local = record["layers"][path]["calibrated_output_error"]
        for kind, difference in (("truncation", dense-hard), ("remapping", hard-decoded), ("total", dense-decoded)):
            assert local[kind+"_mse"] == pytest.approx(float(difference.square().mean()), abs=1e-25, rel=2e-12)
    dense_final = original(x)
    first_decoded = F.linear(x, scalar_decode(objective.model[0]), original[0].bias)
    decoded_final = F.linear(torch.tanh(first_decoded), scalar_decode(objective.model[2]), original[2].bias)
    assert record["end_to_end_calibrated_output_error"]["mse"] == pytest.approx(
        float((dense_final-decoded_final).square().mean()), rel=2e-12, abs=1e-25)
    assert record["layers"]["2"]["calibrated_output_error"]["total_mse"] != pytest.approx(
        record["end_to_end_calibrated_output_error"]["mse"], abs=1e-8)


def test_byte_rebudget_greedy_ties_use_path_order_and_count_all_biases():
    source = nn.Sequential(OrderedDict([
        ("z", nn.Linear(4,7,dtype=torch.float64)),
        ("t", nn.Tanh()),
        ("a", nn.Linear(7,4,dtype=torch.float64)),
    ]))
    x = torch.arange(32,dtype=torch.float64).reshape(8,4) / 10
    objective = prepare_dobi(source, ["z", "a"], DobiSVD(remapping="symmetric_int8_v1"))
    with torch.no_grad():
        for layer in objective.layers.values():
            layer.k.fill_(1)
    # Both layers cost 22 bytes per rank. All eleven fp64 bias values remain.
    record = objective.finalize(objective.model, x, 4, tensor_byte_budget=110)
    assert record["rank_reductions"] == [{"path":"a", "rank":0, "saved_tensor_bytes":22}]
    assert record["tensor_bytes"] == 110
    assert isinstance(objective.model.a, ZeroLinear)
    assert objective.model.z.rank == 1


class FailOnCall(nn.Module):
    def __init__(self, fail_at):
        super().__init__()
        self.calls, self.fail_at = 0, fail_at

    def forward(self, x):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("injected decoded end-to-end failure")
        return x


def test_failed_end_to_end_export_restores_graph_modes_and_allows_retry():
    source = nn.Sequential(nn.Linear(3,5,dtype=torch.float64), FailOnCall(6))
    source.train()
    source[1].eval()
    x = torch.arange(24,dtype=torch.float64).reshape(8,3) / 100
    objective = prepare_dobi(source, ["0"], DobiSVD(initial_fraction=1., remapping="symmetric_int8_v1"))
    work = objective.model
    wrapper = work[0]
    before = deepcopy(work.state_dict())
    modes = {module:module.training for module in work.modules()}
    with pytest.raises(RuntimeError, match="injected decoded"):
        objective.finalize(work, x, 4)
    assert work[0] is wrapper and objective.model is work
    assert not objective.finalized and "export" not in objective.evidence
    assert not wrapper.export_mode and not wrapper._forward_pre_hooks
    assert all(module.training == training for module,training in modes.items())
    assert_state_identical(work, before)
    work[1].fail_at = -1
    objective.finalize(work, x, 4)
    assert objective.finalized and type(objective.model[0]) is RemappedLinear


def test_local_error_sum_overflow_refuses_before_publishing_model():
    first = nn.Linear(1,1,bias=False,dtype=torch.float64)
    with torch.no_grad():
        first.weight.fill_(1e154)
    objective = prepare_dobi(nn.Sequential(first,nn.Tanh()), ["0"],
        DobiSVD(initial_fraction=0.,remapping="symmetric_int8_v1"))
    work = objective.model
    before = deepcopy(work.state_dict())
    x = torch.ones(16,1,dtype=torch.float64)
    with pytest.raises(NumericalDomainViolation, match="finite|overflow|error"):
        objective.finalize(work,x,4)
    assert not objective.finalized and "export" not in objective.evidence
    assert objective.model is work
    assert_state_identical(work,before)


@pytest.mark.parametrize("changes", [{"in_features":True}, {"out_features":0}, {"rank":-1},
    {"rank":3}, {"bias":1}, {"scale_dtype":"float16"}, {"tail_dtype":"not_a_dtype"}])
def test_public_byte_helper_rejects_invalid_descriptions(changes):
    values = dict(in_features=2,out_features=3,rank=1,bias=False)
    values.update(changes)
    with pytest.raises((ValueError,NumericalDomainViolation)):
        remapped_tensor_bytes(**values)


def test_script_retains_finite_input_guards_and_dynamic_leading_dimensions():
    layer = RemappedLinear.from_weight(torch.eye(3,dtype=torch.float64),3,
                                       scale_dtype="float64",compute_dtype="float64")
    scripted = torch.jit.script(layer)
    for shape in ((1,3),(7,3),(2,4,3)):
        x = torch.arange(math.prod(shape),dtype=torch.float64).reshape(shape)
        torch.testing.assert_close(scripted(x),x,rtol=0,atol=0)
    with pytest.raises(torch.jit.Error,match="nonfinite"):
        scripted(torch.full((2,3),math.nan,dtype=torch.float64))
    with pytest.raises(torch.jit.Error,match="compatible"):
        scripted(torch.ones(2,4,dtype=torch.float64))
