"""Independent numerical and lifecycle oracles for bounded Dobi-SVD."""
from copy import deepcopy
from dataclasses import asdict
import io
import json

import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.approximation import NumericalDomainViolation
from fedcore.algorithm.low_rank.dobi_specs import DobiSVD
from fedcore.algorithm.low_rank.dobi_profiles import (
    smooth_activation_truncation, empty_direction_ipca, update_direction_ipca,
    fixed_output_factors, estimate_dobi_bytes, prepare_dobi,
)
from fedcore.algorithm.low_rank.learned_layers import ZeroLinear
from fedcore.algorithm.low_rank.method_errors import MethodSpecError
from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _linear(in_features=4, out_features=4, bias=True, dtype=torch.float64):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(712)
        return nn.Linear(in_features, out_features, bias=bias, dtype=dtype)


def _rows(n=8, d=4):
    return torch.randn(n, d, dtype=torch.float64, generator=torch.Generator().manual_seed(89))


def _reference(a, k, beta):
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    gate = (1 + torch.tanh(beta * (k - torch.arange(1, len(s) + 1, dtype=a.dtype)))) / 2
    return (u * (s * gate)) @ vh


@pytest.mark.parametrize("options", [
    {"updates": 0}, {"updates": True}, {"seed": -1}, {"target_fraction": 1.1},
    {"initial_fraction": float("nan")}, {"beta": 0}, {"budget_weight": -1},
    {"spectrum_policy": "silently_epsilon"}, {"remapping": "float16"},
    {"max_activation_rows": 0}, {"reduction": "none"}, {"export_rounding": "truncate"},
])
def test_pure_spec_rejects_ambiguous_or_invalid_options(options):
    with pytest.raises(MethodSpecError):
        DobiSVD(**options)


def test_pure_spec_is_json_safe():
    assert json.loads(json.dumps(asdict(DobiSVD()))) == asdict(DobiSVD())


@pytest.mark.parametrize("shape", [(5, 3), (3, 5), (4, 4)])
def test_activation_and_cutoff_gradients_match_independent_torch_svd(shape):
    generator = torch.Generator().manual_seed(413)
    a = torch.randn(*shape, generator=generator, dtype=torch.float64, requires_grad=True)
    k = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(*shape, generator=generator, dtype=torch.float64)
    result = smooth_activation_truncation(a, k, beta=1.3, spectrum_policy="error")
    reference = _reference(a, k, 1.3)
    actual = torch.autograd.grad((result * upstream).sum(), (a, k), retain_graph=True)
    expected = torch.autograd.grad((reference * upstream).sum(), (a, k))
    assert torch.allclose(result, reference, atol=1e-12, rtol=1e-12)
    for observed, oracle in zip(actual, expected):
        assert torch.allclose(observed, oracle, atol=2e-11, rtol=2e-10)


@pytest.mark.parametrize("shape", [(5, 3), (3, 5)])
def test_directional_finite_difference_checks_both_derivatives(shape):
    a = _rows(shape[0], shape[1]).requires_grad_()
    k = torch.tensor(1.4, dtype=torch.float64, requires_grad=True)
    da = a.detach().flip(0) / 3
    g = torch.arange(a.numel(), dtype=a.dtype).reshape_as(a) / 10
    y = smooth_activation_truncation(a, k, beta=0.8, spectrum_policy="error")
    ga, gk = torch.autograd.grad((y * g).sum(), (a, k))
    eps = 1e-6
    plus = smooth_activation_truncation(a.detach() + eps * da, k.detach() + eps * 0.37,
                                       beta=0.8, spectrum_policy="error")
    minus = smooth_activation_truncation(a.detach() - eps * da, k.detach() - eps * 0.37,
                                        beta=0.8, spectrum_policy="error")
    finite_difference = ((plus - minus) * g).sum() / (2 * eps)
    assert torch.allclose((ga * da).sum() + gk * 0.37, finite_difference, atol=1e-8, rtol=1e-7)


@pytest.mark.parametrize("singular", [(2.0, 2.0, 0.4), (2.0, 2.0 - 1e-9, 0.4), (0.0, 0.0, 0.0)])
def test_cluster_policy_is_finite_basis_invariant_and_locally_differentiated(singular):
    a = torch.diag(torch.tensor(singular, dtype=torch.float64)).requires_grad_()
    k = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    options = dict(beta=0.7, spectrum_rtol=1e-4, spectrum_atol=1e-4)
    y = smooth_activation_truncation(a, k, **options)
    if singular[0] != 0:
        assert float((y[0, 0] / a[0, 0]).detach()) == pytest.approx(float((y[1, 1] / a[1, 1]).detach()))
    assert torch.autograd.gradcheck(lambda x, cutoff: smooth_activation_truncation(x, cutoff, **options),
                                    (a, k), eps=1e-6, atol=2e-6, rtol=1e-5)


def test_repeated_spectrum_rotation_has_same_map_and_error_policy_refuses_it():
    a = torch.diag(torch.tensor([2.0, 2.0, 0.3], dtype=torch.float64))
    q = torch.tensor([[0.8, -0.6, 0], [0.6, 0.8, 0], [0, 0, 1]], dtype=a.dtype)
    k = torch.tensor(1.1, dtype=a.dtype)
    assert torch.allclose(smooth_activation_truncation(q @ a, k), q @ smooth_activation_truncation(a, k))
    with pytest.raises(NumericalDomainViolation, match="near/repeated"):
        smooth_activation_truncation(a, k, spectrum_policy="error")


def test_finite_beta_is_not_a_hard_rank():
    a = torch.diag(torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64))
    y = smooth_activation_truncation(a, torch.tensor(1.0, dtype=a.dtype), beta=0.5)
    assert torch.linalg.matrix_rank(y) == 3


def test_ipca_transition_is_pure_and_order_dependence_is_explicit():
    state = empty_direction_ipca(2, 1)
    blocks = [torch.tensor([[2.0, 0.0]], dtype=torch.float64),
              torch.tensor([[0.0, 1.9]], dtype=torch.float64),
              torch.tensor([[0.0, 1.9]], dtype=torch.float64)]
    first = update_direction_ipca(state, blocks[0])
    assert state.rows_seen == 0 and state.components.shape == (0, 2)
    forward, reverse = state, state
    for block in blocks:
        forward = update_direction_ipca(forward, block)
    for block in reversed(blocks):
        reverse = update_direction_ipca(reverse, block)
    assert first.rows_seen == 1
    assert forward.rows_seen == reverse.rows_seen == 3
    assert forward.batch_hashes != reverse.batch_hashes
    assert not torch.allclose(forward.components.T @ forward.components,
                              reverse.components.T @ reverse.components)


def test_full_summary_ipca_matches_independent_uncentered_covariance():
    blocks = [_rows(4, 3), _rows(5, 3).flip(0)]
    state = empty_direction_ipca(3, 3)
    for block in blocks:
        state = update_direction_ipca(state, block)
    full = torch.cat(blocks)
    reconstructed_covariance = state.components.T @ torch.diag(state.singular_values.square()) @ state.components
    assert torch.allclose(reconstructed_covariance, full.T @ full, atol=2e-12, rtol=2e-12)


def test_output_factor_orientation_matches_dense_projection():
    w = _linear(4, 5).weight.detach()
    basis = torch.linalg.qr(_rows(5, 2)).Q
    pair = fixed_output_factors(w, basis)
    x = _rows(6, 4)
    assert torch.allclose((x @ pair.right.T) @ pair.left.T,
                          (x @ w.T) @ basis @ basis.T, atol=1e-12, rtol=1e-12)


def test_estimate_and_prepare_do_not_consume_rng_or_modify_frozen_weights():
    model = nn.Sequential(_linear())
    original = deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    costs = estimate_dobi_bytes(model, ["0"], DobiSVD(max_activation_rows=9))
    objective = prepare_dobi(model, ["0"], DobiSVD(max_activation_rows=9))
    assert all(type(value) is int and value >= 0 for value in costs.values())
    assert costs["svd_backward_lower_bound_bytes"] > costs["cutoff_parameter_bytes"]
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(objective.layers["0"].base.weight, original["0.weight"])
    trainable = [name for name, parameter in objective.model.named_parameters() if parameter.requires_grad]
    assert trainable == ["0.k"]


def test_real_optimizer_updates_continuous_k_and_freezes_all_base_parameters():
    model = nn.Sequential(_linear())
    objective = prepare_dobi(model, ["0"], DobiSVD(initial_fraction=0.5, target_fraction=0,
                                                  beta=0.8, budget_weight=2, updates=2))
    before = deepcopy(objective.layers["0"].base.state_dict())
    k_before = float(objective.layers["0"].k.detach())
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.03)
    x = _rows()
    with objective:
        for _ in range(2):
            optimizer.zero_grad()
            loss = objective(model, x, torch.zeros_like(x))
            loss.backward()
            assert torch.isfinite(objective.layers["0"].k.grad)
            optimizer.step()
            objective.after_step(model, optimizer)
    assert float(objective.layers["0"].k.detach()) < k_before
    assert all(torch.equal(value, objective.layers["0"].base.state_dict()[key]) for key, value in before.items())
    assert objective.evidence["completed_optimizer_steps"] == 2
    json.dumps(objective.evidence, allow_nan=False)


def test_two_layer_gradient_reaches_first_cutoff_through_second_activation_svd():
    model = nn.Sequential(_linear(3, 4), nn.Tanh(), _linear(4, 3))
    objective = prepare_dobi(model, ["0", "2"], DobiSVD(beta=0.6, budget_weight=0))
    x = _rows(7, 3)
    with objective:
        loss = objective(model, x, torch.ones_like(x))
        loss.backward()
    assert abs(float(objective.layers["0"].k.grad)) > 1e-8
    assert abs(float(objective.layers["2"].k.grad)) > 1e-8


@pytest.mark.parametrize("fraction, expected_type", [(0.0, ZeroLinear), (0.5, FactorizedLinear), (1.0, nn.Linear)])
def test_fixed_export_zero_partial_full_is_portable_and_has_no_runtime_svd(fraction, expected_type, monkeypatch):
    source = _linear()
    source_state = deepcopy(source.state_dict())
    model = nn.Sequential(source)
    objective = prepare_dobi(model, ["0"], DobiSVD(initial_fraction=fraction, beta=0.8))
    x = _rows()
    rng = torch.get_rng_state().clone()
    record = objective.finalize(model, x, 4)
    exported = objective.model[0]
    assert type(exported) is expected_type
    assert torch.equal(torch.get_rng_state(), rng)
    assert not exported._forward_pre_hooks
    if fraction == 0:
        assert not any(p.shape == (4, 4) for p in exported.parameters())
        expected = source_state["bias"].expand_as(x)
    elif fraction == 1:
        expected = x @ source_state["weight"].T + source_state["bias"]
    else:
        assert record["layers"]["0"]["basis_sha256"] is not None
        expected = x @ (exported.left.weight @ exported.right.weight).T + source_state["bias"]
    def no_svd(*args, **kwargs):
        raise AssertionError("deployment must not run SVD")
    monkeypatch.setattr(torch.linalg, "svd", no_svd)
    assert torch.allclose(model(x), expected, atol=1e-12, rtol=1e-12)
    stream = io.BytesIO()
    torch.save(exported.state_dict(), stream)
    stream.seek(0)
    if fraction == 0:
        restored = ZeroLinear(4, 4, dtype=torch.float64)
    elif fraction == 1:
        restored = _linear()
    else:
        restored = FactorizedLinear(4, 4, 2, dtype=torch.float64)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    assert torch.equal(restored(x), exported(x))
    json.dumps(record, allow_nan=False)


def test_root_linear_prepare_and_finalize_return_the_new_root():
    objective = prepare_dobi(_linear(), [""], DobiSVD(initial_fraction=0.5))
    initial = objective.model
    objective.finalize(initial, _rows(), 4)
    assert type(objective.model) is FactorizedLinear and objective.model is not initial


def test_deterministic_export_reports_ordered_batch_hashes():
    x = _rows()
    results = []
    for _ in range(2):
        objective = prepare_dobi(nn.Sequential(_linear()), ["0"], DobiSVD())
        record = objective.finalize(objective.model, x, 3)
        results.append((record, deepcopy(objective.model.state_dict())))
    assert results[0][0] == results[1][0]
    assert all(torch.equal(v, results[1][1][k]) for k, v in results[0][1].items())
    assert len(results[0][0]["calibration_batch_sha256"]) == 3


def test_failure_cleans_export_hooks_modes_and_transient_flags():
    class Conditional(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = _linear(), _linear()
        def forward(self, x):
            return self.a(x)
    model = Conditional().train()
    objective = prepare_dobi(model, ["a", "b"], DobiSVD())
    with pytest.raises(NumericalDomainViolation, match="every target"):
        objective.finalize(model, _rows(), 4)
    assert model.training
    assert all(not layer.export_mode and not layer._forward_pre_hooks for layer in objective.layers.values())
    assert not objective.finalized


def test_row_limit_fails_before_svd_allocation(monkeypatch):
    objective = prepare_dobi(nn.Sequential(_linear()), ["0"], DobiSVD(max_activation_rows=2))
    def no_svd(*args, **kwargs):
        raise AssertionError("row guard must precede SVD")
    monkeypatch.setattr(torch.linalg, "svd", no_svd)
    with objective, pytest.raises(NumericalDomainViolation, match="activation rows"):
        objective(objective.model, _rows(), _rows())
    assert not objective.active


def test_reused_layer_and_aliases_are_explicitly_refused():
    layer = _linear()
    with pytest.raises(NumericalDomainViolation, match="aliases"):
        prepare_dobi(nn.Sequential(layer, layer), ["0"], DobiSVD())
    class Reused(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = _linear()
        def forward(self, x):
            return self.linear(self.linear(x))
    objective = prepare_dobi(Reused(), ["linear"], DobiSVD())
    with objective, pytest.raises(NumericalDomainViolation, match="exactly once"):
        objective(objective.model, _rows(), _rows())
