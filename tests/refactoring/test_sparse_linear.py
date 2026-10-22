from copy import deepcopy
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fedcore.algorithm.pruning.sparse.linear import (
    sparsegpt_nm, quadratic_objective, scale_coordinates, hassle_free, prepare_losparse,
    cubic_residual_count, LoSparseTrainingCallback, finalize_losparse,
)
from fedcore.algorithm.pruning.sparse.specs import NMStructure, HASSLEFreeSpec, LoSparseSpec, SparseDomainError


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def dense(result, width):
    matrix = torch.zeros(result.values.shape[0], width, dtype=result.values.dtype)
    for row in range(matrix.shape[0]):
        matrix[row, result.indices[row].flatten()] = result.values[row].flatten()
    return matrix


def seeded_linear(width=8, out=5, dtype=torch.float64):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        return nn.Linear(width, out, dtype=dtype)


def inputs(width=8):
    generator = torch.Generator().manual_seed(24)
    rows = torch.randn(32, width, dtype=torch.float64, generator=generator)
    rows[:, 1] = rows[:, 0] * 1.5 + rows[:, 1] * 0.2
    return rows


def test_quadratic_output_loss_regularizer_and_independent_factor_gradients():
    x = inputs(4)
    h0 = x.T @ x
    h = h0 + 0.17 * torch.eye(4, dtype=torch.float64)
    w = seeded_linear(4, 3).weight.detach()
    left = torch.tensor([[.2, .4], [-.1, .3], [.5, -.2]], dtype=torch.float64, requires_grad=True)
    right = torch.tensor([[.1, .6, -.2, .3], [.5, -.7, .4, .2]], dtype=torch.float64, requires_grad=True)
    error = w - left @ right
    loss = quadratic_objective(error, h)
    independent = (x @ error.T).square().sum() + .17 * error.square().sum()
    torch.testing.assert_close(loss, independent)
    loss.backward()
    torch.testing.assert_close(left.grad, -2 * error.detach() @ h @ right.detach().T)
    torch.testing.assert_close(right.grad, -2 * left.detach().T @ error.detach() @ h)
    # Finite-difference derivative provides a second, non-autograd oracle.
    epsilon = 1e-6
    for tensor, grad, location in ((left, left.grad, (1, 0)), (right, right.grad, (0, 2))):
        plus, minus = tensor.detach().clone(), tensor.detach().clone()
        plus[location] += epsilon
        minus[location] -= epsilon
        eplus = w - (plus @ right.detach() if tensor is left else left.detach() @ plus)
        eminus = w - (minus @ right.detach() if tensor is left else left.detach() @ minus)
        oracle = ((x @ eplus.T).square().sum() + .17 * eplus.square().sum()
                  - (x @ eminus.T).square().sum() - .17 * eminus.square().sum()) / (2 * epsilon)
        torch.testing.assert_close(grad[location], oracle, rtol=1e-7, atol=1e-7)


def test_coordinate_transform_preserves_full_objective_and_division_inverse():
    w = seeded_linear(4, 3).weight.detach()
    x = inputs(4)
    h = x.T @ x + .2 * torch.eye(4, dtype=torch.float64)
    scaled, transformed_h, scale = scale_coordinates(w, h)
    error = w - .3
    torch.testing.assert_close(quadratic_objective(error, h), quadratic_objective(error * scale, transformed_h))
    torch.testing.assert_close(scaled / scale, w)
    assert transformed_h[0, 1].abs() > .1
    torch.testing.assert_close(transformed_h.diagonal(), torch.ones(4, dtype=torch.float64))


def test_sparsegpt_identity_matches_group_magnitude():
    w = torch.tensor([[1., -4., 3., .1, -2., 1., 7., 5.], [4., 2., .2, 3., 9., 5., 1., 2.]], dtype=torch.float64)
    result = sparsegpt_nm(w, torch.eye(8, dtype=torch.float64), NMStructure(2, 4))
    expected = torch.tensor([[0., -4., 3., 0., 0., 0., 7., 5.], [4., 0., 0., 3., 9., 5., 0., 0.]], dtype=torch.float64)
    torch.testing.assert_close(dense(result, 8), expected, rtol=0, atol=0)
    assert result.objective == float((w - expected).square().sum())


def test_sparsegpt_one_removed_coordinate_matches_constrained_obs_oracle():
    w = torch.tensor([[.01, 2., 3., 4.]], dtype=torch.float64)
    h = torch.tensor([[2., .6, .3, .2], [.6, 2., .4, .1], [.3, .4, 1.5, .2], [.2, .1, .2, 1.]], dtype=torch.float64)
    result = sparsegpt_nm(w, h, NMStructure(3, 4))
    actual = dense(result, 4)
    # Removing first coordinate leaves all free later coordinates available.
    expected = w.clone()
    expected[0, 0] = 0
    expected[0, 1:] += torch.linalg.solve(h[1:, 1:], h[1:, 0] * w[0, 0])
    torch.testing.assert_close(actual, expected)
    magnitude = w.clone()
    magnitude[0, 0] = 0
    assert quadratic_objective(w - actual, h) < quadratic_objective(w - magnitude, h)


@pytest.mark.parametrize("normalization", ["mean", "sum"])
@pytest.mark.parametrize("rule", ["trace", "mean_diagonal"])
def test_hassle_damping_final_operator_and_seed_contract(normalization, rule):
    layer = seeded_linear()
    state = deepcopy(layer.state_dict())
    rows = inputs()
    spec = HASSLEFreeSpec(NMStructure(2, 4, 2), outer_steps=2, inner_steps=6,
                         normalization=normalization, damping_rule=rule, seed=83)
    rng = torch.random.get_rng_state().clone()
    module, evidence = hassle_free(layer, rows, spec)
    assert torch.equal(torch.random.get_rng_state(), rng)
    for name, value in state.items():
        assert torch.equal(layer.state_dict()[name], value)
    sparse = dense(module, 8)
    weight = sparse + module.factors.left.weight @ module.factors.right.weight
    torch.testing.assert_close(module(rows), F.linear(rows, weight, layer.bias))
    error = layer.weight - weight
    h0 = rows.T @ rows / (len(rows) if normalization == "mean" else 1)
    damping = .01 * float(h0.trace() if rule == "trace" else h0.diagonal().mean())
    assert evidence.damping_value == pytest.approx(damping)
    assert evidence.best_objective == pytest.approx(float(quadratic_objective(error, h0 + damping * torch.eye(8))))
    assert evidence.output_error_squared == pytest.approx(float((rows @ error.T).square().sum()))
    again, repeated = hassle_free(layer, rows, spec)
    for name, value in module.state_dict().items():
        assert torch.equal(again.state_dict()[name], value)
    assert repeated == evidence


def test_full_metric_differs_from_diagonal_on_correlated_inputs():
    w = seeded_linear(4, 3).weight.detach()
    x = inputs(4)
    h = x.T @ x + .1 * torch.eye(4, dtype=torch.float64)
    full = dense(sparsegpt_nm(w, h, NMStructure(2, 4)), 4)
    diagonal = dense(sparsegpt_nm(w, h.diag().diag(), NMStructure(2, 4)), 4)
    assert not torch.allclose(full, diagonal)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rank,scale", [(0, 1.), (2, .3), (2, 2.)])
def test_losparse_exact_initialization_and_rng(dtype, rank, scale):
    layer = seeded_linear(dtype=dtype)
    rng = torch.random.get_rng_state().clone()
    module = prepare_losparse(layer, LoSparseSpec(rank, 2, 6, initial_scale=scale))
    assert torch.equal(torch.random.get_rng_state(), rng)
    x = inputs().to(dtype)
    torch.testing.assert_close(module(x), layer(x), rtol=3e-6 if dtype == torch.float32 else 1e-12,
                               atol=3e-7 if dtype == torch.float32 else 1e-12)


def test_losparse_schedule_frequency_ties_and_real_adam_state_cleanup():
    module = prepare_losparse(seeded_linear(4, 4), LoSparseSpec(1, 0, 6, warmup_steps=1, final_step=5, delta_t=4))
    counts = [cubic_residual_count(t, 4, module.spec) for t in range(7)]
    assert counts[:2] == [4, 4] and counts[-2:] == [0, 0]
    assert counts == sorted(counts, reverse=True)
    objective = LoSparseTrainingCallback(module, lambda model, x, y: F.mse_loss(model(x), y))
    optimizer = torch.optim.Adam(module.parameters(), lr=.01)
    x, target = inputs(4), torch.ones(32, 4, dtype=torch.float64)
    with objective:
        for step in range(1, 7):
            optimizer.zero_grad()
            loss = objective(module, x, target)
            loss.backward()
            objective.before_step(module, optimizer)
            if step == 4:
                module.importance_ema.zero_()  # tied scores retain original row ids
            optimizer.step()
            objective.after_step(module, optimizer)
            if step < 4:
                assert module.support.all()
            if step == 4:
                assert torch.equal(module.support, torch.tensor([True, False, False, False]))
            removed = ~module.support
            assert not module.residual[removed].count_nonzero()
            assert not optimizer.state[module.residual]["exp_avg"][removed].count_nonzero()
    compact = finalize_losparse(module)
    torch.testing.assert_close(compact(x), module(x))
    assert compact.residual_rows.shape == (0, 4)
    assert not any("importance" in key or key == "residual" for key in compact.state_dict())


def test_reentry_gets_inactive_gradient_and_can_restore_selected_row():
    module = prepare_losparse(seeded_linear(4, 3), LoSparseSpec(1, 1, 2, allow_reentry=True, ema_beta=0))
    with torch.no_grad():
        module.support[:] = torch.tensor([True, False, False])
        module.residual[1].fill_(10.)
    x = torch.ones(2, 4, dtype=torch.float64)
    target = module(x).detach().clone()
    target[:, 1] += 1
    objective = LoSparseTrainingCallback(module, lambda model, x, y: F.mse_loss(model(x), y))
    optimizer = torch.optim.Adam(module.parameters(), lr=.01)
    with objective:
        optimizer.zero_grad()
        objective(module, x, target).backward()
        assert module.residual.grad[1].abs().sum() > 0
        objective.before_step(module, optimizer)
        optimizer.step()
        objective.after_step(module, optimizer)
    assert module.support[1]


def test_actual_train_model_hooks_update_factors_and_finalize_compactly():
    from fedcore.experiments.runner import train_model
    from fedcore.experiments.protocol import TensorSplit
    module = prepare_losparse(seeded_linear(), LoSparseSpec(2, 2, 4))
    before = deepcopy(module.state_dict())
    x = inputs()
    split = TensorSplit(x, torch.ones(32, 5, dtype=torch.float64), tuple(f"train-{i}" for i in range(32)))
    callback = LoSparseTrainingCallback(module, lambda model, x, y: F.mse_loss(model(x), y))
    evidence = train_model(module, split, "regression", epochs=1, batch_size=8, learning_rate=.01,
                           batch_objective=callback, max_updates=4)
    assert int(module.completed_steps) == 4 and int(module.support.sum()) == 2
    assert not torch.equal(before["factors.left.weight"], module.factors.left.weight)
    assert not torch.equal(before["factors.right.weight"], module.factors.right.weight)
    assert not torch.equal(before["residual"], module.residual)
    assert evidence["recovery_objective"]["completed_steps"] == [4]
    compact = finalize_losparse(module)
    torch.testing.assert_close(compact(x), module(x))
    assert compact.residual_rows.shape == (2, 8)


def test_invalid_profile_refuses_before_source_or_rng_mutation():
    layer = seeded_linear()
    state = deepcopy(layer.state_dict())
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(SparseDomainError) as error:
        hassle_free(layer, inputs(), HASSLEFreeSpec(NMStructure(2, 4, 1), budget_bytes=1))
    assert error.value.code == "budget_infeasible"
    assert torch.equal(rng, torch.random.get_rng_state())
    for name, value in state.items():
        assert torch.equal(layer.state_dict()[name], value)
    with pytest.raises(SparseDomainError) as error:
        sparsegpt_nm(layer.weight, torch.zeros(8, 8), NMStructure())
    assert error.value.code == "singular_metric"


def test_callback_rejects_duplicate_and_unordered_transitions_without_ema_mutation():
    module = prepare_losparse(seeded_linear(), LoSparseSpec(1, 1, 2))
    callback = LoSparseTrainingCallback(module, lambda model, x, y: F.mse_loss(model(x), y))
    optimizer = torch.optim.Adam(module.parameters(), lr=.01)
    with callback:
        before = module.importance_ema.clone()
        with pytest.raises(SparseDomainError):
            callback.after_step(module, optimizer)
        assert torch.equal(before, module.importance_ema)
        callback(module, inputs(), torch.ones(32, 5, dtype=torch.float64)).backward()
        callback.before_step(module, optimizer)
        recorded = module.importance_ema.clone()
        with pytest.raises(SparseDomainError):
            callback.before_step(module, optimizer)
        assert torch.equal(recorded, module.importance_ema)
        optimizer.step()
        callback.after_step(module, optimizer)
        with pytest.raises(SparseDomainError):
            callback.after_step(module, optimizer)
        assert int(module.completed_steps) == 1
