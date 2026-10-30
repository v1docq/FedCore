from dataclasses import FrozenInstanceError
import pytest
import torch

from fedcore.algorithm.pruning.sparse.specs import (
    NMStructure, HASSLEFreeSpec, LoSparseSpec, SparseDomainError, plan_linear, reject_method,
)


@pytest.mark.parametrize("changes", [dict(n=0), dict(n=True), dict(m=0), dict(n=5),
    dict(rank=-1), dict(axis="out_features"), dict(tail_policy="pad"), dict(kernel="cuda_2_4")])
def test_nm_invalid_contracts(changes):
    with pytest.raises(SparseDomainError):
        NMStructure(**changes)


@pytest.mark.parametrize("changes", [dict(outer_steps=0), dict(inner_steps=True), dict(seed=-1), dict(seed=2**63),
    dict(learning_rate=0), dict(damping=float("nan")), dict(normalization="implicit"),
    dict(damping_rule="hidden"), dict(coordinate_scaling=1), dict(budget_bytes=True)])
def test_hassle_invalid_contracts(changes):
    with pytest.raises(SparseDomainError):
        HASSLEFreeSpec(**changes)


@pytest.mark.parametrize("changes", [dict(rank=True), dict(residual_rows=-1), dict(total_steps=0),
    dict(initial_scale=0), dict(final_step=11), dict(warmup_steps=10), dict(ema_beta=1),
    dict(delta_t=0), dict(allow_reentry="yes")])
def test_losparse_invalid_contracts(changes):
    arguments = dict(rank=1, residual_rows=2, total_steps=10)
    arguments.update(changes)
    with pytest.raises(SparseDomainError):
        LoSparseSpec(**arguments)


def test_immutable_plans_and_exact_declared_costs():
    spec = NMStructure(2, 4, 2)
    plan = plan_linear(12, 5, spec, dtype="float64", bias=True)
    # 30 sparse numbers + 34 factor numbers + 5 bias, 30 int64 indices.
    assert plan.numerical_elements == 69
    assert plan.metadata_bytes == 240
    assert plan.artifact_bytes == 69 * 8 + 240
    assert plan.orientation == "out_in"
    with pytest.raises(FrozenInstanceError):
        spec.rank = 3
    with pytest.raises(FrozenInstanceError):
        plan.rank = 3
    indexed = plan_linear(12, 5, LoSparseSpec(2, 3, 5), bias=False)
    assert indexed.numerical_elements == 34 + 36
    assert indexed.metadata_bytes == 24
    assert indexed.artifact_bytes == 70 * 4 + 24


@pytest.mark.parametrize("spec,kwargs,code", [
    (NMStructure(), dict(in_features=7, out_features=5), "incomplete_nm_group"),
    (NMStructure(rank=6), dict(in_features=8, out_features=5), "invalid_rank"),
    (LoSparseSpec(1, 6, 5), dict(in_features=8, out_features=5), "invalid_support"),
    (HASSLEFreeSpec(budget_bytes=1), dict(in_features=8, out_features=5), "budget_infeasible"),
    (HASSLEFreeSpec(max_input_features=4), dict(in_features=8, out_features=5), "unsupported_shape"),
    (NMStructure(), dict(in_features=8, out_features=5, dtype="float16"), "unsupported_device_dtype"),
    (NMStructure(), dict(in_features=8, out_features=5, device="cuda"), "unsupported_device_dtype"),
])
def test_preflight_refuses_without_rng_consumption(spec, kwargs, code):
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(SparseDomainError) as error:
        plan_linear(spec=spec, **kwargs)
    assert error.value.code == code
    assert torch.equal(rng, torch.random.get_rng_state())


def test_tedepr_is_independent_explicit_refusal():
    with pytest.raises(SparseDomainError) as error:
        reject_method("TEDEPR")
    assert error.value.code == "insufficient_specification"
    assert plan_linear(8, 5, NMStructure()).method == "packed_nm"


def test_rank_zero_and_empty_residual_have_named_physical_cost():
    plan = plan_linear(9, 4, LoSparseSpec(0, 0, 3), bias=False)
    assert plan.rank == 0 and plan.artifact_bytes == 0
