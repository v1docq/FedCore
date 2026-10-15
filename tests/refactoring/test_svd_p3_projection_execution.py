"""Real small CPU P3 execution, recovery, refusal and representation checks."""
from copy import deepcopy
from dataclasses import replace
import io
import random
import numpy as np
import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.projection_specs import ESPACE,PELA,LASER,CTRAFM,CTRMLPAFM
from fedcore.algorithm.low_rank.projection_profiles import (
    espace_statistics,pela_rank,laser_rank,solve_ctr_field)
from fedcore.algorithm.low_rank.projection_execution import (
    ProjectionProfileError,collect_espace_backward,transform_projection,transform_ctr_mlp,
    DiagnoseIntervention,diagnose_laser,export_laser_compact)
from fedcore.algorithm.low_rank.projection_layers import (
    FixedProjectorLinear,CenteredPCAEmbedding,CTRFieldGraph,CTRLinearSum,ReLUFactorizedLinear)
from fedcore.algorithm.low_rank.projection_recovery import PELARecoveryObjective
from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear
from fedcore.algorithm.low_rank.statistics import StatisticsPassport
from fedcore.algorithm.low_rank.method_errors import MethodSpecError
from fedcore.algorithm.low_rank.execution import WeightedProfileError


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def stamp():
    return StatisticsPassport("checkpoint","graph","previous","input","calibration")


def rows(width=6,n=7):
    return torch.randn(n,width,dtype=torch.float64,generator=torch.Generator().manual_seed(13))+2


def model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        return nn.Sequential(nn.Linear(6,8,dtype=torch.float64),nn.Tanh(),nn.Linear(8,4,dtype=torch.float64)).eval()


def same_state(source,state):
    for name,value in source.state_dict().items():
        assert torch.equal(value,state[name])


def train_executor(student,stage,updates,objective):
    data = rows()
    target = torch.zeros(len(data),4,dtype=torch.float64)
    optimizer = torch.optim.Adam((p for p in student.parameters() if p.requires_grad),lr=.001)
    with objective:
        for _ in range(updates):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(student,data,target)
            loss.backward()
            optimizer.step()
    return {"updates":updates,"stage":stage,"loss":float(loss.detach())}


@pytest.mark.parametrize("criterion",["mse","nmse","go_mse_bound","go_normalized"])
def test_espace_public_compact_matches_original_weight_projector_with_single_bias(criterion):
    source = model()
    data = rows()
    snapshot = deepcopy(source.state_dict())
    torch_rng = torch.random.get_rng_state().clone()
    result = transform_projection(source,data,ESPACE(criterion),rank=2,target_paths=("0",))
    assert isinstance(result.model[0],FactorizedLinear)
    from fedcore.algorithm.low_rank.projection_profiles import select_espace_projector
    p = select_espace_projector(espace_statistics(data,stamp(),ESPACE(criterion),weight=source[0].weight),2).basis
    projected = data@p@p.T
    expected = source[2](source[1](source[0](projected)))
    torch.testing.assert_close(result.model(data),expected,rtol=1e-10,atol=1e-10)
    same_state(source,snapshot)
    assert torch.equal(torch_rng,torch.random.get_rng_state())
    assert result.model._fedcore_method_evidence == result.evidence
    assert result.model._fedcore_transform_plan is result.plan


def test_fixed_projector_full_weight_training_compiles_after_updates_without_moving_P():
    source = model()
    data = rows()
    captured = {}
    def recover(student,stage,updates,objective):
        assert isinstance(student[0],FixedProjectorLinear)
        assert student[0].weight.shape == source[0].weight.shape
        p = student[0].projection.clone()
        before = student[0].weight.clone()
        answer = train_executor(student,stage,updates,objective)
        assert torch.equal(p,student[0].projection)
        assert not torch.equal(before,student[0].weight)
        captured["prediction"] = student(data).detach()
        return answer
    result = transform_projection(source,data,ESPACE(updates=2),rank=2,target_paths=("0",),recovery_executor=recover)
    torch.testing.assert_close(result.model(data),captured["prediction"],rtol=1e-10,atol=1e-10)
    assert result.evidence["recovery_stages"][0]["objective"]["resource_reservation"]["adam_moments_reserve_bytes"] > 0


@pytest.mark.parametrize("criterion",["nl_first_order_bound","nl_normalized"])
def test_bounded_backward_matches_individual_manual_gradients_and_restores_source_rng(criterion):
    source = model()
    source[0].weight.grad = torch.ones_like(source[0].weight)
    old_grad = source[0].weight.grad
    data = rows(n=5)
    labels = torch.zeros(5,4,dtype=torch.float64)
    spec = ESPACE(criterion,max_examples=3,max_batches=2)
    state = deepcopy(source.state_dict())
    python_rng,numpy_rng,torch_rng = random.getstate(),np.random.get_state(),torch.random.get_rng_state().clone()
    consumed = []
    def batches():
        for start in range(0,5,2):
            consumed.append(start)
            yield data[start:start+2],labels[start:start+2]
    actual = collect_espace_backward(source,("0",),batches(),stamp(),spec)["0"]
    gradients = []
    for x,y in zip(data[:3],labels[:3]):
        private = x.clone().requires_grad_(True)
        loss = (source(private)-y).square().mean()
        gradients.append(torch.autograd.grad(loss,private)[0].detach())
    expected = espace_statistics(data[:3],stamp(),spec,gradients=torch.stack(gradients))
    torch.testing.assert_close(actual.matrix,expected.matrix,rtol=1e-11,atol=1e-11)
    assert consumed == [0,2]
    assert actual.collection_manifest["backward_passes"] == 3
    same_state(source,state)
    assert source[0].weight.grad is old_grad
    assert random.getstate() == python_rng
    assert np.array_equal(np.random.get_state()[1],numpy_rng[1])
    assert torch.equal(torch_rng,torch.random.get_rng_state())


def test_backward_failure_preserves_mixed_modes_parameters_grad_and_rng():
    source = model()
    source[0].train()
    state = deepcopy(source.state_dict())
    rng = torch.random.get_rng_state().clone()
    source[0].weight.grad = torch.ones_like(source[0].weight)
    gradient = source[0].weight.grad
    with pytest.raises(Exception):
        collect_espace_backward(source,("0",),[(rows(n=1),torch.zeros(1,99))],stamp(),ESPACE("nl_first_order_bound"))
    assert source[0].training and not source.training
    assert source[0].weight.grad is gradient
    assert not source[0]._forward_pre_hooks
    assert torch.equal(rng,torch.random.get_rng_state())
    same_state(source,state)


def test_pela_recovery_uses_private_frozen_teacher_validated_features_and_exact_budget():
    source = model()
    original = deepcopy(source.state_dict())
    spec = PELA(kappa=1.2,block_map=(("1","1"),),updates=2,epsilon=10.)
    result = transform_projection(source,rows(),spec,target_paths=("0",),recovery_executor=train_executor)
    assert isinstance(result.model[0],FactorizedLinear)
    same_state(source,original)
    evidence = result.evidence["recovery_stages"][0]["objective"]
    assert evidence["objective_calls"] == 2 and evidence["teacher"] == "private_frozen_eval"
    assert evidence["regularizer"] == "published_subtract_epsilon"
    assert evidence["feature_normalization"] == "paper_sum_squared"
    assert not result.model[2].weight.requires_grad
    assert all(not layer._forward_hooks for layer in result.model.modules())
    assert pela_rank((8,6),1.2) == 2


def test_pela_block_shape_mismatch_is_rejected_before_training_and_hooks_removed():
    source = model()
    work = deepcopy(source)
    from fedcore.algorithm.low_rank.structured_profiles import FactorPair
    work[0] = FactorizedLinear.from_factors(FactorPair(torch.ones(8,1,dtype=torch.float64),torch.ones(1,6,dtype=torch.float64)))
    spec = PELA(block_map=(("0","2"),))
    objective = PELARecoveryObjective(work,source,spec,("0",))
    with pytest.raises(Exception,match="equal tensor shapes"):
        objective.validate_features(rows(n=1))
    assert not objective.handles
    assert all(not layer._forward_hooks for layer in work.modules())
    assert all(not p.requires_grad for p in objective.teacher.parameters())


def test_pela_refuses_inexact_update_executor_and_invalid_formula_rank():
    source = model()
    with pytest.raises(ProjectionProfileError,match="exact approved"):
        transform_projection(source,rows(),PELA(block_map=(("1","1"),),updates=2),target_paths=("0",),
                             recovery_executor=lambda *args:{"updates":1})
    with pytest.raises(Exception,match="invalid rank"):
        pela_rank((2,2),100)


def test_pela_teacher_memory_refusal_precedes_private_teacher_allocation_and_forward(monkeypatch):
    import psutil
    import fedcore.algorithm.low_rank.projection_execution as execution
    source = model()
    teacher = model()
    teacher.register_buffer("large_teacher_state",torch.zeros(128*1024,dtype=torch.float64))
    class Process:
        def memory_info(self):
            return type("Memory",(),{"rss":128*1024**2})()
    monkeypatch.setattr(psutil,"Process",lambda *args:Process())
    def forbidden(*args,**kwargs):
        raise AssertionError("Teacher allocation and validation forward must follow the named-memory guard")
    monkeypatch.setattr(execution,"PELARecoveryObjective",forbidden)
    with pytest.raises(ProjectionProfileError,match="teacher copy/snapshot/recovery preflight"):
        transform_projection(source,rows(),PELA(block_map=(("1","1"),)),target_paths=("0",),
            base_model=teacher,recovery_executor=train_executor,max_workspace_bytes=64*1024,
            max_peak_bytes=192*1024**2+128*1024)


def test_laser_candidates_are_independent_dense_and_repeatable_then_compact_selected():
    source = model()
    state = deepcopy(source.state_dict())
    data = rows()
    requests = tuple(DiagnoseIntervention(path,1,"checkpoint","validation","mse") for path in ("0","2","0"))
    calls = []
    def evaluator(candidate):
        calls.append(deepcopy(candidate.state_dict()))
        return float(candidate(data).square().mean())
    diagnosis = diagnose_laser(source,requests,evaluator)
    assert diagnosis.candidates[0] == diagnosis.candidates[2]
    assert all(torch.equal(calls[0][key],calls[2][key]) for key in calls[0])
    assert torch.equal(calls[1]["0.weight"],state["0.weight"])
    assert torch.equal(calls[0]["2.weight"],state["2.weight"])
    assert all(item.representation == "dense" for item in diagnosis.candidates)
    result = transform_projection(source,data,LASER(.2),target_paths=("0",))
    assert type(result.model[0]) is nn.Linear
    assert result.evidence["tensor_bytes_before"] == result.evidence["tensor_bytes_after"]
    compact,evidence = export_laser_compact(result.model,"0",1)
    torch.testing.assert_close(compact(data),result.model(data),rtol=1e-10,atol=1e-10)
    assert evidence["tensor_bytes_compact"] < evidence["tensor_bytes_dense"]
    same_state(source,state)
    assert laser_rank((4,10),.5) == 2


def test_laser_test_role_mutating_evaluator_and_uneconomical_export_refuse():
    source = model()
    request = DiagnoseIntervention("0",1,"checkpoint","validation","mse")
    with pytest.raises(ProjectionProfileError,match="validation"):
        diagnose_laser(source,[request],lambda candidate:0.,data_role="test")
    def corrupt(candidate):
        with torch.no_grad():
            candidate[2].weight.zero_()
        return 0.
    with pytest.raises(ProjectionProfileError,match="without mutating"):
        diagnose_laser(source,[request],corrupt)
    with pytest.raises(ProjectionProfileError,match="economical"):
        export_laser_compact(source,"0",5)


def ctr_graph():
    tables = {"a":nn.Embedding(9,3,dtype=torch.float64),"b":nn.Embedding(7,3,dtype=torch.float64)}
    with torch.no_grad():
        tables["a"].weight.copy_(rows(3,9))
        tables["b"].weight.copy_(rows(3,7)+3)
    return CTRFieldGraph(tables,CTRLinearSum({"a":3,"b":3},2,dtype=torch.float64),(("a","b"),)).eval()


def test_ctr_frequency_pca_block_sum_decoded_dot_lookup_and_state_roundtrip():
    source = ctr_graph()
    frequencies = {"a":torch.arange(1,10,dtype=torch.float64),"b":torch.arange(7,0,-1,dtype=torch.float64)}
    spec = CTRAFM((("a",1),("b",2)),weighting="frequency")
    result = transform_projection(source,None,spec,field_frequencies=frequencies)
    tokens = torch.tensor([[0,2],[3,1],[8,6]])
    projected = []
    for name in ("a","b"):
        field = solve_ctr_field(source.fields[name].weight,dict(spec.field_ranks)[name],stamp(),frequencies=frequencies[name],weighting="frequency")
        projected.append(field.projected_table()[tokens[:,0 if name == "a" else 1]])
    expected_linear = source.linear.blocks["a"](projected[0])+source.linear.blocks["b"](projected[1])+source.linear.bias
    actual = result.model(tokens)
    torch.testing.assert_close(actual["linear"],expected_linear,rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(actual["interactions"].squeeze(-1),(projected[0]*projected[1]).sum(-1),rtol=1e-10,atol=1e-10)
    assert result.model.fields["a"](tokens[:,0]).shape == (3,1)
    assert result.evidence["buffer_bytes_after"] > 0
    stream = io.BytesIO()
    torch.save(result.model.state_dict(),stream)
    stream.seek(0)
    restored = deepcopy(result.model)
    restored.load_state_dict(torch.load(stream,weights_only=True))
    for name in actual:
        torch.testing.assert_close(restored(tokens)[name],actual[name])


def test_ctr_interaction_refusal_all_consumers_and_tensor_byte_buffer_budget():
    source = ctr_graph()
    snapshot = deepcopy(source.state_dict())
    with pytest.raises(ProjectionProfileError,match="Interaction"):
        transform_projection(source,None,CTRAFM((("a",1),("b",1)),interaction_policy="refuse"))
    result = transform_projection(source,None,CTRAFM((("a",1),("b",1))))
    with pytest.raises(ProjectionProfileError,match="tensor bytes"):
        transform_projection(source,None,CTRAFM((("a",1),("b",1))),tensor_byte_budget=result.evidence["parameter_bytes_after"])
    source.extra = nn.Identity()
    with pytest.raises(Exception,match="complete explicit"):
        transform_projection(source,None,CTRAFM((("a",1),("b",1))))
    del source.extra
    same_state(source,snapshot)


def test_ctr_mlp_linear_affine_mean_correction_and_separate_trained_relu_variant():
    source = model()
    data = rows()
    result = transform_ctr_mlp(source,data,target_paths=("0",),rank=2)
    output = source[0](data)
    mean = output.mean(0)
    _,vectors = torch.linalg.eigh((output-mean).T@(output-mean)/len(output))
    p = vectors[:,-2:]
    expected_first = (output-mean)@p@p.T+mean
    torch.testing.assert_close(result.model[0](data),expected_first,rtol=1e-10,atol=1e-10)
    nonlinear = transform_ctr_mlp(source,data,target_paths=("0",),rank=2,
        nonlinear_variant="relu_between_factors_v1",updates=2,recovery_executor=train_executor)
    assert type(nonlinear.model[0]) is ReLUFactorizedLinear
    assert nonlinear.evidence["layers"][0]["linearly_equivalent_to_affine_pca"] is False
    assert nonlinear.evidence["recovery_stages"][0]["requested_updates"] == 2
    with pytest.raises(MethodSpecError,match="requires recovery"):
        CTRMLPAFM("relu_between_factors_v1")


def test_recovery_missing_callback_noncalibration_and_insufficient_workspace_refuse_before_source_forward():
    source = model()
    calls = []
    handle = source.register_forward_hook(lambda *args:calls.append(True))
    try:
        with pytest.raises(ProjectionProfileError,match="existing training"):
            transform_projection(source,rows(),ESPACE(updates=1),rank=1,target_paths=("0",))
        with pytest.raises(ProjectionProfileError,match="validation/test"):
            transform_projection(source,rows(),ESPACE(),rank=1,target_paths=("0",),data_role="test")
        with pytest.raises(WeightedProfileError,match="ResourceLimitExceeded"):
            transform_projection(source,rows(),ESPACE(),rank=1,target_paths=("0",),max_workspace_bytes=16)
        assert not calls
    finally:
        handle.remove()
