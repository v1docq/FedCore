"""Independent dense objectives for the supported structured profile algebra."""
import math

import pytest
import torch

from fedcore.algorithm.low_rank.approximation import NumericalDomainViolation,solve_weighted
from fedcore.algorithm.low_rank.plans import MetricPolicy
from fedcore.algorithm.low_rank.structured_profiles import (
    FactorPair,concatenate_factor_pairs,eora_factors,eora_author_gram_update,mixed_rank_metric,drone_factors,
    linear_least_squares_fit,svdllm_v1_fit,svdllm_v2_factors,inverse_log_allocation,
    basis_sharing_factors,groupreduce_factors,frequency_groups,groupreduce_transfer,
)


def matrix(shape,seed=1):
    return torch.randn(shape,generator=torch.Generator().manual_seed(seed),dtype=torch.float64)


@pytest.mark.parametrize("shape",[(3,5),(5,3),(4,4)])
@pytest.mark.parametrize("degenerate",[False,True])
def test_drone_matches_independent_rank_truncation_of_observed_outputs(shape,degenerate):
    w=matrix(shape,4); x=matrix((shape[1],13),8)
    if degenerate:
        w[-1]=w[0]; x[-1]=x[0]
    result=drone_factors(w,x,1)
    output=w@x
    u,s,vh=torch.linalg.svd(output,full_matrices=False)
    oracle=(u[:,:1]*s[:1])@vh[:1]
    torch.testing.assert_close(result.matrix()@x,oracle,atol=1e-11,rtol=1e-11)
    error=float((output-result.matrix()@x).square().sum())
    assert result.diagnostics["weighted_error_squared"]==pytest.approx(error,rel=1e-12)
    assert error==pytest.approx(float(s[1:].square().sum()),rel=1e-11)
    assert result.diagnostics["joint_shape"]==[result.diagnostics["weight_support"],result.diagnostics["input_support"]]
    assert result.parameter_elements==sum(shape)


def test_drone_zero_and_full_observed_support_are_well_defined():
    w=matrix((4,5)); x=matrix((5,11)); x[2:]=0
    full=drone_factors(w,x,4)
    torch.testing.assert_close(full.matrix()@x,w@x,atol=1e-11,rtol=1e-11)
    for weight,inputs in ((w,torch.zeros_like(x)),(torch.zeros_like(w),x)):
        zero=drone_factors(weight,inputs,2)
        assert zero.rank==2 and zero.diagnostics["actual_rank"]==0
        assert zero.diagnostics["weighted_error_squared"]==0
        assert torch.equal(zero.matrix(),torch.zeros_like(w))


@pytest.mark.parametrize("rank",[1,2,3])
def test_v2_rotated_singular_metric_matches_quadratic_oracle(rank):
    w=matrix((3,4),11)
    q,_=torch.linalg.qr(matrix((4,4),12))
    c=q@torch.diag(torch.tensor([0.,.4,2.,9.],dtype=torch.float64))@q.T
    result=svdllm_v2_factors(w,c,rank)
    oracle=solve_weighted(w,c,rank)
    torch.testing.assert_close(result.matrix(),oracle.approximation,atol=1e-11,rtol=1e-11)
    difference=w-result.matrix()
    error=float(torch.trace(difference@c@difference.T))
    assert result.diagnostics["weighted_error_squared"]==pytest.approx(error,abs=1e-12)
    assert result.diagnostics["formula"].endswith("inverse_square_root")


def test_v2_inverse_root_and_right_singular_vectors_are_required():
    w=torch.tensor([[3.,1.],[2.,4.]],dtype=torch.float64)
    c=torch.diag(torch.tensor([1.,9.],dtype=torch.float64))
    result=svdllm_v2_factors(w,c,1)
    u,s,vh=torch.linalg.svd(w@c.sqrt())
    expected=(u[:,:1]*s[:1])@vh[:1]@torch.diag(torch.tensor([1.,1/3],dtype=torch.float64))
    torch.testing.assert_close(result.matrix(),expected)
    wrong=(u[:,:1]*s[:1])@vh[:1]@torch.diag(torch.tensor([1.,1/9],dtype=torch.float64))
    assert not torch.allclose(result.matrix(),wrong)


def test_v2_nullspace_preservation_can_refuse_rank_overflow():
    w=torch.diag(torch.tensor([4.,3.,2.],dtype=torch.float64))
    c=torch.diag(torch.tensor([1.,1.,0.],dtype=torch.float64))
    with pytest.raises(NumericalDomainViolation,match="null-space"):
        svdllm_v2_factors(w,c,1,MetricPolicy(nullspace_policy="preserve_nullspace"))


@pytest.mark.parametrize("ridge",[0.,.3])
@pytest.mark.parametrize("deficient",[False,True])
def test_shared_ls_matches_independent_augmented_lstsq(ridge,deficient):
    h=matrix((14,4),14); y=matrix((14,3),15)
    if deficient:
        h[:,3]=h[:,0]+h[:,1]
    original=h.clone()
    result=linear_least_squares_fit(h,y,ridge=ridge)
    design=torch.cat((h,math.sqrt(ridge)*torch.eye(4,dtype=h.dtype))) if ridge else h
    target=torch.cat((y,torch.zeros(4,3,dtype=y.dtype))) if ridge else y
    oracle=torch.linalg.lstsq(design,target,driver="gelsd").solution.T
    torch.testing.assert_close(result.coefficients,oracle,atol=1e-11,rtol=1e-11)
    assert result.residual_squared==pytest.approx(float((h@oracle.T-y).square().sum()),rel=1e-12)
    assert torch.equal(h,original)


def test_v1_uses_current_input_target_and_preserves_fixed_right():
    w=matrix((4,5),20); b=matrix((2,5),21)
    current=matrix((17,5),22); original=current+2
    saved=b.clone()
    pair=svdllm_v1_fit(w,b,current)
    expected=torch.linalg.lstsq(current@b.T,current@w.T,driver="gelsd").solution.T
    torch.testing.assert_close(pair.left,expected,atol=1e-11,rtol=1e-11)
    old_target=torch.linalg.lstsq(current@b.T,original@w.T,driver="gelsd").solution.T
    assert not torch.allclose(pair.left,old_target)
    assert torch.equal(pair.right,b) and torch.equal(b,saved)
    initial=matrix((4,2),23)
    assert ((current@pair.matrix().T-current@w.T).square().sum()
            <=(current@(initial@b).T-current@w.T).square().sum()+1e-10)


def test_v1_promotes_represented_fp32_inputs_before_cancelling_products():
    w=torch.tensor([[1e8,1e8]],dtype=torch.float32)
    x=torch.tensor([[1.0000001192092896,-1.]],dtype=torch.float32)
    b=torch.tensor([[1.,0.]],dtype=torch.float32)
    result=svdllm_v1_fit(w,b,x)
    target=x.double()@w.double().T
    actual=x.double()@result.matrix().double().T
    torch.testing.assert_close(actual,target,atol=1e-6,rtol=1e-7)
    assert result.left.item()==pytest.approx(11.920928,abs=2e-6)
    assert result.diagnostics["residual_squared"]==pytest.approx(float((actual-target).square().sum()),abs=1e-20)


def test_explicit_ls_support_cutoff_is_not_claimed_to_be_unrestricted_exact_ls():
    h=torch.diag(torch.tensor([1.,1e-8],dtype=torch.float64))
    y=torch.tensor([[0.],[1.]],dtype=torch.float64)
    result=linear_least_squares_fit(h,y,rcond=1e-4)
    assert result.objective=="linear_least_squares_restricted_support"
    assert result.design_rank==1 and result.residual_squared==1
    unrestricted=linear_least_squares_fit(h,y)
    assert unrestricted.residual_squared==pytest.approx(0.,abs=1e-24)


def test_eora_is_exactly_a_weighted_residual_and_does_not_modify_base():
    w=matrix((4,5),24); base=matrix((4,5),25); before=base.clone()
    x=matrix((13,5),26); c=x.T@x/len(x)
    result=eora_factors(w,base,c,2)
    oracle=solve_weighted(w-base,c,2)
    torch.testing.assert_close(result.matrix(),oracle.approximation)
    assert torch.equal(base,before)
    assert result.diagnostics["recovery"]=="analytic_no_gradient_updates"


def test_released_author_eora_gram_is_order_dependent_not_an_empirical_moment():
    zero=torch.zeros(2,2,dtype=torch.float64)
    first=torch.tensor([[1.,0.]],dtype=torch.float64)
    second=torch.tensor([[0.,3.]],dtype=torch.float64)
    a,_=eora_author_gram_update(zero,first,calibration_samples=2)
    a,metadata=eora_author_gram_update(a,second,calibration_samples=2)
    b,_=eora_author_gram_update(zero,second,calibration_samples=2)
    b,_=eora_author_gram_update(b,first,calibration_samples=2)
    torch.testing.assert_close(a,torch.diag(torch.tensor([1/3,4.5],dtype=torch.float64)))
    torch.testing.assert_close(b,torch.diag(torch.tensor([.5,3.],dtype=torch.float64)))
    empirical=(first.T@first+second.T@second)/2
    assert not torch.allclose(a,b) and not torch.allclose(a,empirical)
    assert metadata["order_dependent"] and not metadata["mergeable"]
    assert torch.equal(zero,torch.zeros_like(zero))


def test_exact_residual_concatenation_keeps_both_ranks_visible():
    a=FactorPair(matrix((3,2),1),matrix((2,4),2))
    b=FactorPair(matrix((3,1),3),matrix((1,4),4))
    merged=concatenate_factor_pairs(a,b)
    torch.testing.assert_close(merged.matrix(),a.matrix()+b.matrix())
    assert merged.rank==3 and merged.parameter_elements==21
    assert merged.diagnostics["bias_count"]==1


def test_normalized_output_metric_distinguishes_opposite_inputs_from_mean_heuristic():
    w=torch.eye(2,dtype=torch.float64)
    x=torch.tensor([[1.,0.],[-1.,0.],[0.,2.],[0.,-2.]],dtype=torch.float64)
    metric,diagnostics=mixed_rank_metric(w,x,objective="normalized_output")
    torch.testing.assert_close(metric,torch.eye(2,dtype=torch.float64)/2)
    assert diagnostics["retained_observations"]==4
    with pytest.raises(NumericalDomainViolation,match="zero denominator"):
        mixed_rank_metric(w,x,objective="author_mean_input")
    with_zero=torch.cat((x,torch.zeros(1,2,dtype=x.dtype)))
    skipped,diag=mixed_rank_metric(w,with_zero,objective="normalized_output",zero_policy="skip")
    torch.testing.assert_close(skipped,metric)
    assert diag["observations"]==5 and diag["retained_observations"]==4


@pytest.mark.parametrize("errors",[(0.,2.),(1.,2.),(.5,2.),(-1.,2.),(float("inf"),2.)])
def test_v2_inverse_log_refuses_undefined_or_signed_domain(errors):
    with pytest.raises(NumericalDomainViolation):
        inverse_log_allocation(errors,.4)


def test_v2_inverse_log_scores_are_explicit_and_not_a_cost_certificate():
    ratios,diag=inverse_log_allocation((math.exp(2),math.exp(4),math.exp(8)),.4)
    assert ratios==pytest.approx((.685714285714,.342857142857,.171428571429))
    assert sum(ratios)==pytest.approx(1.2)
    assert diag["integer_budget_verified"] is False
    with pytest.raises(NumericalDomainViolation,match="removal domain"):
        inverse_log_allocation((1.0001,100.),.8)


def test_proportional_basis_sharing_matches_sum_of_individual_output_errors():
    weights=(matrix((3,4),30),matrix((2,4),31))
    x=matrix((19,4),32); c=x.T@x/len(x); scales=(.2,4.)
    result=basis_sharing_factors(weights,c,2,metric_scales=scales)
    errors=sum(scale*float((x@(w-left@result.right).T).square().sum())/len(x)
               for scale,w,left in zip(scales,weights,result.lefts))
    assert errors==pytest.approx(result.diagnostics["weighted_error_squared"],rel=1e-11)
    stacked=torch.cat([math.sqrt(scale)*w for scale,w in zip(scales,weights)])
    u,s,vh=torch.linalg.svd(stacked@torch.linalg.cholesky(c),full_matrices=False)
    assert errors==pytest.approx(float(s[2:].square().sum()),rel=1e-11)
    assert result.parameter_elements==2*(4+3+2)
    with pytest.raises(NumericalDomainViolation,match="input axes"):
        basis_sharing_factors((weights[0],matrix((2,5))),c,2)


def dense_group_table(state):
    result=state.factors[0].right.new_zeros((len(state.token_to_group),state.factors[0].right.shape[1]))
    for tokens,pair in zip(state.group_tokens,state.factors):
        result[tokens]=pair.matrix()
    return result


def test_groupreduce_weighted_rows_permutation_zero_frequency_and_empty_group():
    w=matrix((7,3),40); q=torch.tensor([9.,0.,2.,1.,7.,4.,8.],dtype=w.dtype)
    ids=torch.tensor([1,0,1,0,1,0,1])
    result=groupreduce_factors(w,q,ids,(1,2,0))
    dense=dense_group_table(result)
    objective=float((q[:,None]*(w-dense).square()).sum())
    assert result.diagnostics["weighted_error_squared"]==pytest.approx(objective,rel=1e-12)
    for group,tokens in enumerate(result.group_tokens[:2]):
        singular=torch.linalg.svdvals(q[tokens,None].sqrt()*w[tokens])
        assert result.diagnostics["group_weighted_errors_squared"][group]==pytest.approx(float(singular[result.factors[group].rank:].square().sum()),abs=1e-12)
    assert result.map_bytes==7*2*8
    assert result.parameter_elements==(3+3)*1+(4+3)*2
    with pytest.raises(NumericalDomainViolation,match="zero frequency"):
        groupreduce_factors(w,q,ids,(1,2),zero_policy="error")


def test_frequency_group_ties_and_transfers_are_deterministic_and_budgeted():
    q=torch.ones(6,dtype=torch.float64)
    assert frequency_groups(q,2).tolist()==[0,0,0,1,1,1]
    w=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.],[0.,1.],[1.,0.]],dtype=q.dtype)
    initial=groupreduce_factors(w,q,torch.tensor([0,0,0,1,1,1]),(1,1))
    a=groupreduce_transfer(w,q,initial,max_transfers=2,parameter_budget=10)
    b=groupreduce_transfer(w,q,initial,max_transfers=2,parameter_budget=10)
    assert torch.equal(a.token_to_group,b.token_to_group)
    assert a.diagnostics["transfers"]==b.diagnostics["transfers"]
    assert len(a.token_to_group)==6 and sum(a.diagnostics["group_sizes"])==6
    assert a.diagnostics["weighted_error_squared"]<=initial.diagnostics["weighted_error_squared"]
    assert a.parameter_elements<=10
    assert all(move["parameter_elements"]<=10 for move in a.diagnostics["transfers"])
