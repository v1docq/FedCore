"""Pure rank-policy evidence: complete observations, actual cost and role safety."""
from dataclasses import replace
import json
import math

import pytest

from fedcore.algorithm.low_rank.allocation import StorageCost,BudgetInfeasible,exact_unique_cost
from fedcore.experiments.svd_rank_policies import (
    RankOperatorCost,IsolatedRankEvaluation,RankSelectionPlan,rank_candidate_schedule,
    select_isolated_ranks,rank_search_space,drone_time_allowances,select_drone_step,select_v2_ranks,
)


def observations(operators,grids,*,score_name="validation_kl",graph_version="graph0",statistics_id="stats0"):
    return tuple(IsolatedRankEvaluation(op.path,rank,1/rank,
                 exact_unique_cost(op.storages(rank)),"trained0",score_name,"minimize","validation",
                 "checkpoint0",graph_version,statistics_id) for op in operators for rank in grids[op.path])


def test_isolated_schedules_share_prospective_baseline_and_use_explicit_per_path_ranks():
    operators=(RankOperatorCost("one",(3,4)),RankOperatorCost("two",(4,4)))
    grid={"one":(1,2),"two":(2,3)}
    schedule,evidence=rank_candidate_schedule("asvd",operators,grid,
                                             method_options={"alpha":.5},baseline_id="trained0",checkpoint_id="checkpoint0")
    assert len(schedule)==4
    assert len({candidate.candidate_id for candidate in schedule})==4
    assert all(candidate.method=="asvd" for candidate in schedule)
    assert all(len(candidate.parameters["ranks"])==1 for candidate in schedule)
    assert evidence["baseline_id"]=="trained0"
    assert evidence["statistics_role"]=="calibration" and evidence["score_role"]=="validation"
    json.dumps([c.to_dict() for c in schedule]+[evidence],allow_nan=False)


@pytest.mark.parametrize("family",["asvd","afm","bolaco"])
def test_actual_integer_budget_includes_affine_bias_maps_and_fixed_model(family):
    map_storage=StorageCost("shared_vocab_map",6,8,"map")
    operators=(RankOperatorCost("one",(3,4),3,4,(map_storage,)),
               RankOperatorCost("two",(2,4),2,4,(map_storage,)))
    grid={"one":(1,2),"two":(1,2)}
    fixed=(StorageCost("untouched",5),)
    data=observations(operators,grid)
    # rank1/rank1: 7+6+3+2+shared-map6+fixed5=29; 2/1 costs36.
    plan=select_isolated_ranks(family,operators,grid,data,36,fixed_storages=fixed)
    assert isinstance(plan,RankSelectionPlan)
    assert plan.allocation.total_cost<=36
    ranks=plan.candidate.parameters["ranks"]
    actual=exact_unique_cost((*fixed,*(s for op in operators for s in op.storages(ranks[op.path]))))
    assert actual==plan.allocation.total_cost
    assert plan.evidence["score_semantics"]=="additive_surrogate_of_isolated_observations"
    assert plan.evidence["joint_evaluation_required"] and not plan.evidence["joint_quality_observed"]
    json.dumps(plan.to_dict(),allow_nan=False)
    impossible=select_isolated_ranks(family,operators,grid,data,28,fixed_storages=fixed)
    assert isinstance(impossible,BudgetInfeasible) and impossible.minimum_cost==29


def test_isolated_tables_require_every_rank_same_checkpoint_and_true_complete_cost():
    operators=(RankOperatorCost("one",(3,4),3),)
    grid={"one":(1,2)}; data=observations(operators,grid)
    for bad in (data[:-1],data+data[:1]):
        with pytest.raises(ValueError,match="complete declared"):
            select_isolated_ranks("afm",operators,grid,bad,100)
    for field,value in (("baseline_id","other"),("checkpoint_id","other"),("graph_version","other")):
        with pytest.raises(ValueError,match="cannot mix"):
            select_isolated_ranks("afm",operators,grid,(data[0],replace(data[1],**{field:value})),100)
    with pytest.raises(ValueError,match="complete factors"):
        select_isolated_ranks("afm",operators,grid,(replace(data[0],cost=7),data[1]),100)
    with pytest.raises(ValueError,match="validation"):
        replace(data[0],data_role="test")
    with pytest.raises(ValueError,match="calibration-only"):
        replace(data[0],statistics_role="test")
    with pytest.raises(ValueError,match="same calibration"):
        select_isolated_ranks("afm",operators,grid,(data[0],replace(data[1],statistics_id="different")),100)
    with pytest.raises(ValueError,match="calibration-only"):
        replace(data[0],statistics_id="")


def test_joint_bolaco_domain_is_budgeted_and_is_not_a_new_search_runner():
    operators=(RankOperatorCost("one",(3,4)),RankOperatorCost("two",(2,4)))
    grid={"one":(1,2),"two":(1,2)}
    space,evidence=rank_search_space("bolaco",operators,grid,budget=20)
    assert len(space)==3  # 1/1=13, 1/2=19, 2/1=20; 2/2=26 exceeds.
    for candidate in space:
        ranks=candidate.parameters["ranks"]
        assert exact_unique_cost(s for op in operators for s in op.storages(ranks[op.path]))<=20
    assert evidence["search_executor"].endswith("run_search")
    assert evidence["joint_quality_observed"] is False
    with pytest.raises(ValueError,match="enumeration limit"):
        rank_search_space("bolaco",operators,grid,budget=20,max_combinations=3)
    # A one-pass fixed-storage iterable is captured once, never silently erased
    # after the first Cartesian candidate.
    fixed=(StorageCost("rest",4),)
    a,_=rank_search_space("bolaco",operators,grid,budget=20,fixed_storages=fixed)
    b,_=rank_search_space("bolaco",operators,grid,budget=20,fixed_storages=iter(fixed))
    assert [c.candidate_id for c in a]==[c.candidate_id for c in b]


def test_drone_time_product_identity_does_not_claim_error_independence():
    allowances=drone_time_allowances({"one":1.,"two":3.},.5)
    assert allowances[1].loss_growth>allowances[0].loss_growth
    assert math.prod(1+a.loss_growth for a in allowances)==pytest.approx(1.5)
    # Measurement units cancel; extreme finite common scale must not overflow.
    huge=drone_time_allowances({"one":1e308,"two":1e308},.5)
    assert sum(a.time_fraction for a in huge)==pytest.approx(1.)
    for times in ({"one":0.},{"one":float("inf")}):
        with pytest.raises(ValueError):
            drone_time_allowances(times,.5)


def test_drone_first_accepted_grid_rank_uses_current_graph_and_actual_remaining_cost():
    op=RankOperatorCost("one",(3,4),3)
    grid=(1,2,3); data=observations((op,),{"one":grid},score_name="validation_loss",graph_version="after_step1")
    data=tuple(replace(e,score=score) for e,score in zip(data,(1.4,1.05,1.01)))
    allowance=drone_time_allowances({"one":1.},.1)[0]
    kwargs=dict(reference_loss=1.,baseline_id="trained0",checkpoint_id="checkpoint0",
                graph_version="after_step1",statistics_id="stats0",budget=25)
    selected=select_drone_step(op,grid,data,allowance,**kwargs)
    assert selected.candidate.parameters["ranks"]["one"]==2
    assert selected.evidence["observed_candidate_loss"]==1.05
    assert selected.evidence["product_identity_is_loss_guarantee"] is False
    with pytest.raises(ValueError,match="stale"):
        select_drone_step(op,grid,data,allowance,**{**kwargs,"graph_version":"before_step1"})
    with pytest.raises(ValueError,match="stale"):
        select_drone_step(op,grid,data,allowance,**{**kwargs,"statistics_id":"old_stats"})
    assert isinstance(select_drone_step(op,grid,data,allowance,**{**kwargs,"budget":16}),BudgetInfeasible)
    with pytest.raises(ValueError,match="positive"):
        select_drone_step(op,grid,data,allowance,**{**kwargs,"reference_loss":0.})


def test_v2_domain_projection_respects_different_shapes_and_declared_integer_budget():
    operators=(RankOperatorCost("q0",(3,4),3),RankOperatorCost("q1",(4,5),4))
    errors={"q0":math.exp(2),"q1":math.exp(4)}
    plan=select_v2_ranks(operators,errors,{"Query":("q0","q1")},.4,30)
    assert isinstance(plan,RankSelectionPlan)
    ranks=plan.candidate.parameters["ranks"]
    actual=exact_unique_cost(s for op in operators for s in op.storages(ranks[op.path]))
    assert actual==plan.allocation.total_cost<=30
    assert plan.evidence["published_continuous_formula"] is True
    assert plan.evidence["published_integer_projection"] is False
    assert plan.evidence["joint_evaluation_required"]
    json.dumps(plan.to_dict(),allow_nan=False)
    with pytest.raises(ValueError,match="ell>1"):
        select_v2_ranks(operators,{"q0":0.,"q1":2.},{"Query":("q0","q1")},.4,30)
    with pytest.raises(ValueError,match="disjoint"):
        select_v2_ranks(operators,errors,{"Query":("q0",),"Gate":("q0","q1")},.4,30)


def test_byte_budget_is_emitted_with_its_own_interpreter_key():
    operators=(RankOperatorCost("one",(3,4),bytes_per_element=8),)
    grid={"one":(1,2)}
    data=tuple(replace(e,cost=exact_unique_cost(operators[0].storages(e.rank),"bytes"))
               for e in observations(operators,grid,score_name="validation_loss"))
    allowance=drone_time_allowances({"one":1.},.1)[0]
    operations=(
        lambda:select_isolated_ranks("asvd",operators,grid,data,112,unit="bytes").candidate,
        lambda:rank_search_space("bolaco",operators,grid,budget=112,unit="bytes")[0][-1],
        lambda:select_v2_ranks(operators,{"one":2.},{"Query":("one",)},.3,112,unit="bytes").candidate,
        lambda:select_drone_step(operators[0],grid["one"],data,allowance,reference_loss=1.,baseline_id="trained0",
                               checkpoint_id="checkpoint0",graph_version="graph0",statistics_id="stats0",budget=112,unit="bytes").candidate,
    )
    for operation in operations:
        candidate=operation()
        assert candidate.parameters["tensor_byte_budget"]==112
        assert "parameter_budget" not in candidate.parameters
