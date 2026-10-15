"""Value-only planner contracts, distinct rank meanings and legacy policies."""
from dataclasses import asdict, replace
import json
import pytest
import torch

from fedcore.algorithm.low_rank.plans import (
    ExplicitMask, FixedRank, Goal, GraphWidth, LegacyThreshold, MetricPolicy,
    OperatorDescriptor, ParameterFraction, RankFraction, ResourceLimits,
    SharedBudget, SnapshotRef, TransformPlan, TransformRequest,
    ValidationFailure, legacy_request, plan_transform,
)


def request(structure=FixedRank(2), **changes):
    return replace(TransformRequest(Goal.REPLACE_OPERATOR, "frobenius", "svd", structure), **changes)


def test_plan_is_deterministic_serializable_and_contains_refs_only():
    operators = (OperatorDescriptor("block.proj", (8, 6), "Linear", bias_elements=8),)
    req = request(statistics_refs=(SnapshotRef("moment-1", "ckpt", "graph-1"),))
    first = plan_transform(req, operators)
    assert isinstance(first, TransformPlan)
    assert first == plan_transform(req, operators)
    decoded = json.loads(json.dumps(asdict(first)))
    assert decoded["steps"][0]["rank"] == 2
    assert decoded["request"]["statistics_refs"][0]["snapshot_id"] == "moment-1"
    with pytest.raises(Exception):
        first.schema_version = 2


def test_independent_shape_budget_metric_violations_accumulate_without_effects():
    result = plan_transform(request(SharedBudget(-2), metric_policy=MetricPolicy(ridge=float("nan"))),
                            (OperatorDescriptor("bad", (-1, 0), "Linear"),), ResourceLimits(-1))
    assert isinstance(result, ValidationFailure)
    codes = {item.code for item in result.violations}
    assert {"InvalidBudget", "InvalidShape", "InvalidRidge", "InvalidResourceLimit"} <= codes


@pytest.mark.parametrize("rank", [True, 0, -1, 9, 2.5])
def test_fixed_rank_rejects_noninteger_and_shape_overflow(rank):
    assert isinstance(plan_transform(request(FixedRank(rank)), (OperatorDescriptor("x", (8, 4), "Linear"),)), ValidationFailure)


def test_rank_fraction_and_parameter_fraction_are_distinct():
    operator = (OperatorDescriptor("x", (8, 8), "Linear", bias_elements=8),)
    rank_plan = plan_transform(request(RankFraction(.5)), operator)
    parameter_plan = plan_transform(request(ParameterFraction(.5)), operator)
    assert rank_plan.steps[0].rank == 4
    # 36 total stored elements allow floor((36-8)/(8+8+1))=1.
    assert parameter_plan.steps[0].rank == 1
    assert isinstance(plan_transform(request(ParameterFraction(.1)), operator), ValidationFailure)
    assert isinstance(plan_transform(request(ParameterFraction(.5), representation="one_layer"), operator), ValidationFailure)


@pytest.mark.parametrize("strategy", ["energy", "explained_variance", "absolute_sum", "quantile"])
def test_historical_threshold_is_preserved_as_operator_spectrum_policy(strategy):
    decoded = legacy_request(strategy=strategy, threshold=.7)
    assert decoded.structure == LegacyThreshold(strategy, .7)
    plan = plan_transform(decoded, (OperatorDescriptor("x", (4, 4), "Linear"),))
    assert plan.steps[0].requires_operator_spectrum
    assert plan.steps[0].rank is None
    assert isinstance(legacy_request(rank=2, strategy=strategy, threshold=.7), ValidationFailure)


def test_weighted_profile_validation_precedes_runtime():
    conv = OperatorDescriptor("conv", (8, 2, 3, 2), "Conv2d", groups=2,
                              kernel_size=(3, 2), stride=(2, 1), padding=(1, 0), dilation=(1, 2))
    req = request(objective="input_second_moment")
    plan = plan_transform(req, (conv,))
    assert plan.steps[0].requires_statistics
    assert isinstance(plan_transform(replace(req, decomposing_mode="spatial"), (conv,)), ValidationFailure)
    assert isinstance(plan_transform(req, (replace(conv, dtype="float16"),)), ValidationFailure)
    assert isinstance(plan_transform(req, (replace(conv, padding="same"),)), ValidationFailure)
    assert isinstance(plan_transform(req, (replace(conv, kernel_size=(1, 1)),)), ValidationFailure)


def test_live_resources_and_ambiguous_structures_are_rejected():
    assert isinstance(plan_transform(request(statistics_refs=(torch.ones(2),)), (OperatorDescriptor("x", (4, 4), "Linear"),)), ValidationFailure)
    assert isinstance(plan_transform(request(objective="input_second_moment", decomposing_mode=torch.ones(2)), (OperatorDescriptor("x", (4, 4), "Linear"),)), ValidationFailure)
    assert isinstance(plan_transform(request(ExplicitMask((0, 0))), (OperatorDescriptor("x", (4, 4), "Linear"),)), ValidationFailure)
    assert isinstance(plan_transform(request(GraphWidth(False)), (OperatorDescriptor("x", (4, 4), "Linear"),)), ValidationFailure)
    assert isinstance(plan_transform(request(ExplicitMask((0, 5))), (OperatorDescriptor("x", (4, 4), "Linear"),)), ValidationFailure)


@pytest.mark.parametrize("goal", list(Goal))
def test_future_goal_descriptions_do_not_register_solvers(goal):
    plan = plan_transform(request(goal=goal), (OperatorDescriptor("x", (4, 4), "Linear"),))
    if goal is Goal.REPLACE_OPERATOR:
        assert isinstance(plan, TransformPlan)
    else:
        assert isinstance(plan, ValidationFailure)
        assert any(item.code == "UnsupportedProfile" for item in plan.violations)


@pytest.mark.parametrize("descriptor", [OperatorDescriptor("x", 4, "Linear"), OperatorDescriptor("x", (2, 3, 4), "Linear"), OperatorDescriptor(3, (2, 3), "Linear"), OperatorDescriptor("x", (2, 3), "unknown"), OperatorDescriptor("x", (2, 3, 4), "Conv1d", kernel_size=3)])
def test_malformed_descriptor_has_value_failure(descriptor):
    assert isinstance(plan_transform(request(), (descriptor,)), ValidationFailure)


def test_malformed_limits_and_operator_sequence_have_value_failures():
    assert isinstance(plan_transform(request(), None), ValidationFailure)
    assert isinstance(plan_transform(request(), (OperatorDescriptor("x", (2, 2), "Linear"),), None), ValidationFailure)
