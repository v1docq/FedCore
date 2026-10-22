"""Pure finite rank schedules and decisions for the existing experiment runner.

Isolated whole-model observations are data, never an additive prediction of the
joint model's quality. ``allocate`` solves only the explicitly named surrogate
and the exact stored-element budget. Every selected joint candidate still needs
an independent execution/evaluation by the existing runner/search boundary.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
from itertools import product
import math
from typing import Mapping

from fedcore.algorithm.low_rank.allocation import (
    StorageCost,ScoreObjective,CandidateOption,CandidateTable,Allocation,
    BudgetInfeasible,allocate,exact_unique_cost,
)
from fedcore.algorithm.low_rank.structured_profiles import inverse_log_allocation
from .protocol import CandidateSpec


def _finite(value,name,*,positive=False):
    if (type(value) not in (int,float) or not math.isfinite(value)
            or (positive and value<=0)):
        raise ValueError(f"{name} must be finite"+(" and positive" if positive else ""))


def _execution_unit(unit):
    if unit not in ("parameters","bytes"):
        raise ValueError("explicit parameter-element or tensor-byte budget unit required")


@dataclass(frozen=True)
class RankOperatorCost:
    path: str
    shape: tuple[int,int]
    bias_elements: int = 0
    bytes_per_element: int = 4
    extra_storages: tuple[StorageCost,...] = ()

    def __post_init__(self):
        if (not isinstance(self.path,str) or not isinstance(self.shape,tuple) or len(self.shape)!=2
                or any(type(v) is not int or v<=0 for v in self.shape)
                or type(self.bias_elements) is not int or self.bias_elements<0
                or type(self.bytes_per_element) is not int or self.bytes_per_element<=0
                or not isinstance(self.extra_storages,tuple)):
            raise ValueError("explicit path, matrix shape, bias size and precision required")
        exact_unique_cost(self.extra_storages)

    def storages(self,rank):
        if type(rank) is not int or not 1<=rank<=min(self.shape):
            raise ValueError("integer rank must fit the operator shape")
        namespace=f"rank-policy:{len(self.path)}:{self.path}"
        # Alternative factors need distinct identities; biases/maps/shared fixed
        # state keep their stable identities across the candidate alternatives.
        return (StorageCost(f"{namespace}:{rank}:left",self.shape[0]*rank,self.bytes_per_element),
                StorageCost(f"{namespace}:{rank}:right",self.shape[1]*rank,self.bytes_per_element),
                StorageCost(f"{namespace}:bias",self.bias_elements,self.bytes_per_element,"bias"),
                *self.extra_storages)


@dataclass(frozen=True)
class IsolatedRankEvaluation:
    path: str
    rank: int
    score: float
    cost: int
    baseline_id: str
    score_name: str
    score_direction: str
    data_role: str
    checkpoint_id: str
    graph_version: str = "baseline"
    statistics_id: str = ""
    statistics_role: str = "calibration"
    score_semantics: str = "isolated_observed"

    def __post_init__(self):
        _finite(self.score,"observed score")
        if (not isinstance(self.path,str) or type(self.rank) is not int or self.rank<1
                or type(self.cost) is not int or self.cost<0
                or not all(isinstance(v,str) and v for v in
                           (self.baseline_id,self.checkpoint_id,self.score_name,self.graph_version,self.statistics_id))
                or self.score_direction not in ("minimize","maximize")
                or self.data_role!="validation" or self.statistics_role!="calibration"
                or self.score_semantics!="isolated_observed"):
            raise ValueError("isolated validation observations need one trained baseline/checkpoint and calibration-only statistics")


@dataclass(frozen=True)
class RankSelectionPlan:
    candidate: CandidateSpec
    allocation: Allocation
    evidence: Mapping

    def to_dict(self):
        return {"candidate":self.candidate.to_dict(),"allocation":asdict(self.allocation),
                "evidence":dict(self.evidence)}


@dataclass(frozen=True)
class DRONEAllowance:
    path: str
    measured_time: float
    time_fraction: float
    loss_growth: float

    def __post_init__(self):
        _finite(self.measured_time,"measured module time",positive=True)
        _finite(self.time_fraction,"time_fraction")
        _finite(self.loss_growth,"loss_growth")
        if not isinstance(self.path,str) or not 0<=self.time_fraction<=1 or self.loss_growth<0:
            raise ValueError("named module, time fraction in [0,1] and nonnegative loss allowance required")


def _operators(operators,rank_grid):
    operators=tuple(operators)
    if (not operators or any(not isinstance(op,RankOperatorCost) for op in operators)
            or len({op.path for op in operators})!=len(operators)
            or not isinstance(rank_grid,Mapping) or set(rank_grid)!={op.path for op in operators}):
        raise ValueError("unique operators and complete explicit rank grids required")
    grids={}
    for op in operators:
        ranks=tuple(rank_grid[op.path])
        if not ranks or len(set(ranks))!=len(ranks):
            raise ValueError("nonempty unique rank grid required")
        for rank in ranks:
            op.storages(rank)
        grids[op.path]=ranks
    return operators,grids


def _candidate(family,ranks,method_options,budget=None,unit="parameters"):
    _execution_unit(unit)
    if family not in ("asvd","afm","bolaco","flar_svd","drone","svdllm_v1","svdllm_v2"):
        raise ValueError("explicit supported rank-selection family required")
    parameters={"method_options":dict(method_options or {}),"target_paths":list(ranks),
                "ranks":dict(ranks),"finetune_epochs":0}
    if budget is not None:
        parameters["parameter_budget" if unit=="parameters" else "tensor_byte_budget"]=budget
    return CandidateSpec(family,parameters)


def rank_candidate_schedule(family,operators,rank_grid,*,method_options=None,
                            baseline_id,checkpoint_id):
    """Frozen isolated candidate schedules; all start at the same checkpoint.

    Existing PETRA owns effects and scalar observation collection. The baseline
    identities live in schedule evidence, not unrecognized execution options.
    """
    operators,grids=_operators(operators,rank_grid)
    if not all(isinstance(value,str) and value for value in (baseline_id,checkpoint_id)):
        raise ValueError("trained baseline and checkpoint identities required")
    candidates=tuple(_candidate(family,{op.path:rank},method_options)
                     for op in operators for rank in grids[op.path])
    return candidates,{"family":family,"baseline_id":baseline_id,"checkpoint_id":checkpoint_id,
                       "candidate_ids":[c.candidate_id for c in candidates],
                       "execution":"existing_runner_independent_baseline_copy_per_candidate",
                       "statistics_role":"calibration","score_role":"validation",
                       "score_semantics":"isolated_observed"}


def _evaluations(operators,grids,evaluations,unit):
    _execution_unit(unit)
    evaluations=tuple(evaluations)
    if not evaluations or any(not isinstance(e,IsolatedRankEvaluation) for e in evaluations):
        raise ValueError("complete IsolatedRankEvaluation observations required")
    expected={(op.path,rank) for op in operators for rank in grids[op.path]}
    actual=[(e.path,e.rank) for e in evaluations]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError("isolated evaluation table must cover the complete declared path/rank grid exactly once")
    context={(e.baseline_id,e.checkpoint_id,e.graph_version,e.score_name,e.score_direction,
              e.data_role,e.statistics_role,e.score_semantics) for e in evaluations}
    if len(context)!=1:
        raise ValueError("cannot mix baseline, checkpoint, graph, score or data-role contexts")
    by_path={op.path:op for op in operators}
    for observation in evaluations:
        if observation.cost!=exact_unique_cost(by_path[observation.path].storages(observation.rank),unit):
            raise ValueError("observed cost disagrees with complete factors/bias/maps/extra storage cost")
    for op in operators:
        if len({e.statistics_id for e in evaluations if e.path==op.path})!=1:
            raise ValueError("every rank of one operator must use the same calibration statistics snapshot")
    return {(e.path,e.rank):e for e in evaluations},evaluations[0]


def select_isolated_ranks(family,operators,rank_grid,evaluations,budget,*,
                          method_options=None,fixed_storages=(),unit="parameters"):
    """Allocate integer ranks using a named sum of isolated observed scores.

    For ASVD this can be a validation sensitivity; AAFM can supply observed KL.
    This function never converts abs activations or PCA tails into observed KL.
    Bolaco's full joint search uses ``rank_search_space`` and existing Bayesian
    search, not the sum objective as a fictitious joint observation.
    """
    operators,grids=_operators(operators,rank_grid)
    observed,context=_evaluations(operators,grids,evaluations,unit)
    tables=tuple(CandidateTable(f"target:{op.path}",tuple(
        CandidateOption(f"rank:{rank:012d}",(rank,),observed[op.path,rank].score,op.storages(rank))
        for rank in grids[op.path]),(op.path,),(min(op.shape),)) for op in operators)
    objective=ScoreObjective(f"sum_isolated_{context.score_name}",context.score_direction,"additive_surrogate")
    selected=allocate(tables,budget,objective=objective,fixed_storages=fixed_storages,unit=unit)
    if isinstance(selected,BudgetInfeasible):
        return selected
    table_paths={f"target:{op.path}":op.path for op in operators}
    ranks={table_paths[item.table_id]:item.option.ranks[0] for item in selected.selected}
    candidate=_candidate(family,ranks,method_options,budget,unit)
    return RankSelectionPlan(candidate,selected,
                             {"family":family,"baseline_id":context.baseline_id,"checkpoint_id":context.checkpoint_id,
                              "graph_version":context.graph_version,"isolated_score_name":context.score_name,
                              "statistics_ids":{op.path:observed[op.path,grids[op.path][0]].statistics_id for op in operators},
                              "score_semantics":"additive_surrogate_of_isolated_observations",
                              "joint_quality_observed":False,"joint_evaluation_required":True,
                              "data_roles":{"statistics":"calibration","selection":"validation"},
                              "cost_unit":unit,"actual_total_cost":selected.total_cost})


def rank_search_space(family,operators,rank_grid,*,budget,method_options=None,
                      fixed_storages=(),unit="parameters",max_combinations=100000):
    """Enumerate admissible joint candidates for the EXISTING finite search.

    This is a prospective domain, not evaluations, an optimizer, or a claim that
    all candidates have been observed. Existing run_search(method='bayesian')
    can consume this same frozen CandidateSpec tuple for Bolaco.
    """
    _execution_unit(unit)
    operators,grids=_operators(operators,rank_grid)
    fixed_storages=tuple(fixed_storages)
    if type(max_combinations) is not int or max_combinations<1:
        raise ValueError("positive finite Cartesian-domain limit required")
    if type(budget) is not int or budget<0:
        raise ValueError("nonnegative integer budget required")
    combinations=math.prod(len(grids[op.path]) for op in operators)
    if combinations>max_combinations:
        raise ValueError("joint finite search domain exceeds declared enumeration limit")
    result=[]
    for ranks in product(*(grids[op.path] for op in operators)):
        cost=exact_unique_cost((*fixed_storages,*(s for op,rank in zip(operators,ranks) for s in op.storages(rank))),unit)
        if cost<=budget:
            result.append(_candidate(family,{op.path:rank for op,rank in zip(operators,ranks)},method_options,budget,unit))
    return tuple(result),{"family":family,"unfiltered_combinations":combinations,
                          "feasible_candidates":len(result),"cost_unit":unit,"budget":budget,
                          "joint_quality_observed":False,"joint_evaluation_required":True,
                          "search_executor":"existing_fedcore_experiments_search_run_search"}


def drone_time_allowances(module_times,total_loss_growth):
    """DRONE Algorithm 2: R_i=expm1(log1p(r)*time_i/sum(time)).

    The product identity is allocation arithmetic; it is not a theorem about
    independence of approximation errors or end-to-end loss accumulation.
    Times must share a measurement profile and units, declared by the caller.
    """
    _finite(total_loss_growth,"total_loss_growth")
    if total_loss_growth<0 or not module_times or any(not isinstance(p,str) for p in module_times):
        raise ValueError("nonnegative total growth and named measured module times required")
    for value in module_times.values():
        _finite(value,"module time",positive=True)
    maximum=max(module_times.values())
    scaled={path:value/maximum for path,value in module_times.items()}
    total=math.fsum(scaled.values())
    return tuple(DRONEAllowance(path,float(module_times[path]),scaled[path]/total,
                               math.expm1(math.log1p(total_loss_growth)*(scaled[path]/total)))
                 for path in module_times)


def select_drone_step(operator,rank_grid,evaluations,allowance,*,reference_loss,
                       baseline_id,checkpoint_id,graph_version,statistics_id,budget,
                       fixed_storages=(),unit="parameters",method_options=None,strict=True):
    """First admissible rank in one current-graph step's declared grid.

    Every isolated evaluation must share the accepted predecessor graph and
    statistics identity. ``strict=True`` reproduces the paper's strict ratio
    test; false explicitly admits equality. Remaining model storage is supplied
    as fixed records, so the actual total budget is checked for each candidate.
    """
    _finite(reference_loss,"reference_loss",positive=True)
    if not isinstance(allowance,DRONEAllowance) or allowance.path!=operator.path:
        raise ValueError("matching measured-time allowance required")
    _finite(allowance.loss_growth,"allowance growth")
    if allowance.loss_growth<0 or type(strict) is not bool or not statistics_id:
        raise ValueError("explicit nonnegative allowance, comparison policy and current statistics id required")
    operators,grids=_operators((operator,),{operator.path:rank_grid})
    observed,context=_evaluations(operators,grids,evaluations,unit)
    if (context.baseline_id!=baseline_id or context.checkpoint_id!=checkpoint_id
            or context.graph_version!=graph_version or context.score_direction!="minimize"
            or context.score_name not in ("loss","validation_loss")
            or any(e.statistics_id!=statistics_id for e in observed.values())):
        raise ValueError("stale or incompatible DRONE predecessor/statistics context")
    threshold=reference_loss*(1+allowance.loss_growth)
    _finite(threshold,"accepted loss threshold")
    accepted=[]
    for position,rank in enumerate(grids[operator.path]):
        record=observed[operator.path,rank]
        okay=record.score<threshold if strict else record.score<=threshold
        if okay:
            accepted.append(CandidateOption(f"grid:{position:012d}",(rank,),float(position),operator.storages(rank)))
    if not accepted:
        return None
    table=CandidateTable(f"target:{operator.path}",tuple(accepted),(operator.path,),(min(operator.shape),))
    allocation=allocate((table,),budget,objective=ScoreObjective("declared_drone_grid_position"),
                        fixed_storages=fixed_storages,unit=unit)
    if isinstance(allocation,BudgetInfeasible):
        return allocation
    rank=allocation.selected[0].option.ranks[0]
    return RankSelectionPlan(_candidate("drone",{operator.path:rank},method_options,budget,unit),allocation,
                             {"baseline_id":baseline_id,"checkpoint_id":checkpoint_id,"graph_version":graph_version,
                              "statistics_id":statistics_id,"reference_loss":reference_loss,
                              "observed_candidate_loss":observed[operator.path,rank].score,
                              "time_allowance":asdict(allowance),"strict_comparison":strict,
                              "product_identity_is_loss_guarantee":False,"joint_evaluation_required":True})


def select_v2_ranks(operators,raw_errors,role_groups,removal_fraction,budget,*,
                     normalization=1.,rank_grid=None,fixed_storages=(),unit="parameters",method_options=None):
    """Project V2 continuous inverse-log targets onto feasible integer ranks.

    Published positive-domain scores determine continuous targets per role.
    Exact allocation minimizes squared departure from those targets; this is an
    explicitly named integer projection, not a new published error formula.
    Different matrix shapes, bias, affine extras and untouched storage are all
    counted. Undefined losses are refused by inverse_log_allocation, never fixed
    by automatic clipping, pseudocounts or hidden normalization.
    """
    _execution_unit(unit)
    operators=tuple(operators)
    if rank_grid is None:
        rank_grid={op.path:tuple(range(1,min(op.shape)+1)) for op in operators}
    operators,grids=_operators(operators,rank_grid)
    paths={op.path for op in operators}
    grouped=[path for group in role_groups.values() for path in group]
    if (not role_groups or any(not isinstance(role,str) or not role or not group for role,group in role_groups.items())
            or len(grouped)!=len(set(grouped)) or set(grouped)!=paths or set(raw_errors)!=paths):
        raise ValueError("complete disjoint matrix role groups and raw spectral errors required")
    ratios={}; role_evidence={}
    for role,group in role_groups.items():
        removal,metadata=inverse_log_allocation(tuple(raw_errors[path] for path in group),
                                               removal_fraction,normalization=normalization)
        ratios.update(zip(group,removal)); role_evidence[role]=metadata
    targets={op.path:(1-ratios[op.path])*math.prod(op.shape)/sum(op.shape) for op in operators}
    tables=tuple(CandidateTable(f"target:{op.path}",tuple(
        CandidateOption(f"rank:{rank:012d}",(rank,),(rank-targets[op.path])**2,op.storages(rank))
        for rank in grids[op.path]),(op.path,),(min(op.shape),)) for op in operators)
    allocation=allocate(tables,budget,objective=ScoreObjective("integer_projection_of_v2_inverse_log_rank_targets"),
                        fixed_storages=fixed_storages,unit=unit)
    if isinstance(allocation,BudgetInfeasible):
        return allocation
    table_paths={f"target:{op.path}":op.path for op in operators}
    ranks={table_paths[item.table_id]:item.option.ranks[0] for item in allocation.selected}
    return RankSelectionPlan(_candidate("svdllm_v2",ranks,method_options,budget,unit),allocation,
                             {"method":"svdllm_v2_positive_domain_integer_projection",
                              "source_version":"2503.12340v1","role_groups":{k:list(v) for k,v in role_groups.items()},
                              "role_allocation":role_evidence,"continuous_rank_targets":targets,
                              "integer_ranks":ranks,"actual_total_cost":allocation.total_cost,"cost_unit":unit,
                              "published_continuous_formula":True,"published_integer_projection":False,
                              "score_semantics":"integer_target_distance_surrogate",
                              "joint_quality_observed":False,"joint_evaluation_required":True})
