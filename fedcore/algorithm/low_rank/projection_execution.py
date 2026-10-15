"""Bounded CPU execution for P3, reusing topology, costs and P2 plans.

No runner, optimizer or universal registry lives here. Recovery is a callback
to the existing training boundary; diagnostics return dense interventions.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
from itertools import islice
import json
import math
import os
import time
import torch
from torch import nn

from .allocation import exact_unique_cost
from .execution import (_fingerprint, _hash_tensor, _storage_records, plan_weighted,
                         WeightedProfileError)
from .method_execution import (MethodTransformResult, _collect_rows, _rng_isolated,
                               _check_data, _tensor_state_bytes, _explicit_encoder_graph)
from .plans import FixedRank
from .statistics import StatisticsPassport, _passport
from .structured_profiles import FactorPair
from .structured_layers import FactorizedLinear
from .topology import (module_paths, inspect_topology, validate_parameter_topology,
                       replace_modules_atomically, TopologyError)
from .projection_specs import ESPACE, PELA, LASER, CTRAFM, CTRMLPAFM
from .projection_profiles import (ESPACEStatistics, espace_statistics, select_espace_projector,
    pela_rank, laser_rank, solve_ctr_field, absorb_ctr_linear_blocks, LASER_SOURCE,CTR_SOURCE)
from .projection_layers import FixedProjectorLinear, CenteredPCAEmbedding, CTRFieldGraph, ReLUFactorizedLinear
from .projection_recovery import (PELARecoveryObjective,FixedProjectorRecoveryObjective,
                                  observed_loss)


class ProjectionProfileError(WeightedProfileError):
    pass


def _positive(value,name):
    if type(value) is not int or value <= 0:
        raise ProjectionProfileError(f"{name} must be a positive integer")


def _selected(model,target_paths):
    paths = module_paths(model)
    if target_paths is not None and (not isinstance(target_paths,(tuple,list)) or any(not isinstance(path,str) for path in target_paths)):
        raise ProjectionProfileError("Target paths must be an explicit string sequence")
    selected = tuple(path for path,layer in paths.items() if type(layer) is nn.Linear) if target_paths is None else tuple(target_paths)
    if not selected or len(set(selected)) != len(selected) or any(path not in paths or type(paths[path]) is not nn.Linear for path in selected):
        raise ProjectionProfileError("P3 requires unique standard Linear target paths")
    topology = validate_parameter_topology(model,selected)
    selected_parameters = [paths[path].weight for path in selected]
    if any(value.dtype not in (torch.float32,torch.float64) or not bool(torch.isfinite(value).all()) for value in selected_parameters):
        raise ProjectionProfileError("P3 targets require finite float32/float64 weights")
    if len({id(value) for value in selected_parameters}) != len(selected_parameters):
        raise TopologyError("P3 independent projectors do not support tied target matrices")
    if topology.storage_aliases:
        raise TopologyError("P3 private copies reject storage views")
    return selected


def plan_projection(model,spec,*,rank=None,rank_ratio=None,parameter_fraction=None,ranks=None,
                    target_paths=None,max_peak_bytes=12*1024**3):
    if type(spec) is CTRAFM:
        if type(model) is not CTRFieldGraph:
            raise ProjectionProfileError("CTRAFM requires CTRFieldGraph with all consumers declared")
        model.validate_consumers()
        if set(dict(spec.field_ranks)) != set(model.fields):
            raise ProjectionProfileError("Every CTR field must have one declared rank")
        if any(value is not None for value in (rank,rank_ratio,parameter_fraction,ranks,target_paths)):
            raise ProjectionProfileError("CTR uses its explicit field ranks, not a Linear rank policy")
        if model.interaction_pairs and spec.interaction_policy == "refuse":
            raise ProjectionProfileError("Interaction consumers explicitly refused by CTR policy")
        if spec.nonlinear_variant != "linear_v1":
            raise ProjectionProfileError("CTRFieldGraph additive consumers are linear; ReLUFactorizedLinear requires the separate MLP recovery profile")
        return None
    if type(spec) not in (ESPACE,PELA,LASER,CTRMLPAFM):
        raise ProjectionProfileError("Unknown P3 projection description")
    selected = _selected(model,target_paths)
    if type(spec) in (PELA,LASER):
        if any(value is not None for value in (rank,rank_ratio,parameter_fraction,ranks)):
            raise ProjectionProfileError("PELA kappa and LASER rho are their explicit rank policies")
        if type(spec) is LASER and len(selected) != 1:
            raise ProjectionProfileError("A LASER diagnostic is a single explicit intervention")
        chosen = {path:(pela_rank(tuple(model.get_submodule(path).weight.shape),spec.kappa) if type(spec) is PELA else laser_rank(tuple(model.get_submodule(path).weight.shape),spec.rho)) for path in selected}
    elif ranks is not None:
        if not isinstance(ranks,dict) or set(ranks) != set(selected) or any(value is not None for value in (rank,rank_ratio,parameter_fraction)):
            raise ProjectionProfileError("Explicit ranks must cover every target without another policy")
        chosen = ranks
    else:
        if type(spec) is CTRMLPAFM:
            from .method_execution import plan_method
            from .method_specs import AFM
            # AFM's existing planner includes the newly required affine bias
            # even when the source Linear had no bias.
            plan = plan_method(model,AFM(),rank=rank,rank_ratio=rank_ratio,parameter_fraction=parameter_fraction,
                target_paths=selected,max_peak_bytes=max_peak_bytes)
        else:
            plan = plan_weighted(model,rank=rank,rank_ratio=rank_ratio,parameter_fraction=parameter_fraction,
                target_paths=selected,max_peak_bytes=max_peak_bytes)
        objective = "ctr_mlp_affine_pca_"+spec.nonlinear_variant if type(spec) is CTRMLPAFM else "espace_"+spec.criterion
        return replace(plan,request=replace(plan.request,objective=objective,representation="factor_pair"))
    base = plan_weighted(model,rank_ratio=1.,target_paths=selected,max_peak_bytes=max_peak_bytes)
    steps = []
    for step in base.steps:
        value = chosen[step.operator.path]
        minimum = 0 if type(spec) is LASER else 1
        if type(value) is not int or not minimum <= value <= min(step.operator.shape):
            raise ProjectionProfileError("P3 rank exceeds the target dimensions")
        steps.append(replace(step,rank=value,structure=FixedRank(value)))
    return replace(base,steps=tuple(steps),request=replace(base.request,
        objective="laser_dense_intervention" if type(spec) is LASER else ("pela_svd_initialization" if type(spec) is PELA else ("ctr_mlp_affine_pca_"+spec.nonlinear_variant if type(spec) is CTRMLPAFM else "espace_"+spec.criterion)),
        representation="dense" if type(spec) is LASER else "factor_pair"))


@_rng_isolated
def collect_espace_backward(model,target_paths,examples,passport,spec,*,max_workspace_bytes=256*1024**2):
    """One bounded individual-example backward, accumulating paired x/grad H.

    Hooks exist only on a private model copy. Python/NumPy/Torch CPU RNG and
    the source's parameters, buffers, modes and existing .grad stay unchanged.
    At most max_batches are read and max_examples are differentiated. Each
    target must execute exactly once per example; recurrent reuse is refused.
    """
    _passport(passport)
    if type(spec) is not ESPACE or not spec.criterion.startswith("nl_"):
        raise ProjectionProfileError("Backward ESPACE collection requires an NL criterion")
    selected = _selected(model,target_paths)
    _positive(max_workspace_bytes,"max_workspace_bytes")
    if any(value.device.type != "cpu" for value in (*model.parameters(),*model.buffers())):
        raise ProjectionProfileError("ESPACE backward collection requires CPU")
    dimensions = tuple(model.get_submodule(path).in_features for path in selected)
    matrix_reserve = 8*(sum(d*d for d in dimensions)+7*max(dimensions)**2)
    if matrix_reserve > max_workspace_bytes:
        raise ProjectionProfileError("ResourceLimitExceeded: all ESPACE backward criteria and solve workspace")
    work = _explicit_encoder_graph(deepcopy(model)).eval()
    captured,handles,totals,counts,zeros,constants = {},[],{},{},{},{}
    for path in selected:
        dimension = work.get_submodule(path).in_features
        if 8*8*dimension*dimension > max_workspace_bytes:
            raise ProjectionProfileError("ResourceLimitExceeded: ESPACE backward criterion workspace")
        totals[path] = torch.zeros((dimension,dimension),dtype=torch.float64)
        counts[path],zeros[path],constants[path] = 0,0,0.
        def capture(_module,args,path=path):
            if path in captured or not args or not isinstance(args[0],torch.Tensor):
                raise ProjectionProfileError("ESPACE requires one tensor input per target invocation")
            value = args[0]
            if not value.requires_grad:
                value = value.detach().requires_grad_(True)
            captured_bytes = sum(t.numel()*t.element_size() for t in captured.values())
            if matrix_reserve+8*(captured_bytes+value.numel()*value.element_size()) > max_workspace_bytes:
                raise ProjectionProfileError("ResourceLimitExceeded: individual ESPACE activation")
            captured[path] = value
            return (value,)+args[1:]
        handles.append(work.get_submodule(path).register_forward_pre_hook(capture))
    observed = batches = 0
    try:
        for batch in islice(iter(examples),spec.max_batches):
            if not isinstance(batch,(tuple,list)) or len(batch) != 2:
                raise ProjectionProfileError("ESPACE examples are (inputs, observed_labels)")
            inputs,labels = batch
            if any(not isinstance(v,torch.Tensor) or v.device.type != "cpu" or v.ndim == 0 for v in (inputs,labels)) or len(inputs) != len(labels):
                raise ProjectionProfileError("ESPACE input and observed-label batch axes must match on CPU")
            batches += 1
            for index in range(len(inputs)):
                captured.clear()
                with torch.enable_grad():
                    prediction = work(inputs[index:index+1].detach())
                    if set(captured) != set(selected):
                        raise ProjectionProfileError("An ESPACE target was not executed")
                    loss = observed_loss(prediction,labels[index:index+1].detach(),spec.loss,spec.reduction)
                    gradients = torch.autograd.grad(loss,tuple(captured[path] for path in selected),allow_unused=True)
                for path,gradient in zip(selected,gradients):
                    x = captured[path].detach().reshape(-1,captured[path].shape[-1])
                    g = torch.zeros_like(x) if gradient is None else gradient.detach().reshape_as(x)
                    stats = espace_statistics(x,passport,spec,gradients=g)
                    count = stats.passport.count
                    totals[path].add_(stats.matrix*count)
                    if not bool(torch.isfinite(totals[path]).all()):
                        raise ProjectionProfileError("Nonfinite ESPACE backward accumulation")
                    counts[path] += count
                    zeros[path] += stats.zero_vectors
                    if stats.constant is not None:
                        constants[path] += stats.constant*count
                observed += 1
                if observed == spec.max_examples:
                    break
            if observed == spec.max_examples:
                break
        if not observed:
            raise ProjectionProfileError("No ESPACE backward examples were collected")
        manifest = {"individual_examples":observed,"backward_passes":observed,"observed_batches":batches,
                    "max_examples":spec.max_examples,"max_batches":spec.max_batches,
                    "rng_policy":"restore_python_numpy_torch_cpu","source_model":"never_executed",
                    "gradient_scope":"within_example_loss_paired_activation_rows","hooks_removed":True}
        return {path:ESPACEStatistics(totals[path]/counts[path],spec.criterion,
            replace(passport,observation_point=path+".input",count=counts[path],weight_sum=float(counts[path]),method_version="espace_2410.05437v1_"+spec.criterion),
            zeros[path],"normalized_calibration_heuristic" if spec.criterion == "nl_normalized" else "first_order_network_loss_upper_bound",
            None if spec.criterion == "nl_normalized" else constants[path]/counts[path],manifest) for path in selected}
    finally:
        for handle in handles:
            handle.remove()


def _svd_pair(weight,rank):
    if rank == 0:
        return None,torch.zeros_like(weight)
    u,s,vh = torch.linalg.svd(weight.detach().double(),full_matrices=False)
    root = s[:rank].sqrt()
    pair = FactorPair((u[:,:rank]*root).to(weight),(root[:,None]*vh[:rank]).to(weight))
    return pair,pair.matrix()


def _recovery(model,stage,updates,objective,executor):
    frozen = {name:value.detach().clone() for name,value in model.named_parameters() if not value.requires_grad}
    result = executor(model,stage,updates,objective)
    if not isinstance(result,dict):
        raise ProjectionProfileError("Recovery executor must return JSON evidence")
    json.dumps(result,allow_nan=False)
    if result.get("updates",result.get("training_steps")) != updates:
        raise ProjectionProfileError("Recovery executor did not execute the exact approved update budget")
    if any(not torch.equal(value,dict(model.named_parameters())[name].detach()) for name,value in frozen.items()):
        raise ProjectionProfileError("Recovery executor modified frozen parameters")
    return {"stage":stage,"requested_updates":updates,"training":result,"objective":objective.evidence}


@_rng_isolated
def transform_projection(model,calibration,spec,*,rank=None,rank_ratio=None,parameter_fraction=None,ranks=None,
    target_paths=None,labels=None,base_model=None,recovery_executor=None,batch_size=16,
    max_workspace_bytes=256*1024**2,max_peak_bytes=12*1024**3,graph_version="p3-projection-v1",
    parameter_budget=None,tensor_byte_budget=None,data_role="calibration",field_frequencies=None):
    started = time.perf_counter()
    if data_role != "calibration":
        raise ProjectionProfileError("P3 calibration cannot read validation/test data")
    if not isinstance(graph_version,str) or not graph_version:
        raise ProjectionProfileError("P3 requires an explicit graph version")
    for name,value in (("batch_size",batch_size),("max_workspace_bytes",max_workspace_bytes),("max_peak_bytes",max_peak_bytes)):
        _positive(value,name)
    if max_workspace_bytes > max_peak_bytes or max_peak_bytes > 12*1024**3:
        raise ProjectionProfileError("Workspace must fit an explicit CPU peak <=12 GiB")
    plan = plan_projection(model,spec,rank=rank,rank_ratio=rank_ratio,parameter_fraction=parameter_fraction,
        ranks=ranks,target_paths=target_paths,max_peak_bytes=max_peak_bytes)
    if type(spec) is CTRAFM:
        return _transform_ctr(model,spec,field_frequencies,plan,max_workspace_bytes,max_peak_bytes,
                              graph_version,parameter_budget,tensor_byte_budget,started)
    topology = _check_data(model,calibration,batch_size,max_workspace_bytes,max_peak_bytes)
    if not bool(torch.isfinite(calibration).all()):
        raise ProjectionProfileError("Nonfinite P3 calibration data")
    if (type(spec) is PELA or (type(spec) in (ESPACE,CTRMLPAFM) and spec.updates)) and not callable(recovery_executor):
        raise ProjectionProfileError("Explicit recovery updates require the existing training executor")
    if type(spec) is ESPACE and spec.criterion.startswith("nl_"):
        if not isinstance(labels,torch.Tensor) or labels.device.type != "cpu" or labels.ndim == 0 or len(labels) != len(calibration):
            raise ProjectionProfileError("NL ESPACE requires separate observed calibration labels")
    import psutil
    model_bytes = _tensor_state_bytes(model)
    reserve = psutil.Process(os.getpid()).memory_info().rss+64*1024**2+5*model_bytes+calibration.numel()*calibration.element_size()
    if reserve+max_workspace_bytes > max_peak_bytes:
        raise ProjectionProfileError("ResourceLimitExceeded: P3 copy and workspace reservation")
    for step in plan.steps:
        m,n = step.operator.shape
        if 8*(8*n*n+8*m*n+4*m*m) > max_workspace_bytes:
            raise ProjectionProfileError("ResourceLimitExceeded: P3 dense solve workspace")
    checkpoint_id,data_id = _fingerprint(model,calibration,graph_version)
    work = _explicit_encoder_graph(deepcopy(model)).eval()
    modes = {path:layer.training for path,layer in module_paths(model).items()}
    paths = tuple(step.operator.path for step in plan.steps)
    passport = StatisticsPassport(checkpoint_id,graph_version,checkpoint_id,"input",data_id)
    backward = None
    if type(spec) is ESPACE and spec.criterion.startswith("nl_"):
        data_id = hashlib.sha256((data_id+_tensor_digest(labels)).encode()).hexdigest()
        passport = replace(passport,data_id=data_id)
        examples = ((calibration[start:start+batch_size],labels[start:start+batch_size]) for start in range(0,len(calibration),batch_size))
        backward = collect_espace_backward(work,paths,examples,passport,spec,max_workspace_bytes=max_workspace_bytes)
    replacements,records = {},[]
    for step in plan.steps:
        path = step.operator.path
        layer = work.get_submodule(path)
        if type(spec) is ESPACE:
            stats = backward[path] if backward is not None else espace_statistics(
                _collect_rows(work,layer,calibration[:min(spec.max_examples,spec.max_batches*batch_size)],batch_size=batch_size,workspace=max_workspace_bytes,peak=max_peak_bytes,reserve=reserve),
                replace(passport,observation_point=path+".input"),spec,weight=layer.weight)
            projector = select_espace_projector(stats,step.rank)
            replacements[path] = FixedProjectorLinear.from_linear(layer,projector.basis)
            records.append({"path":path,"rank":step.rank,"statistics":asdict(stats.passport),
                            "numerics":projector.diagnostics,"collection":stats.collection_manifest,
                            "implementation_variant":"bounded_cpu_selected_linear_observation_mean_v1"})
        elif type(spec) is CTRMLPAFM:
            from .method_execution import _centered
            from .statistical_profiles import solve_affine_pca
            rows = _collect_rows(work,layer,calibration,batch_size=batch_size,workspace=max_workspace_bytes,peak=max_peak_bytes,reserve=reserve)
            output_rows = torch.nn.functional.linear(rows.to(layer.weight),layer.weight.detach(),None if layer.bias is None else layer.bias.detach())
            answer = solve_affine_pca(layer.weight.detach(),layer.bias,_centered(output_rows,replace(passport,observation_point=path+".output")),step.rank)
            cls = FactorizedLinear if spec.nonlinear_variant == "linear_v1" else ReLUFactorizedLinear
            replacements[path] = cls.from_factors(FactorPair(answer.left,answer.right),answer.bias)
            records.append({"path":path,"rank":step.rank,"nonlinear_variant":spec.nonlinear_variant,
                "linearly_equivalent_to_affine_pca":spec.nonlinear_variant == "linear_v1",
                "recovery_scope":"selected_factor_parameters",
                "affine_pca":answer.manifest,"source":CTR_SOURCE})
            del rows,output_rows,answer
        else:
            pair,dense = _svd_pair(layer.weight,step.rank)
            if type(spec) is LASER:
                replacement = deepcopy(layer)
                with torch.no_grad():
                    replacement.weight.copy_(dense)
            else:
                replacement = FactorizedLinear.from_factors(pair,layer.bias)
            replacements[path] = replacement
            records.append({"path":path,"rank":step.rank,"weight_squared_error":float((layer.weight.detach()-dense).double().square().sum()),
                            "representation":"dense" if type(spec) is LASER else "factor_pair",
                            "rho_semantics":"retained_fraction_of_min_matrix_dimensions" if type(spec) is LASER else None})
    work = replace_modules_atomically(work,replacements)
    recovery = []
    if type(spec) is PELA:
        teacher = model if base_model is None else base_model
        reserve = max(reserve,_preflight_teacher(work,teacher,spec,paths,max_peak_bytes,max_workspace_bytes))
        objective = PELARecoveryObjective(work,teacher,spec,paths)
        objective.validate_features(calibration[:1])
        reserve = max(reserve,_reserve_recovery(work,objective,max_peak_bytes,max_workspace_bytes))
        recovery.append(_recovery(work,"pela",spec.updates,objective,recovery_executor))
    elif type(spec) is CTRMLPAFM and spec.updates:
        work.requires_grad_(False)
        for path in paths:
            work.get_submodule(path).requires_grad_(True)
        objective = FixedProjectorRecoveryObjective(spec)
        objective.evidence = {"objective":"ctr_mlp_observed_label_recovery","loss":spec.loss,
            "reduction":spec.reduction,"nonlinear_variant":spec.nonlinear_variant,
            "linearly_equivalent_to_affine_pca":spec.nonlinear_variant == "linear_v1"}
        reserve = max(reserve,_reserve_recovery(work,objective,max_peak_bytes,max_workspace_bytes))
        recovery.append(_recovery(work,"ctr_mlp_relu" if spec.nonlinear_variant != "linear_v1" else "ctr_mlp_linear",spec.updates,objective,recovery_executor))
    elif type(spec) is ESPACE:
        if spec.updates:
            work.requires_grad_(False)
            for path in paths:
                work.get_submodule(path).weight.requires_grad_(True)
                if work.get_submodule(path).bias is not None:
                    work.get_submodule(path).bias.requires_grad_(True)
            projectors = {path:work.get_submodule(path).projection.detach().clone() for path in paths}
            objective = FixedProjectorRecoveryObjective(spec)
            reserve = max(reserve,_reserve_recovery(work,objective,max_peak_bytes,max_workspace_bytes))
            recovery.append(_recovery(work,"espace_full_weight",spec.updates,objective,recovery_executor))
            if any(not torch.equal(value,work.get_submodule(path).projection) for path,value in projectors.items()):
                raise ProjectionProfileError("ESPACE recovery modified a fixed projection buffer")
        work = replace_modules_atomically(work,{path:work.get_submodule(path).compact() for path in paths})
    for path,layer in module_paths(work).items():
        if path in modes:
            layer.training = modes[path]
    return _finish(model,work,spec,plan,records,recovery,checkpoint_id,data_id,graph_version,
                    reserve+max_workspace_bytes,max_peak_bytes,max_workspace_bytes,parameter_budget,tensor_byte_budget,started)


def _tensor_digest(value):
    digest = hashlib.sha256()
    _hash_tensor(digest,value)
    return digest.hexdigest()


def _finish(original,work,spec,plan,records,recovery,checkpoint_id,data_id,graph_version,
            estimated_peak,peak,workspace,parameter_budget,tensor_byte_budget,started):
    before,after = _storage_records(original),_storage_records(work)
    parameters = exact_unique_cost(after)
    tensor_bytes = _tensor_state_bytes(work)
    if parameter_budget is not None and (type(parameter_budget) is not int or parameters > parameter_budget):
        raise ProjectionProfileError("BudgetInfeasible: P3 exported stored elements")
    if tensor_byte_budget is not None and (type(tensor_byte_budget) is not int or tensor_bytes > tensor_byte_budget):
        raise ProjectionProfileError("BudgetInfeasible: P3 exported tensor bytes")
    name = {ESPACE:"espace",PELA:"pela",LASER:"laser",CTRAFM:"ctr_afm",CTRMLPAFM:"ctr_mlp_afm"}[type(spec)]
    evidence = {"schema_version":1,"profile":"p3_projection_cpu","method":name,"method_version":1,
        "method_spec":{"method":name,"version":1,"options":asdict(spec)},"checkpoint_id":checkpoint_id,"calibration_id":data_id,"data_id":data_id,"data_role":"calibration",
        "graph_version":graph_version,"layers":records,"recovery_stages":recovery,
        "parameters_before":exact_unique_cost(before),"parameters_after":parameters,
        "tensor_bytes_before":_tensor_state_bytes(original),"tensor_bytes_after":tensor_bytes,
        "parameter_bytes_before":exact_unique_cost(before,"bytes"),"parameter_bytes_after":exact_unique_cost(after,"bytes"),
        "buffer_bytes_after":tensor_bytes-exact_unique_cost(after,"bytes"),
        "representation":"dense_intervention" if type(spec) is LASER else "compact",
        "training_cost_separate":True,"optimizer_rebuild_required":True,
        "collection_order":"same_unmodified_graph_before_atomic_replacements",
        "resources":{"estimated_peak_bytes":estimated_peak,"limit_bytes":peak,"workspace_bytes":workspace,
                     "scope":"named tensors and reserved workspace; arbitrary forward/allocator RSS is not a hard bound"},
        "timings":{"total_seconds":time.perf_counter()-started}}
    json.dumps(evidence,allow_nan=False)
    work._fedcore_method_evidence = evidence
    work._fedcore_requires_optimizer_rebuild = True
    if plan is not None:
        work._fedcore_transform_plan = plan
    return MethodTransformResult(work,plan,evidence)


def _reserve_recovery(model,objective,peak,workspace):
    import psutil
    trainable = tuple(p for p in model.parameters() if p.requires_grad)
    trainable_bytes = sum(p.numel()*p.element_size() for p in trainable)
    student_bytes = _tensor_state_bytes(model)
    teacher_bytes = objective.evidence.get("teacher_bytes",0)
    teacher_snapshot_bytes = objective.evidence.get("teacher_snapshot_bytes",teacher_bytes)
    reference_bytes = objective.evidence.get("original_weight_reference_bytes",0)
    # Teacher snapshot is retained to verify immutability on context exit.
    named = student_bytes+teacher_bytes+teacher_snapshot_bytes+reference_bytes+3*trainable_bytes+4*len(trainable)
    reserve = psutil.Process(os.getpid()).memory_info().rss+64*1024**2+named
    objective.evidence["resource_reservation"] = {"student_tensor_bytes":student_bytes,
        "teacher_tensor_bytes":teacher_bytes,"teacher_snapshot_bytes":teacher_snapshot_bytes,
        "original_weight_reference_bytes":reference_bytes,"gradient_reserve_bytes":trainable_bytes,
        "adam_moments_reserve_bytes":2*trainable_bytes,"adam_step_scalar_reserve_bytes":4*len(trainable),
        "workspace_bytes":workspace,"named_total_bytes":named}
    if reserve+workspace > peak:
        raise ProjectionProfileError("ResourceLimitExceeded: student/teacher/gradient/Adam recovery reservation")
    return reserve


def _preflight_teacher(student,teacher,spec,paths,peak,workspace):
    """Reserve teacher copy/snapshots/references before allocating or forwarding."""
    if not isinstance(teacher,nn.Module) or any(value.device.type != "cpu" for value in (*teacher.parameters(),*teacher.buffers())):
        raise ProjectionProfileError("PELA requires a CPU teacher with explicit block/weight correspondence")
    references = []
    for path in paths:
        try:
            original = teacher.get_submodule(path)
        except AttributeError as error:
            raise ProjectionProfileError("PELA teacher is missing an original Linear target") from error
        factor = student.get_submodule(path)
        if type(original) is not nn.Linear or (original.in_features,original.out_features) != (factor.in_features,factor.out_features):
            raise ProjectionProfileError("PELA teacher and factorized student targets disagree")
        references.append(original.weight)
    for student_path,teacher_path in spec.block_map:
        try:
            student.get_submodule(student_path)
            teacher.get_submodule(teacher_path)
        except AttributeError as error:
            raise ProjectionProfileError("Unknown PELA teacher/student block map") from error
    trainable = tuple(student.parameters()) if spec.freeze == "all_student" else tuple(p for path in paths for p in student.get_submodule(path).parameters())
    trainable_bytes = sum(p.numel()*p.element_size() for p in trainable)
    teacher_copy = _tensor_state_bytes(teacher)
    snapshot = sum(value.numel()*value.element_size() for value in teacher.state_dict().values())
    reference = sum(value.numel()*value.element_size() for value in references)
    import psutil
    named = _tensor_state_bytes(student)+teacher_copy+snapshot+reference+3*trainable_bytes+4*len(trainable)
    reserve = psutil.Process(os.getpid()).memory_info().rss+64*1024**2+named
    if reserve+workspace > peak:
        raise ProjectionProfileError("ResourceLimitExceeded: PELA teacher copy/snapshot/recovery preflight")
    return reserve


def transform_ctr_mlp(model,calibration,*,target_paths,rank=None,nonlinear_variant="linear_v1",
                      updates=0,recovery_executor=None,loss="mse",reduction="mean",**kwargs):
    return transform_projection(model,calibration,CTRMLPAFM(nonlinear_variant,updates,loss,reduction),
        target_paths=target_paths,rank=rank,recovery_executor=recovery_executor,**kwargs)


def _transform_ctr(model,spec,frequencies,plan,workspace,peak,graph_version,parameter_budget,tensor_byte_budget,started):
    model.validate_consumers()
    if inspect_topology(model).parameter_aliases or inspect_topology(model).storage_aliases:
        raise TopologyError("CTRAFM requires independently owned fields and consumers")
    if any(value.device.type != "cpu" for value in (*model.parameters(),*model.buffers())):
        raise ProjectionProfileError("CTRAFM requires CPU")
    if any(type(table) is not nn.Embedding for table in model.fields.values()):
        raise ProjectionProfileError("CTRAFM accepts an uncompressed baseline field graph")
    if spec.weighting == "frequency":
        if not isinstance(frequencies,dict) or set(frequencies) != set(model.fields):
            raise ProjectionProfileError("Frequency CTR requires one category-frequency vector per field")
    elif frequencies is not None:
        raise ProjectionProfileError("Uniform-row CTR cannot consume category frequencies")
    import psutil
    reserve = psutil.Process(os.getpid()).memory_info().rss+64*1024**2+4*_tensor_state_bytes(model)
    for name,table in model.fields.items():
        n,d = table.weight.shape
        if 8*(4*n*d+8*d*d) > workspace or reserve+workspace > peak:
            raise ProjectionProfileError("ResourceLimitExceeded: CTR PCA table/covariance workspace")
    checkpoint = hashlib.sha256(graph_version.encode())
    for name,value in model.state_dict().items():
        checkpoint.update(name.encode()); _hash_tensor(checkpoint,value)
    checkpoint_id = checkpoint.hexdigest()
    data_id = "uniform_vocabulary_rows"
    if frequencies is not None:
        data_id = hashlib.sha256("".join(name+_tensor_digest(frequencies[name]) for name in sorted(frequencies)).encode()).hexdigest()
    work = deepcopy(model)
    names = tuple(work.fields)
    factors = [solve_ctr_field(work.fields[name].weight,dict(spec.field_ranks)[name],
        StatisticsPassport(checkpoint_id,graph_version,checkpoint_id,name+".embedding",data_id,weight_id=spec.weighting),
        frequencies=None if frequencies is None else frequencies[name],weighting=spec.weighting) for name in names]
    blocks,bias = absorb_ctr_linear_blocks([work.linear.blocks[name].weight.detach() for name in names],
        [work.linear.blocks[name].bias for name in names],factors)
    for name,field,weight in zip(names,factors,blocks):
        work.fields[name] = CenteredPCAEmbedding.from_factors(field)
        old = work.linear.blocks[name]
        replacement = nn.Linear(weight.shape[1],weight.shape[0],bias=False,dtype=weight.dtype)
        with torch.no_grad():
            replacement.weight.copy_(weight)
        replacement.train(old.training)
        work.linear.blocks[name] = replacement
    if work.linear.bias is not None:
        bias = bias+work.linear.bias.detach()
    work.linear.bias = nn.Parameter(bias.detach().clone())
    work.validate_consumers()
    records = [{"field":name,"rank":field.basis.shape[1],"weighting":spec.weighting,
        "statistics":asdict(field.passport),"consumers":dict(model.validate_consumers())[name],
        "coefficient_convention":"centered_(E-mean)U","decoder_offset":"mean",
        "affine_correction":"(I-UU.T)mean","linear_contributions":"sum",
        "interaction_policy":spec.interaction_policy,"source":CTR_SOURCE} for name,field in zip(names,factors)]
    return _finish(model,work,spec,plan,records,[],checkpoint_id,data_id,graph_version,reserve+workspace,
                    peak,workspace,parameter_budget,tensor_byte_budget,started)


@dataclass(frozen=True)
class DiagnoseIntervention:
    target_path: str
    rank: int
    checkpoint_id: str
    validation_id: str
    criterion: str


@dataclass(frozen=True)
class LASERCandidateEvidence:
    intervention: DiagnoseIntervention
    task_quality: float
    weight_squared_error: float
    tensor_bytes: int
    representation: str
    candidate_id: str
    baseline_id: str


@dataclass(frozen=True)
class LASERDiagnosis:
    candidates: tuple[LASERCandidateEvidence,...]
    selected: LASERCandidateEvidence
    direction: str


@_rng_isolated
def diagnose_laser(model,interventions,evaluator,*,direction="minimize",data_role="validation",
                   max_candidates=64,max_peak_bytes=12*1024**3,max_workspace_bytes=256*1024**2):
    """Evaluate isolated dense candidates from one immutable baseline.

    The returned frozen selection is ready before a caller can evaluate test.
    No greedy composition is inferred; each record has an empty prior history.
    """
    if data_role != "validation" or direction not in ("minimize","maximize") or not callable(evaluator):
        raise ProjectionProfileError("LASER selection requires a validation evaluator and declared direction")
    for name,value in (("max_candidates",max_candidates),("max_peak_bytes",max_peak_bytes),("max_workspace_bytes",max_workspace_bytes)):
        _positive(value,name)
    if max_peak_bytes > 12*1024**3 or max_workspace_bytes > max_peak_bytes:
        raise ProjectionProfileError("LASER workspace must fit a declared CPU peak <=12 GiB")
    if not isinstance(interventions,(tuple,list)) or not interventions or len(interventions) > max_candidates:
        raise ProjectionProfileError("LASER requires an explicitly bounded candidate sequence")
    if any(value.device.type != "cpu" for value in (*model.parameters(),*model.buffers())):
        raise ProjectionProfileError("LASER diagnosis requires CPU")
    if inspect_topology(model).storage_aliases:
        raise TopologyError("LASER diagnosis rejects shared storage views")
    import psutil
    if psutil.Process(os.getpid()).memory_info().rss+3*_tensor_state_bytes(model)+64*1024**2+max_workspace_bytes > max_peak_bytes:
        raise ProjectionProfileError("ResourceLimitExceeded: LASER private candidate copies")
    source = hashlib.sha256()
    for name,value in model.state_dict().items():
        source.update(name.encode()); _hash_tensor(source,value)
    baseline_id = source.hexdigest()
    records = []
    for request in interventions:
        if not isinstance(request,DiagnoseIntervention) or not all(isinstance(getattr(request,name),str) and getattr(request,name) for name in ("checkpoint_id","validation_id","criterion")):
            raise ProjectionProfileError("LASER requires a typed matrix/checkpoint/validation/criterion intervention")
        _selected(model,(request.target_path,))
        shape = model.get_submodule(request.target_path).weight.shape
        m,n = shape
        if 8*(8*m*n+4*m*m+4*n*n) > max_workspace_bytes:
            raise ProjectionProfileError("ResourceLimitExceeded: LASER candidate dense SVD workspace")
        if type(request.rank) is not int or not 0 <= request.rank <= min(shape):
            raise ProjectionProfileError("LASER rank must fit its target matrix")
    if len({item.checkpoint_id for item in interventions}) != 1 or len({(item.validation_id,item.criterion) for item in interventions}) != 1:
        raise ProjectionProfileError("Independent LASER candidates must share baseline and validation criterion")
    for request in interventions:
        @_rng_isolated
        def evaluate():
            work = deepcopy(model).eval()
            layer = work.get_submodule(request.target_path)
            _,dense = _svd_pair(layer.weight,request.rank)
            error = float((layer.weight.detach()-dense).double().square().sum())
            with torch.no_grad():
                layer.weight.copy_(dense)
            expected = {name:value.detach().clone() for name,value in work.state_dict().items()}
            quality = float(evaluator(work))
            if not math.isfinite(quality) or any(not torch.equal(value,work.state_dict()[name]) for name,value in expected.items()):
                raise ProjectionProfileError("LASER validation must return finite quality without mutating candidate state")
            identity = hashlib.sha256((baseline_id+json.dumps(asdict(request),sort_keys=True)).encode()).hexdigest()
            return LASERCandidateEvidence(request,quality,error,_tensor_state_bytes(work),"dense",identity,baseline_id)
        records.append(evaluate())
    selected = (min if direction == "minimize" else max)(records,key=lambda value:value.task_quality)
    return LASERDiagnosis(tuple(records),selected,direction)


@_rng_isolated
def export_laser_compact(model,target_path,rank,*,max_workspace_bytes=256*1024**2):
    """Separate compact export, preserving the chosen dense intervention.

    Caller supplies the already selected dense graph. No second truncation is
    accepted: numerical rank must fit the frozen decision, and economics must
    be strict r(m+n)<mn. Refusals leave the dense source graph untouched.
    """
    _selected(model,(target_path,))
    _positive(max_workspace_bytes,"max_workspace_bytes")
    if any(value.device.type != "cpu" for value in (*model.parameters(),*model.buffers())):
        raise ProjectionProfileError("LASER compact export requires CPU")
    layer = model.get_submodule(target_path)
    m,n = layer.weight.shape
    if type(rank) is not int or not 1 <= rank <= min(m,n) or rank*(m+n) >= m*n:
        raise ProjectionProfileError("LASER compact export requires a positive economical rank")
    if 8*(8*m*n+4*m*m+4*n*n) > max_workspace_bytes:
        raise ProjectionProfileError("ResourceLimitExceeded: LASER export factorization")
    pair,dense = _svd_pair(layer.weight,rank)
    tolerance = 1e-5 if layer.weight.dtype == torch.float32 else 1e-10
    if not torch.allclose(dense,layer.weight,rtol=tolerance,atol=tolerance):
        raise ProjectionProfileError("Frozen dense intervention is not represented by the selected rank")
    work = deepcopy(model)
    replacement = FactorizedLinear.from_factors(pair,layer.bias)
    replacement.train(layer.training)
    work = replace_modules_atomically(work,{target_path:replacement})
    return work,{"source":LASER_SOURCE,"representation":"compact","selected_rank":rank,
        "tensor_bytes_dense":_tensor_state_bytes(model),
        "tensor_bytes_compact":_tensor_state_bytes(work),
        "function_equivalence":"factor_product_equals_selected_dense_operator"}
