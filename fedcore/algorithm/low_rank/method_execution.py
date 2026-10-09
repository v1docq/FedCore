"""One bounded CPU interpreter for the named P2 Linear method profiles.

Each accepted replacement advances the graph identity. Subsequent calibration
observations are collected on that current graph. Numerical objectives are
implemented by small pure collaborators, never inferred from factor names.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
import time
import random
from functools import wraps
from typing import Callable

import torch
from torch import nn

from .allocation import exact_unique_cost
from .execution import (_fingerprint, _hash_tensor, _storage_records,
                        _finite_observations, plan_weighted, WeightedProfileError)
from .method_specs import (AFM, ASVD, BasisSharing, Bolaco, DRONE, EoRA,
    FLARSVD, FWSVD, GroupReduce, MixedRank, SVDLLMV1, SVDLLMV2, SVDLLMV5,
    method_name, method_payload)
from .plans import SnapshotRef, TransformPlan
from .statistics import (StatisticsPassport, empty_centered_moments,
    empty_second_moment, finish_centered_moments, finish_second_moment,
    finish_within_group_covariance, update_centered_moments,
    update_second_moment, normalized_observations)
from .topology import (TopologyError, inspect_topology, module_paths,
                       replace_modules_atomically, validate_parameter_topology)


class MethodProfileError(WeightedProfileError):
    """A declared P2 capability, provenance or resource condition failed."""


@dataclass(frozen=True)
class MethodTransformResult:
    model: nn.Module
    plan: TransformPlan | None
    evidence: dict


def _rng_isolated(function):
    @wraps(function)
    def run(*args, **kwargs):
        import numpy as np
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                return function(*args, **kwargs)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
    return run


def _tensor_id(value):
    digest = hashlib.sha256()
    _hash_tensor(digest, value)
    return digest.hexdigest()


def _matrix(layer):
    from fedcore.models.network_impl.decomposed_layers import IDecomposed
    return (layer.factor_matrix() if isinstance(layer, IDecomposed) else layer.weight).detach()


def _explicit_encoder_graph(model):
    # PyTorch 2.2's fused Encoder reads dense .weight directly and bypasses
    # module hooks. Disable only that dispatch on the owned copy; the actual
    # activation callable, dropout and mathematical graph remain unchanged.
    for layer in model.modules():
        if type(layer) is nn.TransformerEncoderLayer:
            layer.activation_relu_or_gelu = 0
    return model


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise MethodProfileError(f"{name} must be a positive integer")


def plan_method(model, spec, *, rank=None, rank_ratio=None, parameter_fraction=None,
                ranks=None, target_paths=None, max_peak_bytes=12 * 1024**3):
    """Read capabilities only; no copying, hooks, forward pass or job creation."""
    from fedcore.models.network_impl.decomposed_layers import DecomposedLinear
    method_name(spec)
    if type(spec) in (BasisSharing, GroupReduce):
        return _plan_coupled(model, spec, rank, target_paths, max_peak_bytes,
                             rank_ratio, parameter_fraction, ranks)
    paths = module_paths(model)
    selected = tuple(path for path, layer in paths.items()
                     if type(layer) in (nn.Linear, DecomposedLinear)) if target_paths is None else target_paths
    if not isinstance(selected, (list, tuple)) or not selected:
        raise MethodProfileError("Select a nonempty sequence of Linear paths")
    if any(path not in paths or type(paths[path]) not in (nn.Linear, DecomposedLinear) for path in selected):
        raise MethodProfileError("P2 profile supports standard or known decomposed Linear only")
    if type(spec) is FWSVD and any(type(paths[path]) is not nn.Linear for path in selected):
        raise MethodProfileError("The first Fisher collector requires standard Linear targets")
    if type(spec) in (DRONE, SVDLLMV1) and target_paths is None:
        raise MethodProfileError("Sequential methods require an explicit topological target order")
    if ranks is not None:
        if (not isinstance(ranks, dict) or set(ranks) != set(selected)
                or any(value is not None for value in (rank, rank_ratio, parameter_fraction))):
            raise MethodProfileError("ranks must map every selected path, without another rank policy")
        base = plan_weighted(model, rank_ratio=1.0, target_paths=selected,
                             max_peak_bytes=max_peak_bytes)
        steps = []
        for step in base.steps:
            chosen = ranks[step.operator.path]
            _positive(chosen, "rank")
            if chosen > min(step.operator.shape):
                raise MethodProfileError("A selected rank exceeds its Linear dimensions")
            from .plans import FixedRank
            steps.append(replace(step, rank=chosen, structure=FixedRank(chosen)))
        base = replace(base, steps=tuple(steps))
    else:
        base = plan_weighted(model, rank=rank, rank_ratio=rank_ratio,
            parameter_fraction=parameter_fraction, target_paths=selected,
            max_peak_bytes=max_peak_bytes)
    # This record describes the checked method objective; its rank/capability
    # checks are shared with the existing operator planner, not a new registry.
    objectives = {ASVD: "asvd_diagonal", FWSVD: "aggregated_empirical_fisher",
        AFM: "centered_affine_output_pca", Bolaco: "within_group_affine_output_pca",
        FLARSVD: "centered_shrinkage", DRONE: "drone_current_input_thin_support",
        SVDLLMV1: "svdllm_v1_current_input_ls", SVDLLMV2: "svdllm_v2_corrected_support",
        SVDLLMV5: "svdllm_v5_factor_lora", EoRA: "residual_input_second_moment",
        MixedRank: spec.metric if type(spec) is MixedRank else ""}
    if type(spec) in (AFM, Bolaco) and parameter_fraction is not None:
        from .plans import FixedRank
        adjusted = []
        for step in base.steps:
            m, n = step.operator.shape
            total_budget = math.floor(parameter_fraction*(m*n+step.operator.bias_elements))
            affordable = math.floor((total_budget-m)/(m+n))
            if affordable < 1:
                raise MethodProfileError('BudgetInfeasible: affine mean correction does not fit parameter fraction')
            chosen = min(affordable, min(m,n))
            adjusted.append(replace(step, rank=chosen, structure=FixedRank(chosen)))
        base = replace(base, steps=tuple(adjusted))
    return replace(base, request=replace(base.request, objective=objectives[type(spec)]))


def _check_data(model, calibration, batch_size, workspace, peak):
    if not isinstance(model, nn.Module) or any(t.device.type != "cpu" for t in (*model.parameters(), *model.buffers())):
        raise MethodProfileError("P2 runtime requires a CPU torch Module")
    if (not isinstance(calibration, torch.Tensor) or calibration.device.type != "cpu"
            or calibration.dtype not in (torch.float32, torch.float64)
            or calibration.ndim < 2 or calibration.shape[0] == 0 or calibration.numel() == 0):
        raise MethodProfileError("Separate nonempty CPU float calibration data is required")
    for name, value in (("batch_size", batch_size), ("workspace", workspace), ("peak", peak)):
        _positive(value, name)
    if peak > 12 * 1024**3 or workspace > peak:
        raise MethodProfileError("Workspace must fit a declared peak of at most 12 GiB")
    if calibration[0].numel() * calibration.element_size() > workspace:
        raise MethodProfileError("ResourceLimitExceeded: calibration sample exceeds workspace")
    topology = inspect_topology(model)
    if topology.storage_aliases:
        raise TopologyError("P2 copying rejects storage views before copying any model")
    return topology


def _collect_rows(model, layer, calibration, *, batch_size, workspace, peak, reserve):
    chunks, used = [], 0
    def capture(_layer, args):
        nonlocal used
        if len(args) != 1 or not isinstance(args[0], torch.Tensor):
            raise MethodProfileError("The Linear profile requires one tensor input")
        value = args[0].detach()
        if value.device.type != "cpu" or value.dtype not in (torch.float32, torch.float64):
            raise MethodProfileError("Observed activations must be CPU float32/float64")
        if value.ndim < 2 or value.shape[-1] != layer.in_features:
            raise MethodProfileError("Observed input shape does not match the Linear")
        extra = value.numel() * value.element_size()
        # Chunk copies and their final concatenation coexist; check before copy.
        if 2 * (used + extra) > workspace or reserve + 2 * (used + extra) > peak:
            raise MethodProfileError("ResourceLimitExceeded: bounded observation collection")
        if not bool(torch.isfinite(value).all()):
            raise MethodProfileError("Nonfinite layer observations")
        chunks.append(value.reshape(-1, value.shape[-1]).clone())
        used += extra
    handle = layer.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            for batch in calibration.split(batch_size):
                model(batch)
    finally:
        handle.remove()
    if not chunks:
        raise MethodProfileError("The selected Linear was not executed")
    return torch.cat(chunks)


def _moment(rows, passport):
    return finish_second_moment(update_second_moment(
        empty_second_moment(rows.shape[1], passport), rows))


def _centered(rows, passport):
    return finish_centered_moments(update_centered_moments(
        empty_centered_moments(rows.shape[1], passport), rows))


def _author_eora_gram(model, layer, calibration, batch_size):
    from .structured_profiles import eora_author_gram_update
    gram = torch.zeros((layer.in_features, layer.in_features), dtype=torch.float64)
    updates = []
    def capture(_layer, args):
        nonlocal gram
        value = args[0].detach()
        if value.ndim not in (2, 3):
            raise MethodProfileError('Author EoRA hook supports original 2D/3D inputs only')
        # Literal NVlabs hook: 2D is unsqueezed, hence B=1 for that whole call.
        original_batch = 1 if value.ndim == 2 else len(value)
        gram, evidence = eora_author_gram_update(gram, value.reshape(-1,value.shape[-1]),
            calibration_samples=len(calibration), batch_samples=original_batch)
        updates.append(evidence)
    handle = layer.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            for batch in calibration.split(batch_size):
                model(batch)
    finally:
        handle.remove()
    return gram, {'updates':updates,'batch_size':batch_size,
                  'gram_interpretation':'author_fixed_n_forgetting_not_empirical_second_moment'}


def _install(layer, answer):
    """Install canonical actual factors; preserve an affine correction if present."""
    from fedcore.models.network_impl.decomposed_layers import DecomposedLinear
    factors = getattr(answer, "factors", answer)
    if hasattr(factors, "u"):
        u, s, vh = factors.u, factors.s, factors.vh
    else:
        u, s, vh = torch.linalg.svd(factors.left @ factors.right, full_matrices=False)
        width = factors.left.shape[1]
        u, s, vh = u[:, :width], s[:width], vh[:width]
    bias = getattr(answer, "bias", None)
    original_bias = layer.bias if bias is None else bias
    base = nn.Linear(layer.in_features, layer.out_features, bias=original_bias is not None,
                     device=_matrix(layer).device, dtype=_matrix(layer).dtype)
    if original_bias is not None:
        with torch.no_grad():
            base.bias.copy_(original_bias)
    result = DecomposedLinear(base, decomposing_mode=False, compose_mode="two_layers")
    result.decomposing_mode = True
    result.set_U_S_Vh(u, s, vh)
    result.compose_weight_for_inference()
    result.training = layer.training
    return result


def _numerics(answer):
    manifest = getattr(answer, "manifest", getattr(answer, "diagnostics", {}))
    if manifest:
        return manifest
    fields = getattr(answer, "__dataclass_fields__", {})
    return {name: getattr(answer, name) for name in fields
            if name not in ("u", "s", "vh", "approximation", "left", "right")}


def _statistical_solve(spec, weight, bias, rows, passport, rank, group_labels, fisher):
    from .statistical_profiles import (channel_abs_statistics, solve_asvd, solve_fwsvd,
        solve_affine_pca, ledoit_wolf_metric, solve_flar)
    if type(spec) is ASVD:
        stats = channel_abs_statistics(rows, passport, mode=spec.abs_mode)
        return solve_asvd(weight, stats, rank, alpha=spec.alpha,
            epsilon=spec.epsilon, zero_policy='reject' if spec.zero_policy == 'error' else spec.zero_policy)
    if type(spec) is FWSVD:
        return solve_fwsvd(weight, fisher, rank, epsilon=spec.epsilon,
                           zero_policy='reject' if spec.zero_policy == 'error' else spec.zero_policy)
    if type(spec) in (AFM, Bolaco):
        outputs = rows @ weight.T + (bias if bias is not None else 0)
        if type(spec) is AFM:
            moments = _centered(outputs, passport)
        else:
            if (not isinstance(group_labels, torch.Tensor) or group_labels.ndim != 1
                    or len(group_labels) != len(outputs) or group_labels.is_complex()
                    or group_labels.dtype.is_floating_point):
                raise MethodProfileError("Bolaco requires one integral group ID per observed row")
            groups = []
            for label in torch.unique(group_labels, sorted=True):
                subset = outputs[group_labels == label]
                state = update_centered_moments(empty_centered_moments(outputs.shape[1], passport), subset)
                groups.append((str(int(label)), state))
            moments = finish_within_group_covariance(groups)
            if spec.group_weighting == "equal_groups":
                masses = tuple(1 / len(groups) for _ in groups)
                covariance = sum((item.covariance / len(groups) for item in moments.group_moments),
                                 torch.zeros_like(moments.covariance))
                moments = replace(moments, covariance=covariance, group_weights=masses)
        return solve_affine_pca(weight, bias, moments, rank)
    metric = ledoit_wolf_metric(rows, passport)
    return solve_flar(weight, metric, rank)


def _sequential_order(model, paths, calibration, batch_size):
    seen, handles = [], []
    all_paths = module_paths(model)
    for path in paths:
        def record(_module, _args, path=path):
            seen.append(path)
        handles.append(all_paths[path].register_forward_pre_hook(record))
    try:
        with torch.no_grad():
            model(calibration[:batch_size])
    finally:
        for handle in handles:
            handle.remove()
    if seen != list(paths):
        raise MethodProfileError("Sequential profile requires each selected node once in declared execution order")


@_rng_isolated
def transform_method(model, calibration, spec, *, rank=None, rank_ratio=None,
        parameter_fraction=None, ranks=None, target_paths=None, labels=None,
        group_labels=None, base_model=None, recovery_executor: Callable | None=None,
        batch_size=16, max_workspace_bytes=256 * 1024**2,
        max_peak_bytes=12 * 1024**3, graph_version="p2-profile-v1",
        parameter_budget=None, tensor_byte_budget=None, data_role="calibration"):
    """Return a separate graph with exact unique-parameter costs and provenance."""
    started = time.perf_counter()
    if data_role != "calibration":
        raise MethodProfileError("Public profile calibration cannot use validation/test data")
    if not isinstance(graph_version, str) or not graph_version:
        raise MethodProfileError("Explicit graph_version is required")
    for name,value in (('parameter_budget',parameter_budget),('tensor_byte_budget',tensor_byte_budget)):
        if value is not None:
            _positive(value,name)
    plan = plan_method(model, spec, rank=rank, rank_ratio=rank_ratio,
        parameter_fraction=parameter_fraction, ranks=ranks, target_paths=target_paths,
        max_peak_bytes=max_peak_bytes)
    if type(spec) is GroupReduce:
        return _transform_groupreduce(model, calibration, spec, target_paths,
            max_workspace_bytes, max_peak_bytes, graph_version, parameter_budget,tensor_byte_budget)
    topology = _check_data(model, calibration, batch_size, max_workspace_bytes, max_peak_bytes)
    if type(spec) is FWSVD:
        if (not isinstance(labels, torch.Tensor) or labels.device.type!='cpu' or labels.ndim==0
                or len(labels)!=len(calibration) or labels.is_complex()
                or (spec.loss=='cross_entropy' and labels.dtype not in (torch.int32,torch.int64))):
            raise MethodProfileError('FWSVD requires explicit compatible observed calibration labels')
    if type(spec) is Bolaco and (not isinstance(group_labels,torch.Tensor)
            or group_labels.device.type!='cpu' or group_labels.ndim!=1
            or group_labels.dtype not in (torch.int32,torch.int64)):
        raise MethodProfileError('Bolaco requires explicit integral calibration group IDs')
    import psutil
    rss = psutil.Process(os.getpid()).memory_info().rss
    model_bytes = topology.unique_storage_bytes + sum(b.numel()*b.element_size() for b in model.buffers())
    copy_count = 5 if type(spec) is SVDLLMV5 else (3 if type(spec) in (SVDLLMV1, FWSVD) else 2)
    reserve = rss + 64*1024**2 + copy_count*model_bytes + calibration.numel()*calibration.element_size()
    for step in plan.steps:
        m, n = step.operator.shape
        required = reserve + 8 * (8*n*n + 8*m*n + 4*m*m)
        if required > max_peak_bytes or 8 * (8*n*n + 8*m*n + 4*m*m) > max_workspace_bytes:
            raise MethodProfileError("ResourceLimitExceeded: method collection/solve lower bound")
    if not _finite_observations(calibration):
        raise MethodProfileError("Nonfinite calibration data")
    if type(spec) is EoRA:
        if not isinstance(base_model, nn.Module):
            raise MethodProfileError("EoRA requires an explicit independent dense base model")
        _check_data(base_model, calibration, batch_size, max_workspace_bytes, max_peak_bytes)
        base_paths, source_paths = module_paths(base_model), module_paths(model)
        if set(base_paths) != set(source_paths) or any(
                type(base_paths[path]) is not type(source_paths[path]) for path in source_paths):
            raise MethodProfileError("EoRA base and reference must have the same declared dense graph")
    if type(spec) is SVDLLMV5 and not callable(recovery_executor):
        raise MethodProfileError("v5 requires an explicit training executor and separate train role")
    if parameter_budget is not None:
        _positive(parameter_budget, "parameter_budget")
    checkpoint_id, data_id = _fingerprint(model, calibration, graph_version)
    if labels is not None:
        data_id = hashlib.sha256((data_id + _tensor_id(labels)).encode()).hexdigest()
    if group_labels is not None:
        data_id = hashlib.sha256((data_id + _tensor_id(group_labels)).encode()).hexdigest()
    work = _explicit_encoder_graph(deepcopy(base_model if type(spec) is EoRA else model))
    original_modes = {path: layer.training for path, layer in module_paths(work).items()}
    work.eval()
    source_work = _explicit_encoder_graph(deepcopy(model)).eval() if type(spec) is SVDLLMV1 else None
    records, snapshots = [], []
    with torch.random.fork_rng(devices=[]):
        selected = tuple(step.operator.path for step in plan.steps)
        if type(spec) in (DRONE, SVDLLMV1):
            _sequential_order(work, selected, calibration, batch_size)
        if type(spec) is BasisSharing:
            return _transform_basis(model, work, calibration, spec, plan,
                batch_size, max_workspace_bytes, max_peak_bytes, reserve,
                graph_version, checkpoint_id, data_id, original_modes, started, parameter_budget,tensor_byte_budget)
        for index, step in enumerate(plan.steps):
            path = step.operator.path
            layer = module_paths(work)[path]
            current_version = f"{graph_version}:accepted={index}"
            current_id, _ = _fingerprint(work, calibration, current_version)
            passport = StatisticsPassport(checkpoint_id, current_version, current_id,
                                           path + ":input", data_id)
            rows = _collect_rows(work, layer, calibration, batch_size=batch_size,
                workspace=max_workspace_bytes, peak=max_peak_bytes, reserve=reserve)
            initialization_rows = None
            if source_work is not None:
                initialization_rows = _collect_rows(source_work, module_paths(source_work)[path],
                    calibration, batch_size=batch_size,
                    workspace=max_workspace_bytes - 2*rows.numel()*rows.element_size(),
                    peak=max_peak_bytes, reserve=reserve+2*rows.numel()*rows.element_size())
            weight, bias = _matrix(layer), layer.bias
            author_gram = None
            with torch.no_grad():
                fisher = None
                if type(spec) is FWSVD:
                    fisher = _collect_fisher(work, path, calibration, labels, spec, passport,
                                              max_workspace_bytes, max_peak_bytes)
                if type(spec) in (ASVD, FWSVD, AFM, Bolaco, FLARSVD):
                    answer = _statistical_solve(spec, weight, bias, rows, passport,
                        step.rank, group_labels, fisher)
                    replacement = _install(layer, answer)
                else:
                    author_gram = (_author_eora_gram(work, layer, calibration, batch_size)
                        if type(spec) is EoRA and spec.gram_update_version=='author_fixed_n_v1' else None)
                    answer, replacement = _structured_solve(model, work, path, layer,
                        spec, weight, bias, rows, passport, step.rank, recovery_executor,
                        initialization_rows, author_gram)
            snapshot = hashlib.sha256((current_id + _tensor_id(rows) +
                json.dumps(method_payload(spec), sort_keys=True) +
                ('' if author_gram is None else _tensor_id(author_gram[0])+
                 json.dumps(author_gram[1],sort_keys=True))).encode()).hexdigest()
            snapshots.append(SnapshotRef(snapshot, checkpoint_id, current_version))
            work = replace_modules_atomically(work, {path: replacement})
            work.eval()
            records.append({"path": path, "requested_rank": step.rank,
                "graph_version": current_version, "predecessor_version": current_id,
                "snapshot_id": snapshot, "statistics": asdict(passport),
                "observations": len(rows), "numerics": _numerics(answer)})
            del rows, answer, replacement, initialization_rows
        if type(spec) is SVDLLMV5:
            work, stages = _recover_v5(work, selected, spec, recovery_executor)
            records.append({'factor_recovery_stages': stages})
    for path, layer in module_paths(work).items():
        if path in original_modes:
            layer.training = original_modes[path]
    plan = replace(plan, request=replace(plan.request, statistics_refs=tuple(snapshots)))
    return _finish(model, work, spec, plan, records, checkpoint_id, data_id,
        graph_version, started, reserve, max_peak_bytes, max_workspace_bytes, parameter_budget,tensor_byte_budget)


def _finish(original, model, spec, plan, records, checkpoint_id, data_id,
            graph_version, started, estimate, peak, workspace, parameter_budget,tensor_byte_budget=None):
    before, after = _storage_records(original), _storage_records(model)
    actual = exact_unique_cost(after)
    if parameter_budget is not None and actual > parameter_budget:
        raise MethodProfileError(f"BudgetInfeasible: actual cost {actual} exceeds {parameter_budget}")
    actual_bytes = _tensor_state_bytes(model)
    if tensor_byte_budget is not None and actual_bytes > tensor_byte_budget:
        raise MethodProfileError(f'BudgetInfeasible: tensor bytes {actual_bytes} exceed {tensor_byte_budget}')
    report = {"method": method_name(spec), "method_version": 1,
        "method_spec": method_payload(spec), "checkpoint_id": checkpoint_id,
        "data_id": data_id, "data_role": "calibration", "graph_version": graph_version,
        "collection_order": "current_graph_after_each_accepted_replacement",
        "layers": records, "parameters_before": exact_unique_cost(before),
        "parameters_after": actual, "parameter_bytes_before": exact_unique_cost(before, unit="bytes"),
        "parameter_bytes_after": exact_unique_cost(after, unit="bytes"),
        "tensor_bytes_before": _tensor_state_bytes(original),
        "tensor_bytes_after": _tensor_state_bytes(model),
        "parameter_budget": parameter_budget, "total_seconds": time.perf_counter()-started,
        "tensor_byte_budget": tensor_byte_budget,
        "resources": {"estimated_floor_bytes": estimate, "max_peak_bytes": peak,
            "max_workspace_bytes": workspace,
            "scope": "named buffers and bounded observations; arbitrary forward RSS is not a hard bound"},
        "optimizer_rebuild_required": True}
    json.dumps(report, allow_nan=False)
    model._fedcore_method_evidence = report
    model._fedcore_requires_optimizer_rebuild = True
    if plan is not None:
        model._fedcore_transform_plan = plan
    return MethodTransformResult(model, plan, report)


def _tensor_state_bytes(model):
    storages = {}
    for tensor in (*model.parameters(), *model.buffers()):
        storage = tensor.untyped_storage()
        storages[(str(tensor.device), storage._cdata)] = storage.nbytes()
    return sum(storages.values())


def _recover_v5(model, paths, spec, executor):
    from .factor_recovery import prepare_factor_lora_stage, finish_factor_lora_stage
    stages = []
    for side, epochs in (('left', spec.left_epochs), ('right', spec.right_epochs)):
        stage = prepare_factor_lora_stage(model, paths, factor=side,
            rank=spec.adapter_rank, alpha=spec.adapter_alpha)
        frozen = {name: parameter.detach().clone() for name, parameter in stage.model.named_parameters()
                  if not parameter.requires_grad}
        with torch.enable_grad():
            training = executor(stage.model, side, epochs, stage)
        if not isinstance(training, dict):
            raise MethodProfileError('Factor recovery executor must return JSON-safe training evidence')
        for name, parameter in stage.model.named_parameters():
            if name in frozen and not torch.equal(parameter.detach(), frozen[name]):
                raise MethodProfileError('Recovery executor modified a frozen parameter')
        model, evidence = finish_factor_lora_stage(stage, merge=True)
        stages.append({'stage':side,'training':training,'adapter':evidence})
    return model, stages


def _collect_fisher(model, path, inputs, labels, spec, passport, workspace, peak):
    from .statistical_collectors import collect_empirical_gradient_squares
    from .statistical_profiles import GradientCollectionMetadata
    if (not isinstance(labels, torch.Tensor) or labels.device.type != 'cpu'
            or labels.ndim == 0 or len(labels) != len(inputs)):
        raise MethodProfileError('FWSVD requires separate observed calibration labels')
    if labels.is_floating_point() and not bool(torch.isfinite(labels).all()):
        raise MethodProfileError('Calibration labels must be finite')
    import psutil
    known_bytes = inspect_topology(model).unique_storage_bytes + 4*_matrix(module_paths(model)[path]).numel()*8
    if known_bytes > workspace or psutil.Process(os.getpid()).memory_info().rss + known_bytes + 64*1024**2 > peak:
        raise MethodProfileError('ResourceLimitExceeded: private Fisher copy and gradient accumulators')
    metadata = GradientCollectionMetadata(spec.loss + ':observed_labels',
        _tensor_id(labels), reduction=spec.reduction, model_mode='eval')
    def loss_fn(output, target):
        if spec.loss == 'cross_entropy':
            if target.dtype not in (torch.int32, torch.int64):
                raise MethodProfileError('Cross entropy calibration labels must be integral')
            return nn.functional.cross_entropy(output, target.long(), reduction='none')
        if output.shape != target.shape:
            raise MethodProfileError('MSE calibration labels must match the model output')
        return (output - target.to(output)).square()
    # The collector owns its private model copy and one example graph at a time.
    with torch.enable_grad():
        result = collect_empirical_gradient_squares(model, (path,),
            ((inputs, labels),), loss_fn, passport, metadata,
            max_examples=spec.max_examples, max_batches=1)
    return result[path]


def _structured_solve(reference, model, path, layer, spec, weight, bias, rows,
                      passport, rank, recovery_executor, initialization_rows=None, author_gram=None):
    from .approximation import solve_weighted
    from .structured_profiles import (balanced_factors, drone_factors, eora_factors,
        svdllm_v1_fit, svdllm_v2_factors, mixed_rank_metric, FactorPair)
    from .structured_layers import FactorizedLinear, ResidualLinear
    moment = _moment(rows, passport)
    if type(spec) is DRONE:
        pair = drone_factors(weight, rows.T, rank, rcond=spec.rcond)
        pair = replace(pair, diagnostics={**pair.diagnostics,
            'loss_growth_tolerance': spec.loss_growth_tolerance,
            'tolerance_scope': 'explicit local policy; no independence guarantee'})
        return pair, FactorizedLinear.from_factors(pair, bias)
    if type(spec) is SVDLLMV1:
        initial_moment = _moment(initialization_rows, passport) if initialization_rows is not None else moment
        initial = balanced_factors(solve_weighted(weight, initial_moment, rank))
        pair = svdllm_v1_fit(weight, initial.right, rows, rcond=spec.rcond, ridge=spec.ridge)
        pair = replace(pair, diagnostics={**pair.diagnostics,
            'initialization_input': 'original_reference_graph',
            'fit_input': 'current_graph_after_predecessor_replacements'})
        return pair, FactorizedLinear.from_factors(pair, bias)
    if type(spec) is SVDLLMV2:
        pair = svdllm_v2_factors(weight, moment.matrix, rank)
        return pair, FactorizedLinear.from_factors(pair, bias)
    if type(spec) is EoRA:
        target = _matrix(module_paths(reference)[path])
        metric = moment.matrix if author_gram is None else author_gram[0]
        pair = eora_factors(target, weight, metric, rank)
        pair = replace(pair, diagnostics={**pair.diagnostics,
            'base_kind': spec.base_kind, 'gram_update_version': spec.gram_update_version,
            'gram_interpretation': 'uncentered_observation_second_moment',
            'base_operator_id': _tensor_id(weight),
            **({} if author_gram is None else author_gram[1])})
        return pair, ResidualLinear(layer, pair, freeze_base=True)
    if type(spec) is MixedRank:
        metric, diagnostics = mixed_rank_metric(weight, rows,
            objective=spec.metric, zero_policy=spec.zero_policy)
        pair = balanced_factors(solve_weighted(weight, metric, rank))
        base = FactorizedLinear.from_factors(pair, bias)
        if spec.residual_rank > min(weight.shape):
            raise MethodProfileError('Residual rank exceeds the Linear dimensions')
        result = ResidualLinear.for_training(base, spec.residual_rank, seed=spec.seed)
        result.validate_trainable_initialization()
        pair = replace(pair, diagnostics={**pair.diagnostics, **diagnostics,
            'main_rank': rank, 'residual_rank': spec.residual_rank,
            'trainable_initialization': 'random_right_zero_left',
            'base_frozen': True, 'bias_count': 1})
        return pair, result
    pair = balanced_factors(solve_weighted(weight, moment, rank), method='svdllm_v5_initialization')
    return pair, FactorizedLinear.from_factors(pair, bias)


def _plan_coupled(model, spec, rank, target_paths, peak, rank_ratio, parameter_fraction, ranks):
    if not isinstance(model, nn.Module):
        raise MethodProfileError('A torch Module is required')
    if any(t.device.type != 'cpu' for t in (*model.parameters(), *model.buffers())):
        raise MethodProfileError('Coupled P2 profiles require a CPU graph')
    if not isinstance(target_paths, (tuple, list)) or not target_paths or len(set(target_paths)) != len(target_paths):
        raise MethodProfileError('Coupled profiles require distinct explicit target paths')
    paths = module_paths(model)
    if any(path not in paths for path in target_paths):
        raise MethodProfileError('Unknown coupled target path')
    topology = validate_parameter_topology(model, target_paths)
    if topology.storage_aliases:
        raise TopologyError('Coupled profiles reject storage views before copying')
    if any(value is not None for value in (rank_ratio, parameter_fraction, ranks)):
        raise MethodProfileError('Coupled profiles require explicit integer structures')
    if type(spec) is BasisSharing:
        if len(target_paths) < 2 or any(type(paths[path]) is not nn.Linear for path in target_paths):
            raise MethodProfileError('Basis Sharing requires at least two standard Linear consumers')
        if len({id(paths[path]) for path in target_paths}) != len(target_paths):
            raise MethodProfileError('A common module is already shared, not a group of distinct bases')
        if topology.parameter_aliases:
            raise MethodProfileError('First Basis Sharing profile starts from independent parameters')
        width = paths[target_paths[0]].in_features
        if any(paths[path].in_features != width for path in target_paths):
            raise MethodProfileError('Basis consumers must have the same input axis')
        _positive(rank, 'rank')
        if rank > min(width, sum(paths[path].out_features for path in target_paths)):
            raise MethodProfileError('Shared rank exceeds stacked operator dimensions')
        # The generic planner need only validate shapes/resources; group rank may
        # exceed an individual output dimension, so validate its full local rank.
        base = plan_weighted(model, rank_ratio=1.0, target_paths=target_paths, max_peak_bytes=peak)
        from .plans import FixedRank
        steps = tuple(replace(step, rank=rank, structure=FixedRank(rank)) for step in base.steps)
        return replace(base, steps=steps, request=replace(base.request, objective='shared_input_basis'))
    if rank is not None:
        raise MethodProfileError('GroupReduce ranks are explicitly stored in its method spec')
    if not 1 <= len(target_paths) <= 2 or type(paths[target_paths[0]]) is not nn.Embedding:
        raise MethodProfileError('GroupReduce requires embedding followed by an optional tied Linear head')
    embedding = paths[target_paths[0]]
    if embedding.max_norm is not None or embedding.sparse or embedding.scale_grad_by_freq:
        raise MethodProfileError('First GroupReduce profile excludes norm/sparse/frequency-gradient embedding options')
    if len(target_paths) == 2:
        head = paths[target_paths[1]]
        if type(head) is not nn.Linear or head.weight is not embedding.weight:
            raise MethodProfileError('GroupReduce head must share the original embedding Parameter')
    flat = tuple(token for group in spec.groups for token in group)
    if sorted(flat) != list(range(embedding.num_embeddings)):
        raise MethodProfileError('GroupReduce groups must cover every original vocabulary ID')
    if any(r > min(len(group), embedding.embedding_dim) for group, r in zip(spec.groups, spec.ranks)):
        raise MethodProfileError('Group rank exceeds its vocabulary/feature dimensions')
    return None


def _transform_basis(original, work, calibration, spec, plan, batch_size,
        workspace, peak, reserve, graph_version, checkpoint_id, data_id,
        original_modes, started, parameter_budget,tensor_byte_budget=None):
    from .structured_profiles import basis_sharing_factors
    from .structured_layers import create_shared_basis_consumers
    rows, moments, passports, weights, biases = [], [], [], [], []
    paths = module_paths(work)
    for step in plan.steps:
        layer = paths[step.operator.path]
        passport = StatisticsPassport(checkpoint_id, graph_version, checkpoint_id,
                                      step.operator.path + ':input', data_id)
        values = _collect_rows(work, layer, calibration, batch_size=batch_size,
            workspace=workspace, peak=peak, reserve=reserve+sum(r.numel()*r.element_size() for r in rows))
        rows.append(values)
        moments.append(_moment(values, passport).matrix)
        passports.append(passport)
        weights.append(_matrix(layer))
        biases.append(layer.bias)
    pooled = sum(moments, torch.zeros_like(moments[0])) / len(moments)
    scales = None
    if spec.metric_mode == 'equal_moments':
        if any(not torch.allclose(c, moments[0], rtol=1e-9, atol=1e-12) for c in moments[1:]):
            raise MethodProfileError('Individual moments differ: equal-moment objective is invalid')
        metric = moments[0]
    elif spec.metric_mode == 'proportional_moments':
        scales = spec.proportional_scales
        if len(scales) != len(moments):
            raise MethodProfileError('One metric scale is required per selected consumer')
        metric = moments[0] / scales[0]
        if any(not torch.allclose(c, metric*s, rtol=1e-9, atol=1e-12) for c, s in zip(moments, scales)):
            raise MethodProfileError('Individual moments are not proportional to the declared common metric')
    else:
        metric = pooled
    factors = basis_sharing_factors(weights, metric, plan.steps[0].rank, metric_scales=scales)
    consumers = create_shared_basis_consumers(factors, biases)
    snapshot = hashlib.sha256((_tensor_id(metric)+json.dumps(method_payload(spec),sort_keys=True)).encode()).hexdigest()
    records = []
    for step, weight, moment, consumer, passport in zip(plan.steps, weights, moments, consumers, passports):
        residual = weight.double() - consumer.left.detach().double() @ consumer.right.detach().double()
        direct = float(torch.trace(residual @ moment @ residual.T))
        records.append({'path':step.operator.path,'requested_rank':step.rank,
            'statistics':asdict(passport),'snapshot_id':snapshot,
            'numerics':{**dict(factors.diagnostics),'metric_mode':spec.metric_mode,
                'individual_output_error_squared':direct,
                'exact_individual_objective':spec.metric_mode!='pooled_surrogate'}})
    work = replace_modules_atomically(work,dict(zip((step.operator.path for step in plan.steps),consumers)))
    for path, layer in module_paths(work).items():
        if path in original_modes:
            layer.training=original_modes[path]
    plan=replace(plan,request=replace(plan.request,statistics_refs=(SnapshotRef(snapshot,checkpoint_id,graph_version),)))
    return _finish(original,work,spec,plan,records,checkpoint_id,data_id,graph_version,
                   started,reserve,peak,workspace,parameter_budget,tensor_byte_budget)


def _transform_groupreduce(model, tokens, spec, target_paths, workspace, peak,
                           graph_version, parameter_budget,tensor_byte_budget=None):
    from .structured_profiles import groupreduce_factors, groupreduce_transfer
    from .structured_layers import GroupedEmbedding, GroupedLMHead
    started=time.perf_counter()
    for name,value in (('workspace',workspace),('peak',peak)):
        _positive(value,name)
    if peak>12*1024**3 or workspace>peak:
        raise MethodProfileError('Invalid GroupReduce resource limits')
    if (not isinstance(tokens,torch.Tensor) or tokens.device.type!='cpu'
            or tokens.dtype not in (torch.int32,torch.int64) or not tokens.numel()):
        raise MethodProfileError('GroupReduce calibration requires original integral token IDs')
    topology=inspect_topology(model)
    embedding=module_paths(model)[target_paths[0]]
    if embedding.weight.dtype not in (torch.float32,torch.float64):
        raise MethodProfileError('GroupReduce requires float32/float64 embedding weights')
    if bool(((tokens<0)|(tokens>=embedding.num_embeddings)).any()):
        raise MethodProfileError('Calibration token outside vocabulary')
    import psutil
    reserve=psutil.Process(os.getpid()).memory_info().rss+64*1024**2+2*topology.unique_storage_bytes
    required=embedding.weight.numel()*8*6+embedding.num_embeddings*8*6
    if required>workspace or reserve+required>peak:
        raise MethodProfileError('ResourceLimitExceeded: grouped embedding solve')
    checkpoint_id,data_id=_fingerprint(model,tokens,graph_version)
    frequencies=torch.bincount(tokens.reshape(-1).long(),minlength=embedding.num_embeddings).double()
    group_ids=torch.empty(embedding.num_embeddings,dtype=torch.long)
    for group,ids in enumerate(spec.groups):
        group_ids[list(ids)]=group
    factors=groupreduce_factors(embedding.weight,frequencies,group_ids,spec.ranks,zero_policy=spec.zero_policy)
    if spec.transfer_steps:
        factors=groupreduce_transfer(embedding.weight,frequencies,factors,
            max_transfers=spec.transfer_steps,parameter_budget=factors.parameter_elements)
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        work=deepcopy(model)
        paths=module_paths(work)
        table=GroupedEmbedding.from_factors(factors,padding_idx=embedding.padding_idx)
        table.training=embedding.training
        replacements={target_paths[0]:table}
        if len(target_paths)==2:
            head=paths[target_paths[1]]
            replacements[target_paths[1]]=GroupedLMHead(table,head.bias)
        result=replace_modules_atomically(work,replacements)
    records=[{'path':target_paths[0],'numerics':dict(factors.diagnostics),
        'vocabulary_map_bytes':factors.map_bytes,'frequency_sum':float(frequencies.sum()),
        'calibration_token_count':tokens.numel(),'tied_head':len(target_paths)==2}]
    return _finish(model,result,spec,None,records,checkpoint_id,data_id,graph_version,
        started,reserve+required,peak,workspace,parameter_budget,tensor_byte_budget)
