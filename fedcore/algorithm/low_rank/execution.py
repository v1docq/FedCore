"""One effectful CPU interpreter for the checked weighted-SVD profile.

Statistics and decisions live in pure collaborators. This shell owns copies,
hooks and tensor execution. All input moments refer to the same original graph;
this is not the sequential correction algorithm of SVD-LLM/DRONE.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import time
import os
import torch
from torch import nn
from torch.nn import functional as F

from .allocation import StorageCost, exact_unique_cost
from .approximation import solve_weighted
from .plans import (FixedRank, RankFraction, ParameterFraction, Goal, MetricPolicy,
                    OperatorDescriptor, ResourceLimits, SnapshotRef, TransformPlan,
                    TransformRequest, ValidationFailure, plan_transform)
from .statistics import (StatisticsPassport, ResourceLimitExceeded,
                         empty_second_moment, update_second_moment,
                         finish_second_moment, plan_statistics_memory)
from .topology import (TopologyError, inspect_topology, module_paths,
                       parameter_paths, validate_parameter_topology,
                       replace_modules_atomically)


class WeightedProfileError(ValueError):
    """A runtime capability or declared resource limit was not satisfied."""


@dataclass(frozen=True)
class WeightedTransformResult:
    model: nn.Module
    plan: TransformPlan
    evidence: dict


def _hash_tensor(digest, tensor):
    # Bounded host copies; neither a live tensor nor a pathname is a snapshot ID.
    digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
    value = tensor.detach()
    rows = max(1, (1024 * 1024) // max(1, value[0].numel() * value.element_size())) if value.ndim and len(value) else 1
    for batch in value.split(rows) if value.ndim else (value,):
        digest.update(batch.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())


def _fingerprint(model, calibration, graph_version):
    state = hashlib.sha256(graph_version.encode())
    def config_value(value):
        if value is None or type(value) in (str, int, float, bool):
            return value
        if isinstance(value, (tuple, list)):
            converted = [config_value(item) for item in value]
            if all(item is not NotImplemented for item in converted):
                return converted
        return NotImplemented
    for path, module in module_paths(model).items():
        state.update(str((path, type(module).__module__, type(module).__name__,
                          getattr(module, 'stride', None), getattr(module, 'padding', None),
                          getattr(module, 'dilation', None), getattr(module, 'groups', None),
                          getattr(module, 'padding_mode', None),
                          getattr(module, 'decomposing_mode', None))).encode())
        # Activation slopes, normalization eps, dimensions, etc. need not be in
        # state_dict. Hash data-valued public settings without calling repr on
        # arbitrary runtime objects. Custom hidden/callable state is covered by
        # the caller's declared graph_version, not inferred by this interpreter.
        config = {name: config_value(value) for name, value in vars(module).items()
                  if not name.startswith('_') and name != 'training'}
        config = {name: value for name, value in config.items() if value is not NotImplemented}
        state.update(json.dumps(config, sort_keys=True, allow_nan=False).encode())
    topology = inspect_topology(model)
    state.update(str((topology.module_aliases, topology.parameter_aliases)).encode())
    for name, value in model.state_dict().items():
        if not isinstance(value, torch.Tensor):
            raise WeightedProfileError('Tensor-only state is required by the weighted profile')
        state.update(name.encode())
        _hash_tensor(state, value)
    data = hashlib.sha256()
    _hash_tensor(data, calibration)
    return state.hexdigest(), data.hexdigest()


def _storage_records(model):
    records, seen = [], set()
    for path, parameter in parameter_paths(model).items():
        if id(parameter) not in seen:
            records.append(StorageCost(path, parameter.numel(), parameter.element_size(), 'parameter'))
            seen.add(id(parameter))
    return tuple(records)


def _finite_observations(value):
    rows = max(1, (1024 * 1024) // max(1, value[0].numel() * value.element_size()))
    return all(bool(torch.isfinite(batch).all()) for batch in value.split(rows))


def _descriptor(path, layer):
    from fedcore.models.network_impl.decomposed_layers import IDecomposed
    kind = next(name for cls, name in ((nn.Linear, 'Linear'), (nn.Conv1d, 'Conv1d'),
                                      (nn.Conv2d, 'Conv2d')) if isinstance(layer, cls))
    parameter = next(layer.parameters())
    if isinstance(layer, IDecomposed):
        if layer.decomposing_mode not in (True, 'channel'):
            raise WeightedProfileError('Weighted profile supports channel decomposition only')
        shape = ((layer.out_features, layer.in_features) if kind == 'Linear' else
                 (layer.out_channels, layer.in_channels // layer.groups, *layer.kernel_size))
    else:
        shape = tuple(layer.weight.shape)
    return OperatorDescriptor(path, tuple(shape), kind, str(parameter.dtype).removeprefix('torch.'),
        getattr(layer, 'groups', 1), tuple(getattr(layer, 'kernel_size', ())),
        tuple(getattr(layer, 'stride', ())), getattr(layer, 'padding', ()),
        tuple(getattr(layer, 'dilation', ())),
        layer.bias.numel() if layer.bias is not None else 0, path + ':operator',
        padding_mode=getattr(layer, 'padding_mode', 'zeros'))


def _patch_rows(layer, value, workspace):
    """Yield group-local rows with the exact numeric convolution geometry."""
    if isinstance(layer, nn.Linear):
        if value.shape[-1] != layer.in_features:
            raise WeightedProfileError('Linear observation width does not match the operator')
        yield (value.reshape(-1, layer.in_features),)
        return
    dimensions = 1 if isinstance(layer, nn.Conv1d) else 2
    if value.ndim != dimensions + 2:
        raise WeightedProfileError('Convolution calibration requires a batched tensor')
    spatial = value.shape[2:]
    locations = math.prod((spatial[i] + 2 * layer.padding[i] - layer.dilation[i] *
                           (layer.kernel_size[i] - 1) - 1) // layer.stride[i] + 1
                          for i in range(dimensions))
    width = layer.in_channels * math.prod(layer.kernel_size)
    # unfold plus transposed contiguous rows can coexist. Refuse before unfold.
    patch_bytes = 2 * locations * width * value.element_size()
    if locations <= 0 or patch_bytes > workspace:
        raise WeightedProfileError('ResourceLimitExceeded: one convolution sample exceeds patch workspace')
    for sample in value.split(1):
        padding = layer.padding
        if layer.padding_mode != 'zeros':
            sample = F.pad(sample, layer._reversed_padding_repeated_twice, mode=layer.padding_mode)
            padding = (0,) * dimensions
        if dimensions == 1:
            sample = sample.unsqueeze(2)
            kernel, stride, dilation, padding = ((1, layer.kernel_size[0]), (1, layer.stride[0]),
                                                (1, layer.dilation[0]), (0, padding[0]))
        else:
            kernel, stride, dilation = layer.kernel_size, layer.stride, layer.dilation
        patches = F.unfold(sample, kernel, dilation=dilation, padding=padding, stride=stride)
        # [1, groups * local_width, locations] -> separate group row matrices.
        grouped = patches.reshape(layer.groups, width // layer.groups, locations)
        yield tuple(grouped[group].T for group in range(layer.groups))


def plan_weighted(model, *, rank=None, rank_ratio=None, parameter_fraction=None,
                  target_paths=None, policy=MetricPolicy(), representation='two_layers',
                  max_peak_bytes=12 * 1024**3):
    """Read capabilities and topology without copying or executing a model."""
    from fedcore.models.network_impl.decomposed_layers import (
        DecomposedLinear, DecomposedConv1d, DecomposedConv2d)
    supported = (nn.Linear, nn.Conv1d, nn.Conv2d, DecomposedLinear, DecomposedConv1d, DecomposedConv2d)
    if not isinstance(model, nn.Module):
        raise WeightedProfileError('A torch Module is required')
    if any(t.device.type != 'cpu' for t in (*model.parameters(), *model.buffers())):
        raise WeightedProfileError('This weighted runtime profile requires a CPU model')
    choices = [rank is not None, rank_ratio is not None, parameter_fraction is not None]
    if sum(choices) > 1:
        raise WeightedProfileError('Rank, rank fraction and parameter fraction are alternative policies')
    structure = (FixedRank(rank) if rank is not None else ParameterFraction(parameter_fraction)
                 if parameter_fraction is not None else RankFraction(1.0 if rank_ratio is None else rank_ratio))
    paths = module_paths(model)
    if target_paths is not None and not isinstance(target_paths, (tuple, list)):
        raise WeightedProfileError('target_paths must be an explicit sequence of module paths')
    requested = tuple(path for path, layer in paths.items() if type(layer) in supported) if target_paths is None else tuple(target_paths)
    if not requested or any(not isinstance(path, str) or path not in paths or type(paths[path]) not in supported for path in requested):
        raise WeightedProfileError('Select supported Linear/Conv1d/Conv2d paths')
    selected, identities = [], set()
    for path in requested:
        if id(paths[path]) not in identities:
            selected.append(path)
            identities.add(id(paths[path]))
    topology = validate_parameter_topology(model, selected)
    if topology.storage_aliases:
        raise TopologyError('Weighted profile does not support shared storage views')
    for aliases in topology.parameter_aliases:
        owners = [paths[path.rpartition('.')[0]] for path in aliases]
        if any(id(owner) in identities for owner in owners) and len({id(owner) for owner in owners}) > 1:
            raise TopologyError('Weighted profile requires an explicit pooled objective for tied modules')
    descriptors = tuple(_descriptor(path, paths[path]) for path in selected)
    request = TransformRequest(Goal.REPLACE_OPERATOR, 'input_second_moment', 'svd', structure,
                               representation, 'channel', policy)
    plan = plan_transform(request, descriptors, ResourceLimits(max_peak_bytes))
    if isinstance(plan, ValidationFailure):
        raise WeightedProfileError('; '.join(f'{v.code}: {v.path}: {v.message}' for v in plan.violations))
    return plan


def transform_weighted(model, calibration, *, rank=None, rank_ratio=None,
                       parameter_fraction=None, target_paths=None, policy=MetricPolicy(),
                       representation='two_layers', batch_size=16,
                       max_workspace_bytes=256 * 1024**2, max_peak_bytes=12 * 1024**3,
                       graph_version='weighted-svd-v1'):
    """Return an independent transformed graph and JSON-safe numerical evidence.

    Only CPU float32/float64 Linear and channel Conv1d/Conv2d are executable.
    Module aliases are preserved. Distinct modules with a shared Parameter are
    refused: their pooled calibration objective is a later explicit profile.
    Memory limits bound named live buffers, not arbitrary user-forward RSS.
    Calibration is a separate tensor role, never a test/validation fallback.
    """
    from fedcore.models.network_impl.decomposed_layers import (
        DecomposedLinear, DecomposedConv1d, DecomposedConv2d, IDecomposed)
    constructors = {nn.Linear: DecomposedLinear, nn.Conv1d: DecomposedConv1d,
                    nn.Conv2d: DecomposedConv2d, DecomposedLinear: DecomposedLinear,
                    DecomposedConv1d: DecomposedConv1d, DecomposedConv2d: DecomposedConv2d}
    started = time.perf_counter()
    if not isinstance(model, nn.Module):
        raise WeightedProfileError('A torch Module is required')
    if (not isinstance(calibration, torch.Tensor) or calibration.device.type != 'cpu'
            or calibration.dtype not in (torch.float32, torch.float64) or calibration.ndim < 2
            or calibration.shape[0] == 0 or calibration.numel() == 0):
        raise WeightedProfileError('Calibration requires a nonempty finite CPU float32/float64 tensor')
    if any(type(v) is not int or v <= 0 for v in (batch_size, max_workspace_bytes, max_peak_bytes)):
        raise WeightedProfileError('Batch size and resource limits must be positive integers')
    if max_peak_bytes > 12 * 1024**3 or max_workspace_bytes > max_peak_bytes:
        raise WeightedProfileError('Weighted profile permits at most 12 GiB of declared buffers')
    if calibration[0].numel() * calibration.element_size() > max_workspace_bytes:
        raise WeightedProfileError('ResourceLimitExceeded: one calibration sample exceeds workspace')
    if not isinstance(graph_version, str) or not graph_version:
        raise WeightedProfileError('Explicit graph version is required')
    plan = plan_weighted(model, rank=rank, rank_ratio=rank_ratio, parameter_fraction=parameter_fraction,
        target_paths=target_paths, policy=policy, representation=representation, max_peak_bytes=max_peak_bytes)
    request = plan.request
    descriptors = tuple(step.operator for step in plan.steps)
    topology = inspect_topology(model)
    # Refuse Gram/SVD/copy lower bounds before deepcopy or registering hooks.
    model_bytes = topology.unique_storage_bytes + sum(b.numel() * b.element_size() for b in model.buffers())
    input_bytes = calibration.numel() * calibration.element_size()
    # Account for the already loaded runtime, separately from model buffers.
    # This snapshot is a conservative planning floor, not a measured peak.
    import psutil
    rss_before = psutil.Process(os.getpid()).memory_info().rss
    overhead = rss_before + 64 * 1024**2
    for descriptor in descriptors:
        out = descriptor.shape[0] // descriptor.groups
        width = math.prod(descriptor.shape[1:])
        preflight = plan_statistics_memory(width, 1, groups=descriptor.groups, operator_rows=out,
            model_bytes=2 * model_bytes + input_bytes, runtime_overhead_bytes=overhead,
            max_peak_bytes=max_peak_bytes)
        if isinstance(preflight, ResourceLimitExceeded):
            raise WeightedProfileError(f'ResourceLimitExceeded: requires {preflight.required_bytes} bytes')
        if 3 * descriptor.groups * width * width * 8 > max_workspace_bytes:
            raise WeightedProfileError('ResourceLimitExceeded: Gram accumulation exceeds workspace')
    if not _finite_observations(calibration):
        raise WeightedProfileError('Calibration requires finite observations')
    checkpoint_id, data_id = _fingerprint(model, calibration, graph_version)
    timings = {'planning_seconds': time.perf_counter() - started}
    stage = time.perf_counter()
    work = deepcopy(model)
    work_paths = module_paths(work)
    original_modes = {path: layer.training for path, layer in work_paths.items()}
    work.eval()
    timings['copy_seconds'] = time.perf_counter() - stage
    prepared, evidence, snapshots = {}, [], []
    prepared_bytes, forwards, max_estimate = 0, 0, 0
    collect_seconds = solve_seconds = 0.0
    # Constructors initialize tensors before overwriting them; preserve caller RNG.
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        for step in plan.steps:
            path, descriptor = step.operator.path, step.operator
            layer = work_paths[path]
            out, width = descriptor.shape[0] // descriptor.groups, math.prod(descriptor.shape[1:])
            passports = tuple(StatisticsPassport(checkpoint_id, graph_version, checkpoint_id,
                path + f':input:group={g}', data_id) for g in range(descriptor.groups))
            states = [empty_second_moment(width, passport) for passport in passports]
            layer_peak = 0
            def capture(_module, args):
                nonlocal states, layer_peak, max_estimate
                if len(args) != 1 or not isinstance(args[0], torch.Tensor):
                    raise WeightedProfileError('Weighted calibration supports one tensor input per layer')
                value = args[0].detach()
                if value.device.type != 'cpu' or value.dtype not in (torch.float32, torch.float64):
                    raise WeightedProfileError('Observed activations must be CPU float32/float64')
                # Conservative full batch patch reserve, checked before unfold.
                if isinstance(layer, nn.Linear):
                    patch_reserve = value.numel() * value.element_size()
                else:
                    locations = math.prod((value.shape[i + 2] + 2 * layer.padding[i] - layer.dilation[i] *
                        (layer.kernel_size[i] - 1) - 1) // layer.stride[i] + 1 for i in range(value.ndim - 2))
                    patch_reserve = 2 * max(0, locations) * width * descriptor.groups * value.element_size()
                rows_per_chunk = max(1, min(1024, max_workspace_bytes // max(1, 4 * width * descriptor.groups * 8)))
                memory = plan_statistics_memory(width, rows_per_chunk, groups=descriptor.groups, operator_rows=out,
                    model_bytes=2 * model_bytes + input_bytes, replacement_bytes=prepared_bytes,
                    forward_bytes=2 * value.numel() * value.element_size() + patch_reserve,
                    runtime_overhead_bytes=overhead, max_peak_bytes=max_peak_bytes)
                if isinstance(memory, ResourceLimitExceeded):
                    raise WeightedProfileError(f'ResourceLimitExceeded: requires {memory.required_bytes} bytes')
                layer_peak = max(layer_peak, memory.peak_bytes)
                max_estimate = max(max_estimate, memory.peak_bytes)
                for grouped in _patch_rows(layer, value, max_workspace_bytes):
                    for group, rows in enumerate(grouped):
                        for chunk in rows.split(rows_per_chunk):
                            states[group] = update_second_moment(states[group], chunk)
            stage = time.perf_counter()
            handle = layer.register_forward_pre_hook(capture)
            try:
                for batch in calibration.split(batch_size):
                    work(batch)
                    forwards += 1
            finally:
                handle.remove()
            moments = tuple(finish_second_moment(state) for state in states)
            del states
            collect_seconds += time.perf_counter() - stage
            stage = time.perf_counter()
            # A factorized input is approximated from its actual current operator.
            if isinstance(layer, IDecomposed):
                matrix = layer.factor_matrix().detach()
                from .reassembly.decomposed_recreation import to_standard_module
                base = to_standard_module(layer)
            else:
                base = layer
                matrix = layer.weight.detach()
                if not isinstance(layer, nn.Linear):
                    matrix = matrix.reshape(descriptor.groups, out, width)
                    if descriptor.groups == 1:
                        matrix = matrix[0]
            matrices = matrix.unbind(0) if matrix.ndim == 3 else (matrix,)
            answers = tuple(solve_weighted(weight, moment, step.rank, policy) for weight, moment in zip(matrices, moments))
            constructor = constructors[type(layer)]
            result_layer = constructor(base, decomposing_mode=False, compose_mode=representation)
            result_layer.decomposing_mode = True if isinstance(layer, nn.Linear) else 'channel'
            factors = tuple(torch.stack([getattr(a, name) for a in answers]) for name in ('u', 's', 'vh')) if matrix.ndim == 3 else (answers[0].u, answers[0].s, answers[0].vh)
            result_layer.set_U_S_Vh(*factors)
            result_layer.compose_weight_for_inference()
            result_layer.training = original_modes[path]
            prepared[path] = result_layer
            prepared_bytes += sum(p.numel() * p.element_size() for p in result_layer.parameters())
            solve_seconds += time.perf_counter() - stage
            passport_records = [asdict(moment.passport) for moment in moments]
            snapshot_hash = hashlib.sha256(json.dumps(passport_records, sort_keys=True).encode())
            for moment in moments:
                _hash_tensor(snapshot_hash, moment.matrix)
            snapshot = snapshot_hash.hexdigest()
            snapshots.append(SnapshotRef(snapshot, checkpoint_id, graph_version))
            evidence.append({'path': path, 'kind': descriptor.kind, 'shape': list(descriptor.shape),
                'groups': descriptor.groups, 'requested_rank': step.rank,
                'actual_ranks': [a.actual_rank for a in answers], 'representation': representation,
                'statistics': passport_records, 'snapshot_id': snapshot, 'estimated_peak_bytes': layer_peak,
                'numerics': [{name: getattr(answer, name) for name in answer.__dataclass_fields__
                             if name not in ('u', 's', 'vh', 'approximation')} for answer in answers]})
            del moments, answers, factors, matrix, matrices, base, result_layer
    result = replace_modules_atomically(work, prepared)
    for path, layer in module_paths(result).items():
        if path in original_modes:
            layer.training = original_modes[path]
    plan = replace(plan, request=replace(request, statistics_refs=tuple(snapshots)))
    result._fedcore_transform_plan = plan
    before_records, after_records = _storage_records(model), _storage_records(result)
    timings.update(collection_seconds=collect_seconds, solve_and_assembly_seconds=solve_seconds,
                   total_seconds=time.perf_counter() - started)
    report = {'method': 'weighted_svd', 'method_version': 1, 'objective': 'input_second_moment',
        'policy': asdict(policy), 'checkpoint_id': checkpoint_id, 'data_id': data_id,
        'graph_version': graph_version, 'layers': evidence, 'calibration_forward_calls': forwards,
        'parameters_before': exact_unique_cost(before_records), 'parameters_after': exact_unique_cost(after_records),
        'tensor_bytes_before': exact_unique_cost(before_records, unit='bytes'),
        'tensor_bytes_after': exact_unique_cost(after_records, unit='bytes'), 'timings': timings,
        'resources': {'estimated_peak_bytes': max_estimate, 'limit_bytes': max_peak_bytes,
            'process_rss_before_bytes': rss_before, 'runtime_reserve_bytes': 64 * 1024**2,
            'workspace_bytes': max_workspace_bytes,
            'scope': 'named live buffers; arbitrary model-forward/allocator RSS is not a hard bound'},
        'optimizer_rebuild_required': True}
    return WeightedTransformResult(result, plan, report)
