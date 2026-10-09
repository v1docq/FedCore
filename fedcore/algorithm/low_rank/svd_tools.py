"""Atomic module decomposition and restoration of the actual stored representation."""
from copy import deepcopy
from collections.abc import Mapping
import torch
from torch import nn
from fedcore.models.network_impl.decomposed_layers import IDecomposed, DecomposableLayers
from fedcore.algorithm.low_rank.topology import (
    TopologyError, module_paths, validate_parameter_topology,
    replace_modules_atomically, inspect_topology,
)


def _map_decomposed_cls(inst):
    return None if isinstance(inst, IDecomposed) else DecomposableLayers.get(type(inst))


def decompose_module(model, decomposing_mode=True, decomposer='svd', compose_mode=None, decomposer_params=None):
    paths = module_paths(model)
    selected = {path: module for path, module in paths.items() if _map_decomposed_cls(module) is not None}
    topology = validate_parameter_topology(model, selected)
    from fedcore.algorithm.low_rank.plans import OperatorDescriptor, legacy_request, plan_transform, ValidationFailure
    solver_name = decomposer if isinstance(decomposer, str) else type(decomposer).__module__ + '.' + type(decomposer).__qualname__
    parameters = dict(decomposer_params or {})
    requested_rank = parameters.get('rank', getattr(decomposer, 'rank', None))
    request = legacy_request(rank=requested_rank, solver=solver_name, decomposing_mode=decomposing_mode,
                             representation=compose_mode or 'three_layers')
    if isinstance(request, ValidationFailure):
        raise ValueError('; '.join(violation.message for violation in request.violations))
    descriptors = tuple(OperatorDescriptor(path, tuple(module.weight.shape), type(module).__name__,
        str(module.weight.dtype).removeprefix('torch.'), getattr(module, 'groups', 1),
        tuple(getattr(module, 'kernel_size', ())), tuple(getattr(module, 'stride', ())),
        getattr(module, 'padding', ()), tuple(getattr(module, 'dilation', ())),
        module.bias.numel() if getattr(module, 'bias', None) is not None else 0,
        padding_mode=getattr(module, 'padding_mode', 'zeros')) for path, module in selected.items())
    plan = plan_transform(request, descriptors)
    if isinstance(plan, ValidationFailure):
        raise ValueError('; '.join(violation.message for violation in plan.violations))
    planned_ranks = {step.operator.path: step.rank for step in plan.steps}
    selected_ids = {id(m) for m in selected.values()}
    for aliases in topology.parameter_aliases:
        affected = [(paths[path.rpartition('.')[0]], path.rpartition('.')[2]) for path in aliases
                    if id(paths[path.rpartition('.')[0]]) in selected_ids]
        if affected and (len({name for _, name in affected}) != 1 or
                         len({type(module) for module, name in affected if name == 'weight'}) > 1):
            raise TopologyError(f'Unsupported tied parameter transformation: {aliases}')
    prepared, by_identity = {}, {}
    for path, module in selected.items():
        if id(module) not in by_identity:
            layer_parameters = dict(parameters)
            if requested_rank is not None and isinstance(decomposer, str):
                layer_parameters['rank'] = planned_ranks[path]
            by_identity[id(module)] = _map_decomposed_cls(module)(module, decomposing_mode=decomposing_mode,
                decomposer=decomposer, compose_mode=compose_mode, decomposer_params=layer_parameters or None)
            from fedcore.algorithm.low_rank.decomposer import DECOMPOSERS
            replacement = by_identity[id(module)]
            if (decomposing_mode not in (False, None) and requested_rank is not None and
                    ((isinstance(decomposer, str) and decomposer == 'svd') or type(decomposer) is DECOMPOSERS['svd'])):
                # tdecomp 0.2.18 computes the full SVD even when rank is supplied.
                # Execute the validated structure decision at this boundary.
                rank = planned_ranks[path]
                was_dense = replacement.U is None
                if was_dense:
                    replacement.canonicalize()
                u, s, vh = replacement.get_U_S_Vh()
                if u.shape[-1] != rank:
                    replacement.set_U_S_Vh(u[..., :rank], s[..., :rank], vh[..., :rank, :])
                if was_dense:
                    replacement.compose()
        prepared[path] = by_identity[id(module)]
    for aliases in topology.parameter_aliases:
        transformed = []
        for parameter_path in aliases:
            path, _, name = parameter_path.rpartition('.')
            if path in prepared:
                transformed.append((prepared[path], name))
        if not transformed:
            continue
        first, name = transformed[0]
        names = ('weight', 'U', 'S', 'Vh') if name == 'weight' else (name,)
        for replacement, _ in transformed[1:]:
            for transformed_name in names:
                source = first._parameters[transformed_name]
                target = replacement._parameters[transformed_name]
                if (source is None) != (target is None) or (source is not None and source.shape != target.shape):
                    raise TopologyError(f'Incompatible tied factor shapes: {aliases}')
                replacement.register_parameter(transformed_name, source)
        if name == 'weight' and len({id(replacement) for replacement, _ in transformed}) > 1:
            for replacement, _ in transformed:
                replacement._fedcore_tied_factors = True
    result = replace_modules_atomically(model, prepared)
    result._fedcore_transform_plan = plan
    return result


def restore_svd_representation(module, values, metadata=None):
    """Validate a layer completely before installing any restored Parameters."""
    if not isinstance(values, Mapping) or any(not isinstance(value, torch.Tensor) for value in values.values()):
        raise ValueError('SVD layer state must contain tensors')
    available = set(values) & {'weight', 'U', 'S', 'Vh'}
    forms = {frozenset({'weight'}): 'one_layer', frozenset({'U', 'Vh'}): 'two_layers',
             frozenset({'U', 'S', 'Vh'}): 'three_layers'}
    representation = forms.get(frozenset(available))
    if representation is None:
        raise ValueError('Checkpoint must contain a dense weight, U/Vh, or U/S/Vh')
    if metadata is not None:
        if not isinstance(metadata, Mapping) or type(metadata.get('version')) is not int or metadata['version'] != 1:
            raise ValueError('Unknown SVD representation version')
        if metadata.get('layer_type') != type(module).__name__ or metadata.get('representation') != representation:
            raise ValueError('SVD representation does not match stored parameters')
        if metadata.get('compose_mode') not in (None, 'one_layer', 'two_layers', 'three_layers'):
            raise ValueError('Invalid checkpoint compose mode')
        if type(metadata.get('inference_mode')) is not bool:
            raise ValueError('Invalid checkpoint inference mode')
        from fedcore.algorithm.low_rank.decomposer import DECOMPOSERS
        if metadata.get('solver') not in DECOMPOSERS or not isinstance(metadata.get('decomposer_params'), Mapping):
            raise ValueError('Unknown checkpoint solver')
        if metadata.get('bias') != (getattr(module, 'bias', None) is not None) or metadata.get('groups') != getattr(module, 'groups', 1):
            raise ValueError('Checkpoint bias or groups do not match the layer')
        if metadata.get('shapes') != {name: list(value.shape) for name, value in values.items()}:
            raise ValueError('Checkpoint tensor shapes do not match representation metadata')
    expected_weight = module._get_composed_weight()
    expected_matrix = module._weight_to_matrix(expected_weight)
    if representation == 'one_layer':
        if values['weight'].shape != expected_weight.shape:
            raise ValueError('Invalid dense weight shape')
    else:
        u, vh = values['U'], values['Vh']
        batch, rows, columns = expected_matrix.shape[:-2], expected_matrix.shape[-2], expected_matrix.shape[-1]
        if (u.ndim != expected_matrix.ndim or vh.ndim != expected_matrix.ndim or
                u.shape[:-2] != batch or vh.shape[:-2] != batch or u.shape[-2] != rows or
                vh.shape[-1] != columns or u.shape[-1] != vh.shape[-2] or not 1 <= u.shape[-1] <= min(rows, columns)):
            raise ValueError('Invalid U/Vh factor shapes')
        if representation == 'three_layers' and values['S'].shape != batch + (u.shape[-1],):
            raise ValueError('Invalid S factor shape')
        if any(value.dtype != u.dtype or value.device != u.device for name, value in values.items() if name != 'bias'):
            raise ValueError('Factors must have the same dtype and device')
    if (getattr(module, 'bias', None) is not None) != ('bias' in values) or ('bias' in values and values['bias'].shape != module.bias.shape):
        raise ValueError('Invalid checkpoint bias')
    for name in ('weight', 'U', 'S', 'Vh'):
        value = values.get(name)
        module.register_parameter(name, None if value is None else nn.Parameter(
            value.detach().clone(), requires_grad=module._weight_requires_grad))
    module._representation = representation
    module.inference_mode = metadata['inference_mode'] if metadata is not None else representation != 'three_layers'
    module.compose_mode = metadata['compose_mode'] if metadata is not None else representation
    if metadata is not None:
        module.method = metadata['solver']
        module.decomposer_params = dict(metadata['decomposer_params'])
        module.decomposing_mode = metadata['decomposing_mode']


def _load_svd_params(model, state_dict, prefix=''):
    metadata = getattr(state_dict, '_metadata', {})
    if not isinstance(metadata, Mapping) or any(not isinstance(record, Mapping) for record in metadata.values()):
        raise ValueError('Invalid state dictionary metadata')
    for name, module in model.named_modules():
        if isinstance(module, IDecomposed):
            key = prefix + (name + '.' if name else '')
            values = {field: state_dict[key + field] for field in ('weight', 'U', 'S', 'Vh', 'bias') if key + field in state_dict}
            restore_svd_representation(module, values, metadata.get(name, {}).get('fedcore_svd'))


def validate_tied_state(model, state):
    """A checkpoint cannot silently give conflicting values to one Parameter."""
    topology = inspect_topology(model)
    if topology.storage_aliases:
        raise TopologyError('Restoration of shared storage views is unsupported')
    for aliases in topology.parameter_aliases:
        present = [state[path] for path in aliases if path in state]
        if present and any(value.dtype != present[0].dtype or value.shape != present[0].shape or
                           not torch.equal(value, present[0]) for value in present[1:]):
            raise TopologyError(f'Conflicting checkpoint values for tied Parameters: {aliases}')
    return topology


def retie_parameters(model, aliases):
    paths = module_paths(model)
    for group in aliases:
        first_path, _, first_name = group[0].rpartition('.')
        parameter = paths[first_path]._parameters[first_name]
        for path in group[1:]:
            owner, _, name = path.rpartition('.')
            paths[owner].register_parameter(name, parameter)
        owners = {id(paths[path.rpartition('.')[0]]) for path in group}
        if len(owners) > 1:
            for path in group:
                owner, _, name = path.rpartition('.')
                if isinstance(paths[owner], IDecomposed) and name in ('weight', 'U', 'S', 'Vh'):
                    paths[owner]._fedcore_tied_factors = True


def load_svd_state_dict(model, decomposing_mode, state_dict_path, compose_mode=None, decomposer_params=None):
    state_dict = torch.load(state_dict_path, map_location='cpu', weights_only=True)
    if not isinstance(state_dict, Mapping):
        raise ValueError('Expected a weights-only state dictionary')
    if inspect_topology(model).storage_aliases:
        raise TopologyError('SVD restoration does not support shared storage views anywhere in the model')
    candidate = deepcopy(model)
    metadata = getattr(state_dict, '_metadata', {})
    if not isinstance(metadata, Mapping) or any(not isinstance(record, Mapping) for record in metadata.values()):
        raise ValueError('Invalid state dictionary metadata')
    modes = set()
    for record in metadata.values():
        representation = record.get('fedcore_svd')
        if isinstance(representation, Mapping) and representation.get('layer_type') == 'DecomposedConv2d':
            saved_mode = representation.get('decomposing_mode')
            if saved_mode not in (True, False, None, 'channel', 'spatial'):
                raise ValueError('Invalid checkpoint convolution mode')
            modes.add(saved_mode)
    if len(modes) > 1:
        raise ValueError('Mixed convolution modes require a CheckpointManager architecture descriptor')
    mode = next(iter(modes), decomposing_mode)
    candidate = decompose_module(candidate, False, compose_mode=compose_mode, decomposer_params=decomposer_params)
    if mode == 'spatial':
        for layer in candidate.modules():
            if type(layer).__name__ == 'DecomposedConv2d':
                layer._spatial = True
                layer.decomposing_mode = 'spatial'
    source_topology = inspect_topology(candidate)
    stored_aliases = []
    for group in source_topology.parameter_aliases:
        local_names = {path.rpartition('.')[2] for path in group}
        if local_names == {'weight'}:
            fields = ('weight', 'U', 'S', 'Vh')
            for field in fields:
                aliases = tuple((path.rpartition('.')[0] + '.' if path.rpartition('.')[0] else '') + field for path in group)
                present = tuple(path for path in aliases if path in state_dict)
                if present:
                    if len(present) != len(aliases):
                        raise TopologyError('Tied weights have incompatible checkpoint representations')
                    stored_aliases.append(aliases)
        else:
            stored_aliases.append(group)
    for group in stored_aliases:
        values = [state_dict[path] for path in group if path in state_dict]
        if values and any(value.dtype != values[0].dtype or not torch.equal(value, values[0]) for value in values[1:]):
            raise TopologyError(f'Conflicting checkpoint values for tied Parameters: {group}')
    _load_svd_params(candidate, state_dict)
    retie_parameters(candidate, stored_aliases)
    topology = validate_tied_state(candidate, state_dict)
    candidate.load_state_dict(state_dict, strict=True)
    retie_parameters(candidate, topology.parameter_aliases)
    candidate._fedcore_requires_optimizer_rebuild = True
    return candidate


__all__ = ['decompose_module', 'load_svd_state_dict']
decompose_module_in_place = decompose_module
