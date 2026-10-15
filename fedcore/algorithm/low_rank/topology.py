"""Inspect module references, Parameter identity and storage without changing a model."""
from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Mapping
import torch
from torch import nn


class TopologyError(ValueError):
    """A transformation cannot preserve the model's declared sharing."""


@dataclass(frozen=True)
class StorageAliases:
    parameter_paths: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    strides: tuple[tuple[int, ...], ...]
    offsets: tuple[int, ...]


@dataclass(frozen=True)
class ModelTopology:
    module_aliases: tuple[tuple[str, ...], ...]
    parameter_aliases: tuple[tuple[str, ...], ...]
    storage_aliases: tuple[StorageAliases, ...]
    unique_parameter_numel: int
    unique_storage_bytes: int


def module_paths(model):
    """Include every reference path; reject cycles rather than recursing forever."""
    paths = {}
    def visit(module, path, ancestors):
        if id(module) in ancestors:
            raise TopologyError(f'Cyclic module reference at {path!r}')
        paths[path] = module
        for name, child in module._modules.items():
            if child is not None:
                visit(child, f'{path}.{name}' if path else name, ancestors | {id(module)})
    visit(model, '', set())
    return paths


def parameter_paths(model):
    return {f'{path}.{name}' if path else name: parameter
            for path, module in module_paths(model).items()
            for name, parameter in module._parameters.items() if parameter is not None}


def inspect_topology(model):
    modules, parameters = module_paths(model), parameter_paths(model)
    module_groups, parameter_groups, storage_groups = defaultdict(list), defaultdict(list), defaultdict(list)
    unique_parameters, storages = {}, {}
    for path, module in modules.items():
        module_groups[id(module)].append(path)
    for path, parameter in parameters.items():
        parameter_groups[id(parameter)].append(path)
        unique_parameters[id(parameter)] = parameter
        if parameter.device.type == 'meta' or parameter.layout != torch.strided:
            raise TopologyError(f'Unsupported Parameter storage at {path!r}')
        storage = parameter.untyped_storage()
        identity = (str(parameter.device), storage._cdata)
        storage_groups[identity].append(path)
        storages[identity] = storage.nbytes()
    storage_aliases = []
    for paths in storage_groups.values():
        # A Parameter repeated through module aliases is not a storage view.
        if len({id(parameters[path]) for path in paths}) > 1:
            storage_aliases.append(StorageAliases(tuple(paths),
                tuple(tuple(parameters[path].shape) for path in paths),
                tuple(tuple(parameters[path].stride()) for path in paths),
                tuple(parameters[path].storage_offset() for path in paths)))
    return ModelTopology(tuple(tuple(g) for g in module_groups.values() if len(g) > 1),
        tuple(tuple(g) for g in parameter_groups.values() if len(g) > 1), tuple(storage_aliases),
        sum(p.numel() for p in unique_parameters.values()), sum(storages.values()))


def validate_parameter_topology(model, selected_paths):
    """Reject storage views and partially transformed ties before preparing replacements."""
    topology, modules = inspect_topology(model), module_paths(model)
    selected = set(selected_paths)
    if not selected <= modules.keys():
        raise TopologyError('Unknown selected module path')
    selected_ids = {id(modules[path]) for path in selected}
    roots = {path for path, module in modules.items() if id(module) in selected_ids}
    def affected(parameter_path):
        owner = parameter_path.rpartition('.')[0]
        return any(root == '' or owner == root or owner.startswith(root + '.') for root in roots)
    for aliases in topology.storage_aliases:
        if any(affected(path) for path in aliases.parameter_paths):
            raise TopologyError(f'Shared storage views are unsupported: {aliases.parameter_paths}')
    for aliases in topology.parameter_aliases:
        if any(affected(path) for path in aliases) and not all(affected(path) for path in aliases):
            raise TopologyError(f'All modules sharing a Parameter must be transformed together: {aliases}')
    return topology


def replace_modules_atomically(model, replacements: Mapping[str, nn.Module]):
    """Validate the complete write set, then replace every alias of each selected module."""
    paths = module_paths(model)
    prepared = {}
    for path, replacement in replacements.items():
        if path not in paths or not isinstance(replacement, nn.Module):
            raise TopologyError(f'Invalid replacement at {path!r}')
        identity = id(paths[path])
        if identity in prepared and prepared[identity] is not replacement:
            raise TopologyError(f'Conflicting module alias replacements at {path!r}')
        prepared[identity] = replacement
    writes = [(path, prepared[id(module)]) for path, module in paths.items() if id(module) in prepared]
    for path, _ in writes:
        if any(path != other and (path == '' or other.startswith(path + '.')) for other, _ in writes):
            raise TopologyError('Ancestor and descendant replacements cannot be applied together')
    for path, replacement in writes:
        replacement_ids = {id(module) for module in module_paths(replacement).values()}
        ancestors = {id(module) for ancestor, module in paths.items()
                     if path and (ancestor == '' or path.startswith(ancestor + '.'))}
        if replacement_ids & ancestors:
            raise TopologyError(f'Replacement would create a module cycle at {path!r}')
    if not writes:
        return model
    # All lookups and validation precede mutation. Direct _modules writes cannot invoke user setters.
    assignments = [(paths[path.rpartition('.')[0]], path.rpartition('.')[2], replacement)
                   for path, replacement in writes if path]
    result = prepared.get(id(model), model)
    for parent, name, replacement in assignments:
        parent._modules[name] = replacement
    result._fedcore_requires_optimizer_rebuild = True
    return result
