"""Trusted local checkpoints with explicit architecture restoration.

PyTorch's weights-only loader is not a security boundary on supported Torch
2.2. Never use this module for files supplied through an untrusted service.
"""
import gc
import io
import os
from copy import deepcopy
from collections import OrderedDict
from collections.abc import Mapping
import torch
from torch import nn


class CheckpointError(ValueError):
    """Checkpoint metadata or weights do not satisfy the restoration contract."""


def _clone_state_value(value):
    """Copy only the types accepted by the weights-only serialization contract."""
    if isinstance(value, torch.Tensor):
        if value.is_quantized:
            result = {'__fedcore_state__': 'quantized_tensor', 'int_repr': value.int_repr().cpu(),
                      'qscheme': str(value.qscheme())}
            if value.qscheme() in (torch.per_tensor_affine, torch.per_tensor_symmetric):
                result.update(scale=value.q_scale(), zero_point=value.q_zero_point())
            else:
                result.update(scales=value.q_per_channel_scales().cpu(),
                              zero_points=value.q_per_channel_zero_points().cpu(),
                              axis=value.q_per_channel_axis())
            return result
        return value.detach().cpu().clone()
    if isinstance(value, torch.dtype):
        return {'__fedcore_state__': 'dtype', 'name': str(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return type(value)(_clone_state_value(item) for item in value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise CheckpointError('State mappings require string keys')
        copied = OrderedDict() if isinstance(value, OrderedDict) else {}
        copied.update((key, _clone_state_value(item)) for key, item in value.items())
        if hasattr(value, '_metadata'):
            copied._metadata = _clone_state_value(value._metadata)
        return copied
    raise CheckpointError(f'Unsupported state value: {type(value).__name__}')


def _decode_state_value(value):
    if isinstance(value, Mapping):
        kind = value.get('__fedcore_state__')
        if kind == 'dtype':
            dtypes = {str(dtype): dtype for dtype in (torch.qint8, torch.quint8, torch.qint32,
                torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.int8, torch.uint8, torch.int32, torch.int64)}
            if value.get('name') not in dtypes:
                raise CheckpointError('Unsupported encoded dtype')
            return dtypes[value['name']]
        if kind == 'quantized_tensor':
            if value['qscheme'] in ('torch.per_tensor_affine', 'torch.per_tensor_symmetric'):
                return torch._make_per_tensor_quantized_tensor(value['int_repr'], value['scale'], value['zero_point'])
            if value['qscheme'] in ('torch.per_channel_affine', 'torch.per_channel_symmetric', 'torch.per_channel_affine_float_qparams'):
                return torch._make_per_channel_quantized_tensor(value['int_repr'], value['scales'], value['zero_points'], value['axis'])
            raise CheckpointError('Unsupported encoded quantization scheme')
        if kind is not None:
            raise CheckpointError('Unknown encoded state value')
        copied = OrderedDict() if isinstance(value, OrderedDict) else {}
        copied.update((key, _decode_state_value(item)) for key, item in value.items())
        if hasattr(value, '_metadata'):
            copied._metadata = _decode_state_value(value._metadata)
        return copied
    if isinstance(value, (tuple, list)):
        return type(value)(_decode_state_value(item) for item in value)
    return value


def _same_state_shape(actual, expected):
    if isinstance(expected, torch.Tensor):
        return isinstance(actual, torch.Tensor) and actual.shape == expected.shape
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and set(actual) == set(expected) and all(
            _same_state_shape(actual[key], expected[key]) for key in expected)
    if isinstance(expected, (tuple, list)):
        return type(actual) is type(expected) and len(actual) == len(expected) and all(
            _same_state_shape(a, b) for a, b in zip(actual, expected))
    return type(actual) is type(expected)


def describe_model(model):
    _precheck_cv_checkpoint(model)
    from fedcore.algorithm.pruning.sparse.checkpoint import validate_sparse_storage
    try:
        validate_sparse_storage(model)
    except ValueError as error:
        raise CheckpointError(str(error)) from error
    _validate_remapped_state(model,model.state_dict())
    return _describe_model(model, {}, '')


def _precheck_cv_checkpoint(model):
    """CV is a root-only profile; inspect raw registries before tensor effects.

    In particular, do not let generic topology/state_dict/deepcopy traversal
    reach Tensor overrides or a CV model hidden in an ordinary module container.
    Other checkpoint profiles retain their existing architecture constructors.
    """
    from fedcore.algorithm.pruning.sparse.cv_channels import CifarResNet18, validate_cv_model
    pending, seen = [(model, '')], set()
    while pending:
        layer, path = pending.pop()
        if not isinstance(layer, nn.Module):
            raise CheckpointError('Checkpoint child registry contains a non-module object')
        if id(layer) in seen:
            continue
        seen.add(id(layer))
        if issubclass(type(layer), CifarResNet18):
            if type(layer) is not CifarResNet18 or path:
                raise CheckpointError('The exact CIFAR CV architecture is supported only as the checkpoint root')
            try:
                validate_cv_model(layer)
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                raise CheckpointError('Invalid declared CV architecture before checkpoint effects') from error
            return
        values = object.__getattribute__(layer, '__dict__')
        if type(values) is not dict:
            raise CheckpointError('Checkpoint modules require an ordinary instance dictionary')
        children = values.get('_modules')
        if type(children) not in (dict, OrderedDict):
            raise CheckpointError('Checkpoint child registries require an exact builtin mapping')
        for name, child in children.items():
            if type(name) is not str:
                raise CheckpointError('Checkpoint module names must be strings')
            if child is not None:
                pending.append((child, path + '.' + name if path else name))


def _validate_remapped_state(model,state):
    """Private packed parts cannot share any nonempty model-state storage.

    Validate both the live constructor after topology restoration and incoming
    tensors before assign=True. Generic ties not touching this format remain
    governed by the existing topology contract.
    """
    from fedcore.algorithm.low_rank.dobi_remapping import RemappedLinear
    from fedcore.algorithm.low_rank.topology import module_paths
    packed_keys = set()
    for path,layer in module_paths(model).items():
        if type(layer) is not RemappedLinear:
            continue
        prefix = path+'.' if path else ''
        if any(name in layer.__dict__ for name in ('forward','decoded_factors',
                'representation_config','validate_state','state_dict','_save_to_state_dict',
                '_load_from_state_dict')):
            raise CheckpointError('RemappedLinear checkpoint requires unmodified representation methods')
        try:
            RemappedLinear.validate_state(layer,state,prefix)
        except (ValueError,TypeError,KeyError,RuntimeError) as error:
            raise CheckpointError('Invalid RemappedLinear state') from error
        names = ('left_codes','right_codes','left_scale','right_scale','tail')
        packed_keys.update(prefix+name for name in names)
        if layer.bias is not None:
            packed_keys.add(prefix+'bias')
    if not packed_keys:
        return
    groups = {}
    for name,value in state.items():
        if not isinstance(value,torch.Tensor):
            continue
        storage = value.untyped_storage()
        if storage.nbytes():
            groups.setdefault((str(value.device),storage._cdata),[]).append(name)
    if any(len(names)>1 and packed_keys.intersection(names) for names in groups.values()):
        raise CheckpointError('RemappedLinear components cannot alias any other model state')


def _buffer_paths(model):
    from fedcore.algorithm.low_rank.topology import module_paths
    return {f'{path}.{name}' if path else name: tensor
            for path, module in module_paths(model).items()
            for name, tensor in module._buffers.items() if tensor is not None}


def _buffer_aliases(model):
    groups = {}
    for path, value in _buffer_paths(model).items():
        groups.setdefault(id(value), []).append(path)
    return tuple(tuple(group) for group in groups.values() if len(group)>1)


def _retie_buffers(model, groups):
    from fedcore.algorithm.low_rank.topology import module_paths
    paths, buffers = module_paths(model), _buffer_paths(model)
    for group in groups:
        source = buffers[group[0]]
        for path in group[1:]:
            owner, _, name = path.rpartition('.')
            paths[owner]._buffers[name] = source


def _validate_tied_buffers(model, state):
    groups = _buffer_aliases(model)
    for group in groups:
        values = [state.get(path) for path in group]
        if any(not isinstance(value, torch.Tensor) for value in values) or any(
                value.shape!=values[0].shape or value.dtype!=values[0].dtype
                or not torch.equal(value,values[0]) for value in values[1:]):
            raise CheckpointError('Conflicting state values for a shared buffer')
    return groups


def _validate_p3_state(model, state):
    """Validate fixed graph coordinates and projection invariants before assign."""
    from fedcore.algorithm.low_rank.coupled_layers import ReferenceAttention, ReferenceMLP
    from fedcore.algorithm.low_rank.projection_layers import (FixedProjectorLinear,
        CenteredPCAEmbedding, CTRLinearSum, CTRFieldGraph, ReLUFactorizedLinear)
    from fedcore.algorithm.low_rank.topology import module_paths
    from fedcore.algorithm.low_rank.learned_layers import ZeroLinear
    known = (ReferenceAttention,ReferenceMLP,FixedProjectorLinear,CenteredPCAEmbedding,
             CTRLinearSum,CTRFieldGraph,ReLUFactorizedLinear,ZeroLinear)
    for path, layer in module_paths(model).items():
        if type(layer) not in known:
            continue
        prefix = path+'.' if path else ''
        if type(layer) is ZeroLinear and state[prefix+'_anchor'].dtype != layer._anchor.dtype:
            raise CheckpointError('ZeroLinear state conflicts with its declared dtype')
        for name, expected in layer.named_parameters():
            value = state[prefix+name]
            if value.dtype != expected.dtype or not bool(torch.isfinite(value).all()):
                raise CheckpointError('P3 parameters require finite values and the declared dtype')
        if type(layer) is ReferenceAttention:
            for name in ('head_map','group_coordinates'):
                value, expected = state[prefix+name], getattr(layer,name)
                if value.dtype != expected.dtype or not torch.equal(value.cpu(),expected.cpu()):
                    raise CheckpointError('P3 attention state conflicts with declared immutable coordinates')
        if type(layer) not in (FixedProjectorLinear,CenteredPCAEmbedding):
            continue
        name = 'projection' if type(layer) is FixedProjectorLinear else 'basis'
        basis = state[prefix+name]
        expected = getattr(layer,name)
        if basis.dtype != expected.dtype or not bool(torch.isfinite(basis).all()):
            raise CheckpointError('P3 fixed projection requires finite values and the declared dtype')
        tolerance = 1e-5 if basis.dtype == torch.float32 else 1e-10
        gram = basis.double().T @ basis.double()
        if not torch.allclose(gram,torch.eye(gram.shape[0],dtype=gram.dtype,device=gram.device),
                              rtol=tolerance,atol=tolerance):
            raise CheckpointError('P3 fixed projection must remain orthonormal')
        if type(layer) is CenteredPCAEmbedding:
            mean, correction = state[prefix+'mean'], state[prefix+'affine_correction']
            if any(value.dtype != basis.dtype or not bool(torch.isfinite(value).all())
                   for value in (mean,correction)):
                raise CheckpointError('CTR affine state requires finite values and the declared dtype')
            expected_correction = mean-basis@(basis.T@mean)
            if not torch.allclose(correction,expected_correction,rtol=tolerance,atol=tolerance):
                raise CheckpointError('CTR affine correction conflicts with its basis and mean')


def _describe_model(model, seen, path):
    """Only encode constructors from a fixed allowlist; never arbitrary Python."""
    if id(model) in seen:
        return {'type': 'ModuleAlias', 'target': seen[id(model)]}
    seen[id(model)] = path
    from fedcore.algorithm.quantization.checkpoint import describe_quantized
    try:
        quantized_description = describe_quantized(model)
    except (ValueError, TypeError, KeyError) as error:
        raise CheckpointError('Invalid quantization graph representation') from error
    if quantized_description is not None:
        return quantized_description
    from fedcore.algorithm.pruning.sparse.cv_checkpoint import describe_cv
    try:
        cv_description = describe_cv(model)
    except (ValueError, TypeError, KeyError) as error:
        raise CheckpointError('Invalid declared CV architecture') from error
    if cv_description is not None:
        return cv_description
    from fedcore.algorithm.pruning.sparse.checkpoint import describe_sparse
    try:
        sparse_description = describe_sparse(model)
    except (TypeError, ValueError, KeyError) as error:
        raise CheckpointError('Invalid declared sparse representation') from error
    if sparse_description is not None:
        return sparse_description
    from fedcore.algorithm.pruning.sparse.moe import ReferenceMoE
    if type(model) is ReferenceMoE:
        from fedcore.algorithm.pruning.sparse.checkpoint import _guard, _bound
        try:
            _guard(model)
            model.validate_state()
            config = model.representation_config()
            _bound(config)
        except (ValueError, TypeError, KeyError) as error:
            raise CheckpointError('Invalid MoE representation') from error
        children = {name: _describe_model(child, seen, f'{path}.{name}' if path else name)
                    for name, child in model._modules.items() if child is not None}
        return {'type': 'SparseMoE', 'version': 1, 'config': config,
                'dtype': str(model.router.weight.dtype), 'children': children}
    from fedcore.algorithm.low_rank.architecture_checkpoint import describe_p5
    try:
        description = describe_p5(model)
    except (TypeError, ValueError) as error:
        raise CheckpointError('Invalid declared P5 architecture') from error
    if description is not None:
        return description
    from fedcore.algorithm.low_rank.learned_layers import ZeroLinear
    from fedcore.algorithm.low_rank.dobi_remapping import RemappedLinear
    if type(model) is RemappedLinear:
        if (model._modules or set(model._buffers) != {'left_codes','right_codes','left_scale','right_scale','tail'}
                or set(model._parameters) != ({'bias'} if model.bias is not None else set())
                or any(name in model.__dict__ for name in ('forward','decoded_factors',
                    'representation_config','validate_state','state_dict','_save_to_state_dict','_load_from_state_dict'))
                or model._forward_hooks or model._forward_pre_hooks
                or model._backward_hooks or getattr(model,'_backward_pre_hooks',{})):
            raise CheckpointError('RemappedLinear checkpoint requires its unmodified declared operator')
        try:
            return {'type': 'RemappedLinear', 'config': model.representation_config()}
        except (ValueError,TypeError,KeyError) as error:
            raise CheckpointError('Invalid RemappedLinear state') from error
    if type(model) is ZeroLinear:
        return {'type': 'ZeroLinear', 'config': model.representation_config(),
                'dtype': str(model._anchor.dtype)}
    from fedcore.algorithm.low_rank.projection_layers import (FixedProjectorLinear,
        CenteredPCAEmbedding, CTRLinearSum, CTRFieldGraph, ReLUFactorizedLinear)
    if type(model) in (FixedProjectorLinear, CenteredPCAEmbedding, CTRLinearSum,
                       CTRFieldGraph, ReLUFactorizedLinear):
        children = {name: _describe_model(child, seen, f'{path}.{name}' if path else name)
                    for name, child in model._modules.items() if child is not None}
        if not all(children.values()):
            return None
        return {'type': type(model).__name__, 'config': model.representation_config(),
                'dtype': str(next(model.parameters()).dtype), 'children': children}
    from fedcore.algorithm.low_rank.coupled_layers import ReferenceAttention, ReferenceMLP
    if type(model) in (ReferenceAttention, ReferenceMLP):
        children = {name: _describe_model(child, seen, f'{path}.{name}' if path else name)
                    for name, child in model._modules.items() if child is not None}
        if not all(children.values()):
            return None
        from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear
        from fedcore.algorithm.pruning.sparse.layers import PackedNMLinear, IndexedResidualLinear
        compact_projections = (FactorizedLinear, PackedNMLinear, IndexedResidualLinear)
        if type(model) is ReferenceMLP and any(type(child) in compact_projections for child in model.children()):
            up_bias = model.up_proj.left.bias if type(model.up_proj) is FactorizedLinear else model.up_proj.bias
            config = {'layer_type': 'ReferenceMLP', 'version': 1, 'd_model': model.d_model,
                      'intermediate_size': model.intermediate_size, 'gated': model.gated,
                      'activation': model.activation, 'bias': up_bias is not None, 'channel_indices': list(model.channel_indices)}
            return {'type': 'SparseExpertMLP', 'config': config,
                    'dtype': str(next(model.parameters()).dtype), 'children': children}
        return {'type': type(model).__name__, 'config': model.representation_config(),
                'dtype': str(next(model.parameters()).dtype), 'children': children}
    from fedcore.algorithm.low_rank.structured_layers import (FactorizedLinear,
        ResidualLinear, SharedBasisLinear, GroupedEmbedding, GroupedLMHead)
    structured = (FactorizedLinear, ResidualLinear, SharedBasisLinear, GroupedEmbedding, GroupedLMHead)
    if type(model) in structured:
        children = {}
        names = ('left', 'right') if type(model) is FactorizedLinear else (
            ('base', 'residual') if type(model) is ResidualLinear else (
            ('table',) if type(model) is GroupedLMHead else ()))
        for name in names:
            child_path = f'{path}.{name}' if path else name
            children[name] = _describe_model(getattr(model, name), seen, child_path)
            if children[name] is None:
                return None
        return {'type': type(model).__name__, 'config': model.representation_config(),
                'dtype': str(next(model.parameters()).dtype), 'children': children}
    from fedcore.models.network_impl.decomposed_layers import IDecomposed, DecomposableLayers
    from fedcore.models.network_modules.layers.lora import LoRALayer, Linear, Embedding, Conv2d
    if isinstance(model, IDecomposed):
        if type(model) not in DecomposableLayers.values():
            return None
        base_type = next((base for base in (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding)
                          if isinstance(model, base)), None)
        if base_type is None:
            return None
        return {'type': type(model).__name__, 'base': _dense_description(model, base_type),
                'representation': model.representation_metadata(),
                'dtype': str(next(model.parameters()).dtype)}
    if isinstance(model, LoRALayer):
        if type(model) not in (LoRALayer, Linear, Embedding, Conv2d):
            return None
        if model.merged:
            raise CheckpointError('Unmerge LoRA before saving a restorable adapter checkpoint')
        return {'type': 'LoRA', 'base': _describe_model(model.base_layer, seen, path + '.base_layer' if path else 'base_layer'),
                'adapters': [{'name': name, 'r': rank, 'alpha': model.lora_alpha[name],
                              'dropout': getattr(model.lora_dropout[name], 'p', 0.)} for name, rank in model.r.items()],
                'active': model.active_adapters, 'disabled': model.disable_adapters}
    if type(model) in (nn.quantized.Linear, nn.quantized.dynamic.Linear):
        return {'type': 'DynamicQuantizedLinear' if type(model) is nn.quantized.dynamic.Linear else 'QuantizedLinear',
                'args': {'in_features': model.in_features, 'out_features': model.out_features,
                         'bias_': model.bias() is not None, 'dtype': str(model.weight().dtype)}}
    if type(model) is nn.Linear:
        return {'type': 'Linear', 'args': {'in_features': model.in_features,
                'out_features': model.out_features, 'bias': model.bias is not None}}
    if type(model) in (nn.Conv1d, nn.Conv2d, nn.Embedding):
        return _dense_description(model, type(model))
    if type(model) in (nn.Sequential, nn.ModuleDict, nn.ModuleList):
        children = {name: _describe_model(child, seen, f'{path}.{name}' if path else name)
                    for name, child in model._modules.items() if child is not None}
        return {'type': type(model).__name__, 'children': children} if all(children.values()) else None
    if type(model) in (nn.ReLU, nn.Identity, nn.Sigmoid, nn.Tanh, nn.Flatten, nn.Dropout):
        args = {}
        if type(model) is nn.ReLU:
            args = {'inplace': model.inplace}
        elif type(model) is nn.Flatten:
            args = {'start_dim': model.start_dim, 'end_dim': model.end_dim}
        elif type(model) is nn.Dropout:
            args = {'p': model.p, 'inplace': model.inplace}
        return {'type': type(model).__name__, 'args': args}
    return None


def _dense_description(model, base_type):
    if base_type is nn.Linear:
        args = {'in_features': model.in_features, 'out_features': model.out_features, 'bias': model.bias is not None}
    elif base_type in (nn.Conv1d, nn.Conv2d):
        args = {name: getattr(model, name) for name in ('in_channels', 'out_channels', 'kernel_size',
                'stride', 'padding', 'dilation', 'groups', 'padding_mode')}
        args['bias'] = model.bias is not None
    else:
        args = {name: getattr(model, name) for name in ('num_embeddings', 'embedding_dim', 'padding_idx',
                'max_norm', 'norm_type', 'scale_grad_by_freq', 'sparse')}
    return {'type': base_type.__name__, 'args': args}


def build_model(description):
    return _build_model(description, {}, '')


def _build_model(description, seen, path):
    if not isinstance(description, dict):
        raise CheckpointError('Architecture is missing: provide model= or model_factory=')
    if description.get('type') == 'ModuleAlias':
        if set(description) != {'type', 'target'} or description.get('target') not in seen:
            raise CheckpointError('Invalid or forward module alias reference')
        return seen[description['target']]
    if description.get('type') in ('Sequential', 'ModuleDict', 'ModuleList'):
        if not isinstance(description.get('children'), Mapping):
            raise CheckpointError('Sequential architecture requires named children')
        model = {'Sequential': nn.Sequential, 'ModuleDict': nn.ModuleDict,
                 'ModuleList': nn.ModuleList}[description['type']]()
        seen[path] = model
        for name, child in description['children'].items():
            if not isinstance(name, str) or not name or '.' in name:
                raise CheckpointError('Invalid child module name')
            model.add_module(name, _build_model(child, seen, f'{path}.{name}' if path else name))
        return model
    from fedcore.models.network_impl.decomposed_layers import DecomposableLayers
    decomposed = {cls.__name__: cls for cls in DecomposableLayers.values()}
    kind = description.get('type')
    if kind == 'QuantizationGraph':
        from fedcore.algorithm.quantization.checkpoint import build_quantized
        try:
            model = build_quantized(description)
        except (ValueError, TypeError, KeyError) as error:
            raise CheckpointError('Invalid quantization graph descriptor') from error
        seen[path] = model
        return model
    if kind == 'CVArchitecture':
        if path:
            raise CheckpointError('The CIFAR CV checkpoint profile requires an architecture root')
        from fedcore.algorithm.pruning.sparse.cv_checkpoint import build_cv
        try:
            model = build_cv(description)
        except (ValueError, TypeError, KeyError, RuntimeError) as error:
            raise CheckpointError('Invalid CV architecture descriptor') from error
        seen[path] = model
        return model
    if kind == 'SparseMoE':
        from fedcore.algorithm.pruning.sparse.moe import ReferenceMoE
        from fedcore.algorithm.pruning.sparse.checkpoint import _bound
        if (set(description) != {'type', 'version', 'config', 'dtype', 'children'}
                or type(description['version']) is not int or description['version'] != 1
                or description['dtype'] not in ('torch.float32', 'torch.float64')
                or not isinstance(description['children'], dict)):
            raise CheckpointError('Invalid sparse MoE descriptor')
        config, children = description['config'], description['children']
        try:
            _bound(config)
            if set(children) != ({'router', 'experts', 'shared'} if config['shared'] else {'router', 'experts'}):
                raise ValueError('Invalid MoE children')
            with torch.random.fork_rng(devices=[]):
                model = ReferenceMoE.from_config(config, device='cpu', dtype=getattr(torch, description['dtype'][6:]))
            seen[path] = model
            for name, child in children.items():
                model._modules[name] = _build_model(child, seen, f'{path}.{name}' if path else name)
            if type(model.router) is not nn.Linear or type(model.experts) is not nn.ModuleList or len(model.experts) != model.num_experts:
                raise ValueError('Invalid MoE graph')
        except (ValueError, TypeError, KeyError) as error:
            raise CheckpointError('Invalid sparse MoE architecture') from error
        return model
    if kind == 'SparseRepresentation':
        from fedcore.algorithm.pruning.sparse.checkpoint import build_sparse
        try:
            model = build_sparse(description)
        except (ValueError, TypeError, KeyError, RuntimeError) as error:
            raise CheckpointError('Invalid sparse representation descriptor') from error
        seen[path] = model
        return model
    if kind == 'P5Architecture':
        from fedcore.algorithm.low_rank.architecture_checkpoint import build_p5
        try:
            model = build_p5(description)
        except (ValueError, TypeError, KeyError) as error:
            raise CheckpointError('Invalid P5 architecture descriptor') from error
        seen[path] = model
        return model
    if kind == 'RemappedLinear':
        from fedcore.algorithm.low_rank.dobi_remapping import RemappedLinear
        config = description.get('config')
        if (set(description) != {'type','config'} or not isinstance(config,dict)
                or set(config) != {'layer_type','version','in_features','out_features','rank','bias',
                    'remapping','left_dtype','right_dtype','tail_dtype','scale_dtype','compute_dtype','bias_dtype'}
                or config['layer_type'] != kind or type(config['version']) is not int or config['version'] != 1
                or config['remapping'] != 'symmetric_int8_v1'
                or config['left_dtype'] != 'int8' or config['right_dtype'] != 'int8'):
            raise CheckpointError('Invalid RemappedLinear descriptor')
        try:
            model = RemappedLinear(**{key:config[key] for key in ('in_features','out_features',
                'rank','bias','tail_dtype','scale_dtype','compute_dtype','bias_dtype')})
        except (ValueError,TypeError) as error:
            raise CheckpointError('Invalid RemappedLinear constructor') from error
        seen[path] = model
        return model
    if kind == 'ZeroLinear':
        from fedcore.algorithm.low_rank.learned_layers import ZeroLinear
        config = description.get('config')
        if (set(description) != {'type','config','dtype'} or not isinstance(config,dict)
                or set(config) != {'layer_type','version','in_features','out_features','bias'}
                or config['layer_type'] != kind or type(config['version']) is not int or config['version'] != 1
                or description['dtype'] not in ('torch.float32','torch.float64')):
            raise CheckpointError('Invalid ZeroLinear descriptor')
        try:
            model = ZeroLinear(config['in_features'],config['out_features'],config['bias'],
                               dtype=getattr(torch,description['dtype'][6:]))
        except (ValueError,TypeError) as error:
            raise CheckpointError('Invalid ZeroLinear constructor') from error
        seen[path] = model
        return model
    if kind in ('FixedProjectorLinear','CenteredPCAEmbedding','CTRLinearSum',
                'CTRFieldGraph','ReLUFactorizedLinear'):
        from fedcore.algorithm.low_rank.projection_layers import (FixedProjectorLinear,
            CenteredPCAEmbedding, CTRLinearSum, CTRFieldGraph, ReLUFactorizedLinear)
        config, children = description.get('config'), description.get('children')
        if (not isinstance(config, dict) or config.get('layer_type') != kind
                or type(config.get('version')) is not int or config['version'] != 1
                or description.get('dtype') not in ('torch.float32', 'torch.float64')
                or not isinstance(children, dict)):
            raise CheckpointError('Invalid P3 projection representation descriptor')
        keysets = {
            'FixedProjectorLinear': {'in_features','out_features','rank','bias'},
            'CenteredPCAEmbedding': {'num_embeddings','embedding_dim','rank','coefficient_convention','decoder_offset'},
            'CTRLinearSum': {'field_widths','out_features','bias'},
            'CTRFieldGraph': {'interaction_pairs'},
            'ReLUFactorizedLinear': {'in_features','out_features','rank','bias','nonlinear_variant'},
        }
        if set(config) != keysets[kind] | {'layer_type','version'}:
            raise CheckpointError('Unknown or missing P3 projection options')
        dtype = getattr(torch, description['dtype'][6:])
        def p3_child(name):
            return _build_model(children[name], seen, f'{path}.{name}' if path else name)
        try:
            if kind == 'CTRFieldGraph':
                if set(children) != {'fields','linear'}:
                    raise ValueError('CTR graph requires exactly its declared fields and consumers')
                field_tables, linear = p3_child('fields'), p3_child('linear')
                if type(field_tables) is not nn.ModuleDict or type(linear) is not CTRLinearSum:
                    raise ValueError('Invalid CTR graph children')
                model = CTRFieldGraph(field_tables, linear, config['interaction_pairs'])
            else:
                factory = {'FixedProjectorLinear':FixedProjectorLinear,
                    'CenteredPCAEmbedding':CenteredPCAEmbedding,'CTRLinearSum':CTRLinearSum,
                    'ReLUFactorizedLinear':ReLUFactorizedLinear}[kind]
                options = {key:config[key] for key in keysets[kind]
                           if key not in ('nonlinear_variant','coefficient_convention','decoder_offset')}
                if kind == 'CenteredPCAEmbedding' and (config['coefficient_convention'] != 'centered_(E-mean)U'
                        or config['decoder_offset'] != 'mean'):
                    raise ValueError('Unknown CTR coefficient or affine decoding convention')
                if kind == 'ReLUFactorizedLinear' and config['nonlinear_variant'] != 'relu_between_factors_v1':
                    raise ValueError('Unknown nonlinear CTR architecture')
                model = factory(**options,dtype=dtype)
                seen[path] = model
                expected_children = {name:child for name,child in model._modules.items() if child is not None}
                if set(children) != set(expected_children):
                    raise ValueError('Projection module children differ from the declared graph')
                for name, expected in expected_children.items():
                    restored = p3_child(name)
                    if type(expected) is nn.Linear:
                        if (type(restored) is not nn.Linear or restored.weight.shape != expected.weight.shape
                                or (restored.bias is None) != (expected.bias is None)):
                            raise ValueError('Projection factor geometry differs')
                    elif type(expected) is nn.ModuleDict:
                        if (type(restored) is not nn.ModuleDict or set(restored) != set(expected)
                                or any(type(restored[key]) is not nn.Linear
                                    or restored[key].weight.shape != expected[key].weight.shape
                                    or restored[key].bias is not None for key in expected)):
                            raise ValueError('CTR additive consumer geometry differs')
                    model._modules[name] = restored.to(dtype=dtype)
            seen[path] = model
        except (TypeError, ValueError, KeyError) as error:
            raise CheckpointError('Invalid P3 projection representation constructor') from error
        return model
    if kind in ('ReferenceAttention', 'ReferenceMLP', 'SparseExpertMLP'):
        from fedcore.algorithm.low_rank.coupled_layers import ReferenceAttention, ReferenceMLP
        from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear
        from fedcore.algorithm.pruning.sparse.layers import PackedNMLinear, IndexedResidualLinear
        sparse_expert = kind == 'SparseExpertMLP'
        reference_kind = 'ReferenceMLP' if sparse_expert else kind
        config, children = description.get('config'), description.get('children')
        if (not isinstance(config, dict) or config.get('layer_type') != reference_kind
                or type(config.get('version')) is not int or config['version'] != 1
                or description.get('dtype') not in ('torch.float32', 'torch.float64')
                or not isinstance(children, dict)):
            raise CheckpointError('Invalid P3 reference representation descriptor')
        expected_keys = ({'d_model','num_heads','num_kv_heads','head_dim','qk_dim','value_dim',
            'head_map','group_coordinates','rope','rope_base','logits_scale','causal','bias'}
            if kind == 'ReferenceAttention' else
            {'d_model','intermediate_size','gated','activation','bias','channel_indices'})
        if set(config) != expected_keys | {'layer_type','version'}:
            raise CheckpointError('Unknown or missing reference representation options')
        try:
            factory = ReferenceAttention if kind == 'ReferenceAttention' else ReferenceMLP
            model = factory(**{key: config[key] for key in expected_keys},
                            dtype=getattr(torch, description['dtype'][6:]))
            seen[path] = model
            expected_children = {name: child for name,child in model._modules.items() if child is not None}
            if set(children) != set(expected_children):
                raise ValueError('Reference module children must match the explicit graph')
            for name, expected in expected_children.items():
                restored = _build_model(children[name], seen, f'{path}.{name}' if path else name)
                if sparse_expert and type(restored) in (FactorizedLinear, PackedNMLinear, IndexedResidualLinear):
                    bias = restored.left.bias if type(restored) is FactorizedLinear else restored.bias
                    if (restored.in_features != expected.in_features or restored.out_features != expected.out_features
                            or (bias is None) != (expected.bias is None)):
                        raise ValueError('Sparse expert factor dimensions do not match')
                elif (type(restored) is not nn.Linear or restored.weight.shape != expected.weight.shape
                      or (restored.bias is None) != (expected.bias is None)):
                    raise ValueError('Reference graph projection dimensions do not match')
                model._modules[name] = restored.to(dtype=getattr(torch, description['dtype'][6:]))
        except (TypeError, ValueError, KeyError) as error:
            raise CheckpointError('Invalid P3 reference representation constructor') from error
        return model
    structured_names = ('FactorizedLinear', 'ResidualLinear', 'SharedBasisLinear',
                        'GroupedEmbedding', 'GroupedLMHead')
    if kind in structured_names:
        from fedcore.algorithm.low_rank.structured_layers import (FactorizedLinear,
            ResidualLinear, SharedBasisLinear, GroupedEmbedding, GroupedLMHead)
        from fedcore.algorithm.low_rank.structured_profiles import FactorPair
        config, children = description.get('config'), description.get('children')
        if (not isinstance(config, dict) or config.get('layer_type') != kind
                or type(config.get('version')) is not int or config['version'] != 1
                or description.get('dtype') not in ('torch.float32', 'torch.float64')
                or not isinstance(children, dict)):
            raise CheckpointError('Invalid P2 representation descriptor')
        dtype = getattr(torch, description['dtype'][6:])
        def child(name):
            return _build_model(children.get(name), seen, f'{path}.{name}' if path else name)
        try:
            if kind == 'FactorizedLinear':
                model = FactorizedLinear(config['in_features'], config['out_features'],
                    config['rank'], config['bias'], dtype=dtype)
                seen[path] = model
                model.left, model.right = child('left').to(dtype=dtype), child('right').to(dtype=dtype)
            elif kind == 'SharedBasisLinear':
                model = SharedBasisLinear(config['in_features'], config['out_features'],
                    config['rank'], config['bias'], dtype=dtype)
            elif kind == 'ResidualLinear':
                base = child('base')
                pair = FactorPair(torch.zeros(config['out_features'], config['residual_rank'], dtype=dtype),
                                  torch.zeros(config['residual_rank'], config['in_features'], dtype=dtype))
                model = ResidualLinear(base, pair, freeze_base=config['freeze_base'])
                seen[path] = model
                model.residual = child('residual')
            elif kind == 'GroupedEmbedding':
                model = GroupedEmbedding(config['num_embeddings'], config['embedding_dim'],
                    config['group_sizes'], config['ranks'], padding_idx=config['padding_idx'], dtype=dtype)
            else:
                table = child('table')
                if type(table) is not GroupedEmbedding:
                    raise ValueError('A grouped head requires its explicit grouped table')
                bias = torch.zeros(table.num_embeddings, dtype=dtype) if config['bias'] else None
                model = GroupedLMHead(table, bias)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise CheckpointError(f'Invalid P2 representation: {error}') from error
        seen[path] = model
        return model
    if kind in decomposed:
        from fedcore.algorithm.low_rank.svd_tools import restore_svd_representation
        metadata = description.get('representation')
        if not isinstance(metadata, Mapping) or description.get('dtype') not in ('torch.float16', 'torch.bfloat16', 'torch.float32', 'torch.float64'):
            raise CheckpointError('Invalid decomposed layer descriptor')
        base = _build_model(description.get('base'), {}, '').to(dtype=getattr(torch, description['dtype'][6:]))
        model = decomposed[kind](base, decomposing_mode=False)
        if kind == 'DecomposedConv2d':
            model._spatial = metadata.get('decomposing_mode') == 'spatial'
        model.decomposing_mode = metadata.get('decomposing_mode')
        try:
            values = {name: torch.zeros(shape, dtype=base.weight.dtype) for name, shape in metadata['shapes'].items()}
            restore_svd_representation(model, values, metadata)
        except (TypeError, ValueError, KeyError, RuntimeError) as error:
            raise CheckpointError(f'Invalid decomposed representation: {error}') from error
        seen[path] = model
        return model
    if kind == 'LoRA':
        from fedcore.models.network_modules.layers.lora import Linear, Conv2d, Embedding
        base = _build_model(description.get('base'), seen, path + '.base_layer' if path else 'base_layer')
        mapping = {nn.Linear: Linear, nn.Conv2d: Conv2d, nn.Embedding: Embedding}
        adapters = description.get('adapters')
        if type(base) not in mapping or not isinstance(adapters, list) or not adapters:
            raise CheckpointError('Invalid LoRA architecture')
        try:
            first = adapters[0]
            model = mapping[type(base)](base, first['name'], r=first['r'], lora_alpha=first['alpha'], lora_dropout=first['dropout'])
            for adapter in adapters[1:]:
                model.update_layer(adapter['name'], adapter['r'], adapter['alpha'], adapter['dropout'])
            model.set_adapter(description['active'])
            model.enable_adapters(not description['disabled'])
        except (TypeError, ValueError, KeyError) as error:
            raise CheckpointError(f'Invalid LoRA arguments: {error}') from error
        seen[path] = model
        return model
    constructors = {name: getattr(nn, name) for name in
                    ('Linear', 'Conv1d', 'Conv2d', 'Embedding', 'ReLU', 'Identity', 'Sigmoid', 'Tanh', 'Flatten', 'Dropout')}
    constructors.update({'QuantizedLinear': nn.quantized.Linear,
                         'DynamicQuantizedLinear': nn.quantized.dynamic.Linear})
    constructor = constructors.get(description.get('type'))
    if constructor is None:
        raise CheckpointError(f'Unsupported architecture: {description.get("type")!r}')
    try:
        args = dict(description.get('args', {}))
        if description.get('type') in ('QuantizedLinear', 'DynamicQuantizedLinear'):
            dtypes = {'torch.qint8': torch.qint8, 'torch.float16': torch.float16}
            if args.get('dtype') not in dtypes:
                raise CheckpointError('Unsupported quantized dtype')
            args['dtype'] = dtypes[args['dtype']]
        model = constructor(**args)
        seen[path] = model
        return model
    except (TypeError, ValueError, RuntimeError) as error:
        raise CheckpointError(f'Invalid architecture arguments: {error}') from error


class CheckpointManager:
    FORMAT = 'fedcore.weights'
    VERSION = 1

    def __init__(self, base_dir, auto_cleanup=True):
        self.base_dir, self.auto_cleanup = base_dir, auto_cleanup

    def get_checkpoint_dir(self, fedcore_id):
        return os.path.join(self.base_dir, 'checkpoints', fedcore_id)

    def generate_checkpoint_path(self, fedcore_id, model_id, timestamp):
        directory = self.get_checkpoint_dir(fedcore_id)
        os.makedirs(directory, exist_ok=True)
        safe_id = model_id.replace('/', '_').replace(chr(92), '_')
        return os.path.join(directory, f'{safe_id}_{timestamp}.pt')

    def serialize_to_bytes(self, model=None, model_path=None):
        if model is None and model_path:
            # Validate existing files before adding a registry record.
            payload = torch.load(model_path, map_location='cpu', weights_only=True)
            self._validate_payload(payload)
        elif isinstance(model, nn.Module):
            _precheck_cv_checkpoint(model)
            from fedcore.algorithm.low_rank.topology import inspect_topology
            topology = inspect_topology(model)
            if topology.storage_aliases:
                raise CheckpointError('Shared storage views require a separate checkpoint profile')
            payload = {'format': self.FORMAT, 'version': self.VERSION,
                       'architecture': describe_model(model),
                       'topology': {'version': 1, 'module_aliases': topology.module_aliases,
                                    'parameter_aliases': topology.parameter_aliases,
                                    'buffer_aliases': _buffer_aliases(model)},
                       'state_dict': _clone_state_value(model.state_dict()),
                       'training': {name: child.training for name, child in model.named_modules()}}
            payload['requires_grad'] = {name: param.requires_grad for name, param in model.named_parameters()}
            evidence = getattr(model, '_fedcore_method_evidence', None)
            if evidence is not None:
                import json
                try:
                    payload['method_evidence'] = json.loads(json.dumps(evidence, allow_nan=False))
                except (TypeError, ValueError) as error:
                    raise CheckpointError('Method evidence must be finite JSON metadata') from error
        else:
            raise CheckpointError('A torch.nn.Module or valid weights checkpoint is required')
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        return buffer.getvalue()

    def save_to_file(self, checkpoint_bytes, target_path, cleanup_after_save=None):
        if checkpoint_bytes is None:
            raise CheckpointError('Checkpoint bytes are missing')
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        temporary = target_path + '.tmp'
        try:
            with open(temporary, 'wb') as stream:
                stream.write(checkpoint_bytes)
            os.replace(temporary, target_path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
        if self.auto_cleanup if cleanup_after_save is None else cleanup_after_save:
            self._cleanup_gpu_memory()

    @classmethod
    def _validate_payload(cls, payload):
        if not isinstance(payload, Mapping):
            raise CheckpointError('Expected a weights-only checkpoint mapping')
        if 'format' in payload or 'version' in payload:
            if (payload.get('format') != cls.FORMAT or
                    type(payload.get('version')) is not int or payload['version'] != cls.VERSION):
                raise CheckpointError('Unknown checkpoint format or version')
            state = payload.get('state_dict')
        else:
            state = payload.get('state_dict', payload)  # legacy weights require explicit model
        if not isinstance(state, Mapping):
            raise CheckpointError('state_dict must contain named state values')
        for name in ('training', 'requires_grad'):
            values = payload.get(name, {})
            if not isinstance(values, Mapping) or any(
                    not isinstance(key, str) or type(value) is not bool for key, value in values.items()):
                raise CheckpointError(f'{name} must map names to Boolean flags')
        _clone_state_value(state)  # Reject arbitrary custom extra-state objects.
        try:
            return _decode_state_value(state)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise CheckpointError(f'Invalid encoded state: {error}') from error

    @classmethod
    def restore(cls, payload, device=None, model=None, model_factory=None):
        state = cls._validate_payload(payload)
        if model is not None and model_factory is not None:
            raise CheckpointError('Supply either model or model_factory, not both')
        if model is not None:
            if not isinstance(model, nn.Module):
                raise CheckpointError('The supplied model must be torch.nn.Module')
            _precheck_cv_checkpoint(model)
            from fedcore.algorithm.low_rank.topology import inspect_topology
            if inspect_topology(model).storage_aliases:
                raise CheckpointError('Checkpoint restoration does not support shared storage views anywhere in the model')
            from fedcore.algorithm.pruning.sparse.checkpoint import validate_sparse_storage
            try:
                validate_sparse_storage(model)
            except ValueError as error:
                raise CheckpointError(str(error)) from error
        candidate = deepcopy(model) if model is not None else (
            model_factory() if model_factory is not None else build_model(payload.get('architecture')))
        if not isinstance(candidate, nn.Module):
            raise CheckpointError('The architecture factory must return torch.nn.Module')
        _precheck_cv_checkpoint(candidate)
        from fedcore.algorithm.low_rank.topology import module_paths, parameter_paths, TopologyError
        from fedcore.algorithm.low_rank.svd_tools import retie_parameters, validate_tied_state
        graph = payload.get('topology')
        if graph is not None:
            if not isinstance(graph, Mapping) or type(graph.get('version')) is not int or graph['version'] != 1:
                raise CheckpointError('Unknown checkpoint topology version')
            paths = module_paths(candidate)
            params = parameter_paths(candidate)
            for field, available in (('module_aliases', paths), ('parameter_aliases', params)):
                groups = graph.get(field)
                if not isinstance(groups, (list, tuple)) or any(
                    not isinstance(group, (list, tuple)) or len(group) < 2 or
                    any(not isinstance(path, str) or path not in available for path in group) or
                    len(set(group)) != len(group) for group in groups):
                    raise CheckpointError(f'Invalid checkpoint {field}')
                flattened = [path for group in groups for path in group]
                if len(flattened) != len(set(flattened)):
                    raise CheckpointError(f'Overlapping checkpoint {field}')
            for group in graph['module_aliases']:
                if '' in group or any(type(paths[path]) is not type(paths[group[0]]) for path in group):
                    raise CheckpointError('Incompatible checkpoint module aliases')
                source = paths[group[0]]
                for path in group[1:]:
                    owner, _, name = path.rpartition('.')
                    paths[owner]._modules[name] = source
                paths = module_paths(candidate)
            params = parameter_paths(candidate)
            for group in graph['parameter_aliases']:
                if any(params[path].shape != params[group[0]].shape for path in group):
                    raise CheckpointError('Incompatible checkpoint parameter aliases')
            retie_parameters(candidate, graph['parameter_aliases'])
            groups = graph.get('buffer_aliases', _buffer_aliases(candidate))
            buffers = _buffer_paths(candidate)
            if (not isinstance(groups,(tuple,list)) or any(
                    not isinstance(group,(tuple,list)) or len(group)<2
                    or len(set(group))!=len(group) or any(path not in buffers for path in group)
                    for group in groups)):
                raise CheckpointError('Invalid checkpoint buffer aliases')
            flattened = [path for group in groups for path in group]
            if len(flattened)!=len(set(flattened)):
                raise CheckpointError('Overlapping checkpoint buffer aliases')
            for group in groups:
                source = buffers[group[0]]
                if any(buffers[path].shape!=source.shape or buffers[path].dtype!=source.dtype for path in group):
                    raise CheckpointError('Incompatible checkpoint buffer aliases')
            _retie_buffers(candidate,groups)
        from fedcore.algorithm.pruning.sparse.checkpoint import validate_sparse_storage
        try:
            validate_sparse_storage(candidate)
            validate_sparse_storage(candidate, state)
        except ValueError as error:
            raise CheckpointError(str(error)) from error
        _validate_remapped_state(candidate,candidate.state_dict())
        _validate_remapped_state(candidate,state)
        buffer_ties = _validate_tied_buffers(candidate,state)
        try:
            topology = validate_tied_state(candidate, state)
        except TopologyError as error:
            raise CheckpointError(str(error)) from error
        expected = candidate.state_dict()
        if not _same_state_shape(state, expected):
            raise CheckpointError('Checkpoint keys or tensor shapes do not match the architecture')
        _validate_p3_state(candidate,state)
        from fedcore.algorithm.low_rank.architecture_checkpoint import validate_p5_state
        from fedcore.algorithm.pruning.sparse.checkpoint import validate_sparse_state
        from fedcore.algorithm.pruning.sparse.cv_channels import validate_cv_state
        for path, layer in module_paths(candidate).items():
            try:
                from fedcore.algorithm.quantization.checkpoint import validate_quantized_state
                validate_quantized_state(layer, state, path + '.' if path else '')
                validate_p5_state(layer, state, path + '.' if path else '')
                validate_sparse_state(layer, state, path + '.' if path else '')
                validate_cv_state(layer, state, path + '.' if path else '')
                from fedcore.algorithm.pruning.sparse.moe import ReferenceMoE
                if type(layer) is ReferenceMoE:
                    layer.validate_state(state, path + '.' if path else '')
            except (TypeError, ValueError, KeyError) as error:
                raise CheckpointError('Invalid P5 architecture state') from error
        # assign retains each tensor dtype; restoration never mutates supplied models.
        try:
            candidate.load_state_dict(state, strict=True, assign=True)
        except (RuntimeError, KeyError, TypeError, ValueError) as error:
            raise CheckpointError(f'Checkpoint weights are incompatible: {error}') from error
        candidate.to(device or 'cpu')
        # assign=True and device conversion may each replace Parameter objects.
        retie_parameters(candidate, topology.parameter_aliases)
        _retie_buffers(candidate,buffer_ties)
        training = payload.get('training', {})
        for name, child in candidate.named_modules():
            child.training = training.get(name, child.training)
        for name, param in candidate.named_parameters():
            param.requires_grad_(payload.get('requires_grad', {}).get(name, param.requires_grad))
        if 'method_evidence' in payload:
            import json
            try:
                candidate._fedcore_method_evidence = json.loads(json.dumps(payload['method_evidence'], allow_nan=False))
            except (TypeError, ValueError) as error:
                raise CheckpointError('Invalid finite method evidence') from error
        return candidate

    def load_from_file(self, checkpoint_path, device=None, *, model=None, model_factory=None):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        return self.restore(torch.load(checkpoint_path, map_location=device or 'cpu', weights_only=True),
                            device, model, model_factory)

    def deserialize_from_bytes(self, checkpoint_bytes, device=None, *, model=None, model_factory=None):
        return self.restore(torch.load(io.BytesIO(checkpoint_bytes), map_location=device or 'cpu', weights_only=True),
                            device, model, model_factory)

    def get_gpu_memory_stats(self):
        allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        return {'allocated_gb': allocated / 1024**3, 'reserved_gb': reserved / 1024**3,
                'allocated_mb': allocated / 1024**2, 'reserved_mb': reserved / 1024**2}

    def _cleanup_gpu_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
