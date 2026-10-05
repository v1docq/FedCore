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
                torch.float16, torch.float32, torch.float64, torch.int8, torch.uint8, torch.int32, torch.int64)}
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
    """Only encode constructors from a fixed allowlist; never arbitrary Python."""
    if type(model) in (nn.quantized.Linear, nn.quantized.dynamic.Linear):
        return {'type': 'DynamicQuantizedLinear' if type(model) is nn.quantized.dynamic.Linear else 'QuantizedLinear',
                'args': {'in_features': model.in_features, 'out_features': model.out_features,
                         'bias_': model.bias() is not None, 'dtype': str(model.weight().dtype)}}
    if type(model) is nn.Linear:
        return {'type': 'Linear', 'args': {'in_features': model.in_features,
                'out_features': model.out_features, 'bias': model.bias is not None}}
    if type(model) is nn.Sequential:
        children = {name: describe_model(child) for name, child in model.named_children()}
        return {'type': 'Sequential', 'children': children} if all(children.values()) else None
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


def build_model(description):
    if not isinstance(description, dict):
        raise CheckpointError('Architecture is missing: provide model= or model_factory=')
    if description.get('type') == 'Sequential':
        if not isinstance(description.get('children'), Mapping):
            raise CheckpointError('Sequential architecture requires named children')
        return nn.Sequential(OrderedDict((name, build_model(child))
                             for name, child in description['children'].items()))
    constructors = {name: getattr(nn, name) for name in
                    ('Linear', 'ReLU', 'Identity', 'Sigmoid', 'Tanh', 'Flatten', 'Dropout')}
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
        return constructor(**args)
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
            payload = {'format': self.FORMAT, 'version': self.VERSION,
                       'architecture': describe_model(model),
                       'state_dict': _clone_state_value(model.state_dict()),
                       'training': {name: child.training for name, child in model.named_modules()}}
            payload['requires_grad'] = {name: param.requires_grad for name, param in model.named_parameters()}
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
        candidate = deepcopy(model) if model is not None else (
            model_factory() if model_factory is not None else build_model(payload.get('architecture')))
        if not isinstance(candidate, nn.Module):
            raise CheckpointError('The architecture factory must return torch.nn.Module')
        expected = candidate.state_dict()
        if not _same_state_shape(state, expected):
            raise CheckpointError('Checkpoint keys or tensor shapes do not match the architecture')
        # assign retains each tensor dtype; restoration never mutates supplied models.
        try:
            candidate.load_state_dict(state, strict=True, assign=True)
        except (RuntimeError, KeyError, TypeError, ValueError) as error:
            raise CheckpointError(f'Checkpoint weights are incompatible: {error}') from error
        candidate.to(device or 'cpu')
        training = payload.get('training', {})
        for name, child in candidate.named_modules():
            child.training = training.get(name, child.training)
        for name, param in candidate.named_parameters():
            param.requires_grad_(payload.get('requires_grad', {}).get(name, param.requires_grad))
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
