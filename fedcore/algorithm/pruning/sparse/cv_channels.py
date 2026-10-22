"""Bounded CIFAR ResNet18 and a physical internal-channel L1 profile.

This is ``channel_l1_structural_v1``, not a reproduction of an unrelated
SparseCPU paper. Only internal BasicBlock channels change; residual coordinates
and every block output are fixed. Two ordinary Conv2d layers store SVD factors.
"""
from __future__ import annotations

from collections.abc import Mapping
from collections import OrderedDict
import math
import torch
from torch import nn


BLOCK_PATHS = tuple(f"layer{stage}.{index}" for stage in range(1, 5) for index in range(2))
BLOCK_OUTPUTS = (64, 64, 128, 128, 256, 256, 512, 512)
BLOCK_INPUTS = (64, 64, 64, 128, 128, 256, 256, 512)
BLOCK_STRIDES = (1, 1, 2, 1, 2, 1, 2, 1)
MAX_CV_TENSOR_BYTES = 64 * 1024**2


def _config(num_classes, widths, ranks, channels):
    if type(num_classes) is not int or not 2 <= num_classes <= 1000:
        raise ValueError("CIFAR ResNet18 requires 2..1000 classes")
    if any(type(value) not in (tuple, list) or len(value) != 8 for value in (widths, ranks, channels)):
        raise ValueError("Declare exactly eight BasicBlock configurations")
    for width, rank, coordinates, output in zip(widths, ranks, channels, BLOCK_OUTPUTS):
        if type(width) is not int or not 1 <= width <= output:
            raise ValueError("Internal channels must fit the original block width")
        if rank is not None and (type(rank) is not int or not 1 <= rank <= min(output, 9 * width)):
            raise ValueError("Conv2 SVD rank must fit the reduced convolution")
        if (type(coordinates) not in (tuple, list) or len(coordinates) != width
                or any(type(index) is not int or not 0 <= index < output for index in coordinates)
                or tuple(coordinates) != tuple(sorted(set(coordinates)))):
            raise ValueError("Declare canonical original internal-channel coordinates")
    return num_classes, tuple(widths), tuple(ranks), tuple(tuple(value) for value in channels)


class CVBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, width, stride=1, rank=None, *, device=None, dtype=None):
        super().__init__()
        options = {"device": device, "dtype": dtype}
        self.conv1 = nn.Conv2d(in_channels, width, 3, stride, 1, bias=False, **options)
        self.bn1 = nn.BatchNorm2d(width, **options)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = (nn.Conv2d(width, out_channels, 3, 1, 1, bias=False, **options) if rank is None else
                      nn.Sequential(nn.Conv2d(width, rank, 3, 1, 1, bias=False, **options),
                                    nn.Conv2d(rank, out_channels, 1, bias=False, **options)))
        self.bn2 = nn.BatchNorm2d(out_channels, **options)
        self.downsample = (nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False, **options),
                                        nn.BatchNorm2d(out_channels, **options))
                           if stride != 1 or in_channels != out_channels else None)
        self.stride = stride

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        return self.relu(output + identity)


class CifarResNet18(nn.Module):
    """The torchvision ResNet18 graph with a 3x3 CIFAR stem and no maxpool.

    Dense state_dict names and arithmetic match a CIFAR-adapted torchvision
    ResNet18. Compact constructor metadata records coordinates without adding
    tensors to that dense state_dict contract.
    """
    def __init__(self, num_classes=10, *, block_widths=None, conv2_ranks=None,
                 channel_indices=None, device=None, dtype=None):
        super().__init__()
        widths = BLOCK_OUTPUTS if block_widths is None else block_widths
        ranks = (None,) * 8 if conv2_ranks is None else conv2_ranks
        channels = tuple(tuple(range(width)) for width in widths) if channel_indices is None else channel_indices
        self.num_classes, self.block_widths, self.conv2_ranks, self.channel_indices = _config(
            num_classes, widths, ranks, channels)
        if dtype not in (None, torch.float32):
            raise ValueError("The bounded CV constructor requires float32")
        options = {"device": device, "dtype": dtype}
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False, **options)
        self.bn1 = nn.BatchNorm2d(64, **options)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()
        for stage in range(1, 5):
            blocks = []
            for position in range(2):
                index = 2 * (stage - 1) + position
                blocks.append(CVBasicBlock(BLOCK_INPUTS[index], BLOCK_OUTPUTS[index], widths[index],
                                           BLOCK_STRIDES[index], ranks[index], **options))
            setattr(self, f"layer{stage}", nn.Sequential(*blocks))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes, **options)
        for layer in self.modules():
            if type(layer) is nn.Conv2d:
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            elif type(layer) is nn.BatchNorm2d:
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))

    def representation_config(self):
        return {"version": 1, "num_classes": self.num_classes, "block_widths": list(self.block_widths),
                "conv2_ranks": list(self.conv2_ranks), "channel_indices": [list(value) for value in self.channel_indices]}

    @classmethod
    def from_config(cls, config, *, device=None, dtype=None):
        if (type(config) is not dict or set(config) != {"version", "num_classes", "block_widths", "conv2_ranks", "channel_indices"}
                or type(config["version"]) is not int or config["version"] != 1):
            raise ValueError("Exact version-1 CIFAR ResNet18 config required")
        values = _config(config["num_classes"], config["block_widths"], config["conv2_ranks"], config["channel_indices"])
        return cls(values[0], block_widths=values[1], conv2_ranks=values[2], channel_indices=values[3],
                   device=device, dtype=dtype)


def make_cifar_resnet18(num_classes=10, **options):
    return CifarResNet18(num_classes, **options)


def _children(layer, expected):
    if tuple(layer._modules) != tuple(expected) or any(type(layer._modules[name]) is not kind for name, kind in expected.items()):
        raise ValueError("CV graph must contain exactly the declared child constructors")


_HOOK_FIELDS = ("_backward_pre_hooks", "_backward_hooks", "_forward_hooks", "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called", "_forward_pre_hooks", "_forward_pre_hooks_with_kwargs", "_state_dict_hooks",
    "_state_dict_pre_hooks", "_load_state_dict_pre_hooks", "_load_state_dict_post_hooks")
_BASE_FIELDS = {"training", "_parameters", "_buffers", "_non_persistent_buffers_set", "_modules",
                "_is_full_backward_hook", *_HOOK_FIELDS}


def _safe_metadata(value, depth=0):
    """Exact builtin metadata only: no user iterators, tensor or deepcopy hooks."""
    if depth > 20:
        return False
    if value is None or type(value) in (bool, int, str):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) in (list, tuple):
        return len(value) <= 8192 and all(_safe_metadata(item, depth + 1) for item in value)
    if type(value) is dict:
        return len(value) <= 1024 and all(type(key) is str and _safe_metadata(item, depth + 1)
                                         for key, item in value.items())
    return False


def _instance_guard(layer):
    """Validate raw instance slots and registries before any user-owned iteration."""
    additional = {
        CifarResNet18: {"num_classes", "block_widths", "conv2_ranks", "channel_indices"},
        CVBasicBlock: {"stride", "downsample"},
        nn.Conv2d: {"in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "transposed",
                    "output_padding", "groups", "padding_mode", "_reversed_padding_repeated_twice"},
        nn.BatchNorm2d: {"num_features", "eps", "momentum", "affine", "track_running_stats"},
        nn.Linear: {"in_features", "out_features"}, nn.ReLU: {"inplace"},
        nn.AdaptiveAvgPool2d: {"output_size"}, nn.Sequential: set(), nn.Identity: set(),
    }[type(layer)]
    values = vars(layer)
    if type(values) is not dict:
        raise ValueError("CV module instances require an exact builtin attribute dictionary")
    evidence_fields = {"_fedcore_method_evidence", "_fedcore_requires_optimizer_rebuild"} if type(layer) is CifarResNet18 else set()
    if not set(values).issubset(_BASE_FIELDS | additional | evidence_fields):
        raise ValueError("CV constructors contain unknown instance attributes")
    for name in ("_modules", "_parameters", "_buffers", *_HOOK_FIELDS):
        registry = values.get(name)
        if type(registry) not in (dict, OrderedDict) or any(type(key) is not str for key in registry):
            raise ValueError("CV service registries require exact builtin mappings and string keys")
        if name in _HOOK_FIELDS and len(registry):
            raise ValueError("CV constructors cannot contain runtime or checkpoint hooks")
    if type(values.get("_non_persistent_buffers_set")) is not set or values["_non_persistent_buffers_set"]:
        raise ValueError("CV constructors cannot contain nonpersistent or opaque buffer registries")
    if values.get("_is_full_backward_hook") is not None:
        raise ValueError("CV backward hook state must be unmodified")
    for name in additional | evidence_fields:
        if name in values and not _safe_metadata(values[name]):
            raise ValueError("CV constructor metadata must contain exact safe builtin values only")
    if any(value is not None and type(value) is not nn.Parameter for value in values["_parameters"].values()):
        raise ValueError("CV registered parameters require exact nn.Parameter objects")
    if any(value is not None and type(value) is not torch.Tensor for value in values["_buffers"].values()):
        raise ValueError("CV registered buffers require exact torch.Tensor objects")
    for tensor in (*values["_parameters"].values(), *values["_buffers"].values()):
        if tensor is None:
            continue
        attributes = vars(tensor)
        if type(attributes) is not dict or len(attributes):
            raise ValueError("CV tensor instances cannot contain custom attributes or method overrides")
        for name in ("_backward_hooks", "_post_accumulate_grad_hooks"):
            hooks = getattr(tensor, name, None)
            if hooks is not None and (type(hooks) not in (dict, OrderedDict) or len(hooks)):
                raise ValueError("CV tensors cannot contain active or opaque backward/gradient-accumulation hooks")


def _conv(layer, inputs, outputs, kernel=3, stride=1, padding=1):
    if (type(layer) is not nn.Conv2d or layer.in_channels != inputs or layer.out_channels != outputs
            or layer.kernel_size != (kernel, kernel) or layer.stride != (stride, stride)
            or layer.padding != (padding, padding) or layer.dilation != (1, 1)
            or layer.groups != 1 or layer.padding_mode != "zeros" or layer.bias is not None
            or layer.transposed is not False or layer.output_padding != (0, 0)
            or tuple(layer._reversed_padding_repeated_twice) != (padding,) * 4
            or layer.weight.shape != (outputs, inputs, kernel, kernel) or layer.weight.dtype != torch.float32):
        raise ValueError("CV convolution metadata or tensor shape differs from its constructor")


def _bn(layer, width):
    if (type(layer) is not nn.BatchNorm2d or layer.num_features != width or layer.eps != 1e-5
            or layer.momentum != 0.1 or layer.affine is not True or layer.track_running_stats is not True
            or set(layer._parameters) != {"weight", "bias"}
            or set(layer._buffers) != {"running_mean", "running_var", "num_batches_tracked"}
            or any(value.shape != (width,) for value in (layer.weight, layer.bias, layer.running_mean, layer.running_var))
            or any(value.dtype != torch.float32 for value in (layer.weight, layer.bias, layer.running_mean, layer.running_var))
            or layer.num_batches_tracked.shape != torch.Size([]) or layer.num_batches_tracked.dtype != torch.int64
            or int(layer.num_batches_tracked) < 0 or bool((layer.running_var < 0).any())):
        raise ValueError("CV BatchNorm state differs from the declared channel width")


def validate_cv_model(model, *, cpu=False):
    """Check exact constructors and all raw children before invoking state_dict."""
    if type(model) is not CifarResNet18:
        raise ValueError("Only the explicit CifarResNet18 adapter is supported")
    from torch.nn.modules import module as runtime
    for name in (
            "_global_forward_hooks", "_global_forward_pre_hooks", "_global_backward_hooks", "_global_backward_pre_hooks",
            "_global_module_registration_hooks", "_global_parameter_registration_hooks", "_global_buffer_registration_hooks"):
        registry = getattr(runtime, name, {})
        if type(registry) not in (dict, OrderedDict) or len(registry):
            raise ValueError("CV profile cannot execute under global module hooks or opaque hook registries")
    _config(model.num_classes, model.block_widths, model.conv2_ranks, model.channel_indices)
    allowed = (CifarResNet18, CVBasicBlock, nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Identity,
               nn.Sequential, nn.AdaptiveAvgPool2d, nn.Linear)
    pending, seen, storages, dtype, total = [model], set(), set(), None, 0
    while pending:
        layer = pending.pop()
        if type(layer) not in allowed or id(layer) in seen:
            raise ValueError("CV graphs require exact constructors and no module aliases or cycles")
        seen.add(id(layer))
        _instance_guard(layer)
        if type(layer.training) is not bool:
            raise ValueError("CV training modes must be explicit Boolean flags")
        parameters = {"weight", "bias"} if type(layer) in (nn.Conv2d, nn.Linear, nn.BatchNorm2d) else set()
        buffers = {"running_mean", "running_var", "num_batches_tracked"} if type(layer) is nn.BatchNorm2d else set()
        if set(layer._parameters) != parameters or set(layer._buffers) != buffers:
            raise ValueError("CV graph contains undeclared parameter or buffer slots")
        if type(layer) not in (CifarResNet18, CVBasicBlock, nn.Sequential) and layer._modules:
            raise ValueError("CV primitive operators cannot contain child modules")
        for value in (*layer._parameters.values(), *layer._buffers.values()):
            if value is None:
                continue
            if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                    or value.device.type not in (("cpu",) if cpu else ("cpu", "cuda"))
                    or value.is_conj() or value.is_neg() or value.storage_offset() != 0
                    or value.untyped_storage().nbytes() != value.numel() * value.element_size()
                    or not value.is_contiguous()):
                raise ValueError("CV tensors require complete compact ordinary storage")
            storage = (str(value.device), value.untyped_storage()._cdata)
            if storage in storages:
                raise ValueError("CV profile rejects shared tensor storage")
            storages.add(storage)
            total += value.untyped_storage().nbytes()
            if value.is_floating_point():
                dtype = value.dtype if dtype is None else dtype
                if value.dtype != torch.float32 or value.dtype != dtype or not bool(torch.isfinite(value).all()):
                    raise ValueError("CV tensors require finite float32")
            elif value.dtype != torch.int64:
                raise ValueError("Only BatchNorm counters may use an integer dtype")
        pending.extend(value for value in layer._modules.values() if value is not None)
    if total > MAX_CV_TENSOR_BYTES:
        raise ValueError("CV constructor exceeds the 64 MiB physical tensor limit")
    _children(model, {"conv1": nn.Conv2d, "bn1": nn.BatchNorm2d, "relu": nn.ReLU, "maxpool": nn.Identity,
                      **{f"layer{index}": nn.Sequential for index in range(1, 5)}, "avgpool": nn.AdaptiveAvgPool2d, "fc": nn.Linear})
    _conv(model.conv1, 3, 64)
    _bn(model.bn1, 64)
    if model.relu.inplace is not True or model.avgpool.output_size != (1, 1):
        raise ValueError("CV stem activation or pooling configuration changed")
    if (model.fc.in_features != 512 or model.fc.out_features != model.num_classes
            or model.fc.weight.shape != (model.num_classes, 512) or model.fc.bias is None
            or model.fc.bias.shape != (model.num_classes,) or model.fc.weight.dtype != torch.float32
            or model.fc.bias.dtype != torch.float32):
        raise ValueError("CV classifier configuration changed")
    for stage in range(1, 5):
        _children(getattr(model, f"layer{stage}"), {"0": CVBasicBlock, "1": CVBasicBlock})
    for index, path in enumerate(BLOCK_PATHS):
        block = model.get_submodule(path)
        width, rank = model.block_widths[index], model.conv2_ranks[index]
        incoming, outgoing, stride = BLOCK_INPUTS[index], BLOCK_OUTPUTS[index], BLOCK_STRIDES[index]
        _children(block, {"conv1": nn.Conv2d, "bn1": nn.BatchNorm2d, "relu": nn.ReLU,
                          "conv2": nn.Conv2d if rank is None else nn.Sequential, "bn2": nn.BatchNorm2d,
                          **({"downsample": nn.Sequential} if stride != 1 else {})})
        if block.stride != stride or block.relu.inplace is not True:
            raise ValueError("CV block stride or activation changed")
        _conv(block.conv1, incoming, width, stride=stride)
        _bn(block.bn1, width)
        _bn(block.bn2, outgoing)
        if rank is None:
            _conv(block.conv2, width, outgoing)
        else:
            _children(block.conv2, {"0": nn.Conv2d, "1": nn.Conv2d})
            _conv(block.conv2[0], width, rank)
            _conv(block.conv2[1], rank, outgoing, 1, 1, 0)
        if stride != 1:
            _children(block.downsample, {"0": nn.Conv2d, "1": nn.BatchNorm2d})
            _conv(block.downsample[0], incoming, outgoing, 1, stride, 0)
            _bn(block.downsample[1], outgoing)
    return total


def validate_cv_state(model, state, prefix=""):
    if type(model) is not CifarResNet18:
        return
    validate_cv_model(model)
    if not isinstance(state, Mapping):
        raise ValueError("CV state must be a named tensor mapping")
    groups, cv_keys = {}, set()
    for name, reference in model.state_dict().items():
        value = state.get(prefix + name)
        if (type(value) is not torch.Tensor or value.shape != reference.shape or value.dtype != reference.dtype
                or value.layout != torch.strided or not bool(torch.isfinite(value).all())
                or value.storage_offset() != 0 or not value.is_contiguous()
                or value.untyped_storage().nbytes() != value.numel() * value.element_size()):
            raise ValueError("CV checkpoint requires every finite tensor with its declared shape and dtype")
        if name.endswith("num_batches_tracked") and int(value) < 0:
            raise ValueError("BatchNorm counters cannot be negative")
        if name.endswith("running_var") and bool((value < 0).any()):
            raise ValueError("BatchNorm running variance cannot be negative")
        cv_keys.add(prefix + name)
    for name, value in state.items():
        if isinstance(value, torch.Tensor) and value.layout == torch.strided and value.untyped_storage().nbytes():
            groups.setdefault((str(value.device), value.untyped_storage()._cdata), []).append(name)
    if any(len(names) > 1 and cv_keys.intersection(names) for names in groups.values()):
        raise ValueError("CV checkpoint tensors cannot share backing storage")


def select_internal_channels(block, keep_channels):
    """Stable L1 selection on frozen conv1 weights, preserving original order."""
    width = block.conv1.out_channels
    if type(keep_channels) is not int or not 1 <= keep_channels <= width:
        raise ValueError("A positive bounded channel count is required")
    scores = block.conv1.weight.detach().double().abs().flatten(1).sum(1)
    order = sorted(range(width), key=lambda index: (-float(scores[index]), index))
    return tuple(sorted(order[:keep_channels]))


@torch.no_grad()
def build_compact_block(source, keep, rank):
    """Copy linked Conv/BN coordinates and optionally factor the reduced conv2."""
    if type(source.conv2) is not nn.Conv2d:
        raise ValueError("The CV profile observes a dense frozen source block")
    target = CVBasicBlock(source.conv1.in_channels, source.conv2.out_channels, len(keep), source.stride, rank,
                          dtype=source.conv1.weight.dtype)
    indices = torch.tensor(keep, dtype=torch.int64)
    target.conv1.weight.copy_(source.conv1.weight.index_select(0, indices))
    for name in ("weight", "bias", "running_mean", "running_var"):
        getattr(target.bn1, name).copy_(getattr(source.bn1, name).index_select(0, indices))
    target.bn1.num_batches_tracked.copy_(source.bn1.num_batches_tracked)
    weight = source.conv2.weight.index_select(1, indices).contiguous()
    if rank is None:
        target.conv2.weight.copy_(weight)
    else:
        # Exact truncated Frobenius SVD, not an activation-weighted method.
        u, singular, vh = torch.linalg.svd(weight.flatten(1).double(), full_matrices=False)
        target.conv2[0].weight.copy_(vh[:rank].reshape(rank, len(keep), 3, 3).to(weight))
        target.conv2[1].weight.copy_((u[:, :rank] * singular[:rank]).reshape(weight.shape[0], rank, 1, 1).to(weight))
    target.bn2.load_state_dict(source.bn2.state_dict(), strict=True)
    if source.downsample is not None:
        target.downsample.load_state_dict(source.downsample.state_dict(), strict=True)
    target.eval()
    return target
