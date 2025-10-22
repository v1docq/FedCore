"""
Utilities for structured pruning of convolutional models (ResNet-focused).

This module contains:
- **Filter zeroing helpers** for Conv2d layers:
    * :func:`percentage_filter_zeroing` — zero the lowest-norm filters by a ratio.
    * :func:`energy_filter_zeroing` — zero filters until a target energy is kept.
- **State dict surgery** for ResNet-like models:
    * Functions to parse/collect nested state dicts and to prune channels in a
      topology-aware way, ensuring BatchNorm/Conv/Linear tensors stay consistent.
    * :func:`prune_resnet_state_dict` — prune a ResNet `state_dict` by removing
      entirely-zeroed filters and propagating the channel selection through the
      block graph.
    * :func:`sizes_from_state_dict` — report tensor shapes after pruning.

Notes
-----
These helpers operate purely on tensors and state dicts (no forward graph),
which makes them convenient for offline/serialized pruning pipelines.
"""

from collections import OrderedDict
from typing import Dict, Optional, Tuple, List, Union

import torch
from torch import Tensor
from torch.linalg import vector_norm
from torch.nn import Conv2d


def percentage_filter_zeroing(conv: Conv2d, pruning_ratio: float) -> None:
    """Zero out a percentage of filters in a convolution (in-place).

    The `pruning_ratio` fraction of filters (by smallest L2 norm over
    weights) are set to zero. Biases are not modified.

    Parameters
    ----------
    conv : torch.nn.Conv2d
        The convolutional layer to modify.
    pruning_ratio : float
        Fraction of filters to zero in the open interval (0, 1).

    Raises
    ------
    AssertionError
        If ``pruning_ratio`` is not in (0, 1).
    """
    assert 0 < pruning_ratio < 1, "pruning_ratio must be in the range (0, 1)"
    filter_pruned_num = int(conv.weight.size()[0] * pruning_ratio)
    filter_norms = vector_norm(conv.weight, dim=(1, 2, 3))
    _, indices = filter_norms.sort()
    with torch.no_grad():
        conv.weight[indices[:filter_pruned_num]] = 0


def energy_filter_zeroing(conv: Conv2d, energy_threshold: float) -> None:
    """Zero filters of a convolution until a target energy remains (in-place).

    Filters are sorted by L2 norm; the smallest ones are zeroed iteratively
    until the remaining sum of squared norms falls below
    ``energy_threshold * total_energy``.

    Parameters
    ----------
    conv : torch.nn.Conv2d
        The convolutional layer to modify.
    energy_threshold : float
        Target energy share to preserve, in the interval (0, 1].

    Raises
    ------
    AssertionError
        If ``energy_threshold`` is not in (0, 1].
    """
    assert 0 < energy_threshold <= 1, "energy_threshold must be in the range (0, 1]"
    filter_norms = vector_norm(conv.weight, dim=(1, 2, 3))
    sorted_filter_norms, indices = filter_norms.sort()
    sum = (filter_norms**2).sum()
    threshold = energy_threshold * sum
    for index, filter_norm in zip(indices, sorted_filter_norms):
        with torch.no_grad():
            conv.weight[index] = 0
        sum -= filter_norm**2
        if sum < threshold:
            break


def _check_nonzero_filters(weight: Tensor) -> Tensor:
    """Return indices of filters with at least one non-zero element.

    Parameters
    ----------
    weight : torch.Tensor
        Convolutional weight tensor of shape (out_channels, in_channels, H, W).

    Returns
    -------
    torch.Tensor
        1D tensor of indices for non-zero filters (by any element).
    """
    filters = torch.count_nonzero(weight, dim=(1, 2, 3))
    indices = torch.flatten(torch.nonzero(filters))
    return indices


def _prune_filters(
    weight: Tensor,
    saving_filters: Optional[Tensor] = None,
    saving_channels: Optional[Tensor] = None,
) -> Tensor:
    """Prune filters and/or input channels of a convolution weight tensor.

    Parameters
    ----------
    weight : torch.Tensor
        Convolutional weight tensor.
    saving_filters : Optional[torch.Tensor]
        Indices of **output** filters to keep. If ``None``, keep all.
    saving_channels : Optional[torch.Tensor]
        Indices of **input** channels to keep. If ``None``, keep all.

    Returns
    -------
    torch.Tensor
        A cloned/pruned weight tensor.
    """
    if saving_filters is not None:
        weight = weight[saving_filters].clone()
    if saving_channels is not None:
        weight = weight[:, saving_channels].clone()
    return weight


def _prune_batchnorm(bn: Dict, saving_channels: Tensor) -> Dict[str, Tensor]:
    """Prune BatchNorm1d/2d-like parameter tensors by channel indices.

    Parameters
    ----------
    bn : Dict[str, torch.Tensor]
        Dictionary with BN params: ``weight``, ``bias``, ``running_mean``,
        ``running_var``.
    saving_channels : torch.Tensor
        Indices of channels to keep.

    Returns
    -------
    Dict[str, torch.Tensor]
        BN dictionary with tensors sliced to the selected channels.
    """
    bn["weight"] = bn["weight"][saving_channels].clone()
    bn["bias"] = bn["bias"][saving_channels].clone()
    bn["running_mean"] = bn["running_mean"][saving_channels].clone()
    bn["running_var"] = bn["running_var"][saving_channels].clone()
    return bn


def _index_union(x: Tensor, y: Tensor) -> Tensor:
    """Return the set union of two index tensors.

    Parameters
    ----------
    x, y : torch.Tensor
        1D integer index tensors.

    Returns
    -------
    torch.Tensor
        1D tensor with unique indices from both inputs (order not guaranteed).
    """
    x = set(x.tolist())
    y = set(y.tolist())
    xy = x | y
    return torch.tensor(list(xy))


def _indexes_of_tensor_values(tensor: Tensor, values: Tensor) -> Tensor:
    """Return positions of specific values within a 1D tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        1D tensor whose values are searched.
    values : torch.Tensor
        1D tensor with values to find in ``tensor``.

    Returns
    -------
    torch.Tensor
        1D tensor of integer indices corresponding to ``values`` in ``tensor``.

    Notes
    -----
    This performs linear searches and assumes that all ``values`` are present
    in ``tensor`` (no error handling for missing values).
    """
    indexes = []
    tensor = tensor.tolist()
    for value in values.tolist():
        indexes.append(tensor.index(value))
    return torch.tensor(indexes)


def _parse_sd(state_dict: OrderedDict) -> OrderedDict:
    """Convert a flat state_dict into a nested dictionary structure.

    Keys are split by '.' to create nested mappings.

    Parameters
    ----------
    state_dict : OrderedDict
        Flat state dict as produced by PyTorch modules.

    Returns
    -------
    OrderedDict
        Nested dictionary mirroring the module hierarchy.
    """
    parsed_sd = OrderedDict()
    for k, v in state_dict.items():
        _parse_param(k.split("."), v, parsed_sd)
    return parsed_sd


def _parse_param(param: List, value: Tensor, dictionary: OrderedDict) -> None:
    """Recursive helper to build nested dicts from dotted keys.

    Parameters
    ----------
    param : List[str]
        Key tokens obtained by splitting a full key on '.'.
    value : torch.Tensor
        Tensor to assign.
    dictionary : OrderedDict
        Current nesting level to populate.
    """
    if len(param) > 1:
        dictionary.setdefault(param[0], OrderedDict())
        _parse_param(param[1:], value, dictionary[param[0]])
    else:
        dictionary[param[0]] = value


def _collect_sd(parsed_state_dict: OrderedDict) -> OrderedDict:
    """Flatten a nested dictionary back into a standard state_dict.

    Parameters
    ----------
    parsed_state_dict : OrderedDict
        Nested dictionary (as produced by :func:`_parse_sd`).

    Returns
    -------
    OrderedDict
        Flat state dict with dotted keys.
    """
    state_dict = OrderedDict()
    keys, values = _collect_param(parsed_state_dict)
    for k, v in zip(keys, values):
        key = ".".join(k)
        state_dict[key] = v
    return state_dict


def _collect_param(dictionary: Union[OrderedDict, Tensor]) -> Tuple:
    """Recursive helper to collect keys/values from a nested dictionary.

    Parameters
    ----------
    dictionary : OrderedDict | torch.Tensor
        Either a nested mapping or a leaf tensor.

    Returns
    -------
    Tuple[List[List[str]], List[torch.Tensor]]
        Collected key paths (as token lists) and corresponding leaf tensors.
    """
    if isinstance(dictionary, OrderedDict):
        all_keys = []
        all_values = []
        for k, v in dictionary.items():
            keys, values = _collect_param(v)
            for key in keys:
                key.insert(0, k)
            all_values.extend(values)
            all_keys.extend(keys)
        return all_keys, all_values
    else:
        return [[]], [dictionary]


def _prune_resnet_block(block: Dict, input_channels: Tensor) -> Tensor:
    """Prune a single ResNet basic/bottleneck block by channel connectivity.

    The procedure:
      1) If present, prune the downsample branch (Conv-BN) using the incoming
         channels; record its output channels.
      2) For the main path, iteratively prune each Conv by keeping non-zero
         output filters and slicing input channels to match the previous layer.
      3) For the final Conv, take the union of its non-zero filters and the
         downsample's output channels to preserve residual addition shape.
      4) Prune the final BN accordingly and store indices aligning the
         downsample output with the main path.

    Parameters
    ----------
    block : Dict
        Nested dictionary representing a serialized ResNet block.
    input_channels : torch.Tensor
        Indices of channels entering the block.

    Returns
    -------
    torch.Tensor
        Indices of channels leaving the block (for the next block).
    """
    channels = input_channels
    downsample_channels = input_channels
    keys = list(block.keys())
    if "downsample" in keys:
        filters = _check_nonzero_filters(block["downsample"]["0"]["weight"])
        block["downsample"]["0"]["weight"] = _prune_filters(
            weight=block["downsample"]["0"]["weight"],
            saving_filters=filters,
            saving_channels=downsample_channels,
        )
        downsample_channels = filters
        block["downsample"]["1"] = _prune_batchnorm(
            bn=block["downsample"]["1"], saving_channels=downsample_channels
        )
        keys.remove("downsample")
    final_conv = keys[-2]
    final_bn = keys[-1]
    keys = keys[:-2]
    for key in keys:
        if key.startswith("conv"):
            filters = _check_nonzero_filters(block[key]["weight"])
            block[key]["weight"] = _prune_filters(
                weight=block[key]["weight"],
                saving_filters=filters,
                saving_channels=channels,
            )
            channels = filters
        elif key.startswith("bn"):
            block[key] = _prune_batchnorm(bn=block[key], saving_channels=channels)
    filters = _check_nonzero_filters(block[final_conv]["weight"])
    filters = _index_union(filters, downsample_channels)
    block[final_conv]["weight"] = _prune_filters(
        weight=block[final_conv]["weight"],
        saving_filters=filters,
        saving_channels=channels,
    )
    channels = filters
    block[final_bn] = _prune_batchnorm(bn=block[final_bn], saving_channels=channels)
    block["indices"] = _indexes_of_tensor_values(channels, downsample_channels)
    return channels


def prune_resnet_state_dict(
    state_dict: OrderedDict,
) -> OrderedDict:
    """Prune a serialized ResNet by removing zeroed filters and aligning channels.

    This function inspects the state dict, determines non-zero filters,
    prunes convolution/BN tensors accordingly throughout all blocks/layers,
    and returns a **new** state dict consistent with the pruned topology.

    Parameters
    ----------
    state_dict : OrderedDict
        State dict of a ResNet model.

    Returns
    -------
    OrderedDict
        Pruned state dict with updated tensors and auxiliary indices.
    """
    sd = _parse_sd(state_dict)
    filters = _check_nonzero_filters(sd["conv1"]["weight"])
    sd["conv1"]["weight"] = _prune_filters(
        weight=sd["conv1"]["weight"], saving_filters=filters
    )
    channels = filters
    sd["bn1"] = _prune_batchnorm(bn=sd["bn1"], saving_channels=channels)

    for layer in ["layer1", "layer2", "layer3", "layer4"]:
        for block in sd[layer].values():
            channels = _prune_resnet_block(block=block, input_channels=channels)
    sd["fc"]["weight"] = sd["fc"]["weight"][:, channels].clone()
    sd = _collect_sd(sd)
    return sd


def sizes_from_state_dict(state_dict: OrderedDict) -> Dict:
    """Report tensor shapes for each major module in a (possibly pruned) ResNet.

    Useful for constructing an architecture stub that matches the tensor sizes
    of a pruned state dict.

    Parameters
    ----------
    state_dict : OrderedDict
        (Pruned) state dict as accepted by :func:`prune_resnet_state_dict`.

    Returns
    -------
    dict
        A nested dictionary with shapes for convs/BNs/FC and downsample parts.
    """
    sd = _parse_sd(state_dict)
    sizes = {"conv1": sd["conv1"]["weight"].shape}
    for layer in ["layer1", "layer2", "layer3", "layer4"]:
        sizes[layer] = {}
        for i, block in enumerate(sd[layer].values()):
            sizes[layer][i] = {}
            for k, v in block.items():
                if k.startswith("conv"):
                    sizes[layer][i][k] = v["weight"].shape
                elif k == "downsample":
                    sizes[layer][k] = v["0"]["weight"].shape
                elif k == "indices":
                    sizes[layer][i][k] = v.shape
    sizes["fc"] = sd["fc"]["weight"].shape
    return sizes


#
# def prune_resnet(model: ResNet) -> PrunedResNet:
#     """Prune ResNet
#     ...
#     """
#     ...
#
# def load_sfp_resnet_model(state_dict_path: str) -> torch.nn.Module:
#     """Loads SFP state_dict to PrunedResNet model.
#     ...
#     """
#     ...
