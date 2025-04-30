import abc
from typing import Any, Dict, List, Literal, Optional, Set, Type, Union

import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import conv1x1, conv3x3
import torch
from torch.nn import Conv2d, Parameter
from torch.nn.functional import conv2d
import torch.nn.functional as F


def parameter_value_check(parameter: str, value: Any, valid_values: Set) -> None:
    """Checks if the parameter value is in the set of valid values.

    Args:
        parameter: Name of the checked parameter.
        value: Value of the checked parameter.
        valid_values: Set of the valid parameter values.

    Rises:
        ValueError: If ``value`` is not in ``valid_values``.


    """
    if value not in valid_values:
        raise ValueError(
            f"{parameter} must be one of {valid_values}, but got {parameter}='{value}'"
        )


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            input_dim,
            output_dim,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(input_dim, output_dim, stride=stride)
        self.bn1 = norm_layer(output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(output_dim, output_dim)
        self.bn2 = norm_layer(output_dim)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.index_add_(1, self.indices, identity)
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            sizes: Dict[str, Tensor],
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(sizes["conv1"][1], sizes["conv1"][0])
        self.bn1 = norm_layer(sizes["conv1"][0])
        self.conv2 = conv3x3(sizes["conv2"][1], sizes["conv2"][0], stride=stride)
        self.bn2 = norm_layer(sizes["conv2"][0])
        self.conv3 = conv1x1(sizes["conv3"][1], sizes["conv3"][0])
        self.bn3 = norm_layer(sizes["conv3"][0])
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.register_buffer("indices", torch.zeros(sizes["indices"], dtype=torch.int))

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.index_add_(1, self.indices, identity)
        out = self.relu(out)

        return out


class PrunedResNet(nn.Module):
    """Pruned ResNet for soft filter pruning optimization.

    Args:

    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            depth: dict,
            custom_params: dict
    ) -> None:
        super().__init__()
        # self.inplanes = 64
        self.in_channels = custom_params['initial_conv_output_dim']
        self._init_initial_conv_and_pool(input_dim, self.in_channels)
        self.resnet_layers = nn.ModuleList()
        for layer_idx in range(depth['layers']):
            resnet_layer = self._make_layer(block=custom_params['block'],
                                            blocks=depth['blocks_per_layer'][layer_idx],
                                            output_dim=custom_params['sizes_per_layer'][layer_idx],
                                            stride=custom_params['strides_per_layer'][layer_idx])
            self.resnet_layers.append(resnet_layer)
        input_fc = custom_params['sizes_per_layer'][layer_idx] * custom_params['block'].expansion
        self._init_final_pool_and_predict(input_fc=1, output_dim=output_dim)

    def _init_initial_conv_and_pool(self, input_dim, output_dim):
        self.conv1 = nn.Conv2d(input_dim, output_dim,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False,
                               )
        self.bn1 = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _init_final_pool_and_predict(self, input_fc, output_dim):
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(input_fc, output_dim)

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            blocks: int,
            output_dim: int,
            stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        layers = []
        output_dim = output_dim * block.expansion
        if stride != 1 or self.in_channels != output_dim:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, output_dim, kernel_size=1, stride=stride),
                nn.BatchNorm2d(output_dim)
            )
        first_block = block(input_dim=self.in_channels,
                            output_dim=output_dim,
                            downsample=downsample,
                            stride=stride)
        layers.append(first_block)
        self.in_channels = output_dim
        for i in range(blocks - 1):
            layers.append(block(input_dim=self.in_channels, output_dim=output_dim))
        return nn.Sequential(*layers)

    def _initial_conv_and_pool(self, x: Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def _final_pool_and_predict(self, x: Tensor):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        # stage 1. Apply simple conv to initial signal, then after batchnorm and activation apply maxpooling to get
        # embedding with lower_dim
        x = self._initial_conv_and_pool(x)
        # stage 2. Sequentially apply resnet_layers (conv+skip connection) to embedding
        for idx, resnet_layer in enumerate(self.resnet_layers):
            x = resnet_layer(x)
        # stage 3. Sequentially apply resnet_layers (conv+skip connection) to embedding
        x = self._final_pool_and_predict(x)

        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
