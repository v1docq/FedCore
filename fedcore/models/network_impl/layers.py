from typing import Dict, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torchvision.models.resnet import conv1x1, conv3x3


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            sizes: Dict[str, Tensor],
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(
            sizes['conv1'][1], sizes['conv1'][0], stride=stride)
        self.bn1 = norm_layer(sizes['conv1'][0])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(sizes['conv2'][1], sizes['conv2'][0])
        self.bn2 = norm_layer(sizes['conv2'][0])
        self.downsample = downsample
        self.stride = stride
        self.register_buffer('indices', torch.zeros(
            sizes['indices'], dtype=torch.int))

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
        self.conv1 = conv1x1(sizes['conv1'][1], sizes['conv1'][0])
        self.bn1 = norm_layer(sizes['conv1'][0])
        self.conv2 = conv3x3(
            sizes['conv2'][1], sizes['conv2'][0], stride=stride)
        self.bn2 = norm_layer(sizes['conv2'][0])
        self.conv3 = conv1x1(sizes['conv3'][1], sizes['conv3'][0])
        self.bn3 = norm_layer(sizes['conv3'][0])
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.register_buffer('indices', torch.zeros(
            sizes['indices'], dtype=torch.int))

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
        block: ``'BasicBlock'`` or ``'Bottleneck'``.
        layers: Number of blocks on each layer.
        sizes: Sizes of layers.
        num_classes: Number of classes.
    """

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            sizes: Dict,
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            sizes['conv1'][1], sizes['conv1'][0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(sizes['conv1'][0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block=block,
            blocks=layers[0],
            sizes=sizes['layer1']
        )
        self.layer2 = self._make_layer(
            block=block,
            blocks=layers[1],
            sizes=sizes['layer2'],
            stride=2)
        self.layer3 = self._make_layer(
            block=block,
            blocks=layers[2],
            sizes=sizes['layer3'],
            stride=2)
        self.layer4 = self._make_layer(
            block=block,
            blocks=layers[3],
            sizes=sizes['layer4'],
            stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(sizes['fc'][1], sizes['fc'][0])

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            blocks: int,
            sizes: Dict,
            stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if 'downsample' in sizes.keys():
            downsample = nn.Sequential(
                conv1x1(sizes['downsample'][1],
                        sizes['downsample'][0], stride=stride),
                nn.BatchNorm2d(sizes['downsample'][0]),
            )
        layers = [
            block(
                sizes=sizes[0],
                stride=stride,
                downsample=downsample
            )
        ]
        for i in range(1, blocks):
            layers.append(block(sizes=sizes[i]))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)