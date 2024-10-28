from typing import List, Type, Union, Dict
from typing import Dict, List, Optional, Type, Union

from typing import Set, Any
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torchvision.models.resnet import conv1x1, conv3x3
from typing import Optional
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
        sizes: Dict[str, Tensor],
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(sizes["conv1"][1], sizes["conv1"][0], stride=stride)
        self.bn1 = norm_layer(sizes["conv1"][0])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(sizes["conv2"][1], sizes["conv2"][0])
        self.bn2 = norm_layer(sizes["conv2"][0])
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
            sizes["conv1"][1],
            sizes["conv1"][0],
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(sizes["conv1"][0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block=block, blocks=layers[0], sizes=sizes["layer1"]
        )
        self.layer2 = self._make_layer(
            block=block, blocks=layers[1], sizes=sizes["layer2"], stride=2
        )
        self.layer3 = self._make_layer(
            block=block, blocks=layers[2], sizes=sizes["layer3"], stride=2
        )
        self.layer4 = self._make_layer(
            block=block, blocks=layers[3], sizes=sizes["layer4"], stride=2
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(sizes["fc"][1], sizes["fc"][0])

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        blocks: int,
        sizes: Dict,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if "downsample" in sizes.keys():
            downsample = nn.Sequential(
                conv1x1(sizes["downsample"][1], sizes["downsample"][0], stride=stride),
                nn.BatchNorm2d(sizes["downsample"][0]),
            )
        layers = [block(sizes=sizes[0], stride=stride, downsample=downsample)]
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

class IDecomposed:
    def compose_weight_for_inference(self):
        self.inference_dict[self.forward_mode]()
        self.inference_mode = True

    def decompose(self, W):
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        self.U = Parameter(U)
        self.S = Parameter(S)
        self.Vh = Parameter(Vh)
        self.register_parameter('weight', None)
        self.inference_mode = False

    def compose(self) -> None:
        """Compose the weight matrix from singular value decomposition.
        Replaces U, S, Vh matrices with weights such that weights = U * S * Vh.
        """
        W = self.U @ torch.diag(self.S) @ self.Vh
        self.weight = Parameter(W.reshape(self.decomposing['compose_shape']).permute(self.decomposing['permute']))
        self.register_parameter('U', None)
        self.register_parameter('S', None)
        self.register_parameter('Vh', None)
        self.decomposing = None

    def set_U_S_Vh(self, u: torch.Tensor, s: torch.Tensor, vh: torch.Tensor) -> None:
        """Update U, S, Vh matrices.
        Raises:
            Assertion Error: If ``self.decomposing`` is False.
        """
        assert self.decomposing is not None, "for setting U, S and Vh, the model must be decomposed"
        self.U = Parameter(u)
        self.S = Parameter(s)
        self.Vh = Parameter(vh)

    def _one_layer_forward(self):
        raise NotImplementedError
    
    def _two_layers_forward(self):
        raise NotImplementedError
    
    def _three_layers_forward(self):
        raise NotImplementedError



class DecomposedConv2d(Conv2d, IDecomposed):
    """Extends the Conv2d layer by implementing the singular value decomposition of
    the weight matrix.

    Args:
        base_module:  The convolutional layer whose parameters will be copied
        decomposing_mode: ``'channel'`` or ``'spatial'`` weights reshaping method.
            If ``None`` create layers without decomposition.
        forward_mode: ``'one_layer'``, ``'two_layers'`` or ``'three_layers'`` forward pass calculation method.
    """

    def __init__(
            self,
            base_module: Conv2d,
            decomposing_mode: Optional[str] = 'channel',
            forward_mode: str = 'two_layers',
            device=None,
            dtype=None,
    ) -> None:

        parameter_value_check(
            "forward_mode", forward_mode, {"one_layer", "two_layers", "three_layers"}
        )

        if forward_mode != 'one_layer':
            assert base_module.padding_mode == 'zeros', \
                "only 'zeros' padding mode is supported for '{forward_mode}' forward mode."
            assert base_module.groups == 1, f"only 1 group is supported for '{forward_mode}' forward mode."

        super().__init__(
            base_module.in_channels,
            base_module.out_channels,
            base_module.kernel_size,
            base_module.stride,
            base_module.padding,
            base_module.dilation,
            base_module.groups,
            (base_module.bias is not None),
            base_module.padding_mode,
            device,
            dtype,
        )
        self.load_state_dict(base_module.state_dict())
        self.forward_mode = forward_mode
        self.inference_mode = False
        if decomposing_mode is not None:
            self.decompose(decomposing_mode)
        else:
            self.U = None
            self.S = None
            self.Vh = None
            self.decomposing = None
        self.inference_dict = {'one_layer': self._one_layer_forward,
                               'two_layers': self._two_layers_forward,
                               'three_layers': self._three_layers_forward}
        
    def decompose(self, decomposing_mode: Optional[str] = None) -> None:
        """Decomposes the weight matrix in singular value decomposition.
        Replaces the weights with U, S, Vh matrices such that weights = U * S * Vh.
        Args:
            decomposing_mode: ``'channel'`` or ``'spatial'`` weights reshaping method.
        Raises:
            ValueError: If ``decomposing_mode`` not in valid values.
        """
        self.__set_decomposing_params(decomposing_mode=decomposing_mode)
        W = self.weight.permute(self.decomposing['permute']).reshape(self.decomposing['decompose_shape'])
        super().decompose(W)

    def __set_decomposing_params(self, decomposing_mode):
        n, c, w, h = self.weight.size()
        compose_shape = (n, c, w, h)
        decomposing_modes = {
            "channel": {
                "type": "channel",
                "permute": (0, 1, 2, 3),
                "decompose_shape": (n, c * w * h),
                "compose_shape": compose_shape,
                "U shape": (n, 1, 1, -1),
                "U": {
                    "stride": 1,
                    "padding": 0,
                    "dilation": 1,
                },
                "Vh shape": (-1, c, w, h),
                "Vh": {
                    "stride": self.stride,
                    "padding": self.padding,
                    "dilation": self.dilation,
                },
            },
            "spatial": {
                "type": "spatial",
                "permute": (0, 2, 1, 3),
                "decompose_shape": (n * w, c * h),
                "compose_shape": compose_shape,
                "U shape": (n, w, 1, -1),
                "U": {
                    "stride": (self.stride[0], 1),
                    "padding": (self.padding[0], 0),
                    "dilation": (self.dilation[0], 1),
                },
                "Vh shape": (-1, c, 1, h),
                "Vh": {
                    "stride": (1, self.stride[1]),
                    "padding": (0, self.padding[1]),
                    "dilation": (1, self.dilation[1]),
                },
            },
        }
        parameter_value_check(
            "decomposing_mode", decomposing_mode, set(decomposing_modes.keys())
        )
        self.decomposing = decomposing_modes[decomposing_mode]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.inference_mode:
            self.compose_weight_for_inference()
        if self.forward_mode == 'one_layer':
            x = self._conv_forward(input, self.weight, self.bias)
            return x
        if self.forward_mode == 'two_layers':
            x = conv2d(input=input, weight=self.Vh, groups=self.groups, **self.decomposing['Vh'])
            x = conv2d(input=x, weight=self.U, bias=self.bias, **self.decomposing['U'])
            return x
        if self.forward_mode == "three_layers":
            x = conv2d(
                input=input,
                weight=self.Vh,
                groups=self.groups,
                **self.decomposing["Vh"],
            )
            x = conv2d(input=x, weight=self.S, padding=0)
            return conv2d(
                input=x, weight=self.U, bias=self.bias, **self.decomposing["U"]
            )

    def _one_layer_forward(self):
        W = self.U @ torch.diag(self.S) @ self.Vh
        self.weight = W.reshape(self.decomposing["compose_shape"]).permute(
            self.decomposing["permute"]
        )

    def _two_layers_forward(self):
        SVh = torch.diag(self.S) @ self.Vh
        self.Vh = Parameter(SVh.view(self.decomposing["Vh shape"]))
        self.U = Parameter(
            self.U.reshape(self.decomposing["U shape"]).permute(0, 3, 1, 2)
        )

    def _three_layers_forward(self):
        self.S = torch.diag(self.S).view([len(self.S), len(self.S), 1, 1])
        self.Vh = self.Vh.view(self.decomposing["Vh shape"])
        self.U = self.U.view(self.decomposing["U shape"]).permute(0, 3, 1, 2)

    def set_U_S_Vh(
        self, u: torch.Tensor, s: torch.Tensor, vh: torch.Tensor, rank: int = 1
    ) -> None:
        """Update U, S, Vh matrices.
        Raises:
            Assertion Error: If ``self.decomposing`` is False.
        """
        assert (
            self.decomposing is not None
        ), "for setting U, S and Vh, the model must be decomposed"
        self.U = Parameter(u)
        self.S = Parameter(s)
        self.Vh = Parameter(vh)
        n, c, w, h = self.Vh.size()
        self.Vh = Parameter(self.Vh.reshape((n, c * w * h)))
        n, c, w, h = self.U.size()
        self.U = Parameter(self.U.reshape((n, c * w * h)))


class DecomposedLinear(nn.Linear, IDecomposed):
    """Extends the Linear layer by implementing the singular value decomposition of
    the weight matrix.

    Args:
        base_module:  The linear layer whose parameters will be copied
        decomposing: ``True`` or ``False``
            If ``False`` create layers without decomposition.
        forward_mode: ``'one_layer'``, ``'two_layers'`` or ``'three_layers'`` forward pass calculation method.
    """

    def __init__(
            self,
            base_module: nn.Linear,
            decomposing_mode: bool = True,
            forward_mode: str = 'two_layers',
            device=None,
            dtype=None,
    ) -> None:

        super().__init__(
            in_features=base_module.in_features,
            out_features=base_module.out_features,
            bias=True if base_module.bias is not None else False,
            device=device,
            dtype=dtype,
        )
        self.load_state_dict(base_module.state_dict())
        self.forward_mode = forward_mode
        self.decomposing = decomposing_mode
        self.inference_mode = False
        self.inference_dict = {'one_layer': self._one_layer_forward,
                               'two_layers': self._two_layers_forward}
        if decomposing_mode:
            self.decompose()
        else:
            self.U = None
            self.S = None
            self.Vh = None
            self.decomposing = None

    def decompose(self) -> None:
        """Decomposes the weight matrix in singular value decomposition.
        Replaces the weights with U, S, Vh matrices such that weights = U * S * Vh.
        Args:
        Raises:
            ValueError: If ``decomposing_mode`` not in valid values.
        """
        W = self.weight
        super().decompose(W)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.inference_mode:
            self.compose_weight_for_inference()
        if self.forward_mode == "one_layer":
            x = input @ self.W.T
        if self.forward_mode == "two_layers":
            x = input @ self.Vh.T
            x = x @ self.U.T
        if self.bias is not None:
            x += self.bias
        return x

    def _one_layer_forward(self):
        self.W = Parameter((self.U * self.S) @ self.Vh) 

    def _two_layers_forward(self):
        singular_diag = torch.diag(self.S)
        if singular_diag.shape[1] != self.Vh.shape[0]:
            self.Vh = Parameter(self.Vh)
        self.Vh = Parameter(singular_diag @ self.Vh)
        

class DecomposedEmbedding(nn.Embedding, IDecomposed):
    """Extends the Embedding layer by implementing the singular value decomposition of
    the weight matrix.

    Args:
        base_module:  The linear layer whose parameters will be copied
        decomposing: ``True`` or ``False``
            If ``False`` create layers without decomposition.
        forward_mode: ``'one_layer'``, ``'two_layers'`` or ``'three_layers'`` forward pass calculation method.
    """

    def __init__(
            self,
            base_module: nn.Embedding,
            decomposing_mode: bool = True,
            forward_mode: str = 'two_layers',
            device=None,
            dtype=None,
    ) -> None:
        super().__init__(
            num_embeddings=base_module.num_embeddings,
            embedding_dim=base_module.embedding_dim,
            device=device,
            dtype=dtype,
        )
        self.load_state_dict(base_module.state_dict())
        self.forward_mode = forward_mode
        self.decomposing = decomposing_mode
        self.inference_mode = False
        self.inference_dict = {'one_layer': self._one_layer_forward,
                               'two_layers': self._two_layers_forward}
        if decomposing_mode:
            self.decompose()
        else:
            self.U = None
            self.S = None
            self.Vh = None
            self.decomposing = None

    def decompose(self) -> None:
        W = self.weight
        super().decompose(W)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.inference_mode:
            self.compose_weight_for_inference()
        if self.forward_mode == "one_layer":
            x = self.W[input]
        if self.forward_mode == "two_layers":
            x = self.U[input]
            x = x @ self.Vh
        return x

    def _one_layer_forward(self):
        self.W = Parameter((self.U * self.S) @ self.Vh)

    def _two_layers_forward(self):
        singular_diag = torch.diag(self.S)
        if singular_diag.shape[1] != self.Vh.shape[0]:
            self.Vh = Parameter(self.Vh)
        self.Vh = Parameter(singular_diag @ self.Vh)
