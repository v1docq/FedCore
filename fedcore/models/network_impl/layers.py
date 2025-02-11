import abc

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

class IDecomposed(abc.ABC):
    _weights = ['weight']
    _compose_mode_merices = {
        'one_layer': ['W'],
        'two_layers': ['U', 'Vh'],
        'three_layers': ['U', 'S', 'Vh']
    }

    def __init__(self, compose_mode, decomposing_mode,):
        self.compose_mode : str = compose_mode
        self.inference_mode = False
        self.decomposing_mode = decomposing_mode
        if decomposing_mode is not None:
            self.decompose()
            self._current_forward = self._forward3
        else:
            self.U = None
            self.S = None
            self.Vh = None
            self.decomposing = None
            self._current_forward = self._forward1
        self._compose_dict = {'one_layer': self._one_layer_compose,
                               'two_layers': self._two_layers_compose,
                               'three_layers': self._three_layers_compose}
        self._forward_dict = {'one_layer': self._forward1,
                               'two_layers': self._forward2,
                               'three_layers': self._forward3}


    def compose_weight_for_inference(self):
        self._compose_dict[self.compose_mode]()
        self.inference_mode = True
        self._current_forward = self._forward_dict[self.compose_mode]

    def _get_weights(self):
        return self.weight
    
    def _get_threshold(self):
        return None

    def decompose(self, W):
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        assert U.device.type == W.device.type
        self.U = Parameter(U)
        self.S = Parameter(S)
        self.Vh = Parameter(Vh)
        self.register_parameter('weight', None)
        delattr(self, 'weight')
        self.inference_mode = False

    def compose(self: nn.Module):
        W = self._get_composed_weight()
        self.weight = Parameter(W)
        assert self.U.device.type == self.weight.device.type
        # self.weight = Parameter(W.reshape(self.decomposing['compose_shape']).permute(self.decomposing['permute']))
        self.register_parameter('U', None)
        self.register_parameter('S', None)
        self.register_parameter('Vh', None)
        self.decomposing = None

    def set_U_S_Vh(self, u: torch.Tensor, s: torch.Tensor, vh: torch.Tensor) -> None:
        """Update U, S, Vh matrices."""
        self.U = Parameter(u)
        self.S = Parameter(s)
        self.Vh = Parameter(vh)

    def get_U_S_Vh(self):
        return self.U, self.S, self.Vh
    
    def _get_composed_weight(self): #TODO add assertion if module is decomposed. Forward mode support
        if self.compose_mode == 'two_layers':
            W = self.U @ self.Vh
        elif self.compose_mode == 'three_layers':
            W = (self.U * self.S) @ self.Vh
        else:
            W = self.weight
        return W
    
    @abc.abstractmethod
    def _forward1(self, x): pass
    
    @abc.abstractmethod
    def _forward2(self, x): pass

    @abc.abstractmethod
    def _forward3(self, x): pass

    def _one_layer_compose(self):
        self.W = Parameter((self.U * self.S) @ self.Vh) 
        self.register_parameter('U', None)
        self.register_parameter('S', None)
        self.register_parameter('Vh', None)
    
    def _two_layers_compose(self: nn.Module):
        singular_diag = torch.diag(self.S)
        self.register_parameter('S', None)
        # if singular_diag.shape[1] != self.Vh.shape[0]:
        #     self.Vh = Parameter(self.Vh)
        self.Vh = Parameter(singular_diag @ self.Vh)
    
    def _three_layers_compose(self): pass


class DecomposedConv2d(Conv2d, IDecomposed):
    """Extends the Conv2d layer by implementing the singular value decomposition of
    the weight matrix.

    Args:
        base_module:  The convolutional layer whose parameters will be copied
        decomposing_mode: ``'channel'`` or ``'spatial'`` weights reshaping method.
            If ``None`` create layers without decomposition.
        compose_mode: ``'one_layer'``, ``'two_layers'`` or ``'three_layers'`` forward pass calculation method.
    """

    def __init__(
            self,
            base_module: Conv2d,
            decomposing_mode: Optional[str] = 'channel',
            compose_mode: str = 'two_layers',
            device=None,
            dtype=None,
    ) -> None:

        if compose_mode != 'one_layer':
            assert base_module.padding_mode == 'zeros', \
                "only 'zeros' padding mode is supported for '{forward_mode}' forward mode."
            assert base_module.groups == 1, f"only 1 group is supported for '{compose_mode}' forward mode."

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
        IDecomposed.__init__(self, compose_mode, decomposing_mode,)

    def decompose(self) -> None:
        """Decomposes the weight matrix in singular value decomposition.
        Replaces the weights with U, S, Vh matrices such that weights = U * S * Vh.
        Args:
            decomposing_mode: ``'channel'`` or ``'spatial'`` weights reshaping method.
        Raises:
            ValueError: If ``decomposing_mode`` not in valid values.
        """
        self.__set_decomposing_params(decomposing_mode=self.decomposing_mode)
        W = self._get_weights()
        super().decompose(W)
        self._three_layers_compose()
    
    def _get_weights(self):
        return self.weight.permute(self.decomposing['permute']).reshape(self.decomposing['decompose_shape'])

    def __set_decomposing_params(self, decomposing_mode):
        out, in_, k1, k2 = self.weight.size()
        compose_shape = (out, in_, k1, k2)
        #TODO check if it is effective to choose the largest of (out, in) for decomposition.
        decomposing_modes = {
            "channel": {
                "type": "channel",
                "permute": (0, 1, 2, 3),
                "decompose_shape": (out, in_ * k1 * k2),
                "compose_shape": compose_shape,
                "U2d": (out, -1),
                "U4d": (out, 1, 1, -1),
                "U": {
                    "stride": 1,
                    "padding": 0,
                    "dilation": 1,
                },
                "Vh2d": (-1, in_ * k1 * k2),
                "Vh4d": (-1, in_, k1, k2),
                "Vh": {
                    "stride": self.stride,
                    "padding": self.padding,
                    "dilation": self.dilation,
                },
            },
            "spatial": {
                "type": "spatial",
                "permute": (0, 2, 1, 3),
                "decompose_shape": (out * k1, in_ * k2),
                "compose_shape": compose_shape,
                "U2d": (out * k1, -1),
                "U4d": (out, k1, 1, -1),
                "U": {
                    "stride": (self.stride[0], 1),
                    "padding": (self.padding[0], 0),
                    "dilation": (self.dilation[0], 1),
                },
                "Vh2d": (-1, in_ * k2),
                "Vh4d": (-1, in_, 1, k2),
                "Vh": {
                    "stride": (1, self.stride[1]),
                    "padding": (0, self.padding[1]),
                    "dilation": (1, self.dilation[1]),
                },
            },
        }
        self.decomposing = decomposing_modes[decomposing_mode]

    def compose(self) -> None:
        """Compose the weight matrix from singular value decomposition.
        Replaces U, S, Vh matrices with weights such that weights = U * S * Vh.
        """
        assert all([self.U.ndim == 2, self.Vh.ndim == 2])
        self.weight = self._get_composed_weight()
        self.register_parameter('U', None)
        self.register_parameter('S', None)
        self.register_parameter('Vh', None)
        # self.decomposing = None

    def compose_weight_for_inference(self):
        # here we assume that USVh are set as 2d matrices & training is in 3L mode
        if not self.compose_mode == 'three_layers':
            self._anti_three_layers_compose()
            return super().compose_weight_for_inference()
        else:
            self.inference_mode = True

    def _get_composed_weight(self):
        #TODO suppor for S matrix deletion from another 
        W = self.U @ torch.diag(self.S) @ self.Vh
        W = Parameter(W.reshape(self.decomposing['compose_shape']).permute(self.decomposing['permute']))
        return W
    
    def _forward1(self, x):
        return super().forward(x)
    
    def _forward2(self, x):
        x = conv2d(input=x, weight=self.Vh, groups=self.groups, **self.decomposing['Vh'])
        x = conv2d(input=x, weight=self.U, bias=self.bias, **self.decomposing['U'])
        return x
    
    def _forward3(self, x):
        x = conv2d(
            input=x,
            weight=self.Vh,
            groups=self.groups,
            **self.decomposing["Vh"],
        )
        x = conv2d(
            input=x, weight=self.S * self.U, bias=self.bias, **self.decomposing["U"]
        )
        return x
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self._current_forward(input)
        return x
       
    def _one_layer_compose(self):
        W = self.U @ torch.diag(self.S) @ self.Vh
        self.weight = Parameter(W.reshape(self.decomposing["compose_shape"]).permute(
            self.decomposing["permute"]
        ))

    def _two_layers_compose(self):
        SVh = torch.diag(self.S) @ self.Vh
        self.Vh = Parameter(SVh.view(*self.decomposing["Vh4d"]))
        self.U = Parameter(
            self.U.view(self.decomposing["U4d"]).permute(0, 3, 1, 2)
        )

    def _three_layers_compose(self):
        super().set_U_S_Vh(
            self.U.view(self.decomposing["U4d"]).permute(0, 3, 1, 2),
            self.S[..., None, None],
            self.Vh.view(self.decomposing["Vh4d"])
        )

    def _anti_three_layers_compose(self):
        super().set_U_S_Vh(
            self.U.permute(0, 2, 3, 1).view(*self.decomposing['U2d']),
            self.S[..., 0, 0],
            self.Vh.view(*self.decomposing["Vh2d"])
        )

    def set_U_S_Vh(
        self, u: torch.Tensor, s: torch.Tensor, vh: torch.Tensor, rank: int = 1
    ) -> None:
        self.eval()
        """Update U, S, Vh matrices.
        Raises:
            Assertion Error: If ``self.decomposing`` is False.
        """
        assert (
            self.decomposing is not None
        ), "for setting U, S and Vh, the model must be decomposed"
        assert u.ndim == 2, 'Expected 2d tensors'
        super().set_U_S_Vh(u, s, vh)
        self._three_layers_compose()

    def get_U_S_Vh(self):
        assert not self.inference_mode, 'Only model in training mode have all matrices intact'
        return (
            self.U.reshape(*self.decomposing['U2d']),
            self.S.squeeze(),
            self.Vh.reshape(*self.decomposing['Vh2d'])
        )

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
            compose_mode: str = 'two_layers',
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
        assert self.bias is not None
        IDecomposed.__init__(self, compose_mode, decomposing_mode,)

    def decompose(self) -> None:
        W = self._get_weights()
        super().decompose(W)
    
    def _forward2(self, x):
        x = torch.nn.functional.linear(x, self.Vh)
        x = torch.nn.functional.linear(x, self.U, self.bias)
        return x 
    
    def _forward3(self, x):
        x = torch.nn.functional.linear(x, self.Vh)
        x = torch.nn.functional.linear(x, (self.U * self.S), self.bias)
        return x 
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self._current_forward(input)
        return x
    
    def _forward1(self, x):
        return super().forward(x)
    

class DecomposedEmbedding(nn.Embedding, IDecomposed):
    """Extends the Embedding layer by implementing the singular value decomposition of
    the weight matrix.

    Args:
        base_module:  The linear layer whose parameters will be copied
        decomposing: ``True`` or ``False``
            If ``False`` create layers without decomposition.
        compose_mode: ``'one_layer'``, ``'two_layers'`` or ``'three_layers'`` forward pass calculation method.
    """

    def __init__(
            self,
            base_module: nn.Embedding,
            decomposing_mode: bool = True,
            compose_mode: str = 'two_layers',
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
        IDecomposed.__init__(self, compose_mode, decomposing_mode,)

    def decompose(self) -> None:
        W = self._get_weights()
        super().decompose(W)
    
    def _forward1(self, x):
        return super().forward(x)

    def _forward2(self, x):
        x = nn.functional.embedding(x, self.U)
        x = nn.functional.linear(x, self.Vh)
        return x
    
    def _forward3(self, x):
        x = nn.functional.embedding(x, (self.U * self.S))
        x = nn.functional.linear(x, self.Vh)
        return x
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self._current_forward(input)
        return x
