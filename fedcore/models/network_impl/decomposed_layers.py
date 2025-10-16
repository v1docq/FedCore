import abc
from typing import *

import torch
import torch.nn as nn
from torch.nn import Conv2d, Parameter
from torch.nn.functional import conv1d, conv2d, conv_transpose2d, linear, embedding

from fedcore.algorithm.low_rank.decomposer import DECOMPOSERS
from fedcore.architecture.utils.misc import count_params
from fedcore.architecture.abstraction.placeholders import ParameterPlaceHolder

__all__ = [
    'IDecomposed',
    'DecomposedConv2d',
    'DecomposedLinear',
    'DecomposedEmbedding',
    'DecomposedConv1d'
]

def _diag_tensor_check(t: torch.Tensor):
    return torch.diag(t) if t.ndim == 1 else t
    

class IDecomposed(abc.ABC):
    _weight_name = ['weight']
    _compose_mode_matrices = {
        'one_layer': ['W'],
        'two_layers': ['U', 'Vh'],
        'three_layers': ['U', 'S', 'Vh']
    }

    def __init__(self, decomposing_mode, method: Literal['svd', 'rsvd', 'cur']='svd', compose_mode=None,
                 decomposer_params: dict = None):
        self.compose_mode : str = compose_mode
        self.inference_mode = False
        self.decomposing_mode = decomposing_mode
        self.method = method
        self.decomposer_params = decomposer_params or {}
        if decomposing_mode is not None:
            self.decompose()
            self._current_forward = self._forward3
        else:
            self.U = None
            self.S = None
            self.Vh = None
            self.decomposing = None
            self._current_forward = self._forward1
        self._initial_params_num = count_params(self)
        self._compose_dict = {'one_layer': self._one_layer_compose,
                               'two_layers': self._two_layers_compose,
                               'three_layers': self._three_layers_compose}
        self._forward_dict = {'one_layer': self._forward1,
                               'two_layers': self._forward2,
                               'three_layers': self._forward3}

    def compose_weight_for_inference(self):
        self.compose_mode = self.compose_mode or self._evaluate_compose_mode()
        self._compose_dict[self.compose_mode]()
        self._current_forward = self._forward_dict[self.compose_mode]

    def _evaluate_compose_mode(self: nn.Module):
        """Evaluate the best composition mode to minimize parameters.
        
        Returns:
            'one_layer': Compose all into single weight (when no compression achieved)
            'two_layers': Keep U and Vh separate (preserves low-rank compression)
        """
        nparams = [
            p.numel() for p in self.parameters()
        ]
        l1 = self._initial_params_num
        l2 = nparams[0] + nparams[-1]
        return 'two_layers' if l2 < l1 else 'one_layer'

    def _get_weights(self):
        return self.weight
    
    def _get_threshold(self):
        return None

    def decompose(self, W):
        decomposer_cls = DECOMPOSERS[self.method]
        decomposer = decomposer_cls(**self.decomposer_params)
        U, S, Vh = decomposer.decompose(W)
        assert U.device.type == W.device.type
        self.set_U_S_Vh(U, S, Vh)
        ParameterPlaceHolder(self.weight).set_as(self, 'weight')
        # self.register_parameter('weight', None)
        # self.inference_mode = False

    def compose(self: nn.Module):
        W = self._get_composed_weight()
        self.register_parameter('weight', Parameter(W))
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
    
    def _eliminate_extra_params(self, names):
        for name in names:
            self.register_parameter(name, None)
    
    def _get_composed_weight(self): #TODO add assertion if module is decomposed. Forward mode support
        if self.compose_mode == 'two_layers':
            W = self.U @ self.Vh
        elif self.compose_mode == 'three_layers':
            W = self.U @ _diag_tensor_check(self.S) @ self.Vh
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
        self.register_parameter('U', Parameter((self.U * self.S) @ self.Vh))
        self._eliminate_extra_params(['S', 'Vh'])

    def _two_layers_compose(self: nn.Module):
        singular_diag = _diag_tensor_check(self.S)
        self.register_parameter('Vh', Parameter(singular_diag @ self.Vh))
        self._eliminate_extra_params(['S'])
    
    def _three_layers_compose(self): 
        self._eliminate_extra_params([])

    def _anti_three_layers_compose(self):
        pass


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
            decomposer: Optional[str] = 'svd',
            compose_mode: str = None,
            decomposer_params: dict = None,
            device=None,
            dtype=None,
    ) -> None:
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
        IDecomposed.__init__(self, decomposing_mode, decomposer, compose_mode, decomposer_params)

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
        delattr(self, 'weight')
        self.register_parameter('weight', self._get_composed_weight())
        self._eliminate_extra_params(('U', 'S', 'Vh'))

    # def compose_weight_for_inference(self):
    #     # here we assume that USVh are set as 2d matrices & training is in 3L mode
    #     if not self.compose_mode == 'three_layers':
    #         # self._anti_three_layers_compose()
    #         return super().compose_weight_for_inference()
    #     else:
    #         self.inference_mode = True

    def _get_composed_weight(self):
        #TODO suppor for S matrix deletion from another 
        W = self.U @ torch.diag(self.S) @ self.Vh
        W = Parameter(W.reshape(self.decomposing['compose_shape']).permute(self.decomposing['permute']))
        return W
    
    def _forward1(self, x):
        if self.bias is not None:
            return torch.nn.functional.conv2d(x, self.U, self.bias, 
                self.stride, self.padding, self.dilation, self.groups)
        else:
            return torch.nn.functional.conv2d(x, self.U, None,
                self.stride, self.padding, self.dilation, self.groups)
    
    def _forward2(self, x):
        x = conv2d(input=x, weight=self.Vh, groups=self.groups, **self.decomposing['Vh'])
        if self.bias is not None:
            x = conv2d(input=x, weight=self.U, bias=self.bias, **self.decomposing['U'])
        else:
            x = conv2d(input=x, weight=self.U, bias=None, **self.decomposing['U'])
        return x
    
    def _forward3(self, x):
        x = conv2d(
            input=x,
            weight=self.Vh,
            groups=self.groups,
            **self.decomposing["Vh"],
        )
        if self.bias is not None:
            x = conv2d(
                input=x, weight=self.S * self.U, bias=self.bias, **self.decomposing["U"],             
            )
        else:
            x = conv2d(
                input=x, weight=self.S * self.U, bias=None, **self.decomposing["U"],             
            )
        return x
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self._current_forward(input)
        return x
       
    def _one_layer_compose(self):
        W = (self.U * self.S.unsqueeze(0)) @ self.Vh
        self.register_parameter('U', Parameter(W.reshape(self.decomposing["compose_shape"]).permute(
            self.decomposing["permute"]
        )))
        self._eliminate_extra_params(['S', 'Vh'])

    def _two_layers_compose(self):
        SVh = torch.diag(self.S) @ self.Vh
        self.register_parameter('Vh', Parameter(SVh.view(*self.decomposing["Vh4d"])))
        self.register_parameter('U', 
            Parameter(self.U.view(*self.decomposing["U4d"]).permute(0, 3, 1, 2)))
        self._eliminate_extra_params(['S'])

    def _three_layers_compose(self):
        super().set_U_S_Vh(
            self.U.view(*self.decomposing["U4d"]).permute(0, 3, 1, 2),
            self.S[..., None, None],
            self.Vh.view(*self.decomposing["Vh4d"])
        )
        self._eliminate_extra_params([])

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
        # self._three_layers_compose()

    def get_U_S_Vh(self):
        # assert not self.inference_mode, 'Only model in training mode have all matrices intact'
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
            decomposer: Optional[str] = 'svd',
            compose_mode: str = None,
            decomposer_params: dict = None,
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
        # assert self.bias is not None
        IDecomposed.__init__(self, decomposing_mode, decomposer, compose_mode, decomposer_params)

    def decompose(self) -> None:
        W = self._get_weights()
        super().decompose(W)
    
    def _forward2(self, x):
        x = linear(x, self.Vh)
        if self.bias is not None:
            x = linear(x, self.U, self.bias)
        else:
            x = linear(x, self.U)
        return x 
    
    def _forward3(self, x):
        x = linear(x, self.Vh)
        if self.bias is not None:
            x = linear(x, (self.U * self.S), self.bias)
        else:
            x = linear(x, (self.U * self.S))
        return x 
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self._current_forward(input)
        return x
    
    def _forward1(self, x):
        if self.bias is not None:
            return torch.nn.functional.linear(x, self.U, self.bias)
        else:
            return torch.nn.functional.linear(x, self.U)
    

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
            decomposer: Optional[str] = 'svd',
            compose_mode: str = None,
            decomposer_params: dict = None,
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
        IDecomposed.__init__(self, decomposing_mode, decomposer, compose_mode, decomposer_params)

    def decompose(self) -> None:
        W = self._get_weights()
        super().decompose(W)
    
    def _forward1(self, x):
        return torch.nn.functional.embedding(x, self.U, 
            self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

    def _forward2(self, x):
        x = embedding(x, self.U)
        x = linear(x, self.Vh.T)
        return x
    
    def _forward3(self, x):
        x = embedding(x, (self.U * self.S))
        x = linear(x, self.Vh.T)
        return x
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self._current_forward(input)
        return x
    
class DecomposedConvTranspose2d(nn.ConvTranspose2d, DecomposedConv2d):
    def __init__(
            self,
            base_module: nn.ConvTranspose2d,
            decomposing_mode: Optional[str] = 'channel',
            decomposer: Optional[str] = 'svd',
            compose_mode: str = None,
            decomposer_params: dict = None,
            device=None,
            dtype=None,
    ) -> None:
        super().__init__(
            base_module.in_channels,
            base_module.out_channels,
            base_module.kernel_size,
            base_module.stride,
            base_module.padding,
            base_module.output_padding,
            base_module.groups,
            (base_module.bias is not None),
            base_module.dilation,
            base_module.padding_mode,
            device,
            dtype,
        )
        self.load_state_dict(base_module.state_dict())
        IDecomposed.__init__(self, decomposing_mode, decomposer, compose_mode, decomposer_params)

    def __set_decomposing_params(self, decomposing_mode):
        in_channels, out_channels, kernel_height, kernel_width = self.weight.size()
        decomposing_modes = {
            "channel": {
                "type": "channel",
                "permute": (0, 2, 3, 1),
                "decompose_shape": (in_channels, kernel_height * kernel_width * out_channels),
                "compose_shape": (in_channels, kernel_height, kernel_width, out_channels),
                "compose_permute": (0, 3, 1, 2),
                "U2d": (in_channels, -1),
                "U4d": (in_channels, -1, 1, 1),
                "U4d_permute": (0, 1, 2, 3),
                "U": {
                    "stride": 1,
                    "padding": 0,
                    "dilation": 1,
                },
                "Vh2d": (-1, out_channels * kernel_height * kernel_width),
                "Vh4d": (-1, out_channels, kernel_height, kernel_width),
                "Vh4d_permute": (0, 1, 2, 3),
                "Vh": {
                    "stride": self.stride,
                    "padding": self.padding,
                    "dilation": self.dilation,
                },
            },
            "spatial": {
                "type": "spatial",
                "permute": (0, 2, 3, 1),
                "inverse_permute": (0, 3, 1, 2),
                "decompose_shape": (in_channels * kernel_height, out_channels * kernel_width),
                "compose_shape": (in_channels, kernel_height, kernel_width, out_channels),
                "U2d": (in_channels * kernel_height, -1),
                "U4d": (in_channels, kernel_height, 1, -1),
                "U": {
                    "stride": (self.stride[0], 1),
                    "padding": (self.padding[0], 0),
                    "dilation": (self.dilation[0], 1),
                },
                "Vh2d": (-1, out_channels * kernel_width),
                "Vh4d": (-1, 1, kernel_width, out_channels),
                "Vh": {
                    "stride": (1, self.stride[1]),
                    "padding": (0, self.padding[1]),
                    "dilation": (1, self.dilation[1]),
                },
            },
        }
        self.decomposing = decomposing_modes[decomposing_mode]

    def _forward1(self, x, output_size):
        return super().forward(x, output_size=output_size)
    
    def _compose_transform(self, weight: torch.Tensor, key: str):
        return weight.reshape(*self.decomposing[key]).permute(*self.decomposing['inverse_permute'])
    
    def _decompose_transform(self, weight: torch.Tensor, key: str):
        return weight.permute(*self.decomposing['permute']).reshape(*self.decomposing[key])
    
    def _forward2(self, x, output_size):
        num_spatial_dims = 2
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims, self.dilation)
        x = conv_transpose2d(
            x, self.U, output_padding=output_padding, **self.decomposing['U'])
        x = conv_transpose2d(
            x, self.Vh, output_padding=output_padding, **self.decomposing['Vh'])
        return x
    
    def _forward3(self, x, output_size):
        num_spatial_dims = 2
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims, self.dilation)
        x = conv_transpose2d(
            x, self.U @ _diag_tensor_check(self.S), output_padding=output_padding, **self.decomposing['U'])
        x = conv_transpose2d(
            x, self.Vh, output_padding=output_padding, **self.decomposing['Vh'])
        return x
    
    def forward(self, input: torch.Tensor, output_size=None) -> torch.Tensor:
        x = self._current_forward(input, output_size=output_size)
        return x
       
    def _one_layer_compose(self):
        W = self.U @ _diag_tensor_check(self.S) @ self.Vh
        self.register_parameter('weight', Parameter(self._compose_transform(W, 'compose_shape')))

    def _two_layers_compose(self):
        SVh = _diag_tensor_check(self.S) @ self.Vh
        self.Vh = Parameter(
            self._compose_transform(SVh, 'Vh4d')
        )
        self.U = Parameter(
            self._compose_transform(self.U, 'U4d')
        )

    def _three_layers_compose(self):
        super().set_U_S_Vh(
            self._compose_transform(self.U, 'U4d'),
            self.S[..., None, None],
            self._compose_transform(self.U, 'Vh4d'),
        )

    def _anti_three_layers_compose(self):
        super().set_U_S_Vh(
            self._decompose_transform(self.U, 'U2d'),
            self.S[..., 0, 0],
            self._decompose_transform(self.Vh, 'Vh4d')
        )

class DecomposedConv1d(nn.Conv1d, IDecomposed):
    def __init__(
            self,
            base_module: nn.Conv1d,
            decomposing_mode = True,
            decomposer: Optional[str] = 'svd',
            compose_mode: str = None,
            device=None,
            dtype=None,
    ) -> None:
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
        IDecomposed.__init__(self, decomposing_mode, decomposer, compose_mode)

    def _forward1(self, x):
        return conv1d(
            x, self.U, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
    
    def _forward2(self, x):
        x = conv1d(
            x, self.Vh, None, self.stride, self.padding, self.dilation, self.groups
        )
        x = conv1d(
            x, self.U, self.bias
        )
        return x
    
    def _forward3(self, x):
        x = conv1d(
            x, (self.Vh), None, self.stride, self.padding, self.dilation, self.groups
        )
        x = conv1d(
            x, (self.U * self.S), self.bias
        )
        return x
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self._current_forward(input)
        return x
    
    def decompose(self) -> None:
        """Decomposes the weight matrix in singular value decomposition.
        Replaces the weights with U, S, Vh matrices such that weights = U * S * Vh.
        Args:
            decomposing_mode: ``'channel'`` or ``'spatial'`` weights reshaping method.
        Raises:
            ValueError: If ``decomposing_mode`` not in valid values.
        """
        W = self.weight.reshape(self.out_channels, -1)
        super().decompose(W)
        self._three_layers_compose()

    def _one_layer_compose(self):
        W = (self.U * self.S.unsqueeze(0)) @ self.Vh
        self.register_parameter('U', Parameter(W.reshape((self.out_channels, self.in_channels, self.kernel_size[0]))))
        self._eliminate_extra_params(['S', 'Vh'])

    def _two_layers_compose(self):
        SVh = self.S.unsqueeze(-1) * self.Vh
        # SVh = torch.diag(self.S) @ self.Vh
        self.register_parameter('Vh', Parameter(SVh.reshape((-1, self.in_channels, self.kernel_size[0]))))
        self.register_parameter('U', 
            Parameter(self.U.reshape((self.out_channels, -1, 1))))
        self._eliminate_extra_params(['S'])

    def _three_layers_compose(self):
        self.set_U_S_Vh(
            self.U.reshape((self.out_channels, -1, 1)),
            self.S[..., None],
            self.Vh.reshape((-1, self.in_channels, self.kernel_size[0]))
        )

    def _anti_three_layers_compose(self):
        self.set_U_S_Vh(
            self.U.reshape((self.out_channels, -1)),
            self.S[..., 0],
            self.Vh.reshape((-1, self.in_channels * self.kernel_size[0]))
        )

DecomposableLayers = {
    torch.nn.Linear: DecomposedLinear,
    torch.nn.Embedding: DecomposedEmbedding,
    torch.nn.Conv1d: DecomposedConv1d,
    torch.nn.Conv2d: DecomposedConv2d
}
