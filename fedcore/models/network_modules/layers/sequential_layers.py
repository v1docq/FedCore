from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from fedcore.models.network_modules.reversible import route_args, layer_drop


class SequentialSequence(nn.Module):
    """Applies a sequence of layers with argument routing and LayerDrop.

    Args:
        layers (List[Tuple[nn.Module, nn.Module]]): List of (f, g) layer pairs.
        args_route (Optional[Dict[str, List[Tuple[bool, bool]]]]):
            Maps argument names to routing rules for each layer. Default: `None`.
        layer_dropout (float): Probability of dropping a layer during training. Default: 0.
    """

    def __init__(
            self,
            layers: List[Tuple[nn.Module, nn.Module]],
            args_route: Optional[Dict[str, List[Tuple[bool, bool]]]] = None,
            layer_dropout: float = 0.
    ):
        super().__init__()
        args_route = args_route or {}
        assert all(
            len(route) == len(layers) for route in args_route.values()
        ), "Each route must match the number of layers."
        self.layers = nn.ModuleList(layers)
        self.args_route = args_route
        self.layer_dropout = layer_dropout

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Processes input through the layer sequence.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, ..., features].
            **kwargs: Additional arguments to route to layers.
        """
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))

        if self.training and self.layer_dropout > 0:
            layers_and_args = layer_drop(layers_and_args, self.layer_dropout)

        for (f, g), (f_args, g_args) in layers_and_args:
            x = x + f(x, **f_args)
            x = x + g(x, **g_args)
        return x
