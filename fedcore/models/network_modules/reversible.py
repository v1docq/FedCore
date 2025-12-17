from typing import Any, Dict, List, Tuple, Optional

import torch
from operator import itemgetter
from torch import nn, Tensor
from torch.autograd import Function
from fedcore.architecture.computational.deterministic import Deterministic


def route_args(
        router: Dict[str, List[Tuple[bool, bool]]],
        args: Dict[str, Any],
        depth: int
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Routes input arguments to reversible layers.

    Args:
        router: Maps argument names to routing rules for each layer.
            Example: {"mask": [(True, False), (False, True)]}.
        args: Input arguments to distribute.
        depth: Number of reversible layers.

    Returns:
        List of (f_args, g_args) for each layer.
    """
    routed_args = [({}, {}) for _ in range(depth)]
    for key in args.keys() & router.keys():
        for layer_idx, (routes, (f_args, g_args)) in enumerate(zip(router[key], routed_args)):
            new_f = {key: args[key]} if routes[0] else {}
            new_g = {key: args[key]} if routes[1] else {}
            routed_args[layer_idx] = ({**f_args, **new_f}, {**g_args, **new_g})
    return routed_args


def layer_drop(
        layers: List[Any],
        prob: float
) -> List[Any]:
    """Randomly drops layers during training using LayerDrop.

    Args:
        layers (List[Any]): List of layers to apply dropout to.
        prob (float): Probability of dropping a layer (0 ≤ prob ≤ 1).

    Returns:
        List[Any]: Subset of layers with some layers dropped.

    Example:
        >>> layers = [nn.Linear(10, 10) for _ in range(5)]
        >>> active_layers = layer_drop(layers, prob=0.2)
    """
    if prob == 0 or not layers:
        return layers

    to_drop = torch.rand(len(layers)) < prob
    blocks = [layer for layer, drop in zip(layers, to_drop) if not drop]

    # Ensure at least one layer remains
    return blocks if len(blocks) > 0 else layers[:1]


class ReversibleBlock(nn.Module):
    """Reversible block to save memory during training.

    Implements forward and backward passes with inputs and gradients separated.

    Args:
        f (nn.Module): forward pass function.
        g (nn.Module): backward pass function.
    """

    def __init__(self, f: nn.Module, g: nn.Module):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    def forward(
            self,
            x: torch.Tensor,
            f_args: Dict[str, Any] = {},
            g_args: Dict[str, Any] = {}
    ) -> torch.Tensor:
        """Forward pass through block."""
        x1, x2 = torch.chunk(x, 2, dim=-1)
        with torch.no_grad():
            y1 = x1 + self.f(x2, **f_args)
            y2 = x2 + self.g(y1, **g_args)
        return torch.cat([y1, y2], dim=-1)

    def backward_pass(
            self,
            y: torch.Tensor,
            dy: torch.Tensor,
            f_args: Dict[str, Any] = {},
            g_args: Dict[str, Any] = {}
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward pass through block.

        Args:
            y (torch.Tensor): Tensor of forward pass.
            dy (torch.Tensor): Output gradient.
            f_args (Dict): `f` functions arguments.
            g_args (Dict): `g` functions arguments.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Gradient input and output.
        """
        # Input division
        y1, y2 = torch.chunk(y, 2, dim=-1)
        dy1, dy2 = torch.chunk(dy, 2, dim=-1)

        # g gradient
        with torch.enable_grad():
            y1.requires_grad_(True)
            gy1 = self.g(y1, **g_args)
            torch.autograd.backward(gy1, dy2)

        # x2 and dx1 update
        with torch.no_grad():
            x2 = y2 - gy1.detach()
            dx1 = dy1 + y1.grad
            y1.grad = None

        # f gradient
        with torch.enable_grad():
            x2.requires_grad_(True)
            fx2 = self.f(x2, **f_args)
            torch.autograd.backward(fx2, dx1, retain_graph=True)

        # x1 и dx2 update
        with torch.no_grad():
            x1 = y1 - fx2.detach()
            dx2 = dy2 + x2.grad
            x2.grad = None

        return torch.cat([x1, x2], dim=-1), torch.cat([dx1, dx2], dim=-1)


class _ReversibleFunction(Function):
    """Custom autograd Function for reversible blocks.

    Implements stateful forward and backward passes for a reversible architecture.
    Used in `ReversibleSequence`.
    """

    @staticmethod
    def forward(
            ctx: Any,
            x: torch.Tensor,
            blocks: List[nn.Module],
            args: List[dict]
    ) -> torch.Tensor:
        """Forward through a sequence of reversible blocks.

        Args:
            ctx: Context to store data in.
            x: Input tensor [batch_size, seq_len, dim].
            blocks: List of reversible blocks (ReversibleBlock).
            args: Arguments for each block.

        Returns:
            torch.Tensor: Output tensor after all blocks.
        """
        ctx.args = args
        for block, kwargs in zip(blocks, args):
            x = block(x, **kwargs)
        ctx.save_for_backward(x.detach(), blocks)
        return x

    @staticmethod
    def backward(
            ctx: Any,
            dy: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None]:
        """Backward pass.

        Args:
            ctx: Context with saved data.
            dy: Gradient of output tensor.

        Returns:
            Tuple: Gradients (dx, None, None).
        """
        x, blocks = ctx.saved_tensors
        args = ctx.args
        for block, kwargs in zip(reversed(blocks), reversed(args)):
            x, dy = block.backward_pass(x, dy, **kwargs)
        return dy, None, None


class ReversibleSequence(nn.Module):
    """Reversible block sequence to save memory.

    Implements a reversible architecture where the output of each block is restored during the backward pass without
    storing intermediate values.

    Args:
        blocks (List[Tuple[nn.Module, nn.Module]]): Pairs of functions (f, g).
        args_route (Optional[Dict]): Argument routing. Default: None.
        layer_dropout (float): Probability of layer dropout. Default: 0.
    """

    def __init__(
            self,
            blocks: List[Tuple[nn.Module, nn.Module]],
            args_route: Optional[Dict] = None,
            layer_dropout: float = 0.
    ):
        super().__init__()
        args_route = args_route or {}
        self.args_route = args_route
        self.layer_dropout = layer_dropout
        self.blocks = nn.ModuleList([
            ReversibleBlock(f=f, g=g) for f, g in blocks
        ])

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Direct pass using LayerDrop (if needed)."""
        x = torch.cat([x, x], dim=-1)  # Для обратимой логики

        args = route_args(self.args_route, kwargs, len(self.blocks))
        args = [{"f_args": a[0], "g_args": a[1]} for a in args]

        # Apply LayerDrop
        layers_and_args = list(zip(self.blocks, args))
        if self.training and self.layer_dropout > 0:
            layers_and_args = layer_drop(layers_and_args, self.layer_dropout)
            blocks, args = map(lambda i: list(map(itemgetter(i), layers_and_args)), (0, 1))

        out = _ReversibleFunction.apply(x, self.blocks, args)
        return torch.stack(out.chunk(2, dim=-1)).sum(dim=0)


