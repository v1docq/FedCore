"""
Functions for recreating decomposed layers.

Contains utilities to recreate standard layers from their decomposed counterparts.
"""

import torch
import torch.nn as nn
from fedcore.models.network_impl.decomposed_layers import (
    DecomposedLinear, DecomposedEmbedding, DecomposedConv2d, DecomposedConv1d
)
from .core_reassemblers import RecreatedDecomposed


def _recreate_embedding(E: nn.Embedding):
    """Recreate embedding layer (placeholder for future implementation)."""
    return E  # For now, return as-is


def _recreate_decomposed_linear(L: DecomposedLinear):
    """Recreate linear layer from decomposed version."""
    U, Vh = L.U.detach(), L.Vh.detach()
    h = U.size(-1)
    new = RecreatedDecomposed(
        nn.Linear(L.in_features, h, bias=False),
        nn.Linear(h, L.out_features, bias=True),
        routing={
            'out_features': ('out_features', '1'),
            'weight': ('weight', '1'),
            'bias': ('bias', '1')
        }
    )
    new[0].weight.data = Vh
    new[-1].weight.data = U
    if getattr(L.bias, 'data', None) is not None:
        new[-1].bias.data = L.bias.data
    new._is_recreated = True
    return new


def _recreate_decomposed_embedding(E: DecomposedEmbedding):
    """Recreate embedding layer from decomposed version."""
    U, Vh = E.U.detach(), E.Vh.detach()
    h = U.size(-1)
    new = RecreatedDecomposed(
        nn.Embedding(E.num_embeddings, h),
        nn.Linear(h, E.embedding_dim, False),
        routing={
            'embedding_dim': ('out_features', '1'),
            'weight': ('weight', '1'),
            'bias': ('bias', '1'),
        }
    )
    new[0].weight.data = U
    new[-1].weight.data = Vh.T
    new._is_recreated = True
    return new


def _recreate_decomposed_conv2d(C: DecomposedConv2d):
    """Recreate 2D convolution layer from decomposed version."""
    U, Vh = C.U.detach(), C.Vh.detach()
    assert U.ndim == 4, 'Non composed layers are not supported'
    out_1, in_1, k_11, k_12 = Vh.size()
    out_2, in_2, k_21, k_22 = U.size()
    new = RecreatedDecomposed(
        nn.Conv2d(in_1, out_1, (k_11, k_12), **C.decomposing['Vh'], bias=False),
        nn.Conv2d(in_2, out_2, (k_21, k_22), **C.decomposing['U'], bias=True),
        routing={
            'in_channels': ('in_channels', '0'),
            'out_channels': ('out_channels', '1'),
            'groups': ('groups', '0'),
            'weight': ('weight', '1'),
            'bias': ('bias', '1'),
        }
    )
    new[0].weight.data = Vh
    new[-1].weight.data = U
    new[-1].bias = C.bias
    new._is_recreated = True
    return new


def _recreate_decomposed_conv1d(C: DecomposedConv1d):
    """Recreate 1D convolution layer from decomposed version."""
    U, Vh = C.U.detach(), C.Vh.detach()
    assert U.ndim == 3, 'Non composed layers are not supported'
    out, r, k_2 = U.size()
    r, in_, k_1 = Vh.size()
    C1 = nn.Conv1d(in_, r, k_1, C.stride, C.padding, C.dilation, C.groups, bias=False)
    C2 = nn.Conv1d(r, out, k_2, bias=True)
    C1.weight.data = Vh
    C2.weight.data = U
    C2.bias = C.bias
    new = RecreatedDecomposed(
        C1,
        C2,
        routing={
            'out_channels': ('out_channels', '1'),
            'weight': ('weight', '1'),
            'bias': ('bias', '1'),
        }
    )
    new._is_recreated = True
    return new


# Registry of recreation functions
RECREATION_FUNCTIONS = {
    nn.Embedding: _recreate_embedding,
    DecomposedLinear: _recreate_decomposed_linear,
    DecomposedEmbedding: _recreate_decomposed_embedding,
    DecomposedConv2d: _recreate_decomposed_conv2d,
    DecomposedConv1d: _recreate_decomposed_conv1d,
}


def get_recreation_function(layer_type):
    """Get recreation function for given layer type."""
    return RECREATION_FUNCTIONS.get(layer_type)
