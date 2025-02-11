import pytest

import torch 
import torch.nn as nn 
from fedcore.models.network_impl.layers import DecomposedLinear
from fedcore.algorithm.low_rank.rank_pruning import rank_threshold_pruning
from fedcore.algorithm.low_rank.svd_tools import decompose_module

layer_input = [
    (nn.Linear(100, 90),  torch.randn(12, 100), True),
    (nn.Conv2d(3, 16, 5, 5, 1, 2), torch.randn(2, 3, 32, 32), 'channel'),
    (nn.Conv2d(3, 16, 5, 5, 1, 2), torch.randn(2, 3, 32, 32), 'spatial'),
    (nn.Embedding(100, 64,), torch.randint(0, 99, (20, 23)), True)
]

def test_decomposed_layer():
    for L, T, mode in layer_input:
        wrapping = nn.Sequential(L)
        decompose_module(wrapping, decomposing_mode=mode, compose_mode='two_layers')
        DL = wrapping[0]
        assert torch.allclose(DL(T), L(T), atol=1e-5), 'Wrong decomposition!'
        rank_threshold_pruning(DL, 0.5, 'quantile')
        #check composition
        DL.compose_weight_for_inference()
        #check forward dimensionalities
        DL(T)
