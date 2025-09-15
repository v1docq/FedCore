from copy import deepcopy
import pytest 
import torch
import torch.nn as nn 

from fedcore.algorithm.low_rank.svd_tools import decompose_module
from fedcore.algorithm.reassembly.decomposed_recreation import (
    _recreate_decomposed_conv1d, 
    _recreate_decomposed_embedding,
    _recreate_embedding,
    _recreate_decomposed_linear,
    _recreate_decomposed_conv2d
)

ATOL = 1e-4

def test_conv1d():
    in_, out_, k = 18, 29, 3
    C1 = nn.Conv1d(in_, out_, k)
    DC1 = nn.Sequential(deepcopy(C1))
    decompose_module(DC1, compose_mode='two_layers')
    DC1[0].compose_weight_for_inference()
    RDC1 = _recreate_decomposed_conv1d(DC1[0])
    X = torch.randn(1, in_, 180)
    assert torch.allclose(DC1(X), C1(X), atol=ATOL), 'decomposition wrong'
    assert torch.allclose(RDC1(X), C1(X), atol=ATOL), 'recreation wrong'

def test_conv2d():
    in_, out_, k1, k2 = 18, 29, 3, 5
    C1 = nn.Conv2d(in_, out_, k1, k2)
    DC1 = nn.Sequential(deepcopy(C1))
    decompose_module(DC1, compose_mode='two_layers', decomposing_mode='channel')
    DC1[0].compose_weight_for_inference()
    RDC1 = _recreate_decomposed_conv2d(DC1[0])
    X = torch.randn(1, in_, 100, 100)
    assert torch.allclose(DC1(X), C1(X), atol=ATOL), 'decomposition wrong'
    assert torch.allclose(RDC1(X), C1(X), atol=ATOL), 'recreation wrong'
    
def test_embedding():
    in_, out_, k = 18, 29, 3
    C1 = nn.Embedding(in_, out_,)
    DC1 = nn.Sequential(deepcopy(C1))
    decompose_module(DC1, compose_mode='two_layers')
    DC1[0].compose_weight_for_inference()
    RDC1 = _recreate_decomposed_embedding(DC1[0])
    X = torch.randint(0, in_ - 1, (1, k))
    assert torch.allclose(DC1(X), C1(X), atol=ATOL), 'decomposition wrong'
    assert torch.allclose(RDC1(X), C1(X), atol=ATOL), 'recreation wrong'

def test_linear():
    in_, out_, k = 18, 29, 3
    C1 = nn.Linear(in_, out_,)
    DC1 = nn.Sequential(deepcopy(C1))
    decompose_module(DC1, compose_mode='two_layers')
    DC1[0].compose_weight_for_inference()
    RDC1 = _recreate_decomposed_linear(DC1[0])
    X = torch.randn(k, in_)
    assert torch.allclose(DC1(X), C1(X), atol=ATOL), 'decomposition wrong'
    assert torch.allclose(RDC1(X), C1(X), atol=ATOL), 'recreation wrong'

    