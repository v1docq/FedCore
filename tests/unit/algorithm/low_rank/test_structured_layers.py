"""Physical storage, vocabulary order, gradients and portable constructor tests."""
from copy import deepcopy
import json

import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.structured_profiles import FactorPair,basis_sharing_factors,groupreduce_factors
from fedcore.algorithm.low_rank.structured_layers import (
    FactorizedLinear,ResidualLinear,SharedBasisLinear,create_shared_basis_consumers,
    restore_shared_basis,GroupedEmbedding,GroupedLMHead,
)


def test_residual_keeps_base_identity_bias_once_and_independent_factors():
    base=nn.Linear(4,3).double(); before=deepcopy(base.state_dict())
    pair=FactorPair(torch.randn(3,2,dtype=torch.float64),torch.randn(2,4,dtype=torch.float64))
    layer=ResidualLinear(base,pair)
    x=torch.randn(7,4,dtype=torch.float64)
    expected=nn.functional.linear(x,base.weight+pair.matrix(),base.bias)
    torch.testing.assert_close(layer(x),expected)
    assert layer.base is base
    assert all(not p.requires_grad for p in base.parameters())
    for name,value in before.items():
        assert torch.equal(base.state_dict()[name],value)
    with torch.no_grad():
        pair.left.fill_(4.)
    torch.testing.assert_close(layer(x),expected)


def test_mixed_rank_zero_output_initialization_has_a_real_gradient():
    base=nn.Linear(3,2).double()
    layer=ResidualLinear.for_training(base,1,seed=42)
    layer.validate_trainable_initialization()
    x=torch.tensor([[1.,2.,3.],[2.,-1.,4.]],dtype=torch.float64)
    torch.testing.assert_close(layer(x),base(x))
    layer(x).square().sum().backward()
    assert bool(torch.count_nonzero(layer.residual.left.weight.grad))
    assert all(p.grad is None for p in base.parameters())
    with torch.no_grad():
        layer.residual.right.weight.zero_()
    with pytest.raises(ValueError,match="two zero"):
        layer.validate_trainable_initialization()


def test_factorized_linear_constructor_roundtrip_and_trace():
    pair=FactorPair(torch.randn(3,2),torch.randn(2,4)); bias=torch.randn(3)
    layer=FactorizedLinear.from_factors(pair,bias)
    config=layer.representation_config(); json.dumps(config,allow_nan=False)
    restored=FactorizedLinear(config["in_features"],config["out_features"],config["rank"],config["bias"])
    restored.load_state_dict(layer.state_dict())
    x=torch.randn(9,4)
    expected=nn.functional.linear(x,pair.matrix(),bias)
    torch.testing.assert_close(restored(x),expected)
    torch.testing.assert_close(torch.jit.trace(restored,x)(x),expected)


def test_shared_basis_is_one_parameter_before_and_after_constructor_reload():
    weights=(torch.randn(3,4,dtype=torch.float64),torch.randn(2,4,dtype=torch.float64))
    factors=basis_sharing_factors(weights,torch.eye(4,dtype=torch.float64),2)
    consumers=create_shared_basis_consumers(factors,(torch.randn(3,dtype=torch.float64),None))
    model=nn.ModuleList(consumers)
    assert model[0].right is model[1].right
    assert sum(p.numel() for p in model.parameters())==2*(4+3+2)+3
    restored=nn.ModuleList([SharedBasisLinear(4,3,2,True,dtype=torch.float64),SharedBasisLinear(4,2,2,False,dtype=torch.float64)])
    restored.load_state_dict(model.state_dict())
    restore_shared_basis(restored)
    assert restored[0].right is restored[1].right
    assert restored[0].left is not restored[1].left
    x=torch.randn(7,4,dtype=torch.float64)
    for old,new in zip(model,restored):
        torch.testing.assert_close(new(x),old(x))
    # Equal clones do not have the same unique parameter cost.
    restored[1].right=nn.Parameter(restored[1].right.detach().clone())
    assert sum(p.numel() for p in restored.parameters())==2*(4+3+2)+3+8
    with torch.no_grad():
        restored[1].right.add_(1.)
    with pytest.raises(ValueError,match="disagree"):
        restore_shared_basis(restored)


def grouped_table():
    w=torch.tensor([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.],[2.,1.,4.],[3.,7.,1.]],dtype=torch.float64)
    frequencies=torch.tensor([8.,2.,0.,3.,1.],dtype=w.dtype)
    state=groupreduce_factors(w,frequencies,torch.tensor([1,0,1,0,1]),(2,2))
    dense=torch.empty_like(w)
    for tokens,pair in zip(state.group_tokens,state.factors):
        dense[tokens]=pair.matrix()
    return GroupedEmbedding.from_factors(state),dense


def test_grouped_lookup_and_tied_head_preserve_original_vocabulary_order_and_storage():
    table,dense=grouped_table(); head=GroupedLMHead(table)
    assert head.table is table
    tokens=torch.tensor([[4,0,2],[1,3,0]])
    torch.testing.assert_close(table(tokens),dense[tokens])
    hidden=torch.randn(2,3,3,dtype=dense.dtype)
    torch.testing.assert_close(head(hidden),hidden@dense.T)
    model=nn.ModuleDict({"embedding":table,"head":head})
    assert sum(p.numel() for p in model.parameters())==table.parameter_elements
    config=table.representation_config(); json.dumps(config,allow_nan=False)
    restored_table=GroupedEmbedding(config["num_embeddings"],config["embedding_dim"],config["group_sizes"],config["ranks"],dtype=torch.float64)
    restored=nn.ModuleDict({"embedding":restored_table,"head":GroupedLMHead(restored_table)})
    restored.load_state_dict(model.state_dict())
    assert restored["embedding"] is restored["head"].table
    torch.testing.assert_close(restored["embedding"](tokens),dense[tokens])
    torch.testing.assert_close(restored["head"](hidden),hidden@dense.T)
    assert restored_table.map_bytes==5*16


def test_grouped_table_maps_refuse_invalid_reload_and_lookup_indices():
    table,_=grouped_table()
    state=deepcopy(table.state_dict()); state["token_to_local"].fill_(0)
    with pytest.raises(ValueError,match="bijectively"):
        deepcopy(table).load_state_dict(state)
    with pytest.raises(IndexError):
        table(torch.tensor([5]))
    with pytest.raises(IndexError):
        table(torch.tensor([-1]))
    with pytest.raises(ValueError,match="integer"):
        table(torch.tensor([1.]))


def test_grouped_lookup_trace_preserves_unseen_token_group_positions():
    table,dense=grouped_table()
    traced=torch.jit.trace(table,torch.tensor([0,1,2]))
    tokens=torch.tensor([4,3,4,0])
    torch.testing.assert_close(traced(tokens),dense[tokens])


def test_grouped_padding_value_retained_but_padding_lookup_has_no_gradient():
    table,dense=grouped_table(); table.padding_idx=0
    value=table(torch.tensor([0]))
    torch.testing.assert_close(value,dense[[0]])
    value.sum().backward()
    assert all(p.grad is None or not bool(torch.count_nonzero(p.grad)) for p in table.parameters())
