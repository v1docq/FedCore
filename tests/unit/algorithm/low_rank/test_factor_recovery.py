"""Real tiny left/right updates using the existing LoRA implementation."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear
from fedcore.algorithm.low_rank.structured_profiles import FactorPair
from fedcore.algorithm.low_rank.factor_recovery import prepare_factor_lora_stage,finish_factor_lora_stage
from fedcore.algorithm.low_rank.topology import TopologyError


def test_real_v5_left_then_right_updates_freeze_merge_and_fresh_optimizer():
    torch.manual_seed(41)
    layer=FactorizedLinear.from_factors(FactorPair(torch.randn(3,2,dtype=torch.float64),torch.randn(2,4,dtype=torch.float64)),torch.randn(3,dtype=torch.float64))
    model=nn.Sequential(layer,nn.Tanh(),nn.Linear(3,2).double())
    original=deepcopy(model.state_dict())
    x=torch.randn(13,4,dtype=torch.float64); target=torch.randn(13,2,dtype=torch.float64)
    current=model
    previous_optimizer=None
    for side in ("left","right"):
        stage=prepare_factor_lora_stage(current,("0",),factor=side,rank=1,alpha=1.)
        stage_before=deepcopy(stage.model.state_dict())
        optimizer=torch.optim.SGD(stage.trainable_parameters(),lr=.02)
        stage.validate_optimizer(optimizer)
        if previous_optimizer is not None:
            with pytest.raises(ValueError,match="stale"):
                stage.validate_optimizer(previous_optimizer)
        stage.model.train()
        before=stage.model(x).detach().clone()
        loss_before=float((before-target).square().mean())
        for _ in range(12):
            optimizer.zero_grad()
            loss=(stage.model(x)-target).square().mean()
            loss.backward(); optimizer.step()
        after=stage.model(x).detach()
        assert float((after-target).square().mean())<loss_before
        assert not torch.equal(before,after)
        trainable=set(stage.trainable_names)
        for name,param in stage.model.named_parameters():
            if name not in trainable:
                assert param.grad is None
                assert torch.equal(param,stage_before[name])
        adapter=getattr(stage.model[0],side)
        adapter.eval(); unmerged=stage.model.eval()(x).detach().clone()
        adapter.merge(safe_merge=True)
        torch.testing.assert_close(stage.model(x),unmerged,atol=1e-12,rtol=1e-12)
        adapter.unmerge()
        torch.testing.assert_close(stage.model(x),unmerged,atol=1e-12,rtol=1e-12)
        current,evidence=finish_factor_lora_stage(stage)
        torch.testing.assert_close(current(x),unmerged,atol=1e-12,rtol=1e-12)
        assert evidence["delta_norms"][f"0.{side}"]>0
        assert evidence["optimizer_reusable"] is False
        assert type(getattr(current[0],side)) is nn.Linear
        previous_optimizer=optimizer
    for name,tensor in original.items():
        assert torch.equal(model.state_dict()[name],tensor)
    # The final constructor reload reproduces the trained compact function.
    restored=deepcopy(model); restored.load_state_dict(current.state_dict())
    torch.testing.assert_close(restored(x),current(x))
    assert sum(p.numel() for p in current.parameters())==sum(p.numel() for p in model.parameters())


def test_stage_preserves_repeated_module_aliases_and_refuses_tied_parameters():
    shared=FactorizedLinear(3,3,2,dtype=torch.float64)
    model=nn.ModuleDict({"one":shared,"two":shared})
    stage=prepare_factor_lora_stage(model,("one","two"),factor="left",rank=1)
    assert stage.model["one"] is stage.model["two"]
    assert stage.model["one"].left is stage.model["two"].left
    merged,_=finish_factor_lora_stage(stage)
    assert merged["one"] is merged["two"]
    other=FactorizedLinear(3,3,2,dtype=torch.float64)
    other.left.weight=shared.left.weight
    tied=nn.ModuleDict({"one":shared,"two":other})
    with pytest.raises(TopologyError,match="tied"):
        prepare_factor_lora_stage(tied,("one","two"),factor="left",rank=1)


def test_stage_rejects_accidental_unfreeze_and_retains_explicit_adapter_form():
    layer=FactorizedLinear(4,3,2)
    stage=prepare_factor_lora_stage(layer,("",),factor="left",rank=1)
    with torch.no_grad():
        stage.model.left.lora_B[stage.adapter_name].weight.fill_(.2)
    x=torch.randn(7,4)
    retained,evidence=finish_factor_lora_stage(stage,merge=False)
    torch.testing.assert_close(retained(x),stage.model.eval()(x))
    assert evidence["merge_policy"]=="separate_adapters"
    stage.model.right.weight.requires_grad_(True)
    with pytest.raises(ValueError,match="outside"):
        stage.trainable_parameters()
