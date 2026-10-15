"""Configure SVD-LLM v5 stages using the existing independent LoRA adapters.

The caller owns data, scalar loss, training runner, update count, optimizer
creation/destruction, checkpoint references and resource accounting. This module
only makes the parameter boundary concrete and finalizes an inference merge.
Source: https://arxiv.org/html/2403.07378v5, Section 3.2, equation 7.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import nn

from fedcore.models.network_modules.layers.lora import LoRALayer, apply_lora
from .structured_layers import FactorizedLinear
from .topology import module_paths, replace_modules_atomically


@dataclass(frozen=True)
class FactorLoRAStage:
    model: nn.Module
    factor: str
    factor_paths: tuple[str, ...]
    adapter_paths: tuple[str, ...]
    trainable_names: tuple[str, ...]
    adapter_name: str
    adapter_rank: int
    alpha: float

    def trainable_parameters(self):
        """Return actual live parameters, after structural replacement."""
        allowed=set(self.trainable_names)
        parameters=dict(self.model.named_parameters())
        if any(name not in parameters or not parameters[name].requires_grad for name in allowed):
            raise ValueError("stage trainable Parameters changed; rebuild the stage/optimizer")
        actual={name for name,p in parameters.items() if p.requires_grad}
        if actual!=allowed:
            raise ValueError("parameters outside the factor LoRA stage are trainable")
        return tuple(parameters[name] for name in self.trainable_names)

    def validate_optimizer(self,optimizer):
        """A retained optimizer from a previous stage is explicitly invalid."""
        if not isinstance(optimizer,torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch optimizer")
        expected={id(p) for p in self.trainable_parameters()}
        actual=[id(p) for group in optimizer.param_groups for p in group["params"]]
        if len(actual)!=len(set(actual)) or set(actual)!=expected:
            raise ValueError("optimizer contains stale, duplicate or unauthorized Parameters")

    def evidence(self):
        return {"method":"svdllm_v5_sequential_factor_lora","source_version":"2403.07378v5",
                "factor":self.factor,"factor_paths":list(self.factor_paths),
                "adapter_paths":list(self.adapter_paths),"trainable_names":list(self.trainable_names),
                "adapter_name":self.adapter_name,"adapter_rank":self.adapter_rank,"alpha":self.alpha,
                "trainable_elements":sum(p.numel() for p in self.trainable_parameters()),
                "optimizer_policy":"fresh_optimizer_on_live_stage_parameters"}


def prepare_factor_lora_stage(model,factor_paths,*,factor,rank,alpha=1.,
                              adapter_name="svdllm_v5",dropout=0.):
    """Return an independent model with only the chosen factor's adapters live.

    ``factor_paths`` identify FactorizedLinear parents, including ``''`` for the
    root. Module aliases are preserved by existing apply_lora topology handling;
    distinct modules with tied Parameters/shared storage are explicitly refused.
    For the published profile call left first, finalize it, then call right.
    """
    if factor not in ("left","right"):
        raise ValueError("factor must be left or right")
    if type(rank) is not int or rank<1 or type(alpha) not in (int,float) or not math.isfinite(alpha):
        raise ValueError("positive adapter rank and finite alpha required")
    paths=module_paths(model)
    factor_paths=tuple(factor_paths)
    if not factor_paths or len(set(factor_paths))!=len(factor_paths):
        raise ValueError("distinct explicit factor parent paths required")
    adapter_paths=[]
    for path in factor_paths:
        layer=paths.get(path)
        if not isinstance(layer,FactorizedLinear):
            raise ValueError("factor LoRA requires explicit FactorizedLinear parents")
        target=f"{path}.{factor}" if path else factor
        target_layer=paths.get(target)
        if type(target_layer) is not nn.Linear:
            raise ValueError("finalize the selected factor's previous adapter before a new stage")
        if rank>min(target_layer.weight.shape):
            raise ValueError("adapter rank must fit the selected factor axes")
        adapter_paths.append(target)
    candidate=apply_lora(model,rank=rank,lora_alpha=alpha,target_layers=adapter_paths,
                         adapter_name=adapter_name,lora_dropout=dropout)
    candidate.requires_grad_(False)
    copied=module_paths(candidate)
    seen=set()
    for path in adapter_paths:
        adapter=copied[path]
        if id(adapter) in seen:
            continue
        seen.add(id(adapter))
        adapter.lora_A[adapter_name].requires_grad_(True)
        adapter.lora_B[adapter_name].requires_grad_(True)
    names=tuple(name for name,p in candidate.named_parameters() if p.requires_grad)
    stage=FactorLoRAStage(candidate,factor,factor_paths,tuple(adapter_paths),names,
                          adapter_name,rank,float(alpha))
    stage.trainable_parameters()
    return stage


def finish_factor_lora_stage(stage: FactorLoRAStage,*,merge=True):
    """Return an independent inference model and merge evidence.

    ``merge=False`` keeps adapter/base tensors explicit and frozen. A subsequent
    stage can train the other factor, but exported base references remain the
    caller's responsibility. Merged factors have unchanged dimensions and cost;
    the caller must still count the whole final model's unique storage.
    """
    if not isinstance(stage,FactorLoRAStage):
        raise TypeError("FactorLoRAStage required")
    evidence=stage.evidence()
    result=deepcopy(stage.model).eval()
    paths=module_paths(result)
    identities={}; replacements={}; delta_norms={}
    for path in stage.adapter_paths:
        adapter=paths[path]
        if not isinstance(adapter,LoRALayer):
            raise ValueError("stage adapter topology changed before finalization")
        delta=adapter.get_delta_weight(stage.adapter_name)
        if not bool(torch.isfinite(delta).all()):
            raise ValueError("nonfinite adapter delta cannot finalize")
        delta_norms[path]=float(torch.linalg.vector_norm(delta.detach().double()))
        if merge:
            if id(adapter) not in identities:
                adapter.merge(safe_merge=True)
                identities[id(adapter)]=adapter.get_base_layer()
            replacements[path]=identities[id(adapter)]
    if merge:
        result=replace_modules_atomically(result,replacements)
    result.requires_grad_(False)
    evidence.update({"merge_policy":"merged_into_factors" if merge else "separate_adapters",
                     "delta_norms":delta_norms,"parameter_elements_after":sum(p.numel() for p in result.parameters()),
                     "base_independence":"deepcopy_before_stage_and_finalization",
                     "optimizer_reusable":False})
    return result,evidence
