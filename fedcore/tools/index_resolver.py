import torch
import torch.nn as nn
from typing import List, Dict, Optional, Callable, Dict
from copy import deepcopy

class IndexResolvingParameter(nn.Parameter):
    """
    A custom parameter for controlling group indexing during structural pruning.
    Supports union/intersect and dynamic index rebuilding.
    """
    def __new__(cls, data, *args, **kwargs):
        return super().__new__(cls, data)

    def __init__(
        self,
        data: torch.Tensor,
        group_ids: Optional[List[int]] = None,
        aggregation_mode: str = "union",
        module_name: Optional[str] = None,
    ):
        self.num_original_groups = data.shape[0]
        
        if group_ids is None:
            self.group_ids = list(range(self.num_original_groups))
        else:
            self.group_ids = group_ids
            
        self.original_to_current: Dict[int, Optional[int]] = {idx: None for idx in self.group_ids}
        self.aggregation_mode = aggregation_mode
        
        self.module_name = module_name
        self._update_mapping([])

    def __repr__(self):
        active = sum(1 for v in self.original_to_current.values() if v is not None)
        return (f"IndexResolvingParameter(module={self.module_name}, shape={tuple(self.shape)}, "
                f"active={active}/{self.num_original_groups}, mode={self.aggregation_mode})")
    
    def _update_mapping(self, active_indices: List[int]) -> None:
        for k in self.original_to_current:
            self.original_to_current[k] = None
            
        for new_idx, original_idx in enumerate(active_indices):
            self.original_to_current[original_idx] = new_idx
        
        self.current_to_original = {v: k for k, v in self.original_to_current.items() if v is not None}
    
    def resolve_indices(self, important_groups: List[int]) -> None:
        current_active_original = set(self.current_to_original.values())
        new_important_set = set(important_groups)
        
        if self.aggregation_mode == "union":
            final_active_original = current_active_original.union(new_important_set)
        elif self.aggregation_mode == "intersect":
            if not current_active_original:
                final_active_original = new_important_set
            else:
                final_active_original = current_active_original.intersection(new_important_set)
        else:
            raise ValueError(f"Unsupported aggregation mode: {self.aggregation_mode}")
        final_active_list = sorted(list(final_active_original))
        self._update_mapping(final_active_list)

    def __getattr__(self, name):
        pass
def wrap_parameters_with_resolver(
    module: nn.Module,
    param_filter: Optional[Callable[[nn.Module, str, nn.Parameter], bool]] = None,
    aggregation_mode: str = "union",
    inplace: bool = True,
) -> nn.Module:
    """
    Universal replacement of module parameters with IndexResolvingParameter.

    Args:
        module: PyTorch root module
        param_filter: predicate function that accepts (module, param_name, parameter) and returns True if this parameter should be wrapped. If None, all parameters are wrapped
        with a first dimension > 0 (assuming channels/neurons).
        aggregation_mode: aggregation mode for IndexResolvingParameter
        inplace: modify the module in place or create a copy (if False – deepcopy)

    Returns:
        modified module (same if inplace=True)
    """
    if not inplace:
        module = deepcopy(module)
    
    for submodule in module.modules():
        for param_name, param in list(submodule.named_parameters(recurse=False)):
            if isinstance(param, IndexResolvingParameter):
                continue
            
            if param_filter is not None:
                if not param_filter(submodule, param_name, param):
                    continue
            else:
                if param.dim() == 0 or param.shape[0] == 0:
                    continue
            
            num_groups = param.shape[0]
            
            module_display_name = f"{submodule._get_name()}.{param_name}"
            
            wrapped_param = IndexResolvingParameter(
                data=param.data,
                group_ids=list(range(num_groups)),
                aggregation_mode=aggregation_mode,
                module_name=module_display_name   # <-- передаём имя
            )
            
            if param.grad is not None:
                wrapped_param.grad = param.grad.clone()
            
            setattr(submodule, param_name, wrapped_param)
    
    return module
