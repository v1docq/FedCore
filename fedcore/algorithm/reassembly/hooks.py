"""
Hooks for model reassembly node.

Contains hooks for various stages of the reassembly process.
"""
import torch

from enum import Enum
from typing import Dict, Any
from fedcore.models.network_impl.hooks import BaseHook


class ReassemblyValidationHook(BaseHook):
    """Hook for model validation before reassembly."""
    
    _hook_place = -1  # Executed before main processing
    
    def __init__(self, hook_params: Dict[str, Any], model):
        super().__init__(hook_params, model)
        self.validation_enabled = hook_params.get('enable_validation', True)
    
    def __call__(self, **kwargs):
        """Model validation before reassembly."""
        if not self.validation_enabled:
            return
        
        model = kwargs.get('model', self.model)
        print("[ReassemblyValidationHook] Performing model validation before reassembly")
        
        # Check that model is not empty
        if model is None:
            raise ValueError("Model cannot be None")
        
        # Check for parameters
        param_count = sum(p.numel() for p in model.parameters())
        if param_count == 0:
            raise ValueError("Model contains no parameters")
        
        print(f"[ReassemblyValidationHook] Validation passed successfully. Parameters: {param_count}")


class ReassemblyCompletionHook(BaseHook):
    """Hook for actions after reassembly completion."""
    
    _hook_place = 1  # Executed after main processing
    
    def __init__(self, hook_params: Dict[str, Any], model):
        super().__init__(hook_params, model)
        self.save_enabled = hook_params.get('save_after_reassembly', False)
        self.save_path = hook_params.get('save_path', None)
    
    def __call__(self, **kwargs):
        """Actions after reassembly completion."""
        model = kwargs.get('model', self.model)
        print("[ReassemblyCompletionHook] Performing post-processing after reassembly")
        
        # Check structural changes
        if hasattr(model, '_structure_changed__') and model._structure_changed__:
            print("[ReassemblyCompletionHook] Detected structural changes in model")
        
        # Save model if required
        if self.save_enabled and self.save_path:
            try:
                torch.save(model.state_dict(), self.save_path)
                print(f"[ReassemblyCompletionHook] Model saved to {self.save_path}")
            except Exception as e:
                print(f"[ReassemblyCompletionHook] Save error: {e}")


class ReassemblyHooks(Enum):
    """Enumeration of hooks for reassembly."""
    
    validation_hook = ReassemblyValidationHook
    completion_hook = ReassemblyCompletionHook
