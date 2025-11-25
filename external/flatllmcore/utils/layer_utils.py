"""
Layer utility functions for FLAT-LLM operations.

This module provides helper functions for finding, analyzing, and manipulating
neural network layers during FLAT-LLM transformations.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Type, Union


def find_layers(module: nn.Module, layers: List[Type[nn.Module]] = None, name: str = '') -> Dict[str, nn.Module]:
    """
    Recursively find layers of specified types within a module.
    
    Args:
        module: PyTorch module to search
        layers: List of layer types to find. Defaults to [nn.Linear]
        name: Current module name (used for recursion)
        
    Returns:
        Dictionary mapping layer names to layer modules
    """
    if layers is None:
        layers = [nn.Linear]
        
    if type(module) in layers:
        return {name: module}
    
    result = {}
    for child_name, child in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        result.update(find_layers(child, layers=layers, name=full_name))
    
    return result


def check_sparsity(model: nn.Module) -> float:
    """
    Calculate the overall sparsity ratio of a model.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Sparsity ratio (0.0 = dense, 1.0 = completely sparse)
    """
    use_cache = getattr(model.config, 'use_cache', False)
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False

    layers = model.model.layers if hasattr(model, 'model') else [model]
    total_zeros = 0
    total_params = 0
    
    for i, layer in enumerate(layers):
        layer_modules = find_layers(layer)
        
        layer_zeros = 0
        layer_params = 0
        
        for name, module in layer_modules.items():
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight.data
                layer_zeros += (weight == 0).sum().item()
                layer_params += weight.numel()
        
        total_zeros += layer_zeros
        total_params += layer_params
        
        if layer_params > 0:
            layer_sparsity = layer_zeros / layer_params
            print(f"Layer {i} sparsity: {layer_sparsity:.6f}")
    
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = use_cache
    
    return total_zeros / total_params if total_params > 0 else 0.0


def check_structural_sparsity(model: nn.Module) -> float:
    """
    Calculate structural sparsity considering only linear layers.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Structural sparsity ratio
    """
    use_cache = getattr(model.config, 'use_cache', False)
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False

    layers = model.model.layers if hasattr(model, 'model') else [model]
    total_active = 0
    total_params = 0
    
    for i, layer in enumerate(layers):
        layer_modules = find_layers(layer, layers=[nn.Linear])
        
        layer_active = 0
        layer_params = 0
        
        for name, module in layer_modules.items():
            if isinstance(module, nn.Linear):
                out_features = module.out_features
                in_features = module.in_features
                
                # Count non-zero parameters
                active_params = (module.weight.data != 0).sum().item()
                layer_active += active_params
                layer_params += in_features * out_features
                
                sparsity = active_params / (in_features * out_features)
                print(f"  {name}: sparsity {sparsity:.6f}, active {active_params}, total {in_features * out_features}")
        
        total_active += layer_active
        total_params += layer_params
        
        if layer_params > 0:
            layer_sparsity = layer_active / layer_params
            print(f"Layer {i} structural sparsity: {layer_sparsity:.6f}")
    
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = use_cache
    
    return total_active / total_params if total_params > 0 else 0.0


def get_layer_dimensions(model: nn.Module, model_type: str = "llama") -> Dict[str, int]:
    """
    Extract key dimensions from model for FLAT-LLM calculations.
    
    Args:
        model: PyTorch model
        model_type: Type of model ("llama", "mistral", etc.)
        
    Returns:
        Dictionary containing dimension information
    """
    config = model.config
    
    if model_type.lower() in ["llama", "llama-2", "llama-3"]:
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
        intermediate_size = config.intermediate_size
        
        return {
            'hidden_size': hidden_size,
            'num_heads': num_heads,
            'num_kv_heads': num_kv_heads,
            'head_dim': hidden_size // num_heads,
            'intermediate_size': intermediate_size,
            'dq': hidden_size * hidden_size,
            'dk': hidden_size * (hidden_size if num_kv_heads == num_heads else hidden_size * num_kv_heads // num_heads),
            'dv': hidden_size * (hidden_size if num_kv_heads == num_heads else hidden_size * num_kv_heads // num_heads),
            'do': hidden_size * hidden_size,
            'dmlp': hidden_size * intermediate_size
        }
    
    elif model_type.lower() == "mistral":
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads  
        num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
        intermediate_size = config.intermediate_size
        
        return {
            'hidden_size': hidden_size,
            'num_heads': num_heads,
            'num_kv_heads': num_kv_heads,
            'head_dim': hidden_size // num_heads,
            'intermediate_size': intermediate_size,
            'dq': hidden_size * hidden_size,
            'dk': hidden_size * (hidden_size * num_kv_heads // num_heads),
            'dv': hidden_size * (hidden_size * num_kv_heads // num_heads),
            'do': hidden_size * hidden_size,
            'dmlp': hidden_size * intermediate_size
        }
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def replace_attention_layers(model: nn.Module, custom_attention_class, custom_decoder_class = None):
    """
    Replace standard attention layers with FLAT-LLM optimized versions.
    
    Args:
        model: Model to modify
        custom_attention_class: Custom attention class to use
        custom_decoder_class: Optional custom decoder class
    """
    if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
        raise ValueError("Model structure not supported for attention replacement")
    
    layers = model.model.layers
    
    for i, layer in enumerate(layers):
        # Move to CPU to save memory during replacement
        layer = layer.cpu()
        torch.cuda.empty_cache()
        
        # Create new decoder layer if custom class provided
        if custom_decoder_class is not None:
            new_layer = custom_decoder_class(model.config, layer_idx=i)
            
            # Copy weights from original layer
            new_layer.load_state_dict(layer.state_dict(), strict=True)
            
            # Move to appropriate device
            device = next(layer.parameters()).device if list(layer.parameters()) else 'cuda'
            dtype = next(layer.parameters()).dtype if list(layer.parameters()) else torch.float16
            new_layer = new_layer.to(device=device, dtype=dtype)
            
            # Replace in model
            model.model.layers[i] = new_layer
            
        else:
            # Only replace attention module
            if hasattr(layer, 'self_attn'):
                old_attn = layer.self_attn
                new_attn = custom_attention_class(model.config, layer_idx=i)
                
                # Copy attention weights
                new_attn.load_state_dict(old_attn.state_dict(), strict=True)
                
                # Replace attention
                layer.self_attn = new_attn.to(device=old_attn.q_proj.weight.device, 
                                              dtype=old_attn.q_proj.weight.dtype)
        
        print(f"Replaced layer {i + 1}/{len(layers)}")
        
        # Cleanup
        torch.cuda.empty_cache()
    
    return model
