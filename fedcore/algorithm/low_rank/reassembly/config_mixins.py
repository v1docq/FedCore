"""
Mixin classes for configuration analysis.

Contains shared functionality for analyzing models and hardware,
used by various reassembler configuration classes.
"""

from typing import Optional
import torch
import torch.nn as nn


class ModelAnalysisMixin:
    """Mixin providing model analysis functionality for configuration classes."""
    
    @staticmethod
    def _analyze_model(model: nn.Module) -> dict:
        """
        Analyze model architecture and extract relevant characteristics.
        
        This method extracts common model characteristics that are useful for
        auto-configuration of various compression techniques.
        
        Args:
            model: The transformer model to analyze
            
        Returns:
            dict: Model characteristics including:
                - model_type: Model architecture type (e.g., 'llama', 'qwen')
                - hidden_size: Hidden dimension size
                - num_attention_heads: Number of attention heads
                - num_hidden_layers: Number of transformer layers
                - vocab_size: Vocabulary size
                - num_key_value_heads: Number of KV heads (for GQA models)
                - head_dim: Dimension per attention head
                - total_params: Total number of parameters
                - uses_gqa: Whether model uses Grouped Query Attention
                - size_category: Model size category ('small', 'medium', 'large')
                
        Raises:
            ValueError: If model doesn't have a config attribute
        """
        config = getattr(model, 'config', None)
        if config is None:
            raise ValueError("Model must have a config attribute")
        
        # Extract basic model information
        model_info = {
            'model_type': getattr(config, 'model_type', 'unknown'),
            'hidden_size': getattr(config, 'hidden_size', 768),
            'num_attention_heads': getattr(config, 'num_attention_heads', 12),
            'num_hidden_layers': getattr(config, 'num_hidden_layers', 12),
            'vocab_size': getattr(config, 'vocab_size', 32000),
        }
        
        # Get num_key_value_heads for GQA models
        model_info['num_key_value_heads'] = getattr(
            config, 'num_key_value_heads', model_info['num_attention_heads']
        )
        
        # Calculate derived characteristics
        model_info['head_dim'] = model_info['hidden_size'] // model_info['num_attention_heads']
        model_info['total_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate latent_dim for attention mechanisms (num_kv_heads * head_dim)
        model_info['latent_dim'] = model_info['num_key_value_heads'] * model_info['head_dim']
        
        # Check if model uses GQA (Grouped Query Attention)
        model_info['uses_gqa'] = model_info['num_key_value_heads'] < model_info['num_attention_heads']
        
        # Determine model size category
        if model_info['total_params'] < 1e9:
            model_info['size_category'] = 'small'  # <1B params
        elif model_info['total_params'] < 10e9:
            model_info['size_category'] = 'medium'  # 1B-10B params
        else:
            model_info['size_category'] = 'large'  # >10B params
            
        return model_info


class HardwareAnalysisMixin:
    """Mixin providing hardware analysis functionality for configuration classes."""
    
    @staticmethod
    def _analyze_hardware(hardware_budget: str, model: Optional[nn.Module] = None) -> dict:
        """
        Analyze available hardware resources.
        
        This method detects available hardware (GPU/CPU) and estimates memory budget.
        When model is provided, it detects which device the model is on and checks
        that specific device's properties.
        
        Args:
            hardware_budget: Hardware assumption ("auto", "low", "high")
            model: The model to detect device from (optional, used for auto-detection)
            
        Returns:
            dict: Hardware characteristics including:
                - cuda_available: Whether CUDA is available
                - device_count: Number of CUDA devices
                - budget: Detected or specified budget ('low', 'medium', 'high')
                - gpu_memory_gb: GPU memory in GB (detected or estimated)
        """
        hardware_info = {
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if hardware_budget == "auto" and torch.cuda.is_available():
            # Determine which GPU device to check based on model's location
            device_id = 0  # default
            if model is not None:
                try:
                    # Get device from first model parameter
                    first_param = next(model.parameters())
                    if first_param.is_cuda:
                        device_id = first_param.device.index
                except (StopIteration, AttributeError):
                    pass  # Use default device 0
            
            # Auto-detect GPU memory from the actual device where model is located
            gpu_memory_gb = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
            if gpu_memory_gb < 8:
                hardware_info['budget'] = 'low'
            elif gpu_memory_gb > 24:
                hardware_info['budget'] = 'high'
            else:
                hardware_info['budget'] = 'medium'
            hardware_info['gpu_memory_gb'] = gpu_memory_gb
        else:
            hardware_info['budget'] = hardware_budget if hardware_budget != "auto" else 'medium'
            hardware_info['gpu_memory_gb'] = {
                'low': 6, 'medium': 16, 'high': 32
            }.get(hardware_info['budget'], 16)
        
        return hardware_info


class ConfigAnalysisMixin(ModelAnalysisMixin, HardwareAnalysisMixin):
    """
    Combined mixin providing both model and hardware analysis.
    
    This mixin combines ModelAnalysisMixin and HardwareAnalysisMixin to provide
    a complete analysis suite for auto-configuration classes.
    
    Usage:
        class MyConfig(ConfigAnalysisMixin):
            @classmethod
            def auto_from_model(cls, model, ...):
                model_info = cls._analyze_model(model)
                hardware_info = cls._analyze_hardware("auto", model)
                # Use model_info and hardware_info to calculate params
                ...
    """
    pass

