"""
TransMLA Core - Minimal Python module for Multi-head Latent Attention transformations

This module contains only the essential components needed for TransMLA functionality:
- utils: Dataset and evaluation utilities
- partial_rope: Partial RoPE transformations
- lora_qkv: Low-rank QKV operations
- modify_config: Configuration modifications

Usage:
    from external.transmlacore import utils, partial_rope, lora_qkv, modify_config
"""

__version__ = "1.0.0"
__author__ = "FedCore Team"

# Import main functions for convenience
try:
    from .utils import get_dataset, prepare_dataloader, prepare_test_dataloader, evaluate_ppl
    from .partial_rope import partial_rope
    from .lora_qkv import low_rank_qkv
    from .modify_config import modify_config
    
    __all__ = [
        'get_dataset',
        'prepare_dataloader', 
        'prepare_test_dataloader',
        'evaluate_ppl',
        'partial_rope',
        'low_rank_qkv',
        'modify_config'
    ]
    
except ImportError as e:
    # If some dependencies are missing, still allow module import
    print(f"[Warning] TransMLA Core: Some functions may not be available due to missing dependencies: {e}")
    __all__ = []
