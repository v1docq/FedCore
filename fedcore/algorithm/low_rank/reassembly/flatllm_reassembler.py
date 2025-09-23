"""
FLAT-LLM reassembler implementation for FedCore.

This module provides FLAT-LLM (Fine-grained Low-rank Activation Space 
Transformation) functionality integrated with FedCore's reassembly framework.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from .core_reassemblers import Reassembler


class FlatLLMConfig:
    """Configuration class for FLAT-LLM operations."""
    
    def __init__(
        self,
        target_sparsity: float = 0.5,
        tolerance: float = 0.96,
        cal_dataset: str = "wikitext2",
        cal_nsamples: int = 128,
        cal_batch_size: int = 1,
        cal_max_seqlen: int = 4096,
        importance_method: str = "angular",
        apply_head_transforms: bool = True,
        preserve_qk_layers: bool = True,
        device: str = "auto",
        seed: int = 42
    ):
        """
        Initialize FLAT-LLM configuration.
        
        Args:
            target_sparsity: Target compression ratio (0.5 = 50% compression)
            tolerance: Eigenvalue preservation tolerance for PCA
            cal_dataset: Calibration dataset name
            cal_nsamples: Number of calibration samples
            cal_batch_size: Calibration batch size
            cal_max_seqlen: Maximum sequence length for calibration
            importance_method: Method for computing layer importance
            apply_head_transforms: Whether to apply head-wise transformations
            preserve_qk_layers: Whether to preserve Q,K layers (FLAT-LLM strategy)
            device: Device for computations ("auto", "cuda", "cpu")
            seed: Random seed for reproducibility
        """
        self.target_sparsity = target_sparsity
        self.tolerance = tolerance
        self.cal_dataset = cal_dataset
        self.cal_nsamples = cal_nsamples
        self.cal_batch_size = cal_batch_size
        self.cal_max_seqlen = cal_max_seqlen
        self.importance_method = importance_method
        self.apply_head_transforms = apply_head_transforms
        self.preserve_qk_layers = preserve_qk_layers
        self.device = device
        self.seed = seed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


def get_flatllmcore_path() -> Path:
    """
    Get FlatLLMCore path with multiple fallback strategies.
    
    Priority order:
    1. Environment variable FLATLLMCORE_PATH
    2. Relative path from current file (fallback)
    """
    # Strategy 1: Check environment variable
    if env_path := os.environ.get('FLATLLMCORE_PATH'):
        path = Path(env_path)
        if path.exists():
            return path
    
    # Strategy 2: Fallback to relative path
    fallback_path = Path(__file__).parent.parent.parent.parent / "external" / "flatllmcore"
    return fallback_path


FLATLLMCORE_AVAILABLE = False
FLATLLMCORE_ERROR = None


class FlatLLMFunctions:
    """Container for FLAT-LLM functions - eliminates global state."""
    
    def __init__(self):
        self.FlatLLMPruner = None
        self.ImportancePreservingRankSelector = None
        self.FlatLlamaAttention = None
        self.FlatLlamaDecoderLayer = None
        self.FlatMistralAttention = None
        self.FlatMistralDecoderLayer = None
        self._initialized = False
    
    def initialize(self, flatllmcore_path: Path) -> None:
        """Initialize FLAT-LLM functions from given path."""
        if self._initialized:
            return  # Already initialized
            
        # Setup paths
        flatllmcore_str = str(flatllmcore_path)
        if flatllmcore_str not in sys.path:
            sys.path.insert(0, flatllmcore_str)
        
        try:
            # Import FLAT-LLM modules
            from core.prune import FlatLLMPruner
            from core.rank_allocation import ImportancePreservingRankSelector
            from layers.attention_layers import (
                FlatLlamaAttention, FlatLlamaDecoderLayer,
                FlatMistralAttention, FlatMistralDecoderLayer
            )

            # Store functions in instance
            self.FlatLLMPruner = FlatLLMPruner
            self.ImportancePreservingRankSelector = ImportancePreservingRankSelector
            self.FlatLlamaAttention = FlatLlamaAttention
            self.FlatLlamaDecoderLayer = FlatLlamaDecoderLayer
            self.FlatMistralAttention = FlatMistralAttention
            self.FlatMistralDecoderLayer = FlatMistralDecoderLayer
            
            self._initialized = True
            
        except ImportError as e:
            raise ImportError(f"Failed to import FLAT-LLM modules from {flatllmcore_path}: {e}")


# Global FLAT-LLM functions instance
_flatllm_functions = None


def _initialize_flatllm():
    """Initialize FLAT-LLM if available (called lazily)."""
    global FLATLLMCORE_AVAILABLE, FLATLLMCORE_ERROR, _flatllm_functions
    
    if _flatllm_functions is not None:
        return  # Already initialized or attempted
        
    try:
        flatllmcore_path = get_flatllmcore_path()
        _flatllm_functions = FlatLLMFunctions()
        _flatllm_functions.initialize(flatllmcore_path)
        FLATLLMCORE_AVAILABLE = True
        FLATLLMCORE_ERROR = None
    except Exception as e:
        FLATLLMCORE_AVAILABLE = False
        FLATLLMCORE_ERROR = str(e)
        _flatllm_functions = False  # Mark as attempted but failed


def get_flatllm_status():
    """
    Returns detailed information about FLAT-LLM status.
    
    Returns:
        dict: Status information including availability, error, and path
    """
    _initialize_flatllm()  # Lazy initialization
    return {
        'available': FLATLLMCORE_AVAILABLE,
        'error': FLATLLMCORE_ERROR,
        'path': str(get_flatllmcore_path()) if get_flatllmcore_path().exists() else None,
        'initialized': _flatllm_functions is not False and _flatllm_functions is not None and _flatllm_functions._initialized
    }


class FlatLLM(Reassembler):
    """FLAT-LLM reassembler for FedCore integration."""

    @classmethod
    def _replace_attention_layers(cls, model: nn.Module, architecture: str):
        """Replace standard attention layers with FLAT-LLM versions."""
        _initialize_flatllm()
        assert FLATLLMCORE_AVAILABLE, f"FLAT-LLM Core not available: {FLATLLMCORE_ERROR}"
        
        if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
            raise ValueError("Model structure not supported for attention replacement")
        
        layers = model.model.layers
        
        for i, layer in enumerate(layers):
            # Move to CPU to save memory during replacement
            layer = layer.cpu()
            torch.cuda.empty_cache()
            
            if architecture == 'llama':
                # Create new Llama decoder layer
                new_layer = _flatllm_functions.FlatLlamaDecoderLayer(model.config, layer_idx=i)
            elif architecture == 'mistral':
                # Create new Mistral decoder layer
                new_layer = _flatllm_functions.FlatMistralDecoderLayer(model.config, layer_idx=i)
            else:
                raise ValueError(f"Unsupported architecture: {architecture}")
            
            # Copy weights from original layer
            new_layer.load_state_dict(layer.state_dict(), strict=True)
            
            # Move to appropriate device
            device = next(layer.parameters()).device if list(layer.parameters()) else 'cuda'
            dtype = next(layer.parameters()).dtype if list(layer.parameters()) else torch.float16
            new_layer = new_layer.to(device=device, dtype=dtype)
            
            # Replace in model
            model.model.layers[i] = new_layer
            
            print(f"[FlatLLM] Replaced layer {i + 1}/{len(layers)} with {architecture} FLAT-LLM version")
            
            # Cleanup
            del layer
            torch.cuda.empty_cache()

    @classmethod
    def _apply_flat_llm_pruning(
        cls,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: FlatLLMConfig
    ) -> PreTrainedModel:
        """Apply FLAT-LLM pruning algorithm to the model."""
        _initialize_flatllm()
        assert FLATLLMCORE_AVAILABLE, f"FLAT-LLM Core not available: {FLATLLMCORE_ERROR}"
        
        print("[FlatLLM] Starting FLAT-LLM pruning process...")
        
        # Initialize pruner
        pruner = _flatllm_functions.FlatLLMPruner(
            model=model,
            tokenizer=tokenizer,
            target_sparsity=config.target_sparsity,
            tolerance=config.tolerance,
            device=config.device
        )
        
        # Apply pruning
        pruned_model = pruner.prune_model(
            n_samples=config.cal_nsamples,
            dataset_name=config.cal_dataset
        )
        
        # Print compression statistics
        stats = pruner.get_compression_stats()
        print("[FlatLLM] Compression Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return pruned_model

    @classmethod
    def _validate_inputs(cls, model: nn.Module, tokenizer: Optional[Any]):
        """Validate inputs for FLAT-LLM processing."""
        if not hasattr(model, 'config'):
            raise ValueError("Model must have a config attribute")
        
        if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
            raise ValueError("Model must have transformer layers accessible via model.layers")
        
        if tokenizer is None:
            raise ValueError("Tokenizer is required for FLAT-LLM processing")

    @classmethod
    def reassemble(
        cls,
        model: nn.Module,
        architecture: str,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[FlatLLMConfig] = None,
        additional_mapping: Optional[Dict] = None,
        **kwargs
    ) -> nn.Module:
        """
        Main FLAT-LLM reassembly method.
        
        Args:
            model: Model to process with FLAT-LLM
            tokenizer: Tokenizer for the model (required)
            config: FLAT-LLM configuration
            additional_mapping: Additional module mappings (optional)
            **kwargs: Additional configuration parameters
            
        Returns:
            Model with FLAT-LLM transformations applied
        """
        _initialize_flatllm()
        assert FLATLLMCORE_AVAILABLE, f"FLAT-LLM Core not available: {FLATLLMCORE_ERROR}"
        
        print("[FlatLLM] Starting FLAT-LLM reassembly...")
        
        # Validate inputs
        cls._validate_inputs(model, tokenizer)
        
        # Apply additional mappings if provided
        if additional_mapping:
            cls._apply_additional_mapping(model, additional_mapping)
        
        # Prepare configuration
        if config is None:
            config = FlatLLMConfig()
        
        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Set random seed
        torch.manual_seed(config.seed)
        
        # Phase 1: Replace attention layers with FLAT-LLM versions
        print("[FlatLLM] Phase 1: Replacing attention layers...")
        cls._replace_attention_layers(model, architecture)
        
        # Phase 2: Apply FLAT-LLM pruning algorithm
        print("[FlatLLM] Phase 2: Applying FLAT-LLM pruning...")
        model = cls._apply_flat_llm_pruning(model, tokenizer, config)
        
        # Phase 3: Validate device consistency
        print("[FlatLLM] Phase 3: Validating device consistency...")
        cls._validate_device_consistency(model)
        
        print("[FlatLLM] FLAT-LLM reassembly completed successfully!")
        return model
