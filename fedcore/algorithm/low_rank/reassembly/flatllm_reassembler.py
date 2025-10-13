"""
FLAT-LLM reassembly functionality.

Contains FLAT-LLM-specific reassembly logic with absorption mechanism.
Implements physical model compression through activation-based dimension reduction.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Union, Literal, List
from enum import Enum

import torch
import torch.nn as nn

from .core_reassemblers import Reassembler


# FLAT-LLM configuration
def get_flatllm_path() -> Path:
    """
    Get FLAT-LLM path with multiple fallback strategies for scalability.
    
    Priority order:
    1. Environment variable FLATLLM_PATH
    2. Installed package location
    3. Relative path from current file (fallback)
    """
    # Strategy 1: Check environment variable (highest priority)
    if env_path := os.environ.get('FLATLLM_PATH'):
        path = Path(env_path)
        if path.exists():
            return path
    
    # Strategy 2: Try to find installed package
    try:
        import flatllmcore
        return Path(flatllmcore.__file__).parent
    except ImportError:
        pass
    
    # Strategy 3: Fallback to relative path (for development)
    fallback_path = Path(__file__).parent.parent.parent.parent.parent / "external" / "flatllmcore"
    return fallback_path


FLATLLM_AVAILABLE = False
FLATLLM_ERROR = None


class FlatLLMFunctions:
    """Container for FLAT-LLM functions - eliminates global state for scalability."""
    
    def __init__(self):
        self.AbsorptionCompressor = None
        self.ActivationCollector = None
        self._initialized = False
    
    def initialize(self, flatllm_path: Path) -> None:
        """Initialize FLAT-LLM functions from given path."""
        if self._initialized:
            return  # Already initialized
            
        # Setup paths
        flatllm_str = str(flatllm_path)
        core_path = flatllm_str + "/core"
        if core_path not in sys.path:
            sys.path.insert(0, core_path)
        if flatllm_str not in sys.path:
            sys.path.insert(0, flatllm_str)
        
        try:
            # Import FLAT-LLM modules
            from absorption import AbsorptionCompressor, ActivationCollector
            
            # Store classes in instance (no global state!)
            self.AbsorptionCompressor = AbsorptionCompressor
            self.ActivationCollector = ActivationCollector
            
            self._initialized = True
            
        except ImportError as e:
            raise ImportError(f"Failed to import FLAT-LLM modules from {flatllm_path}: {e}")


# Global FLAT-LLM functions instance
_flatllm_functions = None

def _initialize_flatllm():
    """Initialize FLAT-LLM if available (called lazily)."""
    global FLATLLM_AVAILABLE, FLATLLM_ERROR, _flatllm_functions
    
    if _flatllm_functions is not None:
        return  # Already initialized or attempted
        
    try:
        flatllm_path = get_flatllm_path()
        _flatllm_functions = FlatLLMFunctions()
        _flatllm_functions.initialize(flatllm_path)
        FLATLLM_AVAILABLE = True
        FLATLLM_ERROR = None
    except Exception as e:
        FLATLLM_AVAILABLE = False
        FLATLLM_ERROR = str(e)
        _flatllm_functions = False  # Mark as attempted but failed


def get_flatllm_status():
    """
    Returns detailed information about FLAT-LLM status.
    
    Returns:
        dict: Status information including availability, error, and path
    """
    _initialize_flatllm()  # Lazy initialization
    return {
        'available': FLATLLM_AVAILABLE,
        'error': FLATLLM_ERROR,
        'path': str(get_flatllm_path()) if get_flatllm_path().exists() else None,
        'initialized': _flatllm_functions is not False and _flatllm_functions is not None and _flatllm_functions._initialized
    }


class FlatLLMConfig:
    """Configuration for FLAT-LLM compression."""

    def __init__(self, 
                 target_sparsity: float = 0.7,
                 tolerance: float = 0.96,
                 layer_selection: Union[str, List[int]] = "auto",
                 compression_ratio: float = 0.3,
                 cal_dataset: str = "wikitext2",
                 cal_nsamples: int = 8,
                 cal_batch_size: int = 1,
                 cal_max_seqlen: int = 128,
                 device: str = "auto",
                 verbose: bool = True,
                 seed: int = 42):
        """
        Args:
            target_sparsity: Sparsity ratio (0.7 = keep 70% = 30% compression)
            tolerance: Variance preservation threshold (0.96 = keep 96% variance)
            layer_selection: Which layers to compress
                - "auto": Automatically select based on compression_ratio
                - "all": Compress all layers (not recommended)
                - List[int]: Explicit layer indices
            compression_ratio: Fraction of layers to compress (0.3 = 30% of layers)
            cal_dataset: Calibration dataset name
            cal_nsamples: Number of calibration samples
            cal_batch_size: Batch size for calibration
            cal_max_seqlen: Maximum sequence length for calibration
            device: Device for computation ("auto", "cuda", "cpu")
            verbose: Print progress information
            seed: Random seed
        """
        self.target_sparsity = target_sparsity
        self.tolerance = tolerance
        self.layer_selection = layer_selection
        self.compression_ratio = compression_ratio
        self.cal_dataset = cal_dataset
        self.cal_nsamples = cal_nsamples
        self.cal_batch_size = cal_batch_size
        self.cal_max_seqlen = cal_max_seqlen
        self.device = device
        self.verbose = verbose
        self.seed = seed

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def auto_from_model(cls, 
                       model: nn.Module,
                       priority: Literal["quality", "speed", "memory", "balanced"] = "balanced",
                       compression: Literal["light", "medium", "aggressive"] = "medium",
                       hardware_budget: Literal["auto", "low", "high"] = "auto",
                       **kwargs):
        """
        Automatically configure FLAT-LLM parameters based on model analysis.
        
        Args:
            model: The transformer model to be compressed
            priority: Optimization priority
                - "quality": Maximize model quality, minimal compression
                - "speed": Balance speed and quality
                - "memory": Maximize compression, minimize memory
                - "balanced": Balance between quality, speed and memory (default)
            compression: Compression level
                - "light": ~10-15% compression, best quality
                - "medium": ~20-30% compression, balanced (default)
                - "aggressive": ~40-50% compression, maximum size reduction
            hardware_budget: Hardware resource assumption
                - "auto": Automatically detect available resources (default)
                - "low": Assume limited resources (<8GB GPU memory)
                - "high": Assume high-end hardware (>24GB GPU memory)
            **kwargs: Additional parameters to override automatic settings
            
        Returns:
            FlatLLMConfig: Automatically configured FLAT-LLM configuration
            
        Example:
            >>> config = FlatLLMConfig.auto_from_model(model, priority="quality")
            >>> config = FlatLLMConfig.auto_from_model(model, compression="aggressive")
        """
        # Analyze model characteristics
        model_info = cls._analyze_model(model)
        hardware_info = cls._analyze_hardware(hardware_budget)
        
        # Calculate automatic parameters
        auto_params = cls._calculate_auto_parameters(
            model_info, hardware_info, priority, compression
        )
        
        # Override with any user-provided kwargs
        auto_params.update(kwargs)
        
        return cls(**auto_params)
    
    @staticmethod
    def _analyze_model(model: nn.Module) -> dict:
        """
        Analyze model architecture and extract relevant characteristics.
        
        Args:
            model: The transformer model to analyze
            
        Returns:
            dict: Model characteristics including size, architecture, dimensions
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
    
    @staticmethod
    def _analyze_hardware(hardware_budget: str) -> dict:
        """
        Analyze available hardware resources.
        
        Args:
            hardware_budget: Hardware assumption ("auto", "low", "high")
            
        Returns:
            dict: Hardware characteristics and resource constraints
        """
        hardware_info = {
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if hardware_budget == "auto" and torch.cuda.is_available():
            # Auto-detect GPU memory
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
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
    
    @staticmethod
    def _calculate_auto_parameters(model_info: dict, hardware_info: dict, 
                                 priority: str, compression: str) -> dict:
        """
        Calculate optimal FLAT-LLM parameters based on model and hardware analysis.
        
        Args:
            model_info: Model characteristics from _analyze_model()
            hardware_info: Hardware characteristics from _analyze_hardware()
            priority: Optimization priority
            compression: Compression level
            
        Returns:
            dict: Calculated FLAT-LLM parameters
        """
        params = {}
        
        # Base parameters from model architecture
        num_layers = model_info['num_hidden_layers']
        size_category = model_info['size_category']
        
        # Calculate target_sparsity (what to keep) based on compression level
        # Note: target_sparsity = 0.7 means keep 70% = 30% reduction
        sparsity_map = {
            'light': 0.85,      # Keep 85% = 15% reduction
            'medium': 0.70,     # Keep 70% = 30% reduction
            'aggressive': 0.60  # Keep 60% = 40% reduction
        }
        params['target_sparsity'] = sparsity_map[compression]
        
        # Calculate compression_ratio (which layers to compress)
        layer_ratio_map = {
            'light': 0.15,      # Compress 15% of layers
            'medium': 0.30,     # Compress 30% of layers
            'aggressive': 0.50  # Compress 50% of layers
        }
        params['compression_ratio'] = layer_ratio_map[compression]
        
        # Adjust for priority
        if priority == "quality":
            params['target_sparsity'] = min(params['target_sparsity'] + 0.1, 0.95)
            params['compression_ratio'] = max(params['compression_ratio'] - 0.1, 0.1)
            params['tolerance'] = 0.98  # Higher tolerance = better quality
        elif priority == "memory":
            params['target_sparsity'] = max(params['target_sparsity'] - 0.1, 0.5)
            params['compression_ratio'] = min(params['compression_ratio'] + 0.2, 0.7)
            params['tolerance'] = 0.90  # Lower tolerance = more compression
        else:  # balanced or speed
            params['tolerance'] = 0.96  # Standard tolerance
        
        # Calibration parameters based on model size and hardware
        cal_params = {
            'small': {'nsamples': 8, 'batch_size': 1, 'seqlen': 128},
            'medium': {'nsamples': 16, 'batch_size': 1, 'seqlen': 128},
            'large': {'nsamples': 32, 'batch_size': 1, 'seqlen': 256}
        }
        
        base_cal = cal_params[size_category]
        
        # Adjust based on priority
        if priority == "speed":
            params['cal_nsamples'] = max(4, base_cal['nsamples'] // 2)
            params['cal_batch_size'] = min(base_cal['batch_size'] * 2, 2)
        elif priority == "quality":
            params['cal_nsamples'] = base_cal['nsamples'] * 2
            params['cal_batch_size'] = base_cal['batch_size']
        else:  # balanced or memory
            params['cal_nsamples'] = base_cal['nsamples']
            params['cal_batch_size'] = base_cal['batch_size']
        
        params['cal_max_seqlen'] = base_cal['seqlen']
        
        # Adjust batch sizes based on hardware budget
        if hardware_info['budget'] == 'low':
            params['cal_batch_size'] = 1
            params['cal_max_seqlen'] = min(params['cal_max_seqlen'], 128)
        
        # Memory optimization for memory priority
        if priority == "memory":
            params['cal_batch_size'] = 1
            params['cal_max_seqlen'] = 128
        
        # Other parameters
        params['layer_selection'] = "auto"
        params['cal_dataset'] = "wikitext2"
        params['device'] = "auto"
        params['verbose'] = True
        params['seed'] = 42
        
        return params


def _prepare_calibration_data(model, tokenizer, config: FlatLLMConfig):
    """
    Prepare calibration data for FLAT-LLM compression.
    
    Args:
        model: The model to compress
        tokenizer: Tokenizer for the model
        config: FLAT-LLM configuration
        
    Returns:
        torch.Tensor: Calibration input_ids [N, S]
    """
    import random
    
    # Set seed
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Get calibration dataset
    if config.cal_dataset == "wikitext2":
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        
        # Sample texts
        texts = []
        for _ in range(config.cal_nsamples):
            idx = random.randint(0, len(dataset) - 1)
            text = dataset[idx]["text"]
            if len(text.strip()) > 50:  # Skip empty or very short texts
                texts.append(text)
        
        # If not enough valid texts, use some defaults
        while len(texts) < config.cal_nsamples:
            texts.append("The future of artificial intelligence is bright and full of possibilities.")
    else:
        # Default calibration texts
        default_texts = [
            "The future of artificial intelligence is bright and full of possibilities.",
            "Machine learning models are becoming more efficient and powerful.",
            "Natural language processing has made significant progress in recent years.",
            "Deep learning architectures continue to evolve and improve.",
            "Transformer models have revolutionized the field of natural language processing.",
            "Large language models demonstrate impressive capabilities across various tasks.",
            "Neural networks can learn complex patterns from data efficiently.",
            "Attention mechanisms enable models to focus on relevant information.",
        ]
        texts = default_texts[:config.cal_nsamples]
    
    # Tokenize
    calibration_inputs = tokenizer(
        texts,
        max_length=config.cal_max_seqlen,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    return calibration_inputs.input_ids


def _select_layers_to_compress(model, config: FlatLLMConfig) -> List[int]:
    """
    Select which layers to compress based on configuration.
    
    Args:
        model: The model to compress
        config: FLAT-LLM configuration
        
    Returns:
        List[int]: Indices of layers to compress
    """
    num_layers = len(model.model.layers)
    
    if isinstance(config.layer_selection, list):
        # Explicit layer indices
        return config.layer_selection
    elif config.layer_selection == "all":
        # All layers (not recommended)
        return list(range(num_layers))
    elif config.layer_selection == "auto":
        # Automatically select based on compression_ratio
        # Strategy: Select evenly distributed layers (every Nth layer)
        num_to_compress = max(1, int(num_layers * config.compression_ratio))
        step = num_layers // num_to_compress
        return list(range(0, num_layers, step))[:num_to_compress]
    else:
        raise ValueError(f"Unknown layer_selection: {config.layer_selection}")


def compress_model_with_flatllm(model: nn.Module, tokenizer, config: FlatLLMConfig):
    """
    Complete FLAT-LLM compression of model.
    Simple implementation using AbsorptionCompressor.
    """
    _initialize_flatllm()  # Ensure FLAT-LLM is initialized
    assert FLATLLM_AVAILABLE, f"FLAT-LLM not available: {FLATLLM_ERROR}"
    
    if config.verbose:
        print("[FLAT-LLM] Starting absorption-based compression...")
    
    # Prepare calibration data
    if config.verbose:
        print(f"[FLAT-LLM] Preparing calibration data ({config.cal_nsamples} samples)...")
    
    calibration_input_ids = _prepare_calibration_data(model, tokenizer, config)
    
    # Select layers to compress
    layers_to_compress = _select_layers_to_compress(model, config)
    
    if config.verbose:
        print(f"[FLAT-LLM] Compressing {len(layers_to_compress)}/{len(model.model.layers)} layers")
        print(f"[FLAT-LLM] Layer indices: {layers_to_compress}")
    
    # Handle device
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create compressor
    AbsorptionCompressor = _flatllm_functions.AbsorptionCompressor
    compressor = AbsorptionCompressor(
        model=model,
        target_sparsity=config.target_sparsity,
        tolerance=config.tolerance,
        device=device
    )
    
    # Step 1: Collect activations from all target layers
    if config.verbose:
        print("[FLAT-LLM] Step 1/3: Collecting activations...")
    
    compressor.collect_all_activations(
        layer_indices=layers_to_compress,
        calibration_input_ids=calibration_input_ids
    )
    
    # Step 2: Apply absorption to each layer
    if config.verbose:
        print("[FLAT-LLM] Step 2/3: Applying absorption...")
    
    for layer_idx in layers_to_compress:
        if config.verbose:
            print(f"[FLAT-LLM]   Compressing layer {layer_idx}...")
        
        # MLP compression
        compressor.apply_absorption_mlp(
            layer_idx=layer_idx,
            sparsity_ratio=config.target_sparsity
        )
        
        # Attention compression
        compressor.apply_absorption_attention(
            layer_idx=layer_idx,
            sparsity_ratio=config.target_sparsity
        )
    
    # Step 3: Patch compressed layers for inference
    if config.verbose:
        print("[FLAT-LLM] Step 3/3: Patching compressed layers...")
    
    compressor.patch_compressed_layers(layers_to_compress)
    
    # Step 4: Move model back to original device
    # AbsorptionCompressor moves model to CPU after collection, restore it
    if config.verbose:
        print(f"[FLAT-LLM] Moving model to {device}...")
    
    model = model.to(device)
    
    if config.verbose:
        print("[FLAT-LLM] Compression completed successfully!")
    
    return model


class FlatLLM(Reassembler):
    """Specialized FLAT-LLM reassembler with absorption mechanism."""

    @classmethod
    def reassemble(cls, model: nn.Module, tokenizer, config: Optional[FlatLLMConfig] = None,
                          additional_mapping: dict = None, **kwargs):
        """
        FLAT-LLM compression with absorption.
        
        Args:
            model: Model to compress
            tokenizer: Tokenizer for calibration data
            config: FLAT-LLM configuration (optional)
            additional_mapping: Additional module mappings (optional)
            **kwargs: Additional config parameters to override
            
        Returns:
            nn.Module: Compressed model with physically reduced dimensions
            
        Example:
            >>> from fedcore.algorithm.low_rank.reassembly import FlatLLM, FlatLLMConfig
            >>> config = FlatLLMConfig.auto_from_model(model, compression="medium")
            >>> compressed_model = FlatLLM.reassemble(model, tokenizer, config)
        """
        _initialize_flatllm()  # Ensure FLAT-LLM is initialized
        assert FLATLLM_AVAILABLE, f"FLAT-LLM not available: {FLATLLM_ERROR}"
        
        # Apply mappings
        if additional_mapping:
            cls._apply_additional_mapping(model, additional_mapping)
        
        # Prepare config
        config = config or FlatLLMConfig()
        
        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Handle auto-device
        if config.device == "auto":
            config.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Compress
        model = compress_model_with_flatllm(model, tokenizer, config)
        cls._validate_device_consistency(model)
        return model

