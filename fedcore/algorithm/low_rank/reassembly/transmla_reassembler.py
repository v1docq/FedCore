"""
TransMLA reassembly functionality.

Contains TransMLA-specific reassembly logic moved from quantization utils.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Union, Literal
from enum import Enum

import torch
import torch.nn as nn

from .core_reassemblers import Reassembler
from .config_mixins import ConfigAnalysisMixin


# TransMLA configuration
def get_transmla_path() -> Path:
    """
    Get TransMLA path with multiple fallback strategies for scalability.
    
    Priority order:
    1. Environment variable TRANSMLA_PATH
    2. Installed package location
    3. Relative path from current file (fallback)
    """
    # Strategy 1: Check environment variable (highest priority)
    if env_path := os.environ.get('TRANSMLA_PATH'):
        path = Path(env_path)
        if path.exists():
            return path
    
    # Strategy 2: Try to find installed package
    try:
        import transmlacore
        return Path(transmlacore.__file__).parent
    except ImportError:
        pass
    
    # Strategy 3: Fallback to relative path (for development)
    fallback_path = Path(__file__).parent.parent.parent.parent.parent / "external" / "transmlacore"
    return fallback_path


TRANSMLA_AVAILABLE = False
TRANSMLA_ERROR = None


class TransMLAFunctions:
    """Container for TransMLA functions - eliminates global state for scalability."""
    
    def __init__(self):
        self.partial_rope = None
        self.low_rank_qkv = None
        self.modify_config = None
        self.get_dataset = None
        self.prepare_dataloader = None
        self.prepare_test_dataloader = None
        self.evaluate_ppl = None
        self._initialized = False
    
    def initialize(self, transmla_path: Path) -> None:
        """Initialize TransMLA functions from given path."""
        if self._initialized:
            return  # Already initialized
            
        # Setup paths
        transmla_str = str(transmla_path)
        if transmla_str not in sys.path:
            sys.path.insert(0, transmla_str)
        
        try:
            # Import TransMLA modules
            import partial_rope as transmla_partial_rope
            import utils as transmla_utils
            import lora_qkv as transmla_lora_qkv
            import modify_config as transmla_modify_config

            # Store functions in instance (no global state!)
            self.partial_rope = transmla_partial_rope.partial_rope
            self.low_rank_qkv = transmla_lora_qkv.low_rank_qkv
            self.modify_config = transmla_modify_config.modify_config
            self.get_dataset = transmla_utils.get_dataset
            self.prepare_dataloader = transmla_utils.prepare_dataloader
            self.prepare_test_dataloader = transmla_utils.prepare_test_dataloader
            self.evaluate_ppl = transmla_utils.evaluate_ppl
            
            self._initialized = True
            
        except ImportError as e:
            raise ImportError(f"Failed to import TransMLA modules from {transmla_path}: {e}")


# Global TransMLA functions instance
_transmla_functions = None

def _initialize_transmla():
    """Initialize TransMLA if available (called lazily)."""
    global TRANSMLA_AVAILABLE, TRANSMLA_ERROR, _transmla_functions
    
    if _transmla_functions is not None:
        return  # Already initialized or attempted
        
    try:
        transmla_path = get_transmla_path()
        _transmla_functions = TransMLAFunctions()
        _transmla_functions.initialize(transmla_path)
        TRANSMLA_AVAILABLE = True
        TRANSMLA_ERROR = None
    except Exception as e:
        TRANSMLA_AVAILABLE = False
        TRANSMLA_ERROR = str(e)
        _transmla_functions = False  # Mark as attempted but failed


def get_transmla_status():
    """
    Returns detailed information about TransMLA status.
    
    Returns:
        dict: Status information including availability, error, and path
    """
    _initialize_transmla()  # Lazy initialization
    return {
        'available': TRANSMLA_AVAILABLE,
        'error': TRANSMLA_ERROR,
        'path': str(get_transmla_path()) if get_transmla_path().exists() else None,
        'initialized': _transmla_functions is not False and _transmla_functions is not None and _transmla_functions._initialized
    }


class TransMLAConfig(ConfigAnalysisMixin):
    """Configuration for TransMLA conversion."""

    def __init__(self, 
                 freqfold: Union[str, int] = "auto",
                 collapse: Union[str, int] = "auto", 
                 qk_mqa_dim: int = 64,
                 q_lora_rank: Optional[int] = None,
                 kv_lora_rank: int = 512,
                 balance_kv_ratio: float = 1.0,
                 use_qkv_norm: bool = False,
                 cal_dataset: str = "wikitext2",
                 cal_nsamples: int = 128,
                 cal_batch_size: int = 8,
                 cal_max_seqlen: int = 256,
                 ppl_eval_batch_size: int = 2,
                 deepseek_style: bool = True,
                 dtype: str = "bf16",
                 device: str = "auto",
                 seed: int = 42):
        self.freqfold = freqfold
        self.collapse = collapse
        self.qk_mqa_dim = qk_mqa_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.balance_kv_ratio = balance_kv_ratio
        self.use_qkv_norm = use_qkv_norm
        self.cal_dataset = cal_dataset
        self.cal_nsamples = cal_nsamples
        self.cal_batch_size = cal_batch_size
        self.cal_max_seqlen = cal_max_seqlen
        self.ppl_eval_batch_size = ppl_eval_batch_size
        self.deepseek_style = deepseek_style
        self.dtype = dtype
        self.device = device
        self.seed = seed

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}
    
    def validate(self, model: nn.Module) -> None:
        """
        Validate configuration against model architecture constraints.
        
        This method checks that the configuration satisfies all TransMLA requirements
        and constraints based on the model architecture.
        
        Args:
            model: The transformer model to validate against
            
        Raises:
            AssertionError: If configuration violates TransMLA constraints
            ValueError: If model configuration is invalid
        """
        # Extract model attributes
        config = getattr(model, 'config', None)
        if config is None:
            raise ValueError("Model must have a config attribute")
        
        hidden_size = getattr(config, 'hidden_size', 768)
        num_heads = getattr(config, 'num_attention_heads', 12)
        head_dim = getattr(config, 'head_dim', hidden_size // max(1, num_heads))
        num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
        latent_dim = num_kv_heads * head_dim
        
        # Validation 1: qk_mqa_dim should divide head_dim evenly
        if isinstance(self.qk_mqa_dim, int) and self.qk_mqa_dim > 0:
            if head_dim % self.qk_mqa_dim != 0:
                raise ValueError(
                    f"Configuration validation failed: qk_mqa_dim={self.qk_mqa_dim} "
                    f"must divide head_dim={head_dim} evenly. "
                    f"Valid divisors: {[d for d in [128, 64, 32, 16, 8] if head_dim % d == 0 and d <= head_dim]}"
                )
        
        # Validation 2: kv_lora_rank constraint (critical TransMLA requirement)
        max_kv_rank = 2 * latent_dim - int(self.qk_mqa_dim)
        if self.kv_lora_rank >= max_kv_rank:
            raise ValueError(
                f"Configuration validation failed: kv_lora_rank={self.kv_lora_rank} "
                f"must be < 2*latent_dim - qk_mqa_dim = {max_kv_rank}. "
                f"Suggested max value: {int(max_kv_rank * 0.9)}"
            )
        
        # Validation 3: freqfold and collapse must be 'auto' or positive integers
        if self.freqfold != "auto" and not (isinstance(self.freqfold, int) and self.freqfold > 0):
            raise ValueError(
                f"Configuration validation failed: freqfold must be 'auto' or a positive integer, "
                f"got {self.freqfold}"
            )
        
        if self.collapse != "auto" and not (isinstance(self.collapse, int) and self.collapse > 0):
            raise ValueError(
                f"Configuration validation failed: collapse must be 'auto' or a positive integer, "
                f"got {self.collapse}"
            )
        
        # Validation 4: calibration parameters sanity checks
        if self.cal_nsamples <= 0:
            raise ValueError(
                f"Configuration validation failed: cal_nsamples must be positive, "
                f"got {self.cal_nsamples}"
            )
        
        if self.cal_batch_size <= 0:
            raise ValueError(
                f"Configuration validation failed: cal_batch_size must be positive, "
                f"got {self.cal_batch_size}"
            )
        
        if self.cal_max_seqlen <= 0:
            raise ValueError(
                f"Configuration validation failed: cal_max_seqlen must be positive, "
                f"got {self.cal_max_seqlen}"
            )
    
    @classmethod
    def auto_from_model(cls, 
                       model: nn.Module,
                       priority: Literal["quality", "speed", "memory", "balanced"] = "balanced",
                       compression: Literal["light", "medium", "aggressive"] = "medium",
                       hardware_budget: Literal["auto", "low", "high"] = "auto",
                       **kwargs):
        """
        Automatically configure TransMLA parameters based on model analysis.
        
        This method analyzes the model architecture and available hardware resources
        to automatically determine optimal TransMLA configuration parameters.
        
        Args:
            model: The transformer model to be converted
            priority: Optimization priority
                - "quality": Maximize model quality, slower conversion
                - "speed": Maximize inference speed, may reduce quality
                - "memory": Minimize memory usage
                - "balanced": Balance between quality, speed and memory (default)
            compression: Compression level
                - "light": ~20% compression, best quality
                - "medium": ~50% compression, balanced (default)
                - "aggressive": ~70% compression, maximum size reduction
            hardware_budget: Hardware resource assumption
                - "auto": Automatically detect available resources (default)
                - "low": Assume limited resources (<8GB GPU memory)
                - "high": Assume high-end hardware (>24GB GPU memory)
            **kwargs: Additional parameters to override automatic settings
            
        Returns:
            TransMLAConfig: Automatically configured TransMLA configuration
            
        Example:
            >>> config = TransMLAConfig.auto_from_model(model, priority="quality")
            >>> config = TransMLAConfig.auto_from_model(model, compression="aggressive")
        """
        # Analyze model characteristics
        model_info = cls._analyze_model(model)
        hardware_info = cls._analyze_hardware(hardware_budget, model)
        
        # Calculate automatic parameters
        auto_params = cls._calculate_auto_parameters(
            model_info, hardware_info, priority, compression
        )
        
        # Override with any user-provided kwargs
        auto_params.update(kwargs)
        
        # Create config and validate it
        config = cls(**auto_params)
        config.validate(model)
        
        return config
    
    @staticmethod
    def _resolve_dtype(dtype: str, priority: str = "balanced") -> str:
        """
        Resolve dtype from "auto" to concrete value based on hardware capabilities.
        
        Args:
            dtype: dtype value ("auto", "fp16", "bf16", "fp32")
            priority: optimization priority (for context)
            
        Returns:
            str: Resolved dtype value
        """
        if dtype != "auto":
            return dtype
        
        # Memory priority always uses fp16 for maximum memory savings
        if priority == "memory":
            return "fp16"
        
        # For other priorities, auto-detect based on CUDA availability
        if torch.cuda.is_available():
            # Check if bfloat16 is supported
            if torch.cuda.is_bf16_supported():
                return "bf16"
            else:
                return "fp16"
        else:
            return "fp32"
    
    @staticmethod
    def _calculate_auto_parameters(model_info: dict, hardware_info: dict, 
                                 priority: str, compression: str) -> dict:
        """
        Calculate optimal TransMLA parameters based on model and hardware analysis.
        
        Args:
            model_info: Model characteristics from _analyze_model()
            hardware_info: Hardware characteristics from _analyze_hardware()
            priority: Optimization priority
            compression: Compression level
            
        Returns:
            dict: Calculated TransMLA parameters
        """
        params = {}
        
        # Base parameters from model architecture
        hidden_size = model_info['hidden_size']
        head_dim = model_info['head_dim']
        num_heads = model_info['num_attention_heads']
        size_category = model_info['size_category']
        
        # Get num_key_value_heads (for GQA models)
        num_kv_heads = model_info.get('num_key_value_heads', num_heads)
        
        # Calculate correct latent_dim for TransMLA
        # latent_dim = num_key_value_heads * head_dim (per TransMLA implementation)
        latent_dim = num_kv_heads * head_dim
        
        # Calculate qk_mqa_dim based on head_dim
        # qk_mqa_dim should be a divisor of head_dim for proper collapse calculation
        # Common values: 32, 64, 128
        if head_dim >= 128:
            params['qk_mqa_dim'] = 64  # Standard for larger models
        elif head_dim >= 64:
            params['qk_mqa_dim'] = 64  # Match head_dim for Qwen2.5-0.5B
        else:
            params['qk_mqa_dim'] = 32  # For very small models
        
        # Ensure qk_mqa_dim divides head_dim evenly
        if head_dim % params['qk_mqa_dim'] != 0:
            # Find largest divisor <= 64
            for candidate in [64, 32, 16, 8]:
                if head_dim % candidate == 0 and candidate <= head_dim:
                    params['qk_mqa_dim'] = candidate
                    break
        
        # Calculate kv_lora_rank based on compression level
        # Constraint: kv_lora_rank < 2*latent_dim - qk_mqa_dim
        max_kv_rank = 2 * latent_dim - params['qk_mqa_dim']
        
        compression_factors = {
            'light': 0.7,      # 70% of max_kv_rank
            'medium': 0.5,     # 50% of max_kv_rank
            'aggressive': 0.3  # 30% of max_kv_rank
        }
        factor = compression_factors[compression]
        target_rank = int(max_kv_rank * factor)
        
        # Apply reasonable bounds (adaptive minimum based on model size)
        # For small models (<1B params), allow lower minimum rank for better compression
        min_rank = 32 if size_category == 'small' else (64 if size_category == 'medium' else 128)
        params['kv_lora_rank'] = max(min_rank, min(target_rank, 1024))
        
        # Ensure constraint is satisfied with safety margin
        if params['kv_lora_rank'] >= max_kv_rank:
            params['kv_lora_rank'] = int(max_kv_rank * 0.9)  # 90% of max for safety
        
        # Always use auto-detection for freqfold and collapse - they depend on each other
        params['freqfold'] = "auto"
        params['collapse'] = "auto"
        
        # Calibration parameters based on model size and hardware
        cal_params = {
            'small': {'nsamples': 16, 'batch_size': 2, 'seqlen': 128},
            'medium': {'nsamples': 32, 'batch_size': 2, 'seqlen': 128},
            'large': {'nsamples': 64, 'batch_size': 2, 'seqlen': 256}
        }
        
        base_cal = cal_params[size_category]
        
        # Adjust based on priority
        if priority == "speed":
            params['cal_nsamples'] = max(8, base_cal['nsamples'] // 2)
            params['cal_batch_size'] = min(base_cal['batch_size'] * 2, 8)
            params['ppl_eval_batch_size'] = 0  # Skip evaluation for speed
        elif priority == "quality":
            params['cal_nsamples'] = base_cal['nsamples'] * 2
            params['cal_batch_size'] = max(base_cal['batch_size'] // 2, 1)
            params['ppl_eval_batch_size'] = 2
        else:  # balanced or memory
            params['cal_nsamples'] = base_cal['nsamples']
            params['cal_batch_size'] = base_cal['batch_size']
            params['ppl_eval_batch_size'] = 1 if priority == "balanced" else 0
        
        params['cal_max_seqlen'] = base_cal['seqlen']
        
        # Adjust batch sizes based on hardware budget
        if hardware_info['budget'] == 'low':
            params['cal_batch_size'] = max(params['cal_batch_size'] // 2, 1)
            params['ppl_eval_batch_size'] = min(params['ppl_eval_batch_size'], 1)
        elif hardware_info['budget'] == 'high':
            params['cal_batch_size'] = min(params['cal_batch_size'] * 2, 8)
        
        # Memory optimization for memory priority
        if priority == "memory":
            params['cal_batch_size'] = 1
            params['cal_max_seqlen'] = min(params['cal_max_seqlen'], 128)
        
        # Resolve dtype using centralized logic
        params['dtype'] = TransMLAConfig._resolve_dtype("auto", priority)
        
        # Other parameters
        params['q_lora_rank'] = None  # Keep disabled by default
        params['balance_kv_ratio'] = 1.0
        params['use_qkv_norm'] = False
        params['cal_dataset'] = "wikitext2"
        params['deepseek_style'] = True
        params['device'] = "auto"
        params['seed'] = 42
        
        return params


def convert_model_to_mla(model: nn.Module, tokenizer, config: TransMLAConfig):
    """
    Complete MLA conversion of model using TransMLA.
    Simple implementation without try/except blocks.
    """
    _initialize_transmla()  # Ensure TransMLA is initialized
    assert TRANSMLA_AVAILABLE, f"TransMLA not available: {TRANSMLA_ERROR}"
    
    # Prepare calibration data using encapsulated functions
    dataset = _transmla_functions.get_dataset(config.cal_dataset)
    train_loader = _transmla_functions.prepare_dataloader(
        dataset=dataset["train"],
        tokenizer=tokenizer,
        max_seqlen=config.cal_max_seqlen,
        batch_size=config.cal_batch_size,
        nsamples=config.cal_nsamples,
        seed=config.seed,
    )
    
    test_loader = _transmla_functions.prepare_test_dataloader(
        dataset=dataset["test"], 
        tokenizer=tokenizer, 
        batch_size=max(1, config.ppl_eval_batch_size)
    )
    
    print("[TransMLA] Starting MLA conversion...")
    
    # Stage 1: Partial RoPE
    config_dict = config.to_dict()
    
    # Handle automatic collapse calculation
    collapse_value = config.collapse
    if collapse_value == "auto":
        head_dim = getattr(model.config, 'head_dim', model.config.hidden_size // model.config.num_attention_heads)
        model.config.head_dim = head_dim
        collapse_value = head_dim // config.qk_mqa_dim
        print(f"[TransMLA] Auto collapse: {collapse_value}")
    
    config_dict["collapse"] = int(collapse_value)
    
    model = _transmla_functions.partial_rope(model, tokenizer, train_loader, test_loader, **config_dict)
    
    # Process partial_rope result
    freqfold_value = config.freqfold
    if isinstance(model, tuple):
        if freqfold_value == "auto":
            freqfold_value = model[1]
            print(f"[TransMLA] Auto freqfold: {freqfold_value}")
        model = model[0]
    
    config_dict["freqfold"] = freqfold_value
    
    # Stage 2: Low-rank QKV
    model = _transmla_functions.low_rank_qkv(model, tokenizer, train_loader, test_loader, **config_dict)
    
    print("[TransMLA] MLA conversion completed successfully!")
    return model


class TransMLA(Reassembler):
    """Specialized TransMLA reassembler."""

    @classmethod
    def reassemble(cls, model: nn.Module, tokenizer, config: Optional[TransMLAConfig] = None,
                          additional_mapping: dict = None, **kwargs):
        """TransMLA conversion - simple implementation."""
        _initialize_transmla()  # Ensure TransMLA is initialized
        assert TRANSMLA_AVAILABLE, f"TransMLA not available: {TRANSMLA_ERROR}"
        
        # Apply mappings
        if additional_mapping:
            cls._apply_additional_mapping(model, additional_mapping)
        
        # Prepare config
        config = config or TransMLAConfig()
        
        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Resolve auto-values using centralized logic
        config.dtype = TransMLAConfig._resolve_dtype(config.dtype)
        
        # Validate configuration before conversion
        config.validate(model)
        
        # Convert
        model = convert_model_to_mla(model, tokenizer, config)
        cls._validate_device_consistency(model)
        return model

