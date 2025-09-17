"""
TransMLA reassembly functionality.

Contains TransMLA-specific reassembly logic moved from quantization utils.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Union

import torch.nn as nn

from .core_reassemblers import AttentionReassembler


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
        import transmla_core
        return Path(transmla_core.__file__).parent
    except ImportError:
        pass
    
    # Strategy 3: Fallback to relative path (for development)
    fallback_path = Path(__file__).parent.parent.parent.parent / "external" / "transmla_core"
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


class TransMLAConfig:
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


class TransMLA(AttentionReassembler):
    """Specialized TransMLA reassembler."""

    @classmethod
    def reassemble(cls, model: nn.Module, tokenizer, config: Optional[TransMLAConfig] = None, **kwargs):
        """TransMLA reassembly - direct execution."""
        return cls._convert_trans_mla(
            model=model,
            tokenizer=tokenizer,
            config=config,
            **kwargs
        )

    @classmethod
    def _convert_trans_mla(cls, model: nn.Module, tokenizer, config: Optional[TransMLAConfig] = None,
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
        
        # Convert
        model = convert_model_to_mla(model, tokenizer, config)
        cls._validate_device_consistency(model)
        return model

