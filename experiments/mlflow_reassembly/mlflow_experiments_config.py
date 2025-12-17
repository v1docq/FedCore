"""
Configuration for MLflow experiments with FedCore.

Experimental matrix:
[LowRank (quantile 0.5, one_time)] x [Adam, ULTG] x [FlatLLM, TransMLA] x [sLM, mLM]

This yields 2 x 2 x 2 = 8 experiments.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Callable
import torch
import torch.optim as optim

# Try to import ULTG from tdecomp
try:
    from tdecomp.grad_proj.tensorgrad.prepared_tg import ULTG as _ULTG_Original
    
    # MONKEY PATCH to fix RuntimeError: view size is not compatible
    # After FlatLLM + LowRank tensors can be non-contiguous, while .view() requires contiguous
    # Replace .view(-1) with .reshape(-1) in ALL TensorGradUnstructuredProjector methods
    try:
        from tdecomp.grad_proj.tensorgrad.projectors.tensor_unstructured_sparse_projector import (
            TensorGradUnstructuredProjector
        )
        
        # Patch 1: _build_indices (line ~174)
        _original_build_indices = TensorGradUnstructuredProjector._build_indices
        
        def _patched_build_indices(self, x):
            """Patched _build_indices: reshape instead of view"""
            flat = x.reshape(-1)  # <- ИСПРАВЛЕНИЕ: было x.view(-1)
            k = max(1, int(self.sparse_ratio * flat.numel()))
            if self.sparse_type == 'topk':
                _, indices = torch.topk(flat.abs(), k, sorted=False)
            elif self.sparse_type == 'random':
                indices = torch.randperm(flat.numel(), device=flat.device)[:k]
            else:
                raise ValueError(f"Unknown sparse_type: {self.sparse_type}")
            self._indices = indices
            self._k = k
        
        # Patch 2: project (line ~74)
        _original_project = TensorGradUnstructuredProjector.project
        
        def _patched_project(self, full_grad, step):
            """Patched project: reshape instead of view"""
            if self._indices is None or (self.update_gap_scheduler and self.update_gap_scheduler.should_update(step)):
                self._build_indices(full_grad)
            
            flat = full_grad.reshape(-1)  # <- ИСПРАВЛЕНИЕ: было full_grad.view(-1)
            sparse_flat = flat[self._indices]
            
            if self.scale_by_mask_ratio:
                ratio = self._k / flat.numel()
                sparse_flat = sparse_flat / ratio
            
            return sparse_flat
        
        # Patch 3: project_back (line ~128)
        _original_project_back = TensorGradUnstructuredProjector.project_back
        
        def _patched_project_back(self, sparse_grad, output_buffer=None, alpha=1.0, accumulate=False):
            """Patched project_back: reshape instead of view"""
            if output_buffer is None:
                raise ValueError("output_buffer is required")
            
            flat = output_buffer.reshape(-1)  # <- ИСПРАВЛЕНИЕ: было output_buffer.view(-1)
            
            vals = sparse_grad
            if self.scale_by_mask_ratio:
                ratio = self._k / flat.numel()
                vals = vals * ratio
            
            if not accumulate:
                flat.zero_()
            
            flat.scatter_add_(0, self._indices, vals * alpha)
            return output_buffer
        
        # Apply all patches
        TensorGradUnstructuredProjector._build_indices = _patched_build_indices
        TensorGradUnstructuredProjector.project = _patched_project
        TensorGradUnstructuredProjector.project_back = _patched_project_back
        
        print("✅ Monkey patch applied: TensorGradUnstructuredProjector (3 methods) now use reshape()")
    except Exception as e:
        print(f"⚠️  Warning: Could not apply monkey patch for TensorGradUnstructuredProjector: {e}")
    
    # Create a wrapper for ULTG, subclassing Optimizer
    class ULTG_Wrapper(torch.optim.Optimizer):
        """
        Wrapper for the ULTG optimizer from tdecomp.

        ULTG returns (optimizer, scheduler), but for compatibility with
        the PyTorch API we only need the optimizer.

        In tdecomp 0.2.18 the return_scheduler parameter is not implemented yet,
        so we simply unpack the tuple and take the first element.

        IMPORTANT: setup_optimizer_and_scheduler() in tdecomp does not add
        batch_size/training_samples to param_groups, which causes an error
        in UpdateGapScheduler. We add them manually.
        """
        def __init__(self, model, svd_type=None, rank=128, learning_rate=1e-4, **kwargs):
            # Store batch_size and training_samples to add to param_groups
            batch_size = kwargs.get('batch_size', 1)
            training_samples = kwargs.get('training_samples', 1000)
            
            # FIX: GaLoreProjector expects either None (uses default torch.linalg.svd)
            # or a callable. Strings like 'randomized_svd' are NOT supported by GaLore
            # (only by TensorGradLowRankProjector). Pass None to use default SVD.
            if isinstance(svd_type, str):
                print(f"⚠️  Warning: svd_type='{svd_type}' не поддерживается GaLoreProjector, используем None (torch.linalg.svd)")
                svd_type = None
            
            # ULTG always returns an (optimizer, scheduler) tuple
            # Disable verbose by default
            if 'verbose' not in kwargs:
                kwargs['verbose'] = False
            
            result = _ULTG_Original(
                model=model,
                svd_type=svd_type,
                rank=rank,
                learning_rate=learning_rate,
                **kwargs
            )
            
            # Unpack tuple: keep only optimizer
            if isinstance(result, tuple) and len(result) == 2:
                self._optimizer, self._scheduler = result
            else:
                # Fallback (should not happen)
                self._optimizer = result
                self._scheduler = None
            
            # FIX: Add batch_size and training_samples to param_groups
            # Necessary for UpdateGapScheduler in get_projector()
            for group in self._optimizer.param_groups:
                group['batch_size'] = batch_size
                group['training_samples'] = training_samples
                # Also add epochs and other params if missing
                if 'epochs' not in group:
                    group['epochs'] = kwargs.get('epochs', 1)
                if 'update_proj_gap' not in group:
                    group['update_proj_gap'] = kwargs.get('update_proj_gap', 200)
                if 'update_proj_gap_end' not in group:
                    group['update_proj_gap_end'] = kwargs.get('update_proj_gap_end', 200)
                if 'update_proj_gap_mode' not in group:
                    group['update_proj_gap_mode'] = kwargs.get('update_proj_gap_mode', 'fixed')
                if 'scheduler_T_max' not in group:
                    group['scheduler_T_max'] = kwargs.get('scheduler_T_max', 1)
                if 'optimizer_type' not in group:
                    group['optimizer_type'] = 'tensorgrad'
            
            # Initialize base Optimizer with parameters from _optimizer
            # Needed for isinstance(opt, torch.optim.Optimizer) checks
            super().__init__(self._optimizer.param_groups, self._optimizer.defaults)
        
        def step(self, *args, **kwargs):
            return self._optimizer.step(*args, **kwargs)
        
        def zero_grad(self, *args, **kwargs):
            return self._optimizer.zero_grad(*args, **kwargs)
        
        def state_dict(self):
            return self._optimizer.state_dict()
        
        def load_state_dict(self, state_dict):
            return self._optimizer.load_state_dict(state_dict)
        
        def __getattr__(self, name):
            # Forward all attribute access to the real optimizer
            # Includes state, param_groups, and everything else
            return getattr(self._optimizer, name)
    
    ULTG_AVAILABLE = True
    ULTG = ULTG_Wrapper
except ImportError:
    ULTG_AVAILABLE = False
    ULTG = None
    print("⚠️  WARNING: tdecomp not installed, ULTG optimizer unavailable")
    print("   Install: pip install git+https://github.com/leostre/tensor-decompositions.git")


@dataclass
class ModelConfig:
    """Model configuration for the experiment."""
    name: str
    model_id: str  # HuggingFace model ID
    size_category: str  # 'sLM' (small) or 'mLM' (medium)
    max_length: int = 512
    dtype: str = "float16"


@dataclass
class LowRankConfig:
    """
    LowRank SVD decomposition configuration.

    Parameters:
        strategy: Rank selection strategy ('quantile', 'explained_variance', 'energy')
        threshold: Strategy threshold (0.5 = median for quantile)
        rank_prune_each: Pruning frequency (-1 = one_time at the end of training)
        decomposing_mode: Decomposition mode ('channel' or 'spatial')
        decomposer: Decomposition method ('svd', 'cur', 'rsvd')
    """
    strategy: str = "quantile"
    threshold: float = 0.5  # quantile 0.5 = median
    rank_prune_each: int = -1  # -1 = one_time (applied at the end of training)
    decomposing_mode: str = "channel"
    decomposer: str = "svd"
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    name: str
    optimizer_class: type
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    is_model_based: bool = False  # True for ULTG (accepts model), False for Adam/AdamW (accept params)
    ultg_svd_type: str = None  # Params for ULTG: None (default torch.linalg.svd) or callable
    ultg_rank: int = 128
    ultg_training_samples: int = 500  # For ULTG scheduler
    ultg_batch_size: int = 2  # For ULTG scheduler
    ultg_scale: float = 1.0  # scale parameter for ULTG
    ultg_second_scale: float = 1.0  # second_scale parameter for ULTG
    ultg_second_rank: int = 128  # second_rank parameter for ULTG
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['optimizer_class'] = self.optimizer_class.__name__ if self.optimizer_class else 'ULTG'
        return d


@dataclass
class CompressionConfig:
    """
    Compression method configuration (FlatLLM or TransMLA).

    Parameters:
        method: Method name ('flatllm' or 'transmla')
        config_params: Parameters for the corresponding Config class
    """
    method: str  # 'flatllm' or 'transmla'
    config_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TrainingConfig:
    """Training process configuration."""
    epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    max_grad_norm: float = 1.0
    seed: int = 42
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Full configuration of a single experiment."""
    experiment_name: str
    model: ModelConfig
    lowrank: LowRankConfig
    optimizer: OptimizerConfig
    compression: CompressionConfig
    training: TrainingConfig
    output_dir: str = "./mlflow_experiment_output"
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "fedcore_lowrank_compression"
    
    def to_dict(self) -> Dict:
        """Converts configuration to a dict for logging."""
        return {
            'experiment_name': self.experiment_name,
            'model': asdict(self.model),
            'lowrank': self.lowrank.to_dict(),
            'optimizer': self.optimizer.to_dict(),
            'compression': self.compression.to_dict(),
            'training': self.training.to_dict(),
            'output_dir': self.output_dir,
            'mlflow_tracking_uri': self.mlflow_tracking_uri,
            'mlflow_experiment_name': self.mlflow_experiment_name
        }


# ============================================================================
# Predefined configurations
# ============================================================================

# Models
MODELS = {
    "sLM": ModelConfig(
        name="TinyLlama-1.1B",
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        size_category="sLM",
        max_length=512,
        dtype="bfloat16"
    ),
    "mLM": ModelConfig(
        name="Pythia-1.4B",
        model_id="EleutherAI/pythia-1.4b",
        size_category="mLM",
        max_length=512
    )
}

# Optimizers (Adam and ULTG from tdecomp)
OPTIMIZERS = {
    "Adam": OptimizerConfig(
        name="Adam",
        optimizer_class=optim.Adam,
        learning_rate=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        is_model_based=False
    ),
    "ULTG": OptimizerConfig(
        name="ULTG",
        optimizer_class=ULTG if ULTG_AVAILABLE else None,
        learning_rate=1e-4,
        weight_decay=0.0,  # ULTG does not use weight_decay like Adam
        betas=(0.9, 0.999),
        is_model_based=True,  # ULTG accepts model, not params
        ultg_svd_type=None,  # SVD type: None (default torch.linalg.svd)
        ultg_rank=128,  # Decomposition rank
        ultg_training_samples=500,  # For ULTG scheduler
        ultg_batch_size=2,  # For ULTG scheduler
        ultg_scale=1.0,  # scale parameter for ULTG
        ultg_second_scale=1.0,  # second_scale parameter for ULTG
        ultg_second_rank=128  # second_rank parameter for ULTG
    )
}

# Compression methods
COMPRESSION_METHODS = {
    "FlatLLM": CompressionConfig(
        method="flatllm",
        config_params={
            "target_sparsity": 0.7,
            "tolerance": 0.96,
            "layer_selection": "auto",
            "compression_ratio": 0.3,
            "cal_nsamples": 16,
            "cal_batch_size": 2,
            "cal_max_seqlen": 128,
            "verbose": True
        }
    ),
    "TransMLA": CompressionConfig(
        method="transmla",
        config_params={
            "priority": "balanced",
            "compression": "medium",
            "hardware_budget": "auto",
            "cal_nsamples": 32,
            "cal_batch_size": 2,
            "cal_max_seqlen": 128,
            "ppl_eval_batch_size": 1,
            "seed": 42
        }
    )
}


# ============================================================================
# Experiments combinations generator
# ============================================================================

def generate_experiment_configs() -> List[ExperimentConfig]:
    """
    Generates all 8 experiment combinations:
    [LowRank (quantile 0.5, one_time)] x [Adam, ULTG] x [FlatLLM, TransMLA] x [sLM, mLM]

    Returns:
        List[ExperimentConfig]: List of 8 configurations
    """
    configs = []
    
    # Single LowRank configuration for all experiments
    lowrank_config = LowRankConfig(
        strategy="quantile",
        threshold=0.5,
        rank_prune_each=-1  # one_time mode
    )
    
    # Single Training configuration for all experiments
    training_config = TrainingConfig(
        epochs=3,
        batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=10,
        eval_steps=50
    )
    
    # Generate all combinations
    for model_key, model_config in MODELS.items():
        for opt_key, opt_config in OPTIMIZERS.items():
            for comp_key, comp_config in COMPRESSION_METHODS.items():
                
                # Form a unique experiment name
                exp_name = f"LR_q05_onetime_{opt_key}_{comp_key}_{model_key}"
                
                # Create full configuration
                experiment = ExperimentConfig(
                    experiment_name=exp_name,
                    model=model_config,
                    lowrank=lowrank_config,
                    optimizer=opt_config,
                    compression=comp_config,
                    training=training_config,
                    output_dir=f"./mlflow_experiment_output/{exp_name}",
                    mlflow_experiment_name="fedcore_lowrank_compression"
                )
                
                configs.append(experiment)
    
    return configs


def print_experiment_summary(configs: List[ExperimentConfig]):
    """Prints a summary of all generated experiments."""
    print("=" * 80)
    print("EXPERIMENTAL MATRIX - 8 EXPERIMENTS")
    print("=" * 80)
    print()
    print(f"Total experiments: {len(configs)}")
    print()
    
    # Grouping for readability
    print("Breakdown:")
    print(f"  Models: {list(MODELS.keys())} (2)")
    print(f"  Optimizers: {list(OPTIMIZERS.keys())} (2)")
    print(f"  Compression: {list(COMPRESSION_METHODS.keys())} (2)")
    print(f"  LowRank: quantile 0.5, one_time (1)")
    print()
    print(f"  2 × 2 × 2 = {len(configs)} experiments")
    print()
    print("-" * 80)
    
    for i, config in enumerate(configs, 1):
        print(f"{i:2d}. {config.experiment_name}")
        print(f"    Model: {config.model.name} ({config.model.size_category})")
        print(f"    Optimizer: {config.optimizer.name}")
        print(f"    Compression: {config.compression.method.upper()}")
        print(f"    LowRank: {config.lowrank.strategy} ({config.lowrank.threshold})")
        print()
    
    print("=" * 80)


if __name__ == "__main__":
    # Test configuration generation
    configs = generate_experiment_configs()
    print_experiment_summary(configs)
    
    # Example of accessing a configuration
    print("\nExample configuration of the first experiment:")
    print("-" * 80)
    example = configs[0]
    print(f"Name: {example.experiment_name}")
    print(f"Model: {example.model.model_id}")
    print(f"Output directory: {example.output_dir}")

