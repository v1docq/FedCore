from dataclasses import dataclass
from typing import Optional, Literal, Union, List

@dataclass
class DataConfig:
    batch_size: int = None
    n_train: int = None
    tmp_dir: str = "/tmp/t2t_datagen"  # From tensor2tensor example :cite[4]

@dataclass
class OptimizerConfig:
    # Basic optimizer parameters
    learning_rate: float = 1e-3
    optimizer_type: Literal["tensorgrad", "tensorgrad_sum", "adamw", "lamb", "sgd"] = "tensorgrad"
    n_epochs: int = 100
    scheduler: Literal["cosine", "exponential",
                       "StepLR", "step", "constant", 
                       "ReduceLROnPlateau", "CosineAnnealingLR"] = "cosine"
    gamma: float = 0.1
    scheduler_patience: int = 5
    scheduler_T_max: int = 100
    step_size: int = 30
    
    # TensorGRaD specific parameters :cite[1]:cite[2]
    rank: int = 128
    scale: float = 1.0
    proj_type: Literal["low_rank", "structured_sparse", "unstructured_sparse"] = "low_rank"
    galore_2d_proj_type: Literal["right", "left", "full"] = "left"
    sparse_ratio: float = 0.1  # Ratio of elements to keep in sparse projections
    sparse_type: Literal['topk', 'randK', 'probability'] = "topk"
    scale_by_mask_ratio: bool = True
    reset_sparse_optimizer_states: bool = False
    enforce_full_complex_precision: bool = False
    svd_type: Literal["truncated_svd", "randomized_svd", "full_svd"] = 'truncated_svd'
    
    # Second projector parameters
    second_proj_type: Literal["low_rank", "structured_sparse", "unstructured_sparse"]  = "unstructured_sparse"
    second_sparse_ratio: float = 0.25
    second_sparse_type: Literal['topk', 'randK', 'probability'] = "topk"
    second_scale: float = 1.0
    second_rank: int = 128
    second_scale_by_mask_ratio: bool = False
    
    # Scheduler update gap parameters
    update_proj_gap: int = 100  # Steps between projection updates
    update_proj_gap_end: int = 1000  # Final value for update gap
    update_proj_gap_mode: Literal["linear", "cosine", "step"] = "linear"
    
    # Tucker decomposition parameters :cite[1]
    n_iter_max_tucker: int = 10  # Max iterations for Tucker decomposition
    tucker_warm_restart: bool = True
    
    # Sparsity regularization
    tensorgrad_sum_lambda_sparse: float = 0.05
    
    # Tensor network options :cite[2]
    tensor_network_type: Optional[Literal["mps", "peps", "tree", "mera"]] = None
    tensor_network_chi: Optional[int] = None  # Bond dimension for tensor networks
    
    # Additional flags
    naive_galore: bool = False
    adamw_support_complex: bool = True
    first_dim_rollup: bool = False
    use_checkpoint: bool = False  # For memory optimization :cite[2]
    cuda: Optional[int] = None  # GPU ID if using CUDA :cite[2]

@dataclass
class WandBConfig:
    log_ranks_interval: int = 100
    log_gradients: bool = False
    log_projections: bool = False

@dataclass
class TensorGRaDConfig:
    data: DataConfig
    opt: OptimizerConfig
    # wandb: WandBConfig
    model_type: Literal["symbolic", "neural_network", "tensor_network"] = "neural_network"
    # tensor_diagram_optimization: bool = True  # Enable symbolic tensor optimizations :cite[1]:cite[6]
    # automatic_simplification: bool = True  # Automatically simplify tensor expressions :cite[1]
    # use_fast_jl: bool = False  # Use Fast Johnson-Lindenstrauss transform :cite[1]