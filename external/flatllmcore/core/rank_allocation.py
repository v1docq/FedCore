"""
Importance-Preserving Rank Selection (IPRS) algorithm for FLAT-LLM.

This module implements the core IPRS algorithm that adaptively distributes
ranks across decoder layers based on computed importance scores.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt


class ImportancePreservingRankSelector:
    """
    Implements the Importance-Preserving Rank Selection algorithm from FLAT-LLM.
    
    This class handles computation of layer-wise importance scores and
    adaptive rank allocation to minimize performance degradation.
    """
    
    def __init__(self, model_config: dict, target_compression: float = 0.5):
        """
        Initialize IPRS selector.
        
        Args:
            model_config: Dictionary containing model dimensions
            target_compression: Target compression ratio (0.5 = 50% compression)
        """
        self.model_config = model_config
        self.target_compression = target_compression
        self.importance_scores = None
        self.rank_allocation = None
        
    def compute_importance_scores(
        self, 
        model: torch.nn.Module,
        calibration_data: torch.Tensor,
        method: str = "angular"
    ) -> torch.Tensor:
        """
        Compute layer-wise importance scores using specified method.
        
        Args:
            model: Model to analyze
            calibration_data: Calibration dataset
            method: Importance computation method ("angular", "gradient", "activation")
            
        Returns:
            Tensor of importance scores for each layer
        """
        num_layers = len(model.model.layers)
        importance_scores = torch.zeros(num_layers)
        
        if method == "angular":
            importance_scores = self._compute_angular_importance(model, calibration_data)
        elif method == "gradient":
            importance_scores = self._compute_gradient_importance(model, calibration_data)
        elif method == "activation":
            importance_scores = self._compute_activation_importance(model, calibration_data)
        else:
            raise ValueError(f"Unknown importance method: {method}")
        
        # Normalize importance scores
        importance_scores = importance_scores / (4096 * 128)  # As per original implementation
        
        self.importance_scores = importance_scores
        return importance_scores
    
    def _compute_angular_importance(
        self, 
        model: torch.nn.Module, 
        calibration_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute angular-based importance scores.
        
        This method measures the angular distance between input and output
        representations to assess layer importance.
        """
        num_layers = len(model.model.layers)
        importance_scores = torch.zeros(num_layers)
        
        # Placeholder implementation - would need actual angular computation
        # In real implementation, this would:
        # 1. Pass calibration data through each layer
        # 2. Compute angular distance between input/output vectors
        # 3. Aggregate scores across samples
        
        for layer_idx in range(num_layers):
            # Simulate angular importance computation
            # Real implementation would analyze actual layer activations
            importance_scores[layer_idx] = torch.randn(1).abs()
        
        return importance_scores
    
    def _compute_gradient_importance(
        self, 
        model: torch.nn.Module, 
        calibration_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient-based importance scores.
        """
        num_layers = len(model.model.layers)
        importance_scores = torch.zeros(num_layers)
        
        # Placeholder - would compute gradients w.r.t layer outputs
        for layer_idx in range(num_layers):
            importance_scores[layer_idx] = torch.randn(1).abs()
            
        return importance_scores
    
    def _compute_activation_importance(
        self, 
        model: torch.nn.Module, 
        calibration_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute activation-based importance scores.
        """
        num_layers = len(model.model.layers)
        importance_scores = torch.zeros(num_layers)
        
        # Placeholder - would analyze activation magnitudes and distributions
        for layer_idx in range(num_layers):
            importance_scores[layer_idx] = torch.randn(1).abs()
            
        return importance_scores
    
    def allocate_ranks(
        self, 
        importance_scores: Optional[torch.Tensor] = None,
        target_compression: Optional[float] = None
    ) -> torch.Tensor:
        """
        Allocate ranks across layers using IPRS algorithm.
        
        Args:
            importance_scores: Layer importance scores (uses stored if None)
            target_compression: Target compression ratio (uses stored if None)
            
        Returns:
            Tensor of rank allocation ratios for each layer
        """
        if importance_scores is None:
            importance_scores = self.importance_scores
        if target_compression is None:
            target_compression = self.target_compression
        if importance_scores is None:
            raise ValueError("No importance scores available. Run compute_importance_scores first.")
        
        # Compute remaining ratio for V,O,MLP layers (Q,K not pruned)
        remaining_ratio = self._compute_remaining_ratio(target_compression)
        
        # Apply proportional allocation with capacity constraints
        num_layers = len(importance_scores)
        rank_ratios = self._proportional_allocation_with_cap(
            importance_scores, 
            remaining_ratio * num_layers
        )
        
        self.rank_allocation = rank_ratios
        return rank_ratios
    
    def _compute_remaining_ratio(self, target_compression: float) -> float:
        """
        Compute the remaining parameter ratio for V,O,MLP layers.
        
        Since Q,K layers are not pruned in FLAT-LLM, we need to adjust
        the compression target for the remaining layers.
        """
        # Get model dimensions
        dq = self.model_config.get('dq', 4096 * 4096)
        dk = self.model_config.get('dk', 4096 * 4096) 
        dv = self.model_config.get('dv', 4096 * 4096)
        do = self.model_config.get('do', 4096 * 4096)
        dmlp = self.model_config.get('dmlp', 4096 * 11008)
        
        # Total parameters including all layers
        total_params = dq + dk + dv + do + dmlp * 3  # 3 MLP layers per decoder
        
        # Parameters that will be pruned (V, O, MLP)
        prunable_params = dv + do + dmlp * 3
        
        # Parameters that won't be pruned (Q, K)
        fixed_params = dq + dk
        
        # Calculate remaining ratio for prunable parameters
        target_remaining = (1 - target_compression) * total_params
        remaining_for_prunable = target_remaining - fixed_params
        
        remaining_ratio = remaining_for_prunable / prunable_params
        
        return max(0.0, min(1.0, remaining_ratio))
    
    def _proportional_allocation_with_cap(
        self, 
        importance_scores: torch.Tensor, 
        total_capacity: float
    ) -> torch.Tensor:
        """
        Proportional allocation with capacity constraints.
        
        This implements the core IPRS algorithm that distributes ranks
        proportionally to importance while respecting capacity limits.
        """
        n_layers = len(importance_scores)
        allocation = torch.zeros_like(importance_scores)
        remaining_indices = torch.arange(n_layers)
        
        while len(remaining_indices) > 0:
            # Get scores for remaining layers
            remaining_scores = importance_scores[remaining_indices]
            
            # Proportional allocation
            if remaining_scores.sum() > 0:
                proportional_alloc = remaining_scores / remaining_scores.sum() * total_capacity
            else:
                proportional_alloc = torch.ones_like(remaining_scores) * (total_capacity / len(remaining_scores))
            
            # Check capacity constraints (max ratio = 1.0)
            over_capacity = proportional_alloc > 1.0
            
            if not over_capacity.any():
                # All allocations fit within capacity
                allocation[remaining_indices] = proportional_alloc
                break
            
            # Fix over-capacity allocations at 1.0
            capped_indices = remaining_indices[over_capacity]
            allocation[capped_indices] = 1.0
            
            # Reduce total capacity and continue with remaining layers
            total_capacity -= over_capacity.sum().float()
            remaining_indices = remaining_indices[~over_capacity]
            
            if total_capacity <= 0:
                break
        
        return allocation
    
    def visualize_allocation(
        self, 
        importance_scores: Optional[torch.Tensor] = None,
        rank_allocation: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize importance scores and rank allocation.
        
        Args:
            importance_scores: Importance scores to plot
            rank_allocation: Rank allocation to plot
            save_path: Path to save the plot (optional)
        """
        if importance_scores is None:
            importance_scores = self.importance_scores
        if rank_allocation is None:
            rank_allocation = self.rank_allocation
            
        if importance_scores is None or rank_allocation is None:
            raise ValueError("No data available for visualization")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        layer_indices = range(len(importance_scores))
        
        # Plot importance scores
        ax.plot(layer_indices, importance_scores.numpy(), 
               label='Importance Scores', color='gray', linestyle='--', marker='o')
        
        # Plot rank allocation
        ax.plot(layer_indices, rank_allocation.numpy(),
               label='Rank Allocation', color='blue', marker='s')
        
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Score / Allocation Ratio')
        ax.set_title('FLAT-LLM Importance Scores and Rank Allocation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_layer_compression_ratios(self) -> Dict[str, torch.Tensor]:
        """
        Get compression ratios for different layer types.
        
        Returns:
            Dictionary mapping layer types to compression ratios
        """
        if self.rank_allocation is None:
            raise ValueError("No rank allocation available. Run allocate_ranks first.")
        
        return {
            'attention_v': self.rank_allocation,
            'attention_o': self.rank_allocation, 
            'mlp_up': self.rank_allocation,
            'mlp_gate': self.rank_allocation,
            'mlp_down': self.rank_allocation,
            'attention_q': torch.ones_like(self.rank_allocation),  # Q not pruned
            'attention_k': torch.ones_like(self.rank_allocation)   # K not pruned
        }
    
    def save_allocation(self, path: str):
        """Save rank allocation to file."""
        if self.rank_allocation is None:
            raise ValueError("No allocation to save")
        torch.save(self.rank_allocation, path)
    
    def load_allocation(self, path: str):
        """Load rank allocation from file."""
        self.rank_allocation = torch.load(path)
        return self.rank_allocation


def create_model_specific_selector(model_name: str, target_compression: float = 0.5) -> ImportancePreservingRankSelector:
    """
    Create IPRS selector with model-specific configurations.
    
    Args:
        model_name: Name of the model ("llama-2-7b", "mistral-7b", etc.)
        target_compression: Target compression ratio
        
    Returns:
        Configured IPRS selector
    """
    # Model-specific configurations
    configs = {
        "llama-2-7b": {
            'dq': 4096 * 4096,
            'dk': 4096 * 4096,
            'dv': 4096 * 4096,
            'do': 4096 * 4096,
            'dmlp': 4096 * 11008,
            'num_layers': 32
        },
        "llama-2-13b": {
            'dq': 5120 * 5120,
            'dk': 5120 * 5120,
            'dv': 5120 * 5120,
            'do': 5120 * 5120,
            'dmlp': 5120 * 13824,
            'num_layers': 40
        },
        "mistral-7b": {
            'dq': 4096 * 4096,
            'dk': 4096 * 1024,  # Grouped-query attention
            'dv': 4096 * 1024,
            'do': 4096 * 4096,
            'dmlp': 4096 * 14336,
            'num_layers': 32
        }
    }
    
    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(configs.keys())}")
    
    return ImportancePreservingRankSelector(configs[model_name], target_compression)
