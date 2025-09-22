"""
Core FLAT-LLM pruning algorithm implementation.

This module implements the main FLAT-LLM pruning algorithm that applies
fine-grained low-rank transformations with head-wise PCA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..utils.layer_utils import find_layers, get_layer_dimensions
from ..utils.data_utils import prepare_calibration_data, compute_activation_statistics
from .rank_allocation import ImportancePreservingRankSelector


class FlatLLMPruner:
    """
    Main FLAT-LLM pruning algorithm implementation.
    
    Applies fine-grained low-rank activation space transformations with
    importance-preserving rank selection for efficient model compression.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        target_sparsity: float = 0.5,
        tolerance: float = 0.96,
        device: str = "auto"
    ):
        """
        Initialize FLAT-LLM pruner.
        
        Args:
            model: Model to prune
            tokenizer: Tokenizer for the model
            target_sparsity: Target sparsity level (0.5 = 50% compression)
            tolerance: Tolerance threshold for eigenvalue preservation
            device: Device for computations ("auto", "cuda", "cpu")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.target_sparsity = target_sparsity
        self.tolerance = tolerance
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize components
        self.model_type = self._detect_model_type()
        self.dimensions = get_layer_dimensions(model, self.model_type)
        self.rank_selector = ImportancePreservingRankSelector(
            self.dimensions, target_sparsity
        )
        
        # State tracking
        self.importance_scores = None
        self.rank_allocation = None
        self.pruned = False
        
    def _detect_model_type(self) -> str:
        """Detect model type from configuration."""
        model_type = getattr(self.model.config, 'model_type', '').lower()
        
        if 'llama' in model_type:
            return 'llama'
        elif 'mistral' in model_type:
            return 'mistral'
        else:
            # Try to infer from architecture
            if hasattr(self.model.config, 'num_key_value_heads'):
                if self.model.config.num_key_value_heads < self.model.config.num_attention_heads:
                    return 'mistral'  # Likely GQA
            return 'llama'  # Default fallback
    
    def compute_importance_scores(
        self,
        calibration_data: Optional[torch.Tensor] = None,
        n_samples: int = 128,
        dataset_name: str = "wikitext2"
    ) -> torch.Tensor:
        """
        Compute layer-wise importance scores for rank allocation.
        
        Args:
            calibration_data: Pre-prepared calibration data (optional)
            n_samples: Number of calibration samples to use
            dataset_name: Dataset name for calibration
            
        Returns:
            Tensor of importance scores for each layer
        """
        print("[FLAT-LLM] Computing importance scores...")
        
        # Prepare calibration data if not provided
        if calibration_data is None:
            inputs, _, attention_mask, position_ids = prepare_calibration_data(
                self.model, self.tokenizer, dataset_name, n_samples
            )
        else:
            inputs = calibration_data
            attention_mask = None
            position_ids = None
        
        # Compute importance scores using angular method
        self.importance_scores = self.rank_selector.compute_importance_scores(
            self.model, inputs, method="angular"
        )
        
        print(f"[FLAT-LLM] Computed importance scores: {self.importance_scores}")
        return self.importance_scores
    
    def allocate_ranks(
        self, 
        importance_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Allocate ranks across layers using IPRS algorithm.
        
        Args:
            importance_scores: Layer importance scores (computed if None)
            
        Returns:
            Rank allocation ratios for each layer
        """
        if importance_scores is None and self.importance_scores is None:
            raise ValueError("No importance scores available. Run compute_importance_scores first.")
        
        if importance_scores is None:
            importance_scores = self.importance_scores
        
        print("[FLAT-LLM] Allocating ranks using IPRS algorithm...")
        self.rank_allocation = self.rank_selector.allocate_ranks(importance_scores, self.target_sparsity)
        
        print(f"[FLAT-LLM] Rank allocation: {self.rank_allocation}")
        return self.rank_allocation
    
    def prune_model(
        self,
        calibration_data: Optional[torch.Tensor] = None,
        n_samples: int = 128,
        dataset_name: str = "wikitext2"
    ) -> PreTrainedModel:
        """
        Apply FLAT-LLM pruning to the model.
        
        Args:
            calibration_data: Calibration data for pruning
            n_samples: Number of calibration samples
            dataset_name: Dataset for calibration
            
        Returns:
            Pruned model
        """
        if self.pruned:
            print("[FLAT-LLM] Model already pruned!")
            return self.model
        
        print("[FLAT-LLM] Starting FLAT-LLM pruning...")
        
        # Step 1: Compute importance scores if needed
        if self.importance_scores is None:
            self.compute_importance_scores(calibration_data, n_samples, dataset_name)
        
        # Step 2: Allocate ranks if needed  
        if self.rank_allocation is None:
            self.allocate_ranks()
        
        # Step 3: Apply layer-wise pruning
        self._apply_layer_pruning(calibration_data, n_samples, dataset_name)
        
        self.pruned = True
        print("[FLAT-LLM] Pruning completed!")
        
        return self.model
    
    def _apply_layer_pruning(
        self,
        calibration_data: Optional[torch.Tensor] = None,
        n_samples: int = 128,
        dataset_name: str = "wikitext2"
    ):
        """Apply FLAT-LLM pruning to each layer."""
        # Prepare calibration data
        if calibration_data is None:
            inputs, outputs, attention_mask, position_ids = prepare_calibration_data(
                self.model, self.tokenizer, dataset_name, n_samples
            )
        else:
            inputs = calibration_data
            outputs = torch.zeros_like(inputs)
            attention_mask = None
            position_ids = None
        
        # Disable caching during pruning
        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        
        layers = self.model.model.layers
        num_layers = len(layers)
        
        try:
            for layer_idx in range(num_layers):
                print(f"[FLAT-LLM] Processing layer {layer_idx + 1}/{num_layers}")
                
                layer = layers[layer_idx]
                sparsity_ratio = self.rank_allocation[layer_idx].item()
                
                # Move layer to appropriate device
                if hasattr(self.model, 'hf_device_map') and f"model.layers.{layer_idx}" in self.model.hf_device_map:
                    device = self.model.hf_device_map[f"model.layers.{layer_idx}"]
                    device = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
                else:
                    device = self.device
                
                layer = layer.to(device)
                
                # Move data to same device
                if inputs is not None:
                    inputs = inputs.to(device)
                if outputs is not None:
                    outputs = outputs.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                if position_ids is not None:
                    position_ids = position_ids.to(device)
                
                # Apply FLAT-LLM to this layer
                self._prune_single_layer(
                    layer, layer_idx, sparsity_ratio, 
                    inputs, outputs, attention_mask, position_ids
                )
                
                # Update inputs for next layer (forward pass)
                if inputs is not None and outputs is not None:
                    for sample_idx in range(min(n_samples, inputs.shape[0])):
                        with torch.no_grad():
                            sample_input = inputs[sample_idx].unsqueeze(0)
                            layer_output = layer(
                                sample_input,
                                attention_mask=attention_mask,
                                position_ids=position_ids
                            )[0]
                            outputs[sample_idx] = layer_output.squeeze(0)
                    
                    # Swap inputs and outputs for next iteration
                    inputs, outputs = outputs, inputs
                
                # Memory cleanup
                layer = layer.cpu()
                torch.cuda.empty_cache()
                
        finally:
            # Restore model configuration
            self.model.config.use_cache = use_cache
    
    def _prune_single_layer(
        self,
        layer: nn.Module,
        layer_idx: int,
        sparsity_ratio: float,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor]
    ):
        """
        Apply FLAT-LLM pruning to a single layer.
        
        This method applies head-wise PCA transformations to attention and MLP layers.
        """
        # Find linear layers in this layer
        linear_layers = find_layers(layer, layers=[nn.Linear])
        
        for name, module in linear_layers.items():
            if self._should_prune_layer(name):
                print(f"  [FLAT-LLM] Pruning {name} with ratio {sparsity_ratio:.4f}")
                
                # Apply head-wise transformation
                self._apply_head_wise_pruning(
                    module, name, sparsity_ratio, 
                    inputs, attention_mask, position_ids
                )
    
    def _should_prune_layer(self, layer_name: str) -> bool:
        """
        Determine if a layer should be pruned based on FLAT-LLM strategy.
        
        FLAT-LLM only prunes V, O, and MLP layers (not Q, K).
        """
        # Don't prune query and key projections
        if any(x in layer_name.lower() for x in ['q_proj', 'k_proj']):
            return False
        
        # Prune value, output, and MLP projections
        if any(x in layer_name.lower() for x in ['v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj']):
            return True
        
        return False
    
    def _apply_head_wise_pruning(
        self,
        module: nn.Linear,
        layer_name: str,
        sparsity_ratio: float,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor]
    ):
        """
        Apply head-wise PCA pruning to a linear layer.
        
        This is the core FLAT-LLM transformation that applies PCA
        separately to each attention head's subspace.
        """
        weight = module.weight.data
        out_features, in_features = weight.shape
        
        # For attention layers, apply head-wise transformation
        if any(x in layer_name.lower() for x in ['v_proj', 'o_proj']):
            num_heads = self.model.config.num_attention_heads
            head_dim = in_features // num_heads if 'v_proj' in layer_name else out_features // num_heads
            
            self._apply_attention_head_pruning(
                weight, num_heads, head_dim, sparsity_ratio, layer_name
            )
        
        # For MLP layers, apply standard low-rank approximation
        elif any(x in layer_name.lower() for x in ['up_proj', 'down_proj', 'gate_proj']):
            self._apply_mlp_pruning(weight, sparsity_ratio)
        
        # Update the module weight
        module.weight.data = weight
    
    def _apply_attention_head_pruning(
        self,
        weight: torch.Tensor,
        num_heads: int,
        head_dim: int,
        sparsity_ratio: float,
        layer_name: str
    ):
        """Apply head-wise PCA transformation to attention weights."""
        if 'v_proj' in layer_name:
            # Value projection: [hidden_size, hidden_size]
            # Reshape to [num_heads, head_dim, hidden_size] 
            reshaped = weight.view(num_heads, head_dim, -1)
            
            for head_idx in range(num_heads):
                head_weight = reshaped[head_idx]  # [head_dim, hidden_size]
                
                # Apply PCA-based compression to this head
                compressed_head = self._apply_pca_compression(head_weight, sparsity_ratio)
                reshaped[head_idx] = compressed_head
            
            # Reshape back to original form
            weight.copy_(reshaped.view(weight.shape))
            
        elif 'o_proj' in layer_name:
            # Output projection: [hidden_size, hidden_size]
            # Reshape to [hidden_size, num_heads, head_dim]
            reshaped = weight.view(weight.shape[0], num_heads, head_dim)
            
            for head_idx in range(num_heads):
                head_weight = reshaped[:, head_idx, :]  # [hidden_size, head_dim]
                
                # Apply PCA compression
                compressed_head = self._apply_pca_compression(head_weight, sparsity_ratio)
                reshaped[:, head_idx, :] = compressed_head
            
            # Reshape back
            weight.copy_(reshaped.view(weight.shape))
    
    def _apply_mlp_pruning(self, weight: torch.Tensor, sparsity_ratio: float):
        """Apply low-rank approximation to MLP weights."""
        compressed_weight = self._apply_pca_compression(weight, sparsity_ratio)
        weight.copy_(compressed_weight)
    
    def _apply_pca_compression(
        self, 
        weight: torch.Tensor, 
        sparsity_ratio: float
    ) -> torch.Tensor:
        """
        Apply PCA-based compression to weight matrix.
        
        Args:
            weight: Weight matrix to compress
            sparsity_ratio: Target compression ratio
            
        Returns:
            Compressed weight matrix
        """
        # Perform SVD
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        
        # Determine rank based on sparsity ratio and tolerance
        total_variance = (S ** 2).sum()
        cumsum_variance = torch.cumsum(S ** 2, 0)
        
        # Find rank that preserves required variance
        variance_threshold = self.tolerance * total_variance
        rank = torch.where(cumsum_variance >= variance_threshold)[0]
        
        if len(rank) == 0:
            target_rank = len(S)
        else:
            target_rank = min(rank[0].item() + 1, len(S))
        
        # Apply sparsity constraint
        max_rank = int((1 - sparsity_ratio) * min(weight.shape))
        final_rank = min(target_rank, max_rank, len(S))
        
        if final_rank <= 0:
            final_rank = 1
        
        # Reconstruct with reduced rank
        U_trunc = U[:, :final_rank]
        S_trunc = S[:final_rank]
        Vh_trunc = Vh[:final_rank, :]
        
        # Reconstruct approximation
        compressed = U_trunc @ torch.diag(S_trunc) @ Vh_trunc
        
        return compressed
    
    def get_compression_stats(self) -> Dict[str, float]:
        """Get compression statistics for the pruned model."""
        if not self.pruned:
            raise ValueError("Model not pruned yet. Call prune_model() first.")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        non_zero_params = sum((p != 0).sum().item() for p in self.model.parameters())
        
        # Calculate sparsity
        sparsity = 1 - (non_zero_params / total_params)
        compression_ratio = total_params / non_zero_params if non_zero_params > 0 else float('inf')
        
        return {
            'total_params': total_params,
            'non_zero_params': non_zero_params,
            'sparsity': sparsity,
            'compression_ratio': compression_ratio,
            'target_sparsity': self.target_sparsity
        }
    
    def save_pruned_model(self, save_path: str):
        """Save the pruned model."""
        if not self.pruned:
            raise ValueError("Model not pruned yet. Call prune_model() first.")
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save additional FLAT-LLM metadata
        metadata = {
            'target_sparsity': self.target_sparsity,
            'tolerance': self.tolerance,
            'importance_scores': self.importance_scores,
            'rank_allocation': self.rank_allocation,
            'model_type': self.model_type,
            'compression_stats': self.get_compression_stats()
        }
        
        torch.save(metadata, f"{save_path}/flat_llm_metadata.pt")
