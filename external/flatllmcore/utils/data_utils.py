"""
Data utilities for FLAT-LLM calibration and evaluation.

This module provides functions for preparing calibration data and managing
data flow during FLAT-LLM transformation processes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Any, Iterator
from transformers import PreTrainedTokenizer
from datasets import Dataset


def prepare_calibration_data(
    model: nn.Module, 
    tokenizer: PreTrainedTokenizer,
    dataset_name: str = "wikitext2",
    n_samples: int = 128,
    max_seq_len: int = 4096,
    batch_size: int = 1,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Prepare calibration data for FLAT-LLM algorithm.
    
    Args:
        model: Model to calibrate on
        tokenizer: Tokenizer for the model
        dataset_name: Name of calibration dataset
        n_samples: Number of calibration samples
        max_seq_len: Maximum sequence length
        batch_size: Batch size for processing
        seed: Random seed
        
    Returns:
        Tuple of (inputs, outputs, attention_mask, position_ids)
    """
    # Set random seed
    torch.manual_seed(seed)
    
    # Prepare data structures
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # Initialize input/output tensors
    inputs = torch.zeros((n_samples, max_seq_len, model.config.hidden_size), 
                        dtype=dtype, device='cpu')
    inputs.requires_grad = False
    
    outputs = torch.zeros_like(inputs, device='cpu')
    
    # Cache for attention mask and position ids
    cache = {
        'i': 0, 
        'attention_mask': None, 
        'position_ids': None
    }
    
    # Create input capture module
    class InputCatcher(nn.Module):
        """Module to capture inputs to first transformer layer."""
        
        def __init__(self, original_layer):
            super().__init__()
            self.original_layer = original_layer
            
        def forward(self, hidden_states, **kwargs):
            # Pad sequence if needed
            if hidden_states.shape[1] < max_seq_len:
                pad_size = max_seq_len - hidden_states.shape[1]
                hidden_states = F.pad(hidden_states, (0, 0, 0, pad_size))
            
            # Store input
            inputs[cache['i']] = hidden_states.cpu()
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask')
            cache['position_ids'] = kwargs.get('position_ids')
            
            # Raise exception to stop forward pass
            raise StopIteration("Input captured")
    
    # Get dataset (placeholder - would need actual dataset loading)
    dataloader = _create_dummy_dataloader(tokenizer, n_samples, max_seq_len, batch_size)
    
    # Temporarily replace first layer
    original_first_layer = model.model.layers[0]
    model.model.layers[0] = InputCatcher(original_first_layer)
    
    try:
        # Capture inputs
        for batch_idx, batch in enumerate(dataloader):
            if cache['i'] >= n_samples:
                break
                
            try:
                if isinstance(batch, (tuple, list)):
                    input_ids = batch[0].to(device)
                else:
                    input_ids = batch.to(device)
                    
                model(input_ids)
            except StopIteration:
                # Expected - input was captured
                pass
            except Exception as e:
                print(f"Warning: Error processing batch {batch_idx}: {e}")
                continue
                
    finally:
        # Restore original layer
        model.model.layers[0] = original_first_layer
    
    return inputs, outputs, cache['attention_mask'], cache['position_ids']


def _create_dummy_dataloader(
    tokenizer: PreTrainedTokenizer, 
    n_samples: int, 
    max_seq_len: int, 
    batch_size: int
) -> Iterator[torch.Tensor]:
    """
    Create a dummy dataloader for testing purposes.
    
    In a real implementation, this would load actual dataset like WikiText-2.
    """
    # Generate dummy text sequences
    dummy_texts = [
        "This is a sample text for calibration purposes. " * (max_seq_len // 50)
        for _ in range(n_samples)
    ]
    
    for i in range(0, len(dummy_texts), batch_size):
        batch_texts = dummy_texts[i:i + batch_size]
        
        # Tokenize
        encodings = tokenizer(
            batch_texts,
            max_length=max_seq_len,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        yield encodings.input_ids


def compute_activation_statistics(
    activations: torch.Tensor,
    method: str = "pca"
) -> dict:
    """
    Compute statistics for activation analysis in FLAT-LLM.
    
    Args:
        activations: Activation tensor [batch, seq_len, hidden_dim]
        method: Statistical method to use ("pca", "svd", etc.)
        
    Returns:
        Dictionary containing computed statistics
    """
    batch_size, seq_len, hidden_dim = activations.shape
    
    # Reshape for analysis: [batch * seq_len, hidden_dim]
    flat_activations = activations.view(-1, hidden_dim)
    
    if method.lower() == "pca":
        # Compute covariance matrix
        centered = flat_activations - flat_activations.mean(dim=0, keepdim=True)
        cov_matrix = torch.mm(centered.T, centered) / (flat_activations.shape[0] - 1)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        indices = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]
        
        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'explained_variance_ratio': eigenvalues / eigenvalues.sum(),
            'cumulative_variance_ratio': torch.cumsum(eigenvalues, 0) / eigenvalues.sum()
        }
        
    elif method.lower() == "svd":
        # SVD decomposition
        U, S, Vh = torch.linalg.svd(flat_activations, full_matrices=False)
        
        return {
            'singular_values': S,
            'left_vectors': U,
            'right_vectors': Vh,
            'explained_variance_ratio': (S ** 2) / (S ** 2).sum()
        }
    
    else:
        raise ValueError(f"Unsupported method: {method}")


def estimate_rank_from_tolerance(
    eigenvalues: torch.Tensor,
    tolerance: float = 0.96
) -> int:
    """
    Estimate optimal rank based on explained variance tolerance.
    
    Args:
        eigenvalues: Eigenvalues from PCA/SVD
        tolerance: Minimum fraction of variance to preserve
        
    Returns:
        Estimated rank
    """
    explained_variance = eigenvalues / eigenvalues.sum()
    cumulative_variance = torch.cumsum(explained_variance, 0)
    
    # Find first index where cumulative variance exceeds tolerance
    rank = torch.where(cumulative_variance >= tolerance)[0]
    
    if len(rank) == 0:
        return len(eigenvalues)  # Use full rank if tolerance not met
    
    return rank[0].item() + 1  # +1 because index is 0-based


def create_low_rank_approximation(
    weight_matrix: torch.Tensor,
    rank: int,
    method: str = "svd"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create low-rank approximation of weight matrix.
    
    Args:
        weight_matrix: Original weight matrix [out_features, in_features]
        rank: Target rank for approximation
        method: Decomposition method ("svd", "pca")
        
    Returns:
        Tuple of (left_matrix, right_matrix) such that 
        weight_matrix ≈ left_matrix @ right_matrix
    """
    if method.lower() == "svd":
        U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)
        
        # Keep only top 'rank' components
        U_trunc = U[:, :rank]
        S_trunc = S[:rank]
        Vh_trunc = Vh[:rank, :]
        
        # Create factorization: W ≈ U_trunc @ (S_trunc * Vh_trunc)
        left_matrix = U_trunc
        right_matrix = torch.diag(S_trunc) @ Vh_trunc
        
        return left_matrix, right_matrix
        
    else:
        raise ValueError(f"Unsupported decomposition method: {method}")


def validate_approximation_quality(
    original: torch.Tensor,
    approximation: torch.Tensor,
    metrics: list = ["mse", "frobenius", "spectral"]
) -> dict:
    """
    Validate quality of low-rank approximation.
    
    Args:
        original: Original tensor
        approximation: Approximated tensor  
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of computed metrics
    """
    results = {}
    
    diff = original - approximation
    
    if "mse" in metrics:
        results["mse"] = torch.mean(diff ** 2).item()
        
    if "frobenius" in metrics:
        results["frobenius_norm"] = torch.norm(diff, 'fro').item()
        results["relative_frobenius"] = (torch.norm(diff, 'fro') / torch.norm(original, 'fro')).item()
        
    if "spectral" in metrics:
        results["spectral_norm"] = torch.norm(diff, 2).item()
        results["relative_spectral"] = (torch.norm(diff, 2) / torch.norm(original, 2)).item()
        
    if "max" in metrics:
        results["max_error"] = torch.max(torch.abs(diff)).item()
        
    return results
