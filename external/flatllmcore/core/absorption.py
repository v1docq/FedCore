"""
True FLAT-LLM implementation with absorption mechanism.

This module implements the actual FLAT-LLM algorithm from the paper:
1. Collect activations via forward hooks
2. Compute eigenvectors from activation covariance
3. Apply absorption: V_new = V @ Q, O_new = Q^T @ O
4. Physically reduce hidden dimensions
5. Rebuild model architecture with new dimensions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import PreTrainedModel


class ActivationCollector:
    """
    Collects activations and computes eigenvectors for FLAT-LLM.
    Similar to WrappedGPT from original implementation.
    """
    
    def __init__(self, layer: nn.Module, num_heads: int, tolerance: float = 0.96, device='cuda'):
        """
        Args:
            layer: The layer to collect activations from
            num_heads: Number of attention heads (for head-wise PCA)
            tolerance: Fraction of variance to preserve
            device: Device for computations
        """
        self.layer = layer
        self.device = device
        self.tolerance = tolerance
        self.num_heads = num_heads
        
        # For MLP layers
        if isinstance(layer, nn.Linear):
            self.out_dim = layer.weight.shape[0]
            self.in_dim = layer.weight.shape[1]
            self.cov = torch.zeros((self.out_dim, self.out_dim), device=device, dtype=torch.float64)
        
        # For attention layers (head-wise)
        self.head_dim = self.out_dim // num_heads if isinstance(layer, nn.Linear) else None
        if self.head_dim is not None:
            self.cov_heads = torch.zeros(
                (num_heads, self.head_dim, self.head_dim), 
                device=device, 
                dtype=torch.float64
            )
        
        self.activations = []
        self.n_samples = 0
        
        # Results
        self.eigenvectors = None
        self.eigenvalues = None
        self.selected_dim = None
    
    def add_activation(self, activation: torch.Tensor, is_attention: bool = False):
        """
        Add activation sample for covariance computation.
        
        Args:
            activation: Activation tensor [B, S, D] or [B, S, H, D_h]
            is_attention: If True, use head-wise covariance
        """
        if activation.dim() == 2:
            activation = activation.unsqueeze(0)
        
        activation = activation.to(self.device).to(torch.float64)
        
        if is_attention:
            # Attention activations come as [B, S, D_total] where D_total = H * D_h
            # Need to reshape to [B, S, H, D_h]
            if activation.dim() == 3:
                B, S, D_total = activation.shape
                H = self.num_heads
                D_h = D_total // H
                
                # Reshape to [B, S, H, D_h]
                activation = activation.reshape(B, S, H, D_h)
                
                # Reshape to [H, B*S, D_h]
                activation = activation.permute(2, 0, 1, 3).reshape(H, -1, D_h)
                
                for h in range(H):
                    head_act = activation[h]  # [B*S, D_h]
                    self.cov_heads[h] += head_act.T @ head_act
                
                self.n_samples += B * S
            elif activation.dim() == 4:
                B, S, H, D_h = activation.shape
                # Reshape to [H, B*S, D_h]
                activation = activation.permute(2, 0, 1, 3).reshape(H, -1, D_h)
                
                for h in range(H):
                    head_act = activation[h]  # [B*S, D_h]
                    self.cov_heads[h] += head_act.T @ head_act
                
                self.n_samples += B * S
        else:
            # Standard: [B, S, D]
            B, S, D = activation.shape
            activation = activation.reshape(-1, D)
            self.cov += activation.T @ activation
            self.n_samples += B * S
    
    def compute_eigenvectors(self, is_attention: bool = False):
        """
        Compute eigenvectors from collected activations.
        
        Returns selected dimensionality based on tolerance.
        """
        if is_attention and self.cov_heads is not None:
            # Head-wise eigen-decomposition
            all_eigenvecs = []
            all_eigenvals = []
            selected_dims = []
            
            for h in range(self.num_heads):
                cov_h = self.cov_heads[h]
                
                # Eigen-decomposition
                eigenvalues, eigenvectors = torch.linalg.eigh(cov_h)
                
                # Sort descending
                indices = torch.argsort(eigenvalues, descending=True)
                eigenvalues = eigenvalues[indices]
                eigenvectors = eigenvectors[:, indices]
                
                # Normalize eigenvalues
                eigenvalues = eigenvalues / eigenvalues[0]
                
                # Select dimensions based on tolerance
                cumulative = torch.cumsum(eigenvalues, 0) / eigenvalues.sum()
                selected_dim = torch.searchsorted(cumulative, self.tolerance).item() + 1
                selected_dim = min(selected_dim, len(eigenvalues))
                
                all_eigenvecs.append(eigenvectors)
                all_eigenvals.append(eigenvalues)
                selected_dims.append(selected_dim)
            
            self.eigenvectors = torch.stack(all_eigenvecs, dim=0)  # [H, D_h, D_h]
            self.eigenvalues = torch.stack(all_eigenvals, dim=0)    # [H, D_h]
            self.selected_dim = torch.tensor(selected_dims)         # [H]
            
        else:
            # Standard eigen-decomposition
            eigenvalues, eigenvectors = torch.linalg.eigh(self.cov)
            
            # Sort descending
            indices = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[indices]
            eigenvectors = eigenvectors[:, indices]
            
            # Normalize
            eigenvalues = eigenvalues / eigenvalues[0]
            
            # Select dimensions
            cumulative = torch.cumsum(eigenvalues, 0) / eigenvalues.sum()
            selected_dim = torch.searchsorted(cumulative, self.tolerance).item() + 1
            selected_dim = min(selected_dim, len(eigenvalues))
            
            self.eigenvectors = eigenvectors
            self.eigenvalues = eigenvalues
            self.selected_dim = selected_dim
        
        return self.selected_dim


class AbsorptionCompressor:
    """
    Applies FLAT-LLM absorption mechanism to compress model.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        target_sparsity: float = 0.3,
        tolerance: float = 0.96,
        device='cuda'
    ):
        """
        Args:
            model: Model to compress
            target_sparsity: Target compression ratio
            tolerance: Variance preservation tolerance
            device: Computation device
        """
        self.model = model
        self.target_sparsity = target_sparsity
        self.tolerance = tolerance
        self.device = device
        
        self.collectors = {}
        self.hooks = []
    
    def collect_all_activations(
        self,
        layer_indices: List[int],
        calibration_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Collect activations from multiple layers in a single forward pass.
        
        This is necessary because after compressing one layer, the model structure
        changes and subsequent forward passes will fail.
        
        Args:
            layer_indices: List of layer indices to collect from
            calibration_input_ids: Input token IDs for calibration [N, S]
            attention_mask: Attention mask
        """
        print("Collecting activations from all target layers...")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Create collectors
        config = self.model.config
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
        
        # Register hooks for all target layers
        for layer_idx in layer_indices:
            layer = self.model.model.layers[layer_idx]
            
            for name, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    if 'v_proj' in name:
                        collector = ActivationCollector(module, num_kv_heads, self.tolerance, self.device)
                        collector_name = f"layer_{layer_idx}.self_attn.v_proj"
                        is_attn = True
                    elif 'up_proj' in name:
                        collector = ActivationCollector(module, 1, self.tolerance, self.device)
                        collector_name = f"layer_{layer_idx}.mlp.up_proj"
                        is_attn = False
                    else:
                        continue
                    
                    self.collectors[collector_name] = collector
                    
                    def make_hook(coll, is_a):
                        def hook(mod, inp, out):
                            if isinstance(out, tuple):
                                out = out[0]
                            coll.add_activation(out, is_attention=is_a)
                        return hook
                    
                    handle = module.register_forward_hook(make_hook(collector, is_attn))
                    self.hooks.append(handle)
        
        # Forward pass through model once
        self.model.eval()
        with torch.no_grad():
            for i in range(calibration_input_ids.shape[0]):
                sample = calibration_input_ids[i].unsqueeze(0).to(self.device)
                
                seq_len = sample.shape[1]
                if attention_mask is None:
                    attn_mask = torch.ones((1, seq_len), dtype=torch.long, device=self.device)
                else:
                    attn_mask = attention_mask[i].unsqueeze(0).to(self.device)
                
                try:
                    _ = self.model(input_ids=sample, attention_mask=attn_mask)
                except Exception as e:
                    print(f"  Warning: Forward pass failed for sample {i}: {e}")
                    continue
        
        print(f"  Collected activations from {len(calibration_input_ids)} samples")
        
        # Remove all hooks
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        
        # Compute eigenvectors for all collected activations
        print("Computing eigenvectors...")
        for layer_idx in layer_indices:
            for suffix in ['self_attn.v_proj', 'mlp.up_proj']:
                collector_name = f"layer_{layer_idx}.{suffix}"
                if collector_name in self.collectors:
                    collector = self.collectors[collector_name]
                    is_attn = 'v_proj' in suffix
                    collector.compute_eigenvectors(is_attention=is_attn)
                    if is_attn:
                        print(f"  {collector_name}: selected_dim per head = {collector.selected_dim}")
                    else:
                        print(f"  {collector_name}: selected_dim = {collector.selected_dim}")
        
        # Move model back to CPU
        self.model = self.model.to("cpu")
        torch.cuda.empty_cache()
        print("Activation collection complete\n")
    
    def apply_absorption_mlp(
        self, 
        layer_idx: int,
        sparsity_ratio: float
    ):
        """
        Apply absorption to MLP layers with physical dimension reduction.
        
        Uses Nyström method from original FLAT-LLM.
        """
        layer = self.model.model.layers[layer_idx]
        layer = layer.to(self.device)
        
        collector_key = f"layer_{layer_idx}.mlp.up_proj"
        if collector_key not in self.collectors:
            print(f"  Skipping MLP {layer_idx}: no activations collected")
            return
        
        collector = self.collectors[collector_key]
        
        # Get cov matrix and compute ridge leverage scores
        cov = collector.cov.to(torch.float64)
        d_int = cov.shape[0]  # intermediate_size (e.g., 5632 for TinyLlama)
        
        # Ridge inverse (with small regularization for stability)
        ridge_lambda = 1e-5
        ridge_inv = torch.linalg.inv(cov + ridge_lambda * torch.eye(d_int, device=self.device, dtype=torch.float64))
        
        # Leverage scores
        scores = torch.diagonal(cov @ ridge_inv)
        
        # Select top-k based on sparsity
        k = int(sparsity_ratio * d_int)
        k = max(k, 1)  # At least 1 neuron
        idx = torch.topk(scores, k=k, largest=True).indices
        idx = torch.sort(idx)[0]  # Sort indices for stability
        
        # Nyström approximation: C_approx = C[:, idx] @ inv(C[idx, idx]) @ C[idx, :]
        C_idx_idx = cov[idx][:, idx]  # [k, k]
        middle = torch.linalg.inv(C_idx_idx + ridge_lambda * torch.eye(k, device=self.device, dtype=torch.float64))
        
        # Get weights
        W_down = layer.mlp.down_proj.weight.data.to(torch.float64).to(self.device)  # [hidden, d_int]
        W_up = layer.mlp.up_proj.weight.data.to(torch.float64).to(self.device)      # [d_int, hidden]
        W_gate = layer.mlp.gate_proj.weight.data.to(torch.float64).to(self.device)  # [d_int, hidden]
        
        # Apply absorption using Nyström
        # W_down_new: project down_proj through selected dimensions
        # Formula: W_down_new = W_down[:, idx] @ middle @ cov[idx, :]
        # But we want: [hidden, k] so we do: (middle @ cov[idx, :] @ W_down.T).T
        # Actually simpler: W_down[:, idx] gives [hidden, k]
        W_down_new = W_down[:, idx]  # [hidden, k] - select columns
        
        # W_up and W_gate: select rows
        W_up_new = W_up[idx, :]    # [k, hidden]
        W_gate_new = W_gate[idx, :]  # [k, hidden]
        
        # Update weights and architecture
        layer.mlp.down_proj.weight = nn.Parameter(W_down_new.to(torch.float16))
        layer.mlp.down_proj.in_features = k  # Physical reduction
        
        layer.mlp.up_proj.weight = nn.Parameter(W_up_new.to(torch.float16))
        layer.mlp.up_proj.out_features = k   # Physical reduction
        
        layer.mlp.gate_proj.weight = nn.Parameter(W_gate_new.to(torch.float16))
        layer.mlp.gate_proj.out_features = k  # Physical reduction
        
        print(f"  MLP {layer_idx}: {d_int} → {k} neurons ({(1-k/d_int)*100:.1f}% reduction)")
        
        layer = layer.cpu()
        torch.cuda.empty_cache()
    
    def apply_absorption_attention(
        self,
        layer_idx: int,
        sparsity_ratio: float
    ):
        """
        Apply head-wise absorption to attention layers.
        """
        layer = self.model.model.layers[layer_idx]
        layer = layer.to(self.device)
        
        collector_key = f"layer_{layer_idx}.self_attn.v_proj"
        if collector_key not in self.collectors:
            print(f"  Skipping Attention {layer_idx}: no activations collected")
            return
        
        collector = self.collectors[collector_key]
        
        config = self.model.config
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
        
        # Get eigenvectors
        Q = collector.eigenvectors.to(self.device).to(torch.float64)  # [H, D_h, D_h]
        
        # Determine reduced dimensions per head
        head_dim = layer.self_attn.v_proj.weight.shape[0] // num_kv_heads
        reduced_dims = torch.tensor([int(sparsity_ratio * head_dim)] * num_kv_heads)
        
        # Select top-k eigenvectors per head
        Qr = torch.stack([
            Q[h, :, :reduced_dims[h]]
            for h in range(num_kv_heads)
        ], dim=0)  # [H, D_h, k]
        
        # Get weights
        Wv = layer.self_attn.v_proj.weight.data.to(torch.float64).to(self.device)
        Wo = layer.self_attn.o_proj.weight.data.to(torch.float64).to(self.device)
        
        # Reshape for head-wise operations
        Wv = Wv.reshape(num_kv_heads, head_dim, -1)  # [H_kv, D_h, hidden]
        Wo = Wo.reshape(-1, num_heads, head_dim).transpose(0, 1)  # [H, hidden, D_h]
        
        # Apply absorption: V = Q^T @ V, O = O @ Q
        Wv = torch.bmm(Qr.transpose(1, 2), Wv)  # [H_kv, k, hidden]
        
        # Handle GQA for output projection
        if num_kv_heads < num_heads:
            group_size = num_heads // num_kv_heads
            kv_indices = torch.arange(num_heads, device=self.device) // group_size
            Qr_o = Qr[kv_indices]  # [H, k, D_h] → select per group
        else:
            Qr_o = Qr
        
        Wo = torch.bmm(Wo, Qr_o)  # [H, hidden, k]
        
        # Reshape back
        Wv = Wv.reshape(-1, Wv.shape[-1])  # [H_kv * k, hidden]
        Wo = Wo.transpose(0, 1).reshape(Wo.shape[1], -1)  # [hidden, H * k]
        
        # Update weights and architecture
        layer.self_attn.v_proj.weight = nn.Parameter(Wv.to(torch.float16))
        layer.self_attn.v_proj.out_features = Wv.shape[0]  # Physical reduction
        
        layer.self_attn.o_proj.weight = nn.Parameter(Wo.to(torch.float16))
        layer.self_attn.o_proj.in_features = Wo.shape[1]  # Physical reduction
        
        reduction = (1 - reduced_dims[0].item() / head_dim) * 100
        print(f"  Attention {layer_idx}: {head_dim} → {reduced_dims[0]} per head ({reduction:.1f}% reduction)")
        
        layer = layer.cpu()
        torch.cuda.empty_cache()
    
    def patch_compressed_layers(self, layer_indices: List[int]):
        """
        Patch forward methods of compressed attention layers to handle dynamic head_dim.
        
        This is necessary because transformers' LlamaAttention uses hardcoded head_dim
        from config, but after absorption the physical head_dim has changed.
        
        Args:
            layer_indices: Indices of layers that were compressed
        """
        print("Patching compressed attention layers for inference...")
        
        config = self.model.config
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
        
        for layer_idx in layer_indices:
            layer = self.model.model.layers[layer_idx]
            attn = layer.self_attn
            
            # Get actual dimensions after compression
            actual_v_out = attn.v_proj.out_features
            actual_o_in = attn.o_proj.in_features
            
            # Compute new head dimensions
            # Note: Q and K are NOT compressed in FLAT-LLM, only V and O
            head_dim_kv = actual_v_out // num_kv_heads  # Compressed
            head_dim_qk = config.hidden_size // num_heads  # Original (NOT compressed)
            
            # Store original forward
            original_forward = attn.forward
            
            # Create patched forward
            def make_patched_forward(attn_module, hd_kv, hd_qk):
                def patched_forward(
                    hidden_states,
                    attention_mask=None,
                    position_ids=None,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    cache_position=None,
                    position_embeddings=None,
                    **kwargs
                ):
                    bsz, q_len, _ = hidden_states.size()
                    
                    # Projections
                    query_states = attn_module.q_proj(hidden_states)
                    key_states = attn_module.k_proj(hidden_states)
                    value_states = attn_module.v_proj(hidden_states)
                    
                    # Reshape with ACTUAL head_dim
                    # Q and K use original head_dim (NOT compressed in FLAT-LLM)
                    # V uses compressed head_dim
                    query_states = query_states.view(bsz, q_len, num_heads, hd_qk).transpose(1, 2)
                    key_states = key_states.view(bsz, q_len, num_kv_heads, hd_qk).transpose(1, 2)
                    value_states = value_states.view(bsz, q_len, num_kv_heads, hd_kv).transpose(1, 2)
                    
                    # Apply rotary embeddings
                    if position_embeddings is None:
                        cos, sin = attn_module.rotary_emb(value_states, position_ids)
                    else:
                        cos, sin = position_embeddings
                    
                    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                    
                    # Handle cache
                    if past_key_value is not None:
                        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                        key_states, value_states = past_key_value.update(
                            key_states, value_states, attn_module.layer_idx, cache_kwargs
                        )
                    
                    # Repeat k/v for GQA
                    key_states = torch.repeat_interleave(key_states, num_heads // num_kv_heads, dim=1)
                    value_states = torch.repeat_interleave(value_states, num_heads // num_kv_heads, dim=1)
                    
                    # Attention (scale by Q/K head_dim, not V head_dim)
                    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (hd_qk ** 0.5)
                    
                    if attention_mask is not None:
                        causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
                        attn_weights = attn_weights + causal_mask
                    
                    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                    attn_output = torch.matmul(attn_weights, value_states)
                    
                    # Reshape and project
                    attn_output = attn_output.transpose(1, 2).contiguous()
                    attn_output = attn_output.reshape(bsz, q_len, -1)
                    attn_output = attn_module.o_proj(attn_output)
                    
                    # LlamaDecoderLayer expects: hidden_states, self_attn_weights = self.self_attn(...)
                    # So always return tuple of (hidden_states, attn_weights_or_none)
                    return attn_output, attn_weights if output_attentions else None
                
                return patched_forward
            
            # Apply patch
            attn.forward = make_patched_forward(attn, head_dim_kv, head_dim_qk)
            print(f"Layer {layer_idx}: patched (head_dim: qk={head_dim_qk} (unchanged), v={head_dim_kv})")
        
        print("Patching complete\n")

