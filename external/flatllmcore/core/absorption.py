"""
Calibration-based absorption using a documented Nystrom least-squares variant.

This module contains a calibration prototype, with a documented Nyström restoration variant:
1. Collect activations via forward hooks
2. Compute eigenvectors from activation covariance
3. Apply absorption: V_new = V @ Q, O_new = Q^T @ O
4. Physically reduce hidden dimensions
5. Rebuild model architecture with new dimensions
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from typing import Dict, List, Tuple, Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
        if not 0 < tolerance <= 1:
            raise ValueError("tolerance must be in (0, 1]")
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

    def compute_eigenvectors(self, is_attention=False):
        if not self.n_samples:
            raise ValueError('No calibration activations collected')
        covariance = self.cov_heads if is_attention else self.cov
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        eigenvalues = eigenvalues.flip(-1).clamp_min(0)
        eigenvectors = eigenvectors.flip(-1)
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        spectra = eigenvalues.unbind(0) if is_attention else (eigenvalues,)
        ranks = []
        for spectrum in spectra:
            total = spectrum.sum()
            rank = 1 if total == 0 else int(torch.searchsorted(spectrum.cumsum(0), total*self.tolerance)) + 1
            ranks.append(min(rank, len(spectrum)))
        self.selected_dim = torch.tensor(ranks) if is_attention else ranks[0]
        return self.selected_dim


class AbsorptionCompressor:
    """
    Applies FLAT-LLM absorption mechanism to compress model.
    """

    def __init__(
        self,
        model: "PreTrainedModel",
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
        if not 0 < tolerance <= 1:
            raise ValueError("tolerance must be finite and in (0, 1]")
        self.model = deepcopy(model)
        self.target_sparsity = target_sparsity
        self.tolerance = tolerance
        self.device = device

        self.collectors = {}
        self.hooks = []
        self.compression_results = []

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
        self.model.to(self.device)
        config = self.model.config
        heads = config.num_attention_heads
        kv_heads = getattr(config, 'num_key_value_heads', heads)
        if heads % kv_heads:
            raise ValueError('Attention heads must be divisible by KV heads')
        self.collectors = {}
        self.hooks = []
        for layer_idx in layer_indices:
            layer = self.model.model.layers[layer_idx]
            mlp_col = ActivationCollector(layer.mlp.up_proj, 1, self.tolerance, self.device)
            attn_col = ActivationCollector(layer.self_attn.v_proj, kv_heads, self.tolerance, self.device)
            self.collectors[f'layer_{layer_idx}.mlp.up_proj'] = mlp_col
            self.collectors[f'layer_{layer_idx}.self_attn.v_proj'] = attn_col
            def mlp_hook(module, inputs, coll=mlp_col):
                coll.add_activation(inputs[0].detach(), is_attention=False)
            def attn_hook(module, inputs, coll=attn_col):
                values = inputs[0].detach()
                dims = values.shape[:-1]
                group_size = heads // kv_heads
                values = values.reshape(-1, kv_heads, group_size, coll.head_dim)
                values = values.permute(0,2,1,3).reshape(1,-1,kv_heads,coll.head_dim)
                coll.add_activation(values, is_attention=True)
            self.hooks.append(layer.mlp.down_proj.register_forward_pre_hook(mlp_hook))
            self.hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(attn_hook))
        training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for i in range(calibration_input_ids.shape[0]):
                    sample = calibration_input_ids[i:i+1].to(self.device)
                    mask = torch.ones_like(sample) if attention_mask is None else attention_mask[i:i+1].to(self.device)
                    self.model(input_ids=sample, attention_mask=mask)
        finally:
            for handle in self.hooks:
                handle.remove()
            self.hooks = []
            self.model.train(training)
        for name, collector in self.collectors.items():
            collector.compute_eigenvectors(is_attention='v_proj' in name)

    def apply_absorption_mlp(
        self,
        layer_idx: int,
        sparsity_ratio: float
    ):
        """
        Apply absorption to MLP layers with physical dimension reduction.

        Uses the named pseudoinverse least-squares Nystrom variant.
        """
        layer = self.model.model.layers[layer_idx]
        collector = self.collectors.get(f'layer_{layer_idx}.mlp.up_proj')
        if collector is None or not collector.n_samples:
            raise ValueError('MLP calibration is required')
        cov = collector.cov.double()
        d = cov.shape[-1]
        if not 0 < sparsity_ratio <= 1:
            raise ValueError('sparsity_ratio is the retained fraction in (0, 1]')
        k = max(1, int(sparsity_ratio*d))
        # Measured ridge leverage scores; pseudoinverse handles dependent channels.
        ridge = torch.eye(d, device=cov.device, dtype=cov.dtype)
        scores = (cov @ torch.linalg.solve(cov + ridge, ridge)).diagonal()
        idx = torch.argsort(scores, descending=True, stable=True)[:k].sort().values
        selected = cov[idx][:,idx]
        inverse = torch.linalg.pinv(selected, hermitian=True)
        reconstruction = cov[:,idx] @ inverse @ cov[idx,:]
        total = cov.trace()
        retained = 1.0 if total == 0 else float((reconstruction.trace()/total).clamp(0,1))
        if retained + 1e-10 < self.tolerance:
            raise ValueError(f'Unattainable MLP tolerance {self.tolerance} with retained rank {k}: {retained}')
        down = layer.mlp.down_proj
        # PyTorch row-weight convention: W_down C[:,I] pinv(C[I,I]).
        new_down = down.weight.detach().to(cov) @ cov[:,idx] @ inverse
        new_up = layer.mlp.up_proj.weight.detach()[idx.to(layer.mlp.up_proj.weight.device)].clone()
        new_gate = layer.mlp.gate_proj.weight.detach()[idx.to(layer.mlp.gate_proj.weight.device)].clone()
        down.weight = nn.Parameter(new_down.to(down.weight), requires_grad=down.weight.requires_grad)
        down.in_features = k
        for projection, weight in ((layer.mlp.up_proj,new_up),(layer.mlp.gate_proj,new_gate)):
            projection.weight = nn.Parameter(weight, requires_grad=projection.weight.requires_grad)
            if projection.bias is not None:
                projection.bias = nn.Parameter(projection.bias.detach()[idx.to(projection.bias.device)].clone(), requires_grad=projection.bias.requires_grad)
            projection.out_features = k
        if hasattr(layer.mlp,'intermediate_size'):
            layer.mlp.intermediate_size = k
        result = {'layer':layer_idx, 'component':'mlp', 'method':'nystrom_least_squares_pseudoinverse',
                  'rank':k,'retained_calibration_energy':retained,'tolerance':self.tolerance,
                  'target_retained_fraction':sparsity_ratio}
        self.compression_results.append(result)
        return result

    def apply_absorption_attention(
        self,
        layer_idx: int,
        sparsity_ratio: float
    ):
        """
        Apply head-wise absorption to attention layers.
        """
        layer = self.model.model.layers[layer_idx]
        collector = self.collectors.get(f'layer_{layer_idx}.self_attn.v_proj')
        if collector is None or collector.selected_dim is None:
            raise ValueError('Attention calibration is required')
        config = self.model.config
        heads = config.num_attention_heads
        kv_heads = getattr(config, 'num_key_value_heads', heads)
        attn = layer.self_attn
        head_dim = attn.v_proj.out_features // kv_heads
        if not 0 < sparsity_ratio <= 1:
            raise ValueError('sparsity_ratio is a retained fraction in (0, 1]')
        k = max(1,int(sparsity_ratio*head_dim))
        if k < int(collector.selected_dim.max()):
            raise ValueError(f'Unattainable attention tolerance at rank {k}; requires {int(collector.selected_dim.max())}')
        q = collector.eigenvectors[...,:k]
        wv = attn.v_proj.weight.detach().to(q).reshape(kv_heads, head_dim, -1)
        wo = attn.o_proj.weight.detach().to(q).reshape(-1, heads, head_dim).transpose(0,1)
        groups = torch.arange(heads, device=q.device) // (heads // kv_heads)
        new_v = torch.bmm(q.transpose(-2,-1),wv).reshape(kv_heads*k,-1)
        new_o = torch.bmm(wo,q[groups]).transpose(0,1).reshape(attn.o_proj.out_features,heads*k)
        if attn.v_proj.bias is not None:
            bias = torch.bmm(q.transpose(-2,-1),attn.v_proj.bias.detach().to(q).reshape(kv_heads,head_dim,1)).reshape(-1)
            attn.v_proj.bias = nn.Parameter(bias.to(attn.v_proj.bias),requires_grad=attn.v_proj.bias.requires_grad)
        attn.v_proj.weight = nn.Parameter(new_v.to(attn.v_proj.weight),requires_grad=attn.v_proj.weight.requires_grad)
        attn.o_proj.weight = nn.Parameter(new_o.to(attn.o_proj.weight),requires_grad=attn.o_proj.weight.requires_grad)
        attn.v_proj.out_features = kv_heads*k
        attn.o_proj.in_features = heads*k
        values = collector.eigenvalues
        energy = values[...,:k].sum(-1)/values.sum(-1).clamp_min(torch.finfo(values.dtype).tiny)
        energy = torch.where(values.sum(-1)>0,energy,torch.ones_like(energy))
        result = {'layer':layer_idx,'component':'attention','method':'headwise_pca_absorption','rank':k,
                  'retained_calibration_energy':energy.tolist(),'tolerance':self.tolerance}
        self.compression_results.append(result)
        return result

    def patch_compressed_layers(self, layer_indices: List[int]):
        """
        Patch forward methods of compressed attention layers to handle dynamic head_dim.

        This is necessary because transformers' LlamaAttention uses hardcoded head_dim
        from config, but after absorption the physical head_dim has changed.

        Args:
            layer_indices: Indices of layers that were compressed
        """
        from fedcore.repository.capabilities import require_supported
        require_supported("flatllm.attention_cache")
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
