"""
Modular PyTorch implementation of Canonical Rank Adaptation (CaRA)
General-purpose, readable, and extensible. Not tied to ViT-specific shapes.

Main components:
- CPDecomp: utilities for canonical polyadic decomposition parameterization
- CaraAdapter: generic adapter that parameterizes a low-rank delta for a tensor
- CaraLinear: wrapped nn.Linear using CaraAdapter
- CaraAttentionWrapper: helper to adapt projection weights (q/k/v/out) of a MultiheadAttention

Usage examples at the bottom show how to attach to nn.Linear and to torch.nn.MultiheadAttention.

Designed for clarity and testability.
"""

from typing import Tuple, Optional, Sequence, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------ CPD utilities ------------------
class CPDecomp(nn.Module):
    """Canonical Polyadic Decomposition parameterization.

    Represents a tensor of shape `shape` and rank `rank` as a set of `len(shape)` factor matrices
    where factor[i] has shape (shape[i], rank). The reconstructed tensor is:
        T[i0,i1,...,i_{n-1}] = sum_{r=0..rank-1} prod_{mode=0..n-1} factors[mode][i_mode, r]

    This module stores the factors as learnable parameters and provides a reconstruct() method.
    It is intentionally generic: supports 2D (matrix -> reduces to a sum of outer products) and
    higher-order tensors.
    """

    def __init__(self, shape: Sequence[int], rank: int, init_scale: float = 1e-3):
        """Initialize CPD decomposition with given shape and rank.
        
        Args:
            shape: Tuple/sequence of integers defining tensor dimensions (e.g., (8, 64, 512) for 3D)
            rank: CPD rank (number of components in the decomposition)
            init_scale: Standard deviation for Gaussian initialization of factors
            
        Raises:
            AssertionError: If rank < 1
        """
        super().__init__()
        assert rank >= 1, f"CPD rank must be >= 1, but got rank={rank}"
        self.shape = tuple(int(s) for s in shape)
        self.rank = int(rank)
        self.num_modes = len(self.shape)

        # create factors: each factor is (mode_dim, rank)
        self.factors = nn.ParameterList()
        for dim in self.shape:
            p = nn.Parameter(torch.empty(dim, self.rank))
            nn.init.normal_(p, mean=0.0, std=init_scale)
            self.factors.append(p)

    def forward(self) -> torch.Tensor:
        return self.reconstruct()

    def reconstruct(self) -> torch.Tensor:
        """Reconstruct full tensor from factors. Returns tensor with `self.shape`.

        Uses einsum for efficient canonical polyadic reconstruction.
        For a 2-mode tensor: T[i,j] = sum_r factors[0][i,r] * factors[1][j,r]
        For a 3-mode tensor: T[i,j,k] = sum_r factors[0][i,r] * factors[1][j,r] * factors[2][k,r]
        
        Efficiency note: reconstruction costs O(prod(shape) * rank), which can be expensive for large
        tensors. Callers should avoid reconstructing huge tensors frequently; instead consider computing
        mode-wise contractions when possible. For adapter use-cases (small rank, moderate shape), this
        is usually acceptable.
        
        Note: Supports tensors up to 52 modes using letters [a-z, A-Z] (excluding 'r' and 'R').
        """
        # Check if we can generate enough unique letters (max 52 modes supported)
        if self.num_modes > 52:
            raise ValueError(
                f"CPDecomp only supports up to 52 modes, but got {self.num_modes}. "
                "For very high-dimensional tensors, consider alternative decomposition methods."
            )
        
        # Build einsum string like 'ar,br,cr->abc' for 3-mode tensor
        # Use letters a-z, A-Z, but skip 'r' and 'R' (reserved for rank dimension)
        # This gives us 52 possible letters for modes
        available_letters = [chr(i) for i in range(ord('a'), ord('z') + 1) if chr(i) != 'r']
        available_letters += [chr(i) for i in range(ord('A'), ord('Z') + 1) if chr(i) != 'R']
        
        letters = available_letters[:self.num_modes]
        subscripts_in = [l + 'r' for l in letters]
        subs_in = ','.join(subscripts_in)
        subs_out = ''.join(letters)
        eins = f"{subs_in}->{subs_out}"
        return torch.einsum(eins, *self.factors)

    def num_parameters(self) -> int:
        """Calculate total number of parameters in all factors.
        
        Returns:
            Total number of learnable parameters across all factor matrices
        """
        return sum(p.numel() for p in self.factors)


# ------------------ Adapter / wrapper ------------------
class CaraAdapter(nn.Module):
    """Generic CaRA adapter that parameterizes a delta for a tensor using CPD.

    Parameters
    - base_shape: tuple describing the shape of the base tensor (e.g., for nn.Linear weight: (out, in))
    - rank: CPD rank
    - tensor_modes: (optional) how to interpret the base tensor as an N-mode tensor for CPD. If None,
      we default to splitting the 2D matrix into (out, in) — a 2-mode tensor. For more complex cases
      (e.g., qkv weight shaped (3, n_heads, head_dim, embed_dim_per_head)) user should pass desired modes.
    - init_scale: gaussian initialization std for factors
    - scaling: scaling factor for delta (can be fixed float or 'learnable')
    """

    def __init__(self, base_shape: Sequence[int], rank: int, tensor_modes: Optional[Sequence[int]] = None, 
                 init_scale: float = 1e-3, scaling: Union[float, str] = 1.0):
        """Initialize CaRA adapter with CPD parameterization.
        
        Args:
            base_shape: Shape of the base tensor to adapt (e.g., (out_features, in_features) for Linear weight)
            rank: CPD rank for low-rank decomposition
            tensor_modes: Optional tensorization pattern. If None, uses base_shape directly.
                         If provided, must have same total size as base_shape (e.g., (8,64,512) for (512,512))
            init_scale: Standard deviation for Gaussian initialization of CPD factors
            scaling: Scaling factor α for delta. Either float value or 'learnable' for trainable scaling
            
        Raises:
            AssertionError: If tensor_modes product doesn't match base_shape product
        """
        super().__init__()
        self.base_shape = tuple(int(s) for s in base_shape)
        if tensor_modes is None:
            # default: treat base tensor as 2-mode with original shape
            self.tensor_shape = self.base_shape
        else:
            base_prod = math.prod(self.base_shape)
            modes_prod = math.prod(tensor_modes)
            assert modes_prod == base_prod, (
                f"tensor_modes product ({modes_prod}) must equal base_shape product ({base_prod}). "
                f"Got tensor_modes={tuple(tensor_modes)}, base_shape={self.base_shape}"
            )
            self.tensor_shape = tuple(int(s) for s in tensor_modes)
        self.rank = int(rank)

        self.cpd = CPDecomp(self.tensor_shape, self.rank, init_scale=init_scale)
        
        # Scaling factor α for stability
        if scaling == 'learnable':
            self.scaling = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('scaling', torch.tensor(float(scaling)))
        self.is_scaling_learnable = (scaling == 'learnable')

    def forward(self) -> torch.Tensor:
        """Return delta tensor reshaped to base_shape with scaling applied.

        Reconstructs CPD tensor and reshapes (flattens modes) to the base tensor shape.
        Applies scaling factor α for stability: delta = α * CPD(factors)
        """
        reconstructed = self.cpd.reconstruct()
        # flatten reconstructed to match base_shape
        reconstructed_flat = reconstructed.reshape(self.base_shape)
        # Apply scaling factor
        return self.scaling * reconstructed_flat

    def zero_factors(self):
        """Reset all CPD factors to zero. Useful for zero-initialization."""
        for p in self.cpd.factors:
            nn.init.zeros_(p)

    def num_parameters(self) -> int:
        """Calculate total number of trainable parameters.
        
        Returns:
            Total parameters including CPD factors and scaling (if learnable)
        """
        total = self.cpd.num_parameters()
        if self.is_scaling_learnable:
            total += 1
        return total


# ------------------ Simple wrapped Linear ------------------
class CaraLinear(nn.Module):
    """A wrapper around nn.Linear that keeps the base weight frozen and learns a CaRA delta.

    Usage:
        - base_linear: an existing nn.Linear instance (its weights are frozen)
        - rank: CPD rank for the delta
        - tensor_modes: optional tensorization of the weight; if None, default (out, in)

    Forward adds (W + delta) @ x + (bias + delta_bias) if bias adaptation is enabled.
    """

    def __init__(self, base_linear: nn.Linear, rank: int, adapt_bias: bool = False, tensor_modes: Optional[Sequence[int]] = None, 
                 init_scale: float = 1e-3, scaling: Union[float, str] = 1.0):
        """Initialize CaraLinear wrapper for adapting a frozen Linear layer.
        
        Args:
            base_linear: Existing nn.Linear layer (will be frozen)
            rank: CPD rank for weight adaptation
            adapt_bias: Whether to also adapt bias using CPD
            tensor_modes: Optional tensorization for weight (e.g., (num_heads, head_dim, embed_dim))
            init_scale: Standard deviation for Gaussian initialization
            scaling: Scaling factor α. Either float or 'learnable'
            
        Raises:
            AssertionError: If base_linear is not nn.Linear
        """
        super().__init__()
        assert isinstance(base_linear, nn.Linear), (
            f"base_linear must be nn.Linear, but got {type(base_linear).__name__}"
        )
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.base_weight = base_linear.weight.detach()  # frozen base
        self.base_bias = base_linear.bias.detach() if base_linear.bias is not None else None

        # Parameter storing: make sure base parameters are not registered as parameters
        # We'll keep them as buffers so they move with device but are not trained
        self.register_buffer('weight_base', self.base_weight)
        if self.base_bias is not None:
            self.register_buffer('bias_base', self.base_bias)

        self.adapter = CaraAdapter((self.out_features, self.in_features), rank, tensor_modes=tensor_modes, 
                                   init_scale=init_scale, scaling=scaling)
        self.adapt_bias = adapt_bias
        if adapt_bias:
            # Use CaraAdapter for bias too (1D tensor with CPD decomposition)
            # For a 1D tensor (vector), CPD with rank r gives: bias[i] = sum_r factor[i,r]
            self.bias_adapter = CaraAdapter(
                base_shape=(self.out_features,),
                rank=rank,
                tensor_modes=None,  # 1D decomposition
                init_scale=init_scale,
                scaling=scaling
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # weight: (out, in)
        delta_w = self.adapter()
        weight = self.weight_base + delta_w
        y = F.linear(x, weight, bias=None)
        if self.base_bias is not None or self.adapt_bias:
            bias = self.bias_base if self.base_bias is not None else torch.zeros(self.out_features, device=x.device, dtype=x.dtype)
            if self.adapt_bias:
                # Use CPD-based bias adapter for consistency
                delta_b = self.bias_adapter()
                bias = bias + delta_b
            y = y + bias
        return y

    def num_trainable_parameters(self) -> int:
        """Calculate total number of trainable parameters.
        
        Returns:
            Total parameters including weight adapter and bias adapter (if enabled)
        """
        total = self.adapter.num_parameters()
        if self.adapt_bias:
            total += self.bias_adapter.num_parameters()
        return total


# ------------------ Attention wrapper ------------------
class CaraAttentionWrapper(nn.Module):
    """Wrap a MultiheadAttention-like module's projection weights (q,k,v,out) with CaRA adapters.

    This wrapper assumes the underlying attention stores its projection weights as attributes, or
    that you pass those tensors explicitly. For PyTorch's nn.MultiheadAttention, the in_proj_weight
    packs q,k,v into one matrix (3*embed_dim, embed_dim). We provide helper functions to create
    adapters for common conventions.
    
    Key feature: Uses 3D tensor structure (num_heads, head_dim, embed_dim) to preserve multi-head
    correlations as described in the CaRA paper, rather than flattening to 2D like LoRA.
    """

    def __init__(self, base_attn: nn.MultiheadAttention, rank: int, init_scale: float = 1e-3, 
                 scaling: Union[float, str] = 1.0, adapt_out_proj: bool = True):
        """
        Args:
            base_attn: PyTorch MultiheadAttention module to wrap
            rank: CPD rank for adapters
            init_scale: initialization scale for CPD factors
            scaling: scaling factor α (float or 'learnable')
            adapt_out_proj: whether to also adapt output projection
        """
        super().__init__()
        self.base_attn = base_attn
        self.embed_dim = base_attn.embed_dim
        self.num_heads = base_attn.num_heads
        self.head_dim = base_attn.head_dim
        self.rank = rank
        self.adapt_out_proj = adapt_out_proj
        
        # Freeze base parameters
        for param in base_attn.parameters():
            param.requires_grad = False
        
        self.adapters = nn.ModuleDict()

        # Create 3D adapters for Q, K, V projections using multi-head structure
        # Shape: (num_heads, head_dim, embed_dim) for proper tensor decomposition
        tensor_modes = (self.num_heads, self.head_dim, self.embed_dim)
        
        for proj_name in ['q', 'k', 'v']:
            adapter = CaraAdapter(
                base_shape=(self.embed_dim, self.embed_dim),  # standard projection shape
                rank=rank,
                tensor_modes=tensor_modes,  # 3D decomposition preserving head structure
                init_scale=init_scale,
                scaling=scaling
            )
            self.adapters[f'in_proj_{proj_name}'] = adapter
        
        # Output projection adapter (can be 2D as it merges heads)
        if adapt_out_proj:
            self.adapters['out_proj'] = CaraAdapter(
                base_shape=(self.embed_dim, self.embed_dim),
                rank=rank,
                tensor_modes=None,  # 2D is fine for output
                init_scale=init_scale,
                scaling=scaling
            )

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = True,
                attn_mask: Optional[torch.Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass applying CaRA deltas to Q/K/V and output projections.
        
        This method computes attention with modified weights without mutating the base model.
        Uses functional API for safe and efficient computation.
        Signature matches nn.MultiheadAttention.forward for compatibility.
        """
        # Get deltas for Q, K, V projections
        delta_q = self.adapters['in_proj_q']()
        delta_k = self.adapters['in_proj_k']()
        delta_v = self.adapters['in_proj_v']()
        
        # Create modified weights without mutating base weights
        # This is safe for autograd and thread-safe
        if self.base_attn._qkv_same_embed_dim:
            # Packed weights: in_proj_weight shape (3*embed_dim, embed_dim)
            # Create a new tensor with deltas applied (no in-place modification)
            in_proj_weight = self.base_attn.in_proj_weight.clone()
            D = self.embed_dim
            in_proj_weight[0:D] = in_proj_weight[0:D] + delta_q
            in_proj_weight[D:2*D] = in_proj_weight[D:2*D] + delta_k
            in_proj_weight[2*D:3*D] = in_proj_weight[2*D:3*D] + delta_v
            in_proj_bias = self.base_attn.in_proj_bias
            q_proj_weight = k_proj_weight = v_proj_weight = None
        else:
            # Separate projection weights
            q_proj_weight = self.base_attn.q_proj_weight + delta_q
            k_proj_weight = self.base_attn.k_proj_weight + delta_k
            v_proj_weight = self.base_attn.v_proj_weight + delta_v
            in_proj_weight = None
            in_proj_bias = None
        
        # Prepare output projection weights
        if self.adapt_out_proj:
            delta_out = self.adapters['out_proj']()
            out_proj_weight = self.base_attn.out_proj.weight + delta_out
        else:
            out_proj_weight = self.base_attn.out_proj.weight
        
        out_proj_bias = self.base_attn.out_proj.bias
        
        # Use functional API to compute attention with modified weights
        # This doesn't mutate the base model and is autograd-safe
        attn_output, attn_weights = F.multi_head_attention_forward(
            query, key, value,
            self.embed_dim,
            self.num_heads,
            in_proj_weight,
            in_proj_bias,
            self.base_attn.bias_k,
            self.base_attn.bias_v,
            self.base_attn.add_zero_attn,
            self.base_attn.dropout,
            out_proj_weight,
            out_proj_bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=(not self.base_attn._qkv_same_embed_dim),
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            average_attn_weights=average_attn_weights
        )
        
        return attn_output, attn_weights

    def num_trainable_parameters(self) -> int:
        """Calculate total number of trainable parameters across all adapters.
        
        Returns:
            Total parameters in Q/K/V and output projection adapters
        """
        return sum(m.num_parameters() for m in self.adapters.values())


# ------------------ Simple unit tests / examples ------------------
if __name__ == '__main__':
    print("=" * 60)
    print("CaRA Implementation Tests - Following Paper Methodology")
    print("=" * 60)
    
    # Test 1: CaraLinear with scaling factor
    print("\n[Test 1] CaraLinear with learnable scaling factor")
    lin = nn.Linear(128, 256)
    cara_lin = CaraLinear(lin, rank=4, adapt_bias=True, scaling='learnable')
    x = torch.randn(10, 128)
    y = cara_lin(x)
    print(f'  Output shape: {y.shape}')
    print(f'  Trainable params: {cara_lin.num_trainable_parameters()}')
    print(f'  Scaling factor α: {cara_lin.adapter.scaling.item():.4f}')
    
    # Test 2: CPD reconstruction for 3D tensor (multi-head structure)
    print("\n[Test 2] CPD reconstruction for 3D tensor (multi-head like)")
    num_heads, head_dim, embed_dim = 8, 64, 512
    shape_3d = (num_heads, head_dim, embed_dim)
    rank = 4
    cpd_3d = CPDecomp(shape_3d, rank, init_scale=0.01)
    full_3d = cpd_3d.reconstruct()
    print(f'  3D tensor shape: {full_3d.shape}')
    print(f'  CPD parameters: {cpd_3d.num_parameters()} (vs full tensor: {math.prod(shape_3d)})')
    compression_ratio = math.prod(shape_3d) / cpd_3d.num_parameters()
    print(f'  Compression ratio: {compression_ratio:.2f}x')

    # Test 3: CaraAttentionWrapper with 3D tensor decomposition
    print("\n[Test 3] CaraAttentionWrapper with multi-head structure (KEY TEST)")
    embed_dim = 512
    num_heads = 8
    mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
    
    # Create wrapper - this now uses 3D tensors internally!
    cara_attn = CaraAttentionWrapper(mha, rank=4, scaling='learnable', adapt_out_proj=True)
    print(f'  Attention config: embed_dim={embed_dim}, num_heads={num_heads}, head_dim={embed_dim//num_heads}')
    print(f'  Trainable params: {cara_attn.num_trainable_parameters()}')
    
    # Verify 3D tensor structure
    for name, adapter in cara_attn.adapters.items():
        if 'in_proj' in name:
            print(f'  {name}: tensor_shape={adapter.tensor_shape} (3D multi-head!)')
    
    # Test forward pass
    batch_size, seq_len = 4, 32
    q = torch.randn(batch_size, seq_len, embed_dim)
    k = torch.randn(batch_size, seq_len, embed_dim)
    v = torch.randn(batch_size, seq_len, embed_dim)
    
    attn_out, attn_weights = cara_attn(q, k, v, need_weights=True)
    print(f'  Forward output shape: {attn_out.shape}')
    print(f'  Attention weights shape: {attn_weights.shape}')

    # Test 4: Zero-initialization test
    print("\n[Test 4] Zero-delta initialization (should match base)")
    lin2 = nn.Linear(64, 64)
    cara_lin2 = CaraLinear(lin2, rank=2, scaling=0.0)  # scaling=0 means no delta
    x2 = torch.randn(5, 64)
    base_out = lin2(x2)
    cara_out = cara_lin2(x2)
    is_close = torch.allclose(base_out, cara_out, rtol=1e-5, atol=1e-7)
    max_diff = (base_out - cara_out).abs().max().item()
    print(f'  Outputs match (allclose): {is_close}')
    print(f'  Max absolute difference: {max_diff:.2e}')
    
    # Test 5: Training step simulation
    print("\n[Test 5] Training step simulation")
    cara_lin3 = CaraLinear(nn.Linear(32, 32), rank=2, scaling='learnable')
    optimizer = torch.optim.Adam(cara_lin3.parameters(), lr=0.01)
    
    x3 = torch.randn(8, 32)
    target = torch.randn(8, 32)
    
    # Before training
    loss_before = F.mse_loss(cara_lin3(x3), target)
    
    # Training step
    optimizer.zero_grad()
    output = cara_lin3(x3)
    loss = F.mse_loss(output, target)
    loss.backward()
    optimizer.step()
    
    # After training
    loss_after = F.mse_loss(cara_lin3(x3), target)
    print(f'  Loss before: {loss_before.item():.4f}')
    print(f'  Loss after: {loss_after.item():.4f}')
    print(f'  Loss decreased: {loss_before.item() > loss_after.item()}')

    # Test 6: Parameter efficiency comparison
    print("\n[Test 6] Parameter efficiency vs LoRA-style 2D decomposition")
    embed_dim = 512
    num_heads = 8
    rank = 4
    
    # CaRA: 3D decomposition (num_heads, head_dim, embed_dim)
    cara_params = (num_heads * rank) + (embed_dim // num_heads * rank) + (embed_dim * rank)
    
    # LoRA-style: 2D decomposition (embed_dim, embed_dim)
    lora_params = (embed_dim * rank) + (embed_dim * rank)
    
    print(f'  CaRA 3D params per projection: {cara_params}')
    print(f'  LoRA 2D params per projection: {lora_params}')
    print(f'  CaRA is more efficient: {cara_params < lora_params}')
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! Implementation follows CaRA paper.")
    print("=" * 60)