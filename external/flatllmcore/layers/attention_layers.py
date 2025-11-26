"""
Custom attention layers optimized for FLAT-LLM transformation.

This module contains modified attention layers that support fine-grained
low-rank transformations and head-wise PCA operations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

# Transformers compatibility: try SDPA classes first, fall back to generic classes.
try:
    # newer or older installs that still expose SDPA names
    from transformers.models.llama.modeling_llama import (
        LlamaSdpaAttention, LlamaConfig, LlamaDecoderLayer,
        apply_rotary_pos_emb, repeat_kv
    )
except Exception:
    # fallback mapping: if SDPA variant disappeared, use the generic LlamaAttention
    from transformers.models.llama.modeling_llama import (
        LlamaAttention as LlamaSdpaAttention,
        LlamaConfig, LlamaDecoderLayer,
        apply_rotary_pos_emb, repeat_kv
    )

# Mistral compatibility (same approach)
try:
    from transformers.models.mistral.modeling_mistral import (
        MistralSdpaAttention, MistralConfig, MistralDecoderLayer
    )
except Exception:
    from transformers.models.mistral.modeling_mistral import (
        MistralAttention as MistralSdpaAttention,
        MistralConfig, MistralDecoderLayer
    )

from transformers.cache_utils import Cache
from transformers.utils import logging

logger = logging.get_logger(__name__)

# Helper to call rotary embedding in a version-tolerant way.
# Some transformers versions expect (value_states, position_ids) -> (cos, sin)
# others expect (value_states, seq_len=...) -> (cos, sin). Wrap both.
def _rotary_cos_sin(rotary_emb, value_states, position_ids=None, seq_len=None):
    try:
        # preferred: (value_states, position_ids=...) or (value_states, seq_len=...)
        if position_ids is not None:
            return rotary_emb(value_states, position_ids)
        if seq_len is not None:
            # some rotary implementations accept seq_len keyword
            return rotary_emb(value_states, seq_len=seq_len)
        return rotary_emb(value_states)
    except TypeError:
        # fallback: try positional seq_len
        if seq_len is not None:
            return rotary_emb(value_states, seq_len)
        # last resort: call without extras
        return rotary_emb(value_states)


class FlatLlamaAttention(LlamaSdpaAttention):
    """
    FLAT-LLM optimized Llama attention with support for head-wise transformations.
    
    This attention layer supports fine-grained low-rank activation space 
    transformations applied at the head level for efficient compression.
    """
    
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config=config, layer_idx=layer_idx)
        
        # Store original dimensions for FLAT-LLM transformations
        self.original_head_dim = self.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        
        # Flags for FLAT-LLM state
        self._flat_transformed = False
        self._head_transforms = None
        
    def set_head_transforms(self, transforms: dict):
        """
        Set head-wise transformation matrices from FLAT-LLM algorithm.
        
        Args:
            transforms: Dictionary containing transformation matrices for each head
        """
        self._head_transforms = transforms
        self._flat_transformed = True
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with optional FLAT-LLM head-wise transformations.
        """
        if output_attentions:
            logger.warning_once(
                "FlatLlamaAttention is using scaled_dot_product_attention, but "
                "`output_attentions=True`. Falling back to manual attention implementation."
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )
        
        bsz, q_len, _ = hidden_states.size()
        head_dim = self.v_proj.weight.shape[0] // self.num_key_value_heads
        
        if position_ids is None:
            position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
        
        # Projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, head_dim).transpose(1, 2)

        # Apply RoPE
        rotary_out = _rotary_cos_sin(self.rotary_emb, value_states, position_ids=position_ids)
        if rotary_out is not None:
            cos, sin = rotary_out
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            # Newer transformers versions handle RoPE internally
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states)

        # Cache handling
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Repeat k,v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Attention mask preparation
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # Ensure contiguous tensors for CUDA
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # Scaled dot-product attention
        is_causal = True if causal_mask is None and q_len > 1 else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.v_proj.weight.shape[0] * self.num_key_value_groups)

        # Apply head-wise transformations if available (FLAT-LLM)
        if self._flat_transformed and self._head_transforms is not None:
            attn_output = self._apply_head_transforms(attn_output, bsz, q_len)

        # Final output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, past_key_value
    
    def _apply_head_transforms(self, attn_output: torch.Tensor, bsz: int, q_len: int) -> torch.Tensor:
        """
        Apply head-wise transformations from FLAT-LLM algorithm.
        
        Args:
            attn_output: Attention output tensor
            bsz: Batch size
            q_len: Sequence length
            
        Returns:
            Transformed attention output
        """
        # This is a placeholder for the actual FLAT-LLM head transformation logic
        # The actual implementation would apply PCA-based transformations per head
        return attn_output


class FlatLlamaDecoderLayer(LlamaDecoderLayer):
    """
    FLAT-LLM optimized Llama decoder layer.
    
    Replaces standard attention with FlatLlamaAttention for efficient compression.
    """
    
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        
        # Replace attention with FLAT-LLM version
        self.self_attn = FlatLlamaAttention(config=config, layer_idx=layer_idx)
        
        # Ensure device consistency
        if hasattr(self, 'input_layernorm'):
            device = self.input_layernorm.weight.device
            self.self_attn = self.self_attn.to(device)


class FlatMistralAttention(MistralSdpaAttention):
    """
    FLAT-LLM optimized Mistral attention with support for head-wise transformations.
    """
    
    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None):
        super().__init__(config=config, layer_idx=layer_idx)
        
        # Store original dimensions
        self.original_head_dim = self.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        
        # FLAT-LLM state
        self._flat_transformed = False
        self._head_transforms = None
        
    def set_head_transforms(self, transforms: dict):
        """Set head-wise transformation matrices from FLAT-LLM algorithm."""
        self._head_transforms = transforms
        self._flat_transformed = True
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Forward pass with FLAT-LLM optimizations."""
        if output_attentions:
            logger.warning_once(
                "FlatMistralAttention does not support output_attentions=True. "
                "Falling back to manual implementation."
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()
        head_dim = self.v_proj.weight.shape[0] // self.num_key_value_heads

        # Projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, head_dim).transpose(1, 2)

        # RoPE
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        rotary_out = _rotary_cos_sin(self.rotary_emb, value_states, seq_len=kv_seq_len)
        if rotary_out is not None:
            cos, sin = rotary_out
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, position_ids=position_ids)

        # Cache handling
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Expand k,v heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Attention computation
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(f"Attention mask should be {(bsz, 1, q_len, kv_seq_len)}, got {attention_mask.size()}")

        # Ensure contiguous for CUDA
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.v_proj.weight.shape[0] * self.num_key_value_groups)
        
        # Apply FLAT-LLM transformations if available
        if self._flat_transformed and self._head_transforms is not None:
            attn_output = self._apply_head_transforms(attn_output, bsz, q_len)
        
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value
    
    def _apply_head_transforms(self, attn_output: torch.Tensor, bsz: int, q_len: int) -> torch.Tensor:
        """Apply head-wise FLAT-LLM transformations."""
        # Placeholder for actual transformation logic
        return attn_output


class FlatMistralDecoderLayer(MistralDecoderLayer):
    """
    FLAT-LLM optimized Mistral decoder layer.
    """
    
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        
        # Replace attention with FLAT-LLM version
        self.self_attn = FlatMistralAttention(config=config, layer_idx=layer_idx)
