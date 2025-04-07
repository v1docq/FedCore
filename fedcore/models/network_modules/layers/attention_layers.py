from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.torch_core import Module
from torch import Tensor

from fedcore.architecture.settings.computational import backend_methods as np
from fedcore.architecture.utils.misc import default_value
from fedcore.models.network_modules.other import init_tensor


class ScaledDotProductAttention(Module):
    """Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual
    attention from previous layer (Realformer: Transformer likes residual attention by He et al, 2020) and locality
    self attention (Vision Transformer for Small-Size Datasets by Lee et al, 2021)

    """

    def __init__(
        self, d_model, n_heads, attn_dropout=0.0, res_attention=False, lsa=False
    ):
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim**-0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        """
        Method for forward pass of scaled dot-product attention.

        Args:
            q: [bs x n_heads x max_q_len x d_k]
            k: [bs x n_heads x d_k x seq_len]
            v: [bs x n_heads x seq_len x d_v]
            prev: [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask: [1 x seq_len x seq_len]

        Returns:
            output: [bs x n_heads x max_q_len x d_v]
            attn: [bs x n_heads x max_q_len x seq_len]
            scores: [bs x n_heads x max_q_len x seq_len]

        """

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        # attn_scores : [bs x n_heads x max_q_len x q_len]
        attn_scores = torch.matmul(q, k) * self.scale

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None:
            attn_scores = attn_scores + prev

        # Attention mask (optional)
        # attn_mask with shape [q_len x seq_len] - only used when q_len ==
        # seq_len
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        # mask with shape [bs x q_len] (only when max_w_len == q_len)
        if key_padding_mask is not None:
            attn_scores.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf
            )

        # normalize the attention weights
        # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        # output: [bs x n_heads x max_q_len x d_v]
        output = torch.matmul(attn_weights, v)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class MultiHeadAttention(Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        res_attention=False,
        attn_dropout=0.0,
        proj_dropout=0.0,
        qkv_bias=True,
        lsa=False,
    ):
        """Multi Head Attention Layer

        Args:
            d_model: model dimensionality
            n_heads: number of heads
            d_k: dimensionality of K and Q
            d_v: dimensionality of V
            res_attention: whether to use residual attention from previous layer
            attn_dropout: dropout for attention weights
            proj_dropout: dropout for output
            qkv_bias: whether to use bias for q, k, v projections
            lsa: whether to use learnable scale for attention scores

        """

        if d_k is None:
            d_k = d_model // n_heads
        if d_v is None:
            d_v = d_model // n_heads

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = ScaledDotProductAttention(
            d_model,
            n_heads,
            attn_dropout=attn_dropout,
            res_attention=self.res_attention,
            lsa=lsa,
        )

        # Poject output
        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout)
        )

    def forward(
        self,
        Q: Tensor,
        K: Optional[Tensor] = None,
        V: Optional[Tensor] = None,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):

        bs = Q.size(0)
        if K is None:
            K = Q
        if V is None:
            V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        # v_s    : [bs x n_heads x q_len x d_v]
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(
                q_s,
                k_s,
                v_s,
                prev=prev,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            output, attn_weights = self.sdp_attn(
                q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        # output: [bs x n_heads x q_len x d_v],
        # attn: [bs x n_heads x q_len x q_len],
        # scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        # output = (
        #     output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        # )
        output = output.view(bs, -1, self.n_heads * self.d_v).transpose(1, 2).contiguous()
        # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class LinformerSelfAttention(nn.Module):
    """Self-attention implementation from Linformer.

    Args:
        dim (int): Dimensionality of input embeddings.
        seq_len (int): Maximum sequence length.
        k (int): Projection dimension of keys/values. Default: 256.
        n_heads (int): Number of attention heads. Default: 8.
        dim_head (int): Dimensionality of each head. Default: dim // n_heads.
        one_kv_head (bool): Use one head for keys/values. Default: False.
        share_kv (bool): Share projections of keys and values. Default: False.
        dropout (float): Dropout probability. Default: 0.
    """

    def __init__(
            self,
            dim: int,
            seq_len: int,
            k: int = 256,
            n_heads: int = 8,
            dim_head: Optional[int] = None,
            one_kv_head: bool = False,
            share_kv: bool = False,
            dropout: float = 0.
    ):
        super().__init__()
        assert (dim % n_heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = n_heads

        dim_head = default_value(dim_head, dim // n_heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * n_heads, bias=False)

        kv_dim = dim_head if one_kv_head else (dim_head * n_heads)
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.proj_k = nn.Parameter(init_tensor(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias=False)
            self.proj_v = nn.Parameter(init_tensor(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * n_heads, dim)

    def forward(self, x, context=None, **kwargs):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        kv_len = n if context is None else context.shape[1]
        assert kv_len <= self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.to_q(x)

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        kv_input = x if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # allow for variable sequence lengths (less than maximum sequence length) by slicing projections

        if kv_len < self.seq_len:
            kv_projs = map(lambda t: t[:kv_len], kv_projs)

        # project keys and values along the sequence length dimension to k

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention
        keys_transposed = keys.transpose(-1, -2)  # bhkd → bhdk
        dots = torch.matmul(queries, keys_transposed)  # [bhnd] @ [bhdk] → [bhnk]
        dots *= d_h ** -0.5
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)
