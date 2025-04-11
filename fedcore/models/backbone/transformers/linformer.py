from typing import Optional

import torch
import torch.nn as nn
from fedot.core.operations.operation_parameters import OperationParameters
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.models.network_impl.base_nn_model import BaseNeuralForecaster
from fedcore.models.network_modules.layers.attention_layers import LinformerSelfAttention
from fedcore.models.network_modules.layers.sequential_layers import SequentialSequence
from fedcore.models.network_modules.layers.special import PreNorm
from fedcore.models.network_modules.other import FeedForward
from fedcore.models.network_modules.reversible import ReversibleSequence


class Linformer(nn.Module):
    """Implementation of the Linformer model.

    Args:
        dim (int): Dimensionality of input embeddings.
        seq_len (int): Maximum sequence length.
        depth (int): Number of layers.
        k (int, optional): Projection dimension of keys/values. Default: 256.
        n_heads (int, optional): Number of attention heads. Default: 8.
        dim_head (int, optional): Dimensionality of attention head. Default: dim // n_heads.
        one_kv_head (bool, optional): Use one head for keys/values. Default: False.
        share_kv (bool, optional): Share keys and values. Default: False.
        reversible (bool, optional): Use reversibility to save memory. Default: False.
        dropout (float, optional): Dropout probability. Default: 0.
    """

    def __init__(
            self,
            dim: int,
            seq_len: int,
            depth: int,
            k: int = 256,
            n_heads: int = 8,
            dim_head: Optional[int] = None,
            one_kv_head: bool = False,
            share_kv: bool = False,
            reversible: bool = False,
            dropout: float = 0.
    ):
        super().__init__()
        layers = nn.ModuleList()
        for _ in range(depth):
            attn = LinformerSelfAttention(
                dim=dim,
                seq_len=seq_len,
                k=k,
                n_heads=n_heads,
                dim_head=dim_head,
                one_kv_head=one_kv_head,
                share_kv=share_kv,
                dropout=dropout
            )
            ff = FeedForward(dim=dim, dropout=dropout)
            layers.append(nn.ModuleList([
                PreNorm(dim, attn),
                PreNorm(dim, ff)
            ]))

        self.net = ReversibleSequence(layers) if reversible else SequentialSequence(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Model forward pass"""
        return self.net(x)


class LinformerModel(BaseNeuralForecaster):
    """Linformer implementation for forecasting."""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model_params = self._get_model_params()
        self._init_model()

    def _init_model(self):
        """Linformer initialization."""
        input_dim = self._get_input_dim()
        self.model = Linformer(
            dim=input_dim,
            seq_len=self.params.get("seq_len", 64),
            depth=self.params.get("depth", 3),
            n_heads=self.params.get("n_heads", 8),
            reversible=self.params.get("reversible", False),
            dropout=self.params.get("dropout", 0.1)
        ).to(default_device())

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Model forward pass."""
        return self.model(x)

    def out_of_sample_predict(self, tensor_endogen: torch.Tensor, tensor_exogen: torch.Tensor, target: torch.Tensor):
        """Out-of-sample forecasting."""
        if self.use_exog_features:
            x = torch.cat([tensor_endogen, tensor_exogen], dim=-1)
        else:
            x = tensor_endogen

        pred = self.model(x)  # [batch_size, seq_len, horizon]
        return pred[:, -1, :].unsqueeze(1)  # [batch_size, 1, train_horizon]


class LinformerLM(nn.Module):
    """Linformer-based language model.

    Args:
        num_tokens (int): Number of unique tokens in the dictionary.
        dim (int): Embedding dimension.
        seq_len (int): Maximum sequence length.
        depth (int): Number of Linformer layers.
        k (int, optional): Projection dimension of keys/values. Default: 256.
        n_heads (int, optional): Number of attention heads. Default: 8.
        dim_head (int, optional): Dimensionality of attention head. Default: dim // n_heads.
        one_kv_head (bool, optional): Use one head for keys/values. Default: False.
        share_kv (bool, optional): Share keys and values. Default: False.
        reversible (bool, optional): Use reversibility. Default: False.
        dropout (float, optional): Dropout probability. Default: 0.
    """
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        seq_len: int,
        depth: int,
        k: int = 256,
        n_heads: int = 8,
        dim_head: Optional[int] = None,
        one_kv_head: bool = False,
        share_kv: bool = False,
        reversible: bool = False,
        dropout: float = 0.
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.linformer = Linformer(dim, seq_len, depth, k = k, n_heads = n_heads, dim_head = dim_head,
                one_kv_head = one_kv_head, share_kv = share_kv, reversible = reversible, dropout = dropout)
        self.to_logits = nn.Linear(dim, num_tokens)

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input token indices [batch_size, seq_len].

        Returns:
            torch.Tensor: Logits for predicting tokens [batch_size, seq_len, num_tokens].
        """
        x = self.token_emb(x)
        x = self.pos_emb(torch.arange(x.shape[1], device=x.device)) + x
        x = self.linformer(x)
        out = self.to_logits(x)
        return out
