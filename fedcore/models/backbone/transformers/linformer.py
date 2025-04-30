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

BASE_LINFORMER_PARAMS = {
    'k': 256,
    'n_heads': 8,
    'dim_head': None,
    'one_kv_head': False,
    'share_kv': False,
    'reversible': False,
    'dropout': 0.1
}

class Linformer(nn.Module):
    """Implementation of the Linformer model.

    Args:
        input_dim (int): Dimensionality of input embeddings.
        output_dim (int): Maximum sequence length.
        depth (int): Number of layers.
        custom_params (dict, optional): Additional parameters. Default: None.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            depth: int = 6,
            custom_params: Optional[dict] = None,
            **kwargs
    ):
        super().__init__()
        params = custom_params.copy() if custom_params is not None else {}
        params.update(kwargs)

        k = params.get('k', 256)
        n_heads = params.get('n_heads', 8)
        dim_head = params.get('dim_head', None)
        one_kv_head = params.get('one_kv_head', False)
        share_kv = params.get('share_kv', False)
        reversible = params.get('reversible', False)
        dropout = params.get('dropout', 0.)

        layers = nn.ModuleList()
        for _ in range(depth):
            attn = LinformerSelfAttention(
                dim=input_dim,
                seq_len=output_dim,
                k=k,
                n_heads=n_heads,
                dim_head=dim_head,
                one_kv_head=one_kv_head,
                share_kv=share_kv,
                dropout=dropout
            )
            ff = FeedForward(dim=input_dim, dropout=dropout)
            layers.append(nn.ModuleList([
                PreNorm(input_dim, attn),
                PreNorm(input_dim, ff)
            ]))

        self.net = ReversibleSequence(layers) if reversible else SequentialSequence(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Model forward pass"""
        return self.net(x)


class LinformerModel(BaseNeuralForecaster):
    """Linformer implementation for forecasting."""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params or OperationParameters())
        self.input_dim = self.params.model_architecture.get("input_dim", 1)
        self.output_dim = self.params.model_architecture.get("output_dim", 1)
        self.depth = self.params.model_architecture.get("depth", 3)
        self.custom_params = self.params.model_architecture.get("custom_params", BASE_LINFORMER_PARAMS)
        self._init_model()

    def _init_model(self):
        """Model initialization."""
        self.model = Linformer(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            depth=self.depth,
            custom_params=self.custom_params
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

        pred = self.model(x)
        return pred[:, -1, :].unsqueeze(1)


class LinformerLM(nn.Module):
    """Linformer-based language model.

    Args:
        input_dim (int): Number of unique tokens in the dictionary.
        output_dim (int): Maximum sequence length.
        depth (int): Number of Linformer layers.
        custom_params (dict, optional): Additional parameters. Default: None.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            depth: int = 6,
            custom_params: Optional[dict] = None,
            **kwargs
    ):
        super().__init__()
        params = custom_params.copy() if custom_params is not None else {}
        params.update(kwargs)

        dim = params.get('dim', 512)
        self.token_emb = nn.Embedding(input_dim, dim)
        self.pos_emb = nn.Embedding(output_dim, dim)
        self.linformer = Linformer(
            input_dim=dim,
            output_dim=output_dim,
            depth=depth,
            custom_params=params
        )
        self.to_logits = nn.Linear(dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(x)
        x += self.pos_emb(torch.arange(x.shape[1], device=x.device))
        x = self.linformer(x)
        return self.to_logits(x)