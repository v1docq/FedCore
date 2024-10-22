from typing import Optional

import loralib as lora
import torch

from fedot.core.operations.operation_parameters import OperationParameters
from torch import nn

from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
from fedcore.repository.constanst_repository import default_device


class LoraTrainer:
    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__()
        self.lora_strategy = params.get("lora_strategy", None)
        self.device = default_device()
        self.trainer = BaseNeuralModel(params)

        self.model = None

    def _init_model(self, input_data):
        self.model = input_data.target
        lora.mark_only_lora_as_trainable(model=self.model, bias=self.lora_strategy)

    def fit(self, input_data):
        self._init_model(input_data)
        self.trainer.model = self.model
        self.model = self.trainer.fit(input_data)
        # self.compress(model=self.model, params=input_data.features, ft_params=None)


class LoRAParametrization(nn.Module):
    def __init__(self, features_in, features_out, rank=1, alpha=1, device="cpu"):
        super().__init__()
        # Section 4.1 of the paper:
        #   We use a random Gaussian initialization for A and zero for B, so ∆W = BA is zero at the beginning of training
        self.lora_A = nn.Parameter(torch.zeros((rank, features_out)).to(device))
        self.lora_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))
        nn.init.normal_(self.lora_A, mean=0, std=1)

        # Section 4.1 of the paper:
        #   We then scale ∆Wx by α/r , where α is a constant in r.
        #   When optimizing with Adam, tuning α is roughly the same as tuning the learning rate if we scale the initialization appropriately.
        #   As a result, we simply set α to the first r we try and do not tune it.
        #   This scaling helps to reduce the need to retune hyperparameters when we vary r.
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, original_weights):
        if self.enabled:
            # Return W + (B * A) * scale
            return (
                original_weights
                + torch.matmul(self.lora_B, self.lora_A).view(original_weights.shape)
                * self.scale
            )
        else:
            return original_weights


def linear_layer_parameterization(layer, device, rank=1, lora_alpha=1):
    # Only add the parameterization to the weight matrix, ignore the Bias

    # From section 4.2 of the paper:
    #   We limit our study to only adapting the attention weights for downstream tasks and freeze the MLP modules (so they are not trained in downstream tasks) both for simplicity and parameter-efficiency.
    #   [...]
    #   We leave the empirical investigation of [...], and biases to a future work.

    features_in, features_out = layer.weight.shape
    return LoRAParametrization(
        features_in, features_out, rank=rank, alpha=lora_alpha, device=device
    )
