"""
LoRA (Low-Rank Adaptation) helpers for FedCore.

This module provides a minimal training wrapper and a parameterization module
to fine-tune neural networks with low-rank adapters on top of a (largely)
frozen base model.

Components
---------
- LoraTrainer:
    A thin orchestrator that (a) takes a ready model from the incoming
    ``InputData`` container, (b) freezes non-LoRA parameters, enabling only
    LoRA parameters for training via ``loralib.mark_only_lora_as_trainable``,
    and (c) delegates the training loop to :class:`BaseNeuralModel`.

- LoRAParametrization:
    A drop-in parameterization that implements the classic LoRA update
    ΔW = (B @ A) * (α / r) added to the original weight matrix W.
    Initialization follows Section 4.1 of the LoRA paper:
      * A ~ N(0, 1) (random Gaussian),
      * B = 0 so that ΔW starts at 0.

- linear_layer_parameterization:
    Convenience factory that creates :class:`LoRAParametrization` for a given
    linear layer's weight dimensions.

References
----------
Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021/2022.
"""

from typing import Optional 

import loralib as lora
import torch

from fedot.core.operations.operation_parameters import OperationParameters
from torch import nn

from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
from fedcore.repository.constanst_repository import default_device


class LoraTrainer:
    """
    Minimal trainer wrapper for LoRA fine-tuning.

    This class expects that the incoming ``input_data`` object carries a
    pre-constructed model in ``input_data.target``. It freezes all non-LoRA
    parameters and enables training only for LoRA parameters according to
    the configured bias strategy.

    Parameters
    ----------
    params : Optional[OperationParameters], default={}
        Operation parameters. Recognized keys:
          - ``lora_strategy`` : Optional[str]
                Bias handling policy forwarded to
                ``loralib.mark_only_lora_as_trainable(..., bias=...)``.
                Typical values: ``None``, ``'all'``, ``'lora_only'``.

    Attributes
    ----------
    lora_strategy : Optional[str]
        Current bias policy for LoRA training.
    device : torch.device
        Target computation device.
    trainer : BaseNeuralModel
        Training loop driver used to fit the model.
    model : Optional[nn.Module]
        The underlying model being fine-tuned.
    """
    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__()
        self.lora_strategy = params.get("lora_strategy", None)
        self.device = default_device()
        self.trainer = BaseNeuralModel(params)

        self.model = None

    def _init_model(self, input_data):
        """
        Initialize the model and restrict trainable params to LoRA.

        Notes
        -----
        The model is taken directly from ``input_data.target``. All parameters
        except LoRA parameters are frozen via ``loralib.mark_only_lora_as_trainable``.

        Parameters
        ----------
        input_data : Any
            Container that must expose the model in ``input_data.target``.
        """
        self.model = input_data.target
        lora.mark_only_lora_as_trainable(model=self.model, bias=self.lora_strategy)

    def fit(self, input_data):
        """
        Run the training loop with LoRA-enabled parameters.

        Steps
        -----
        1) Initialize model and mark only LoRA parameters as trainable.
        2) Attach the model to :attr:`trainer`.
        3) Delegate training to :meth:`BaseNeuralModel.fit`.

        Parameters
        ----------
        input_data : Any
            Training data structure compatible with :class:`BaseNeuralModel`.

        Returns
        -------
        nn.Module
            The trained model instance with updated LoRA parameters.
        """
        self._init_model(input_data)
        self.trainer.model = self.model
        self.model = self.trainer.fit(input_data)
        # self.compress(model=self.model, params=input_data.features, ft_params=None)
        return self.model


class LoRAParametrization(nn.Module):
    """
    LoRA parameterization for a weight matrix.

    Implements ΔW = (B @ A) * (α / r) and returns W + ΔW at forward time.
    The shapes follow the paper's convention and this code's implementation:
      * ``A`` has shape (r, features_out),
      * ``B`` has shape (features_in, r).

    Initialization
    --------------
    - ``A`` ~ N(0, 1)
    - ``B`` = 0
    so that ΔW initially equals 0 (Section 4.1 in the paper).

    Parameters
    ----------
    features_in : int
        Input dimensionality of the base weight matrix W.
    features_out : int
        Output dimensionality of W.
    rank : int, default=1
        Low-rank dimension ``r``.
    alpha : float, default=1
        Scaling coefficient ``α`` used as ``α / r``.
    device : str or torch.device, default="cpu"
        Device for the LoRA parameters.

    Attributes
    ----------
    lora_A : nn.Parameter
        Right factor A ∈ R^{r × features_out}.
    lora_B : nn.Parameter
        Left factor B ∈ R^{features_in × r}.
    scale : float
        Precomputed scaling factor α / r.
    enabled : bool
        Gate to toggle the LoRA update on/off at forward time.
    """
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
        """
        Apply the LoRA update to the given base weights.

        Parameters
        ----------
        original_weights : torch.Tensor
            The original (frozen) weight tensor W.

        Returns
        -------
        torch.Tensor
            Updated weights W + ΔW if :attr:`enabled` is True; otherwise W.
        """
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
    """
    Create LoRA parameterization for a linear layer's weight.

    Only the weight matrix is parameterized; the bias, if present, is ignored.
    This mirrors the common LoRA practice of adapting attention/linear weights
    while keeping biases and certain MLP parts frozen (cf. Section 4.2).

    Parameters
    ----------
    layer : nn.Linear or compatible
        Module whose ``weight.shape`` is used to size the LoRA factors.
    device : str or torch.device
        Device on which to allocate the LoRA parameters.
    rank : int, default=1
        Low-rank dimension r.
    lora_alpha : float, default=1
        Scaling coefficient α (used as α / r).

    Returns
    -------
    LoRAParametrization
        A parameterization module sized to the layer's weight.
    """
    # Only add the parameterization to the weight matrix, ignore the Bias

    # From section 4.2 of the paper:
    #   We limit our study to only adapting the attention weights for downstream tasks and freeze the MLP modules (so they are not trained in downstream tasks) both for simplicity and parameter-efficiency.
    #   [...]
    #   We leave the empirical investigation of [...], and biases to a future work.

    features_in, features_out = layer.weight.shape
    return LoRAParametrization(
        features_in, features_out, rank=rank, alpha=lora_alpha, device=device
    )
