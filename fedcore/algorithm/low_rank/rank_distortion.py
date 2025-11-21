"""LoRA-based fine-tuning utilities.

This module provides:

* :class:`LoraTrainer` – a thin wrapper around :class:`BaseNeuralModel`
  that configures training for LoRA-augmented models (only LoRA parameters
  are trainable).
* :class:`LoRAParametrization` – a parametrization module implementing the
  low-rank update ΔW = BA scaled by α / r, following the original LoRA paper.
* :func:`linear_layer_parameterization` – a helper to create a
  :class:`LoRAParametrization` for a given linear layer.
"""

from typing import Optional

import loralib as lora
import torch

from fedot.core.operations.operation_parameters import OperationParameters
from torch import nn

from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
from fedcore.repository.constanst_repository import default_device


class LoraTrainer:
    """Trainer wrapper that enables LoRA-style fine-tuning.

    This class configures a :class:`BaseNeuralModel` to train only LoRA
    parameters on top of a frozen backbone model. All non-LoRA weights remain
    fixed; training updates only the low-rank adapters created via
    :mod:`loralib`.

    Parameters
    ----------
    params : Optional[OperationParameters], optional
        Configuration parameters passed to :class:`BaseNeuralModel` and used
        to control LoRA behaviour. Recognized keys include:

        * ``"lora_strategy"`` – bias handling strategy for
          :func:`loralib.mark_only_lora_as_trainable` (e.g. ``"lora_only"``,
          ``"all"``, etc.; see loralib docs).
        * Any additional keys supported by :class:`BaseNeuralModel`.
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__()
        self.lora_strategy = params.get("lora_strategy", None)
        self.device = default_device()
        self.trainer = BaseNeuralModel(None, params)

        self.model = None

    def _init_model(self, input_data):
        """Initialize model reference and mark only LoRA params as trainable.

        Parameters
        ----------
        input_data :
            Object providing access to the target model to be trained. The
            trainer expects ``input_data.target`` to hold the model instance.

        Notes
        -----
        After this call, only parameters registered by LoRA (and optionally
        biases, depending on ``self.lora_strategy``) are left trainable.
        """
        self.model = input_data.target
        lora.mark_only_lora_as_trainable(model=self.model, bias=self.lora_strategy)

    def fit(self, input_data):
        """Run training for LoRA parameters using the internal trainer.

        Parameters
        ----------
        input_data :
            Training input object expected by :class:`BaseNeuralModel`
            (e.g. experimenter, dataset descriptor, etc.). It must provide
            a ``target`` attribute with the model to be fine-tuned.

        Returns
        -------
        Any
            Result of :meth:`BaseNeuralModel.fit`. Typically a trained model
            instance with updated LoRA parameters.
        """
        self._init_model(input_data)
        self.trainer.model = self.model
        self.model = self.trainer.fit(input_data)
        # self.compress(model=self.model, params=input_data.features, ft_params=None)


class LoRAParametrization(nn.Module):
    """LoRA weight parametrization module.

    This module implements a low-rank update of a weight matrix:

    .. math::

        W' = W + \\Delta W, \\quad \\Delta W = BA \\cdot \\frac{\\alpha}{r},

    where ``B`` and ``A`` are learnable low-rank factors of shapes
    ``(features_in, rank)`` and ``(rank, features_out)`` respectively,
    ``r`` is the rank, and ``α`` is a scaling factor.

    Parameters
    ----------
    features_in : int
        Input dimensionality of the underlying linear layer (rows of W).
    features_out : int
        Output dimensionality of the underlying linear layer (columns of W).
    rank : int, optional
        LoRA rank ``r`` (the dimensionality of the low-rank bottleneck),
        by default ``1``.
    alpha : float, optional
        Scaling factor ``α`` used as ``alpha / rank`` in the update, by
        default ``1``.
    device : str or torch.device, optional
        Device on which to allocate LoRA parameters, by default ``"cpu"``.
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
        """Apply LoRA update to the original weight matrix.

        Parameters
        ----------
        original_weights : torch.Tensor
            Original (frozen) weight tensor to which the low-rank update
            should be applied.

        Returns
        -------
        torch.Tensor
            Updated weights ``W'`` if ``self.enabled`` is ``True``, otherwise
            ``original_weights`` unchanged.
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
    """Create a LoRA parametrization for a given linear layer.

    The parametrization is applied only to the weight matrix of the layer;
    the bias term (if present) is intentionally ignored, following the LoRA
    paper that focuses on adapting attention weights and leaves biases for
    future investigation.

    Parameters
    ----------
    layer : torch.nn.Module
        Linear-like layer with a ``weight`` attribute of shape
        ``(features_out, features_in)``.
    device : str or torch.device
        Device on which to allocate LoRA parameters.
    rank : int, optional
        LoRA rank ``r`` (bottleneck dimensionality), by default ``1``.
    lora_alpha : float, optional
        Scaling factor ``α`` used as ``lora_alpha / rank``, by default ``1``.

    Returns
    -------
    LoRAParametrization
        A parametrization module that can be used to wrap the layer's weight
        in a LoRA-style update.
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
