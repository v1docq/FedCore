"""
Low-rank optimization model for FedCore.

This module defines :class:`LowRankModel`, a compression-aware training wrapper that:
  - clones the original model graph,
  - applies structural low-rank decomposition to supported layers
    (via :func:`fedcore.algorithm.low_rank.svd_tools.decompose_module`),
  - trains the decomposed model with optional pruning hooks (see ``LRHooks``),
  - optionally composes factorized weights back for inference.

Integration notes
-----------------
* The class extends :class:`fedcore.algorithm.base_compression_model.BaseCompressionModel`
  and reuses its trainer lifecycle, device handling, and parameter estimation utilities.
* Layer decomposition, rank pruning, and state loading are delegated to utilities from
  ``fedcore.algorithm.low_rank``.
* The original (pre-decomposition) model is kept in ``self.model_before`` for analysis and
  parameter comparison; the decomposed/optimized copy is stored in ``self.model_after``.
"""

from copy import deepcopy
from typing import Dict, Optional
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
import torch_pruning as tp
import torch
from torch import nn

from fedcore.algorithm.low_rank.rank_pruning import rank_threshold_pruning
from fedcore.algorithm.low_rank.svd_tools import load_svd_state_dict, decompose_module
from fedcore.architecture.comptutaional.devices import default_device, extract_device
from fedcore.losses.low_rank_loss import HoyerLoss, OrthogonalLoss
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
from fedcore.models.network_impl.decomposed_layers import IDecomposed
from fedcore.repository.constanst_repository import (
    DECOMPOSE_MODE,
    LRHooks
)
from fedcore.algorithm.base_compression_model import BaseCompressionModel


class LowRankModel(BaseCompressionModel):
    """Low-rank (SVD-family) optimization for neural network structures.

    This wrapper prepares a decomposed copy of the model, trains it with
    low-rank–specific regularizers/hooks, and (optionally) composes weights
    for efficient inference.

    Attributes
    ----------
    decomposing_mode : str
        Structural decomposition mode (taken from ``DECOMPOSE_MODE`` by default).
    decomposer : str
        Decomposition algorithm key (e.g., ``'svd'``, ``'rsvd'``, ``'cur'``).
    compose_mode : Optional[str]
        Composition strategy used when reassembling factors for inference.
    device : torch.device
        Device on which the decomposed model is placed.
    model_before : nn.Module
        Original model instance returned by the base initializer (kept for analysis).
    model_after : nn.Module
        Deep-copied and decomposed model that is actually trained.
    _additional_hooks : list
        Registry of hook groups enabled for this optimization (here: ``[LRHooks]``).

    Parameters
    ----------
    params : Optional[OperationParameters], default={}
        Configuration dictionary (``OperationParameters``-like) that may contain:
          - ``'decomposing_mode'``: str, decomposition structure mode
          - ``'decomposer'``: str, decomposition algorithm key (default: ``'svd'``)
          - ``'compose_mode'``: Optional[str], inference composition strategy
          - other keys consumed by the base compression/trainer pipeline.
    """
    _additional_hooks = [LRHooks]

    def __init__(self, params: Optional[OperationParameters] = {}):
        """Initialize the low-rank optimization wrapper.

        Notes
        -----
        No graph changes are performed here. Actual layer decomposition
        happens inside :meth:`_init_model`.

        Parameters
        ----------
        params : Optional[OperationParameters], default={}
            Operation parameters; see class docstring for supported keys.
        """
        super().__init__(params)
        self.decomposing_mode = params.get("decomposing_mode", DECOMPOSE_MODE)
        self.decomposer = params.get('decomposer', 'svd')
        self.compose_mode = params.get("compose_mode", None)
        self.device = default_device()

    def _init_model(self, input_data):
        """Create and prepare a decomposed copy of the base model.

        Steps
        -----
        1. Call the base initializer to obtain the original model.
        2. Deep-copy the model into ``self.model_after``.
        3. Apply structural decomposition to supported layers according to
           ``self.decomposing_mode`` / ``self.decomposer`` / ``self.compose_mode``.
        4. Move the decomposed model to ``self.device``.

        Parameters
        ----------
        input_data : fedot.core.data.data.InputData or compatible
            Data object passed to the underlying base initializer.

        Returns
        -------
        nn.Module
            Decomposed model placed on the target device.
        """
        model = super()._init_model(input_data, self._additional_hooks)
        self.model_before = model
        self.model_after = deepcopy(model)
        decompose_module(
            self.model_after, self.decomposing_mode, self.decomposer, self.compose_mode
        )
        self.model_after.to(self.device)
        return self.model_after

    def fit(self, input_data) -> None:
        """Train the decomposed model and finalize bookkeeping.

        The method:
          * builds the decomposed model via :meth:`_init_model`,
          * sets it into the trainer and runs the training loop,
          * estimates/compares parameters of the before/after models,
          * marks the resulting model as structurally changed.

        Parameters
        ----------
        input_data : fedot.core.data.data.InputData
            Training (and validation) data consumed by the trainer.

        Returns
        -------
        nn.Module
            The trained decomposed model (``self.model_after``).
        """
        model_after = self._init_model(input_data)
        # base_params = self._estimate_params(self.model_before, example_batch)
        self.trainer.model = self.model_after
        self.model_after = self.trainer.fit(input_data)
        # self.compress(self.model_after)
        # check params
        example_batch = self._get_example_input(input_data)#.to(extract_device(self.model_before))
        self.estimate_params(example_batch, self.model_before, self.model_after)
        self.model_after._structure_changed__ = True
        return self.model_after

    def compress(self, model: nn.Module):
        """Compose factorized weights for inference for all decomposed layers.

        This utility iterates over modules and calls
        :meth:`fedcore.models.network_impl.decomposed_layers.IDecomposed.compose_weight_for_inference`
        on each decomposed layer.

        Parameters
        ----------
        model : nn.Module
            Model whose decomposed layers should be composed.
        """
        for module in model.modules():
            if isinstance(module, IDecomposed):
                # module.inference_mode = True
                module.compose_weight_for_inference()

    def load_model(self, model, state_dict_path: str) -> None:
        """Load a previously saved low-rank state into a model.

        Uses :func:`fedcore.algorithm.low_rank.svd_tools.load_svd_state_dict`
        to correctly restore factorized parameters according to
        ``self.decomposing_mode`` and ``self.compose_mode``.

        Parameters
        ----------
        model : nn.Module
            Target model instance to load weights into.
        state_dict_path : str
            Path to the serialized ``state_dict`` file.

        Returns
        -------
        None
        """
        load_svd_state_dict(
            model=model,
            state_dict_path=state_dict_path,
            decomposing_mode=self.decomposing_mode,
            compose_mode=self.compose_mode,
        )
        model.to(self.device)
