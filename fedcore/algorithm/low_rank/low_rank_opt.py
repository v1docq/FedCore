"""Low-rank (SVD-based) compression model wrapper.

This module defines :class:`LowRankModel`, a compression wrapper around
:class:`BaseNeuralModel` that:

* injects low-rank–specific hooks (see :data:`LRHooks`);
* trains the base model;
* decomposes supported layers in-place using SVD (or another decomposer);
* optionally composes weights for inference and loads decomposed checkpoints.
"""

from typing import Dict, Optional
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
import torch_pruning as tp
import torch
from torch import nn
import inspect

from transformers import AutoTokenizer

from fedcore.algorithm.low_rank.rank_pruning import rank_threshold_pruning
from fedcore.algorithm.low_rank.svd_tools import load_svd_state_dict, decompose_module
from fedcore.architecture.computational.devices import default_device, extract_device
from fedcore.losses.low_rank_loss import HoyerLoss, OrthogonalLoss
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
from fedcore.models.network_impl.decomposed_layers import IDecomposed
from fedcore.repository.constant_repository import (
    DECOMPOSE_MODE,
    LRHooks
)
from fedcore.algorithm.base_compression_model import BaseCompressionModel
from tdecomp._base import Decomposer
from tdecomp.matrix.decomposer import DECOMPOSERS
from fedcore.algorithm.low_rank.reassembly import TransMLA, FlatLLM
from external.transmlacore.modify_config import settings


def _get_all_decomposer_params():
    """Dynamically extract all unique parameters from all decomposer classes.

    Returns:
        set: Set of all parameter names across all decomposer implementations
    """
    all_params = set()
    for decomposer_cls in DECOMPOSERS.values():
        sig = inspect.signature(decomposer_cls.__init__)
        all_params.update(sig.parameters.keys())
    all_params.discard('self')
    return all_params


_DECOMPOSER_PARAMS = _get_all_decomposer_params()


class LowRankModel(BaseCompressionModel):
    """Compression model that applies low-rank (SVD-based) decomposition.

    The class wraps training of a base neural network and replaces eligible
    layers with low-rank decomposed counterparts. It also registers rank
    pruning hooks defined in :data:`LRHooks` and provides utilities for
    composing decomposed weights and loading decomposed checkpoints.

    Parameters
    ----------
    params : dict, optional
        Configuration dictionary. Common keys include:

        * ``decomposing_mode``: decomposition mode passed to
          :func:`decompose_module_in_place` (defaults to
          :data:`DECOMPOSE_MODE`).
        * ``decomposer``: name of the decomposer to use
          (e.g. ``"svd"``; default is ``"svd"``).
        * ``compose_mode``: compose mode passed to
          :func:`decompose_module_in_place` and
          :func:`load_svd_state_dict` (or ``None`` to use default).
        * Any additional keys supported by :class:`BaseCompressionModel`
          and :class:`BaseNeuralModel` (e.g. optimizer, scheduler, device).
    """
    _additional_hooks = [LRHooks]

    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params or {})
        self.decomposing_mode = params.get("decomposing_mode", DECOMPOSE_MODE)
        self.decomposer = params.get('decomposer', 'svd')
        self.compose_mode = params.get("compose_mode", None)
        self.device = default_device()

        self.decomposer_params = self._extract_decomposer_params(params)

    def _extract_decomposer_params(self, params: OperationParameters) -> dict:
        """Extract decomposer parameters from operation parameters.

        Args:
            params: Operation parameters from config

        Returns:
            dict: Parameters for tdecomp decomposer (rank, distortion_factor, etc.)
        """
        params_dict = params.to_dict() if hasattr(params, 'to_dict') else dict(params)
        filtered_params = {
            key: value for key, value in params_dict.items()
            if key in _DECOMPOSER_PARAMS
        }

        if 'rank' not in filtered_params:
            filtered_params['rank'] = None

        if self.decomposer == 'svd':
            filtered_params.pop('power', None)
            filtered_params.pop('random_init', None)
        elif self.decomposer == 'two_sided':
            filtered_params.pop('power', None)
        elif self.decomposer == 'cur':
            filtered_params.pop('power', None)
            filtered_params.pop('random_init', None)

        return filtered_params

    def _get_example_input(self, input_data):
        """Override to handle tuples/lists with 2 or 3 elements (for LLM data),
        as well as dictionary inputs.

        Args:
            input_data: InputData or CompressionInputData with dataloader

        Returns:
            Tensor or structure that can be used as model input
        """
        batch = super()._get_example_input(input_data)

        if isinstance(batch, dict):
            return batch

        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0]

        return batch

    def _init_model(self, input_data):
        model = super()._init_model(input_data, self._additional_hooks)

        if self._model_id_before:
            self._registry.update_metrics(
                fedcore_id=self._fedcore_id,
                model_id=self._model_id_before,
                metrics={},
                stage="before",
                mode=self.__class__.__name__
            )


        decompose_module(
            model,
            self.decomposing_mode,
            self.decomposer,
            self.compose_mode,
            self.decomposer_params
        )
        model.to(self.device)

        self.model_after = model

        return self.model_after

    def fit(self, input_data) -> None:
        """Train the model with low-rank–aware hooks and estimate compression.

        The method prepares the trainer and models, runs training, and then
        evaluates parameter statistics before and after decomposition.

        Parameters
        ----------
        input_data :
            Object required by the trainer to perform fitting
            (e.g. experimenter instance, dataloaders, or config).

        Returns
        -------
        nn.Module
            Trained and decomposed model instance (``self.model_after``) with
            ``_structure_changed__`` flag set to ``True``.
        """
        model_after = self._init_model(input_data)
        # base_params = self._estimate_params(self.model_before, example_batch)
        self.trainer.model = self.model_after
        self.model_after = self.trainer.fit(input_data)

        if self._model_id_after:
            self._registry.update_metrics(
                fedcore_id=self._fedcore_id,
                model_id=self._model_id_after,
                metrics={},
                stage="after",
                mode=None,
                trainer=self.trainer
            )

        # self.compress(self.model_after)
        # check params
        example_batch = self._get_example_input(input_data)#.to(extract_device(self.model_before))
        self.estimate_params(example_batch, self.model_before, self.model_after)
        self.model_after._structure_changed__ = True
        return self.model_after

    def compress(self, model: nn.Module):
        """Compose weights of all decomposed layers for inference.

        This helper iterates over all modules of the given ``model`` and,
        for each instance of :class:`IDecomposed`, calls
        :meth:`IDecomposed.compose_weight_for_inference` to materialize the
        effective weight matrix.

        Parameters
        ----------
        model : nn.Module
            Model whose decomposed layers should be switched to inference form.
        """
        for module in model.modules():
            if isinstance(module, IDecomposed):
                # module.inference_mode = True
                module.compose_weight_for_inference()

        model_type = getattr(getattr(model, 'config', None), 'model_type', None)

        if model_type in settings:
            from transformers import AutoTokenizer

            model_name = getattr(model.config, "name_or_path", None)
            if model_name is None:
                raise ValueError("Can't find model name in config to load tokenizer")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            trans_mla = TransMLA()
            model = trans_mla.reassemble(model, tokenizer=tokenizer)

        if model_type in ['llama', 'mistral']:
            from transformers import AutoTokenizer

            model_name = getattr(model.config, "name_or_path", None)
            if model_name is None:
                raise ValueError("Can't find model name in config to load tokenizer")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            flat_llm = FlatLLM()
            model = flat_llm.reassemble(model, architecture=model_type, tokenizer=tokenizer)

    def load_model(self, model, state_dict_path: str) -> None:
        """Load a decomposed (SVD-based) checkpoint into a model.

        The method uses :func:`load_svd_state_dict` to restore a state dict
        that already contains low-rank–decomposed parameters, and then moves
        the model to ``self.device``.

        Parameters
        ----------
        model : nn.Module
            Model instance into which the state dict will be loaded.
        state_dict_path : str
            Path to the serialized state dict file containing decomposed
            parameters.
        """
        load_svd_state_dict(
            model=model,
            state_dict_path=state_dict_path,
            decomposing_mode=self.decomposing_mode,
            compose_mode=self.compose_mode,
            decomposer_params=self.decomposer_params,
        )
        model.to(self.device)

    def predict_for_fit(self, input_data, output_mode: str = 'fedcore'):
        """Return model after training."""
        return self.predict(input_data, output_mode)

    def predict(self, input_data, output_mode: str = 'fedcore'):
        """Prediction using compressed model."""
        if output_mode == 'fedcore':
            self.trainer.model = self.model_after
        else:
            self.trainer.model = self.model_before
        return self.trainer.predict(input_data, output_mode)
