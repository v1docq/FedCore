from copy import deepcopy
from typing import Optional
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
import torch_pruning as tp
import torch
from torch import nn

from transformers import AutoTokenizer

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
from fedcore.algorithm.reassembly.transmla_reassembler import TransMLA
from external.transmlacore.modify_config import settings


class LowRankModel(BaseCompressionModel):
    """Singular value decomposition for model structure optimization.
    
    This class performs low-rank decomposition of models for compression.
    
    Args:
        decomposing_mode: Decomposition mode ('channel' or 'spatial')
        decomposer: Decomposition method ('svd', 'cur', 'rsvd')
        compose_mode: Composition mode ('one_layer', 'two_layers', 'three_layers')
    """
    _additional_hooks = [LRHooks]

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params or {})
        self.decomposing_mode = params.get("decomposing_mode", DECOMPOSE_MODE)
        self.decomposer = params.get('decomposer', 'svd')
        self.compose_mode = params.get("compose_mode", None)
        self.device = default_device()

    def _init_model(self, input_data):
        model = super()._init_model(input_data, self._additional_hooks)
        self.model_before = model
        self.model_after = deepcopy(model)
        decompose_module(
            self.model_after, self.decomposing_mode, self.decomposer, self.compose_mode
        )
        self.model_after.to(self.device)
        return self.model_after

    def fit(self, input_data) -> None:
        """Perform low-rank decomposition and model training.

        Args:
            input_data: Instance of model class
        """
        model_after = self._init_model(input_data)
        self.trainer.model = self.model_after
        self.model_after = self.trainer.fit(input_data)
        # self.compress(self.model_after)
        # Parameter check
        example_batch = self._get_example_input(input_data)
        self.estimate_params(example_batch, self.model_before, self.model_after)
        self.model_after._structure_changed__ = True
        return self.model_after

    def compress(self, model: nn.Module):
        """Model compression by composing weights for inference."""
        for module in model.modules():
            if isinstance(module, IDecomposed):
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

    def load_model(self, model, state_dict_path: str) -> None:
        """Load optimized model.

        Args:
            model: Model to load
            state_dict_path: Path to state dict file
        """
        load_svd_state_dict(
            model=model,
            state_dict_path=state_dict_path,
            decomposing_mode=self.decomposing_mode,
            compose_mode=self.compose_mode,
        )
        model.to(self.device)
    
    def predict_for_fit(self, input_data, output_mode: str = 'fedcore'):
        """Return model after training."""
        return self.model_after if output_mode == 'fedcore' else self.model_before

    def predict(self, input_data, output_mode: str = 'fedcore'):
        """Prediction using compressed model."""
        if output_mode == 'fedcore':
            self.trainer.model = self.model_after
        else:
            self.trainer.model = self.model_before
        return self.trainer.predict(input_data, output_mode)
