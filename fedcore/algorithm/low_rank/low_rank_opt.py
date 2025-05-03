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
    """Singular value decomposition for model structure optimization.

    Args:
    """
    _additional_hooks = [LRHooks]

    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)
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
        return self.model_after

    def fit(self, input_data) -> None:
        """Run model training with optimization.

        Args:
            input_data: An instance of the model class
        """
        model_after = self._init_model(input_data)
        example_batch = self._get_example_input(input_data).to(extract_device(self.model_before))
        base_params = self._estimate_params(self.model_before, example_batch)
        self.trainer.model = self.model_after
        self.model_after = self.trainer.fit(input_data)
        self.compress(self.model_after)
        # check params
        self.estimate_params(example_batch, self.model_before, self.model_after)
        self.model_after._structure_changed__ = True
        return self.model_after

    def compress(self, model: nn.Module):
        for module in model.modules():
            if isinstance(module, IDecomposed):
                # module.inference_mode = True
                module.compose_weight_for_inference()

    def load_model(self, model, state_dict_path: str) -> None:
        """Loads the optimized model into the experimenter.

        Args:
            exp: An instance of the experimenter class, e.g. ``ClassificationExperimenter``.
            state_dict_path: Path to state_dict file.
        """
        load_svd_state_dict(
            model=model,
            state_dict_path=state_dict_path,
            decomposing_mode=self.decomposing_mode,
            compose_mode=self.compose_mode,
        )
        model.to(self.device)
