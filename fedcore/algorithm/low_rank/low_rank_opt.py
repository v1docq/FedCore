from copy import deepcopy
from typing import Dict, Optional
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
import torch_pruning as tp
import torch
from torch import nn
import inspect

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
        decompose_module(
            model, 
            self.decomposing_mode, 
            self.decomposer, 
            self.compose_mode,
            self.decomposer_params
        )
        return model

    def fit(self, input_data) -> None:
        """Run model training with optimization.

        Args:
            input_data: An instance of the model class
        """
        model_before = self._init_model(input_data)
        example_batch = self._get_example_input(input_data)
        device = extract_device(model_before)
        
        if isinstance(example_batch, dict):
            example_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                           for k, v in example_batch.items()}
        elif isinstance(example_batch, (list, tuple)):
            example_batch = type(example_batch)(
                item.to(device) if isinstance(item, torch.Tensor) else item 
                for item in example_batch
            )
        else:
            example_batch = example_batch.to(device)
        
        base_params = self._estimate_params(model_before, example_batch)
        self.trainer.model = deepcopy(model_before)
        self.model_after = self.trainer.fit(input_data)
        self.compress(self.model_after)
        # check params
        self._diagnose(self.model_after, example_batch, *base_params,
            "After low rank truncation".center(80, '='))
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
            decomposer_params=self.decomposer_params,
        )
        model.to(self.device)
