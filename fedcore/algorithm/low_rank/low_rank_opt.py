from copy import deepcopy
from typing import Dict, Optional
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
import torch_pruning as tp
import torch
from torch import nn

from fedcore.algorithm.low_rank.rank_pruning import rank_threshold_pruning
from fedcore.algorithm.low_rank.svd_tools import load_svd_state_dict, decompose_module
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.losses.low_rank_loss import HoyerLoss, OrthogonalLoss
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
from fedcore.models.network_impl.decomposed_layers import IDecomposed
from fedcore.repository.constanst_repository import (
    DECOMPOSE_MODE,
    COMPOSE_MODE,
    # HOER_LOSS,
    # ORTOGONAL_LOSS
)
from fedcore.algorithm.base_compression_model import BaseCompressionModel


class LowRankModel(BaseCompressionModel):
    """Singular value decomposition for model structure optimization.

    Args:
        energy_thresholds: List of pruning hyperparameters.
        decomposing_mode: ``'channel'`` or ``'spatial'`` weights reshaping method.
        compose_mode: ``'one_layer'``, ``'two_layers'`` or ``'three_layers'`` forward pass calculation method.
        hoer_loss_factor: The hyperparameter by which the Hoyer loss function is
            multiplied.
        orthogonal_loss_factor: The hyperparameter by which the orthogonal loss
            function is multiplied.
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        self.epochs = params.get("epochs", 30)
        self.ft_params = params.get("finetune_params", None)
        if self.ft_params is None:
            self.ft_params = {}
            self.ft_params["custom_loss"] = None
            self.ft_params["epochs"] = round(self.epochs / 2)
        self.non_adaptive_threshold = params.get('non_adaptive_threshold', 0.5)
        # self.energy_thresholds = params.get("energy_thresholds", ENERGY_THR)
        self.decomposing_mode = params.get("decomposing_mode", DECOMPOSE_MODE)
        self.decomposer = params.get('decomposer', 'svd')
        self.compose_mode = params.get("compose_mode", None)
        # self.hoer_loss = HoyerLoss(params.get("hoyer_loss", HOER_LOSS))
        # self.orthogonal_loss = OrthogonalLoss(
        #     params.get("orthogonal_loss", ORTOGONAL_LOSS)
        # )
        self.strategy = params.get("spectrum_pruning_strategy", "quantile")
        self.learning_rate = params.get("learning_rate", 0.001)
        self.finetuning = False
        self.device = default_device()
        self.trainer = BaseNeuralModel(params)
        # self.trainer.custom_loss = self.__loss()

    def _init_model(self, input_data):
        self.model = (
            input_data.features.target
            if hasattr(input_data.features, "features")
            else input_data.predict.target
        )
        decompose_module(
            self.model, self.decomposing_mode, self.decomposer, self.compose_mode
        )

    def fit(self, input_data) -> None:
        """Run model training with optimization.

        Args:
            input_data: An instance of the model class
        """
        self._init_model(input_data)
        self.trainer.model = self.model
        self.model = self.trainer.fit(input_data)
        self.optimised_model = deepcopy(self.model)  ### do actually need deepcopy???
        self.compress(input_data=input_data)
        return self.optimised_model

    def predict_for_fit(self, input_data: InputData, output_mode: str = 'fedcore'):
        return (
            self.optimised_model if output_mode == 'fedcore' else self.model
        )  ### any case same model

    def predict(self, input_data: InputData, output_mode: str = 'fedcore'):
        self.trainer.model = (
            self.optimised_model if output_mode == 'fedcore' else self.model
        )  ### any case same model
        return self.trainer.predict(input_data, output_mode)

    def _prune_weight_rank(self, model: nn.Module, non_adaptive_thr: float):
        for name, module in model.named_modules():
            if isinstance(module, IDecomposed): 
                rank_threshold_pruning(decomposed_module=module,
                                       threshold=non_adaptive_thr,
                                       strategy=self.strategy,
                                       module_name=name)
            

    def _prepare_model_for_inference(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, IDecomposed):
                module.inference_mode = True
                module.compose_weight_for_inference()

    def compress(self, input_data) -> None:
        """Prunes the trained model at the given thresholds.

        Args:
            model: An instance of the experimenter class, e.g. ``ClassificationExperimenter``.
            params: An object containing training parameters.
            ft_params: An object containing fine-tuning parameters for optimized model.
        """
        batch_iter = (b[0] for b in input_data.features.train_dataloader)
        example_batch = self._get_example_input(input_data).to(self.device)
        example_batch = next(batch_iter).to(self.device)
        base_macs, base_nparams = tp.utils.count_ops_and_params(self.model, example_batch)

        print("==============Truncate rank for each weight matrix=================") 
        self._prune_weight_rank(self.optimised_model, self.non_adaptive_threshold)
        self._prepare_model_for_inference(self.optimised_model)

        print("==============Finetune truncated model=================")
        self.trainer.model = self.optimised_model
        # self.trainer.custom_loss = self.ft_params["custom_loss"]
        self.trainer.epochs = self.ft_params["epochs"]
        self.optimised_model = self.trainer.fit(input_data, loader_type='train')

        self._diagnose(example_batch, base_nparams, base_macs,
            "==============After low rank truncation=================")

    def __loss(self) -> Dict[str, torch.Tensor]:
        """
        Computes the orthogonal and the Hoer loss for the model.

        Args:
            model: model with IDecomposed layers.

        Returns:
            A dict ``{loss_name: loss_value}``
                where loss_name is a string and loss_value is a floating point tensor with a single value.
        """
        losses = {"orthogonal_loss": self.orthogonal_loss}
        if not self.finetuning:
            losses["hoer_loss"] = self.hoer_loss
        return losses

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
