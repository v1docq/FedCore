from functools import reduce
from operator import iadd
from typing import Dict, List, Optional

from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from tqdm import tqdm

from fedcore.algorithm.low_rank.rank_pruning import rank_threshold_pruning
from fedcore.algorithm.low_rank.svd_tools import load_svd_state_dict, decompose_module
from fedcore.losses.low_rank_loss import HoyerLoss, OrthogonalLoss
from fedcore.losses.utils import _get_loss_metric
from fedcore.models.network_impl.layers import DecomposedConv2d
from fedcore.neural_compressor.compression.pruner.utils import nn
from fedcore.repository.constanst_repository import ENERGY_THR, DECOMPOSE_MODE, FORWARD_MODE, HOER_LOSS, ORTOGONAL_LOSS
import torch
from fedcore.repository.constanst_repository import default_device


class LowRankModel:
    """Singular value decomposition for model structure optimization.

    Args:
        energy_thresholds: List of pruning hyperparameters.
        decomposing_mode: ``'channel'`` or ``'spatial'`` weights reshaping method.
        forward_mode: ``'one_layer'``, ``'two_layers'`` or ``'three_layers'`` forward pass calculation method.
        hoer_loss_factor: The hyperparameter by which the Hoyer loss function is
            multiplied.
        orthogonal_loss_factor: The hyperparameter by which the orthogonal loss
            function is multiplied.
    """

    def __init__(
            self, params: Optional[OperationParameters] = {}):
        self.epochs = params.get('epochs', 5)
        self.energy_thresholds = params.get('energy_thresholds', ENERGY_THR)
        self.decomposing_mode = params.get('decomposing_mode', DECOMPOSE_MODE)
        self.forward_mode = params.get('forward_mode', FORWARD_MODE)
        self.hoer_loss = HoyerLoss(params.get('hoyer_loss', HOER_LOSS))
        self.orthogonal_loss = OrthogonalLoss(params.get('orthogonal_loss', ORTOGONAL_LOSS))
        self.learning_rate = params.get('learning_rate', 0.001)
        self.finetuning = False
        self.device = default_device()

    def _init_model(self, input_data):
        self.loss_fn = _get_loss_metric(input_data)
        self.model = input_data.target
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, input_data.features.num_classes)
        )
        decompose_module(self.model, self.decomposing_mode,
                         forward_mode=self.forward_mode)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _fit_loop(self,
                  train_loader,
                  model,
                  total_iterations_limit=None,
                  custom_loss: dict = None):

        total_iterations = 0
        for epoch in range(1, self.epochs + 1):
            loss_sum = 0
            self.model.train()
            for batch in tqdm(train_loader):
                self.optimizer.zero_grad()
                total_iterations += 1
                inputs, targets = batch
                output = self.model(inputs.to(self.device))
                if custom_loss:
                    model_loss = {key: val(model) for key, val in custom_loss.items()}
                    model_loss['metric_loss'] = self.loss_fn(torch.argmax(output, dim=1).float(),
                                                targets.to(self.device).float())
                    loss_sum += sum([loss.item() for loss in model_loss.values()])
                    quality_loss = reduce(iadd, [loss for loss in model_loss.values()])
                else:
                    quality_loss = self.loss_fn(output, targets)
                    loss_sum += quality_loss.item()
                avg_loss = loss_sum / total_iterations
                quality_loss.backward()
                self.optimizer.step()

            print('Epoch: {}, {}, Training Loss: {:.2f}'.format(
                epoch, 'low_rank_loss', avg_loss))

            if total_iterations_limit is not None and total_iterations >= total_iterations_limit:
                return

    def fit(
            self,
            input_data
    ) -> None:
        """Run model training with optimization.

        Args:
            input_data: An instance of the model class
        """
        self._init_model(input_data)
        self._fit_loop(train_loader=input_data.features.train_dataloader,
                       model=self.model,
                       custom_loss=self.__loss())
        return self.optimize(model=self.model, params=input_data.features, ft_params=None)

    def predict_for_fit(self,input_data: InputData, output_mode: str = 'default'):
        return self.model

    def predict(self,
                input_data: InputData, output_mode: str = 'default'):
        return self.predict_for_fit(input_data, output_mode)

    def optimize(
            self,
            model,
            params,
            ft_params
    ) -> None:
        """Prunes the trained model at the given thresholds.

        Args:
            model: An instance of the experimenter class, e.g. ``ClassificationExperimenter``.
            params: An object containing training parameters.
            ft_params: An object containing fine-tuning parameters for optimized model.
        """
        self.finetuning = True
        for thr in self.energy_thresholds:
            for name, module in model.named_children():
                if isinstance(module, DecomposedConv2d):
                    rank_threshold_pruning(module, thr, name)
            if ft_params is not None:
                self._fit_loop(train_loader=params.train_dataloader,
                               model=self.model,
                               custom_loss=self.__loss())
        return self.model

    def __loss(self) -> Dict[str, torch.Tensor]:
        """
        Computes the orthogonal and the Hoer loss for the model.

        Args:
            model: CNN model with DecomposedConv layers.

        Returns:
            A dict ``{loss_name: loss_value}``
                where loss_name is a string and loss_value is a floating point tensor with a single value.
        """
        losses = {'orthogonal_loss': self.orthogonal_loss}
        if not self.finetuning:
            losses['hoer_loss'] = self.hoer_loss
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
            forward_mode=self.forward_mode
        )
        model.to(self.device)
