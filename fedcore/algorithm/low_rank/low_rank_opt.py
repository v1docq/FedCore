from typing import Dict, List, Optional
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from tqdm import tqdm
import torch_pruning as tp
from fedcore.algorithm.low_rank.rank_pruning import rank_threshold_pruning
from fedcore.algorithm.low_rank.svd_tools import load_svd_state_dict, decompose_module
from fedcore.losses.low_rank_loss import HoyerLoss, OrthogonalLoss
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
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
        self.epochs = params.get('epochs', 30)
        self.energy_thresholds = params.get('energy_thresholds', ENERGY_THR)
        self.decomposing_mode = params.get('decomposing_mode', DECOMPOSE_MODE)
        self.forward_mode = params.get('forward_mode', FORWARD_MODE)
        self.hoer_loss = HoyerLoss(params.get('hoyer_loss', HOER_LOSS))
        self.orthogonal_loss = OrthogonalLoss(params.get('orthogonal_loss', ORTOGONAL_LOSS))
        self.strategy = params.get('spectrum_pruning_strategy', 'median')
        self.learning_rate = params.get('learning_rate', 0.001)
        self.finetuning = False
        self.device = default_device()
        self.trainer = BaseNeuralModel(params)
        self.trainer.custom_loss = self.__loss()

    def _init_model(self, input_data):
        self.model = input_data.target
        self.model.fc = nn.Sequential(nn.Linear(self.model.fc.in_features, input_data.features.num_classes))
        decompose_module(self.model, self.decomposing_mode,
                         forward_mode=self.forward_mode)

    def _evaluate_model_acc(self, train_loader):
        correct = 0
        for batch in tqdm(train_loader):
            inputs, targets = batch
            output = self.model(inputs.to(self.device))
            correct += (torch.argmax(output, 1) == targets.to(self.device)).sum().item()
        acc = round(100 * correct / len(train_loader.dataset))
        return acc

    def fit(
            self,
            input_data
    ) -> None:
        """Run model training with optimization.

        Args:
            input_data: An instance of the model class
        """
        self._init_model(input_data)
        self.trainer.model = self.model
        self.model = self.trainer.fit(input_data)
        return self.optimize(model=self.model, params=input_data.features, ft_params=None)

    def predict_for_fit(self, input_data: InputData, output_mode: str = 'default'):
        return self.model

    def predict(self,
                input_data: InputData, output_mode: str = 'default'):
        return self.predict_for_fit(input_data, output_mode)

    def _prune_weight_rank(self, model, thr):
        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                self._prune_weight_rank(module, thr)
            if isinstance(module, DecomposedConv2d):
                rank_threshold_pruning(conv=module,
                                       threshold=thr,
                                       strategy=self.strategy,
                                       module_name=name)

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
        batch_iter = (b[0] for b in params.train_dataloader)
        first_batch = next(batch_iter).to(self.device)
        base_macs, base_nparams = tp.utils.count_ops_and_params(self.model, first_batch)
        # acc_before_pruning = self._evaluate_model_acc(params.train_dataloader)
        for thr in self.energy_thresholds:
            self._prune_weight_rank(model,thr)
            if self.strategy.__contains__('median'):
                break
            if ft_params is not None:
                self._fit_loop(train_loader=params.train_dataloader,
                               model=self.model,
                               custom_loss=self.__loss())
        print("==============After pruning=================")
        macs, nparams = tp.utils.count_ops_and_params(self.model, first_batch)
        # acc_after_pruning = self._evaluate_model_acc(params.train_dataloader)
        print("Params: %.2f M => %.2f M" % (base_nparams / 1e6, nparams / 1e6))
        print("MACs: %.2f G => %.2f G" % (base_macs / 1e9, macs / 1e9))
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
