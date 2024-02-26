from copy import deepcopy

import numpy as np
from fedot.core.data.data import InputData
from fedcore.algorithm.base_compression_model import BaseCompressionModel
import torch_pruning as tp
from typing import Optional
import torch
from fedot.core.operations.operation_parameters import OperationParameters

from fedcore.repository.model_repository import PRUNER_MODELS


class BasePruner(BaseCompressionModel):
    """Class responsible for Pruning model implementation.
    Example:
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)
        self.epochs = params.get('epochs', 5)
        self.pruner_name = params.get('pruner_name', 'magnitude_pruner')
        self.pruner = PRUNER_MODELS[self.pruner_name]

    def __repr__(self):
        return self.pruner_name

    def fit(self,
            input_data: InputData):
        self.model = input_data.supplementary_data['model']
        self.model_for_inference = deepcopy(self.model)
        # build dependency graph for model
        DG = tp.DependencyGraph().build_dependency(self.model,
                                                   example_inputs=input_data.features)

        # Select some channels to prune. Here we prune the channels indexed by [2, 6, 9].
        pruning_idxs = input_data.supplementary_data['channels_to_prune']
        self.pruning_group = DG.get_pruning_group(self.model.conv1, tp.prune_conv_out_channels, idxs=pruning_idxs)

        # importance criterion for parameter selections
        self.importance = tp.importance.MagnitudeImportance(p=2, group_reduction='mean')

    def predict_for_fit(self,
                        input_data: InputData, output_mode: str = 'default') -> np.array:

        # ignore some layers that should not be pruned, e.g., the final classifier layer.
        ignored_layers = []
        for m in self.model.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
                ignored_layers.append(m)  # DO NOT prune the final classifier!

        # Pruner initialization
        self.pruner = self.pruner(
            self.model,
            input_data.features,
            global_pruning=False,  # If False, a uniform ratio will be assigned to different layers.
            importance=self.importance,  # importance criterion for parameter selection
            iterative_steps=self.epochs,  # the number of iterations to achieve target ratio
            pruning_ratio=0.5,
            # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            ignored_layers=ignored_layers,
        )

        base_macs, base_nparams = tp.utils.count_ops_and_params(self.model, input_data.features)

        for i in range(self.epochs):
            # 3. the pruner.step will remove some channels from the model with least importance
            self.pruner.step()
            macs, nparams = tp.utils.count_ops_and_params(self.model, input_data.features)
            print("Iter %d/%d, Params: %.2f M => %.2f M" % (i + 1, self.epochs, base_nparams / 1e6, nparams / 1e6))
            print(" Iter %d/%d, MACs: %.2f G => %.2f G" % (i + 1, self.epochs, base_macs / 1e9, macs / 1e9))
        return self.model

    def predict(self,
                input_data: InputData, output_mode: str = 'default') -> np.array:
        return self.predict_for_fit(input_data, output_mode)
