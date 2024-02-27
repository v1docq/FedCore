from copy import deepcopy

import numpy as np
from fedot.core.data.data import InputData
from fedcore.algorithm.base_compression_model import BaseCompressionModel
import torch_pruning as tp
from typing import Optional
import torch
from fedot.core.operations.operation_parameters import OperationParameters

from fedcore.repository.constanst_repository import PRUNERS, PRUNING_IMPORTANCE


class BasePruner(BaseCompressionModel):
    """Class responsible for Pruning model implementation.
    Example:
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)
        self.epochs = params.get('epochs', 5)
        self.pruner_name = params.get('pruner_name', 'magnitude_pruner')
        self.pruning_channels = params.get('channels_to_prune', [2])
        self.pruning_ratio = params.get('pruning_ratio', 0.5)
        self.importance = params.get('importance', 'MagnitudeImportance')
        self.importance_norm = params.get('importance_norm', 2)
        self.importance_reduction = params.get('importance_reduction', 'mean')
        self.importance_normalize = params.get('importance_normalize', 'mean')
        # importance criterion for parameter selections
        self.importance = PRUNING_IMPORTANCE[self.importance](p=self.importance_norm,
                                                              group_reduction=self.importance_reduction,
                                                              normalizer=self.importance_normalize)
        self.pruner = None

    def __repr__(self):
        return self.pruner_name

    def fit(self,
            input_data: InputData):
        self.model = input_data.target
        self.model_for_inference = deepcopy(self.model)
        # build dependency graph for model
        DG = tp.DependencyGraph().build_dependency(self.model,
                                                   example_inputs=input_data.features)

        # Select some channels to prune. Here we prune the channels indexed by [2, 6, 9].
        self.pruning_group = DG.get_pruning_group(self.model.conv1,
                                                  tp.prune_conv_out_channels,
                                                  idxs=self.pruning_channels)

        # ignore some layers that should not be pruned, e.g., the final classifier layer.
        self.ignored_layers = []
        for m in self.model.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
                self.ignored_layers.append(m)  # DO NOT prune the final classifier!

    def predict_for_fit(self,
                        input_data: InputData, output_mode: str = 'default') -> np.array:

        # Pruner initialization
        self.pruner = PRUNERS[self.pruner_name]
        self.pruner = self.pruner(
            self.model,
            input_data.features,
            global_pruning=False,  # If False, a uniform ratio will be assigned to different layers.
            importance=self.importance,  # importance criterion for parameter selection
            iterative_steps=self.epochs,  # the number of iterations to achieve target ratio
            pruning_ratio=self.pruning_ratio,
            # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            ignored_layers=self.ignored_layers,
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
        return self.model if self.pruner is not None else self.predict_for_fit(input_data, output_mode)
