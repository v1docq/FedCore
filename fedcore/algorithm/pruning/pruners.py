from copy import deepcopy

import numpy as np
import torchvision
from fedot.core.data.data import InputData
from fedcore.algorithm.base_compression_model import BaseCompressionModel
import torch_pruning as tp
from typing import Optional
import torch
from fedot.core.operations.operation_parameters import OperationParameters

from fedcore.repository.constanst_repository import PRUNERS, PRUNING_IMPORTANCE, PRUNING_LAYERS_IMPL


class BasePruner(BaseCompressionModel):
    """Class responsible for Pruning model implementation.
    Example:
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)
        self.epochs = params.get('epochs', 5)
        self.pruner_name = params.get('pruner_name', 'meta_pruner')
        self.pruning_ratio = params.get('pruning_ratio', 0.5)
        self.importance = params.get('importance', 'GroupNormImportance')
        self.importance_norm = params.get('importance_norm', 2)
        self.importance_reduction = params.get('importance_reduction', 'mean')
        self.importance_normalize = params.get('importance_normalize', 'mean')
        # importance criterion for parameter selections
        self.importance = PRUNING_IMPORTANCE[self.importance](group_reduction=self.importance_reduction,
                                                              normalizer=self.importance_normalize)
        self.pruner = None

    def __repr__(self):
        return self.pruner_name

    def fit(self,
            input_data: InputData):
        self.model = input_data.target
        self.model_for_inference = deepcopy(self.model)
        # DG = tp.DependencyGraph().build_dependency(self.model,
        #                                            example_inputs=input_data.features[0][0][np.newaxis, :, :, :])
        #
        # # 2. Group coupled layers for model.conv1
        # group = DG.get_pruning_group(self.model.conv2, tp.prune_conv_out_channels,idxs=[2,6,9])
        #
        # # 3. Prune grouped layers altogether
        # if DG.check_pruning_group(group):  # avoid full pruning, i.e., channels=0.
        #     group.prune()

        self.ignored_layers = []

        for m in self.model.modules():
            if isinstance(m, PRUNING_LAYERS_IMPL):
                continue
            else:
                self.ignored_layers.append(m)  # DO NOT prune the final classifier!

    def predict_for_fit(self,
                        input_data: InputData, output_mode: str = 'default') -> np.array:

        # Pruner initialization
        self.pruner = PRUNERS[self.pruner_name]
        example_inputs = [input_data.features[i][0][np.newaxis, :, :, :] for i in range(10)]
        example_inputs = torch.concat(example_inputs)
        targets = input_data.features[0][1]
        self.pruner = self.pruner(
            self.model,
            example_inputs,
            global_pruning=False,  # If False, a uniform ratio will be assigned to different layers.
            importance=self.importance,  # importance criterion for parameter selection
            iterative_steps=self.epochs,  # the number of iterations to achieve target ratio
            pruning_ratio=self.pruning_ratio,
            ignored_layers=self.ignored_layers,
        )

        base_macs, base_nparams = tp.utils.count_ops_and_params(self.model, example_inputs)
        for i in range(self.epochs):
            for group in self.pruner.step(interactive=True):
                print(group)
                dep, idxs = group[0]
                target_module = dep.target.module
                pruning_fn = dep.handler
                group.prune()
                _ = 1
            macs, nparams = tp.utils.count_ops_and_params(self.model, example_inputs)
        # for i in range(self.epochs):
        #     self.pruner.step()
        #     macs, nparams = tp.utils.count_ops_and_params(self.model, example_inputs)
        #     print("Params: %.2f M => %.2f M" % (base_nparams / 1e6, nparams / 1e6))
        #     print("MACs: %.2f G => %.2f G" % (base_macs / 1e9, macs / 1e9))
        #     _ = 1

        return self.model

    def predict(self,
                input_data: InputData, output_mode: str = 'default') -> np.array:
        return self.model if self.pruner is not None else self.predict_for_fit(input_data, output_mode)
