from copy import deepcopy

import numpy as np
from fedot.core.data.data import InputData
from pymonad.either import Either
from torch import nn, optim

from fedcore.algorithm.base_compression_model import BaseCompressionModel
import torch_pruning as tp
from typing import Optional
from fedot.core.operations.operation_parameters import OperationParameters

from fedcore.algorithm.pruning.pruning_validation import PruningValidator
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
from fedcore.repository.constanst_repository import (
    PRUNERS,
    PRUNING_IMPORTANCE,
    PRUNER_REQUIRED_REG,
    PRUNER_WITHOUT_REQUIREMENTS,
    PRUNING_FUNC,
    MANUAL_PRUNING_STRATEGY,
)


class BasePruner(BaseCompressionModel):
    """Class responsible for Pruning model implementation.
    Example:
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)
        # finetune params
        self.epochs = params.get("epochs", 5)
        self.ft_params = params.get("finetune_params", None)
        if self.ft_params is None:
            self.ft_params = {}
            self.ft_params["custom_loss"] = None
            self.ft_params["epochs"] = round(self.epochs / 3)
        self.criterion = params.get("loss", nn.CrossEntropyLoss())
        self.optimizer = params.get("optimizer", optim.Adam)
        self.learning_rate = params.get("lr", 0.001)

        # pruning gradients params
        self.criterion_for_grad = params.get("loss", nn.CrossEntropyLoss())
        self.learning_rate_for_grad = params.get("lr", 0.001)

        # pruning params
        self.pruner_name = params.get("pruner_name", "meta_pruner")
        self.importance_name = params.get("importance", "MagnitudeImportance")

        # pruning hyperparams
        self.pruning_ratio = params.get("pruning_ratio", 0.5)
        self.pruning_iterations = params.get("pruning_iterations", 2)
        self.importance_norm = params.get("importance_norm", 1)
        self.importance_reduction = params.get("importance_reduction", "mean")
        self.importance_normalize = params.get("importance_normalize", "mean")

        # importance criterion for parameter selections
        self.importance = PRUNING_IMPORTANCE[self.importance_name](
            group_reduction=self.importance_reduction,
            normalizer=self.importance_normalize,
        )
        self.trainer = BaseNeuralModel(params)

    def __repr__(self):
        return self.pruner_name

    def _init_model(self, input_data):
        print("==============Prepare original model for pruning=================")
        self.model = input_data.target
        self.trainer.model = self.model
        print(
            f"==============Initialisation of {self.pruner_name} pruning agent================="
        )
        print(
            f"==============Pruning importance - {self.importance_name} ================="
        )
        print(f"==============Pruning ratio -  {self.pruning_ratio} =================")
        print(
            f"==============Pruning importance norm -  {self.importance_norm} ================="
        )
        # Pruner initialization
        self.pruner = PRUNERS[self.pruner_name]
        # list of tensors with dim size n_samples x n_channel x height x width
        batch_generator = (b for b in input_data.features.calib_dataloader)
        # take first batch
        input, target = next(batch_generator)
        self.data_batch_for_calib = input.to(self.trainer.device)
        self.validator = PruningValidator(self.model, input_data.features.num_classes)
        self.ignored_layers = self.validator.filter_ignored_layers(
            self.model, str(self.model.__class__)
        )
        self.channel_groups = self.validator.validate_channel_groups()
        self.optimizer_for_grad = optim.Adam(
            self.model.parameters(), lr=self.learning_rate_for_grad
        )

    def _accumulate_grads(self, data, target, return_false=False):
        data, target = data.to(default_device()), target.to(default_device())
        out = self.model(data)
        loss = self.criterion_for_grad(out, target)
        loss.backward()
        if return_false:
            return loss

    def _define_root_layer(self):
        root_layer_dict = {
            "manual_conv": self.optimised_model.conv1,
        }

        return root_layer_dict[self.pruner_name]

    def _pruner_step_with_grads(self, pruner, input_data):
        for i, (data, target) in enumerate(input_data.features.calib_dataloader):
            if i != 0:
                print(f"Gradients accumulation iter- {i}")
                print(f"==========================================")
                # we using 1 batch as example of pruning quality
                self._accumulate_grads(data, target)
        return pruner

    def _pruner_step_with_reg(self, pruner, input_data):
        pruner.update_regularizer()
        # <== initialize regularizer
        for i, (data, target) in enumerate(input_data.features.calib_dataloader):
            if i != 0:
                print(f"Pruning reg iter- {i}")
                print(f"==========================================")
                # we using 1 batch as example of pruning quality
                self.optimizer_for_grad.zero_grad()
                loss = self._accumulate_grads(
                    data, target, True
                )  # after loss.backward()
                pruner.regularize(self.optimised_model, loss)  # <== for sparse training
                self.optimizer_for_grad.step()
        return pruner

    def _dependency_graph_pruner(self):
        DG = tp.DependencyGraph().build_dependency(
            self.optimised_model,
            example_inputs=self.data_batch_for_calib,
            ignored_layers=self.ignored_layers,
        )
        group = DG.get_pruning_group(
            self.optimised_model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9]
        )

        print(group.details())

        if DG.check_pruning_group(group):  # avoid over-pruning, i.e., channels=0.
            group.prune()

    def pruner_step(self, pruner: callable, input_data):

        pruner_without_grads = isinstance(
            self.importance, tuple(PRUNER_WITHOUT_REQUIREMENTS.values())
        )
        pruner_with_reg = isinstance(
            self.importance, tuple(PRUNER_REQUIRED_REG.values())
        )
        pruner_with_grads = not all([pruner_with_reg, pruner_without_grads])

        pruner = Either(value=pruner, monoid=[input_data, pruner_without_grads]).either(
            left_function=lambda data: (
                self._pruner_step_with_grads(pruner, data)
                if pruner_with_grads
                else self._pruner_step_with_reg(pruner, data)
            ),
            right_function=lambda pruner_agent: pruner_agent,
        )
        return pruner

    def fit(self, input_data: InputData):
        self._init_model(input_data)
        if input_data.target.training:
            self.model = input_data.target
        else:
            self.model = self.trainer.fit(input_data)
        self.optimised_model = deepcopy(self.model)
        self.compress(input_data=input_data)
        return self.optimised_model

    def _manual_pruning_iter(self, pruner):
        for i in range(self.pruning_iterations):
            for pruning_func in MANUAL_PRUNING_STRATEGY[self.pruner_name]:
                root_layer = self._define_root_layer()
                pruner.manual_prune(
                    root_layer, PRUNING_FUNC[pruning_func], self.pruning_ratio
                )

    def _default_pruning_iter(self, pruner):
        pruning_hist = []
        for i in range(self.pruning_iterations):
            for group in pruner.step(interactive=True):
                dep, idxs = group[0]
                layer = dep.layer
                pruning_fn = dep.pruning_fn
                valid_to_prune = self.validator.validate_pruned_layers(
                    layer, pruning_fn
                )
                if valid_to_prune:
                    pruning_hist.append((layer, idxs, pruning_fn))
                    group.prune()
                else:
                    continue

    def compress(self, input_data: InputData) -> np.array:
        self.pruner = self.pruner(
            self.optimised_model,
            self.data_batch_for_calib,
            # global_pruning=False,  # If False, a uniform ratio will be assigned to different layers.
            importance=self.importance,  # importance criterion for parameter selection
            iterative_steps=self.pruning_iterations,  # the number of iterations to achieve target ratio
            pruning_ratio=self.pruning_ratio,
            ignored_layers=self.ignored_layers,
            channel_groups=self.channel_groups,
            round_to=None,
            unwrapped_parameters=None,
        )

        base_macs, base_nparams = tp.utils.count_ops_and_params(
            self.model, self.data_batch_for_calib
        )
        manual_pruning = self.pruner_name.__contains__("manual")

        self.pruner = self.pruner_step(self.pruner, input_data)
        self.pruner = Either(
            value=self.pruner, monoid=[self.pruner, manual_pruning]
        ).either(
            left_function=lambda pruner_agent: self._default_pruning_iter(pruner_agent),
            right_function=lambda pruner_agent: self._manual_pruning_iter(pruner_agent),
        )

        print("==============Finetune pruned model=================")
        self.trainer.model = self.optimised_model
        self.trainer.custom_loss = self.ft_params["custom_loss"]
        self.trainer.epochs = self.ft_params["epochs"]
        self.optimised_model = self.trainer.fit(input_data)

        print("==============After pruning=================")
        macs, nparams = tp.utils.count_ops_and_params(
            self.optimised_model, self.data_batch_for_calib
        )
        print("Params: %.2f M => %.2f M" % (base_nparams / 1e6, nparams / 1e6))
        print("MACs: %.2f G => %.2f G" % (base_macs / 1e9, macs / 1e9))
        return self.model

    def predict_for_fit(self, input_data: InputData, output_mode: str = "compress"):
        return self.optimised_model if output_mode == "compress" else self.model

    def predict(self, input_data: InputData, output_mode: str = "compress"):
        self.trainer.model = (
            self.optimised_model if output_mode == "compress" else self.model
        )
        return self.trainer.predict(input_data, output_mode)
