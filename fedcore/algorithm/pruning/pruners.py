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
    PRUNING_FUNC
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
        self.trainer = BaseNeuralModel(self.ft_params)

    def __repr__(self):
        return self.pruner_name

    def _check_model_before_prune(self, input_data):
        # list of tensors with dim size n_samples x n_channel x height x width
        batch_generator = (b for b in input_data.features.calib_dataloader)
        # take first batch
        batch_list = next(batch_generator)
        self.data_batch_for_calib = batch_list[0].to(default_device())
        n_classes = input_data.task.task_params['forecast_length'] \
            if input_data.task.task_type.value.__contains__('forecasting') else input_data.features.num_classes
        self.validator = PruningValidator(model=self.model_after_pruning,
                                          output_dim=n_classes, input_dim=input_data.features.input_dim)
        self.ignored_layers = self.validator.filter_ignored_layers(self.model_after_pruning,
                                                                   str(self.model_after_pruning.__class__))
        self.channel_groups = self.validator.validate_channel_groups()

    def _init_model(self, input_data):
        print("==============Prepare original model for pruning=================")
        self.model_before_pruning = input_data.target
        if hasattr(self.model_before_pruning, 'model'):
            self.trainer = self.model_before_pruning
            self.model_before_pruning = self.model_before_pruning.model
        else:
            self.trainer.model = self.model_before_pruning
        self.model_before_pruning.to(default_device())
        self.model_after_pruning = deepcopy(self.model_before_pruning)

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
        self._check_model_before_prune(input_data)
        self.optimizer_for_grad = optim.Adam(self.model_after_pruning.parameters(), lr=self.learning_rate_for_grad)

    def _accumulate_grads(self, data, target, return_false=False):
        data, target = data.to(default_device()), target.to(default_device())
        out = self.model_after_pruning(data)
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
                pruner.regularize(self.model_after_pruning, loss)  # <== for sparse training
                self.optimizer_for_grad.step()
        return pruner

    # def _dependency_graph_pruner(self):
    #     DG = tp.DependencyGraph().build_dependency(
    #         self.model_after_pruning,
    #         example_inputs=self.data_batch_for_calib,
    #         ignored_layers=self.ignored_layers,
    #     )
    #     group = DG.get_pruning_group(
    #         self.optimised_model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9]
    #     )
    #
    #     print(group.details())
    #
    #     if DG.check_pruning_group(group):  # avoid over-pruning, i.e., channels=0.
    #         group.prune()

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
        self.prune(input_data=input_data)
        return self.model_after_pruning

    def _manual_pruning_iter(self, pruner):
        # for i in range(self.pruning_iterations):
        #     for pruning_func in MANUAL_PRUNING_STRATEGY[self.pruner_name]:
        #         root_layer = self._define_root_layer()
        #         pruner.manual_prune(
        #             root_layer, PRUNING_FUNC[pruning_func], self.pruning_ratio
        #         )
        pass

    def _default_pruning_iter(self, pruner):
        pruning_hist = []
        for i in range(self.pruning_iterations):
            potential_groups_to_prune = list(pruner.step(interactive=True))
            for group in potential_groups_to_prune:
                dep, idxs = group[0]
                layer = dep.layer
                pruning_fn = dep.pruning_fn
                valid_to_prune = self.validator.validate_pruned_layers(layer, pruning_fn)
                if valid_to_prune:
                    pruning_hist.append((layer, idxs, pruning_fn))
                    group.prune()
                else:
                    continue

    def prune(self, input_data: InputData) -> np.array:
        self.pruner = self.pruner(
            self.model_after_pruning,
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
            self.model_before_pruning, self.data_batch_for_calib
        )
        manual_pruning = self.pruner_name.__contains__("manual")

        self.pruner = self.pruner_step(self.pruner, input_data)
        self.pruner = Either(
            value=self.pruner, monoid=[self.pruner, manual_pruning]
        ).either(
            left_function=lambda pruner_agent: self._default_pruning_iter(pruner_agent),
            right_function=lambda pruner_agent: self._manual_pruning_iter(pruner_agent),
        )
        print("==============After pruning=================")
        macs, nparams = tp.utils.count_ops_and_params(
            self.model_after_pruning, self.data_batch_for_calib
        )
        print("Params: %.6f M => %.6f M" % (base_nparams / 1e6, nparams / 1e6))
        print("MACs: %.6f G => %.6f G" % (base_macs / 1e9, macs / 1e9))

        print("==============Finetune pruned model=================")
        self.trainer.model = self.model_after_pruning
        # self.trainer.custom_loss = self.ft_params["custom_loss"]
        # self.trainer.epochs = self.ft_params["epochs"]
        self.model_after_pruning = self.trainer.fit(input_data)

        return self.model_after_pruning

    def predict_for_fit(self, input_data: InputData, output_mode: str = "compress"):
        return self.model_after_pruning if output_mode == "compress" else self.model

    def predict(self, input_data: InputData, output_mode: str = "compress"):
        if output_mode == "compress":
            self.trainer.model = self.model_after_pruning
        else:
            self.trainer.model = self.model_before_pruning
        return self.trainer.predict(input_data, output_mode)
