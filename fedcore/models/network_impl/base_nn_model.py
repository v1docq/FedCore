import os
from copy import deepcopy
from datetime import datetime
from functools import reduce
from operator import iadd
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from pymonad.either import Either
from torch import Tensor
from tqdm import tqdm

from fedcore.api.utils.data import DataLoaderHandler
from fedcore.data.data import CompressionInputData
from fedcore.losses.utils import _get_loss_metric
from fedcore.repository.constanst_repository import default_device


def now_for_file():
    return datetime.now().strftime("%m-%d-%Y_%H-%M-%S")


class BaseNeuralModel:
    """Class responsible for NN model implementation.

    Attributes:
        self.num_features: int, the number of features.

    Example:
        To use this operation you can create pipeline as follows::
            from fedot.core.pipelines.pipeline_builder import PipelineBuilder
            from examples.fedot.fedot_ex import init_input_data
            from fedot_ind.tools.loader import DataLoader
            from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

            train_data, test_data = DataLoader(dataset_name='Ham').load_data()
            with IndustrialModels():
                pipeline = PipelineBuilder().add_node('resnet_model').add_node('rf').build()
                input_data = init_input_data(train_data[0], train_data[1])
                pipeline.fit(input_data)
                features = pipeline.predict(input_data)
                print(features)
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        self.params = params or {}
        self.epochs = self.params.get("epochs", 1)
        self.batch_size = self.params.get("batch_size", 16)
        self.learning_rate = self.params.get("learning_rate", 0.001)
        self.custom_loss = self.params.get(
            "custom_loss", None
        )  # loss which evaluates model structure
        self.enforced_training_loss = self.params.get("enforced_training_loss", None)
        self.device = default_device()

        self.is_operation = self.params.get('is_operation', False) ###
        self.save_each = self.params.get('save_each', None)
        self.checkpoint_folder = self.params.get('checkpoint_folder', None) ###
        self.batch_limit = self.params.get('batch_limit', None)
        self.calib_batch_limit = self.params.get('calib_batch_limit', None)
        self.batch_type = self.params.get('batch_type', None)

        self.label_encoder = None
        self.is_regression_task = False
        self.model = None
        self.target = None
        self.task_type = None

    def __check_and_substitute_loss(self, train_data: InputData):
        if (
            train_data.supplementary_data.col_type_ids is not None
            and "loss" in train_data.supplementary_data.col_type_ids
        ):
            self.loss_fn = train_data.supplementary_data.col_type_ids["loss"]()
            print("Forcely substituted loss to", self.loss_fn)

    def fit(self, input_data: InputData, supplementary_data: dict = None):
        custom_fit_process = supplementary_data is not None
        loader = input_data.features.train_dataloader

        self.loss_fn = _get_loss_metric(input_data)
        self.__check_and_substitute_loss(input_data)
        if self.model is None:
            self.model = input_data.target
        self.optimised_model = self.model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.model.to(self.device)

        fit_output = Either(
            value=supplementary_data, monoid=[self.custom_loss, custom_fit_process]
        ).either(
            left_function=lambda custom_loss: self._default_train(
                loader, self.model, custom_loss
            ),
            right_function=lambda sup_data: self._custom_train(
                loader, self.model, sup_data["callback"]
            ),
        )
        self._clear_cache()
        return self.model

    def _train_loop(self, train_loader, model, custom_loss: dict = None):
        loss_sum = 0
        total_iterations = 0
        losses = None
        train_loader = DataLoaderHandler.check_convert(dataloader=train_loader,
                                                       mode=self.batch_type,
                                                       max_batches=self.batch_limit,
                                                       enumerate=False)
        for batch in tqdm(train_loader, desc='Batch #'):
            self.optimizer.zero_grad()
            total_iterations += 1
            inputs, targets = batch
            output = self.model(inputs.to(self.device))
            if custom_loss:
                model_loss = {key: val(model) for key, val in custom_loss.items()}
                model_loss["metric_loss"] = self.loss_fn(
                    output, targets.to(self.device)
                )
                quality_loss = reduce(iadd, [loss for loss in model_loss.values()])
                loss_sum += model_loss["metric_loss"].item()
            else:
                quality_loss = self.loss_fn(output, targets.to(self.device))
                loss_sum += quality_loss.item()
                model_loss = quality_loss
            quality_loss.backward()
            self.optimizer.step()
        avg_loss = loss_sum / total_iterations
        if custom_loss:
            losses = reduce(iadd, list(model_loss.items()))
            losses = [x.item() if not isinstance(x, str) else x for x in losses]

        return losses, avg_loss

    def _custom_train(self, train_loader, model, callback: Callable):
        # callback.callbacks.on_train_end()
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            model_loss, avg_loss = self._train_loop(train_loader, model)
            print("Epoch: {}, Average loss {}".format(epoch, avg_loss))
            if epoch > 3:
                # Freeze quantizer parameters
                self.model.apply(torch.quantization.disable_observer)
            if epoch > 2:
                # Freeze batch norm mean and variance estimates
                self.model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            if self._check_saving(epoch):
                torch.save(
                    self.model,
                    Path(
                        self.checkpoint_folder,
                        f"model_train{now_for_file()}_{epoch}.pth",
                    ),
                )

        # callback.callbacks.on_train_end()

    def _check_saving(self, epoch) -> bool:
        if not self.save_each:
            return False
        if self.save_each != -1:
            return not epoch % self.save_each
        else:
            return epoch == self.epochs

    def _default_train(self, train_loader, model, custom_loss: dict = None):
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            model_loss, avg_loss = self._train_loop(train_loader, model, custom_loss)
            if model_loss is not None:
                print(
                    "Epoch: {}, Average loss {}, {}: {:.6f}, {}: {:.6f}, {}: {:.6f}".format(
                        epoch, avg_loss, *model_loss
                    )
                )
            else:
                print("Epoch: {}, Average loss {}".format(epoch, avg_loss))
            if self._check_saving(epoch):
                torch.save(
                    self.model,
                    Path(
                        self.checkpoint_folder,
                        f"model_train{now_for_file()}_{epoch}.pth",
                    ),
                )

    def predict(self, input_data: InputData, output_mode: str = "default"):
        """
        Method for feature generation for all series
        """
        return self._predict_model(input_data.features, output_mode)

    def predict_for_fit(self, input_data: InputData, output_mode: str = "default"):
        """
        Method for feature generation for all series
        """
        return self._predict_model(input_data.features, output_mode)

    def _predict_model(
        self, x_test: CompressionInputData, output_mode: str = "default"
    ):
        assert type(x_test) is CompressionInputData
        model = self.model or x_test.target
        model.eval()
        prediction = []
        dataloader = DataLoaderHandler.check_convert(x_test.calib_dataloader,
                                                     mode=self.batch_type,
                                                     max_batches=self.calib_batch_limit)
        for batch in tqdm(dataloader): ###TODO why calib_dataloader???
            inputs, targets = batch
            x_test = inputs.to(self.device)
            prediction.append(model(x_test))
        return self._convert_predict(torch.concat(prediction), output_mode)

    def _convert_predict(self, pred: Tensor, output_mode: str = "labels"):
        have_encoder = all([self.label_encoder is not None, output_mode == "labels"])
        output_is_clf_labels = all(
            [not self.is_regression_task, output_mode == "labels"]
        )

        pred = (
            pred.cpu().detach().numpy()
            if self.is_regression_task
            else F.softmax(pred, dim=1)
        )
        y_pred = (
            torch.argmax(pred, dim=1).cpu().detach().numpy()
            if output_is_clf_labels
            else pred
        )
        y_pred = (
            self.label_encoder.inverse_transform(y_pred) if have_encoder else y_pred
        )

        predict = OutputData(
            idx=np.arange(len(y_pred)),
            task=self.task_type,
            predict=y_pred,
            target=self.target,
            data_type=DataTypesEnum.table,
        )
        return predict

    def _clear_cache(self):
        with torch.no_grad():
            torch.cuda.empty_cache()

    @staticmethod
    def get_validation_frequency(epoch, lr):
        if epoch < 10:
            return 1  # Validate frequently in early epochs
        elif lr < 0.01:
            return 5  # Validate less frequently after learning rate decay
        else:
            return 2  # Default validation frequency
