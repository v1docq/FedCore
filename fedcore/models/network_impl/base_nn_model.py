import os
from copy import deepcopy
from datetime import datetime
from functools import reduce, partial
from operator import iadd
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from pymonad.either import Either
from torch import Tensor
from torch.optim import lr_scheduler
from tqdm import tqdm

from fedcore.api.utils.data import DataLoaderHandler
from fedcore.data.data import CompressionInputData
from fedcore.losses.utils import _get_loss_metric
from fedcore.models.network_modules.layers.special import EarlyStopping
from fedcore.repository.constanst_repository import default_device, TorchLossesConstant
from fedcore.architecture.abstraction.accessor import Accessor


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
        self.custom_loss = self.params.get("custom_loss", None)  # loss which evaluates model structure
        self.loss = self.params.get('loss', None)
        self.enforced_training_loss = self.params.get("enforced_training_loss", None)
        self.device = self.params.get('device', default_device())
        self.model_params = self.params.get('model_params', {})
        self.learning_params = self.params.get('learning_params', {})

        # TODO move to learning or model params
        # self.is_operation = self.params.get('is_operation', False)  ###
        # self.save_each = self.params.get('save_each', None)
        # self.eval_each = self.params.get('eval_each', 5)
        # self.name = self.params.get('name', '')

    def _init_null_object(self):
        self.label_encoder = None
        self.is_regression_task = False
        self.model = None
        self.target = None
        self.task_type = None
        self.checkpoint_folder = self.params.get('checkpoint_folder', None)
        self.batch_limit = self.learning_params.get('batch_limit', None)
        self.calib_batch_limit = self.learning_params.get('calib_batch_limit', None)
        self.batch_type = self.learning_params.get('batch_type', None)
        self.train_loss_hist = []
        self.val_loss_hist = []

    def _init_model(self, dataloader):
        pass
    def get_loss(self):
        return TorchLossesConstant[self.loss].value()

    def get_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, dataloader, path: str):
        if self.model is None:
            self._init_model(dataloader)
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()

    def __check_and_substitute_loss(self, train_data: InputData):
        if (train_data.supplementary_data.col_type_ids is not None
                and train_data.supplementary_data.col_type_ids.get("loss", None)
        ):
            criterion = train_data.supplementary_data.col_type_ids["loss"]
            try:
                self.loss_fn = criterion()
            except:
                self.loss_fn = criterion
            print("Forcely substituted loss to", self.loss_fn)

    def __substitute_device_quant(self):
        if getattr(self.model, '_is_quantized', False):
            self.device = default_device('cpu')
            self.model.to(self.device)
            print('Quantized model inference supports CPU only')

    def _loss_callback(self, loss_fn, model_output, target):
        if custom_loss:
            model_loss = {key: val(model) for key, val in custom_loss.items()}
            model_loss["metric_loss"] = loss_fn(
                output, targets.to(self.device)
            )
            quality_loss = reduce(iadd, [loss for loss in model_loss.values()])
            loss_sum += model_loss["metric_loss"].item()
        else:
            model_loss = loss_fn(model_output, target)
        return model_loss

    def fit(self, input_data: InputData, supplementary_data: dict = None, loader_type='train'):
        self.custom_fit_process = supplementary_data is not None
        train_loader = getattr(input_data.features, f'{loader_type}_dataloader', 'train_dataloader')
        val_loader = getattr(input_data.features, 'calib_dataloader', None)
        loss_fn = self.get_loss()
        self.__check_and_substitute_loss(input_data)
        optimizer = self.get_optimizer()
        self.model = input_data.target if self.model is None else self.model
        self.optimised_model = self.model
        self.model.to(self.device)
        self._train_loop(
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer
        )

        self._clear_cache()
        return self.model

    @torch.no_grad()
    def _eval_one_epoch(self, epoch, dataloader, loss_fn, optimizer):

        training_loss = 0.0
        total_iterations = 0
        self.model.train()
        verified_dataloader = DataLoaderHandler.check_convert(dataloader=dataloader,
                                                              mode=self.batch_type,
                                                              max_batches=self.calib_batch_limit,
                                                              enumerate=False)
        for batch in tqdm(verified_dataloader, desc='Batch #'):
            total_iterations += 1
            inputs, targets = batch
            output = self.model(inputs.to(self.device))
            loss = self._loss_callback(loss_fn, output,
                                       targets.to(self.device))  # output [bs x last_hist_val x train_horizon]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            training_loss += loss.item()
        avg_loss = training_loss / total_iterations
        return optimizer, loss_fn, avg_loss

    def _train_loop(self, train_loader, val_loader, loss_fn, optimizer):
        # define custom fit logic after each training epoch
        fit_callback = Either(value=self.supplementary_data,
                              monoid=[self.custom_loss, self.custom_fit_process]).either(
            left_function=lambda custom_loss: partial(self._default_fit_callback, custom_loss),
            right_function=lambda sup_data: partial(self._custom_fit_callback, sup_data))
        losses = None
        train_loader = DataLoaderHandler.check_convert(dataloader=train_loader,
                                                       mode=self.batch_type,
                                                       max_batches=self.batch_limit,
                                                       enumerate=False)
        early_stopping = EarlyStopping(**self.learning_params['use_early_stopping'])
        scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
                                            steps_per_epoch=max(1, len(train_loader)),
                                            epochs=self.epochs,
                                            max_lr=self.learning_rate)
        for epoch in range(1, self.epochs + 1):
            optimizer, loss, train_loss, val_loss = self._eval_one_epoch(epoch, train_loader,
                                                                         val_loader, loss_fn,
                                                                         optimizer)
            scheduler.step()
            early_stopping(loss=train_loss)
            self.train_loss_hist.append(train_loss)
            self.val_loss_hist.append(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
            fit_callback()
        # for batch in tqdm(train_loader, desc='Batch #'):
        #     self.optimizer.zero_grad()
        #     total_iterations += 1
        #     inputs, targets = batch
        #     output = self.model(inputs.to(self.device))
        #     if custom_loss:
        #         model_loss = {key: val(model) for key, val in custom_loss.items()}
        #         model_loss["metric_loss"] = self.loss_fn(
        #             output, targets.to(self.device)
        #         )
        #         quality_loss = reduce(iadd, [loss for loss in model_loss.values()])
        #         loss_sum += model_loss["metric_loss"].item()
        #     else:
        #         quality_loss = self.loss_fn(output, targets.to(self.device))
        #         loss_sum += quality_loss.item()
        #         model_loss = quality_loss
        #     quality_loss.backward()
        #     self.optimizer.step()
        # avg_loss = loss_sum / total_iterations
        # if custom_loss:
        #     losses = reduce(iadd, list(model_loss.items()))
        #     losses = [x.item() if not isinstance(x, str) else x for x in losses]
        return losses, avg_loss

    def _custom_fit_callback(self, train_loader, model, callback: Callable, val_loader=None):
        # callback.callbacks.on_train_end()
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            model_loss, avg_loss = self._train_loop(train_loader, model)
            print("Epoch: {}, Average loss {}".format(epoch, avg_loss))
            if epoch % self.eval_each == 0 and val_loader is not None:
                print('Model Validation:', self._eval(self.model, val_loader, ))
            if epoch > 3:
                # Freeze quantizer parameters
                self.model.apply(torch.quantization.disable_observer)
            if epoch > 2:
                # Freeze batch norm mean and variance estimates
                self.model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            if self._check_saving(epoch):
                self.save_model(epoch, self.name)

    def _check_saving(self, epoch) -> bool:
        if not self.save_each:
            return False
        if self.save_each != -1:
            return not epoch % self.save_each
        else:
            return epoch == self.epochs

    def _default_fit_callback(self, model_loss: dict, current_epoch: int, custom_loss: dict = None, val_loader=None):

        if model_loss is not None:
            print(
                "Epoch: {}, Average loss {}, {}: {:.6f}, {}: {:.6f}, {}: {:.6f}".format(
                    current_epoch, avg_loss, *model_loss
                )
            )
        else:
            print("Epoch: {}, Average loss {}".format(current_epoch, avg_loss))
        if current_epoch % self.eval_each == 0 and val_loader is not None:
            print('Model Validation:', self._eval(self.model, val_loader, custom_loss))

    def save_model(self, epoch, name=''):
        name = name or self.params.get('name', '')
        path_pref = Path(self.checkpoint_folder)
        save_only = self.params.get('save_only', '')
        to_save = self.model if not save_only else Accessor.get_module(self.model, save_only)
        try:
            path = path_pref.joinpath(f"model_{name}{now_for_file()}_{epoch}.pth")
            torch.save(
                to_save,
                path,
            )
        except Exception as x:
            if os.path.exists(path):
                os.remove(path)
            print('Basic saving failed. Trying to use jit. \nReason: ', x.args[0])
            try:
                path = path_pref.joinpath(f"model_{name}{now_for_file()}_{epoch}_jit.pth")
                torch.jit.save(torch.jit.script(to_save), path)
            except Exception as x:
                if os.path.exists(path):
                    os.remove(path)
                print('JIT saving failed. saving weights only. \nReason: ', x.args[0])
                torch.save(to_save.state_dict(),
                           path_pref.joinpath(f"model_{name}{now_for_file()}_{epoch}_state.pth")
                           )

    def predict(self, input_data: InputData, output_mode: str = "default"):
        """
        Method for feature generation for all series
        """
        self.__substitute_device_quant()
        return self._predict_model(input_data.features, output_mode)

    def predict_for_fit(self, input_data: InputData, output_mode: str = "default"):
        """
        Method for feature generation for all series
        """
        self.__substitute_device_quant()
        return self._predict_model(input_data.features, output_mode)

    def _predict_model(
            self, x_test: CompressionInputData, output_mode: str = "default"
    ):
        assert type(x_test) is CompressionInputData
        # print('### IS_QUANTIZED', getattr(self.model, '_is_quantized', False))
        model: torch.nn.Module = self.model or x_test.target
        model.eval()
        prediction = []
        dataloader = DataLoaderHandler.check_convert(x_test.calib_dataloader,
                                                     mode=self.batch_type,
                                                     max_batches=self.calib_batch_limit)
        for batch in tqdm(dataloader):  ###TODO why calib_dataloader???
            inputs, targets = batch
            inputs = inputs.to(self.device)
            prediction.append(model(inputs))
        # print('### PREDICTION', prediction)
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

    @property
    def is_quantised(self):
        return getattr(self, '_is_quantised', False)
