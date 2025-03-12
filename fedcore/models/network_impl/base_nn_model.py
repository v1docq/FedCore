from datetime import datetime
from functools import reduce, partial
from operator import iadd
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from torch import Tensor
from tqdm import tqdm

from fedcore.api.utils.data import DataLoaderHandler
from fedcore.data.data import CompressionInputData
from fedcore.repository.constanst_repository import TorchLossesConstant, ModelLearningHooks, \
    LoggingHooks
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
        self.custom_loss = self.params.get("custom_loss", None)  # loss which evaluates model structure
        self.loss = self.params.get('loss', None)
        self.enforced_training_loss = self.params.get("enforced_training_loss", None)
        self.device = self.params.get('device', default_device())
        self.model_params = self.params.get('model_params', {})
        self.learning_params = self.params.get('custom_learning_params', {})
        self._init_empty_object()
        self._init_null_object()
        self._init_hook_params()
        self._init_hooks(hook_type='logging')

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

    def _init_empty_object(self):
        self.train_loss_hist = []
        self.val_loss_hist = []
        # add hooks
        self._on_epoch_end = []
        self._on_epoch_start = []
        self.hook_params = {}
        self.learning_hook_params = {}

    def _init_hook_params(self):
        self.hook_params = {'is_operation': self.params.get('is_operation', False),
                            'save_each': self.params.get('save_each', None),
                            'eval_each': self.params.get('eval_each', 5),
                            'log_each': self.params.get('log_each', 5),
                            'n_plateau': self.params.get('n_plateau', None),
                            'use_scheduler': self.learning_params.get('use_early_stopping', None),
                            'name': ''}

    def _init_model(self):
        pass

    def _init_hooks(self, hook_type: str = 'logging'):
        hook_dict = {'logging': LoggingHooks,
                     'learning_control': ModelLearningHooks}
        for hook_elem in hook_dict[hook_type]:
            hook = hook_elem.value
            if self.hook_params[hook._SUMMON_KEY] is None:
                continue
            if hook_type.__contains__('logging'):
                hook = hook(self.hook_params, self.model)
            else:
                hook = hook(self.learning_hook_params, self.model)
            if hook._hook_place == 'post':
                self._on_epoch_end.append(hook)
            else:
                self._on_epoch_start.append(hook)

    def get_loss(self):
        return TorchLossesConstant[self.loss].value()

    def get_optimizer(self):
        optimizer = self.learning_params.get('optimizer', partial(torch.optim.Adam, lr=self.learning_rate))
        return optimizer(self.model.parameters())

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        if self.model is None:
            self._init_model()
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
        if self.custom_loss:
            model_loss = {key: val(self.model) for key, val in self.custom_loss.items()}
            model_loss["metric_loss"] = loss_fn(model_output, target)
            quality_loss = reduce(iadd, [loss for loss in model_loss.values()])
            model_loss += model_loss["metric_loss"].item()
        else:
            model_loss = loss_fn(model_output, target)
        return model_loss

    def fit(self, input_data: InputData, supplementary_data: dict = None, loader_type='train'):
        # define data for fit process
        self.custom_fit_process = supplementary_data is not None
        train_loader = getattr(input_data.features, f'{loader_type}_dataloader', 'train_dataloader')
        val_loader = getattr(input_data.features, 'calib_dataloader', None)
        # define model for fit process
        self.model = input_data.target if self.model is None else self.model
        self.optimised_model = self.model
        self.model.to(self.device)
        # define loss and optimizer for fit process
        loss_fn = self.get_loss()
        self.__check_and_substitute_loss(input_data)
        optimizer = self.get_optimizer()
        self.learning_hook_params.update({'optimizer': optimizer, 'learning_params': self.learning_params,
                                          'epochs': self.epochs, 'learning_rate': self.learning_rate,
                                          'train_loader': train_loader})
        self._init_hooks(hook_type='learning_control')
        self._train_loop(
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer
        )

        self._clear_cache()
        return self.model

    def _eval_one_epoch(self, epoch, dataloader, loss_fn, optimizer):

        training_loss = 0.0
        total_iterations = 0
        self.model.train()
        for batch in tqdm(dataloader, desc='Batch #'):
            total_iterations += 1
            inputs, targets = batch
            output = self.model(inputs.to(self.device))
            loss = self._loss_callback(loss_fn, output,
                                       targets.to(self.device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            training_loss += loss.item()
        avg_loss = training_loss / total_iterations
        return optimizer, loss_fn, avg_loss

    def _train_loop(self, train_loader, val_loader, loss_fn, optimizer):
        train_loader = DataLoaderHandler.check_convert(dataloader=train_loader,
                                                       mode=self.batch_type,
                                                       max_batches=self.batch_limit,
                                                       enumerate=False)
        for epoch in range(1, self.epochs + 1):
            for hook in self._on_epoch_start:
                hook(epoch=epoch)
            optimizer, loss, train_loss = self._eval_one_epoch(epoch=epoch,
                                                               dataloader=train_loader,
                                                               loss_fn=loss_fn,
                                                               optimizer=optimizer)

            self.train_loss_hist.append(train_loss)
            for hook in self._on_epoch_end:
                hook(epoch=epoch, val_loader=val_loader, custom_loss=self.custom_loss, train_loss=train_loss)
        return self

    # def _custom_fit_callback(self, train_loader, model, callback: Callable, val_loader=None):
    #     # callback.callbacks.on_train_end()
    #     for epoch in range(1, self.epochs + 1):
    #         self.model.train()
    #         model_loss, avg_loss = self._train_loop(train_loader, model)
    #         print("Epoch: {}, Average loss {}".format(epoch, avg_loss))
    #         if epoch % self.eval_each == 0 and val_loader is not None:
    #             print('Model Validation:', self._eval(self.model, val_loader, ))
    #         if epoch > 3:
    #             # Freeze quantizer parameters
    #             self.model.apply(torch.quantization.disable_observer)
    #         if epoch > 2:
    #             # Freeze batch norm mean and variance estimates
    #             self.model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    #

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


class BaseNeuralForecaster(BaseNeuralModel):
    """Class responsible for NN model implementation.

    Attributes:
        self.num_features: int, the number of features.

    Example:
        To use this operation you can create pipeline as follows::
            from fedot.core.pipelines.pipeline_builder import PipelineBuilder
            from fedot_ind.tools.loader import DataLoader
            from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

            train_data, test_data = DataLoader(dataset_name='Ham').load_data()
            with IndustrialModels():
                pipeline = PipelineBuilder().add_node('resnet_model').add_node('rf').build()
                pipeline.fit(input_data)
                features = pipeline.predict(input_data)
                print(features)
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.train_horizon = self.params.get('train_horizon', 1)
        self.test_horizon = self.params.get('test_horizon', 1)
        self.in_sample_regime = self.params.get('use_in_sample', True)
        self.use_exog_features = self.params.get('use_exog_features', False)
        self.forecasting_blocks = int(self.test_horizon / self.train_horizon)
        self.val_interval = 5
        self.device = default_device()

    def out_of_sample_predict(self, tensor_endogen: Tensor, tensor_exogen: Tensor, target: Tensor):
        pred = self.model(x=tensor_endogen, mask=None)  # output [bs x seq_len x horizon]
        predict = pred[:, -1, :][:, None, :]  # take predict from last point as output [bs x 1 x train_horizon]
        return predict

    def create_features_from_predict(self, tensor_endogen: Tensor, tensor_exogen: Tensor, predict: Tensor):
        features_from_predict = torch.concat([predict, tensor_exogen], dim=1)
        tensor_endogen = tensor_endogen[:, :, self.train_horizon:]
        tensor_endogen = torch.concat([tensor_endogen, features_from_predict], dim=2)
        return tensor_endogen

    def in_sample_predict(self, tensor_endogen: Tensor, tensor_exogen: Tensor, target: Tensor):
        all_predict = []
        new_tensor_endogen = tensor_endogen
        for block in range(self.forecasting_blocks):
            start_idx, end_idx = block * self.train_horizon, block * self.train_horizon + self.train_horizon
            exog_feature = tensor_exogen[:, :, start_idx:end_idx]
            horizon_pred = self.out_of_sample_predict(new_tensor_endogen, exog_feature,
                                                      target)  # output [bs x 1 x train_horizon]
            all_predict.append(horizon_pred)
            new_tensor_endogen = self.create_features_from_predict(new_tensor_endogen, exog_feature, horizon_pred)
        in_sample_predict = torch.concat(all_predict, dim=2)  # output [bs x 1 x test_horizon]
        return in_sample_predict

    def _eval_one_epoch(self, epoch, dataloader, loss_fn, optimizer):
        training_loss = 0.0
        self.model.train()
        for batch in dataloader:
            x_hist, x_fut, y = [b.to(self.device) for b in batch]
            predict = self.out_of_sample_predict(x_hist, x_fut, y)
            loss = loss_fn(predict, y)  # output [bs x last_hist_val x train_horizon]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            training_loss += loss.item()
        return optimizer, loss_fn, training_loss

    def _predict_model(self, test_loader, output_mode: str = 'default'):
        self.model.eval()

        def predict_loop(batch):
            x_hist, x_fut, y = [b.to(self.device).transpose(2, 1) for b in batch]
            if self.in_sample_regime:
                predict = self.in_sample_predict(x_hist, x_fut, y)
            else:
                predict = self.out_of_sample_predict(x_hist, x_fut, y)
            predict = predict.cpu().detach().numpy().squeeze()
            target = y.cpu().detach().numpy().squeeze()
            return predict, target

        prediction = list(map(lambda batch: predict_loop(batch), test_loader))
        all_prediction = np.concatenate([x[0] for x in prediction])
        all_target = np.concatenate([x[1] for x in prediction])
        return all_prediction, all_target
