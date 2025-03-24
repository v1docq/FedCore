from datetime import datetime
from enum import Enum
from functools import reduce
from itertools import chain
from typing import Iterable, Optional, Any, Union
from pymonad.maybe import Maybe
import numpy as np
import torch
import torch.nn.functional as F
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from torch import Tensor
from tqdm import tqdm

from fedcore.api.utils.data import DataLoaderHandler
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.data.data import CompressionInputData
from fedcore.repository.constanst_repository import (
    ModelLearningHooks,
    LoggingHooks,
    StructureCriterions,
    TorchLossesConstant,
)

from fedcore.models.network_impl.hooks import BaseHook


def now_for_file():
    return datetime.now().strftime("%m-%d-%Y_%H-%M-%S")


class BaseNeuralModel(torch.nn.Module):
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
        super().__init__()
        self.params = params or {}
        self.learning_params = self.params.get('custom_learning_params', {})
        self._init_empty_object()
        self._init_null_object()
        self.epochs = self.params.get("epochs", 1)
        self.batch_size = self.params.get("batch_size", 16)
        self.learning_rate = self.params.get("learning_rate", 0.001)
        self._init_custom_criterions(
            self.params.get("custom_criterions", {}))  # let it be dict[name : coef], let nodes add it to trainer
        self.criterion = self.__get_criterion()
        self.device = self.params.get('device', default_device())
        self.model_params = self.params.get('model_params', {})
        self._hooks = [LoggingHooks, ModelLearningHooks]

    def _init_custom_criterions(self, custom_criterions: dict):
        for name, coef in custom_criterions.items():
            if hasattr(StructureCriterions, name):
                criterion = StructureCriterions[name].value
            elif hasattr(TorchLossesConstant, name):
                criterion = TorchLossesConstant[name].value
            else:
                raise ValueError(f'Unknown type `{name}` of custom loss')
            self.history[f'train_{name}_loss'] = []
            self.history[f'val_{name}_loss'] = []
            custom_criterions[name] = (criterion(), coef)
        self.custom_criterions = custom_criterions

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
        self.trainer_objects = {
            'optimizer': None,
            'scheduler': None,
        }

    def _init_empty_object(self):
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
        # add hooks
        self._on_epoch_end = []
        self._on_epoch_start = []

    def __repr__(self):
        return self.__class__.__name__ + '\nStart hooks:\n' + '\n'.join(
            str(hook) for hook in self._on_epoch_start
        ) + '\nEnd hooks:\n' + '\n'.join(
            str(hook) for hook in self._on_epoch_end
        )

    def clear_hooks(self):
        self._hooks.clear()
        self._on_epoch_end.clear()
        self._on_epoch_start.clear()

    @classmethod
    def wrap(cls, model, params: Optional[OperationParameters] = None):
        bnn = cls(params)
        bnn.clear_hooks()
        bnn.model = model
        return bnn

    def _init_model(self):
        pass

    def _init_hooks(self):
        for hook_elem in chain(*self._hooks):
            hook: BaseHook = hook_elem.value
            if not hook.check_init(self.params):
                continue
            hook = hook(self.params, self.model)
            if hook._hook_place == 'post':
                self._on_epoch_end.append(hook)
            else:
                self._on_epoch_start.append(hook)

    def register_additional_hooks(self, hooks: Iterable[Enum]):
        self._hooks.extend(hooks)

    def __get_criterion(self):
        key = self.params.get('loss', None) or self.params.get('criterion', None)
        if hasattr(TorchLossesConstant, key):
            return TorchLossesConstant[key].value()
        if hasattr(key, '__call__'):
            return key
        raise ValueError('No loss specified!')

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        if self.model is None:
            self._init_model()
        try:  # path to state_dict
            self.model.load_state_dict(torch.load(path, weights_only=False))
        except Exception:  # path to model_impl
            self.model = torch.load(path, map_location=self.device)
        self.model.eval()

        # add hooks
        self._on_epoch_end = []
        self._on_epoch_start = []

    def __check_and_substitute_loss(self, train_data: InputData):
        # TODO delete 
        names = ['loss', 'criterion']
        for name in names:
            if (train_data.supplementary_data.col_type_ids is not None
                    and train_data.supplementary_data.col_type_ids.get(name, None)
            ):
                criterion = train_data.supplementary_data.col_type_ids[name]
                try:
                    self.criterion = criterion()
                except:
                    self.criterion = criterion
                print("Forcely substituted criterion[loss] to", self.criterion)

    def __substitute_device_quant(self):
        if getattr(self.model, '_is_quantized', False):
            self.device = default_device('cpu')
            self.model.to(self.device)
            print('Quantized model inference supports CPU only')

    def _compute_loss(self, criterion, model_output, target, stage='train', epoch=None):
        if hasattr(model_output, 'loss'):
            quality_loss = model_output.loss
        else:
            quality_loss = criterion(model_output, target)
        if isinstance(model_output, torch.Tensor):
            additional_losses = {name: coef * criterion(model_output, target)
                                 for name, (criterion, coef) in self.custom_criterions.items()
                                 if hasattr(TorchLossesConstant, name)}
            additional_losses.update({name: coef * criterion(self.model)
                                      for name, (criterion, coef) in self.custom_criterions.items()
                                      if hasattr(StructureCriterions, name)})
            for name, val in additional_losses.items():
                self.history[f'{stage}_{name}_loss'].append((epoch, val))
        final_loss = reduce(torch.add, additional_losses.values(), quality_loss)
        return final_loss

    def fit(self, input_data: InputData, supplementary_data: dict = None, loader_type='train'):
        # define data for fit process
        self.custom_fit_process = supplementary_data is not None
        train_loader = getattr(input_data.features, f'{loader_type}_dataloader', 'train_dataloader')
        val_loader = getattr(input_data.features, 'val_dataloader', None)
        self.task_type = input_data.task.task_type
        # define model for fit process
        self.model = input_data.target if self.model is None else self.model
        self.optimised_model = self.model
        self.model.to(self.device)
        self.__check_and_substitute_loss(input_data)
        self._init_hooks()
        self._train_loop(
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=self.criterion,
        )
        self._clear_cache()
        return self.model

    def _run_one_epoch(self, epoch, dataloader, loss_fn, optimizer):
        training_loss = 0.0
        self.model.train()
        for batch in tqdm(dataloader, desc='Batch #'):
            *inputs, targets = batch
            inputs = tuple(inputs_.to(self.device) for inputs_ in inputs if hasattr(inputs_, 'to'))
            output = self.model(*inputs)
            loss = self._compute_loss(loss_fn, output,
                                      targets.to(self.device), epoch=epoch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            training_loss += loss.item()
        avg_loss = training_loss / len(dataloader)
        self.history['train_loss'].append((epoch, avg_loss))  # changed to match epoch and loss

    def _train_loop(self, train_loader, val_loader, loss_fn):
        train_loader = DataLoaderHandler.check_convert(dataloader=train_loader,
                                                       mode=self.batch_type,
                                                       max_batches=self.batch_limit,
                                                       enumerate=False)
        for epoch in range(1, self.epochs + 1):
            for hook in self._on_epoch_start:
                hook(epoch=epoch, trainer_objects=self.trainer_objects,
                     learning_rate=self.learning_rate)
            self._run_one_epoch(epoch=epoch,
                                dataloader=train_loader,
                                loss_fn=loss_fn,
                                optimizer=self.optimizer)
            from functools import partial
            for hook in self._on_epoch_end:
                hook(epoch=epoch, val_loader=val_loader,
                     criterion=partial(self._compute_loss, criterion=loss_fn),
                     history=self.history)
        return self

    def predict(self, input_data: InputData, output_mode: str = "default"):
        """
        Method for feature generation for all series
        """
        print('###', 'BNN predict')
        self.__substitute_device_quant()
        return self._predict_model(input_data, output_mode)

    def predict_for_fit(self, input_data: InputData, output_mode: str = "default"):
        """
        Method for feature generation for all series
        """
        print('###', 'BNN predict4fit')

        self.__substitute_device_quant()
        return self._predict_model(input_data.features, output_mode)

    def _predict_model(
            self, x_test: Union[CompressionInputData, InputData], output_mode: str = "default"
    ):
        model: torch.nn.Module = self.model or x_test.target
        model.eval()
        prediction = []
        dataloader = DataLoaderHandler.check_convert(x_test.val_dataloader,
                                                     mode=self.batch_type,
                                                     max_batches=self.calib_batch_limit)
        for batch in tqdm(dataloader):  ###TODO why val_dataloader???
            *inputs, targets = batch
            inputs = tuple(inputs_.to(self.device) for inputs_ in inputs if hasattr(inputs_, 'to'))
            prediction.append(model(*inputs))
        return self._convert_predict(torch.concat(prediction), output_mode)

    def _convert_predict(self, pred: Tensor, output_mode: str = "labels"):
        have_encoder = all([self.label_encoder is not None, output_mode == "labels"])
        if self.task_type.name == 'regression':
            self.is_regression_task = True
        output_is_clf_labels = all(
            [not self.is_regression_task, output_mode == "labels"]
        )
        pred = Maybe.insert(pred). \
            then(lambda predict: predict.cpu().detach().numpy() if self.is_regression_task else F.softmax(predict, dim=1)). \
            then(lambda predict: torch.argmax(predict, dim=1).cpu().detach().numpy() if output_is_clf_labels else predict). \
            then(lambda predict: self.label_encoder.inverse_transform(predict) if have_encoder else predict). \
            maybe(None, lambda output: output)

        predict = OutputData(
            idx=np.arange(len(pred)),
            task=self.task_type,
            predict=pred,
            target=self.target,
            data_type=DataTypesEnum.table,
        )
        return predict

    def _clear_cache(self):
        with torch.no_grad():
            torch.cuda.empty_cache()

    def __wrap(self, model):
        if not isinstance(model, BaseNeuralModel):
            return BaseNeuralModel.wrap(model, self.params)
        return model

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

    @property
    def optimizer(self):
        return self.trainer_objects['optimizer']

    @property
    def scheduler(self):
        return self.trainer_objects['scheduler']


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
        self.loss = self.params.get('loss', 'smape')
        self.val_interval = 5

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

    def _run_one_epoch(self, epoch, dataloader, loss_fn, optimizer):
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

    def _predict_model(self, input_data: CompressionInputData, output_mode: str = 'default'):
        self.model.eval()

        def predict_loop(batch):
            x_hist, x_fut, y = [b.to(self.device) for b in batch]
            if self.in_sample_regime:
                predict = self.in_sample_predict(x_hist, x_fut, y)
            else:
                predict = self.out_of_sample_predict(x_hist, x_fut, y)
            predict = predict.cpu().detach().numpy().squeeze()
            target = y.cpu().detach().numpy().squeeze()
            return predict, target

        prediction = list(map(lambda batch: predict_loop(batch), input_data.test_dataloader))
        all_prediction = np.concatenate([x[0] for x in prediction])
        # all_target = np.concatenate([x[1] for x in prediction])
        return all_prediction
