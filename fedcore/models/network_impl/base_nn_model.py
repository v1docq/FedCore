import os
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from torch import Tensor
from tqdm import tqdm
from functools import reduce
from operator import iadd

from fedcore.losses.utils import _get_loss_metric
from fedcore.repository.constanst_repository import default_device


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
        self.epochs = self.params.get('epochs', 30)
        self.batch_size = self.params.get('batch_size', 16)
        self.learning_rate = self.params.get('learning_rate', 0.001)
        self.custom_loss = self.params.get('custom_loss', None)
        self.device = default_device()

        self.label_encoder = None
        self.model = None
        self.model_for_inference = None
        self.target = None
        self.task_type = None

    def fit(self, input_data: InputData):
        self.loss_fn = _get_loss_metric(input_data)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.to(self.device)
        self._train_loop(train_loader=input_data.features.train_dataloader,
                         model=self.model,
                         custom_loss=self.custom_loss
                         )
        # self._save_and_clear_cache()
        return self.model

    def _train_loop(self,
                    train_loader,
                    model,
                    total_iterations_limit=None,
                    custom_loss: dict = None):
        for epoch in range(1, self.epochs + 1):
            loss_sum = 0
            total_iterations = 0
            self.model.train()
            for batch in tqdm(train_loader):
                self.optimizer.zero_grad()
                total_iterations += 1
                inputs, targets = batch
                output = self.model(inputs.to(self.device))
                if custom_loss:
                    model_loss = {key: val(model) for key, val in custom_loss.items()}
                    model_loss['metric_loss'] = self.loss_fn(torch.argmax(output, dim=1).float(),
                                                             targets.to(self.device).float())
                    quality_loss = reduce(iadd, [loss for loss in model_loss.values()])
                    loss_sum += quality_loss.item()
                else:
                    quality_loss = self.loss_fn(output, targets)
                    loss_sum += quality_loss.item()
                quality_loss.backward()
                self.optimizer.step()
            avg_loss = loss_sum / total_iterations
            losses = reduce(iadd, list(model_loss.items()))
            losses = [x.item() / total_iterations if not isinstance(x, str) else x for x in losses]
            print('Epoch: {}, Average loss {}, {} Loss: {:.2f}, {} Loss: {:.2f}, {} Loss: {:.2f}'.format(
                epoch, avg_loss, *losses))

            if total_iterations_limit is not None and total_iterations >= total_iterations_limit:
                return

    def predict(
            self,
            input_data: InputData,
            output_mode: str = 'default'):
        """
        Method for feature generation for all series
        """
        return self._predict_model(input_data.features, output_mode)

    def predict_for_fit(
            self,
            input_data: InputData,
            output_mode: str = 'default'):
        """
        Method for feature generation for all series
        """
        return self._predict_model(input_data.features, output_mode)

    def _predict_model(self, x_test, output_mode: str = 'default'):
        self.model.eval()
        x_test = Tensor(x_test).to(self._device)
        pred = self.model(x_test)
        return self._convert_predict(pred, output_mode)

    def _convert_predict(self, pred, output_mode: str = 'labels'):
        have_encoder = all([self.label_encoder is not None, output_mode == 'labels'])
        output_is_clf_labels = all([not self.is_regression_task, output_mode == 'labels'])

        pred = pred.cpu().detach().numpy() if self.is_regression_task else F.softmax(pred, dim=1)
        y_pred = torch.argmax(pred, dim=1).cpu().detach().numpy() if output_is_clf_labels else pred
        y_pred = self.label_encoder.inverse_transform(y_pred) if have_encoder else y_pred

        predict = OutputData(
            idx=np.arange(len(y_pred)),
            task=self.task_type,
            predict=y_pred,
            target=self.target,
            data_type=DataTypesEnum.table)
        return predict

    def _save_and_clear_cache(self):
        prefix = f'model_{self.__repr__()}_activation_{self.activation}_epochs_{self.epochs}_bs_{self.batch_size}.pth'
        torch.save(self.model.state_dict(), prefix)
        del self.model
        with torch.no_grad():
            torch.cuda.empty_cache()
        self.model = self.model_for_inference.to(torch.device('cpu'))
        self.model.load_state_dict(torch.load(
            prefix, map_location=torch.device('cpu')))
        os.remove(prefix)

    @staticmethod
    def get_validation_frequency(epoch, lr):
        if epoch < 10:
            return 1  # Validate frequently in early epochs
        elif lr < 0.01:
            return 5  # Validate less frequently after learning rate decay
        else:
            return 2  # Default validation frequency
