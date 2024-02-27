import numpy as np
from fedot.core.data.data import InputData
from torch import optim, nn

from fedcore.algorithm.base_compression_model import BaseCompressionModel
from typing import Optional
import torch
from fedot.core.operations.operation_parameters import OperationParameters

from fedcore.architecture.comptutaional.devices import default_device
from fedcore.neural_compressor import QuantizationAwareTrainingConfig
from fedcore.neural_compressor.training import prepare_compression


class QuantAwareModel(BaseCompressionModel):
    """Class responsible for Pruning model implementation.
    Example:
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)
        self.epochs = params.get('epochs', 5)
        self.loss = params.get('loss', nn.CrossEntropyLoss())
        self.learning_rate = params.get('loss', 3e-3)
        self.optimizer = params.get('optimizewr', optim.AdamW)
        self.scheduler = params.get('optimizewr', torch.optim.lr_scheduler.StepLR)
        self.quantisation_config = params.get('quantisation_config', QuantizationAwareTrainingConfig())
        self.quantisation_model = prepare_compression

    def __repr__(self):
        return 'QuantisationAware'

    def _fit_model(self,
                   input_data: InputData):
        criterion = self.loss
        self.optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = self.scheduler(self.optimizer, step_size=10, gamma=0.1)
        train_losses, train_acces = [], []
        total_step = len(input_data.features)
        for epoch in range(self.epochs):
            print(f'Epoch {epoch}\n')
            self.model.train()
            running_loss = 0.0
            running_corrects = 0
            for batch_idx, (inputs, labels) in enumerate(input_data.features):
                inputs = inputs.to(default_device())
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels.to(default_device()))
                loss.backward()
                self.optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item()
                running_corrects += torch.sum(preds.detach().cpu() == labels.data)
                if (batch_idx) % 20 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, self.epochs - 1, batch_idx,
                                                                             total_step, loss.item()))
            self.scheduler.step()
            epoch_loss = running_loss / total_step
            epoch_acc = running_corrects / total_step
            train_acces.append(epoch_acc * 100)
            train_losses.append(epoch_loss)

        return train_losses, train_acces

    def fit(self,
            input_data: InputData):
        self.quantisation_model = self.quantisation_model(input_data.supplementary_data['model_to_quant'],
                                                          self.quantisation_config)
        self.quantisation_model.callbacks.on_train_begin()
        self.model = self.quantisation_model.model
        self._fit_model(input_data)

    def predict_for_fit(self,
                        input_data: InputData, output_mode: str = 'default') -> np.array:
        test_losses, test_acces = [], []
        # evaluation on test
        self.model.eval()
        test_loss = 0.0
        test_corrects = 0
        for batch_idx, (inputs, labels) in enumerate(input_data.features):
            inputs = inputs.to(default_device())
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels.to(default_device()))
            _, preds = torch.max(outputs, 1)
            test_loss += loss.item()
            test_corrects += torch.sum(preds.detach().cpu() == labels.data)

        epoch_loss = test_loss / len(input_data.features)
        epoch_acc = test_corrects / len(input_data.features)
        test_acces.append(epoch_acc * 100)
        test_losses.append(epoch_loss)
        return self.model

    def predict(self,
                input_data: InputData, output_mode: str = 'default') -> np.array:
        self.model = self.model.to('cpu')
        self.quantisation_model.callbacks.on_train_end()
        self.quantisation_model.save("./output")

