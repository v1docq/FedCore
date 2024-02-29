import numpy as np
from fedot.core.data.data import InputData
from torch import optim, nn

from fedcore.algorithm.base_compression_model import BaseCompressionModel
from typing import Optional
import torch
from fedot.core.operations.operation_parameters import OperationParameters

from fedcore.architecture.comptutaional.devices import default_device
from fedcore.data.data import CompressionInputData
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
        self.learning_rate = params.get('lr', 3e-3)
        self.optimizer = params.get('optimizer', optim.AdamW)
        self.scheduler = params.get('optimizer', torch.optim.lr_scheduler.StepLR)
        self.quantisation_config = params.get('quantisation_config', QuantizationAwareTrainingConfig())
        self.quantisation_model = prepare_compression

    def __repr__(self):
        return 'QuantisationAware'

    def _fit_model(self, input_data: CompressionInputData):
        criterion = self.loss
        self.optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = self.scheduler(self.optimizer, step_size=10, gamma=0.1)
        for epoch in range(self.epochs):
            print(f'Epoch {epoch}\n')
            self.model.train()
            running_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(input_data.train_dataloader):
                inputs = inputs.to(default_device())
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels.to(default_device()))
                loss.backward()
                self.optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item()
                if batch_idx % 20 == 0:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, batch_idx + 1, running_loss / 20))
                    running_loss = 0.0
            self.scheduler.step()
            if epoch > 3:
                # Freeze quantizer parameters
                self.model.apply(torch.quantization.disable_observer)
            if epoch > 2:
                # Freeze batch norm mean and variance estimates
                self.model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        self.model = self.model.cpu()

    def fit(self,
            input_data: CompressionInputData):
        self.quantisation_model = self.quantisation_model(input_data.target,
                                                          self.quantisation_config)
        self.quantisation_model.callbacks.on_train_begin()
        self.model = self.quantisation_model.model
        self._fit_model(input_data)
        self.quantisation_model.callbacks.on_train_end()

    def predict_for_fit(self,
                        input_data: CompressionInputData, output_mode: str = 'default') -> np.array:
        return self.model

    def predict(self,
                input_data: CompressionInputData, output_mode: str = 'default') -> np.array:
        return self.model
