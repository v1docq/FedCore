from typing import Optional

import torch
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from fedcore.architecture.computational.devices import default_device
from fedcore.data.data import CompressionOutputData, CompressionInputData
from fedcore.repository.model_repository import (
    PRUNER_MODELS,
    QUANTIZATION_MODELS,
    DISTILATION_MODELS,
    LOW_RANK_MODELS,
    DETECTION_MODELS,
    TRAINING_MODELS,
)

from fedcore.models.network_impl.llm_trainer import LLMTrainer
from fedcore.models.network_impl.utils.trainer_factory import create_trainer_from_input_data
from fedcore.architecture.preprocessing.data_convertor import CompressionDataConverter

class FedCoreStrategy(EvaluationStrategy):
    def _convert_to_output(
        self,
        prediction,
        predict_data: InputData,
        output_data_type: DataTypesEnum = DataTypesEnum.table,
    ) -> OutputData:
        output_data = CompressionOutputData(
            features=None,  
            val_dataloader=getattr(predict_data, 'val_dataloader', None),
            task=predict_data.task,
            num_classes=getattr(predict_data, 'num_classes', None),
            data_type=DataTypesEnum.image,
            supplementary_data=predict_data.supplementary_data,
        )
        output_data.predict = prediction
        return output_data

    def __init__(
        self, operation_type: str, params: Optional[OperationParameters] = None
    ):
        super().__init__(operation_type, params)
        self.operation_impl = self._convert_to_operation(operation_type)(
            self.params_for_fit
        )

    def fit(self, train_data: InputData):
        compression_data = CompressionDataConverter.convert(train_data)
        self.operation_impl.fit(compression_data)
        return self.operation_impl

    def predict(
        self, trained_operation, predict_data: InputData, output_mode: str = "default"
    ) -> CompressionOutputData:
        compression_data = CompressionDataConverter.convert(predict_data)
        prediction = trained_operation.predict(compression_data, output_mode)
        converted = self._convert_to_output(prediction, compression_data)
        return converted

    def predict_for_fit(
        self, trained_operation, predict_data: InputData, output_mode: str = "default"
    ) -> CompressionOutputData:
        compression_data = CompressionDataConverter.convert(predict_data)
        prediction = trained_operation.predict_for_fit(compression_data, output_mode)
        converted = self._convert_to_output(prediction, compression_data)
        return converted


class FedcoreTrainingStrategy(FedCoreStrategy):
    _operations_by_types = TRAINING_MODELS

    def __init__(
        self, operation_type: str, params: Optional[OperationParameters] = None
    ):
        from functools import partial
        self.operation_impl = partial(
            create_trainer_from_input_data,
            params=params
        )

    def fit(self, train_data: InputData):
        compression_data = CompressionDataConverter.convert(train_data)
        self.operation_impl = self.operation_impl(compression_data)
        self.trained_model = self.operation_impl.fit(compression_data)
        return self.operation_impl

    def predict(
        self, trained_operation, predict_data: InputData, output_mode: str = "default"
    ) -> CompressionOutputData:
        compression_data = CompressionDataConverter.convert(predict_data)
        prediction = trained_operation.predict(compression_data, output_mode)
        if isinstance(prediction, CompressionOutputData):
            return prediction
        converted = self._convert_to_output(prediction, compression_data)
        return converted

    def predict_for_fit(
        self, trained_operation, predict_data: InputData, output_mode: str = "default"
    ) -> CompressionOutputData:
        compression_data = CompressionDataConverter.convert(predict_data)
        prediction = trained_operation.predict_for_fit(compression_data, output_mode)
        if isinstance(prediction, CompressionOutputData):
            return prediction
        converted = self._convert_to_output(prediction, compression_data)
        return converted


class FedcoreLowRankStrategy(FedCoreStrategy):
    _operations_by_types = LOW_RANK_MODELS


class FedcorePruningStrategy(FedCoreStrategy):
    _operations_by_types = PRUNER_MODELS


class FedcoreQuantizationStrategy(FedCoreStrategy):
    _operations_by_types = QUANTIZATION_MODELS


class FedcoreDistilationStrategy(FedcoreQuantizationStrategy):
    _operations_by_types = DISTILATION_MODELS


class FedcoreDetectionStrategy(EvaluationStrategy):
    _operations_by_types = DETECTION_MODELS

    def __init__(
        self, operation_type: str, params: Optional[OperationParameters] = None
    ):
        super().__init__(operation_type, params)
        self.operation_impl = self._convert_to_operation(operation_type)(
            self.params_for_fit
        )

    def fit(self, train_data: InputData):
        if train_data.idx == 0:
            num_classes = len(train_data.features_names)
            in_features = (
                self.operation_impl.roi_heads.box_predictor.cls_score.in_features
            )
            self.operation_impl.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_classes
            )
        self.operation_impl.to(default_device())
        self.operation_impl.train()
        target = [{"boxes": train_data.target[0], "labels": train_data.target[1]}]
        print(target)
        self.operation_impl(torch.unsqueeze(train_data.features, dim=0), target)
        return self.operation_impl

    def predict(
        self,
        trained_operation,
        predict_data: CompressionInputData,
        output_mode: str = "default",
    ) -> CompressionOutputData:
        trained_operation.eval()
        pred = trained_operation(torch.unsqueeze(predict_data.features, dim=0))
        prediction = [
            pred[0]["boxes"].cpu().detach().numpy(),
            pred[0]["labels"].cpu().detach().numpy(),
            pred[0]["scores"].cpu().detach().numpy(),
        ]
        converted = self._convert_to_output(prediction, predict_data)
        return converted

    def predict_for_fit(
        self,
        trained_operation,
        predict_data: CompressionInputData,
        output_mode: str = "default",
    ) -> CompressionOutputData:
        trained_operation.eval()
        pred = trained_operation(torch.unsqueeze(predict_data.features, dim=0))
        converted = self._convert_to_output(pred, predict_data)
        return converted

    def _convert_to_output(
        self,
        prediction,
        predict_data: CompressionInputData,
        output_data_type: DataTypesEnum,
    ) -> OutputData:
        output_data = CompressionOutputData(
            features=predict_data.features,
            # train_dataloader=predict_data.train_dataloader,
            # val_dataloader=predict_data.val_dataloader,
            predict=prediction,
            task=predict_data.task,
            data_type=output_data_type,
            supplementary_data=predict_data.supplementary_data,
        )
        return output_data