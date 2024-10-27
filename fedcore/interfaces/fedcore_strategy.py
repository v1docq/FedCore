from typing import Optional

import torch
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from fedcore.data.data import CompressionOutputData, CompressionInputData
from fedcore.repository.constanst_repository import default_device
from fedcore.repository.model_repository import (
    PRUNER_MODELS,
    QUANTISATION_MODELS,
    DISTILATION_MODELS,
    LOW_RANK_MODELS,
    DETECTION_MODELS,
    TRAINING_MODELS,
)

from fedcore.models.network_impl.base_nn_model import BaseNeuralModel

class FedCoreStrategy(EvaluationStrategy):
    def _convert_to_output(
        self,
        prediction,
        predict_data: InputData,
        output_data_type: DataTypesEnum = DataTypesEnum.table,
    ) -> OutputData:
        output_data = CompressionOutputData(
            features=predict_data.features,
            idx=[1, 2],
            calib_dataloader=predict_data.features.calib_dataloader,
            task=predict_data.task,
            num_classes=predict_data.features.num_classes,
            target=predict_data.features.target,
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
        self.operation_impl.fit(train_data)
        return self.operation_impl

    def predict(
        self, trained_operation, predict_data: InputData, output_mode: str = "default"
    ) -> OutputData:
        pruned_model = trained_operation.predict(predict_data, output_mode)
        converted = self._convert_to_output(pruned_model, predict_data)
        return converted

    def predict_for_fit(
        self, trained_operation, predict_data: InputData, output_mode: str = "default"
    ) -> OutputData:
        pruned_model = trained_operation.predict_for_fit(predict_data, output_mode)
        converted = self._convert_to_output(pruned_model, predict_data)
        return converted


class FedcoreTrainingStrategy(FedCoreStrategy):
    _operations_by_types = TRAINING_MODELS

    def fit(self, train_data: InputData):
        self.original_model = train_data.features.target
        self.trained_model = self.operation_impl.fit(train_data)
        return self.operation_impl

    def predict(
        self, trained_operation, predict_data: InputData, output_mode: str = "default"
    ) -> OutputData:
        converted = self._convert_to_output(trained_operation.model, predict_data)
        return converted

    def predict_for_fit(
        self, trained_operation, predict_data: InputData, output_mode: str = "default"
    ) -> OutputData:
        converted = self._convert_to_output(trained_operation.model, predict_data)
        return converted


class FedcoreLowRankStrategy(FedCoreStrategy):
    _operations_by_types = LOW_RANK_MODELS


class FedcorePruningStrategy(FedCoreStrategy):
    _operations_by_types = PRUNER_MODELS


class FedcoreQuantisationStrategy(FedCoreStrategy):
    _operations_by_types = QUANTISATION_MODELS


class FedcoreDistilationStrategy(FedcoreQuantisationStrategy):
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
    ) -> OutputData:
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
    ) -> OutputData:
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
            idx=[1, 2],
            features=predict_data.features,
            # train_dataloader=predict_data.train_dataloader,
            # calib_dataloader=predict_data.calib_dataloader,
            predict=prediction,
            task=predict_data.task,
            target=predict_data.target,
            data_type=output_data_type,
            supplementary_data=predict_data.supplementary_data,
        )
        return output_data
