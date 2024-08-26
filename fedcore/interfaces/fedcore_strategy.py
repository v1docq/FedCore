from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum

from fedcore.data.data import CompressionOutputData, CompressionInputData
from fedcore.repository.model_repository import PRUNER_MODELS, QUANTISATION_MODELS, DISTILATION_MODELS, LOW_RANK_MODELS


class FedcoreLowRankStrategy(EvaluationStrategy):
    __operations_by_types = LOW_RANK_MODELS

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(
                f'Impossible to obtain custom preprocessing strategy for {operation_type}')

    def _convert_to_output(self, prediction, predict_data: InputData,
                           output_data_type: DataTypesEnum = DataTypesEnum.table) -> OutputData:
        predict_data = predict_data.features if type(predict_data) is InputData else predict_data
        output_data = CompressionOutputData(features=predict_data.features,
                                            calib_dataloader=predict_data.calib_dataloader,
                                            task=predict_data.task,
                                            num_classes=predict_data.num_classes,
                                            target=predict_data.target,
                                            data_type=DataTypesEnum.image,
                                            supplementary_data=predict_data.supplementary_data)
        output_data.predict = prediction
        return output_data

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.operation_impl = self._convert_to_operation(operation_type)(self.params_for_fit)

    def fit(self, train_data: InputData):
        self.operation_impl.fit(train_data)
        return self.operation_impl

    def predict(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        pruned_model = trained_operation.predict(predict_data)
        converted = self._convert_to_output(pruned_model, predict_data)
        return converted

    def predict_for_fit(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        pruned_model = trained_operation.predict_for_fit(predict_data)
        converted = self._convert_to_output(pruned_model, predict_data)
        return converted


class FedcorePruningStrategy(EvaluationStrategy):
    __operations_by_types = PRUNER_MODELS

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(
                f'Impossible to obtain custom preprocessing strategy for {operation_type}')

    def _convert_to_output(self, prediction, predict_data: InputData,
                           output_data_type: DataTypesEnum = DataTypesEnum.table) -> OutputData:
        predict_data = predict_data.features if type(predict_data) is InputData else predict_data
        output_data = CompressionOutputData(features=predict_data.features,
                                            calib_dataloader=predict_data.calib_dataloader,
                                            task=predict_data.task,
                                            num_classes=predict_data.num_classes,
                                            target=predict_data.target,
                                            data_type=DataTypesEnum.image,
                                            supplementary_data=predict_data.supplementary_data)
        output_data.predict = prediction
        return output_data

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.operation_impl = self._convert_to_operation(operation_type)(self.params_for_fit)

    def fit(self, train_data: InputData):
        self.operation_impl.fit(train_data)
        return self.operation_impl

    def predict(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        pruned_model = trained_operation.predict(predict_data)
        converted = self._convert_to_output(pruned_model, predict_data)
        return converted

    def predict_for_fit(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        pruned_model = trained_operation.predict_for_fit(predict_data)
        converted = self._convert_to_output(pruned_model, predict_data)
        return converted


class FedcoreQuantisationStrategy(EvaluationStrategy):
    __operations_by_types = QUANTISATION_MODELS

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(
                f'Impossible to obtain custom preprocessing strategy for {operation_type}')

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.operation_impl = self._convert_to_operation(operation_type)(self.params_for_fit)

    def fit(self, train_data: InputData):
        self.operation_impl.fit(train_data)
        return self.operation_impl

    def predict(self, trained_operation, predict_data: CompressionInputData,
                output_mode: str = 'default') -> OutputData:
        pruned_model = trained_operation.predict(predict_data)
        converted = self._convert_to_output(pruned_model, predict_data)
        return converted

    def predict_for_fit(self, trained_operation, predict_data: CompressionInputData,
                        output_mode: str = 'default') -> OutputData:
        pruned_model = trained_operation.predict_for_fit(predict_data)
        converted = self._convert_to_output(pruned_model, predict_data)
        return converted

    def _convert_to_output(self, prediction, predict_data: InputData,
                           output_data_type: DataTypesEnum = DataTypesEnum.table) -> OutputData:
        output_data = CompressionOutputData(features=predict_data.features,
                                            train_dataloader=predict_data.features.train_dataloader,
                                            calib_dataloader=predict_data.features.calib_dataloader,
                                            predict=prediction,
                                            task=predict_data.task,
                                            target=predict_data.target,
                                            data_type=output_data_type,
                                            supplementary_data=predict_data.supplementary_data)
        return output_data


class FedcoreDistilationStrategy(FedcoreQuantisationStrategy):
    __operations_by_types = DISTILATION_MODELS

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.operation_impl = self._convert_to_operation(operation_type)(self.params_for_fit)

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(
                f'Impossible to obtain custom preprocessing strategy for {operation_type}')
