from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.operation_parameters import OperationParameters
from fedcore.repository.model_repository import PRUNER_MODELS, QUANTISATION_MODELS


class FedcorePruningStrategy(EvaluationStrategy):
    __operations_by_types = PRUNER_MODELS

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(
                f'Impossible to obtain custom preprocessing strategy for {operation_type}')

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.operation_impl = self._convert_to_operation(operation_type)

    def fit(self, train_data: InputData):
        return self.operation_impl.fit(train_data)

    def predict(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        return self.operation_impl.predict(trained_operation, predict_data, output_mode=output_mode)

    def predict_for_fit(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        return self.operation_impl.predict_for_fit(trained_operation, predict_data, output_mode=output_mode)


class FedcoreQuantisationStrategy(FedcorePruningStrategy):
    __operations_by_types = QUANTISATION_MODELS

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.operation_impl = self._convert_to_operation(operation_type)

