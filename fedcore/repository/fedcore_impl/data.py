from fedcore.data.data import CompressionInputData
from functools import partial
from typing import Union
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.optimisers.objective import DataSource


def build_holdout_producer(self, data: CompressionInputData):
    """
    Build trivial data producer for hold-out validation
    that always returns same data split. Equivalent to 1-fold validation.
    """

    def convert_compression_to_input(data):
        is_input_data = isinstance(data, InputData)
        converted = InputData(
            idx=[1],
            features=data,
            target=data.target,
            task=data.task,
            data_type=None,
            supplementary_data=data.supplementary_data,
        )
        return data if is_input_data else converted

    train_data, test_data = convert_compression_to_input(
        data
    ), convert_compression_to_input(data)
    train_data.supplementary_data.is_auto_preprocessed = True
    test_data.supplementary_data.is_auto_preprocessed = True
    return partial(self._data_producer, train_data, test_data)


def build_fedcore_dataproducer(
    self, data: Union[InputData, MultiModalData]
) -> DataSource:
    return self._build_holdout_producer(data)
