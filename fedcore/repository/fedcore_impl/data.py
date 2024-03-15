from functools import partial

from fedot.core.data.data import InputData

from fedcore.data.data import CompressionInputData


def build_holdout_producer(self, data: CompressionInputData):
    """
    Build trivial data producer for hold-out validation
    that always returns same data split. Equivalent to 1-fold validation.
    """

    def convert_compression_to_input(data):
        return InputData(idx=[1],
                         features=data,
                         target=data.target,
                         task=data.task,
                         data_type=None,
                         supplementary_data=data.supplementary_data)

    train_data, test_data = convert_compression_to_input(data), convert_compression_to_input(data)
    train_data.supplementary_data.is_auto_preprocessed = True
    test_data.supplementary_data.is_auto_preprocessed = True
    return partial(self._data_producer, train_data, test_data)
