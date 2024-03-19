from enum import Enum
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from golem.utilities.memory import MemoryAnalytics


class TaskCompression(Enum):
    classification = 'classification'
    regression = 'regression'
    pruning = 'pruning'
    quantisation = 'quantisation'
    distilation = 'distilation'


def _fit_assumption_and_check_correctness(self,
                                          pipeline,
                                          pipelines_cache,
                                          preprocessing_cache,
                                          eval_n_jobs: int = -1):
    """
    Check if initial pipeline can be fitted on a presented data

    :param pipeline: pipeline for checking
    :param pipelines_cache: Cache manager for fitted models, optional.
    :param preprocessing_cache: Cache manager for optional preprocessing encoders and imputers, optional.
    :param eval_n_jobs: number of jobs to fit the initial pipeline
    """
    try:
        data_train, data_test = _train_test_data_setup(self.data)
        self.log.info('Initial pipeline fitting started')
        # load preprocessing
        pipeline.try_load_from_cache(pipelines_cache, preprocessing_cache)
        pipeline.fit(data_train, n_jobs=eval_n_jobs)

        try:
            if pipelines_cache is not None:
                pipelines_cache.save_pipeline(pipeline)
            if preprocessing_cache is not None:
                preprocessing_cache.add_preprocessor(pipeline)
        except Exception:
            _ = 1

        pipeline.predict(data_test)
        self.log.info('Initial pipeline was fitted successfully')

        MemoryAnalytics.log(self.log,
                            additional_info='fitting of the initial pipeline',
                            logging_level=45)  # message logging level

    except Exception as ex:
        self._raise_evaluating_exception(ex)
    return pipeline


def _merge(self) -> 'InputData':
    return self.main_output


def _fit(self, params, data):
    """This method is used for defining and running of the evaluation strategy
    to train the operation with the data provided

    Args:
        params: hyperparameters for operation
        data: data used for operation training

    Returns:
        tuple: trained operation and prediction on train data
    """
    self._init(data.task, params=params, n_samples_data=data.features.shape[0])

    self.fitted_operation = self._eval_strategy.fit(train_data=data)

    predict_train = self.predict_for_fit(self.fitted_operation, data, params)

    return self.fitted_operation, predict_train


def _train_test_data_setup(data):
    """ Function for train and test split for both InputData and MultiModalData

    :param data: InputData object to split
    :param split_ratio: share of train data between 0 and 1
    :param shuffle: is data needed to be shuffled or not
    :param shuffle_flag: same is shuffle, use for backward compatibility
    :param stratify: make stratified sample or not
    :param random_seed: random_seed for shuffle
    :param validation_blocks: validation blocks are used for test

    :return: data for train, data for validation
    """

    if isinstance(data, InputData):
        train_data, test_data = data, data
    elif isinstance(data, MultiModalData):
        train_data, test_data = MultiModalData(), MultiModalData()
    else:
        raise ValueError((f'Dataset {type(data)} is not supported. Supported types:'
                          ' InputData, MultiModalData'))

    return train_data, test_data
