import pathlib

import fedot.core.data.data_split as sp
from fedot.api.api_utils.assumptions.assumptions_handler import AssumptionsHandler
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.utilities.memory import MemoryAnalytics

from fedcore.architecture.utils.paths import PROJECT_PATH
from fedcore.interfaces.search_space import get_fedcore_search_space


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

        if pipelines_cache is not None:
            pipelines_cache.save_pipeline(pipeline)
        if preprocessing_cache is not None:
            preprocessing_cache.add_preprocessor(pipeline)

        pipeline.predict(data_test)
        self.log.info('Initial pipeline was fitted successfully')

        MemoryAnalytics.log(self.log,
                            additional_info='fitting of the initial pipeline',
                            logging_level=45)  # message logging level

    except Exception as ex:
        self._raise_evaluating_exception(ex)
    return pipeline
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


class FedcoreModels:
    def __init__(self):
        self.fedcore_data_operation_path = pathlib.Path(PROJECT_PATH, 'fedcore',
                                                        'repository',
                                                        'data',
                                                        'compression_data_operation_repository.json')
        self.base_data_operation_path = pathlib.Path(
            'data_operation_repository.json')

        self.fedcore_model_path = pathlib.Path(PROJECT_PATH, 'fedcore',
                                               'repository',
                                               'data',
                                               'compression_model_repository.json')
        self.base_model_path = pathlib.Path('model_repository.json')

    def setup_repository(self):
        OperationTypesRepository.__repository_dict__.update(
            {'data_operation': {'file': self.fedcore_data_operation_path,
                                'initialized_repo': True,
                                'default_tags': []}})

        OperationTypesRepository.assign_repo(
            'data_operation', self.fedcore_data_operation_path)

        OperationTypesRepository.__repository_dict__.update(
            {'model': {'file': self.fedcore_model_path,
                       'initialized_repo': True,
                       'default_tags': []}})
        OperationTypesRepository.assign_repo(
            'model', self.fedcore_model_path)

        setattr(PipelineSearchSpace, "get_parameters_dict",
                get_fedcore_search_space)
        setattr(AssumptionsHandler, "fit_assumption_and_check_correctness",
                _fit_assumption_and_check_correctness)

        return OperationTypesRepository

    def __enter__(self):
        """
        Switching to industrial models
        """
        OperationTypesRepository.__repository_dict__.update(
            {'data_operation': {'file': self.industrial_data_operation_path,
                                'initialized_repo': True,
                                'default_tags': []}})

        OperationTypesRepository.assign_repo(
            'data_operation', self.industrial_data_operation_path)

        OperationTypesRepository.__repository_dict__.update(
            {'model': {'file': self.industrial_model_path,
                       'initialized_repo': True,
                       'default_tags': []}})
        OperationTypesRepository.assign_repo(
            'model', self.industrial_model_path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Switching to fedot models.
        """
        OperationTypesRepository.__repository_dict__.update(
            {'data_operation': {'file': self.base_data_operation_path,
                                'initialized_repo': None,
                                'default_tags': [
                                    OperationTypesRepository.DEFAULT_DATA_OPERATION_TAGS]}})
        OperationTypesRepository.assign_repo(
            'data_operation', self.base_data_operation_path)

        OperationTypesRepository.__repository_dict__.update(
            {'model': {'file': self.base_model_path,
                       'initialized_repo': None,
                       'default_tags': []}})
        OperationTypesRepository.assign_repo('model', self.base_model_path)
