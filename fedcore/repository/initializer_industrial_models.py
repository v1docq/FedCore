import pathlib
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedcore.architecture.utils.paths import PROJECT_PATH


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

        # setattr(PipelineSearchSpace, "get_parameters_dict",
        #         get_industrial_search_space)
        # setattr(ApiParamsRepository, "_get_default_mutations",
        #         _get_default_industrial_mutations)
        # setattr(Crossover, '_crossover_by_type', _crossover_by_type)

        # setattr(ImageDataMerger, "preprocess_predicts", preprocess_predicts)
        # setattr(ImageDataMerger, "merge_predicts", merge_predicts)
        # setattr(Operation, "_predict", predict_operation)
        # setattr(Operation, "predict", predict)
        # setattr(Operation, "predict_for_fit", predict_for_fit)
        # setattr(TSDataMerger, 'postprocess_predicts', postprocess_predicts)
        #
        # # setattr(F1, "metric", metric_f1)
        # # setattr(Accuracy, "metric", metric_acc)
        #
        # setattr(LaggedImplementation,
        #         '_update_column_types', update_column_types)
        # setattr(LaggedImplementation, 'transform', transform_lagged)
        # setattr(LaggedImplementation, 'transform_for_fit',
        #         transform_lagged_for_fit)
        # setattr(TsSmoothingImplementation, 'transform', transform_smoothing)

        # class_rules.append(has_no_data_flow_conflicts_in_industrial_pipeline)

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
