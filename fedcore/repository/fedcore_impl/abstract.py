import gc
from copy import deepcopy
from enum import Enum
from typing import Optional, Sequence

from fedot.core.data.multi_modal import MultiModalData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.api.time import ApiTime
from fedot.core.data.data import InputData
from fedot.utilities.composer_timer import fedot_composer_timer
from golem.core.dag.graph_utils import map_dag_nodes
from fedot.core.pipelines.pipeline import Pipeline

from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.utilities.memory import MemoryAnalytics
from fedot.core.pipelines.node import PipelineNode
from fedcore.repository.fedcore_impl.optimisation import FedcoreMutations


@property
def is_fitted_fedcore(self, value: bool = None) -> bool:
    """Property showing whether pipeline is fitted

    Returns:
        flag showing if all of the pipeline nodes are fitted already
    """
    if value is None:
        return all(node.fitted_operation is not None for node in self.nodes)
    else:
        return value


def adapter_restore(opt_graph, metadata=None):
    def transform_to_pipeline_node(node):
        content = deepcopy(node.content)
        return PipelineNode(operation_type=content['name'], content=content)

    restored_nodes = map_dag_nodes(transform_to_pipeline_node, opt_graph.nodes)
    pipeline = Pipeline(restored_nodes, use_input_preprocessing=False)
    metadata = metadata or {}
    pipeline.computation_time = metadata.get('computation_time_in_seconds')

    return pipeline


def restore_pipeline_fedcore(self, opt_result):
    multi_objective = self.optimizer.objective.is_multi_objective
    best_pipelines = [adapter_restore(graph) for graph in opt_result]
    if not best_pipelines:
        return None, []
    chosen_best_pipeline = best_pipelines if multi_objective else best_pipelines[0]
    return chosen_best_pipeline, best_pipelines


def fit_fedcore(self,
                features,
                target='target',
                predefined_model=None):
    """Composes and fits a new pipeline, or fits a predefined one.

    Args:
        features: train data feature values in one of the supported features formats.
        target: train data target values in one of the supported target formats.
        predefined_model: the name of a single model or a :class:`Pipeline` instance, or ``auto``.
            With any value specified, the method does not perform composing and tuning.
            In case of ``auto``, the method generates a single initial assumption and then fits
            the created pipeline.

    Returns:
        :class:`Pipeline` object.
    """

    MemoryAnalytics.start()
    self.target = target
    with fedot_composer_timer.launch_data_definition('fit'):
        self.train_data = self.data_processor.define_data(features=features, target=target, is_predict=False)

    self.params.update_available_operations_by_preset(self.train_data)
    self._init_remote_if_necessary()

    if isinstance(self.train_data, InputData) and self.params.get('use_auto_preprocessing'):
        with fedot_composer_timer.launch_preprocessing():
            self.train_data = self.data_processor.fit_transform(self.train_data)

    self.current_pipeline, self.best_models, self.history = self.api_composer.obtain_model(self.train_data)
    MemoryAnalytics.finish()
    return self.current_pipeline


def obtain_model_fedcore(self, train_data: InputData):
    """ Function for composing FEDOT pipeline model """

    with fedot_composer_timer.launch_composing():
        timeout: float = self.params.timeout
        with_tuning = self.params.get('with_tuning')

        self.timer = ApiTime(time_for_automl=timeout, with_tuning=with_tuning)

        # skip fit init_assumption for Fedcore
        # initial_assumption, fitted_assumption = self.propose_and_fit_initial_assumption(train_data)
        initial_assumption, fitted_assumption = self.params.get('initial_assumption'), None
        multi_objective = len(self.metrics) > 1
        self.params.init_params_for_composing(self.timer.timedelta_composing, multi_objective)

        self.log.message(f"AutoML configured."
                         f" Parameters tuning: {with_tuning}."
                         f" Time limit: {timeout} min."
                         f" Set of candidate models: {self.params.get('available_operations')}.")

        best_pipeline, best_pipeline_candidates, gp_composer = self.compose_pipeline(
            train_data,
            initial_assumption,
            fitted_assumption
        )

    if with_tuning:
        with fedot_composer_timer.launch_tuning('composing'):
            best_pipeline = self.tune_final_pipeline(train_data, best_pipeline)

    if gp_composer.history:
        adapter = self.params.graph_generation_params.adapter
        gp_composer.history.tuning_result = adapter.adapt(best_pipeline)
    # enforce memory cleaning
    gc.collect()

    self.log.message('Model generation finished')
    return best_pipeline, best_pipeline_candidates, gp_composer.history


class TaskCompression(Enum):
    classification = "classification"
    regression = "regression"
    pruning = "pruning"
    quantisation = "quantisation"
    distilation = "distilation"


def _fit_assumption_and_check_correctness(
        self, pipeline, pipelines_cache, preprocessing_cache, eval_n_jobs: int = -1
):
    """
    Check if initial pipeline can be fitted on a presented data

    :param pipeline: pipeline for checking
    :param pipelines_cache: Cache manager for fitted models, optional.
    :param preprocessing_cache: Cache manager for optional preprocessing encoders and imputers, optional.
    :param eval_n_jobs: number of jobs to fit the initial pipeline
    """
    try:
        data_train, data_test = _train_test_data_setup(self.data)
        self.log.info("Initial pipeline fitting started")
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
        self.log.info("Initial pipeline was fitted successfully")

        MemoryAnalytics.log(
            self.log,
            additional_info="fitting of the initial pipeline",
            logging_level=45,
        )  # message logging level

    except Exception as ex:
        self._raise_evaluating_exception(ex)
    return pipeline


def _merge(self) -> "InputData":
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
    """Function for train and test split for both InputData and MultiModalData

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
        raise ValueError(
            (
                f"Dataset {type(data)} is not supported. Supported types:"
                " InputData, MultiModalData"
            )
        )

    return train_data, test_data


def predict_operation_fedcore(
        self,
        fitted_operation,
        data: InputData,
        params: Optional[OperationParameters] = None,
        output_mode: str = "default",
        is_fit_stage: bool = False,
):
    is_main_target = data.supplementary_data.is_main_target
    data_flow_length = data.supplementary_data.data_flow_length
    self._init(
        data.task,
        output_mode=output_mode,
        params=params,
        n_samples_data=data.features.shape[0],
    )

    if is_fit_stage:
        prediction = self._eval_strategy.predict_for_fit(
            trained_operation=fitted_operation,
            predict_data=data,
            output_mode=output_mode,
        )
    else:
        prediction = self._eval_strategy.predict(
            trained_operation=fitted_operation,
            predict_data=data,
            output_mode=output_mode,
        )
    prediction = self.assign_tabular_column_types(prediction, output_mode)

    # any inplace operations here are dangerous!
    if is_main_target is False:
        prediction.supplementary_data.is_main_target = is_main_target

    prediction.supplementary_data.data_flow_length = data_flow_length
    return prediction


def fedcore_preprocess_predicts(self, predicts):
    return predicts


def merge_fedcore_predicts(self, predicts):
    sample_shape, channel_shape, elem_shape = [
        (x.shape[0], x.shape[1], x.shape[2]) for x in predicts
    ][0]

    sample_wise_concat = [x.shape[0] == sample_shape for x in predicts]
    chanel_concat = [x.shape[1] == channel_shape for x in predicts]
    element_wise_concat = [x.shape[2] == elem_shape for x in predicts]

    all(chanel_concat)
    all(element_wise_concat)
    sample_match = all(sample_wise_concat)
    return sample_match


def _get_default_fedcore_mutations(
        task_type: TaskTypesEnum, params
) -> Sequence[MutationTypesEnum]:
    ind_mutations = FedcoreMutations(task_type=task_type)
    mutations = [
        ind_mutations.parameter_change_mutation,
        ind_mutations.single_change,
        # ind_mutations.single_drop,
        # ind_mutations.single_add
    ]
    return mutations


@staticmethod
def divide_operations_fedcore(available_operations, task):
    """ Function divide operations for primary and secondary """

    primary_operations = available_operations
    secondary_operations = available_operations
    return primary_operations, secondary_operations
