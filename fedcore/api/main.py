import warnings
from copy import deepcopy
from functools import partial
from typing import Union, Optional, Callable
import numpy as np
import pandas as pd
import torch
import torch.nn
from fedot.api.main import Fedot
from fedot.core.data.data import InputData, OutputData
from fedot.core.pipelines.pipeline import Pipeline
from pymonad.either import Either
from pymonad.maybe import Maybe
from torch import Tensor
from fedcore.api.utils.api_init import ApiManager
from fedcore.api.utils.checkers_collection import DataCheck
from fedcore.architecture.abstraction.decorators import DaskServer, exception_handler
from fedcore.architecture.dataset.api_loader import ApiLoader
from fedcore.inference.onnx import ONNXInferenceModel
from fedcore.metrics.cv_metrics import CV_quality_metric
from fedcore.neural_compressor.config import Torch2ONNXConfig
from fedcore.repository.constanst_repository import (
    FEDCORE_CV_DATASET,
    FEDOT_API_PARAMS,
    FEDOT_ASSUMPTIONS,
    FEDOT_GET_METRICS,
)
from fedcore.repository.initializer_industrial_models import FedcoreModels

warnings.filterwarnings("ignore")


class FedCore(Fedot):
    """This class is used to run Fedot in model compression mode as FedCore.

    Args:
        input_config: dictionary with the parameters of the experiment.
        output_folder: path to the folder where the results will be saved.

    Example:
        First, configure experiment and instantiate FedotIndustrial class::

            from fedcore.api.main import FedCore

            model = FedCore()

    """

    def __init__(self, **kwargs):
        super(Fedot, self).__init__()
        self.manager = ApiManager().build(kwargs)
        self.logger = self.manager.logger

    def __init_fedcore_backend(self, input_data: Optional[Union[InputData, np.array]] = None):
        self.logger.info('-' * 50)
        self.logger.info('Initialising Industrial Repository')
        if self.manager.industrial_config.is_default_fedot_context:
            self.logger.info(f'-------------------------------------------------')
            self.logger.info('Initialising Fedot Evolutionary Optimisation params')
            self.repo = FedcoreModels().setup_default_repository()
            self.manager.automl_config.optimisation_strategy = self.manager.optimisation_agent['Fedot']
        else:
            self.logger.info(f'-------------------------------------------------')
            self.logger.info('Initialising Industrial Evolutionary Optimisation params')
            self.repo = FedcoreModels().setup_repository(backend=self.manager.compute_config.backend)
            optimisation_agent = self.manager.automl_config.optimisation_strategy['optimisation_agent']
            optimisation_params = self.manager.automl_config.optimisation_strategy['optimisation_strategy']
            self.manager.automl_config.optimisation_strategy = partial(
                self.manager.optimisation_agent[optimisation_agent],
                optimisation_params=optimisation_params)
        return input_data

    def __init_solver(self, input_data: Optional[Union[InputData, np.array]] = None):
        self.logger.info('-' * 50)
        self.logger.info('Initialising Dask Server')
        if self.manager.automl_config.config['initial_assumption'] is None:
            self.manager.automl_config.config['initial_assumption'] = \
                self.manager.industrial_config.config['initial_assumption'].build()
        else:
            self.manager.automl_config.config['initial_assumption'] = \
                self.manager.automl_config.config['initial_assumption'].build()
        dask_server = DaskServer(self.manager.compute_config.distributed)
        self.manager.dask_client = dask_server.client
        self.manager.dask_cluster = dask_server.cluster
        self.logger.info(f'Link Dask Server - {self.manager.dask_client.dashboard_link}')
        self.logger.info('-' * 50)
        self.logger.info('Initialising solver')
        self.manager.solver = Fedot(
            **self.manager.learning_config.config['learning_strategy_params'],
            metric=self.manager.learning_config.config['optimisation_loss'],
            problem=self.manager.automl_config.config['task'],
            task_params=self.manager.industrial_config.task_params
            if self.manager.industrial_config.is_forecasting_context else self.manager.automl_config.config
            ['task_params'], optimizer=self.manager.automl_config.optimisation_strategy,
            available_operations=self.manager.automl_config.config['available_operations'],
            initial_assumption=self.manager.automl_config.config['initial_assumption'])
        return input_data

    def _process_input_data(self, input_data):
        train_data = Either.insert(input_data).then(lambda data: deepcopy(data)). \
            then(lambda data: DataCheck(input_data=data, task=self.manager.automl_config.config['task'],
                                        task_params=self.manager.automl_config.config['task_params'], fit_stage=True,
                                        industrial_task_params=self.manager.industrial_config.strategy_params)). \
            then(lambda data_cls: (data_cls.check_input_data(), data_cls.get_target_encoder())).value

        train_data.features = train_data.features.squeeze() if self.manager.industrial_config.is_default_fedot_context \
            else train_data.features
        return train_data

    def _pretrain_before_optimise(self, fedot_pipeline: Pipeline):
        pretrained_model = fedot_pipeline.fit(self.train_data)
        self.train_data.target = pretrained_model.predict
        return pretrained_model

    def __abstract_predict(self, predict_data):
        custom_predict = all([not self.manager.condition_check.solver_is_fedot_class(self.manager.solver),
                              not self.manager.condition_check.solver_is_pipeline_class(self.manager.solver)])
        predict = Either(value=predict_data,
                         monoid=[predict_data, custom_predict]).either(
            left_function=lambda predict_for_solver:  self.manager.solver.predict(predict_for_solver),
            right_function=lambda predict_for_custom: self.manager.solver.predict(predict_for_custom))
        predict = Maybe.insert(predict).then(lambda x: x.predict if isinstance(predict, OutputData) else x).value
        return predict

    def fit(self, input_data: tuple, manually_done: bool = False, **kwargs):
        """
        Method for training Industrial model.

        Args:
            input_data: tuple with train_features and train_target
            **kwargs: additional parameters

        """

        def fit_function(train_data):
            return Either(value=train_data, monoid=[FEDOT_ASSUMPTIONS["training"].build(),
                                                    self.manager.learning_config.
                          config['learning_strategy'].__contains__('from_scratch')]). \
                either(left_function=self.manager.solver.fit,
                       right_function=self._pretrain_before_optimise)

        with exception_handler(Exception, on_exception=self.shutdown, suppress=False):
            solver = Maybe.insert(self._process_input_data(input_data)). \
                then(self.__init_fedcore_backend). \
                then(self.__init_solver). \
                then(fit_function). \
                maybe(None, lambda solver: solver)
        self.optimised_model = solver.root_node.fitted_operation.optimised_model
        return solver

    def predict(self, predict_data: tuple, **kwargs):
        """
        Method to obtain prediction labels from trained Industrial model.

        Args:
            predict_data: tuple with test_features and test_target

        Returns:
            the array with prediction values

        """
        self.manager.predicted_labels = Maybe. \
            insert(self._process_input_data(predict_data)). \
            then(self.__init_fedcore_backend). \
            then(self.__abstract_predict). \
            maybe(None, lambda labels: labels)
        return self.manager.predicted_labels

    def finetune(self, train_data, tuning_params=None):
        """
        Method to obtain prediction probabilities from trained Industrial model.

        Args:
            train_data: raw train data
            tuning_params: dictionary with tuning parameters
            mode: str, ``default='full'``. Defines the mode of fine-tuning. Could be 'full' or 'head'.

        """

        def _fit_pipeline(data_dict):
            data_dict['model_to_tune'].fit(data_dict['train_data'])
            return data_dict

        is_fedot_datatype = self.manager.condition_check.input_data_is_fedot_type(train_data)
        tuning_params['metric'] = FEDOT_TUNING_METRICS[self.manager.automl_config.config['task']]
        tuning_params['tuner'] = FEDOT_TUNER_STRATEGY[tuning_params.get('tuner', 'sequential')]

        with exception_handler(Exception, on_exception=self.shutdown, suppress=False):
            model_to_tune = Either.insert(train_data). \
                then(lambda data: self._process_input_data(data) if not is_fedot_datatype else data). \
                then(lambda data: self.__init_fedcore_backend(data)). \
                then(lambda processed_data: {'train_data': processed_data} |
                                            {'model_to_tune': model_to_tune.build()} |
                                            {'tuning_params': tuning_params}). \
                then(lambda dict_for_tune: _fit_pipeline(dict_for_tune)['model_to_tune'] if return_only_fitted
            else build_tuner(self, **dict_for_tune)).value

        self.manager.is_finetuned = True
        self.manager.solver = model_to_tune

    def _metric_evaluation_loop(
            self, target, predicted_labels, predicted_probs, problem, metric_type
    ):
        prediction_dict = dict(
            target=target, labels=predicted_labels, probs=predicted_probs
        )
        inference_metric = metric_type.__contains__("computational")
        inference_model = (
            self.optimised_model
            if metric_type.__contains__("optimised")
            else self.original_model
        )
        inference_eval = CV_quality_metric()

        prediction_dataframe = Either(
            value=inference_model, monoid=[prediction_dict, inference_metric]
        ).either(
            left_function=lambda pred: FEDOT_GET_METRICS[problem](**pred),
            right_function=lambda model: inference_eval.metric(
                model=model, dataset=self.predict_data.features.calib_dataloader
            ),
        )

        return prediction_dataframe

    def evaluate_metric(
            self,
            predicton: Union[Tensor, np.array],
            target: Union[list, np.array],
            metric_type: str = "quality",
    ) -> pd.DataFrame:
        """
        Method to calculate metrics for Industrial model.

        Available metrics for classification task: 'f1', 'accuracy', 'precision', 'roc_auc', 'logloss'.

        Available metrics for regression task: 'r2', 'rmse', 'mse', 'mae', 'median_absolute_error',
        'explained_variance_score', 'max_error', 'd2_absolute_error_score', 'msle', 'mape'.

        Args:
            metric_type:
            predicton:
            target: target values

        Returns:
            pandas DataFrame with calculated metrics

        """
        from sklearn.preprocessing import OneHotEncoder

        model_output = (
            predicton.cpu().detach().numpy()
            if isinstance(predicton, Tensor)
            else predicton
        )
        model_output_is_probs = all(
            [len(model_output.shape) > 1, model_output.shape[1] > 1]
        )
        labels, predicted_probs = Either(
            value=model_output, monoid=[model_output, model_output_is_probs]
        ).either(
            left_function=lambda output: (
                output,
                OneHotEncoder().fit_transform(output),
            ),
            right_function=lambda output: (np.argmax(output, axis=1), output),
        )
        metric_dict = self._metric_evaluation_loop(
            target=target,
            problem=self.cv_task,
            predicted_labels=labels,
            predicted_probs=predicted_probs,
            metric_type=metric_type,
        )
        return metric_dict

    def load(self, path):
        """Loads saved Industrial model from disk

        Args:
            path (str): path to the model

        """

    def load_data(self, path: str = None, supplementary_data: dict = None):
        pretrained_scenario = all(
            [
                any([path.__contains__("CIFAR"), path.__contains__("MNIST")]),
                supplementary_data is not None,
            ]
        )
        torchvision_scenario = all(
            [
                supplementary_data is not None,
                "torchvision_dataset" in supplementary_data.keys(),
                "torchvision_dataset" == True,
            ]
        )
        custom_scenario = "pretrain" if pretrained_scenario else "directory"
        data_loader = ApiLoader()
        self.train_data = Either(
            value="torchvision", monoid=[custom_scenario, torchvision_scenario]
        ).either(
            left_function=lambda loader_type: data_loader.load_data(
                loader_type, supplementary_data, path
            ),
            right_function=lambda loader_type: data_loader.load_data(
                loader_type, supplementary_data, path
            ),
        )
        self.target = np.array(self.train_data[0].calib_dataloader.dataset.targets)
        return self.train_data

    def save(self, mode: str = 'all', **kwargs):
        is_fedot_solver = self.manager.condition_check.solver_is_fedot_class(self.manager.solver)

        def save_model(api_manager):
            return Either(value=api_manager.solver,
                          monoid=[api_manager.solver,
                                  api_manager.condition_check.solver_is_fedot_class(
                                      api_manager.solver)]). \
                either(left_function=lambda pipeline: pipeline.save(path=api_manager.compute_config.output_folder,
                                                                    create_subdir=True, is_datetime_in_path=True),
                       right_function=lambda solver: solver.current_pipeline.save(
                           path=api_manager.compute_config.output_folder,
                           create_subdir=True,
                           is_datetime_in_path=True))

        def save_opt_hist(api_manager):
            return self.manager.solver.history.save(
                f"{self.manager.compute_config.output_folder}/optimization_history.json")

        def save_metrics(api_manager):
            return self.metric_dict.to_csv(
                f'{self.manager.compute_config.output_folder}/metrics.csv')

        def save_preds(api_manager):
            return pd.DataFrame(api_manager.predicted_labels).to_csv(
                f'{self.manager.compute_config.output_folder}/labels.csv')

        method_dict = {'metrics': save_metrics, 'model': save_model, 'opt_hist': save_opt_hist,
                       'prediction': save_preds}
        self.manager.create_folder(self.manager.compute_config.output_folder)
        if not is_fedot_solver:
            del method_dict['opt_hist']

        def save_all(api_manager):
            for method in method_dict.values():
                try:
                    method(api_manager)
                except Exception as ex:
                    self.manager.logger.info(f'Error during saving. Exception - {ex}')

        Either(value=self.manager, monoid=[self.manager, mode.__contains__('all')]). \
            either(left_function=lambda api_manager: method_dict[mode](self.manager),
                   right_function=lambda api_manager: save_all(api_manager))

    def export(
            self,
            framework: str = "ONNX",
            framework_config: dict = None,
            supplementary_data: dict = None,
    ):
        if self.framework_config is None and framework_config is None:
            return self.logger.info(
                "You must specify configuration for model convertation"
            )
        else:
            if framework == "ONNX":
                example_input = next(iter(self.train_data.features.calib_dataloader))[
                    0
                ][0]
                self.framework_config["example_inputs"] = torch.unsqueeze(
                    example_input, dim=0
                )
                onnx_config = Torch2ONNXConfig(**self.framework_config)
                supplementary_data["model_to_export"].export(
                    "converted-model.onnx", onnx_config
                )
                converted_model = ONNXInferenceModel("converted-model.onnx")
        return converted_model

    def shutdown(self):
        """Shutdown Dask client"""
        if self.manager.dask_client is not None:
            self.manager.dask_client.close()
            del self.manager.dask_client
        if self.manager.dask_cluster is not None:
            self.manager.dask_cluster.close()
            del self.manager.dask_cluster
