import os
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
from fedcore.inference.onnx import ONNXInferenceModel
from fedcore.neural_compressor.config import Torch2ONNXConfig
from fedcore.repository.constanst_repository import (
    FEDCORE_CV_DATASET,
    FEDOT_API_PARAMS,
    FEDOT_ASSUMPTIONS,
    FEDOT_GET_METRICS,
)
from fedcore.repository.initializer_industrial_models import FedcoreModels
from fedcore.repository.model_repository import default_fedcore_availiable_operation

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

    def __init_fedcore_backend(self, input_data: Optional[InputData] = None):
        self.logger.info('-' * 50)
        self.logger.info('Initialising Fedcore Repository')
        self.logger.info('Initialising Fedcore Evolutionary Optimisation params')
        self.repo = FedcoreModels().setup_repository()
        if not isinstance(self.manager.automl_config.config['optimizer'], partial):
            optimisation_agent = self.manager.automl_config.optimizer['optimisation_agent']
            optimisation_params = self.manager.automl_config.optimizer['optimisation_strategy']
            fedcore_opt = partial(self.manager.optimisation_agent[optimisation_agent],
                                  optimisation_params=optimisation_params)
            self.manager.automl_config.optimizer = fedcore_opt
            self.manager.automl_config.config.update({'optimizer': fedcore_opt})
        return input_data

    def __init_solver(self, input_data: Optional[Union[InputData, np.array]] = None):
        self.logger.info('Initialising solver')
        self.manager.solver = Fedot(**self.manager.automl_config.config,
                                    use_input_preprocessing=False,
                                    use_auto_preprocessing=False)
        initial_assumption = FEDOT_ASSUMPTIONS[self.manager.learning_config.peft_strategy](params=self.manager.learning_config.peft_strategy_params)
        initial_assumption.heads[0].parameters = self.manager.learning_config.peft_strategy_params
        self.manager.solver.params.data.update({'initial_assumption': initial_assumption.build()})
        return input_data

    def __init_dask(self, input_data):
        self.logger.info('-' * 50)
        self.logger.info('Initialising Dask Server')
        dask_server = DaskServer(self.manager.compute_config.distributed)
        self.manager.dask_client = dask_server.client
        self.manager.dask_cluster = dask_server.cluster
        self.logger.info(f'Link Dask Server - {self.manager.dask_client.dashboard_link}')
        self.logger.info('-' * 50)
        return input_data

    def __init_solver_no_evo(self, input_data: Optional[Union[InputData, np.array]] = None):
        self.logger.info('Initialising solver')
        self.manager.solver = Fedot(**self.manager.automl_config.config,
                                    use_input_preprocessing=False,
                                    use_auto_preprocessing=False)
        initial_assumption = FEDOT_ASSUMPTIONS[self.manager.learning_config.peft_strategy](params=self.manager.learning_config.peft_strategy_params)
        self.manager.solver = initial_assumption.build()
        return input_data

    def _process_input_data(self, input_data):
        data_cls = DataCheck(peft_task=self.manager.learning_config.config['peft_strategy'],
                             model=self.manager.automl_config.config['initial_assumption'],
                             learning_params=self.manager.learning_config.learning_strategy_params
                             )
        train_data = Either.insert(input_data).then(deepcopy).then(data_cls.check_input_data).value
        return train_data

    def _pretrain_before_optimise(self, fedot_pipeline: Pipeline, train_data: InputData):
        pretrained_model = fedot_pipeline.fit(train_data)
        fedcore_trainer = fedot_pipeline.operator.root_node.operation.fitted_operation
        path_to_save_pretrain = os.path.join(self.manager.compute_config.config['output_folder'],
                                             f'pretrain_model_checkpoint_at_{fedcore_trainer.epochs}_epoch.pt')
        os.makedirs(self.manager.compute_config.config['output_folder'])
        fedcore_trainer.save_model(path_to_save_pretrain)
        train_data.target = pretrained_model.predict
        return train_data

    def __abstract_predict(self, predict_data, output_mode):
        predict = self.fedcore_model.predict(predict_data, output_mode)
        return predict

    def fit(self, input_data: tuple, manually_done: bool = False, **kwargs):
        """
        Method for training Industrial model.

        Args:
            input_data: tuple with train_features and train_target
            **kwargs: additional parameters

        """

        def fit_function(train_data):
            model_learning_pipeline = FEDOT_ASSUMPTIONS["training"]
            model_learning_pipeline.heads[0].parameters = self.manager.learning_config.config[
                'learning_strategy_params']
            pretrain_before_optimise = self.manager.learning_config.config['learning_strategy'].__contains__(
                'scratch')
            fitted_solver = Maybe.insert(train_data).then(
                lambda data: self._pretrain_before_optimise(model_learning_pipeline.build(),
                                                            data) if pretrain_before_optimise else data). \
                then(self.manager.solver.fit).maybe(None, lambda solver: solver)
            return fitted_solver

        #with exception_handler(Exception, on_exception=self.shutdown, suppress=False):
        self.fedcore_model = Maybe.insert(self._process_input_data(input_data)). \
            then(self.__init_fedcore_backend). \
            then(self.__init_dask). \
            then(self.__init_solver). \
            then(fit_function). \
            maybe(None, lambda solver: solver)
        return self.fedcore_model

    def fit_no_evo(self, input_data: tuple, manually_done=False, **kwargs):
        with exception_handler(Exception, on_exception=self.shutdown, suppress=False):
            x = self._process_input_data(input_data)
            x = self.__init_fedcore_backend(x)
            x = self.__init_dask(x)
            x = self.__init_solver_no_evo(x)
            fitted_solver = self.manager.solver.fit(x)

            # fitted_solver = (
            #     Maybe.insert(self._process_input_data(input_data))
            #     .then(self.__init_fedcore_backend)
            #     .then(self.__init_solver)
            #     .then(self.manager.solver.fit)
            #     .maybe(None, lambda solver: solver)
            # )
        self.optimised_model = fitted_solver.target
        return fitted_solver

    def predict(self, predict_data: tuple, output_mode:str = 'fedcore', **kwargs):
        """
        Method to obtain prediction labels from trained Industrial model.

        Args:
            predict_data: tuple with test_features and test_target

        Returns:
            the array with prediction values

        """
        self.manager.predicted_labels = Maybe.insert(self._process_input_data(predict_data)). \
                                         then(self.__init_fedcore_backend). \
                                         then(lambda data: self.__abstract_predict(data,output_mode)). \
                                         maybe(None, lambda output: output)

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

    def evaluate_metric(
            self,
            predicton: Union[Tensor, np.array],
            target: Union[list, np.array],
            problem: str = "computational",
    ) -> pd.DataFrame:
        """
        Method to calculate metrics.

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

        model_output = predicton.cpu().detach().numpy() if isinstance(predicton, Tensor) else predicton
        model_output_is_probs = all([len(model_output.shape) > 1, model_output.shape[1] > 1])
        if model_output_is_probs:
            labels = np.argmax(model_output, axis=1)
            predicted_probs = model_output

        inference_metric = problem.__contains__("computational")
        if inference_metric:
            if problem.__contains__('fedcore'):
                model_regime = 'model_after'
            else:
                model_regime = 'model_before'
            prediction_dict = dict(model=self.fedcore_model, dataset=target, model_regime=model_regime)
        else:
            prediction_dict = dict(target=target, labels=labels, probs=predicted_probs)
        prediction_dataframe = FEDOT_GET_METRICS[problem](**prediction_dict)
        return prediction_dataframe

    def get_report(self, test_data: InputData):
        def create_df(iterator):
            df_list = []
            for metric_dict, col in iterator:
                df = pd.DataFrame.from_dict(metric_dict)
                df.columns = [f'{col}_{x}' for x in df.columns]
                df_list.append(df)
            return pd.concat(df_list, axis=1)

        eval_regime = ['original', 'fedcore']
        prediction_list = [self.predict(test_data, output_mode=mode) for mode in eval_regime]
        quality_metrics_list = [self.evaluate_metric(predicton=prediction.predict.predict,
                                                     target=test_data.target,
                                                     problem=self.manager.automl_config.problem)
                                for prediction in prediction_list]
        computational_metrics_list = [self.evaluate_metric(predicton=prediction,
                                                           target=test_data.val_dataloader,
                                                           problem=f'computational_{regime}')
                                      for prediction, regime in zip(prediction_list, eval_regime)]

        quality_df = create_df(zip(quality_metrics_list, eval_regime))
        compute_df = create_df(zip(computational_metrics_list, eval_regime))
        return dict(quality_comparasion=quality_df, computational_comparasion=compute_df)

    def load(self, path):
        """Loads saved Industrial model from disk

        Args:
            path (str): path to the model

        """

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
                example_input = next(iter(self.train_data.features.val_dataloader))[
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
