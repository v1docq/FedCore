import logging
import os
import warnings
from copy import deepcopy
from datetime import datetime
from functools import partial
from typing import Union, Optional, Callable
import numpy as np
import pandas as pd
import torch
import torch.nn
from fedot.api.main import Fedot
from fedot.core.data.data import InputData, OutputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum

from pymonad.either import Either
from pymonad.maybe import Maybe
from torch import Tensor
from torch.utils.data import DataLoader
from fedcore.api.utils.checkers_collection import DataCheck
from fedcore.architecture.abstraction.decorators import DaskServer, exception_handler
from fedcore.data.data import CompressionInputData
from fedcore.inference.onnx import ONNXInferenceModel
from fedcore.models.network_impl.utils.trainer_factory import create_trainer
from fedcore.repository.constant_repository import (
    FEDOT_API_PARAMS,
    FEDOT_ASSUMPTIONS,
    FEDOT_GET_METRICS,
)
from fedcore.repository.initializer_industrial_models import FedcoreModels
from fedcore.api.api_configs import ConfigTemplate
from fedcore.interfaces.fedcore_optimizer import FedcoreEvoOptimizer
from fedcore.tools.registry.model_registry import ModelRegistry
from fedcore.api.utils.misc import extract_fitted_operation

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

    def __init__(self, api_config: ConfigTemplate, **kwargs):
        super(Fedot, self).__init__()
        api_config.update(kwargs)
        self.manager = api_config
        self.logger = logging.Logger('Fedcore')
        self.fedcore_model = None

    def __init_fedcore_backend(self, input_data: Optional[InputData] = None):
        self.logger.info('-' * 50)
        self.logger.info('Initialising Fedcore Repository')
        self.logger.info('Initialising Fedcore Evolutionary Optimisation params')
        self.repo = FedcoreModels().setup_repository()
        if not isinstance(self.manager.automl_config.optimizer, partial):
            fedcore_opt = partial(FedcoreEvoOptimizer, optimisation_params={
                'mutation_strategy': self.manager.automl_config.mutation_strategy,
                'mutation_agent': self.manager.automl_config.mutation_agent})
            self.manager.automl_config.optimizer = fedcore_opt
            self.manager.automl_config.fedot_config.optimizer = fedcore_opt
            # self.manager.automl_config.config.update({'optimizer': fedcore_opt})
        return input_data

    def __init_solver(self, input_data: Optional[Union[InputData, np.array]] = None):
        self.logger.info('Initialising solver')
        self.manager.solver = Fedot(**self.manager.automl_config.fedot_config,
                                    use_input_preprocessing=False,
                                    use_auto_preprocessing=False)
        initial_assumption = FEDOT_ASSUMPTIONS[self.manager.learning_config.peft_strategy]
        initial_assumption = initial_assumption(
            params=self.manager.learning_config.peft_strategy_params.to_dict())
        initial_pipeline = initial_assumption.build()
        self.manager.solver.params.data.update({'initial_assumption': initial_pipeline})
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
        self.manager.solver = Fedot(**self.manager.automl_config.fedot_config,
                                    use_input_preprocessing=False,
                                    use_auto_preprocessing=False)
        initial_assumption = FEDOT_ASSUMPTIONS[self.manager.learning_config.peft_strategy](
            params=self.manager.learning_config.peft_strategy_params.to_dict()
        )
        self.manager.solver = initial_assumption.build()
        return input_data

    @property
    def compressed_model(self):
        """Get compressed (optimized) model.
        Returns:
            torch.nn.Module or None: Compressed model
        """
        if self.fedcore_model is None:
            return None
        return getattr(self.fedcore_model, 'model_after', self.fedcore_model)
    
    @property
    def original_model(self):
        """Get original (before compression) model.        
        Returns:
            torch.nn.Module or None: Original model
        """
        if self.fedcore_model is None:
            return None
        return getattr(self.fedcore_model, 'model_before', self.fedcore_model)
    
    def get_model_by_regime(self, regime: str = 'model_after'):
        """Get model by regime name.
        Args:
            regime: 'model_after' for compressed, 'model_before' for original
            
        Returns:
            torch.nn.Module: Requested model or fallback to fedcore_model
            
        Raises:
            ValueError: If fedcore_model is not initialized
        """
        if self.fedcore_model is None:
            raise ValueError("fedcore_model is not initialized. Call fit() first.")
        
        model = getattr(self.fedcore_model, regime, None)
        if model is None:
            self.logger.warning(
                f"Regime '{regime}' not found in fedcore_model. "
                f"Using fedcore_model directly."
            )
            model = self.fedcore_model
        return model

    def _save_metrics_from_evaluator(self):
        """Collect and save metrics from evaluator to registry after fit."""        
        if not hasattr(self.manager, 'solver') or self.manager.solver is None:
            return
        
        if not hasattr(self.manager.solver, 'history') or self.manager.solver.history is None:
            return
        
        fedcore_id = None
        model_id = None
        
        if self.fedcore_model is not None:
            if hasattr(self.fedcore_model, 'operator') and hasattr(self.fedcore_model.operator, 'root_node'):
                fitted_op = getattr(self.fedcore_model.operator.root_node, 'fitted_operation', None)
                if fitted_op is not None:
                    fedcore_id = getattr(fitted_op, '_fedcore_id', None)
                    if fedcore_id:
                        model_id = getattr(fitted_op, '_model_id_after', None) or getattr(fitted_op, '_model_id_before', None)
        
        if fedcore_id and model_id:
            registry = ModelRegistry()
            registry.save_metrics_from_evaluator(
                solver=self.manager.solver,
                fedcore_id=fedcore_id,
                model_id=model_id
            )

    def _process_input_data(self, input_data):
        data_cls = DataCheck(peft_task=self.manager.learning_config.config['peft_strategy'],
                             model=self.manager.automl_config.fedot_config['initial_assumption'],
                             learning_params=self.manager.learning_config.learning_strategy_params
                             )
        train_data = Either.insert(input_data).then(data_cls.check_input_data).value
        ### TODO del workaround
        train_data.train_dataloader = train_data.features.train_dataloader
        train_data.val_dataloader = train_data.features.val_dataloader
        ###
        return train_data

    def _pretrain_before_optimise(self, fedot_pipeline: Pipeline, train_data: InputData):
        pretrained_model = fedot_pipeline.fit(train_data)
        fedcore_trainer = fedot_pipeline.operator.root_node.operation.fitted_operation
        path_to_save_pretrain = os.path.join(self.manager.compute_config.output_folder)
        os.makedirs(path_to_save_pretrain, exist_ok=True)
        path_to_model = os.path.join(path_to_save_pretrain,
                                     f'pretrain_model_checkpoint_at_{fedcore_trainer.epochs}_epoch.pt')
        fedcore_trainer.save_model(path_to_model)
        train_data.target = pretrained_model.predict
        return train_data

    def __abstract_predict(self, predict_data: InputData, output_mode):
        if self.fedcore_model is None:
            learning_params = self.manager.learning_config.peft_strategy_params.to_dict()
            learning_params['model'] = predict_data.target
            # scenario where we load pretrain model and use it only for inference
            task_type = learning_params.get('task_type', 'training')
            self.fedcore_model = create_trainer(task_type=task_type, params=learning_params, model=learning_params['model'])
            #predict_data = predict_data.features # InputData to CompressionInputData
        predict = self.fedcore_model.predict(predict_data, output_mode)
        return predict

    def fit(self, input_data: CompressionInputData, manually_done: bool = False, **kwargs):
        """
        Method for training Industrial model.

        Args:
            input_data: tuple with train_features and train_target
            **kwargs: additional parameters

        """

        def fit_function(train_data):
            pretrain_before_optimise = self.manager.learning_config.config['learning_strategy'] == 'from_scratch'
            if pretrain_before_optimise:
                model_learning_pipeline = FEDOT_ASSUMPTIONS["training"](
                    params=self.manager.learning_config.learning_strategy_params.to_dict()
                )
                model_learning_pipeline = model_learning_pipeline.build()
                train_data = self._pretrain_before_optimise(model_learning_pipeline, train_data)
            fitted_solver = self.manager.solver.fit(train_data)
            return fitted_solver

        # with exception_handler(Exception, on_exception=self.shutdown, suppress=False):
        try:
            self.fedcore_model = Maybe.insert(self._process_input_data(input_data)). \
                then(self.__init_fedcore_backend). \
                then(self.__init_dask). \
                then(self.__init_solver). \
                then(fit_function). \
                maybe(None, lambda solver: solver)
            
            self._save_metrics_from_evaluator()
            
            return self.fedcore_model
        except KeyboardInterrupt:
            self.fedcore_model = self.manager.solver
            self._save_metrics_from_evaluator()
            return self.fedcore_model

    def fit_no_evo(self, input_data: tuple, manually_done=False, **kwargs):
        with exception_handler(Exception, on_exception=self.shutdown, suppress=False):
            x = self._process_input_data(input_data)
            x = self.__init_fedcore_backend(x)
            x = self.__init_dask(x)
            x = self.__init_solver_no_evo(x)
            fitted_solver = self.manager.solver.fit(x)
        self.optimised_model = fitted_solver.model
        
        self.fedcore_model = extract_fitted_operation(self.manager.solver)
        return fitted_solver

    def predict(self, predict_data: tuple, output_mode: str = 'fedcore', **kwargs):
        """
        Method to obtain prediction labels from trained Industrial model.

        Args:
            predict_data: tuple with test_features and test_target

        Returns:
            the array with prediction values

        """
        result = Maybe.insert(self._process_input_data(predict_data)). \
            then(self.__init_fedcore_backend). \
            then(lambda data: self.__abstract_predict(data, output_mode)). \
            maybe(None, lambda output: output)
        
        if hasattr(result, 'predictions') and hasattr(result, 'label_ids'):
            pred_values = torch.tensor(result.predictions)
            target_values = torch.tensor(result.label_ids) if result.label_ids is not None else None
            
            self.manager.predicted_labels = OutputData(
                idx=torch.arange(len(pred_values)),
                task=getattr(predict_data, 'task', None),
                predict=pred_values,
                target=target_values,
                data_type=DataTypesEnum.table,
            )
            
        elif isinstance(result, OutputData):
            self.manager.predicted_labels = result
        elif hasattr(result, 'predict'):
            pred_value = result.predict if result.predict is not None else getattr(result, 'model', None)
            if pred_value is None:
                raise ValueError("Result has 'predict' attribute but it is None, and 'model' is also unavailable")
            self.manager.predicted_labels = pred_value if isinstance(pred_value, OutputData) else result
        else:
            pred_values = torch.tensor(result) if not isinstance(result, torch.Tensor) else result
            self.manager.predicted_labels = OutputData(
                idx=torch.arange(len(pred_values)),
                task=getattr(predict_data, 'task', None),
                predict=pred_values,
                target=None,
                data_type=DataTypesEnum.table,
            )

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
        tuning_params['metric'] = FEDOT_TUNING_METRICS[self.manager.automl_config.fedot_config['task']]
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
            prediction: OutputData,
            target: DataLoader,
            problem: str = "computational",
            metrics: list = ['latency']
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
        is_inference_metric = problem.__contains__("computational")
        is_fedcore_model = problem.__contains__('fedcore')
        model_regime = 'model_after' if is_fedcore_model else 'model_before'

        def preproc_predict(prediction):
            prediction = prediction.predict
            model_output = prediction.cpu().detach().numpy() if isinstance(prediction, Tensor) else prediction
            model_output_is_probs = all([len(model_output.shape) > 1, model_output.shape[1] > 1])
            if model_output_is_probs and not self.manager.automl_config.fedot_config.problem.__contains__(
                    'forecasting'):
                labels = np.argmax(model_output, axis=1)
                predicted_probs = model_output
            else:
                labels = model_output
                predicted_probs = model_output
            return labels, predicted_probs

        def preproc_target(target):
            if hasattr(target, 'targets'):
                target = target.dataset.targets
            # else:
            #     iter_object = iter(target.dataset)
            #     target = np.array([batch[1] for batch in iter_object])
            # return target
            all_targets = []
            for batch in target:
                labels = None
                if isinstance(batch, dict):
                    labels = batch.get('labels')
                    if labels is None:
                        labels = batch.get('targets')
                elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    labels = batch[1]
                elif hasattr(batch, 'labels'):
                    labels = batch.labels
                
                if labels is not None:
                    all_targets.append(labels.cpu())
            
            return torch.cat(all_targets) if all_targets else None         
        
        if is_inference_metric:
            model_to_evaluate = self.get_model_by_regime(model_regime)
            prediction_dict = dict(model=model_to_evaluate, dataset=target, model_regime=model_regime)
        else:
            preproc_labels, preproc_probs = preproc_predict(prediction)
            preproc_target = preproc_target(target)
            prediction_dict = dict(target=preproc_target, labels=preproc_labels, probs=preproc_probs, metric_names=metrics)
        prediction_dataframe = FEDOT_GET_METRICS[problem](**prediction_dict)
        
        if is_inference_metric:
            registry = ModelRegistry()
            registry.force_cleanup()
        
        return prediction_dataframe

    def get_report(self, test_data: CompressionInputData):
        def create_df(iterator):
            df_list = []
            for metric_dict, col in iterator:
                if isinstance(metric_dict, dict):
                    df = pd.DataFrame.from_dict(metric_dict).T
                elif isinstance(metric_dict, pd.DataFrame):
                    df = metric_dict
                    df = df.T
                else:
                    raise TypeError('Unknown type of metrics passed')
                df['mode'] = col
                df_list.append(df)
            df_total = pd.concat(df_list, axis=0)
            return df_total

        def calculate_metric_changes(metric_df: pd.DataFrame, metric: list = None):
            orig = metric_df[(metric_df['mode'] != 'fedcore')].drop('mode', axis=1)
            opt = metric_df[metric_df['mode'] == 'fedcore'].drop('mode', axis=1)
            change_val = ((opt - orig) / orig * 100).round(2)
            change_val['mode'] = 'change'
            return change_val

        eval_regime = ['original', 'fedcore']
        prediction_list = [self.predict(test_data, output_mode=mode) for mode in eval_regime]
        prediction_list = [x if isinstance(x, OutputData) else x.predict for x in prediction_list]
        problem = self.manager.automl_config.fedot_config.problem
        if any([problem == 'ts_forecasting', problem == 'regression']):
            quality_metrics = ["r2", "mse", "rmse", "mae", "msle", "mape", 
                               "median_absolute_error", "explained_variance_score", 
                               "max_error", "d2_absolute_error_score"]
        else:
            quality_metrics = ["accuracy", "f1", "precision"]
        computational_metrics = ["latency", "throughput"]
        quality_metrics_list = [self.evaluate_metric(prediction=prediction,
                                                     target=test_data.val_dataloader,
                                                     problem=self.manager.automl_config.fedot_config.problem,
                                                     metrics=quality_metrics)
                                for prediction in prediction_list]
        computational_metrics_list = [self.evaluate_metric(prediction=prediction,
                                                           target=test_data.val_dataloader,
                                                           problem=f'computational_{regime}',
                                                           metrics=computational_metrics)
                                      for prediction, regime in zip(prediction_list, eval_regime)]
        
        quality_df = create_df(zip(quality_metrics_list, eval_regime))
        compute_df = create_df(zip(computational_metrics_list, eval_regime))
        result = dict(quality_comparison=quality_df, computational_comparison=compute_df)
        for tp, df in result.items():
            result[tp] = (pd.concat([df, calculate_metric_changes(df)], axis=0)
                          .reset_index()
                          .rename(columns={'index': 'metric'})
                          .pivot(index='metric', columns='mode')
                          .reindex(columns=['original', 'fedcore', 'change'], level=1)
            )
        return result

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
        # if self.framework_config is None and framework_config is None:
        #     return self.logger.info(
        #         "You must specify configuration for model convertation"
        #     )
        # else:
        #     if framework == "ONNX":
        #         example_input = next(iter(self.train_data.features.val_dataloader))[
        #             0
        #         ][0]
        #         self.framework_config["example_inputs"] = torch.unsqueeze(
        #             example_input, dim=0
        #         )
        #         onnx_config = Torch2ONNXConfig(**self.framework_config)
        #         supplementary_data["model_to_export"].export(
        #             "converted-model.onnx", onnx_config
        #         )
        #         converted_model = ONNXInferenceModel("converted-model.onnx")
        # return converted_model
        pass

    def shutdown(self):
        """Shutdown Dask client"""
        # if self.manager.dask_client is not None:
        if hasattr(self.manager, 'dask_client'):
            self.manager.dask_client.close()
            del self.manager.dask_client
        if hasattr(self.manager, 'dask_cluster'):
            # if self.manager.dask_cluster is not None:
            self.manager.dask_cluster.close()
            del self.manager.dask_cluster