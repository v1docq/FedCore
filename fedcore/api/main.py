import logging
import os
import warnings
from copy import deepcopy
from datetime import datetime
from copy import deepcopy
from datetime import datetime
from functools import partial
from typing import Union, Optional, Callable
import numpy as np
import pandas as pd
import torch
import torch.nn

from fedcore.api.utils import camel_to_snake
from fedcore.repository.initializer_industrial_models import FedcoreModels
FEDCORE_IMPLEMENTATIONS = FedcoreModels().setup_repository()

from fedot.api.main import Fedot
from fedot.core.data.data import InputData, OutputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from pymonad.either import Either
from pymonad.maybe import Maybe
from torch import Tensor
from torch.utils.data import DataLoader
from fedcore.api.utils.checkers_collection import DataCheck
from fedcore.architecture.abstraction.decorators import DaskServer, exception_handler
from fedcore.data.data import CompressionInputData
from fedcore.inference.onnx import ONNXInferenceModel
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
from fedcore.repository.constanst_repository import (
    FEDOT_API_PARAMS,
    FEDOT_ASSUMPTIONS,
    # FEDOT_GET_METRICS,
)
from fedcore.metrics.quality import calculate_metrics
from fedcore.api.api_configs import ConfigTemplate
from fedcore.interfaces.fedcore_optimizer import FedcoreEvoOptimizer

warnings.filterwarnings("ignore")

# TODO
COMPUTATIONAL_METRICS = ['latency', 'power', 'throughput']


class FedCore(Fedot):
    """High-level FedCore API entrypoint for model compression on top of FEDOT.

    This class wraps :class:`fedot.api.main.Fedot` and extends it with:

    * FedCore-specific model repository and evolutionary optimizer;
    * PEFT-aware initial pipeline construction;
    * compression-aware training, evaluation and reporting;
    * optional Dask-based distributed execution.

    Parameters
    ----------
    api_config : ConfigTemplate
        Top-level API configuration object (see :class:`APIConfigTemplate`).
        Must contain at least ``automl_config``, ``learning_config`` and
        ``compute_config`` sub-configs.
    **kwargs
        Additional overrides that are merged into ``api_config`` via
        :meth:`ConfigTemplate.update`.
    """

    def __init__(self, api_config: ConfigTemplate, **kwargs):
        super(Fedot, self).__init__()
        api_config.update(kwargs)
        self.manager = api_config
        self.logger = logging.Logger('Fedcore')
        self.fedcore_model = None

    def __init_fedcore_backend(self, input_data: Optional[InputData] = None):
        """Initialize FedCore repository and evolutionary optimizer.

        This method:

        * logs backend initialization;
        * attaches FedCore implementations repository to ``self.repo``;
        * configures :class:`FedcoreEvoOptimizer` as the FEDOT optimizer
          if one is not already provided in ``automl_config``.

        Parameters
        ----------
        input_data : InputData, optional
            Input data being passed through the initialization chain.

        Returns
        -------
        InputData or None
            The same ``input_data`` object, for use in monadic chains.
        """
        self.logger.info('-' * 50)
        self.logger.info('Initialising Fedcore Repository')
        self.logger.info('Initialising Fedcore Evolutionary Optimisation params')
        self.repo = FEDCORE_IMPLEMENTATIONS

        if not isinstance(self.manager.automl_config.optimizer, partial):
            fedcore_opt = partial(
                FedcoreEvoOptimizer,
                optimisation_params={
                    'mutation_strategy': self.manager.automl_config.mutation_strategy,
                    'mutation_agent': self.manager.automl_config.mutation_agent
                }
            )
            self.manager.automl_config.optimizer = fedcore_opt
            self.manager.automl_config.fedot_config.optimizer = fedcore_opt
            # self.manager.automl_config.config.update({'optimizer': fedcore_opt})
        return input_data

    def __init_solver(self, input_data: Optional[Union[InputData, np.array]] = None):
        """Initialize FEDOT solver with FedCore-specific initial assumption.

        This method:

        * creates :class:`Fedot` instance using ``automl_config.fedot_config``;
        * builds initial pipeline according to PEFT strategy;
        * injects this pipeline into FEDOT solver parameters as
          ``initial_assumption``.

        Parameters
        ----------
        input_data : InputData or numpy.ndarray, optional
            Input data being passed through the initialization chain.

        Returns
        -------
        InputData or numpy.ndarray or None
            The same ``input_data`` object, for chaining.
        """
        self.logger.info('Initialising solver')
        self.manager.solver = Fedot(
            **self.manager.automl_config.fedot_config,
            use_input_preprocessing=False,
            use_auto_preprocessing=False
        )
        # initial_assumption = FEDOT_ASSUMPTIONS[self.manager.learning_config.peft_strategy]
        # initial_assumption = initial_assumption(
        #     params=self.manager.learning_config.peft_strategy_params.to_dict())
        initial_pipeline = self.__build_assumption()
        self.manager.solver.params.data.update({'initial_assumption': initial_pipeline})
        return input_data

    def __init_dask(self, input_data):
        """Initialize Dask server and attach client/cluster to the manager.

        Parameters
        ----------
        input_data :
            Input data being passed through the initialization chain.

        Returns
        -------
        Any
            The same ``input_data`` object, for chaining.
        """
        self.logger.info('-' * 50)
        self.logger.info('Initialising Dask Server')
        dask_server = DaskServer(self.manager.compute_config.distributed)
        self.manager.dask_client = dask_server.client
        self.manager.dask_cluster = dask_server.cluster
        self.logger.info(f'Link Dask Server - {self.manager.dask_client.dashboard_link}')
        self.logger.info('-' * 50)
        return input_data

    def __init_solver_no_evo(self, input_data: Optional[Union[InputData, np.array]] = None):
        """Initialize solver without evolutionary optimization.

        Instead of a full FEDOT solver, this method builds an initial
        pipeline via :meth:`__build_assumption` and assigns it directly
        to ``self.manager.solver``.

        Parameters
        ----------
        input_data : InputData or numpy.ndarray, optional
            Input data being passed through the initialization chain.

        Returns
        -------
        InputData or numpy.ndarray or None
            The same ``input_data`` object, for chaining.
        """
        self.logger.info('Initialising solver')
        # self.manager.solver = Fedot(**self.manager.automl_config.fedot_config,
        #                             use_input_preprocessing=False,
        #                             use_auto_preprocessing=False)
        self.manager.solver = self.__build_assumption()
        return input_data

    def __build_assumption(self):
        """Build initial PEFT-aware pipeline assumption.

        The pipeline is constructed based on the configuration in
        ``self.manager.learning_config.peft_strategy_params``. Each
        PEFT config is converted into a corresponding FEDOT node with
        ``*_model`` operation type and parameters from ``to_dict()``.

        Returns
        -------
        Pipeline
            FEDOT :class:`Pipeline` object that is used as an initial
            assumption for optimization.
        """
        initial_assumption = PipelineBuilder()
        peft_strategy_params = self.manager.learning_config.peft_strategy_params
        # check if atomized strategy
        if not isinstance(peft_strategy_params, (list, tuple)):
            peft_strategy_params = (peft_strategy_params,)
        for peft_strategy_conf in peft_strategy_params:
            initial_assumption.add_node(
                operation_type=camel_to_snake(peft_strategy_conf.__class__.__name__) + '_model',
                params=peft_strategy_conf.to_dict()
            )

        return initial_assumption.build()

    def _process_input_data(self, input_data):
        """Validate and normalize input data for compression pipeline.

        This method:

        * runs :class:`DataCheck` to ensure that input data and learning
          parameters are consistent;
        * attaches ``train_dataloader`` and ``val_dataloader`` shortcuts
          for downstream components.

        Parameters
        ----------
        input_data :
            Raw input data, typically :class:`CompressionInputData` or
            :class:`InputData`.

        Returns
        -------
        Any
            Processed data object with attached dataloaders.
        """
        data_cls = DataCheck(
            # peft_task=self.manager.learning_config.config['peft_strategy'],
            model=self.manager.automl_config.fedot_config['initial_assumption'],
            learning_params=self.manager.learning_config.learning_strategy_params
        )
        train_data = Either.insert(input_data).then(data_cls.check_input_data).value
        # TODO: remove workaround when DataCheck returns consistent interface
        train_data.train_dataloader = train_data.features.train_dataloader
        train_data.val_dataloader = train_data.features.val_dataloader
        return train_data

    def _pretrain_before_optimise(self, fedot_pipeline: Pipeline, train_data: InputData):
        """Optionally pretrain model before running evolutionary optimization.

        The pipeline is first fitted on ``train_data``, then the underlying
        FedCore trainer is used to save a checkpoint. After that, the fitted
        model's predictions are injected back into ``train_data.target`` for
        further optimization.

        Parameters
        ----------
        fedot_pipeline : Pipeline
            FEDOT pipeline used for pretraining.
        train_data : InputData
            Training data to be used for pretraining.

        Returns
        -------
        InputData
            Modified training data with updated ``target``.
        """
        pretrained_model = fedot_pipeline.fit(train_data)
        fedcore_trainer = fedot_pipeline.operator.root_node.operation.fitted_operation
        path_to_save_pretrain = os.path.join(self.manager.compute_config.output_folder)
        os.makedirs(path_to_save_pretrain, exist_ok=True)
        path_to_model = os.path.join(
            path_to_save_pretrain,
            f'pretrain_model_checkpoint_at_{fedcore_trainer.epochs}_epoch.pt'
        )
        fedcore_trainer.save_model(path_to_model)
        train_data.target = pretrained_model.predict
        return train_data

    def __abstract_predict(self, predict_data: InputData, output_mode):
        """Run prediction using internal FedCore model abstraction.

        If ``self.fedcore_model`` is not yet initialized, it is created
        as :class:`BaseNeuralModel` with learning parameters taken from
        ``peft_strategy_params`` and the model from ``predict_data.target``.

        Parameters
        ----------
        predict_data : InputData
            Data to predict on.
        output_mode : str
            Output mode passed to :meth:`BaseNeuralModel.predict`.

        Returns
        -------
        Any
            Prediction object produced by :class:`BaseNeuralModel`.
        """
        if self.fedcore_model is None:
            learning_params = self.manager.learning_config.peft_strategy_params.to_dict()
            learning_params['model'] = predict_data.target
            # scenario where we load pretrain model and use it only for inference
            self.fedcore_model = BaseNeuralModel(learning_params)
            # predict_data = predict_data.features # InputData to CompressionInputData
        predict = self.fedcore_model.predict(predict_data, output_mode)
        return predict

    def fit(self, input_data: CompressionInputData, manually_done: bool = False, **kwargs):
        """
        Method for training Industrial model with evolutionary optimization.

        The training pipeline typically includes:

        * data validation and preprocessing;
        * FedCore backend initialization (repository, optimizer);
        * optional Dask setup;
        * FEDOT solver initialization;
        * optional pretraining step (depending on ``learning_strategy``);
        * evolutionary optimization via FEDOT.

        Parameters
        ----------
        input_data : CompressionInputData
            Input data for training (features, target, loaders).
        manually_done : bool, optional
            Reserved for compatibility; not used.
        **kwargs
            Additional keyword arguments (currently ignored).

        Returns
        -------
        Any
            Trained FEDOT solver or FedCore model, depending on the pipeline.
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
            return self.fedcore_model
        except KeyboardInterrupt:
            self.fedcore_model = self.manager.solver
            return self.fedcore_model

    def fit_no_evo(self, input_data: tuple, manually_done=False, **kwargs):
        """Train model without evolutionary search.

        This variant uses a fixed initial assumption (built from PEFT
        configuration) and runs a single FEDOT ``fit`` call without
        running the evolutionary optimizer.

        Parameters
        ----------
        input_data : tuple
            Training data (features, target) in FEDOT-compatible format.
        manually_done : bool, optional
            Reserved for compatibility; not used.
        **kwargs
            Additional keyword arguments (currently ignored).

        Returns
        -------
        Any
            Fitted solver or pipeline, depending on the backend.
        """
        with exception_handler(Exception, on_exception=self.shutdown, suppress=False):
            x = self._process_input_data(input_data)
            x = self.__init_fedcore_backend(x)
            x = self.__init_dask(x)
            x = self.__init_solver_no_evo(x)
            fitted_solver = self.manager.solver.fit(x)
        self.optimised_model = fitted_solver.target
        return fitted_solver

    def predict(self, predict_data: tuple, output_mode: str = 'fedcore', **kwargs):
        """
        Method to obtain prediction labels from trained Industrial model.

        Parameters
        ----------
        predict_data : tuple
            Data to predict on (features, target) in FEDOT-compatible format.
        output_mode : str, default='fedcore'
            Output mode passed to the underlying FedCore predictor.
        **kwargs
            Additional keyword arguments (currently ignored).

        Returns
        -------
        Any
            Prediction object produced by the FedCore model.
        """
        self.manager.predicted_labels = Maybe.insert(self._process_input_data(predict_data)). \
            then(self.__init_fedcore_backend). \
            then(lambda data: self.__abstract_predict(data, output_mode)). \
            maybe(None, lambda output: output)

        return self.manager.predicted_labels

    def finetune(self, train_data, tuning_params=None):
        """
        Method to fine-tune a trained Industrial model.

        Depending on the configuration, this can either return a fitted
        model directly or construct and run a FEDOT tuner.

        Parameters
        ----------
        train_data :
            Raw training data for fine-tuning.
        tuning_params : dict, optional
            Dictionary with tuning parameters, including tuner type and
            metric configuration.
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
        Method to calculate metrics for quality and computational evaluation.

        Available metrics for classification task: 'f1', 'accuracy',
        'precision', 'roc_auc', 'logloss'.

        Available metrics for regression task: 'r2', 'rmse', 'mse', 'mae',
        'median_absolute_error', 'explained_variance_score', 'max_error',
        'd2_absolute_error_score', 'msle', 'mape'.

        For ``problem`` values containing ``"computational"`` the method
        switches to inference/computational metrics and uses the FedCore
        model plus dataset instead of label predictions.

        Parameters
        ----------
        prediction : OutputData
            FEDOT prediction object (or compatible wrapper).
        target : DataLoader
            Ground truth target or inference dataset.
        problem : str, default="computational"
            Problem type string; determines whether to compute quality or
            computational metrics.
        metrics : list of str, default=['latency']
            List of metric names to compute.

        Returns
        -------
        pandas.DataFrame
            DataFrame with calculated metrics.
        """
        is_inference_metric = problem.__contains__("computational")
        is_fedcore_model = problem.__contains__('fedcore')
        model_regime = 'model_after' if is_fedcore_model else 'model_before'
        prediction_dict = dict(target=target, predict=prediction.predict)
        if is_inference_metric:
            prediction_dict = dict(model=self.fedcore_model, dataset=target, model_regime=model_regime)
            # preproc_target = preproc_target(target)
        metrics = metrics or self.manager.automl_config.fedot_config.metric
        prediction_dataframe = calculate_metrics(metrics, **prediction_dict)
        return prediction_dataframe

    def get_report(self, test_data: CompressionInputData):
        """Build comparison report for original vs FedCore-compressed model.

        The report includes both quality and computational metrics, and
        aggregates:

        * metrics for the original model;
        * metrics for the FedCore-compressed model;
        * relative changes between them (in percents).

        Parameters
        ----------
        test_data : CompressionInputData
            Data used to evaluate both original and compressed models.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Dictionary with two tables:

            * ``"quality_comparison"`` – quality metrics comparison;
            * ``"computational_comparison"`` – computational metrics
              comparison.
        """
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
        quality_metrics_list  = [name for name in self.manager.automl_config.fedot_config.metrics if name not in COMPUTATIONAL_METRICS]
        computational_metrics = [name for name in self.manager.automl_config.fedot_config.metrics if name in COMPUTATIONAL_METRICS]
        quality_metrics_list = [self.evaluate_metric(prediction=prediction,
                                                     target=test_data.val_dataloader,
                                                     problem=self.manager.automl_config.fedot_config.problem,
                                                     metrics=quality_metrics_list)
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
        """Load a previously saved Industrial model from disk.

        Parameters
        ----------
        path : str
            Filesystem path to the saved model artifact.

        Notes
        -----
        The actual loading logic is not implemented yet and should be
        implemented according to the chosen serialization format.
        """
        """Loads saved Industrial model from disk

        Args:
            path (str): path to the model

        """

    def save(self, mode: str = 'all', **kwargs):
        """Save models, metrics and optimization artifacts to disk.

        Depending on the selected ``mode``, this method can save:

        * ``"model"`` – FEDOT pipeline or FedCore model checkpoint;
        * ``"metrics"`` – metrics CSV file;
        * ``"opt_hist"`` – optimization history (if available);
        * ``"prediction"`` – predicted labels;
        * ``"all"`` – everything listed above.

        Parameters
        ----------
        mode : {'all', 'metrics', 'model', 'opt_hist', 'prediction'}, default='all'
            What exactly should be saved.
        **kwargs
            Additional keyword arguments (currently ignored).
        """
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
        """Export trained model to an external inference framework.

        Currently only ONNX export is supported. The method prepares
        example inputs, builds ONNX conversion config and exports the
        model using :class:`ONNXInferenceModel`.

        Parameters
        ----------
        framework : str, default="ONNX"
            Target framework identifier.
        framework_config : dict, optional
            Framework-specific configuration for export (e.g. ONNX
            conversion parameters).
        supplementary_data : dict, optional
            Additional objects required for export (e.g. model instance
            under ``'model_to_export'`` key).

        Returns
        -------
        ONNXInferenceModel or None
            Inference wrapper for the exported ONNX model, or ``None`` if
            export configuration is missing.
        """
        if self.framework_config is None and framework_config is None:
            return self.logger.info(
                "You must specify configuration for model convertation"
            )
        else:
            if framework == "ONNX":
                example_input = next(iter(self.train_data.features.val_dataloader))[0][0]
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
        """Shutdown Dask resources attached to the FedCore manager.

        This method safely closes and deletes Dask client and cluster
        objects if they are present in ``self.manager``.
        """
        # if self.manager.dask_client is not None:
        if hasattr(self.manager, 'dask_client'):
            self.manager.dask_client.close()
            del self.manager.dask_client
        if hasattr(self.manager, 'dask_cluster'):
            # if self.manager.dask_cluster is not None:
            self.manager.dask_cluster.close()
            del self.manager.dask_cluster
