import logging
import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Union
from tqdm import tqdm
import torch
from fedcore.api.utils.checkers_collection import DataCheck
from fedcore.architecture.dataset.prediction_datasets import CustomDatasetForImages
from fedcore.architecture.utils.paths import DEFAULT_PATH_RESULTS as default_path_to_save_results, PROJECT_PATH
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from fedot.api.main import Fedot
from fedot.core.pipelines.pipeline import Pipeline

from fedcore.data.data import CompressionInputData
from fedcore.inference.onnx import ONNXInferenceModel
from fedcore.interfaces.fedcore_optimizer import FedcoreEvoOptimizer
from fedcore.neural_compressor.config import Torch2ONNXConfig
from fedcore.repository.constanst_repository import FEDOT_ASSUMPTIONS, FEDOT_API_PARAMS, FEDOT_TASK
from fedcore.repository.initializer_industrial_models import FedcoreModels
from fedcore.repository.model_repository import default_fedcore_availiable_operation
from fedcore.architecture.utils.loader import collate

warnings.filterwarnings("ignore")


class FedCore(Fedot):
    """This class is used to run Fedot in model compression mode as FedCore.

    Args:
        input_config: dictionary with the parameters of the experiment.
        output_folder: path to the folder where the results will be saved.

    Example:
        First, configure experiment and instantiate FedotIndustrial class::

            from fedot_ind.api.main import FedotIndustrial
            from fedot_ind.tools.loader import DataLoader


            industrial = FedotIndustrial(problem='ts_classification',
                                         use_cache=False,
                                         timeout=15,
                                         n_jobs=2,
                                         logging_level=20)

        Next, download data from UCR archive::

            train_data, test_data = DataLoader(dataset_name='ItalyPowerDemand').load_data()

        Finally, fit the model and get predictions::

            model = industrial.fit(train_features=train_data[0], train_target=train_data[1])
            labels = industrial.predict(test_features=test_data[0])
            probs = industrial.predict_proba(test_features=test_data[0])
            metric = industrial.get_metrics(target=test_data[1], metric_names=['f1', 'roc_auc'])

    """

    def __init__(self, **kwargs):

        # init Fedot and Industrial hyperparams and path to results
        self.output_folder = kwargs.get('output_folder', None)
        self.preprocessing = kwargs.get('fedcore_preprocessing', False)
        self.framework_config = kwargs.get('framework_config', None)
        self.backend_method = kwargs.get('backend', 'cpu')
        self.task_params = kwargs.get('task_params', None)
        self.model_params = kwargs.get('model_params', None)
        self.path_to_composition_results = kwargs.get('history_dir', None)

        # create dirs with results
        prefix = './composition_results' if self.path_to_composition_results is None else \
            self.path_to_composition_results
        Path(prefix).mkdir(parents=True, exist_ok=True)
        # create dirs with results
        if self.output_folder is None:
            self.output_folder = default_path_to_save_results
            Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        else:
            Path(self.output_folder).mkdir(parents=True, exist_ok=True)
            del kwargs['output_folder']
        # init logger
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(name)s - %(message)s',
                            handlers=[logging.FileHandler(Path(self.output_folder) / 'log.log'),
                                      logging.StreamHandler()])
        super(Fedot, self).__init__()
        # init hidden state variables
        self.logger = logging.getLogger('FedCoreAPI')
        self.solver = None
        # map Fedot params to Industrial params
        self.config_dict = kwargs
        self.config_dict['history_dir'] = prefix
        self.config_dict['available_operations'] = kwargs.get('available_operations',
                                                              default_fedcore_availiable_operation(
                                                                  self.config_dict['problem']))

        self.config_dict['optimizer'] = kwargs.get('optimizer', FedcoreEvoOptimizer)
        self.config_dict['initial_assumption'] = kwargs.get('initial_assumption',
                                                            FEDOT_ASSUMPTIONS[self.config_dict['problem']])
        self.__init_experiment_setup()

    def __init_experiment_setup(self):
        self.logger.info('Initialising experiment setup')
        fedcore_params = [param for param in self.config_dict.keys() if param not in list(FEDOT_API_PARAMS.keys())]
        [self.config_dict.pop(x, None) for x in fedcore_params]

    def __init_solver(self):
        self.logger.info('Initialising Industrial Repository')
        self.repo = FedcoreModels().setup_repository()
        self.config_dict['initial_assumption'] = self.config_dict['initial_assumption'].build()
        self.logger.info('Initialising solver')
        self.__init_experiment_setup()
        self.config_dict['problem'] = 'classification'
        solver = Fedot(**self.config_dict)
        return solver

    def fit(self,
            input_data: tuple,
            **kwargs):
        """
        Method for training Industrial model.

        Args:
            input_data: tuple with train_features and train_target
            **kwargs: additional parameters

        """
        if self.config_dict['problem'] == 'detection':
            classes = list(input_data.classes.keys())
            loader = DataLoader(
                input_data, 
                batch_size=1,
                shuffle=False,
                collate_fn=collate
            )
            desc='Fitting'
            for i, (images, targets) in enumerate(tqdm(loader, desc=desc)):
                target = [
                    targets[0]['boxes'],
                    targets[0]['labels']
                ]
                self.train_data = deepcopy([images, target])
                input_preproc = DataCheck(input_data=self.train_data,
                                        task=self.config_dict['problem'],
                                        task_params=self.task_params,
                                        classes=classes,
                                        idx=i)
                self.train_data = input_preproc.check_input_data()
                self.solver = self.__init_solver()
                self.solver.fit(self.train_data)
            
        else:
            self.train_data = deepcopy(input_data)  # we do not want to make inplace changes
            input_preproc = DataCheck(input_data=self.train_data,
                                    task=self.config_dict['problem'],
                                    task_params=self.task_params)
            self.train_data = input_preproc.check_input_data()
            self.solver = self.__init_solver()
            self.solver.fit(self.train_data)

    def predict(self,
                predict_data: tuple,
                **kwargs):
        """
        Method to obtain prediction labels from trained Industrial model.

        Args:
            predict_data: tuple with test_features and test_target

        Returns:
            the array with prediction values

        """
        self.predict_data = deepcopy(
            predict_data)  # we do not want to make inplace changes
        self.predict_data = DataCheck(input_data=self.predict_data,
                                      task=self.config_dict['problem'],
                                      task_params=self.task_params).check_input_data()
        predict = self.solver.predict(self.predict_data)
        return predict

    def finetune(self,
                 train_data,
                 tuning_params=None,
                 mode: str = 'head'):
        """
            Method to obtain prediction probabilities from trained Industrial model.

            Args:
                train_data: raw train data
                tuning_params: dictionary with tuning parameters
                mode: str, ``default='full'``. Defines the mode of fine-tuning. Could be 'full' or 'head'.

            """

        # train_data = DataCheck(input_data=train_data, task=self.config_dict['problem']).check_input_data()
        # tuning_params = {} if tuning_params is None else tuning_params
        # tuned_metric = 0
        # tuning_params['metric'] = FEDOT_TUNING_METRICS[self.config_dict['problem']]
        # for tuner_name, tuner_type in FEDOT_TUNER_STRATEGY.items():
        #     model_to_tune = deepcopy(self.solver.current_pipeline) if isinstance(self.solver, Fedot) \
        #         else deepcopy(self.solver)
        #     tuning_params['tuner'] = tuner_type
        #     pipeline_tuner, model_to_tune = build_tuner(
        #         self, model_to_tune, tuning_params, train_data, mode)
        #     if abs(pipeline_tuner.obtained_metric) > tuned_metric:
        #         tuned_metric = abs(pipeline_tuner.obtained_metric)
        #         self.solver = model_to_tune
        pass

    def get_metrics(self,
                    target: Union[list, np.array] = None,
                    metric_names: tuple = ('f1', 'roc_auc', 'accuracy'),
                    rounding_order: int = 3,
                    **kwargs) -> pd.DataFrame:
        """
        Method to calculate metrics for Industrial model.

        Available metrics for classification task: 'f1', 'accuracy', 'precision', 'roc_auc', 'logloss'.

        Available metrics for regression task: 'r2', 'rmse', 'mse', 'mae', 'median_absolute_error',
        'explained_variance_score', 'max_error', 'd2_absolute_error_score', 'msle', 'mape'.

        Args:
            target: target values
            metric_names: list of metric names
            rounding_order: rounding order for metrics

        Returns:
            pandas DataFrame with calculated metrics

        """
        pass

    def load(self, path):
        """Loads saved Industrial model from disk

        Args:
            path (str): path to the model

        """
        self.repo = FedcoreModels().setup_repository()

        dir_list = os.listdir(path)
        if 'fitted_operations' in dir_list:
            self.solver = Pipeline().load(path)
        else:
            self.solver = []
            for p in dir_list:
                self.solver.append(Pipeline().load(
                    f'{path}/{p}/0_pipeline_saved'))

    def load_data(self, path: str = None, supplementary_data: dict = None):
        if path is None and supplementary_data is not None:
            # load data dynamically
            torch_model = supplementary_data['torch_model']
            torch_dataset = CompressionInputData(features=np.zeros((2, 2)),
                                                 idx=None,
                                                 calib_dataloader=supplementary_data['test_dataset'],
                                                 task=FEDOT_TASK['classification'],
                                                 data_type=None,
                                                 target=torch_model
                                                 )
            torch_dataset.supplementary_data.is_auto_preprocessed = True
        else:
            # load data from directory
            path_to_data = os.path.join(PROJECT_PATH, path)
            dir_list = os.listdir(path_to_data)
            for x in dir_list:
                if x.__contains__('dataset'):
                    directory = os.path.join(path_to_data, x)
                elif x.__contains__('model'):
                    model_dir = os.path.join(path_to_data, x)
                    _ = [y for y in os.listdir(model_dir) if y.__contains__('.pt')][0]
                    path_to_model = os.path.join(model_dir, _)
                else:
                    annotations = os.path.join(path_to_data, x)
            torch_model = torch.load(path_to_model, map_location=torch.device('cpu'))
            torch_dataset = CustomDatasetForImages(annotations=annotations,
                                                   directory=directory)
        return (torch_dataset, torch_model)

    def save_best_model(self):
        if isinstance(self.solver, Fedot):
            return self.solver.current_pipeline.save(path=self.output_folder, create_subdir=True,
                                                     is_datetime_in_path=True)
        elif isinstance(self.solver, Pipeline):
            return self.solver.save(path=self.output_folder, create_subdir=True,
                                    is_datetime_in_path=True)
        else:
            for idx, p in enumerate(self.solver.ensemble_branches):
                Pipeline(p).save(f'./raf_ensemble/{idx}_ensemble_branch', create_subdir=True)
            Pipeline(self.solver.ensemble_head).save(f'./raf_ensemble/ensemble_head', create_subdir=True)

    def convert_model(self, framework: str = 'ONNX', framework_config: dict = None):
        if self.framework_config is None and framework_config is None:
            return self.logger.info('You must specify configuration for model convertation')
        else:
            if framework == 'ONNX':
                framework_config['example_inputs'] = 1
                int8_onnx_config = Torch2ONNXConfig(**framework_config)
                self.solver.export("int8-model.onnx", int8_onnx_config)
                converted_model = ONNXInferenceModel("int8-model.onnx")
        return converted_model
