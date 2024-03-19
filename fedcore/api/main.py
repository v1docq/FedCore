import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Union
import logging

import numpy as np
import pandas as pd

import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn

from fedcore.api.utils.checkers_collection import DataCheck
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.architecture.utils.paths import DEFAULT_PATH_RESULTS as default_path_to_save_results, PROJECT_PATH
from fedot.api.main import Fedot
from fedot.core.pipelines.pipeline import Pipeline
from fedcore.inference.onnx import ONNXInferenceModel
from fedcore.interfaces.fedcore_optimizer import FedcoreEvoOptimizer
from fedcore.neural_compressor.config import Torch2ONNXConfig
from fedcore.repository.constanst_repository import FEDOT_ASSUMPTIONS, FEDOT_API_PARAMS, FEDOT_TASK, FEDCORE_CV_DATASET
from fedcore.repository.model_repository import default_fedcore_availiable_operation, BACKBONE_MODELS
from fedcore.architecture.utils.paths import data_path
from fedcore.data.data import CompressionInputData
from fedcore.repository.initializer_industrial_models import FedcoreModels

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

        # init FedCore hyperparams
        self.compression_task = kwargs.get('compression_task', 'pruning')
        self.cv_task = kwargs.get('cv_task', 'classification')
        self.cv_dataset = FEDCORE_CV_DATASET[self.cv_task]

        # init backend and convertation params
        self.framework_config = kwargs.get('framework_config', None)
        self.backend_method = kwargs.get('backend', 'cpu')

        # init path to results
        self.path_to_composition_results = kwargs.get('history_dir', None)
        self.output_folder = kwargs.get('output_folder', None)

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
        # map Fedot params to FedCore params
        self.config_dict = kwargs
        self.config_dict['history_dir'] = prefix
        self.config_dict['available_operations'] = kwargs.get('available_operations',
                                                              default_fedcore_availiable_operation(
                                                                  self.config_dict['problem']))

        self.config_dict['optimizer'] = kwargs.get('optimizer', FedcoreEvoOptimizer)
        self.config_dict['initial_assumption'] = kwargs.get('initial_assumption',
                                                            FEDOT_ASSUMPTIONS[self.config_dict['problem']])
        self.__init_experiment_setup()

    def _init_pretrain_dataset(self, dataset: str = 'CIFAR10'):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        default_dataset = {'CIFAR10': torchvision.datasets.CIFAR10}
        train_dataset = default_dataset[dataset](data_path(dataset), train=True, download=True,
                                                 transform=transform)
        val_dataset = default_dataset[dataset](data_path(dataset), train=False, download=True,
                                               transform=transform)
        val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [0.1, 0.9])
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=1
        )

        val_dataloader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=1)
        return train_dataloader, val_dataloader

    def _init_pretrain_model(self, model_name):
        model = BACKBONE_MODELS[model_name](pretrained=True).to(default_device())
        model.fc = nn.Linear(512, 10).to(default_device())
        model.train()
        return model

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
        solver = self.config_dict['initial_assumption']
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
        self.train_data = deepcopy(input_data)  # we do not want to make inplace changes
        input_preproc = DataCheck(input_data=self.train_data,
                                  cv_dataset=self.cv_dataset)
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
        self.predict_data = deepcopy(predict_data)  # we do not want to make inplace changes
        self.predict_data = DataCheck(input_data=self.predict_data,
                                      cv_dataset=self.cv_dataset).check_input_data()
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
        pass

    def load_data(self, path: str = None, supplementary_data: dict = None):
        if path is None and supplementary_data is not None:
            train_dataloader, val_dataloader = self._init_pretrain_dataset(dataset=supplementary_data['dataset_name'])
            model = self._init_pretrain_model(supplementary_data['model_name'])
            supplementary_data = {'torch_model': model.cpu(),
                                  'test_dataset': val_dataloader,
                                  'train_dataset': train_dataloader}
            # load data dynamically
            torch_model = supplementary_data['torch_model']
            torch_dataset = CompressionInputData(features=np.zeros((2, 2)),
                                                 num_classes=10,
                                                 calib_dataloader=supplementary_data['test_dataset'],
                                                 train_dataloader=supplementary_data['train_dataset'],
                                                 target=torch_model
                                                 )
            torch_dataset.supplementary_data.is_auto_preprocessed = True
            self.train_data = (torch_dataset, torch_model)
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
                elif x.__contains__('txt'):
                    annotations = os.path.join(path_to_data, x)
            self.train_data = DataCheck(input_data=(directory, annotations, path_to_model),
                                        cv_dataset=self.cv_dataset).check_input_data()
        return self.train_data

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

    def convert_model(self, framework: str = 'ONNX',
                      framework_config: dict = None,
                      supplementary_data: dict = None):
        if self.framework_config is None and framework_config is None:
            return self.logger.info('You must specify configuration for model convertation')
        else:
            if framework == 'ONNX':
                example_input = next(iter(self.train_data.features.calib_dataloader))[0][0]
                self.framework_config['example_inputs'] = torch.unsqueeze(example_input,
                                                                          dim=0)
                onnx_config = Torch2ONNXConfig(**self.framework_config)
                supplementary_data['model_to_export'].export("converted-model.onnx", onnx_config)
                converted_model = ONNXInferenceModel("converted-model.onnx")
        return converted_model
