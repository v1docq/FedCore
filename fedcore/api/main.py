import warnings
from copy import deepcopy
from typing import Union
import numpy as np
import pandas as pd
import torch
import torch.nn
from fedot.api.main import Fedot
from fedot.core.pipelines.pipeline import Pipeline
from pymonad.either import Either
from torch import Tensor
from fedcore.api.utils.api_init import ApiManager
from fedcore.api.utils.checkers_collection import DataCheck
from fedcore.architecture.abstraction.decorators import DaskServer
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
        self.manager = ApiManager(**kwargs)
        # init FedCore hyperparams
        self.cv_dataset = FEDCORE_CV_DATASET[self.manager.automl_config.task]
        self.logger = self.manager.logger
        # self.cluster_params = self.manager.dask_cluster_params
        self.solver = self.manager.solver

    def __init_solver(self):
        self.logger.info("Initialising FedCore Repository")
        self.repo = FedcoreModels().setup_repository()
        self.logger.info("Initialising solver")

        if self.manager.compute_config.distributed is not None:
            self.dask_client = DaskServer().client
            self.logger.info(f"LinK Dask Server - {self.dask_client.dashboard_link}")
            self.logger.info(f"-------------------------------------------------")

        if self.manager.automl_config.use_automl:
            solver = Fedot(**self.manager.automl_config.config)
        else:
            solver = self.manager.automl_config.initial_assumption

        return solver

    def fit(self, input_data: tuple, manually_done: bool = False, **kwargs):
        """
        Method for training Industrial model.

        Args:
            input_data: tuple with train_features and train_target
            **kwargs: additional parameters

        """

        self.train_data = deepcopy(input_data)  # we do not want to make inplace changes TODO it copies not only model, but dataloaders and datasets.
        self.original_model = input_data[1]
        input_preproc = DataCheck(input_data=self.train_data, task=self.cv_task)
        self.train_data = input_preproc.check_input_data(manually_done)
        self.solver = self.__init_solver()
        if self.need_fedot_pretrain:
            fedcore_training = FEDOT_ASSUMPTIONS["training"].build()
            pretrained_model = fedcore_training.fit(self.train_data)
            self.train_data.target = pretrained_model.predict
        self.solver.fit(self.train_data)
        self.optimised_model = self.solver.root_node.fitted_operation.optimised_model
        # self.original_model = self.solver.root_node.fitted_operation.model

    def predict(self, predict_data: tuple, **kwargs):
        """
        Method to obtain prediction labels from trained Industrial model.

        Args:
            predict_data: tuple with test_features and test_target

        Returns:
            the array with prediction values

        """
        self.predict_data = deepcopy(
            predict_data
        )  # we do not want to make inplace changes
        self.predict_data = DataCheck(
            input_data=self.predict_data, task=self.cv_task
        ).check_input_data()
        output = self.solver.predict(self.predict_data, **kwargs)
        return output.predict

    def finetune(self, train_data, tuning_params=None, mode: str = "head"):
        """
        Method to obtain prediction probabilities from trained Industrial model.

        Args:
            train_data: raw train data
            tuning_params: dictionary with tuning parameters
            mode: str, ``default='full'``. Defines the mode of fine-tuning. Could be 'full' or 'head'.

        """
        pass

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

    def save_best_model(self):
        if isinstance(self.solver, Fedot):
            return self.solver.current_pipeline.save(
                path=self.output_folder, create_subdir=True, is_datetime_in_path=True
            )
        elif isinstance(self.solver, Pipeline):
            return self.solver.save(
                path=self.output_folder, create_subdir=True, is_datetime_in_path=True
            )
        else:
            for idx, p in enumerate(self.solver.ensemble_branches):
                Pipeline(p).save(
                    f"./raf_ensemble/{idx}_ensemble_branch", create_subdir=True
                )
            Pipeline(self.solver.ensemble_head).save(
                f"./raf_ensemble/ensemble_head", create_subdir=True
            )

    def convert_model(
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
