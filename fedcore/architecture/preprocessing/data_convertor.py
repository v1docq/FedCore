from functools import partial
from inspect import signature

import pandas as pd
import torch
import torch.nn as nn
from fedot import Fedot
from fedot.core.data.data import InputData, OutputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from pymonad.list import ListMonad
from sklearn.preprocessing import LabelEncoder

from fedcore.api.utils.data import check_multivariate_data
from fedcore.architecture.settings.computational import backend_methods as np
from fedcore.architecture.comptutaional.devices import default_device


class CustomDatasetTS:
    def __init__(self, ts):
        self.x = torch.from_numpy(
            DataConverter(data=ts.features).convert_to_torch_format()
        ).float()
        self.y = torch.from_numpy(
            DataConverter(data=ts.target).convert_to_torch_format()
        ).float()

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class CustomDatasetCLF:
    def __init__(self, ts):
        self.x = torch.from_numpy(ts.features).to(default_device()).float()
        if ts.task.task_type.value == "classification":
            label_1 = max(ts.class_labels)
            label_0 = min(ts.class_labels)
            self.classes = ts.num_classes
            if self.classes == 2 and label_1 != 1:
                ts.target[ts.target == label_0] = 0
                ts.target[ts.target == label_1] = 1
            elif self.classes == 2 and label_0 != 0:
                ts.target[ts.target == label_0] = 0
                ts.target[ts.target == label_1] = 1
            elif self.classes > 2 and label_0 == 1:
                ts.target = ts.target - 1
            if type(min(ts.target)) is np.str_:
                self.label_encoder = LabelEncoder()
                ts.target = self.label_encoder.fit_transform(ts.target)
            else:
                self.label_encoder = None

            try:
                self.y = (
                    torch.nn.functional.one_hot(
                        torch.from_numpy(ts.target).long(), num_classes=self.classes
                    )
                    .to(default_device())
                    .squeeze(1)
                )
            except Exception:
                self.y = (
                    torch.nn.functional.one_hot(torch.from_numpy(ts.target).long())
                    .to(default_device())
                    .squeeze(1)
                )
                self.classes = self.y.shape[1]
        else:
            self.y = torch.from_numpy(ts.target).to(default_device()).float()
            self.classes = 1
            self.label_encoder = None

        self.n_samples = ts.features.shape[0]
        self.supplementary_data = ts.supplementary_data

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


class FedotConverter:
    def __init__(self, data):
        self.input_data = self.convert_to_input_data(data)

    def convert_to_input_data(self, data):
        if isinstance(data, InputData):
            return data
        elif isinstance(data, OutputData):
            return data
        elif isinstance(data[0], (np.ndarray, pd.DataFrame)):
            return self.__init_input_data(features=data[0], target=data[1])
        elif isinstance(data, list):
            return data[0]
        else:
            try:
                return torch.tensor(data)
            except Exception:
                print(f"Can't convert {type(data)} to InputData", Warning)

    def __init_input_data(
        self, features: pd.DataFrame, target: np.ndarray, task: str = "classification"
    ) -> InputData:
        if type(features) is np.ndarray:
            features = pd.DataFrame(features)
        is_multivariate_data = check_multivariate_data(features)
        task_dict = {
            "classification": Task(TaskTypesEnum.classification),
            "regression": Task(TaskTypesEnum.regression),
        }
        if is_multivariate_data:
            input_data = InputData(
                idx=np.arange(len(features)),
                features=np.array(features.values.tolist()).astype(float),
                target=target.astype(float).reshape(-1, 1),
                task=task_dict[task],
            )
        else:
            input_data = InputData(
                idx=np.arange(len(features)),
                features=features.values,
                target=np.ravel(target).reshape(-1, 1),
                task=task_dict[task],
            )
        return input_data

    def convert_to_output_data(self, prediction, predict_data, output_data_type):
        if isinstance(prediction, OutputData):
            output_data = prediction
        elif isinstance(prediction, list):
            output_data = prediction[0]
            target = NumpyConverter(
                data=np.concatenate([p.target for p in prediction], axis=0)
            ).convert_to_torch_format()
            predict = NumpyConverter(
                data=np.concatenate([p.predict for p in prediction], axis=0)
            ).convert_to_torch_format()
            output_data = OutputData(
                idx=predict_data.idx,
                features=predict_data.features,
                predict=predict,
                task=predict_data.task,
                target=target,
                data_type=output_data_type,
                supplementary_data=predict_data.supplementary_data,
            )
        else:
            output_data = OutputData(
                idx=predict_data.idx,
                features=predict_data.features,
                predict=prediction,
                task=predict_data.task,
                target=predict_data.target,
                data_type=output_data_type,
                supplementary_data=predict_data.supplementary_data,
            )
        return output_data

    def unwrap_list_to_output(self):
        data_type = self.input_data.data_type
        predict_data_copy = self.input_data
        return data_type, predict_data_copy

    def convert_input_to_output(self):
        return OutputData(
            idx=self.input_data.idx,
            features=self.input_data.features,
            task=self.input_data.task,
            data_type=self.input_data.data_type,
            target=self.input_data.target,
            predict=self.input_data.features,
        )

    def convert_to_industrial_composing_format(self, mode):
        if mode == "one_dimensional":
            new_features, new_target = [
                (
                    array.reshape(array.shape[0], array.shape[1] * array.shape[2])
                    if array is not None and len(array.shape) > 2
                    else array
                )
                for array in [self.input_data.features, self.input_data.target]
            ]
            # if new_features.shape[0] != new_target.shape[0]:
            #     min_samples = min(new_features.shape[0], new_target.shape[0])
            #     new_features, new_target = new_features[:min_samples], new_target[:min_samples]
            input_data = InputData(
                idx=self.input_data.idx,
                features=new_features,
                target=new_target,
                task=self.input_data.task,
                data_type=self.input_data.data_type,
                supplementary_data=self.input_data.supplementary_data,
            )
        elif mode == "channel_independent":
            feats = self.input_data.features
            flat_input = self.input_data.features.shape[0] == 1
            if len(self.input_data.features.shape) == 1:
                feats = self.input_data.features.reshape(1, -1)
            elif (
                len(self.input_data.features.shape) == 3
                and self.input_data.features.shape[0] == 1
            ):
                feats = self.input_data.features.reshape(
                    self.input_data.features.shape[1],
                    1 * self.input_data.features.shape[2],
                )
            elif not flat_input:
                feats = self.input_data.features.swapaxes(1, 0)
            input_data = [
                InputData(
                    idx=self.input_data.idx,
                    features=features,
                    target=self.input_data.target,
                    task=self.input_data.task,
                    data_type=self.input_data.data_type,
                    supplementary_data=self.input_data.supplementary_data,
                )
                for features in feats
            ]
        elif mode == "multi_dimensional":
            features = NumpyConverter(
                data=self.input_data.features
            ).convert_to_torch_format()
            input_data = InputData(
                idx=self.input_data.idx,
                features=features,
                target=self.input_data.target,
                task=self.input_data.task,
                data_type=DataTypesEnum.image,
                supplementary_data=self.input_data.supplementary_data,
            )

        return input_data


class TensorConverter:
    def __init__(self, data):
        self.tensor_data = self.convert_to_tensor(data)

    def convert_to_tensor(self, data):
        if isinstance(data, tuple):
            data = data[0]

        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, pd.DataFrame):
            if data.values.dtype == object:
                return torch.from_numpy(np.array(data.values.tolist()).astype(float))
            else:
                return torch.from_numpy(data.values)
        elif isinstance(data, InputData):
            return torch.from_numpy(data.features)
        else:
            print(f"Can't convert {type(data)} to torch.Tensor", Warning)

    def convert_to_1d_tensor(self):
        if self.tensor_data.ndim == 1:
            return self.tensor_data
        elif self.tensor_data.ndim == 3:
            return self.tensor_data[0, 0]
        if self.tensor_data.ndim == 2:
            return self.tensor_data[0]
        assert False, f"Please, review input dimensions {self.tensor_data.ndim}"

    def convert_to_2d_tensor(self):
        if self.tensor_data.ndim == 2:
            return self.tensor_data
        elif self.tensor_data.ndim == 1:
            return self.tensor_data[None]
        elif self.tensor_data.ndim == 3:
            return self.tensor_data[0]
        assert False, f"Please, review input dimensions {self.tensor_data.ndim}"

    def convert_to_3d_tensor(self):
        if self.tensor_data.ndim == 3:
            return self.tensor_data
        elif self.tensor_data.ndim == 1:
            return self.tensor_data[None, None]
        elif self.tensor_data.ndim == 2:
            return self.tensor_data[:, None]
        assert False, f"Please, review input dimensions {self.tensor_data.ndim}"


class NumpyConverter:
    def __init__(self, data):
        self.numpy_data = self.convert_to_array(data)
        self.numpy_data = np.where(np.isnan(self.numpy_data), 0, self.numpy_data)
        self.numpy_data = np.where(np.isinf(self.numpy_data), 0, self.numpy_data)

    def convert_to_array(self, data):
        if isinstance(data, tuple):
            data = data[0]

        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.detach().numpy()
        elif isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, InputData):
            return data.features
        elif isinstance(data, CustomDatasetTS):
            return data.x
        elif isinstance(data, CustomDatasetCLF):
            return data.x
        else:
            try:
                return np.asarray(data)
            except Exception:
                print(f"Can't convert {type(data)} to np.array", Warning)

    def convert_to_1d_array(self):
        if self.numpy_data.ndim == 1:
            return self.numpy_data
        elif self.numpy_data.ndim > 2:
            return np.squeeze(self.numpy_data)
        elif self.numpy_data.ndim == 2:
            return self.numpy_data.flatten()
        assert False, print(f"Please, review input dimensions {self.numpy_data.ndim}")

    def convert_to_2d_array(self):
        if self.numpy_data.ndim == 2:
            return self.numpy_data
        elif self.numpy_data.ndim == 1:
            return self.numpy_data.reshape(1, -1)
        elif self.numpy_data.ndim == 3:
            return self.numpy_data[0]
        assert False, print(f"Please, review input dimensions {self.numpy_data.ndim}")

    def convert_to_3d_array(self):
        if self.numpy_data.ndim == 3:
            return self.numpy_data
        elif self.numpy_data.ndim == 1:
            return self.numpy_data[None, None]
        elif self.numpy_data.ndim == 2:
            return self.numpy_data[:, None]
        assert False, print(f"Please, review input dimensions {self.numpy_data.ndim}")

    def convert_to_4d_torch_format(self):
        if self.numpy_data.ndim == 4:
            if self.numpy_data.shape[1] in range(1, 5):
                # because image.shape[1] could be maximum RGB(a) channels
                return self.numpy_data
            else:
                return self.numpy_data.swapaxes(1, 3)
        elif self.numpy_data.ndim == 1:
            return self.numpy_data.reshape(-1, 1, 1)
        else:
            return self.numpy_data.reshape(
                self.numpy_data.shape[0],
                1,
                self.numpy_data.shape[1],
                self.numpy_data.shape[2],
            )

    def convert_to_torch_format(self):
        if self.numpy_data.ndim == 3:
            return self.numpy_data
        elif self.numpy_data.ndim == 1:
            return self.numpy_data.reshape(self.numpy_data.shape[0], 1, 1)
        elif self.numpy_data.ndim == 2 and self.numpy_data.shape[0] != 1:
            # add 1 channel
            return self.numpy_data.reshape(
                self.numpy_data.shape[0], 1, self.numpy_data.shape[1]
            )
        elif self.numpy_data.ndim == 2 and self.numpy_data.shape[0] == 1:
            # add 1 channel
            return self.numpy_data.reshape(1, 1, self.numpy_data.shape[1])

        elif self.numpy_data.ndim > 3:
            return self.numpy_data.squeeze()
        assert False, print(f"Please, review input dimensions {self.numpy_data.ndim}")

    def convert_to_ts_format(self):
        if self.numpy_data.ndim > 1:
            return self.numpy_data.squeeze()
        else:
            return self.numpy_data


class ConditionConverter:
    def __init__(self, train_data, operation_implementation, mode):
        self.train_data = train_data
        self.operation_implementation = operation_implementation
        self.operation_example = (
            operation_implementation[0]
            if isinstance(operation_implementation, list)
            else operation_implementation
        )
        self.mode = mode

    @property
    def have_transform_method(self):
        return "transform" in dir(self.operation_example)

    @property
    def have_fit_method(self):
        return "fit" in dir(self.operation_example)

    @property
    def have_predict_method(self):
        return "predict" in dir(self.operation_example)

    @property
    def have_predict_for_fit_method(self):
        return "predict_for_fit" in dir(self.operation_example)

    @property
    def is_one_dim_operation(self):
        return self.mode == "one_dimensional"

    @property
    def is_channel_independent_operation(self):
        return self.mode == "channel_independent"

    @property
    def is_multi_dimensional_operation(self):
        return self.mode == "multi_dimensional"

    @property
    def is_list_container(self):
        return type(self.train_data) is list

    @property
    def is_operation_is_list_container(self):
        return type(self.operation_implementation) is list

    @property
    def have_predict_atr(self):
        return (
            "predict" in vars(self.operation_example)
            if self.is_operation_is_list_container
            else False
        )

    @property
    def is_fit_input_fedot(self):
        return (
            str(list(signature(self.operation_example.fit).parameters.keys())[0])
            == "input_data"
        )

    @property
    def is_transform_input_fedot(self):
        return (
            str(list(signature(self.operation_example.transform).parameters.keys())[0])
            == "input_data"
        )

    @property
    def is_predict_input_fedot(self):
        return (
            str(list(signature(self.operation_example.predict).parameters.keys())[0])
            == "input_data"
        )

    @property
    def is_regression_of_forecasting_task(self):
        return self.train_data.task.task_type.value in ["regression", "ts_forecasting"]

    @property
    def is_multi_output_target(self):
        return isinstance(self.operation_example.classes_, list)

    @property
    def solver_is_fedot_class(self):
        return isinstance(self.operation_example, Fedot)

    @property
    def solver_is_none(self):
        return self.operation_example is None

    def output_mode_converter(self, output_mode, n_classes):
        if output_mode == "labels":
            return self.operation_example.predict(self.train_data.features).reshape(
                -1, 1
            )
        else:
            return self.probs_prediction_converter(output_mode, n_classes)

    def probs_prediction_converter(self, output_mode, n_classes):
        try:
            prediction = self.operation_example.predict_proba(self.train_data.features)
        except Exception:
            prediction = self.operation_example.predict_proba(
                self.train_data.features.T
            )
        if n_classes < 2:
            raise ValueError(
                "Data set contain only 1 target class. Please reformat your data."
            )
        elif n_classes == 2 and output_mode != "probs":
            if self.is_multi_output_target:
                prediction = np.stack([pred[:, 1] for pred in prediction]).T
            else:
                prediction = prediction[:, 1]
        return prediction


class ApiConverter:

    @staticmethod
    def solver_is_fedot_class(operation_implementation):
        return isinstance(operation_implementation, Fedot)

    @staticmethod
    def solver_is_none(operation_implementation):
        return operation_implementation is None

    @staticmethod
    def solver_is_pipeline_class(operation_implementation):
        return isinstance(operation_implementation, Pipeline)

    @staticmethod
    def solver_is_dict(operation_implementation):
        return isinstance(operation_implementation, dict)

    @staticmethod
    def tuning_params_is_none(tuning_params):
        return {} if tuning_params is None else tuning_params

    @staticmethod
    def ensemble_mode(predict_mode):
        return predict_mode == "RAF_ensemble"

    @staticmethod
    def solver_have_target_encoder(encoder):
        return encoder is not None

    @staticmethod
    def input_data_is_fedot_type(input_data):
        return isinstance(input_data, (InputData, MultiModalData))

    def is_multiclf_with_labeling_problem(self, problem, target, predict):
        clf_problem = problem == "classification"
        uncorrect_labels = target.min() - predict.min() != 0
        multiclass = len(np.unique(predict).shape) != 1
        return clf_problem and uncorrect_labels and multiclass


class DataConverter(TensorConverter, NumpyConverter):
    def __init__(self, data):
        super().__init__(data)
        self.data = data
        self.numpy_data = self.convert_to_array(data)

    @property
    def is_nparray(self):
        return isinstance(self.data, np.ndarray)

    @property
    def is_tensor(self):
        return isinstance(self.data, torch.Tensor)

    @property
    def is_zarr(self):
        return hasattr(self.data, "oindex")

    @property
    def is_dask(self):
        return hasattr(self.data, "compute")

    @property
    def is_memmap(self):
        return isinstance(self.data, np.memmap)

    @property
    def is_slice(self):
        return isinstance(self.data, slice)

    @property
    def is_tuple(self):
        return isinstance(self.data, tuple)

    @property
    def is_torchvision_dataset(self):
        if self.is_tuple:
            return self.data[1] == "torchvision_dataset"
        else:
            return False

    @property
    def is_none(self):
        return self.data is None

    @property
    def is_fedot_data(self):
        return isinstance(self.data, InputData)

    @property
    def is_exist(self):
        return self.data is not None

    def convert_to_data_type(self):
        if isinstance(self.data, torch.Tensor):
            self.data = self.data.to(dtype=torch.Tensor)
        elif isinstance(self.data, np.ndarray):
            self.data = self.data.astype(np.ndarray)

    def convert_to_list(self):
        if isinstance(self.data, list):
            return self.data
        elif isinstance(self.data, (np.ndarray, torch.Tensor)):
            return self.data.tolist()
        else:
            try:
                return list(self.data)
            except Exception:
                print(
                    f"passed object needs to be of type L, list, np.ndarray or torch.Tensor but is {type(self.data)}",
                    Warning,
                )

    def convert_data_to_1d(self):
        if self.data.ndim == 1:
            return self.data
        if isinstance(self.data, np.ndarray):
            return self.convert_to_1d_array()
        if isinstance(self.data, torch.Tensor):
            return self.convert_to_1d_tensor()

    def convert_data_to_2d(self):
        if self.data.ndim == 2:
            return self.data
        if isinstance(self.data, np.ndarray):
            return self.convert_to_2d_array()
        if isinstance(self.data, torch.Tensor):
            return self.convert_to_2d_tensor()

    def convert_data_to_3d(self):
        if self.data.ndim == 3:
            return self.data
        if isinstance(self.data, (np.ndarray, pd.self.dataFrame)):
            return self.convert_to_3d_array()
        if isinstance(self.data, torch.Tensor):
            return self.convert_to_3d_tensor()

    def convert_to_monad_data(self):
        if self.is_fedot_data:
            features = np.array(ListMonad(*self.data.features.tolist()).value)
        else:
            features = np.array(ListMonad(*self.data.tolist()).value)

        if len(features.shape) == 2 and features.shape[1] == 1:
            features = features.reshape(1, -1)
        elif len(features.shape) == 1:
            features = features.reshape(1, 1, -1)
        elif len(features.shape) == 3 and features.shape[1] == 1:
            features = features.squeeze()
        return features

    def convert_to_eigen_basis(self):
        if self.is_fedot_data:
            features = self.data.features
        else:
            features = np.array(ListMonad(*self.data.values.tolist()).value)
            features = np.array([series[~np.isnan(series)] for series in features])
        return features


class NeuralNetworkConverter:
    def __init__(self, layer):
        self.layer = layer

    @property
    def is_layer(self, *args):
        def _is_layer(cond=args):
            return isinstance(self.layer, cond)

        return partial(_is_layer, cond=args)

    @property
    def is_linear(self):
        return isinstance(self.layer, nn.Linear)

    @property
    def is_batch_norm(self):
        types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
        return isinstance(self.layer, types)

    @property
    def is_convolutional_linear(self):
        types = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
        return isinstance(self.layer, types)

    @property
    def is_affine(self):
        return self.has_bias or self.has_weight

    @property
    def is_convolutional(self):
        types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
        return isinstance(self.layer, types)

    @property
    def has_bias(self):
        return hasattr(self.layer, "bias") and self.layer.bias is not None

    @property
    def has_weight(self):
        return hasattr(self.layer, "weight")

    @property
    def has_weight_or_bias(self):
        return any((self.has_weight, self.has_bias))
