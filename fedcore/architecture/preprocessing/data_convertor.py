"""Utilities for converting data between FEDOT, NumPy, PyTorch and custom formats.

This module contains helper classes for:

* wrapping FEDOT time series / tabular data into PyTorch datasets;
* converting between :class:`InputData` / :class:`OutputData`,
  NumPy arrays, pandas DataFrames and torch.Tensors;
* reshaping data to a suitable dimensionality for neural networks
  (1D/2D/3D/4D, image-like, time-series-like formats);
* checking capabilities and expected I/O of FEDOT/FedCore operations;
* introspecting neural network layers for structural properties
  (linear/conv/batch-norm, presence of weights/bias, etc.).
"""

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
from fedcore.architecture.computational.devices import default_device


class CustomDatasetTS:
    """Torch dataset wrapper for time series data.

    This class is intended to hold time series features and targets
    converted to 3D torch tensors in a format suitable for sequence
    models.

    Parameters
    ----------
    ts :
        FEDOT-like data object with ``features`` and ``target`` attributes.
    """

    def __init__(self, ts):
        self.x = torch.from_numpy(
            DataConverter(data=ts.features).convert_to_torch_format()
        ).float()
        self.y = torch.from_numpy(
            DataConverter(data=ts.target).convert_to_torch_format()
        ).float()

    def __getitem__(self, index):
        """Return a single sample by index.

        Notes
        -----
        This method is not yet implemented and should be filled with
        the desired indexing logic.
        """
        pass

    def __len__(self):
        """Return the number of samples in the dataset.

        Notes
        -----
        This method is not yet implemented and should be filled with
        the desired size logic.
        """
        pass


class CustomDatasetCLF:
    """Torch dataset wrapper for classification / regression tasks.

    This dataset:

    * converts features to :class:`torch.FloatTensor` on the default device;
    * normalizes class labels for binary and multi-class cases;
    * optionally encodes string labels using :class:`LabelEncoder`;
    * produces one-hot encoded targets for classification, or float
      targets for regression.

    Parameters
    ----------
    ts :
        FEDOT-like data object with attributes:

        * ``features`` – feature matrix;
        * ``target`` – array of labels or regression targets;
        * ``task`` – FEDOT task with ``task_type.value``;
        * ``class_labels`` and ``num_classes`` for classification tasks.
    """

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
        """Return ``(features, target)`` pair for a given index."""
        return self.x[index], self.y[index]

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.n_samples


class FedotConverter:
    """Converter between generic Python structures and FEDOT data types.

    This helper encapsulates typical conversions:

    * from raw tuples / NumPy / pandas into :class:`InputData`;
    * from FEDOT prediction objects / arrays into :class:`OutputData`;
    * reshaping inputs for industrial composing (1D / channel-wise /
      multi-dimensional).
    """

    def __init__(self, data):
        """Initialize converter and build :class:`InputData` if possible.

        Parameters
        ----------
        data :
            Object to convert. Can be a FEDOT ``InputData``/``OutputData``,
            tuple ``(features, target)``, list of inputs, or raw tensor-like
            data.
        """
        self.input_data = self.convert_to_input_data(data)

    def convert_to_input_data(self, data):
        """Convert arbitrary input into FEDOT :class:`InputData` or compatible.

        Parameters
        ----------
        data :
            Input object. Supported cases:

            * :class:`InputData` – returned as is;
            * :class:`OutputData` – returned as is;
            * tuple/sequence ``(features, target)`` with NumPy or pandas
              features – converted via :meth:`__init_input_data`;
            * list – first element is taken as input;
            * other – attempted to be converted to ``torch.tensor``.

        Returns
        -------
        InputData or OutputData or torch.Tensor or Any
            Converted object or original data if conversion is not possible.
        """
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
        """Create :class:`InputData` from features and targets.

        Parameters
        ----------
        features : pandas.DataFrame or np.ndarray
            Feature matrix.
        target : np.ndarray
            Target values.
        task : {"classification", "regression"}, default="classification"
            FEDOT task type to be assigned.

        Returns
        -------
        InputData
            FEDOT input data object with proper task and type settings.
        """
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
        """Wrap prediction results into FEDOT :class:`OutputData`.

        Parameters
        ----------
        prediction :
            Prediction object – either :class:`OutputData`, list of them,
            or raw NumPy/torch predictions.
        predict_data : InputData
            FEDOT input data that was used for prediction.
        output_data_type : DataTypesEnum
            FEDOT data type for the resulting :class:`OutputData`.

        Returns
        -------
        OutputData
            FEDOT output data with merged predictions and targets.
        """
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
        """Extract data type and data copy from internal :class:`InputData`.

        Returns
        -------
        tuple
            Pair ``(data_type, input_data_copy)`` where:

            * ``data_type`` – FEDOT :class:`DataTypesEnum`;
            * ``input_data_copy`` – original :class:`InputData`.
        """
        data_type = self.input_data.data_type
        predict_data_copy = self.input_data
        return data_type, predict_data_copy

    def convert_input_to_output(self):
        """Create :class:`OutputData` by reusing features as predictions.

        Useful in scenarios where a pipeline step expects
        :class:`OutputData`, but only :class:`InputData` is available.

        Returns
        -------
        OutputData
            FEDOT output data with ``predict`` set to ``features``.
        """
        return OutputData(
            idx=self.input_data.idx,
            # features=self.input_data,
            task=self.input_data.task,
            data_type=self.input_data.data_type,
            target=self.input_data.target,
            predict=self.input_data,
        )

    def convert_to_industrial_composing_format(self, mode):
        """Convert :class:`InputData` to industrial composing format.

        Parameters
        ----------
        mode : {"one_dimensional", "channel_independent", "multi_dimensional"}
            Conversion strategy:

            * ``"one_dimensional"`` – flatten time/channel dimensions;
            * ``"channel_independent"`` – split channels into separate
              :class:`InputData` objects;
            * ``"multi_dimensional"`` – convert to image-like format
              with :class:`DataTypesEnum.image`.

        Returns
        -------
        InputData or list[InputData]
            Converted data in the desired format.
        """
        if mode == "one_dimensional":
            new_features, new_target = [
                (
                    array.reshape(array.shape[0], array.shape[1] * array.shape[2])
                    if array is not None and len(array.shape) > 2
                    else array
                )
                for array in [self.input_data, self.input_data.target]
            ]
            input_data = InputData(
                idx=self.input_data.idx,
                features=new_features,
                target=new_target,
                task=self.input_data.task,
                data_type=self.input_data.data_type,
                supplementary_data=self.input_data.supplementary_data,
            )
        elif mode == "channel_independent":
            feats = self.input_data
            flat_input = self.input_data.shape[0] == 1
            if len(self.input_data.shape) == 1:
                feats = self.input_data.reshape(1, -1)
            elif (
                len(self.input_data.shape) == 3
                and self.input_data.shape[0] == 1
            ):
                feats = self.input_data.reshape(
                    self.input_data.shape[1],
                    1 * self.input_data.shape[2],
                )
            elif not flat_input:
                feats = self.input_data.swapaxes(1, 0)
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
                data=self.input_data
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
    """Base converter for turning various data types into :class:`torch.Tensor`."""

    def __init__(self, data):
        """Initialize converter from arbitrary data.

        Parameters
        ----------
        data :
            Input object to be converted (tuple, Tensor, np.ndarray,
            DataFrame, :class:`InputData`, etc.).
        """
        self.tensor_data = self.convert_to_tensor(data)

    def convert_to_tensor(self, data):
        """Convert input object to :class:`torch.Tensor`.

        Parameters
        ----------
        data :
            Data to convert. If ``data`` is a tuple, the first element
            is used.

        Returns
        -------
        torch.Tensor or None
            Converted tensor or ``None`` if conversion fails.
        """
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
        """Ensure the underlying tensor is 1D.

        Returns
        -------
        torch.Tensor
            1D tensor view of the data.

        Raises
        ------
        AssertionError
            If the tensor has unsupported dimensionality.
        """
        if self.tensor_data.ndim == 1:
            return self.tensor_data
        elif self.tensor_data.ndim == 3:
            return self.tensor_data[0, 0]
        if self.tensor_data.ndim == 2:
            return self.tensor_data[0]
        assert False, f"Please, review input dimensions {self.tensor_data.ndim}"

    def convert_to_2d_tensor(self):
        """Ensure the underlying tensor is 2D.

        Returns
        -------
        torch.Tensor
            2D tensor view of the data.

        Raises
        ------
        AssertionError
            If the tensor has unsupported dimensionality.
        """
        if self.tensor_data.ndim == 2:
            return self.tensor_data
        elif self.tensor_data.ndim == 1:
            return self.tensor_data[None]
        elif self.tensor_data.ndim == 3:
            return self.tensor_data[0]
        assert False, f"Please, review input dimensions {self.tensor_data.ndim}"

    def convert_to_3d_tensor(self):
        """Ensure the underlying tensor is 3D.

        Returns
        -------
        torch.Tensor
            3D tensor view of the data.

        Raises
        ------
        AssertionError
            If the tensor has unsupported dimensionality.
        """
        if self.tensor_data.ndim == 3:
            return self.tensor_data
        elif self.tensor_data.ndim == 1:
            return self.tensor_data[None, None]
        elif self.tensor_data.ndim == 2:
            return self.tensor_data[:, None]
        assert False, f"Please, review input dimensions {self.tensor_data.ndim}"


class NumpyConverter:
    """Base converter for turning various data types into NumPy arrays."""

    def __init__(self, data):
        """Initialize converter from arbitrary data.

        Parameters
        ----------
        data :
            Object to convert to a NumPy array.
        """
        self.numpy_data = self.convert_to_array(data)
        # sanitize NaNs and infs
        self.numpy_data = np.where(np.isnan(self.numpy_data), 0, self.numpy_data)
        self.numpy_data = np.where(np.isinf(self.numpy_data), 0, self.numpy_data)

    def convert_to_array(self, data):
        """Convert input object to :class:`numpy.ndarray`.

        Parameters
        ----------
        data :
            Data to convert. If ``data`` is a tuple, the first element
            is used.

        Returns
        -------
        np.ndarray or None
            Converted NumPy array or ``None`` if conversion fails.
        """
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
        """Ensure the underlying array is 1D.

        Returns
        -------
        np.ndarray
            1D view of the data.

        Raises
        ------
        AssertionError
            If the array has unsupported dimensionality.
        """
        if self.numpy_data.ndim == 1:
            return self.numpy_data
        elif self.numpy_data.ndim > 2:
            return np.squeeze(self.numpy_data)
        elif self.numpy_data.ndim == 2:
            return self.numpy_data.flatten()
        assert False, print(f"Please, review input dimensions {self.numpy_data.ndim}")

    def convert_to_2d_array(self):
        """Ensure the underlying array is 2D.

        Returns
        -------
        np.ndarray
            2D view of the data.

        Raises
        ------
        AssertionError
            If the array has unsupported dimensionality.
        """
        if self.numpy_data.ndim == 2:
            return self.numpy_data
        elif self.numpy_data.ndim == 1:
            return self.numpy_data.reshape(1, -1)
        elif self.numpy_data.ndim == 3:
            return self.numpy_data[0]
        assert False, print(f"Please, review input dimensions {self.numpy_data.ndim}")

    def convert_to_3d_array(self):
        """Ensure the underlying array is 3D.

        Returns
        -------
        np.ndarray
            3D view of the data.

        Raises
        ------
        AssertionError
            If the array has unsupported dimensionality.
        """
        if self.numpy_data.ndim == 3:
            return self.numpy_data
        elif self.numpy_data.ndim == 1:
            return self.numpy_data[None, None]
        elif self.numpy_data.ndim == 2:
            return self.numpy_data[:, None]
        assert False, print(f"Please, review input dimensions {self.numpy_data.ndim}")

    def convert_to_4d_torch_format(self):
        """Convert array to 4D NCHW-like format for torch image models.

        Returns
        -------
        np.ndarray
            4D array with shape ``(N, C, H, W)`` or similar.
        """
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
        """Convert array to 3D torch-friendly format (N, C, L) / (N, C, H).

        Returns
        -------
        np.ndarray
            3D array suitable for torch models.

        Raises
        ------
        AssertionError
            If the array has unsupported dimensionality.
        """
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
        """Convert array to time-series-like format.

        Returns
        -------
        np.ndarray
            Squeezed array for time series models.
        """
        if self.numpy_data.ndim > 1:
            return self.numpy_data.squeeze()
        else:
            return self.numpy_data


class ConditionConverter:
    """Inspection helper for operations and their I/O conventions.

    This class inspects a given operation implementation (or list of
    implementations) and training data to answer questions like:

    * whether operation has ``fit/transform/predict/predict_for_fit`` methods;
    * whether it expects FEDOT-style inputs (``input_data``);
    * whether it is used for regression/forecasting vs classification;
    * how to convert predictions to labels or probabilities.
    """

    def __init__(self, train_data, operation_implementation, mode):
        """Initialize converter with operation and training data.

        Parameters
        ----------
        train_data :
            Training data object (FEDOT ``InputData`` or similar).
        operation_implementation :
            Operation instance or list of instances to inspect.
        mode : {"one_dimensional", "channel_independent", "multi_dimensional"}
            Mode used for data representation.
        """
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
        """bool: Whether the operation implements ``transform``."""
        return "transform" in dir(self.operation_example)

    @property
    def have_fit_method(self):
        """bool: Whether the operation implements ``fit``."""
        return "fit" in dir(self.operation_example)

    @property
    def have_predict_method(self):
        """bool: Whether the operation implements ``predict``."""
        return "predict" in dir(self.operation_example)

    @property
    def have_predict_for_fit_method(self):
        """bool: Whether the operation implements ``predict_for_fit``."""
        return "predict_for_fit" in dir(self.operation_example)

    @property
    def is_one_dim_operation(self):
        """bool: True if operation is used in one-dimensional mode."""
        return self.mode == "one_dimensional"

    @property
    def is_channel_independent_operation(self):
        """bool: True if operation is used in channel-independent mode."""
        return self.mode == "channel_independent"

    @property
    def is_multi_dimensional_operation(self):
        """bool: True if operation is used in multi-dimensional mode."""
        return self.mode == "multi_dimensional"

    @property
    def is_list_container(self):
        """bool: True if train_data is a list container."""
        return type(self.train_data) is list

    @property
    def is_operation_is_list_container(self):
        """bool: True if operation implementation is a list container."""
        return type(self.operation_implementation) is list

    @property
    def have_predict_atr(self):
        """bool: Whether operation stores predictions as an attribute."""
        return (
            "predict" in vars(self.operation_example)
            if self.is_operation_is_list_container
            else False
        )

    @property
    def is_fit_input_fedot(self):
        """bool: Whether ``fit`` expects ``input_data`` as the first argument."""
        return (
            str(list(signature(self.operation_example.fit).parameters.keys())[0])
            == "input_data"
        )

    @property
    def is_transform_input_fedot(self):
        """bool: Whether ``transform`` expects ``input_data`` as the first argument."""
        return (
            str(list(signature(self.operation_example.transform).parameters.keys())[0])
            == "input_data"
        )

    @property
    def is_predict_input_fedot(self):
        """bool: Whether ``predict`` expects ``input_data`` as the first argument."""
        return (
            str(list(signature(self.operation_example.predict).parameters.keys())[0])
            == "input_data"
        )

    @property
    def is_regression_of_forecasting_task(self):
        """bool: Whether the task is regression or time-series forecasting."""
        return self.train_data.task.task_type.value in ["regression", "ts_forecasting"]

    @property
    def is_multi_output_target(self):
        """bool: Whether the operation has multi-output targets."""
        return isinstance(self.operation_example.classes_, list)

    @property
    def solver_is_fedot_class(self):
        """bool: Whether the solver is an instance of :class:`fedot.Fedot`."""
        return isinstance(self.operation_example, Fedot)

    @property
    def solver_is_none(self):
        """bool: Whether solver/operation is ``None``."""
        return self.operation_example is None

    def output_mode_converter(self, output_mode, n_classes):
        """Convert operation predictions according to desired output mode.

        Parameters
        ----------
        output_mode : {"labels", "probs"}
            Desired output representation: labels or probabilities.
        n_classes : int
            Number of classes in the prediction task.

        Returns
        -------
        np.ndarray
            Array of predictions in the requested representation.
        """
        if output_mode == "labels":
            return self.operation_example.predict(self.train_data.features).reshape(
                -1, 1
            )
        else:
            return self.probs_prediction_converter(output_mode, n_classes)

    def probs_prediction_converter(self, output_mode, n_classes):
        """Convert raw model probabilities to a consistent format.

        Parameters
        ----------
        output_mode : str
            Desired output representation (currently affects binary case).
        n_classes : int
            Number of classes.

        Returns
        -------
        np.ndarray
            Converted probability predictions.

        Raises
        ------
        ValueError
            If the dataset contains only one target class.
        """
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
    """Static predicates and helpers for high-level API logic."""

    @staticmethod
    def solver_is_fedot_class(operation_implementation):
        """Return True if solver is an instance of :class:`fedot.Fedot`."""
        return isinstance(operation_implementation, Fedot)

    @staticmethod
    def solver_is_none(operation_implementation):
        """Return True if solver/operation implementation is ``None``."""
        return operation_implementation is None

    @staticmethod
    def solver_is_pipeline_class(operation_implementation):
        """Return True if solver is a FEDOT :class:`Pipeline`."""
        return isinstance(operation_implementation, Pipeline)

    @staticmethod
    def solver_is_dict(operation_implementation):
        """Return True if solver is represented as a dictionary."""
        return isinstance(operation_implementation, dict)

    @staticmethod
    def tuning_params_is_none(tuning_params):
        """Return a non-None tuning params dict.

        Parameters
        ----------
        tuning_params : dict or None
            Tuning parameters or ``None``.

        Returns
        -------
        dict
            Original dict or an empty dict if input was ``None``.
        """
        return {} if tuning_params is None else tuning_params

    @staticmethod
    def ensemble_mode(predict_mode):
        """Return True if ensemble mode is enabled."""
        return predict_mode == "RAF_ensemble"

    @staticmethod
    def solver_have_target_encoder(encoder):
        """Return True if target encoder is present."""
        return encoder is not None

    @staticmethod
    def input_data_is_fedot_type(input_data):
        """Return True if input data is FEDOT :class:`InputData` or :class:`MultiModalData`."""
        return isinstance(input_data, (InputData, MultiModalData))

    def is_multiclf_with_labeling_problem(self, problem, target, predict):
        """Detect multi-class classification with misaligned labels.

        Parameters
        ----------
        problem : str
            Problem type (e.g. ``"classification"``).
        target : np.ndarray
            True labels.
        predict : np.ndarray
            Predicted labels.

        Returns
        -------
        bool
            True if the problem is multi-class classification and
            labels are misaligned.
        """
        clf_problem = problem == "classification"
        uncorrect_labels = target.min() - predict.min() != 0
        multiclass = len(np.unique(predict).shape) != 1
        return clf_problem and uncorrect_labels and multiclass


class DataConverter(TensorConverter, NumpyConverter):
    """High-level data converter combining tensor and NumPy utilities.

    This class provides a uniform interface for checking input types,
    reshaping data and converting it to different representations
    (lists, monads, eigen-basis, etc.).
    """

    def __init__(self, data):
        """Initialize converter from arbitrary data."""
        super().__init__(data)
        self.data = data
        self.numpy_data = self.convert_to_array(data)

    @property
    def is_nparray(self):
        """bool: Whether underlying data is a NumPy array."""
        return isinstance(self.data, np.ndarray)

    @property
    def is_tensor(self):
        """bool: Whether underlying data is a torch.Tensor."""
        return isinstance(self.data, torch.Tensor)

    @property
    def is_zarr(self):
        """bool: Whether underlying data has a zarr-like ``oindex`` attribute."""
        return hasattr(self.data, "oindex")

    @property
    def is_dask(self):
        """bool: Whether underlying data has a Dask-like ``compute`` method."""
        return hasattr(self.data, "compute")

    @property
    def is_memmap(self):
        """bool: Whether underlying data is a NumPy memmap."""
        return isinstance(self.data, np.memmap)

    @property
    def is_slice(self):
        """bool: Whether underlying data is a :class:`slice`."""
        return isinstance(self.data, slice)

    @property
    def is_tuple(self):
        """bool: Whether underlying data is a tuple."""
        return isinstance(self.data, tuple)

    @property
    def is_torchvision_dataset(self):
        """bool: Whether data represents a torchvision dataset sentinel."""
        if self.is_tuple:
            return self.data[1] == "torchvision_dataset"
        else:
            return False

    @property
    def is_none(self):
        """bool: Whether underlying data is ``None``."""
        return self.data is None

    @property
    def is_fedot_data(self):
        """bool: Whether underlying data is FEDOT :class:`InputData`."""
        return isinstance(self.data, InputData)

    @property
    def is_exist(self):
        """bool: Whether data is not ``None``."""
        return self.data is not None

    def convert_to_data_type(self):
        """Attempt to enforce tensor/array dtype on data in-place.

        Notes
        -----
        This method relies on ``astype`` / ``to`` behavior of NumPy and
        torch, and should be used with caution.
        """
        if isinstance(self.data, torch.Tensor):
            self.data = self.data.to(dtype=torch.Tensor)
        elif isinstance(self.data, np.ndarray):
            self.data = self.data.astype(np.ndarray)

    def convert_to_list(self):
        """Convert underlying data to a Python list.

        Returns
        -------
        list or None
            List representation of data, or ``None`` if conversion fails.
        """
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
        """Convert underlying data to 1D form.

        Returns
        -------
        np.ndarray or torch.Tensor
            1D representation of the data.
        """
        if self.data.ndim == 1:
            return self.data
        if isinstance(self.data, np.ndarray):
            return self.convert_to_1d_array()
        if isinstance(self.data, torch.Tensor):
            return self.convert_to_1d_tensor()

    def convert_data_to_2d(self):
        """Convert underlying data to 2D form.

        Returns
        -------
        np.ndarray or torch.Tensor
            2D representation of the data.
        """
        if self.data.ndim == 2:
            return self.data
        if isinstance(self.data, np.ndarray):
            return self.convert_to_2d_array()
        if isinstance(self.data, torch.Tensor):
            return self.convert_to_2d_tensor()

    def convert_data_to_3d(self):
        """Convert underlying data to 3D form.

        Returns
        -------
        np.ndarray or torch.Tensor
            3D representation of the data.
        """
        if self.data.ndim == 3:
            return self.data
        if isinstance(self.data, (np.ndarray, pd.self.dataFrame)):
            return self.convert_to_3d_array()
        if isinstance(self.data, torch.Tensor):
            return self.convert_to_3d_tensor()

    def convert_to_monad_data(self):
        """Convert data to monadic representation suitable for ListMonad.

        Returns
        -------
        np.ndarray
            Array constructed from a :class:`ListMonad` over features.
        """
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
        """Prepare data for eigen-basis or spectral decomposition.

        Returns
        -------
        np.ndarray
            2D array where each row corresponds to a cleaned series
            (NaNs removed).
        """
        if self.is_fedot_data:
            features = self.data.features
        else:
            features = np.array(ListMonad(*self.data.values.tolist()).value)
            features = np.array([series[~np.isnan(series)] for series in features])
        return features


class NeuralNetworkConverter:
    """Helper for introspecting basic properties of torch.nn layers."""

    def __init__(self, layer):
        """Initialize converter with a specific layer.

        Parameters
        ----------
        layer : nn.Module
            Layer to inspect.
        """
        self.layer = layer

    @property
    def is_layer(self, *args):
        """Callable[[], bool]: Predicate checking whether layer is instance of given types."""
        def _is_layer(cond=args):
            return isinstance(self.layer, cond)

        return partial(_is_layer, cond=args)

    @property
    def is_linear(self):
        """bool: Whether the layer is :class:`nn.Linear`."""
        return isinstance(self.layer, nn.Linear)

    @property
    def is_batch_norm(self):
        """bool: Whether the layer is batch normalization (1D/2D/3D)."""
        types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
        return isinstance(self.layer, types)

    @property
    def is_convolutional_linear(self):
        """bool: Whether the layer is convolutional or linear."""
        types = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
        return isinstance(self.layer, types)

    @property
    def is_affine(self):
        """bool: Whether the layer has affine parameters (weights/bias)."""
        return self.has_bias or self.has_weight

    @property
    def is_convolutional(self):
        """bool: Whether the layer is a convolutional layer."""
        types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
        return isinstance(self.layer, types)

    @property
    def has_bias(self):
        """bool: Whether the layer has a non-None ``bias`` attribute."""
        return hasattr(self.layer, "bias") and self.layer.bias is not None

    @property
    def has_weight(self):
        """bool: Whether the layer has a ``weight`` attribute."""
        return hasattr(self.layer, "weight")

    @property
    def has_weight_or_bias(self):
        """bool: Whether the layer has either weights or bias."""
        return any((self.has_weight, self.has_bias))
