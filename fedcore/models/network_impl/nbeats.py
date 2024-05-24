import pickle
import random
from enum import Enum
from time import time
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn, optim
from torch.optim import Optimizer
from torch.nn import functional as F
from torch.nn.functional import mse_loss, l1_loss, binary_cross_entropy, cross_entropy
from torch.utils.data import DataLoader

from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedcore.architecture.comptutaional.devices import default_device

from fedcore.models.network_impl.bases import _NBeatsBlock, _TrendBasis, _GenericBasis, _SeasonalityBasis
from fedcore.models.nn.network_modules.losses import SMAPELoss


class NBeatsModel(ModelImplementation):
    """Class responsible for NBeats model implementation.

    Attributes:
        self.num_features: int, the number of features.

    Example:
        To use this operation you can create pipeline as follows::
            from fedot.core.pipelines.pipeline_builder import PipelineBuilder
            from examples.fedot.fedot_ex import init_input_data
            from fedot_ind.tools.loader import DataLoader
            from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
            train_data, test_data = DataLoader(dataset_name="Lightning7").load_data()
            input_data = init_input_data(train_data[0], train_data[1])
            val_data = init_input_data(test_data[0], test_data[1])
            with IndustrialModels():
                pipeline = PipelineBuilder().add_node("tst_model",
                                                      params={"epochs": 100,
                                                              "batch_size": 10}
                                                     ) \
                                            .build()
                pipeline.fit(input_data)
                target = pipeline.predict(val_data).predict
                metric = evaluate_metric(target=test_data[1], prediction=target)

    References:
        @inproceedings{
            Oreshkin2020:N-BEATS,
            title={{N-BEATS}: Neural basis expansion analysis for interpretable time series forecasting},
            author={Boris N. Oreshkin and Dmitri Carpov and Nicolas Chapados and Yoshua Bengio},
            booktitle={International Conference on Learning Representations},
            year={2020},
            url={https://openreview.net/forum?id=r1ecqn4YwB}
        }
        Original paper: https://arxiv.org/abs/1905.10437
        Original code:  https://github.com/ServiceNow/N-BEATS
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        self.is_generic_architecture = params.get("is_generic_architecture", True)

        self.n_stacks = params.get("n_stacks", 30)
        self.layers = params.get("layers", 4)
        self.layer_size = params.get("layer_size", 512)

        self.n_trend_blocks = params.get("n_trend_blocks", 3)
        self.n_trend_layers = params.get("n_trend_layers", 4)
        self.trend_layer_size = params.get("trend_layer_size", 2)
        self.degree_of_polynomial = params.get("degree_of_polynomial", 20)

        self.n_seasonality_blocks = params.get("n_seasonality_blocks", 3)
        self.n_seasonality_layers = params.get("n_seasonality_layers", 4)
        self.seasonality_layer_size = params.get("seasonality_layer_size", 2048)
        self.n_of_harmonics = params.get("n_of_harmonics", 1)

    def _init_model(self, ts) -> tuple:
        self.model = NBeats(
            input_dim=ts.features.shape[1],
            output_dim=self.num_classes).to(default_device())
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # In article, you could choose from: MAPE, MASE, SMAPE
        # https://github.com/ServiceNow/N-BEATS/blob/master/experiments/trainer.py#L79
        loss_fn = SMAPELoss
        return loss_fn, optimizer


class NBeats(nn.Module):
    """Class responsible for NBeats Model.

    Args:
        input_dim: the number of features (aka variables, dimensions, channels) in the time series dataset
        output_dim: the number of target classes
        is_generic_architecture: indicating whether the generic architecture of N-BEATS is used.
            If not, the interpretable architecture outlined in the paper (consisting of one trend
            and one seasonality stack with appropriate waveform generator functions)
        n_stacks: the number of: The number of stacks that make up the whole model.
            Only used if `is_generic_architecture` is set to `True`.
            The interpretable architecture always uses two stacks - one for trend and one for seasonality.
        n_trend_blocks:
            Used if `is_generic_architecture` is set to `False`
        n_trend_layers:
            Used if `is_generic_architecture` is set to `False`
        degree_of_polynomial:
            Used if `is_generic_architecture` is set to `False`
        n_seasonality_blocks
            Used if `is_generic_architecture` is set to `False`
        n_seasonality_layers
            Used if `is_generic_architecture` is set to `False`
        seasonality_layer_size:
            Used if `is_generic_architecture` is set to `False`
        n_of_harmonics:
            Used if `is_generic_architecture` is set to `False`

    References:
        Original paper: https://arxiv.org/abs/1905.10437
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 is_generic_architecture: bool,
                 n_stacks: int,
                 n_trend_blocks: int,
                 n_trend_layers: int,
                 trend_layer_size: int,
                 degree_of_polynomial: int,
                 n_seasonality_blocks: int,
                 n_seasonality_layers: int,
                 seasonality_layer_size: int,
                 n_of_harmonics: int,
                 ):

        super().__init__()

        self.blocks = None

        if is_generic_architecture:
            self.stack = _NBeatsStack(
                input_dim=input_dim,
                output_dim=output_dim,
                is_generic_architecture=is_generic_architecture,
                n_stacks=n_stacks,
            )
        else:
            # The overall interpretable architecture consists of two stacks:
            # the trend stack is followed by the seasonality stack
            self.stack = _NBeatsStack(
                input_dim=input_dim,
                output_dim=output_dim,
                is_generic_architecture=is_generic_architecture,
                n_trend_blocks=n_trend_blocks,
                n_trend_layers=n_trend_layers,
                trend_layer_size=trend_layer_size,
                degree_of_polynomial=degree_of_polynomial,
                n_seasonality_blocks=n_seasonality_blocks,
                n_seasonality_layers=n_seasonality_layers,
                seasonality_layer_size=seasonality_layer_size,
                n_of_harmonics=n_of_harmonics,
            )

        self.blocks = nn.ModuleList(self.stacks)

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        residuals = x.flip(dims=(1,))
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast

        return forecast


class _NBeatsStack(nn.Module):
    """Class responsible for NBeats Model Stack.

        Args:
            input_dim: the number of features (aka variables, dimensions, channels) in the time series dataset
            output_dim: the number of target classes
            is_generic_architecture: indicating whether the generic architecture of N-BEATS is used.
                If not, the interpretable architecture outlined in the paper (consisting of one trend
                and one seasonality stack with appropriate waveform generator functions)
            n_stacks: the number of: The number of stacks that make up the whole model.
                Only used if `is_generic_architecture` is set to `True`.
                The interpretable architecture always uses two stacks - one for trend and one for seasonality.
            n_trend_blocks:
                Used if `is_generic_architecture` is set to `False`
            n_trend_layers:
                Used if `is_generic_architecture` is set to `False`
            degree_of_polynomial:
                Used if `is_generic_architecture` is set to `False`
            n_seasonality_blocks
                Used if `is_generic_architecture` is set to `False`
            n_seasonality_layers
                Used if `is_generic_architecture` is set to `False`
            seasonality_layer_size:
                Used if `is_generic_architecture` is set to `False`
            n_of_harmonics:
                Used if `is_generic_architecture` is set to `False`

        References:
            Original paper: https://arxiv.org/abs/1905.10437
        """
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            is_generic_architecture: bool,
            n_stacks: int,
            n_trend_blocks: int,
            n_trend_layers: int,
            trend_layer_size: int,
            degree_of_polynomial: int,
            n_seasonality_blocks: int,
            n_seasonality_layers: int,
            seasonality_layer_size: int,
            n_of_harmonics: int,
    ):
        self.block = None

        if is_generic_architecture:
            self.block = _NBeatsBlock(
                input_size=input_dim,
                theta_size=input_dim + output_dim,
                basis_function=_GenericBasis(
                    backcast_size=input_dim,
                    forecast_size=output_dim
                )
            )
            self.blocks = [self.block for _ in range(n_stacks)]

        else:
            trend_block = _NBeatsBlock(
                input_size=input_dim,
                theta_size=2 * (degree_of_polynomial + 1),
                basis_function=_TrendBasis(
                    degree_of_polynomial=degree_of_polynomial,
                    backcast_size=input_dim,
                    forecast_size=output_dim),
                layers=n_trend_layers,
                layer_size=trend_layer_size,
            )

            seasonality_block = _NBeatsBlock(
                input_size=input_dim,
                theta_size=4 * int(np.ceil(n_of_harmonics / 2 * output_dim) - (n_of_harmonics - 1)),
                basis_function=_SeasonalityBasis(
                    harmonics=n_of_harmonics,
                    backcast_size=input_dim,
                    forecast_size=output_dim),
                layers=n_seasonality_layers,
                layer_size=seasonality_layer_size
            )

            self.blocks = [trend_block for _ in range(n_trend_blocks)] + [seasonality_block for _ in
                                                                          range(n_seasonality_blocks)]


class NBeatsNet(nn.Module):
    """
    The NBeatsNet class is an implementation of the N-BEATS model for interpretable time series forecasting.
    It employs a deep neural architecture that models time series data by learning and combining
    several basis expansion blocks, such as trend and seasonality blocks.

    Attributes:
        forecast_length: The length of the forecast horizon.
        backcast_length: The length of the input series considered for the forecast.
        hidden_layer_units: The number of units in the hidden layer.
        nb_blocks_per_stack: Number of blocks per stack.
        share_weights_in_stack: If True, blocks within the same stack share weights.
        nb_harmonics: The number of harmonics for seasonality modeling, if applicable.
        stack_types: Types of stacks used in the model ('trend', 'seasonality', or 'generic').
        stacks: The list of stack instances built during initialization.
        thetas_dim: The dimensionality of the theta parameters for each stack.
        device: The device on which the model is running.

    Methods:
        create_stack(stack_id: int):
            Creates a stack with the given ID.

        compile(loss: str, optimizer: Union[str, Optimizer]):
            Configures the model for training with given loss and optimizer.

        fit(x_train, y_train, validation_data=None, epochs=10, batch_size=32):
            Trains the model on the given training data.

        predict(x, return_backcast=False):
            Generates output predictions for the given input data.

        save(filename: str):
            Saves the current state of the model.

        load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
            Loads the model from a saved state.

        disable_intermediate_outputs():
            Disables recording of intermediate outputs.

        enable_intermediate_outputs():
            Enables recording of intermediate outputs.

        get_generic_and_interpretable_outputs():
            Retrieves the outputs of the generic and interpretable layers.

        forward(backcast):
            Defines the forward pass of the model.

    Example:
        Using the NBeatsNet class involves first instantiating the model with desired configurations and then
        calling its methods to train, predict, and potentially save the model's state.

        Example usage:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = NBeatsNet(device=device, stack_types=("trend", "seasonality"), ...)
            model.compile(loss='mae', optimizer='adam')
            model.fit(x_train, y_train, epochs=15, batch_size=64)
            predictions = model.predict(x_test)

    References:
        The implementation is based on the original paper and the code provided by Philippe Remy.

        @misc{NBeatsPRemy,
            author = {Philippe Remy},
            title = {N-BEATS: Neural basis expansion analysis for interpretable time series forecasting},
            year = {2020},
            publisher = {GitHub},
            journal = {GitHub repository},
            howpublished = {\url{https://github.com/philipperemy/n-beats}},
        }
    """
    SEASONALITY_BLOCK = "seasonality"
    TREND_BLOCK = "trend"
    GENERIC_BLOCK = "generic"

    def __init__(
            self,
            device=torch.device("cpu"),
            stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
            nb_blocks_per_stack=3,
            forecast_length=5,
            backcast_length=10,
            thetas_dim=(4, 8),
            share_weights_in_stack=False,
            hidden_layer_units=256,
            nb_harmonics=None
    ):
        super(NBeatsNet, self).__init__()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.nb_harmonics = nb_harmonics
        self.stack_types = stack_types
        self.stacks = []
        self.thetas_dim = thetas_dim
        self.parameters = []
        self.device = device
        print("| N-Beats")
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)
        self.to(self.device)
        self._loss = None
        self._opt = None
        self._gen_intermediate_outputs = False
        self._intermediary_outputs = []

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        print(f"| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})")
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            block_init = NBeatsNet.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]  # pick up the last one when we share weights.
            else:
                block = block_init(
                    self.hidden_layer_units, self.thetas_dim[stack_id],
                    self.device, self.backcast_length, self.forecast_length,
                    self.nb_harmonics
                )
                self.parameters.extend(block.parameters())
            print(f"     | -- {block}")
            blocks.append(block)
        return blocks

    def disable_intermediate_outputs(self):
        self._gen_intermediate_outputs = False

    def enable_intermediate_outputs(self):
        self._gen_intermediate_outputs = True

    def save(self, filename: str):
        torch.save(self, filename)

    @staticmethod
    def load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
        return torch.load(f, map_location, pickle_module, **pickle_load_args)

    @staticmethod
    def select_block(block_type):
        if block_type == NBeatsNet.SEASONALITY_BLOCK:
            return SeasonalityBlock
        elif block_type == NBeatsNet.TREND_BLOCK:
            return TrendBlock
        else:
            return GenericBlock

    def compile(self, loss: str, optimizer: Union[str, Optimizer]):
        """
        Configures the model for training by assigning a loss function and an optimizer.

        Args:
            loss (str): A string identifier for the desired loss function.
                Supported losses: "mae" for Mean Absolute Error, "mse" for Mean Squared Error,
                "cross_entropy" for Cross Entropy Loss, and "binary_crossentropy" for Binary Cross-Entropy Loss.

            optimizer (Union[str, Optimizer]): A string identifier or an optimizer instance.
                If a string is provided, it should be one of the following: "adam", "sgd", or "rmsprop",
                which will be initialized with a default learning rate of 1e-4.
                If an optimizer instance is provided, it will be used as is.

        Raises:
            ValueError: If an unknown string identifier for loss or optimizer is provided.
        """
        try:
            losses = {
                "mae": l1_loss,
                "mse": mse_loss,
                "cross_entropy": cross_entropy,
                "binary_crossentropy": binary_cross_entropy
            }
            loss_ = losses[loss]
        except KeyError:
            raise ValueError(f"Unknown loss name: {loss}.")
        # noinspection PyArgumentList
        if isinstance(optimizer, str):
            try:
                optimizers = {
                    "adam": optim.Adam,
                    "sgd": optim.SGD,
                    "rmsprop": optim.RMSprop
                }
                opt_ = optimizers[optimizer]
            except KeyError:
                raise ValueError(f"Unknown opt name: {optimizer}.")
            opt_ = opt_(lr=1e-4, params=self.parameters())
        else:
            opt_ = optimizer
        self._opt = opt_
        self._loss = loss_

    def fit(self, x_train, y_train, validation_data=None, epochs: int = 10, batch_size: int = 32):
        """
        Trains the NBeatsNet model using given training data and parameters.

        Args:
            x_train: Training features (predictors).
            y_train: Training labels (targets). Dimensions must match x_train.
            validation_data:  A tuple (x_val, y_val) where x_val is the validation features and
                y_val is the validation labels. If None, no validation is performed.
            epochs: Number of epochs to run training for. Defaults to 10.
            batch_size: Number of samples per gradient update. Defaults to 32.

        Raises:
            AssertionError: If dimensions of split training features and labels do not match.
            ValueError: If the provided data leads to an undefined validation loss, e.g., due to incorrect dimensions.

        """

        for epoch in range(epochs):
            x_train_dl = DataLoader(x_train, batch_size)
            y_train_dl = DataLoader(y_train, batch_size)
            assert len(x_train_dl) == len(y_train_dl)

            shuffled_indices = list(range(len(x_train_dl)))
            random.shuffle(shuffled_indices)
            self.train()
            train_loss = []
            timer = time()

            for batch_id in shuffled_indices:
                batch_x, batch_y = list(x_train_dl)[batch_id], list(y_train_dl)[batch_id]
                self._opt.zero_grad()
                _, forecast = self(torch.tensor(batch_x, dtype=torch.float).to(self.device))
                loss = self._loss(forecast, squeeze_last_dim(torch.tensor(batch_y, dtype=torch.float).to(self.device)))
                train_loss.append(loss.item())
                loss.backward()
                self._opt.step()

            elapsed_time = time() - timer
            train_loss = np.mean(train_loss)

            test_loss = '[undefined]'
            if validation_data is not None:
                x_test, y_test = validation_data
                self.eval()
                _, forecast = self(torch.tensor(x_test, dtype=torch.float).to(self.device))
                test_loss = self._loss(forecast, squeeze_last_dim(torch.tensor(y_test, dtype=torch.float))).item()

            num_samples = len(x_train_dl)
            time_per_step = int(elapsed_time / num_samples * 1000)
            print(f"Epoch {str(epoch + 1).zfill(len(str(epochs)))}/{epochs}")
            try:
                print(f"{num_samples}/{num_samples} [==============================] - "
                      f"{int(elapsed_time)}s {time_per_step}ms/step - "
                      f"loss: {train_loss:.4f} - val_loss: {test_loss:.4f}")
            except ValueError:
                print(f"{num_samples}/{num_samples} [==============================] - "
                      f"{int(elapsed_time)}s {time_per_step}ms/step - "
                      f"loss: {train_loss:.4f} - val_loss: [undefined]")

    def predict(self, x, return_backcast=False):
        self.eval()
        backcast, forecast = self(torch.tensor(x, dtype=torch.float).to(self.device))
        backcast, forecast = backcast.detach().numpy(), forecast.detach().numpy()
        if len(x.shape) == 3:
            backcast = np.expand_dims(backcast, axis=-1)
            forecast = np.expand_dims(forecast, axis=-1)
        if return_backcast:
            return backcast
        return forecast

    @staticmethod
    def name():
        return "NBeatsNetPytorch"

    def get_generic_and_interpretable_outputs(self):
        g_pred = sum([a["value"][0] for a in self._intermediary_outputs if "generic" in a["layer"].lower()])
        i_pred = sum([a["value"][0] for a in self._intermediary_outputs if "generic" not in a["layer"].lower()])
        outputs = {o["layer"]: o['value'][0] for o in self._intermediary_outputs}
        return g_pred, i_pred, outputs

    def forward(self, backcast):
        """
        Compute the backcast and forecast from the input parameters using the harmonic function templates.

        Args:
            backcast: The input tensor representing the historical data to be processed
            for forecasting. Should have the appropriate dimensions expected by the model.

        Returns:
            tuple:
                - backcast (torch.Tensor): The final residuals after processing through all blocks.
                - forecast (torch.Tensor): The forecasted values generated by the model.
        """
        self._intermediary_outputs = []
        backcast = squeeze_last_dim(backcast)
        forecast = torch.zeros(size=(backcast.size()[0], self.forecast_length,))  # maybe batch size here.

        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)
                backcast = backcast.to(self.device) - b
                forecast = forecast.to(self.device) + f
                block_type = self.stacks[stack_id][block_id].__class__.__name__
                layer_name = f"stack_{stack_id}-{block_type}_{block_id}"

                if self._gen_intermediate_outputs:
                    self._intermediary_outputs.append({"value": f.detach().numpy(), "layer": layer_name})

        return backcast, forecast


def squeeze_last_dim(tensor):
    if len(tensor.shape) == 3 and tensor.shape[-1] == 1:  # (128, 10, 1) => (128, 10).
        return tensor[..., 0]
    return tensor


def seasonality_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= thetas.shape[1], "thetas_dim is too big."
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor(np.array([np.cos(2 * np.pi * i * t) for i in range(p1)])).float()  # H/2-1
    s2 = torch.tensor(np.array([np.sin(2 * np.pi * i * t) for i in range(p2)])).float()
    S = torch.cat([s1, s2])
    return thetas.mm(S.to(device))


def trend_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= 4, "thetas_dim is too big."
    T = torch.tensor(np.array([t ** i for i in range(p)])).float()
    return thetas.mm(T.to(device))


def linear_space(backcast_length, forecast_length, is_forecast=True):
    horizon = forecast_length if is_forecast else backcast_length
    return np.arange(0, horizon) / horizon


class Block(nn.Module):
    """
    Represents a building block of the NBeats neural network architecture, responsible for capturing a specific
    temporal pattern within sequential data.

    Each Block instance is a self-contained neural network module equipped with fully connected layers and
    applicable non-linear activation functions. It receives an input tensor representing historical values of the
    time series and outputs processed tensors for both backcast and forecast values.

    Parameters:
        units: The number of neurons in each layer of the block.
        thetas_dim: The dimensionality of the theta vector, which represents the learned parameters
                          for the polynomial basis expansion.
        device: The device (CPU or CUDA) on which to perform calculations.
        backcast_length: The length of the input sequence used to compute backcasts.
        forecast_length: The length of the output sequence for the forecast.
        share_thetas: Determines if the model should use the same theta parameters for both backcast
                             and forecast computations to reduce the number of parameters.
        nb_harmonics: The number of harmonic terms to use when processing seasonal patterns.
                                      This parameter is only applicable for seasonal blocks and is ignored
                                      for other types of blocks.

    Attributes:
        fc1, fc2, fc3, fc4 (nn.Linear): Fully connected layers of the block.
        backcast_linspace, forecast_linspace: Linspace vectors for polynomial basis expansion in backcast
                                              and forecast computations respectively.

    Methods:
        forward(x): The forward pass receiving an input tensor `x`, applying a series of transformations,
                    and outputting the transformed tensor.

    Examples:
        To instantiate a `Block` you must provide the required parameters: units, thetas_dim, and device.

            block = Block(units=64, thetas_dim=4, device=torch.device('cuda:0'))

        To use this Block within an NBeats model, you could perform a forward pass with:

            backcast, forecast = block(input_tensor)

    Note:
        The __str__ method of the class is overridden to provide a string representation of the
        Block configuration, including the unique memory address, which can be useful for debugging or logging.
    """

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, share_thetas=False,
                 nb_harmonics=None):
        super(Block, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.fc1 = nn.Linear(backcast_length, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)
        self.device = device
        self.backcast_linspace = linear_space(backcast_length, forecast_length, is_forecast=False)
        self.forecast_linspace = linear_space(backcast_length, forecast_length, is_forecast=True)

        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
            self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)

    def forward(self, x):
        x = squeeze_last_dim(x)
        x = F.relu(self.fc1(x.to(self.device)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}'


class GenericBlock(Block):
    """
    A subclass of Block that extends functionality to capture complex or non-standard temporal patterns
    in time series data beyond the conventional trend and seasonality components.

    `GenericBlock` is designed to capture patterns that are less predictable, such as residuals, noise,
    or other irregularities. With this flexibility, it serves as a catch-all block that can be tuned to
    model the idiosyncrasies of a specific dataset when traditional periodicities do not suffice.

    Inherits from:
        Block (nn.Module): The base block class that `GenericBlock` expands upon.

    Parameters:
        units: The number of neurons in each layer of the block.
        thetas_dim: The dimensionality of the theta vector for expansion coefficients.
        device: The computation device (CPU or GPU/CUDA).
        backcast_length: The number of past time steps used to predict the present.
        forecast_length: The number of future time steps to predict.
        nb_harmonics: The number of harmonics terms to use when processing seasonal
                                      patterns. Not utilized in `GenericBlock`; present for interface compatibility.

    Attributes:
        backcast_fc (nn.Linear): Linear layer to transform theta to the backcast output.
        forecast_fc (nn.Linear): Linear layer to transform theta to the forecast output.

    Methods:
        forward(x): Processes the input data and returns the backcast and forecast for the block.

    Example:
        To create an instance of `GenericBlock`, the same initialization parameters as `Block` can be used.

            generic_block = GenericBlock(units=128, thetas_dim=8, device=torch.device('cpu'),
                                         backcast_length=20, forecast_length=10)

        Once instantiated, `GenericBlock` can be used in the forward pass of the model.

            backcast, forecast = generic_block(input_data)

    Note:
        `GenericBlock` should be integrated within the larger NBeats model architecture. It is
        typically not used on its own but as part of a sequence of blocks within a model stack.
    """

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics=None):
        super(GenericBlock, self).__init__(units, thetas_dim, device, backcast_length, forecast_length)

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        # No constraint for generic arch.
        x = super(GenericBlock, self).forward(x)

        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)

        backcast = self.backcast_fc(theta_b)  # Generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # Generic. 3.3.

        return backcast, forecast


class SeasonalityBlock(Block):
    """
    A specialized implementation of the Block class, designed to specifically capture and
    model the seasonal variations within time series data. This block focuses on identifying
    and learning patterns that recur at consistent, predictable intervals, such as daily,
    weekly, monthly, or annually.

    The SeasonalityBlock leverages a unique configuration of its neural layers and trainable
    parameters (thetas) to encapsulate the essence of periodic fluctuations, making it an
    essential component for analyzing data with strong seasonal trends.

    Inherits from:
        Block (nn.Module): Base class providing the foundational neural network structure.

    Parameters:
        units: Number of neurons in each fully connected layer of the block.
        thetas_dim: The dimensionality of the theta vector that captures the coefficients
                          relevant to seasonality patterns. It typically corresponds to the
                          number of harmonics used in modeling the seasonality.
        device: The device (CPU or CUDA) where the calculations will be performed.
        backcast_length: Length of the input time series data for generating backcasts.
        forecast_length: Length of the generated output forecast.
        nb_harmonics: Specifies the number of harmonic terms to be used in modeling
                                      the seasonality. It directly influences the dimensionality of the
                                      model's seasonality coefficients (thetas).

    Attributes:
        Inherits all attributes from the Block class with specific modifications to `theta_b_fc` and
        `theta_f_fc` linear layers based on whether `nb_harmonics` is provided or not.

    Methods:
        forward(x): Processes the input time series `x` through the block's network to output both
                    backcasts and forecasts encapsulating the learned seasonal patterns.

    Example:
        seasonality_block = SeasonalityBlock(units=32, thetas_dim=10, device=torch.device('cuda'),
                                             backcast_length=24, forecast_length=12, nb_harmonics=5)

        # Given input data `x`, the forward pass produces backcasts and forecasts
        backcast, forecast = seasonality_block(x)

    Note:
        This block should be selected when dealing with time series data known to contain strong
        seasonal components. It can be used as part of a larger NBeats model consisting of multiple
        specialized blocks to comprehensively model complex time series datasets.
    """

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics=None):
        if nb_harmonics:
            super(SeasonalityBlock, self).__init__(units, nb_harmonics, device, backcast_length,
                                                   forecast_length, share_thetas=True)
        else:
            super(SeasonalityBlock, self).__init__(units, forecast_length, device, backcast_length,
                                                   forecast_length, share_thetas=True)

    def forward(self, x):
        x = super(SeasonalityBlock, self).forward(x)
        backcast = seasonality_model(self.theta_b_fc(x), self.backcast_linspace, self.device)
        forecast = seasonality_model(self.theta_f_fc(x), self.forecast_linspace, self.device)
        return backcast, forecast


class TrendBlock(Block):
    """
    A dedicated subclass of Block tailored to understand and represent the trend aspect of time
    series data. TrendBlock focuses on capturing long-term increase or decrease patterns in the dataset,
    such as overall growth or decline trends over time.

    By isolating and modeling the data's trend, this block can be crucial in forecasting where a given
    time series is heading, based on its historical trajectory. This makes it a valuable tool for
    time series forecasting tasks where trend plays a significant role in the underlying process.

    Inherits from:
        Block (nn.Module): The base class for various types of blocks in an NBeats architecture, defining
                           the core neural network components.

    Parameters:
        units: Number of neurons for the fully connected layers within the block.
        thetas_dim: Size of the theta vector that represents the coefficients of the polynomial
                          functions for the trend component.
        device: Computation device ('cpu' or 'cuda') on which the block will operate.
        backcast_length: The number of previous time points the block will use to make predictions.
        forecast_length: The number of future time points the block aims to predict.
        nb_harmonics: The number of harmonic terms used in the block for capturing
                                      periodic components, not directly used for trend blocks.

    Attributes:
        Inherits all attributes from the Block class with an emphasis on modeling the trend.

    Methods:
        forward(x): Takes an input `x` representing the time series data and runs it through the block
                    to output both the backcast and forecast that reflect the trend component.

    Example:
        # Initialize the TrendBlock with appropriate parameters.
        trend_block = TrendBlock(units=64, thetas_dim=2, device=torch.device('cuda:0'),
                                 backcast_length=24, forecast_length=12)

        # Using the initialized block to process data and obtain trend predictions.
        backcast, forecast = trend_block(input_sequence)

    Note:
        The TrendBlock is optimized to work as a part of a composite model in the NBeats architecture,
        often combined with other blocks like SeasonalityBlock or GenericBlock to achieve comprehensive
        forecasting that covers various aspects of time series data.
    """

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics=None):
        super(TrendBlock, self).__init__(units, thetas_dim, device, backcast_length,
                                         forecast_length, share_thetas=True)

    def forward(self, x):
        x = super(TrendBlock, self).forward(x)
        backcast = trend_model(self.theta_b_fc(x), self.backcast_linspace, self.device)
        forecast = trend_model(self.theta_f_fc(x), self.forecast_linspace, self.device)
        return backcast, forecast
