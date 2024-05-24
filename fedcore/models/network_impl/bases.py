import torch
import numpy as np
from torch import nn
from typing import Tuple


class _NBeatsBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.

        Args:
            input_size: the size of the input layer,
            theta_size: the dimension of the theta layer, which represents
                the flattened parameters used by basis functions to make forecasts,
            basis_function: a function that defines
                the operation of translating the model's parameters (theta) into the output space.
                This function essentially shapes the output of the N-BEATS block,
            layers: the number of hidden layers in the N-BEATS block,
            layer_size: the size of the output layer
    """

    def __init__(
            self,
            input_size,
            theta_size: int,
            basis_function: nn.Module,
            layers: int,
            layer_size: int
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(in_features=input_size, out_features=layer_size)] +
            [nn.Linear(in_features=layer_size, out_features=layer_size) for _ in range(layers - 1)]
        )

        self.basis_parameters = nn.Linear(in_features=layer_size, out_features=theta_size)
        self.basis_function = basis_function

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        block_input = x
        for layer in self.layers:
            block_input = torch.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)

        return self.basis_function(basis_parameters)


class _GenericBasis(nn.Module):
    """
    A basis function block for the N-BEATS neural network that applies a generic transformation.

    This module uses a simple linear projection of the input parameters without
    incorporating any domain-specific knowledge about time series data.
    The output is divided into two tensors: one for backcasting and one for forecasting,
    which are linear projections of the model's parameters (theta).

    The backcast (`g^b_l`) and forecast (`g^f_l`) outputs are created from the learned theta vector,
    enabling the model to make predictions based on learned representations.

    Attributes:
        backcast_size (int): The size of the output tensor representing the backcast.
        forecast_size (int): The size of the output tensor representing the forecast.

    Args:
        backcast_size (int): The dimension of the backcast output, i.e., the number of time steps to predict in the past.
        forecast_size (int): The dimension of the forecast output, i.e., the number of time steps to predict in the future.

    Methods:
        forward(theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            Defines the computation performed at every call. It takes a tensor 'theta',
            which includes parameters learned by earlier layers in the network, and returns
            two tensors: the backcast and the forecast.

    Usage:
        Instantiate the _GenericBasis module by providing backcast and forecast sizes. An input
        tensor 'theta' is passed through the forward method to get the backcast and forecast outputs.

        Example:
            generic_basis = _GenericBasis(backcast_size=10, forecast_size=5)
            backcast, forecast = generic_basis(theta)
    """

    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform the forward pass for generating backcast and forecast from theta.

        Args:
            theta (torch.Tensor): The flattened parameter tensor from the previous layer.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the backcast and forecast tensor projections.
        """
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]


class _TrendBasis(nn.Module):
    """
    A polynomial basis function module for modeling trends in time series data.

    This trend model captures the behavior of time series data that tends to be
    monotonic or changing slowly over time. It constrains the backcasting (g^b_sl)
    and forecasting (g^f_sl) outputs to be a polynomial function of a specified
    degree. This low-degree polynomial provides a simple yet effective way of
    capturing slowly varying trends across the forecast window.

    Attributes:
        polynomial_size: The size of the polynomial, which is one more than the degree
            of the polynomial due to the constant term.
        backcast_time: The time indices for the backcast, represented
            as a parameter tensor that is not trainable.
        forecast_time: The time indices for the forecast, similarly
            represented as a non-trainable parameter tensor.

    Args:
        degree_of_polynomial: The degree of the polynomial used to model the trend.
        backcast_size: The number of time steps to predict into the past, defining the size of the backcast output.
        forecast_size: The number of time steps to predict into the future, defining the size of the forecast output.

    Methods:
        forward(theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            Defines the forward pass of the module. It applies the polynomial transformation
            to the parameters theta to produce the backcast and forecast outputs.

    Example:
        Instantiate the _TrendBasis module by providing the degree of the polynomial
        along with the backcast and forecast sizes. The forward method takes a theta tensor
        and returns the trend-based backcast and forecast.

        Example usage:
            trend_basis = _TrendBasis(degree_of_polynomial=2, backcast_size=10, forecast_size=5)
            backcast, forecast = trend_basis(theta)
    """

    def __init__(
            self,
            degree_of_polynomial: int,
            backcast_size: int,
            forecast_size: int
    ):
        super().__init__()

        self.polynomial_size = degree_of_polynomial + 1  # degree of polynomial with constant term
        self.backcast_time = nn.Parameter(
            torch.tensor(np.concatenate(
                [np.power(np.arange(backcast_size, dtype=np.float) / backcast_size, i)[None, :] for i in
                 range(self.polynomial_size)]),
                dtype=torch.float32),
            requires_grad=False
        )

        self.forecast_time = nn.Parameter(
            torch.tensor(np.concatenate(
                [np.power(np.arange(forecast_size, dtype=np.float) / forecast_size, i)[None, :] for i in
                 range(self.polynomial_size)]),
                dtype=torch.float32),
            requires_grad=False
        )

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform the trend polynomial transformation during the forward computation.

        Args:
            theta: Tensor of parameters learned by earlier model layers.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The resulting backcast and forecast tensors
            after applying the trend-based polynomial transformation to theta.
        """
        backcast = torch.einsum(
            "bp,pt->bt",
            theta[:, self.polynomial_size:],
            self.backcast_time
        )

        forecast = torch.einsum(
            "bp,pt->bt",
            theta[:, :self.polynomial_size],
            self.forecast_time
        )

        return backcast, forecast


class _SeasonalityBasis(_NBeatsBlock):
    """
    A neural network block that models seasonality in time-series data using harmonic functions.

    Seasonal effects in time series are recurrent patterns or cycles arising within specific periods.
    To capture such effects, this block uses a basis expansion with harmonic functions (sine and cosine)
    allowing it to represent a broad class of periodic functions. The model parameters are optimized
    to fit these functions to the seasonal patterns observed in the data.

    Attributes:
        frequency: Frequencies of the harmonic functions used to model the seasonality.
        backcast_cos_template: Template for the cosine basis functions for backcasting.
        backcast_sin_template: Template for the sine basis functions for backcasting.
        forecast_cos_template: Template for the cosine basis functions for forecasting.
        forecast_sin_template: Template for the sine basis functions for forecasting.

    Args:
        harmonics: The number of harmonics, or pairs of sine/cosine functions, to use in the model.
        backcast_size: The length of the backcast, the number of time steps to predict into the past.
        forecast_size: The length of the forecast, the number of time steps to predict into the future.

    Methods:
        forward(theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            Processes the input through the seasonality basis to produce backcast and forecast values.

    Example:
        Initializing a `_SeasonalityBasis` block typically requires specifying the number of harmonics to
        include in the model, alongside the sizes of the backcast and forecast windows.

        Example usage:
            seasonality_basis = _SeasonalityBasis(harmonics=10, backcast_size=100, forecast_size=50)
            backcast, forecast = seasonality_basis(theta)
    """

    def __init__(
            self,
            harmonics: int,
            backcast_size: int,
            forecast_size: int
    ):
        super().__init__()

        self.frequency = np.append(np.zeros(1, dtype=np.float32),
                                   np.arange(harmonics, harmonics / 2 * forecast_size,
                                             dtype=np.float32) / harmonics)[None, :]
        backcast_grid = -2 * np.pi * (
                np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency

        forecast_grid = 2 * np.pi * (
                np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency

        self.backcast_cos_template = nn.Parameter(
            torch.tensor(np.transpose(np.cos(backcast_grid)), dtype=torch.float32),
            requires_grad=False
        )

        self.backcast_sin_template = nn.Parameter(
            torch.tensor(np.transpose(np.sin(backcast_grid)), dtype=torch.float32),
            requires_grad=False
        )

        self.forecast_cos_template = nn.Parameter(
            torch.tensor(np.transpose(np.cos(forecast_grid)), dtype=torch.float32),
            requires_grad=False
        )

        self.forecast_sin_template = nn.Parameter(
            torch.tensor(np.transpose(np.sin(forecast_grid)), dtype=torch.float32),
            requires_grad=False
        )

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the backcast and forecast from the input parameters using the harmonic function templates.

        Args:
            theta: The parameters of the model, which are used to weight the harmonic basis functions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The backcast (past) and forecast (future) values generated by
            combining the harmonic functions with the model parameters.
        """
        params_per_harmonic = theta.shape[1] // 4
        backcast_harmonics_cos = torch.einsum(
            "bp,pt->bt",
            theta[:, 2 * params_per_harmonic:3 * params_per_harmonic],
            self.backcast_cos_template
        )

        backcast_harmonics_sin = torch.einsum(
            "bp,pt->bt",
            theta[:, 3 * params_per_harmonic:],
            self.backcast_sin_template
        )

        backcast = backcast_harmonics_sin + backcast_harmonics_cos

        forecast_harmonics_cos = torch.einsum(
            "bp,pt->bt",
            theta[:, :params_per_harmonic],
            self.forecast_cos_template
        )

        forecast_harmonics_sin = torch.einsum(
            "bp,pt->bt",
            theta[:, params_per_harmonic:2 * params_per_harmonic],
            self.forecast_sin_template
        )

        forecast = forecast_harmonics_sin + forecast_harmonics_cos

        return backcast, forecast

