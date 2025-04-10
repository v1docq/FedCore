import torch
from torch.nn.modules import Module


class RegularizationLoss(Module):
    """Base class for weight regularizers.

    Args:
        factor: The hyperparameter by which the calculated loss function is multiplied
            (default: ``0.001``).
    """

    def __init__(self, factor: float = 0.001) -> None:
        super().__init__()
        self.factor = factor


class LaiMSE(RegularizationLoss):
    """MSE with adaptive weighting based on residuals magnitude.

    Implements an adaptive MSE loss where the weight of each residual is determined
    by the combination of two terms, controlled by the balancing factor.

    Args:
        factor: Balances between error-sensitive and constant regularization terms
            (default: ``0.5``).
    """
    
    def __init__(self, factor: float = 0.5) -> None:
        super().__init__(factor=factor)

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Calculates adaptive MSE loss.
        
        Args:
            y_pred: Model predictions tensor of shape (N, *).
            y_true: Ground truth tensor of same shape as predictions.
        """
        residuals = y_pred - y_true
        n = y_pred.size(0)
        mse_term = residuals.pow(2)

        k_i = 2 * residuals / n
        k_i_sq = k_i.pow(2)

        if self.factor >= 1:
            term1 = k_i_sq / (1 + k_i_sq)
            term2 = self.factor / (1 + k_i_sq)
        else:
            term1 = k_i_sq / (self.factor * (1 + k_i_sq))
            term2 = 1 / (1 + k_i_sq)

        weight = torch.max(term1, term2)
        return (mse_term * weight).mean()


class LaiMAE(RegularizationLoss):
    """MAE with adaptive weighting based on residuals sign and magnitude.

    Implements an adaptive MAE loss with sign-aware weighting controlled by
    the balancing factor.

    Args:
        factor: Balances between sign-sensitive and constant regularization terms
            (default: ``0.5``).
    """
    
    def __init__(self, factor: float = 0.5) -> None:
        super().__init__(factor=factor)

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Calculates adaptive MAE loss.
        
        Args:
            y_pred: Model predictions tensor of shape (N, *).
            y_true: Ground truth tensor of same shape as predictions.
        """
        residuals = y_pred - y_true
        n = y_pred.size(0)
        mae_term = residuals.abs()

        k_i = residuals.sign() / n
        k_i_sq = k_i.pow(2)
        k_i_abs = k_i.abs()
        denom = (1 + k_i_sq).sqrt()

        if self.factor >= 1:
            term1 = k_i_abs / denom
            term2 = self.factor / denom
        else:
            term1 = k_i_abs / (self.factor * denom)
            term2 = 1 / denom

        weight = torch.max(term1, term2)
        return (mae_term * weight).mean()


class NormLoss(RegularizationLoss):
    """Norm regularizer for weight matrices.

    This regularizer encourages the L2 norms of neuron weights to be close to 1.

    Args:
        factor: The hyperparameter by which the calculated loss function is multiplied
            (default: ``0.001``).
    """

    def __init__(self, factor: float = 0.001) -> None:
        super().__init__(factor=factor)

    def forward(self, model: Module) -> Tensor:
        """Calculates the norm regularization loss.

        Args:
            model: The neural network model containing layers with weights.
        """
        loss = 0.0
        for module in model.modules():
            if hasattr(module, "weight") and module.weight is not None and module.weight.dim() >= 2:
                weights = module.weight
                # Calculate L2 norms along all dimensions except the first (output neurons)
                dims = tuple(range(1, weights.dim()))
                norms = torch.norm(weights, p=2, dim=dims)
                # Sum the squared deviations from 1 for each neuron's norm
                loss += torch.sum((1 - norms) ** 2)
        return self.factor * loss


class AdaptiveRegularizationLoss(RegularizationLoss):
    """Adaptive regularization based on parameter gradients.

    This regularizer applies an adaptive penalty based on the exponential of the
    negative absolute gradients, encouraging parameters with small gradients to
    have smaller magnitudes.

    Args:
        factor: The hyperparameter by which the calculated loss function is multiplied
            (default: ``0.001``).
    """
    
    def __init__(self, factor: float = 0.001) -> None:
        super().__init__(factor=factor)

    def forward(self, model: Module, main_loss: Tensor) -> Tensor:
        """Calculates the adaptive regularization loss.

        Args:
            model: The neural network model to regularize.
            main_loss: The primary loss value based on which gradients are computed.
        """
        # Compute gradients of main_loss with respect to model parameters
        grads = torch.autograd.grad(
            outputs=main_loss,
            inputs=model.parameters(),
            retain_graph=True,
            create_graph=True,
            allow_unused=True  # Handle parameters not used in the loss
        )

        reg_loss = 0.0
        for param, grad in zip(model.parameters(), grads):
            if grad is not None:
                I = torch.exp(-torch.abs(grad))
                reg_loss += torch.sum(I * (param ** 2))

        return self.factor * reg_loss
