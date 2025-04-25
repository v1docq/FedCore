import torch
import pytest

from fedcore.losses.regularization_losses import (
    LaiMSE, LaiMAE, NormLoss, AdaptiveRegularizationLoss
)


class SimpleModel(torch.nn.Module):
    """Dummy model"""
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 5)
        self.linear2 = torch.nn.Linear(5, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


# ==========================
# LaiMSE
# ==========================
def test_lai_mse_basic():
    """Test basic functionality with equal-sized tensors."""
    loss_fn = LaiMSE(factor=0.5)
    y_pred = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([1.1, 1.9, 3.1])
    loss = loss_fn(y_pred, y_true)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])  # scalar

def test_lai_mse_factor_above_1():
    """Test behavior when factor >= 1."""
    loss_fn = LaiMSE(factor=1.5)
    y_pred = torch.tensor([1.0, 2.0])
    y_true = torch.tensor([1.5, 2.5])
    loss = loss_fn(y_pred, y_true)
    assert loss > 0

def test_lai_mse_factor_below_1():
    """Test behavior when factor < 1."""
    loss_fn = LaiMSE(factor=0.3)
    y_pred = torch.tensor([1.0, 2.0])
    y_true = torch.tensor([1.5, 2.5])
    loss = loss_fn(y_pred, y_true)
    assert loss > 0

def test_lai_mse_zero_residuals():
    """Test with perfect predictions (zero residuals)."""
    loss_fn = LaiMSE(factor=0.5)
    y_pred = torch.tensor([1.0, 2.0])
    y_true = torch.tensor([1.0, 2.0])
    loss = loss_fn(y_pred, y_true)
    assert loss == 0.0

def test_lai_mse_large_residuals():
    """Test with large residuals."""
    loss_fn = LaiMSE(factor=0.5)
    y_pred = torch.tensor([0.0])
    y_true = torch.tensor([100.0])
    loss = loss_fn(y_pred, y_true)
    assert loss > 0

# ==========================
# LaiMAE
# ==========================
def test_lai_mae_basic():
    """Test basic functionality with equal-sized tensors."""
    loss_fn = LaiMAE(factor=0.5)
    y_pred = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([1.1, 1.9, 3.1])
    loss = loss_fn(y_pred, y_true)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])

def test_lai_mae_factor_above_1():
    """Test behavior when factor >= 1."""
    loss_fn = LaiMAE(factor=2.0)
    y_pred = torch.tensor([1.0, 2.0])
    y_true = torch.tensor([1.5, 2.5])
    loss = loss_fn(y_pred, y_true)
    assert loss > 0

def test_lai_mae_zero_residuals():
    """Test with perfect predictions (zero residuals)."""
    loss_fn = LaiMAE(factor=0.5)
    y_pred = torch.tensor([1.0, 2.0])
    y_true = torch.tensor([1.0, 2.0])
    loss = loss_fn(y_pred, y_true)
    assert loss == 0.0

def test_lai_mae_large_residuals():
    """Test with large residuals."""
    loss_fn = LaiMAE(factor=0.5)
    y_pred = torch.tensor([0.0])
    y_true = torch.tensor([100.0])
    loss = loss_fn(y_pred, y_true)
    assert loss > 0

# ==========================
# NormLoss
# ==========================
def test_norm_loss_basic():
    """Test basic functionality with a simple model."""
    model = SimpleModel()
    loss_fn = NormLoss(factor=0.001)
    loss = loss_fn(model)
    assert isinstance(loss, torch.Tensor)
    assert loss > 0

def test_norm_loss_zero_factor():
    """Test with zero factor should return zero loss."""
    model = SimpleModel()
    loss_fn = NormLoss(factor=0.0)
    loss = loss_fn(model)
    assert loss == 0.0

def test_norm_loss_ideal_weights():
    """Test with weights already normalized."""
    model = SimpleModel()
    # Manually set weights to have norm 1
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, 'weight'):
                module.weight.data = torch.nn.functional.normalize(module.weight.data, p=2, dim=1)
    loss_fn = NormLoss(factor=0.001)
    loss = loss_fn(model)
    assert loss < 1e-6  # Should be very close to zero

# ==========================
# AdaptiveRegularizationLoss
# ==========================
def test_adaptive_reg_basic():
    """Test basic functionality with a simple model."""
    model = SimpleModel()
    loss_fn = AdaptiveRegularizationLoss(factor=0.001)
    # Create a dummy loss
    dummy_input = torch.randn(3, 10)
    dummy_output = model(dummy_input)
    main_loss = torch.nn.MSELoss()(dummy_output, torch.randn(3, 2))
    
    reg_loss = loss_fn(model, main_loss)
    assert isinstance(reg_loss, torch.Tensor)
    assert reg_loss > 0


def test_adaptive_reg_zero_factor():
    """Test with zero factor should return zero loss."""
    model = SimpleModel()
    loss_fn = AdaptiveRegularizationLoss(factor=0.0)
    dummy_input = torch.randn(3, 10, requires_grad=True)
    dummy_output = model(dummy_input)
    target = torch.randn(3, 2)
    main_loss = torch.nn.MSELoss()(dummy_output, target)
    reg_loss = loss_fn(model, main_loss)
    assert reg_loss == 0.0


def test_adaptive_reg_zero_gradients():
    """Test when gradients are zero."""
    model = SimpleModel()
    loss_fn = AdaptiveRegularizationLoss(factor=0.001)
    dummy_input = torch.randn(3, 10, requires_grad=True)
    dummy_output = model(dummy_input)
    with torch.no_grad():
        target = torch.randn(3, 2)
    main_loss = torch.nn.MSELoss()(dummy_output, target)
    # With zero grads this loss should degenerate to L2
    reg_loss = loss_fn(model, main_loss)
    assert reg_loss > 0
