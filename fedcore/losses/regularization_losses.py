import torch
from torch.nn.modules import Module
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class LaiMSE(Module):
    def __init__(self, lambda_reg=0.5):
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(self, y_pred, y_true):
        residuals = y_pred - y_true
        n = y_pred.size(0)
        mse_term = residuals.pow(2)

        k_i = 2 * residuals / n
        k_i_sq = k_i.pow(2)

        if self.lambda_reg >= 1:
            term1 = k_i_sq / (1 + k_i_sq)
            term2 = self.lambda_reg / (1 + k_i_sq)
        else:
            term1 = k_i_sq / (self.lambda_reg * (1 + k_i_sq))
            term2 = 1 / (1 + k_i_sq)

        weight = torch.max(term1, term2)
        return (mse_term * weight).mean()


class LaiMAE(Module):
    def __init__(self, lambda_reg=0.5):
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(self, y_pred, y_true):
        residuals = y_pred - y_true
        n = y_pred.size(0)
        mae_term = residuals.abs()

        k_i = residuals.sign() / n
        k_i_sq = k_i.pow(2)
        k_i_abs = k_i.abs()
        denom = (1 + k_i_sq).sqrt()

        if self.lambda_reg >= 1:
            term1 = k_i_abs / denom
            term2 = self.lambda_reg / denom
        else:
            term1 = k_i_abs / (self.lambda_reg * denom)
            term2 = 1 / denom

        weight = torch.max(term1, term2)
        return (mae_term * weight).mean()


class NormLoss(Module):
    def __init__(self, lambda_reg, model, l_target):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.model = model
        self.l_target = l_target

    def forward(self, outputs, targets):
        loss_target = self.l_target(outputs, targets)
        L_nl = 0

        for module in self.model.modules():
            if hasattr(module, "weight") and module.weight is not None and module.weight.dim() >= 2:
                weights = module.weight
                dims = tuple(range(1, weights.dim()))
                norms = torch.norm(weights, p=2, dim=dims)
                L_nl += torch.sum((1 - norms) ** 2)

        total_loss = loss_target + self.lambda_reg * L_nl
        return total_loss


class AdaptiveRegularizationLoss(Module):
    def __init__(self, lambda_reg, model, l_target):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.model = model
        self.l_target = l_target

    def forward(self, outputs, targets, model, lambda_reg):
        main_loss = self.l_target(outputs, targets)

        grads = torch.autograd.grad(
            outputs=main_loss,
            inputs=model.parameters(),
            retain_graph=True,
            create_graph=True
        )

        reg_loss = 0.0
        for param, grad in zip(model.parameters(), grads):
            if grad is not None:
                I = torch.exp(-torch.abs(grad))
                reg_loss += torch.sum(I * (param ** 2))

        return main_loss + lambda_reg * reg_loss
