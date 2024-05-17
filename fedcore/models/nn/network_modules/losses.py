import torch
from fastai.torch_core import Module
from torch import Tensor


class SMAPELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return 100 * torch.mean(2 * torch.abs(input - target) / (torch.abs(target) + torch.abs(input)) + 1e-8)
