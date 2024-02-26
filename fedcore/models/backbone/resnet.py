from typing import Any
import torch
from torch import nn
from torchvision.models import ResNet, resnet101, resnet152, resnet18, resnet34, resnet50
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.models.network_impl.layers import PrunedResNet, Bottleneck, BasicBlock


def resnet18_one_channel(**kwargs) -> ResNet:
    """ResNet18 for one input channel"""
    model = resnet18(**kwargs)
    model.conv1 = nn.Conv2d(in_channels=1,
                            out_channels=64,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False)
    return model


def resnet34_one_channel(**kwargs) -> ResNet:
    """ResNet34 for one input channel"""
    model = resnet34(**kwargs)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                            stride=2, padding=3, bias=False)
    return model


def resnet50_one_channel(**kwargs) -> ResNet:
    """ResNet50 for one input channel"""
    model = resnet50(**kwargs)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                            stride=2, padding=3, bias=False)
    return model


def resnet101_one_channel(**kwargs) -> ResNet:
    """ResNet101 for one input channel"""
    model = resnet101(**kwargs)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                            stride=2, padding=3, bias=False)
    return model


def resnet152_one_channel(**kwargs) -> ResNet:
    """ResNet152 for one input channel"""
    model = resnet152(**kwargs)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                            stride=2, padding=3, bias=False)
    return model


def pruned_resnet18(**kwargs: Any) -> PrunedResNet:
    """Pruned ResNet-18."""
    return PrunedResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def pruned_resnet34(**kwargs: Any) -> PrunedResNet:
    """Pruned ResNet-34."""
    return PrunedResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def pruned_resnet50(**kwargs: Any) -> PrunedResNet:
    """Pruned ResNet-50."""
    return PrunedResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def pruned_resnet101(**kwargs: Any) -> PrunedResNet:
    """Pruned ResNet-101."""
    return PrunedResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def pruned_resnet152(**kwargs: Any) -> PrunedResNet:
    """Pruned ResNet-152."""
    return PrunedResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


class ResNet:
    def __init__(self,
                 input_dim,
                 output_dim,
                 model_name: str = 'ResNet18one'):
        model_list = {**CLF_MODELS, **CLF_MODELS_ONE_CHANNEL}
        self.model = model_list[model_name](num_classes=output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implements the forward method of the model and returns predictions."""
        x = x.to(default_device())
        return self.model(x)
