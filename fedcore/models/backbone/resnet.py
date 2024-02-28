from typing import Any
import torch
from torch import nn
from torchvision.models import resnet101, resnet152, resnet18, resnet34, resnet50

from fedcore.architecture.comptutaional.devices import default_device
from fedcore.models.network_impl.layers import PrunedResNet, Bottleneck, BasicBlock


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


CLF_MODELS = {
    'ResNet18': resnet18,
    'ResNet34': resnet34,
    'ResNet50': resnet50,
    'ResNet101': resnet101,
    'ResNet152': resnet152,
}

PRUNED_MODELS = {
    "ResNet18": pruned_resnet18,
    "ResNet34": pruned_resnet34,
    "ResNet50": pruned_resnet50,
    "ResNet101": pruned_resnet101,
    "ResNet152": pruned_resnet152,
}


class ResNet:
    def __init__(self,
                 input_dim,
                 output_dim,
                 model_name: str = 'ResNet18'):
        model_list = {**CLF_MODELS}
        self.model = model_list[model_name](num_classes=output_dim)

        if input_dim != 3:
            self.model.conv1 = nn.Conv2d(in_channels=input_dim,
                                         out_channels=64,
                                         kernel_size=7,
                                         stride=2,
                                         padding=3,
                                         bias=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implements the forward method of the model and returns predictions."""
        x = x.to(default_device())
        return self.model(x)
