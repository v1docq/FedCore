from typing import Any, Optional
import torch
from torch import nn, optim, Tensor
from torchvision.models import resnet101, resnet152, resnet18, resnet34, resnet50

from fedot.core.operations.operation_parameters import OperationParameters
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.models.network_impl.layers import PrunedResNet, Bottleneck, BasicBlock
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel

BASE_RESNET_PARAMS = {'block': BasicBlock,
                      'layers': 4,
                      'initial_conv_output_dim': 64,
                      'sizes_per_layer': [64, 128, 256, 512],
                      'strides_per_layer': [1, 2, 2, 2],
                      }
BASE_RESNET_ARCH = {'layers': 2,
                    'blocks_per_layer': [3, 2]}


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
    "ResNet18": resnet18,
    "ResNet34": resnet34,
    "ResNet50": resnet50,
    "ResNet101": resnet101,
    "ResNet152": resnet152,
}

PRUNED_MODELS = {
    "ResNet18": pruned_resnet18,
    "ResNet34": pruned_resnet34,
    "ResNet50": pruned_resnet50,
    "ResNet101": pruned_resnet101,
    "ResNet152": pruned_resnet152,
}


class ResNet:
    def __init__(self, input_dim, output_dim, depth: dict = None, custom_params: dict = {}):
        model_list = {**CLF_MODELS}
        self.model = None
        for base_resnet_model_name in model_list.keys():
            if base_resnet_model_name.split('ResNet')[1] == str(depth['layers']):
                self.model = model_list[base_resnet_model_name](num_classes=output_dim)
        if self.model is None:
            # create custom resnet model
            unified_params = custom_params | BASE_RESNET_PARAMS
            self.model = PrunedResNet(input_dim=input_dim, output_dim=output_dim,
                                      depth=depth,
                                      custom_params=unified_params)

        if input_dim != 3:
            self.model.conv1 = nn.Conv2d(
                in_channels=input_dim,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implements the forward method of the model and returns predictions."""
        x = x.to(default_device())
        return self.model(x)


class ResNetModel(BaseNeuralModel):
    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)
        self.input_dim = params.model_architecture.get("input_dim", 1)
        self.output_dim = params.model_architecture.get("output_dim", 1)
        self.depth = params.model_architecture.get("depth", BASE_RESNET_ARCH)
        self.custom_params = params.model_architecture.get("custom_params", BASE_RESNET_PARAMS)
        self._init_model()

    def _init_model(self):
        self.model = ResNet(input_dim=self.input_dim, output_dim=self.output_dim,
                            depth=self.depth, custom_params=self.custom_params)

    def forward(self, *inputs):
        return self.model(*inputs)

