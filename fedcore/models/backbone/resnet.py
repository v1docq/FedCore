from typing import Any, Optional
import torch
from torch import nn, optim, Tensor
from torchvision.models import resnet101, resnet152, resnet18, resnet34, resnet50

from fedot.core.operations.operation_parameters import OperationParameters
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.models.network_impl.layers import PrunedResNet, Bottleneck, BasicBlock
from fedcore.models.network_impl.base_nn_model import BaseNeuralModel
from fedcore.repository.constanst_repository import (
    CROSS_ENTROPY,
    MULTI_CLASS_CROSS_ENTROPY,
    MSE,
)

from torch.nn import Dropout
from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential
from torch.nn import Sigmoid


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
    def __init__(self, input_dim, output_dim, model_name: str = "ResNet18"):
        model_list = {**CLF_MODELS}
        self.model = model_list[model_name](num_classes=output_dim)

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


class ObjectDetector(Module):
    def __init__(self, baseModel, numClasses) -> None:
        super(ObjectDetector, self).__init__()

        # intialize base model and number of classes
        self.baseModel = baseModel
        self.numClasses = numClasses

        # build regressor head for outputting the bounding
        # box coordinates
        self.regressor = Sequential(
            Linear(baseModel.fc.in_features, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, 4),
            Sigmoid(),
        )

        # build the classifier head that predicts the
        # class labels
        self.classifier = Sequential(
            Linear(baseModel.fc.in_features, 512),
            ReLU(),
            Dropout(),
            Linear(512, 512),
            ReLU(),
            Dropout(),
            Linear(512, 9),
        )

        # set classifier of our base model to produce
        # outputs from last convolutional block
        self.baseModel.fc = Identity()

    # we take the output of the base model and pass it through our heads
    def forward(self, x):
        # pass the inputs through the base model and then obtain
        # predictions from two different branches of the network
        features = self.baseModel(x)
        bboxes = self.regressor(features)
        classLogits = self.classifier(features)

        # return outputs as tuple
        return (bboxes, classLogits)


class ResNetModel(BaseNeuralModel):
    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)
        self.epochs = params.get("epochs", 25)
        self.batch_size = params.get("batch_size", 10)
        self.model_name = params.get("model_name", "ResNet18")
        self.input_dim = params.get("input_dim", 1)
        self.output_dim = params.get("output_dim", 10)

    def _init_model(self, input_data):
        self.model = ResNet(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            model_name=self.model_name,
        )
        self.model_for_inference = ResNet(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            model_name=self.model_name,
        ).model
        self.model = self.model.model
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        if self.task_type == "classification":
            if input_data.shape[0] == 2:
                loss_fn = CROSS_ENTROPY()
            else:
                loss_fn = MULTI_CLASS_CROSS_ENTROPY()
        else:
            loss_fn = MSE()
        return loss_fn, optimizer

    def _predict_model(self, x_test):
        self.model.eval()
        x_test = Tensor(x_test).to(default_device())
        pred = self.model(x_test)
        return self._convert_predict(pred)
