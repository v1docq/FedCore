from torch import nn
from torchvision.models.densenet import (
    densenet121,
    densenet161,
    densenet169,
    densenet201,
)

from fedcore.architecture.comptutaional.devices import default_device

DENSE_MODELS = {
    "densenet121": densenet121,
    "densenet169": densenet169,
    "densenet201": densenet201,
    "densenet161": densenet161,
}


class DenseNetwork:
    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 2,
        model_name: str = "densenet121",
        **kwargs,
    ):
        assert model_name in DENSE_MODELS, (
            f"Unknown model name: {model_name}. "
            f"Available models: {list(DENSE_MODELS.keys())}"
        )

        self.model = DENSE_MODELS[model_name](num_classes=output_dim)

        if input_dim != 3:
            self.model.features.conv0 = nn.Conv2d(
                input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

    def forward(self, x):
        x = x.to(default_device())
        return self.model(x)
