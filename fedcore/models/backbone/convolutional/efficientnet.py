from torch import nn
from torchvision.models.efficientnet import (
    _efficientnet_conf,
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
)
from torchvision.ops import Conv2dNormActivation

EFFICIENTNET_MODELS = {
    "efficientnet_b0": efficientnet_b0,
    "efficientnet_b1": efficientnet_b1,
    "efficientnet_b2": efficientnet_b2,
    "efficientnet_b3": efficientnet_b3,
    "efficientnet_b4": efficientnet_b4,
    "efficientnet_b5": efficientnet_b5,
    "efficientnet_b6": efficientnet_b6,
    "efficientnet_b7": efficientnet_b7,
}


class EfficientNet:
    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 2,
        model_name: str = "efficientnet_b0",
        **kwargs,
    ):
        if model_name not in EFFICIENTNET_MODELS:
            raise ValueError(
                f"Unknown model name: {model_name}. Available models: {list(EFFICIENTNET_MODELS.keys())}"
            )
        self.model = EFFICIENTNET_MODELS[model_name](num_classes=output_dim, **kwargs)

        inverted_residual_setting, last_channel = _efficientnet_conf(
            model_name, width_mult=1.0, depth_mult=1.1
        )
        norm_layer = nn.BatchNorm2d

        #     redefine first layer to accept `input_dim` channels
        if input_dim != 3:
            firstconv_output_channels = inverted_residual_setting[0].input_channels
            first_layer = Conv2dNormActivation(
                input_dim,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
            self.model.features._modules["0"] = first_layer

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def forward(self, x):
        return self.model(x)
