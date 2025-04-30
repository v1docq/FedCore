"""U-Net model from https://github.com/milesial/Pytorch-UNet with small fix for compatibility."""

import torch
import torch.nn as nn
from segmentation_models_pytorch import UnetPlusPlus
from typing import Optional

from fedcore.architecture.comptutaional.devices import default_device
from fedcore.models.network_impl.layers import DoubleConv, Down, OutConv, Up


BASE_UNET_PLUS_PLUS_PARAMS = {
    'encoder_name': 'timm-efficientnet-b0',
    'encoder_weights': 'imagenet',
    'encoder_depth': 5,
    'decoder_channels': (256, 128, 64, 32, 16),
    'activation': None
}

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # return {'out': logits}
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class UNetwork:
    def __init__(self, input_dim, output_dim):
        self.model = UNet(n_channels=input_dim, n_classes=output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(default_device())
        return self.model(x)


class UNetPlusPlusNetwork(nn.Module):
    """Unet++ realization with new parameters interface.

    Args:
        input_dim (int):  Input channels number.
        output_dim (int): Output channels number.
        depth (int): Encoder depth (redefines encoder_depth when it is absent at custom_params).
        custom_params (dict): Additional parameters.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            depth: int = 6,
            custom_params: Optional[dict] = None,
            **kwargs
    ):
        super().__init__()

        # Parameters union
        params = BASE_UNET_PLUS_PLUS_PARAMS.copy()
        if custom_params:
            params.update(custom_params)
        params.update(kwargs)

        # Priority: custom_params > depth > base settings
        params['encoder_depth'] = params.get('encoder_depth', depth)

        self.model = UnetPlusPlus(
            encoder_name=params['encoder_name'],
            encoder_weights=params['encoder_weights'],
            encoder_depth=params['encoder_depth'],
            decoder_channels=params['decoder_channels'],
            in_channels=input_dim,
            classes=output_dim,
            activation=params['activation']
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.model(x.to(default_device()))
